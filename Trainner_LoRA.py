from typing import Any, Optional, List
import gc
import os
import time
import torch
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from peft import PeftModel
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map
from accelerate.utils import get_balanced_memory
from genrl.data import DataManager
from genrl.logging_utils.global_defs import get_logger
from genrl.logging_utils.ml_logger import LoggerMixin
from genrl.rewards import RewardManager
from genrl.state import GameState
from genrl.trainer.grpo_trainer import GRPOLanguageTrainerModule
from reasoning_gym.utils import SYSTEM_PROMPTS
from rgym_exp.src.utils.judge_client import JudgeClient
from rgym_exp.src.prg_module import PRGGameStatus

# INLINE ROBUST COMMUNICATION - No external dependencies
class EmergencyTrainingWrapper:
    """Emergency wrapper to prevent training crashes from communication errors"""
    
    def __init__(self, communication_backend):
        self.backend = communication_backend
        self.emergency_mode = False
        self.consecutive_errors = 0
        self.max_consecutive_errors = 3
        self.total_emergency_calls = 0
        
    def safe_all_gather(self, obj):
        """Ultra-safe wrapper around all_gather_object"""
        try:
            if self.emergency_mode:
                self.total_emergency_calls += 1
                if self.total_emergency_calls % 100 == 0:
                    get_logger().warning(f"Emergency mode: {self.total_emergency_calls} single-node calls")
                return {self.backend.get_id(): obj}
                
            result = self.backend.all_gather_object(obj)
            
            if self.consecutive_errors > 0:
                get_logger().info(f"Communication recovered after {self.consecutive_errors} errors")
                self.consecutive_errors = 0
                
            return result
            
        except Exception as e:
            error_msg = str(e)
            self.consecutive_errors += 1
            
            get_logger().error(f"EMERGENCY CATCH #{self.consecutive_errors}: {error_msg}")
            
            critical_patterns = [
                "ran out of input", "pipe", "broken", "connection", "timeout", 
                "eof", "resource temporarily unavailable", "blocking"
            ]
            
            if any(pattern in error_msg.lower() for pattern in critical_patterns):
                get_logger().error("Critical communication error detected - enabling emergency mode")
                self.emergency_mode = True
                
            if self.consecutive_errors >= self.max_consecutive_errors:
                get_logger().error(f"Too many consecutive errors ({self.consecutive_errors}) - emergency mode")
                self.emergency_mode = True
                
            return {self.backend.get_id(): obj}

    def all_gather_object(self, obj):
        return self.safe_all_gather(obj)

    def get_id(self):
        return self.backend.get_id()

    def shutdown(self):
        if hasattr(self.backend, 'shutdown'):
            self.backend.shutdown()


class FallbackBackend:
    """Simple fallback backend for single-node operation"""
    
    def __init__(self):
        self.agent_id = f"fallback_{os.getpid()}"
        self.mode = "single_node_fallback"
        
    def all_gather_object(self, obj):
        return {self.agent_id: obj}
        
    def safe_all_gather(self, obj):
        return {self.agent_id: obj}
        
    def get_id(self):
        return self.agent_id
        
    def get_training_mode(self):
        return self.mode
        
    def shutdown(self):
        pass


def create_robust_communication_wrapper(existing_backend):
    """Create robust wrapper around existing communication backend"""
    
    if existing_backend is None:
        get_logger().warning("No existing backend - creating fallback")
        return FallbackBackend()
    
    if hasattr(existing_backend, 'safe_all_gather'):
        get_logger().info("Backend already has robust features")
        return existing_backend
    
    get_logger().info("Wrapping existing backend with emergency handler")
    return EmergencyTrainingWrapper(existing_backend)


def emergency_disable_dht():
    """Emergency function to disable DHT completely"""
    os.environ['DISABLE_DHT'] = 'true'
    os.environ['FORCE_SINGLE_NODE'] = 'true'
    get_logger().warning("DHT EMERGENCY DISABLED - all communication will use single-node mode")


# Enhanced trainer prompts
PRG_SYSTEM_PROMPT = """Given a question, hints, and possible answers, your task is to answer the question by thinking step-by-step in a clear and specific manner for 1 line only.
Your answer MUST be one of the possible answers. Provide the answer in the following format:
<answer>answer here</answer>
Do not explain your reasoning inside the answer tags, provide only the final answer.
"""

PRG_SYSTEM_PROMPT_NO_THINKING = """Given a question, hints, and possible answers, your task is to answer the question.
Your answer MUST be one of the possible answers. Give your answer in the following format:
<answer>answer here</answer>
Do not explain your reasoning at all, provide only the final answer in the answer tag.
"""


class LoRAConfig:
    """Configuration for LoRA parameters - optimized for VRAM"""
    
    # Standard configs
    ULTRA_LOW_VRAM = {
        'r': 8,
        'lora_alpha': 16,
        'target_modules': ["q_proj", "v_proj"],
        'lora_dropout': 0.05,
        'bias': "none",
        'task_type': TaskType.CAUSAL_LM,
        'inference_mode': False
    }
    
    LOW_VRAM = {
        'r': 16,
        'lora_alpha': 32,
        'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj"],
        'lora_dropout': 0.05,
        'bias': "none",
        'task_type': TaskType.CAUSAL_LM,
        'inference_mode': False
    }
    
    BALANCED = {
        'r': 32,
        'lora_alpha': 64,
        'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        'lora_dropout': 0.1,
        'bias': "none",
        'task_type': TaskType.CAUSAL_LM,
        'inference_mode': False
    }
    
    HIGH_PERFORMANCE = {
        'r': 64,
        'lora_alpha': 128,
        'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        'lora_dropout': 0.1,
        'bias': "none",
        'task_type': TaskType.CAUSAL_LM,
        'inference_mode': False
    }
    
    @staticmethod
    def get_config(preset='balanced'):
        """Get LoRA config by preset name"""
        presets = {
            'ultra_low': LoRAConfig.ULTRA_LOW_VRAM,
            'low': LoRAConfig.LOW_VRAM,
            'balanced': LoRAConfig.BALANCED,
            'high': LoRAConfig.HIGH_PERFORMANCE
        }
        return presets.get(preset, LoRAConfig.BALANCED)


class MemoryMonitor:
    """Monitor both VRAM and RAM with aggressive offloading support"""
    
    def __init__(self, vram_threshold_gb=0.5, ram_threshold_gb=2.0, max_vram_gb=20.0):
        self.vram_threshold_gb = vram_threshold_gb
        self.ram_threshold_gb = ram_threshold_gb
        self.max_vram_gb = max_vram_gb
        self.last_log_time = 0
        self.log_interval = 30
        self.peak_vram = 0.0
        self.peak_ram = 0.0
        self.cleanup_counter = 0
        
    def get_vram_usage(self):
        """Get current VRAM usage in GB"""
        if not torch.cuda.is_available():
            return 0.0, 0.0
        
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        
        if allocated > self.peak_vram:
            self.peak_vram = allocated
        
        return allocated, reserved
    
    def get_ram_usage(self):
        """Get current RAM usage in GB"""
        process = psutil.Process(os.getpid())
        ram_used = process.memory_info().rss / 1024**3
        ram_available = psutil.virtual_memory().available / 1024**3
        
        if ram_used > self.peak_ram:
            self.peak_ram = ram_used
        
        return ram_used, ram_available
    
    def is_vram_critical(self):
        """Check if VRAM is critically high"""
        allocated, _ = self.get_vram_usage()
        return allocated > (self.max_vram_gb * 0.85)
    
    def is_ram_critical(self):
        """Check if RAM is critically low"""
        _, available = self.get_ram_usage()
        return available < self.ram_threshold_gb
    
    def log_usage(self, force=False, context=""):
        """Log both VRAM and RAM usage"""
        current_time = time.time()
        if not force and (current_time - self.last_log_time) < self.log_interval:
            return
        
        vram_alloc, vram_res = self.get_vram_usage()
        ram_used, ram_avail = self.get_ram_usage()
        
        prefix = f"[{context}] " if context else ""
        get_logger().info(
            f"{prefix}VRAM: {vram_alloc:.2f}/{vram_res:.2f}GB (peak {self.peak_vram:.2f}GB) | "
            f"RAM: {ram_used:.2f}GB used, {ram_avail:.2f}GB free (peak {self.peak_ram:.2f}GB)"
        )
        self.last_log_time = current_time
    
    def smart_clear(self, aggressive=False):
        """Smart cache clearing"""
        self.cleanup_counter += 1
        
        if aggressive or self.is_vram_critical():
            # Aggressive VRAM cleanup
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            for _ in range(3):
                gc.collect()
            
            vram_alloc, _ = self.get_vram_usage()
            get_logger().warning(f"AGGRESSIVE cleanup #{self.cleanup_counter} - VRAM: {vram_alloc:.2f}GB")
        else:
            # Normal cleanup
            torch.cuda.empty_cache()
            gc.collect()
    
    def force_cleanup(self):
        """Force aggressive cleanup of both VRAM and RAM"""
        get_logger().warning("FORCING AGGRESSIVE MEMORY CLEANUP...")
        
        for i in range(5):
            gc.collect()
            torch.cuda.empty_cache()
            if i < 4:
                time.sleep(0.1)
        
        torch.cuda.synchronize()
        
        vram_alloc, vram_res = self.get_vram_usage()
        ram_used, ram_avail = self.get_ram_usage()
        get_logger().warning(
            f"After cleanup: VRAM {vram_alloc:.2f}/{vram_res:.2f}GB | RAM {ram_used:.2f}GB used"
        )


class CPUOffloadOptimizer:
    """
    Wrapper for optimizer with CPU offloading
    Offload optimizer states to CPU RAM to save VRAM
    """
    
    def __init__(self, optimizer, pin_memory=True):
        self.optimizer = optimizer
        self.pin_memory = pin_memory
        self._offloaded_state = {}
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        get_logger().info("Initializing CPU Offload Optimizer")
        self._setup_offload()
    
    def _setup_offload(self):
        """Setup initial offload of optimizer state"""
        # Offload optimizer state dict to CPU
        if hasattr(self.optimizer, 'state'):
            for param_id, state in self.optimizer.state.items():
                if isinstance(state, dict):
                    cpu_state = {}
                    for key, value in state.items():
                        if isinstance(value, torch.Tensor) and value.is_cuda:
                            cpu_state[key] = value.cpu().pin_memory() if self.pin_memory else value.cpu()
                        else:
                            cpu_state[key] = value
                    self._offloaded_state[param_id] = cpu_state
    
    def step(self, closure=None):
        """Step with CPU offload"""
        # Move optimizer states to GPU temporarily
        self._move_state_to_device('cuda')
        
        # Perform optimizer step
        loss = self.optimizer.step(closure)
        
        # Move states back to CPU
        self._move_state_to_device('cpu')
        
        return loss
    
    def _move_state_to_device(self, device):
        """Move optimizer state between CPU and GPU"""
        if not hasattr(self.optimizer, 'state'):
            return
        
        target_device = torch.device(device)
        
        for param_id, state in self.optimizer.state.items():
            if isinstance(state, dict):
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        if device == 'cuda':
                            state[key] = value.to(target_device, non_blocking=True)
                        else:
                            state[key] = value.cpu().pin_memory() if self.pin_memory else value.cpu()
    
    def zero_grad(self, set_to_none=True):
        """Zero gradients"""
        self.optimizer.zero_grad(set_to_none=set_to_none)
    
    def __getattr__(self, name):
        """Delegate other attributes to underlying optimizer"""
        return getattr(self.optimizer, name)


class GRPOTrainerModule(GRPOLanguageTrainerModule, LoggerMixin):
    """
    AGGRESSIVE CPU OFFLOAD GRPO Trainer - Optimized for multiple instances
    
    Strategy:
    - Base model weights: CPU (loaded on-demand)
    - LoRA adapters: GPU (small, needs speed)
    - Optimizer states: CPU (offloaded)
    - Activations: CPU when not in use (gradient checkpointing)
    - Current batch: GPU only
    
    Expected savings: 70-80% VRAM reduction
    """

    def __init__(self, models: List[Any], **kwargs):
        """
        Initialize with aggressive CPU offloading for multi-instance training
        
        Args:
            models: List of pre-loaded models
            **kwargs: Configuration including:
                - lora_config: LoRA configuration
                - enable_lora: bool (default True)
                - vram_threshold_gb: float (default 1.0)
                - max_vram_gb: float (default 20.0)
                - offload_folder: str (default "./offload")
                - cpu_offload: bool (default True)
                - accumulation_steps: int (default 8)
                - max_memory_per_gpu: str (default "4GB")
        """
        get_logger().info("=" * 80)
        get_logger().info("INITIALIZING AGGRESSIVE CPU OFFLOAD GRPO TRAINER")
        get_logger().info("Strategy: Base Model → CPU, LoRA → GPU, Optimizer → CPU")
        get_logger().info("=" * 80)
        
        # Initialize robust communication
        self._init_robust_communication(kwargs)
        
        # Offload configuration
        self.cpu_offload = kwargs.get("cpu_offload", True)
        self.offload_folder = kwargs.get("offload_folder", "./offload")
        self.max_memory_per_gpu = kwargs.get("max_memory_per_gpu", "4GB")
        os.makedirs(self.offload_folder, exist_ok=True)
        
        # LoRA configuration
        self.enable_lora = kwargs.get("enable_lora", True)
        self.lora_config_dict = self._get_lora_config(kwargs.get("lora_config", "balanced"))
        self.lora_model = None
        
        # Memory monitoring
        vram_threshold = kwargs.get("vram_threshold_gb", 1.0)
        max_vram = kwargs.get("max_vram_gb", 20.0)
        self.memory_monitor = MemoryMonitor(
            vram_threshold_gb=vram_threshold,
            ram_threshold_gb=2.0,
            max_vram_gb=max_vram
        )
        
        # Gradient accumulation for even lower memory
        self.accumulation_steps = kwargs.get("accumulation_steps", 8)
        self._accumulation_counter = 0
        
        # Log initial memory state
        self.memory_monitor.log_usage(force=True, context="BEFORE INIT")
        
        # === Process models with AGGRESSIVE offloading ===
        if not models or len(models) == 0:
            raise ValueError("Models list is empty! Please provide pre-loaded models.")
        
        get_logger().info(f"Received {len(models)} pre-loaded model(s)")
        
        # Aggressive cleanup before processing
        self.memory_monitor.force_cleanup()
        
        processed_models = []
        for idx, model in enumerate(models):
            get_logger().info(f"Processing model {idx + 1}/{len(models)} with CPU offload...")
            
            # Apply CPU offload to base model
            if self.cpu_offload:
                model = self._apply_cpu_offload(model, idx)
            
            # Add LoRA (stays on GPU)
            if self._has_lora(model):
                get_logger().info(f"  → Model {idx + 1} already has LoRA")
                processed_models.append(model)
            elif self.enable_lora:
                get_logger().info(f"  → Adding LoRA to model {idx + 1} (GPU)")
                lora_model = self._apply_lora_to_model(model)
                processed_models.append(lora_model)
                self.memory_monitor.smart_clear(aggressive=True)
            else:
                get_logger().info(f"  → Using model {idx + 1} as-is")
                processed_models.append(model)
            
            self.memory_monitor.log_usage(force=True, context=f"MODEL {idx+1}")
        
        # Call parent class init
        super().__init__(processed_models, **kwargs)
        
        # Enable gradient checkpointing
        if kwargs.get("gradient_checkpointing", True):
            self._enable_gradient_checkpointing()
        
        # Wrap optimizer with CPU offload
        self._cpu_offload_optimizer = None
        
        # Judge client
        judge_base_url = kwargs.get("judge_base_url", None)
        self.judge_client = JudgeClient(judge_base_url) if judge_base_url else None
        
        # Communication tracking
        self._communication_errors = 0
        self._emergency_mode_enabled = False
        self._last_successful_gather = 0
        self._step_counter = 0
        self._last_cleanup_step = 0
        
        # Final cleanup
        self.memory_monitor.force_cleanup()
        self._log_setup_info()

    def _apply_cpu_offload(self, model, model_idx):
        """
        Apply aggressive CPU offload to base model using Accelerate
        Keeps only essential parts on GPU
        """
        get_logger().info(f"  Applying CPU offload to model {model_idx}...")
        
        try:
            # Create device map for aggressive offloading
            # Only keep LoRA-related layers on GPU, rest on CPU
            device_map = self._create_aggressive_device_map(model)
            
            # Log device map
            gpu_layers = sum(1 for v in device_map.values() if v == 'cuda' or v == 0)
            cpu_layers = sum(1 for v in device_map.values() if v == 'cpu')
            get_logger().info(f"  Device map: {gpu_layers} layers GPU, {cpu_layers} layers CPU")
            
            # Apply device map
            from accelerate import dispatch_model
            model = dispatch_model(
                model,
                device_map=device_map,
                offload_dir=self.offload_folder,
                offload_buffers=True
            )
            
            get_logger().info(f"  ✓ CPU offload applied successfully")
            self.memory_monitor.log_usage(force=True, context=f"AFTER OFFLOAD {model_idx}")
            
            return model
            
        except Exception as e:
            get_logger().warning(f"  Failed to apply CPU offload: {e}")
            get_logger().warning(f"  Continuing without offload for model {model_idx}")
            return model
    
    def _create_aggressive_device_map(self, model):
        """
        Create aggressive device map:
        - Embedding layers: CPU
        - Most transformer layers: CPU
        - Last few layers: GPU (for LoRA)
        - LM head: CPU
        """
        device_map = {}
        
        # Get model structure
        if hasattr(model, 'hf_device_map'):
            # Model already has device map
            return model.hf_device_map
        
        # Calculate memory budgets
        max_memory = {
            0: self.max_memory_per_gpu,  # GPU
            "cpu": "100GB"  # CPU (virtually unlimited)
        }
        
        # Use Accelerate's auto device map with strict GPU limit
        try:
            device_map = infer_auto_device_map(
                model,
                max_memory=max_memory,
                no_split_module_classes=["LlamaDecoderLayer", "Qwen2DecoderLayer"],  # Don't split transformer blocks
                dtype=torch.float16
            )
            
            get_logger().info(f"  Auto device map created: {len(device_map)} layers mapped")
            
        except Exception as e:
            get_logger().warning(f"  Auto device map failed: {e}, using manual map")
            
            # Manual aggressive mapping
            for name, module in model.named_modules():
                if 'embed' in name.lower():
                    device_map[name] = 'cpu'
                elif 'lm_head' in name.lower() or 'output' in name.lower():
                    device_map[name] = 'cpu'
                elif 'layers' in name.lower():
                    # Only keep last 2 layers on GPU for LoRA
                    layer_num = self._extract_layer_number(name)
                    if layer_num is not None:
                        total_layers = self._get_total_layers(model)
                        if total_layers is not None and layer_num >= total_layers - 2:
                            device_map[name] = 0  # GPU
                        else:
                            device_map[name] = 'cpu'
                    else:
                        device_map[name] = 'cpu'
                else:
                    device_map[name] = 'cpu'
        
        return device_map
    
    def _extract_layer_number(self, name):
        """Extract layer number from module name"""
        import re
        match = re.search(r'layers\.(\d+)', name)
        if match:
            return int(match.group(1))
        return None
    
    def _get_total_layers(self, model):
        """Get total number of transformer layers"""
        if hasattr(model, 'config'):
            if hasattr(model.config, 'num_hidden_layers'):
                return model.config.num_hidden_layers
            elif hasattr(model.config, 'n_layer'):
                return model.config.n_layer
        return None

    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to save VRAM"""
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            try:
                self.model.gradient_checkpointing_enable()
                get_logger().info("✓ Gradient checkpointing ENABLED")
            except Exception as e:
                get_logger().warning(f"Could not enable gradient checkpointing: {e}")
        else:
            get_logger().warning("Model does not support gradient checkpointing")

    def _get_lora_config(self, config_input):
        """Get LoRA configuration"""
        if isinstance(config_input, dict):
            return config_input
        elif isinstance(config_input, str):
            return LoRAConfig.get_config(config_input)
        else:
            return LoRAConfig.BALANCED

    def _has_lora(self, model) -> bool:
        """Check if model has LoRA"""
        if isinstance(model, PeftModel):
            return True
        if hasattr(model, 'peft_config') and model.peft_config is not None:
            return True
        model_str = str(type(model)).lower()
        if 'peft' in model_str or 'lora' in model_str:
            return True
        return False

    def _apply_lora_to_model(self, model):
        """Apply LoRA adapter (stays on GPU)"""
        if self._has_lora(model):
            get_logger().warning("Model already has LoRA")
            return model
        
        get_logger().info("Applying LoRA configuration (GPU)...")
        get_logger().info(f"  LoRA rank: {self.lora_config_dict['r']}")
        get_logger().info(f"  LoRA alpha: {self.lora_config_dict['lora_alpha']}")
        
        try:
            # Prepare for k-bit training
            model = prepare_model_for_kbit_training(model)
            
            # Create LoRA config
            lora_config = LoraConfig(**self.lora_config_dict)
            
            # Apply LoRA
            model = get_peft_model(model, lora_config)
            self.lora_model = model
            
            # Log trainable params
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            
            get_logger().info(f"  ✓ LoRA applied: {trainable:,}/{total:,} params ({100*trainable/total:.2f}%)")
            
            return model
            
        except Exception as e:
            get_logger().error(f"Failed to apply LoRA: {e}")
            return model

    def setup_optimizer(self, optimizer):
        """
        Wrap optimizer with CPU offload
        Call this AFTER creating your optimizer
        """
        get_logger().info("Setting up CPU-offloaded optimizer...")
        self._cpu_offload_optimizer = CPUOffloadOptimizer(optimizer, pin_memory=True)
        get_logger().info("✓ Optimizer states will be offloaded to CPU RAM")
        return self._cpu_offload_optimizer

    def _log_setup_info(self):
        """Log setup information"""
        get_logger().info("=" * 80)
        get_logger().info("AGGRESSIVE CPU OFFLOAD TRAINER - SETUP COMPLETE")
        get_logger().info("=" * 80)
        get_logger().info(f"CPU Offload: {self.cpu_offload}")
        get_logger().info(f"Offload Folder: {self.offload_folder}")
        get_logger().info(f"Max VRAM per GPU: {self.max_memory_per_gpu}")
        get_logger().info(f"LoRA Enabled: {self.enable_lora}")
        if self.enable_lora:
            get_logger().info(f"LoRA Rank: {self.lora_config_dict['r']}")
        get_logger().info(f"Gradient Accumulation: {self.accumulation_steps} steps")
        get_logger().info(f"VRAM Threshold: {self.memory_monitor.vram_threshold_gb:.2f}GB")
        self.memory_monitor.log_usage(force=True, context="FINAL SETUP")
        get_logger().info("=" * 80)

    def _init_robust_communication(self, kwargs):
        """Initialize communication backend"""
        existing_backend = None
        
        if hasattr(self, 'communication'):
            existing_backend = self.communication
        elif 'communication' in kwargs:
            existing_backend = kwargs['communication']
        
        world_size = int(os.environ.get("HIVEMIND_WORLD_SIZE", 1))
        
        if existing_backend is not None:
            self._robust_backend = create_robust_communication_wrapper(existing_backend)
        elif world_size > 1:
            self._robust_backend = FallbackBackend()
        else:
            self._robust_backend = FallbackBackend()

    def robust_all_gather(self, data, step_info=None):
        """Robust distributed gathering"""
        if not self._robust_backend:
            return {"single_node": data}
        
        if isinstance(step_info, dict):
            step_num = step_info.get('step', self._step_counter)
        elif isinstance(step_info, int):
            step_num = step_info
        else:
            step_num = self._step_counter
            
        self._step_counter = max(self._step_counter, step_num) + 1
        
        try:
            if hasattr(self._robust_backend, 'safe_all_gather'):
                result = self._robust_backend.safe_all_gather(data)
                
                if step_num % 1000 == 0 and step_num > 0:
                    self._log_communication_status(step_num, result)
                
                return result
            elif hasattr(self._robust_backend, 'all_gather_object'):
                return self._robust_backend.all_gather_object(data)
            else:
                return {self._robust_backend.get_id(): data}
                
        except Exception as e:
            return self._handle_communication_error(e, data, step_num)

    def _handle_communication_error(self, error, data, step_num):
        """Handle communication errors"""
        self._communication_errors += 1
        agent_id = getattr(self._robust_backend, 'get_id', lambda: f"emergency_{os.getpid()}")()
        return {agent_id: data}

    def _enable_emergency_mode(self):
        """Enable emergency mode"""
        if not self._emergency_mode_enabled:
            self._emergency_mode_enabled = True
            emergency_disable_dht()

    def _log_communication_status(self, step_num, gathered_results):
        """Log communication status"""
        mode = getattr(self._robust_backend, 'get_training_mode', lambda: 'unknown')()
        get_logger().info(f"Step {step_num}: {len(gathered_results)} agents, mode={mode}")
        self.memory_monitor.log_usage(context=f"STEP {step_num}")

    def train_step_with_communication(self, batch_data, step_num, loss_fn=None, optimizer=None):
        """
        Training step with CPU offload optimization
        """
        try:
            # Memory check
            if self.memory_monitor.is_vram_critical():
                get_logger().warning(f"Step {step_num}: VRAM CRITICAL")
                self.memory_monitor.force_cleanup()
            
            # Use CPU-offloaded optimizer if available
            if optimizer is not None and self._cpu_offload_optimizer is None:
                optimizer = self.setup_optimizer(optimizer)
            elif self._cpu_offload_optimizer is not None:
                optimizer = self._cpu_offload_optimizer
            
            # Forward pass with autocast
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = self.forward(batch_data)
                loss = outputs.get('loss', None)
                
                if loss is None and loss_fn is not None:
                    loss = loss_fn(outputs, batch_data)
                
                # Scale for gradient accumulation
                if self.accumulation_steps > 1:
                    loss = loss / self.accumulation_steps
            
            # Backward pass
            if loss is not None:
                loss.backward()
                
                self._accumulation_counter += 1
                
                # Update only after accumulation
                if self._accumulation_counter >= self.accumulation_steps:
                    if hasattr(self, 'model'):
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    if optimizer is not None:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                    
                    self._accumulation_counter = 0
                    self.memory_monitor.smart_clear(aggressive=False)
            
            # Communication
            step_data = {
                'step': step_num,
                'loss': loss.item() * self.accumulation_steps if loss is not None else 0.0,
                'agent_id': self._robust_backend.get_id(),
                'vram_gb': self.memory_monitor.get_vram_usage()[0],
                'ram_gb': self.memory_monitor.get_ram_usage()[0]
            }
            
            gathered_results = self.robust_all_gather(step_data, step_num)
            processed_outputs = self._process_distributed_results(gathered_results, outputs, step_num)
            
            # Periodic cleanup
            if step_num % 10 == 0:
                self.memory_monitor.smart_clear(aggressive=False)
            
            if step_num % 50 == 0:
                self.memory_monitor.smart_clear(aggressive=True)
                self.memory_monitor.log_usage(force=True, context=f"STEP {step_num}")
            
            return processed_outputs
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                get_logger().error(f"Step {step_num}: OOM - Emergency cleanup")
                self.memory_monitor.force_cleanup()
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                raise e
            else:
                raise e

    def _process_distributed_results(self, gathered_results, original_outputs, step_num):
        """Process distributed results"""
        processed = original_outputs.copy() if isinstance(original_outputs, dict) else {}
        processed['step'] = step_num
        processed['num_agents'] = len(gathered_results)
        processed['gathered_results'] = gathered_results
        
        if len(gathered_results) > 1:
            losses = [d['loss'] for d in gathered_results.values() if 'loss' in d]
            if losses:
                processed['distributed_avg_loss'] = sum(losses) / len(losses)
            
            vrams = [d['vram_gb'] for d in gathered_results.values() if 'vram_gb' in d]
            if vrams:
                processed['avg_vram_gb'] = sum(vrams) / len(vrams)
                processed['max_vram_gb'] = max(vrams)
            
            rams = [d['ram_gb'] for d in gathered_results.values() if 'ram_gb' in d]
            if rams:
                processed['avg_ram_gb'] = sum(rams) / len(rams)
        
        return processed

    # Override communication methods
    def all_gather_object(self, obj):
        try:
            return self.robust_all_gather(obj, self._step_counter)
        except:
            return {self.get_id(): obj}

    def get_id(self):
        try:
            if self._robust_backend:
                return self._robust_backend.get_id()
        except:
            pass
        return f"trainer_{os.getpid()}"

    def _initialize_model(self, enable_gradient_checkpointing: bool = True):
        """Override model initialization - skip device placement for offloaded models"""
        # CPU-offloaded models already have device map, don't move them
        if self.cpu_offload:
            get_logger().info("CPU-offloaded model - skipping device placement")
        else:
            if not self._has_lora(self.model):
                self.model = self.model.to(device=self.device, dtype=self.dtype)
        
        if enable_gradient_checkpointing:
            self._enable_gradient_checkpointing()

    @torch.no_grad()
    def evaluate(self, state: GameState, data_manager: DataManager, reward_manager: RewardManager):
        """Evaluate with memory management"""
        if not self.judge_client:
            return
        
        self.memory_monitor.smart_clear(aggressive=True)
        
        try:
            model_name = self.model.name_or_path
        except AttributeError:
            model_name = "none"

        result = self.judge_client.request_question(
            user_id=state.peer_id,
            round_number=state.round,
            model_name=model_name
        )
        
        if not result:
            return

        prompt = [
            {"role": "system", "content": SYSTEM_PROMPTS["default"]},
            {"role": "user", "content": result["question"]},
        ]
        input_ids = self.processing_class.apply_chat_template(
            prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )

        input_ids = input_ids.to(self.model.device)
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = self.model.generate(input_ids, max_new_tokens=512)
        
        answer = self.processing_class.decode(outputs[0], skip_special_tokens=True)
        
        del outputs, input_ids
        self.memory_monitor.smart_clear(aggressive=True)
        
        self.judge_client.submit_answer(
            session_id=result["session_id"],
            round_number=state.round,
            user_answer=answer
        )

    @torch.no_grad()
    def play_prg_game_logits(self, prg_history_dict: dict) -> dict:
        """Play PRG game with memory management"""
        if not self.judge_client:
            return {'status': PRGGameStatus.ERROR}

        self.memory_monitor.smart_clear(aggressive=True)
        
        game_clue_dict = self.judge_client.get_current_clue()
        
        if not isinstance(game_clue_dict, dict):
            return {'status': PRGGameStatus.ERROR}
        
        game_id = game_clue_dict.get("game_id", -1)
        clue_id = game_clue_dict.get("clue_id", -1)
        rounds_remaining = game_clue_dict.get("rounds_remaining", -1)
        clue = game_clue_dict.get("clue") or ""
        choices = game_clue_dict.get("choices") or []
        
        if any(val < 0 for val in (game_id, clue_id, rounds_remaining)):
            return {'status': PRGGameStatus.NO_ACTIVE_GAME}
        if game_id in prg_history_dict and clue_id <= prg_history_dict[game_id]:
            return {'status': PRGGameStatus.ALREADY_ANSWERED}
        if not clue or not choices:
            return {'status': PRGGameStatus.ERROR}

        try:
            choices_str = ", ".join(choices)
            custom_prompt = f"{clue}\nPossible Answers: {choices_str}\nAnswer:"
            
            prompt = [
                {"role": "system", "content": PRG_SYSTEM_PROMPT_NO_THINKING},
                {"role": "user", "content": custom_prompt},
            ]
            input_ids = self.processing_class.apply_chat_template(
                prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            )

            input_ids = input_ids.to(self.model.device)
            choice_logits = self._get_choice_logits(input_ids, choices)
            choice_idx = torch.argmax(choice_logits).item()
            
            del input_ids, choice_logits
            self.memory_monitor.smart_clear(aggressive=True)
            
            return {
                "game_idx": game_id,
                "clue_idx": clue_id,
                "choice_idx": choice_idx,
                "choice": choices[choice_idx],
                "rounds_remaining": rounds_remaining,
                "status": PRGGameStatus.SUCCESS
            }

        except Exception as e:
            get_logger().error(f"Error in play_prg_game_logits: {e}")
            return {'status': PRGGameStatus.ERROR}

    def _get_choice_logits(self, input_ids: torch.Tensor, choices: List[str]) -> torch.Tensor:
        """Get choice logits with memory management"""
        device = input_ids.device
        batch_size, prompt_len = input_ids.shape
        logits_list = []

        for i, choice in enumerate(choices):
            answer_str = f"<answer>{choice}</answer>"
            choice_ids = self.processing_class(
                answer_str, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(device)

            seq = torch.cat([input_ids, choice_ids], dim=1)
            labels = seq.clone()
            labels[:, :prompt_len] = -100
            
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = self.model(input_ids=seq, labels=labels)
            
            total_log_prob = -outputs.loss * choice_ids.size(1)
            logits_list.append(total_log_prob)
            
            del choice_ids, seq, labels, outputs
            
            if i > 0 and i % 2 == 0:
                self.memory_monitor.smart_clear(aggressive=True)

        return torch.stack(logits_list)

    def save_lora_weights(self, save_path):
        """Save LoRA weights"""
        if not self.enable_lora or self.lora_model is None:
            get_logger().warning("No LoRA to save")
            return
        
        os.makedirs(save_path, exist_ok=True)
        self.lora_model.save_pretrained(save_path)
        get_logger().info(f"LoRA saved: {save_path}")

    def load_lora_weights(self, load_path):
        """Load LoRA weights"""
        if not self.enable_lora:
            return
        
        try:
            self.lora_model = PeftModel.from_pretrained(self.model, load_path)
            get_logger().info(f"LoRA loaded: {load_path}")
        except Exception as e:
            get_logger().error(f"Failed to load LoRA: {e}")

    def merge_and_unload_lora(self):
        """Merge LoRA into base model"""
        if not self.enable_lora or self.lora_model is None:
            return self.model
        
        merged = self.lora_model.merge_and_unload()
        get_logger().info("LoRA merged successfully")
        return merged

    def cleanup(self):
        """Cleanup with memory management"""
        get_logger().info("Starting cleanup...")
        
        if self._robust_backend:
            try:
                if hasattr(self._robust_backend, 'shutdown'):
                    self._robust_backend.shutdown()
            except Exception as e:
                get_logger().warning(f"Communication cleanup error: {e}")
        
        if torch.cuda.is_available():
            self.memory_monitor.force_cleanup()
        
        get_logger().info("Cleanup completed")

    def __del__(self):
        try:
            self.cleanup()
        except:
            pass
