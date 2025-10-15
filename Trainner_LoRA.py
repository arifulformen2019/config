from typing import Any, Optional, List
import gc
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from peft import PeftModel
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


class VRAMMonitor:
    """Enhanced VRAM monitor with aggressive cleanup"""
    
    def __init__(self, threshold_gb=0.5, max_vram_gb=20.0):
        self.threshold_gb = threshold_gb
        self.max_vram_gb = max_vram_gb
        self.last_log_time = 0
        self.log_interval = 30  # Log every 30 seconds
        self.peak_vram = 0.0
        self.cleanup_counter = 0
        
    def get_vram_usage(self):
        """Get current VRAM usage in GB"""
        if not torch.cuda.is_available():
            return 0.0, 0.0
        
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        
        # Track peak
        if allocated > self.peak_vram:
            self.peak_vram = allocated
        
        return allocated, reserved
    
    def should_clear_cache(self):
        """Check if cache should be cleared"""
        allocated, _ = self.get_vram_usage()
        return allocated > self.threshold_gb
    
    def is_vram_critical(self):
        """Check if VRAM is critically high"""
        allocated, _ = self.get_vram_usage()
        return allocated > (self.max_vram_gb * 0.85)  # 85% of max
    
    def log_usage(self, force=False, context=""):
        """Log VRAM usage periodically"""
        current_time = time.time()
        if not force and (current_time - self.last_log_time) < self.log_interval:
            return
        
        allocated, reserved = self.get_vram_usage()
        prefix = f"[{context}] " if context else ""
        get_logger().info(
            f"{prefix}VRAM: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, "
            f"Peak: {self.peak_vram:.2f}GB"
        )
        self.last_log_time = current_time
    
    def smart_clear(self, aggressive=False):
        """Smart cache clearing with levels"""
        if self.should_clear_cache() or aggressive:
            self.cleanup_counter += 1
            
            if aggressive or self.is_vram_critical():
                # Aggressive cleanup
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Force garbage collection multiple times
                for _ in range(3):
                    gc.collect()
                
                allocated, _ = self.get_vram_usage()
                get_logger().warning(
                    f"AGGRESSIVE cache clear #{self.cleanup_counter} - VRAM: {allocated:.2f}GB"
                )
            else:
                # Normal cleanup
                torch.cuda.empty_cache()
                gc.collect()
                allocated, _ = self.get_vram_usage()
                if self.cleanup_counter % 10 == 0:
                    get_logger().info(f"Cache cleared #{self.cleanup_counter} - VRAM: {allocated:.2f}GB")
    
    def force_cleanup(self):
        """Force aggressive cleanup"""
        get_logger().warning("FORCING AGGRESSIVE VRAM CLEANUP...")
        
        # Multiple rounds of cleanup
        for i in range(5):
            gc.collect()
            torch.cuda.empty_cache()
            if i < 4:
                time.sleep(0.1)  # Brief pause between cleanups
        
        torch.cuda.synchronize()
        allocated, reserved = self.get_vram_usage()
        get_logger().warning(f"After forced cleanup: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


class GRPOTrainerModule(GRPOLanguageTrainerModule, LoggerMixin):
    """
    Enhanced GRPO Trainer with LoRA, robust communication, and AGGRESSIVE VRAM optimization.
    """

    def __init__(self, models: List[Any], **kwargs):
        """
        Initialize the GRPO trainer module with LoRA and robust communication.
        
        Args:
            models: List of models ĐÃ ĐƯỢC TRUYỀN VÀO SẴN
            **kwargs: Configuration parameters including:
                - lora_config: Dict or str preset ('ultra_low', 'low', 'balanced', 'high')
                - enable_lora: bool (default True)
                - vram_threshold_gb: float (default 2.0) - Giảm xuống để cleanup sớm hơn
                - max_vram_gb: float (default 20.0) - Giới hạn VRAM tối đa
                - gradient_checkpointing: bool (default True)
                - accumulation_steps: int (default 4) - Gradient accumulation
        """
        # Initialize robust communication first
        self._init_robust_communication(kwargs)
        
        # LoRA configuration
        self.enable_lora = kwargs.get("enable_lora", True)
        self.lora_config_dict = self._get_lora_config(kwargs.get("lora_config", "balanced"))
        self.lora_model = None
        
        # VRAM monitoring - AGGRESSIVE SETTINGS
        vram_threshold = kwargs.get("vram_threshold_gb", 2.0)  # Giảm xuống để cleanup sớm
        max_vram = kwargs.get("max_vram_gb", 20.0)
        self.vram_monitor = VRAMMonitor(threshold_gb=vram_threshold, max_vram_gb=max_vram)
        
        # Gradient accumulation for lower memory
        self.accumulation_steps = kwargs.get("accumulation_steps", 4)
        self._accumulation_counter = 0
        
        # Log VRAM trước khi xử lý model
        get_logger().info("=" * 60)
        get_logger().info("INITIALIZING GRPO TRAINER WITH AGGRESSIVE VRAM OPTIMIZATION")
        get_logger().info("=" * 60)
        self.vram_monitor.log_usage(force=True, context="INIT START")
        
        # === Process models - CHỈ THÊM LoRA ===
        if not models or len(models) == 0:
            raise ValueError("Models list is empty! Please provide pre-loaded models.")
        
        get_logger().info(f"Received {len(models)} pre-loaded model(s)")
        
        # Clear cache trước khi xử lý
        self.vram_monitor.smart_clear(aggressive=True)
        
        processed_models = []
        for idx, model in enumerate(models):
            get_logger().info(f"Processing model {idx + 1}/{len(models)}...")
            
            if self._has_lora(model):
                get_logger().info(f"  → Model {idx + 1} already has LoRA")
                processed_models.append(model)
            elif self.enable_lora:
                get_logger().info(f"  → Adding LoRA to model {idx + 1}...")
                lora_model = self._apply_lora_to_model(model)
                processed_models.append(lora_model)
                # Clear sau khi thêm LoRA
                self.vram_monitor.smart_clear(aggressive=True)
            else:
                get_logger().info(f"  → Using model {idx + 1} as-is (LoRA disabled)")
                processed_models.append(model)
            
            self.vram_monitor.log_usage(force=True, context=f"MODEL {idx+1}")
        
        # Gọi parent class init
        super().__init__(processed_models, **kwargs)
        
        # Enable gradient checkpointing - CRITICAL for VRAM
        if kwargs.get("gradient_checkpointing", True):
            self._enable_gradient_checkpointing()
        
        # Judge client initialization
        judge_base_url = kwargs.get("judge_base_url", None)
        self.judge_client = JudgeClient(judge_base_url) if judge_base_url else None
        
        # Communication tracking
        self._communication_errors = 0
        self._emergency_mode_enabled = False
        self._last_successful_gather = 0
        self._step_counter = 0
        
        # Training state tracking
        self._last_cleanup_step = 0
        self._emergency_cleanup_triggered = False
        
        # Final aggressive cleanup
        self.vram_monitor.force_cleanup()
        
        # Log final setup
        self._log_setup_info()

    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to save VRAM"""
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            try:
                self.model.gradient_checkpointing_enable()
                get_logger().info("✓ Gradient checkpointing ENABLED (saves ~30-50% VRAM)")
            except Exception as e:
                get_logger().warning(f"Could not enable gradient checkpointing: {e}")
        else:
            get_logger().warning("Model does not support gradient checkpointing")

    def _get_lora_config(self, config_input):
        """Get LoRA configuration from input"""
        if isinstance(config_input, dict):
            return config_input
        elif isinstance(config_input, str):
            return LoRAConfig.get_config(config_input)
        else:
            return LoRAConfig.BALANCED

    def _has_lora(self, model) -> bool:
        """Check if model already has LoRA adapter"""
        if isinstance(model, PeftModel):
            return True
        if hasattr(model, 'peft_config') and model.peft_config is not None:
            return True
        model_str = str(type(model)).lower()
        if 'peft' in model_str or 'lora' in model_str:
            return True
        return False

    def _apply_lora_to_model(self, model):
        """Apply LoRA to a model - KHÔNG RELOAD MODEL"""
        if self._has_lora(model):
            get_logger().warning("Model already has LoRA - skipping")
            return model
        
        get_logger().info("Applying LoRA configuration...")
        get_logger().info(f"  LoRA rank (r): {self.lora_config_dict['r']}")
        get_logger().info(f"  LoRA alpha: {self.lora_config_dict['lora_alpha']}")
        get_logger().info(f"  Target modules: {self.lora_config_dict['target_modules']}")
        
        try:
            # Prepare model for k-bit training
            get_logger().info("  Preparing model for k-bit training...")
            model = prepare_model_for_kbit_training(model)
            
            # Create LoRA config
            lora_config = LoraConfig(**self.lora_config_dict)
            
            # Apply LoRA
            get_logger().info("  Applying LoRA adapter...")
            model = get_peft_model(model, lora_config)
            self.lora_model = model
            
            # Log trainable parameters
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            trainable_pct = 100 * trainable_params / total_params
            
            get_logger().info(f"  ✓ LoRA applied successfully")
            get_logger().info(f"  Trainable: {trainable_params:,} ({trainable_pct:.2f}%)")
            get_logger().info(f"  Total: {total_params:,}")
            
            return model
            
        except Exception as e:
            get_logger().error(f"Failed to apply LoRA: {e}")
            return model

    def _log_setup_info(self):
        """Log setup information"""
        get_logger().info("=" * 60)
        get_logger().info("GRPO TRAINER SETUP COMPLETE")
        get_logger().info("=" * 60)
        get_logger().info(f"LoRA Enabled: {self.enable_lora}")
        if self.enable_lora:
            get_logger().info(f"LoRA Rank: {self.lora_config_dict['r']}")
            get_logger().info(f"LoRA Alpha: {self.lora_config_dict['lora_alpha']}")
        get_logger().info(f"Gradient Accumulation Steps: {self.accumulation_steps}")
        get_logger().info(f"VRAM Threshold: {self.vram_monitor.threshold_gb:.2f}GB")
        get_logger().info(f"Max VRAM: {self.vram_monitor.max_vram_gb:.2f}GB")
        self.vram_monitor.log_usage(force=True, context="SETUP COMPLETE")
        get_logger().info("=" * 60)

    def _init_robust_communication(self, kwargs):
        """Initialize robust communication backend"""
        existing_backend = None
        
        if hasattr(self, 'communication'):
            existing_backend = self.communication
        elif 'communication' in kwargs:
            existing_backend = kwargs['communication']
        
        world_size = int(os.environ.get("HIVEMIND_WORLD_SIZE", 1))
        
        if existing_backend is not None:
            get_logger().info(f"Found existing communication backend, wrapping with robust features")
            self._robust_backend = create_robust_communication_wrapper(existing_backend)
        elif world_size > 1:
            get_logger().info(f"Distributed environment detected (world_size={world_size}) but no backend found - using fallback")
            self._robust_backend = FallbackBackend()
        else:
            get_logger().info("Single-node environment - using fallback backend")
            self._robust_backend = FallbackBackend()

    def robust_all_gather(self, data, step_info=None):
        """Public method for robust distributed gathering"""
        
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
                
                if len(result) > 1 and self._communication_errors > 0:
                    get_logger().info(f"Communication recovered at step {step_num}")
                    self._communication_errors = 0
                    self._last_successful_gather = step_num
                elif len(result) > 1:
                    self._last_successful_gather = step_num
                
                return result
                
            elif hasattr(self._robust_backend, 'all_gather_object'):
                return self._robust_backend.all_gather_object(data)
            else:
                return {self._robust_backend.get_id(): data}
                
        except Exception as e:
            return self._handle_communication_error(e, data, step_num)

    def _handle_communication_error(self, error, data, step_num):
        """Handle communication errors gracefully"""
        
        self._communication_errors += 1
        error_msg = str(error)
        
        if self._communication_errors <= 5 or self._communication_errors % 100 == 0:
            get_logger().error(f"Step {step_num}: Communication failed ({self._communication_errors}): {error_msg}")
        
        critical_patterns = [
            "ran out of input", "pipe", "connection", "dht", "hivemind", 
            "eof", "resource temporarily unavailable", "blocking"
        ]
        
        if any(pattern in error_msg.lower() for pattern in critical_patterns):
            if self._communication_errors <= 3:
                get_logger().warning(f"Critical communication error detected at step {step_num}")
            self._enable_emergency_mode()
        
        agent_id = getattr(self._robust_backend, 'get_id', lambda: f"emergency_{os.getpid()}")()
        return {agent_id: data}

    def _enable_emergency_mode(self):
        """Enable emergency mode"""
        if not self._emergency_mode_enabled:
            self._emergency_mode_enabled = True
            get_logger().warning("EMERGENCY MODE ENABLED - switching to single-node training")
            
            if hasattr(self._robust_backend, 'backend'):
                if hasattr(self._robust_backend.backend, '_emergency_mode'):
                    self._robust_backend.backend._emergency_mode = True
            
            emergency_disable_dht()

    def _log_communication_status(self, step_num, gathered_results):
        """Log communication status"""
        mode = getattr(self._robust_backend, 'get_training_mode', lambda: 'unknown')()
        
        get_logger().info(f"Step {step_num} Communication Status:")
        get_logger().info(f"  Agents: {len(gathered_results)}")
        get_logger().info(f"  Mode: {mode}")
        get_logger().info(f"  Emergency: {self._emergency_mode_enabled}")
        
        # Also log VRAM
        self.vram_monitor.log_usage(context=f"STEP {step_num}")

    def train_step_with_communication(self, batch_data, step_num, loss_fn=None, optimizer=None):
        """
        Enhanced training step with AGGRESSIVE VRAM management and gradient accumulation
        """
        
        try:
            # === VRAM CHECK - CRITICAL ===
            if self.vram_monitor.is_vram_critical():
                get_logger().warning(f"Step {step_num}: VRAM CRITICAL - forcing cleanup")
                self.vram_monitor.force_cleanup()
                self._emergency_cleanup_triggered = True
            elif (step_num - self._last_cleanup_step) >= 5:
                # Cleanup mỗi 5 steps
                self.vram_monitor.smart_clear(aggressive=False)
                self._last_cleanup_step = step_num
            
            # === FORWARD PASS ===
            # Sử dụng autocast để giảm memory
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = self.forward(batch_data)
                loss = outputs.get('loss', None)
                
                if loss is None and loss_fn is not None:
                    loss = loss_fn(outputs, batch_data)
                
                # Scale loss for gradient accumulation
                if self.accumulation_steps > 1:
                    loss = loss / self.accumulation_steps
            
            # === BACKWARD PASS với gradient accumulation ===
            if loss is not None:
                loss.backward()
                
                self._accumulation_counter += 1
                
                # Chỉ update weights sau N accumulation steps
                if self._accumulation_counter >= self.accumulation_steps:
                    # Gradient clipping
                    if hasattr(self, 'model'):
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    # Optimizer step
                    if optimizer is not None:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)  # set_to_none=True saves memory
                    
                    # Reset counter
                    self._accumulation_counter = 0
                    
                    # Clear cache after optimizer step
                    self.vram_monitor.smart_clear(aggressive=False)
            
            # === COMMUNICATION ===
            step_data = {
                'step': step_num,
                'loss': loss.item() * self.accumulation_steps if loss is not None else 0.0,  # Un-scale loss
                'agent_id': self._robust_backend.get_id(),
                'batch_size': len(batch_data) if hasattr(batch_data, '__len__') else 1,
                'timestamp': time.time(),
                'vram_gb': self.vram_monitor.get_vram_usage()[0]
            }
            
            if 'logits' in outputs:
                step_data['has_logits'] = True
            if 'rewards' in outputs:
                step_data['avg_reward'] = torch.mean(outputs['rewards']).item()
            
            # Distributed gathering
            gathered_results = self.robust_all_gather(step_data, step_num)
            
            # Process results
            processed_outputs = self._process_distributed_results(gathered_results, outputs, step_num)
            
            # === AGGRESSIVE CLEANUP SCHEDULE ===
            if step_num % 10 == 0:
                self.vram_monitor.smart_clear(aggressive=False)
            
            if step_num % 50 == 0:
                self.vram_monitor.smart_clear(aggressive=True)
                self.vram_monitor.log_usage(force=True, context=f"STEP {step_num}")
            
            # Emergency cleanup nếu cần
            if self._emergency_cleanup_triggered and step_num % 100 == 0:
                get_logger().warning(f"Step {step_num}: Running emergency VRAM check")
                self.vram_monitor.force_cleanup()
                self._emergency_cleanup_triggered = False
            
            return processed_outputs
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                get_logger().error(f"Step {step_num}: OUT OF MEMORY - Emergency cleanup")
                self.vram_monitor.force_cleanup()
                
                # Try to recover
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                
                torch.cuda.empty_cache()
                
                raise e
            else:
                raise e
                
        except Exception as e:
            get_logger().error(f"Training step {step_num} failed: {e}")
            return {
                'loss': torch.tensor(0.0) if 'loss' not in locals() else loss,
                'step': step_num,
                'error': str(e),
                'gathered_results': {self._robust_backend.get_id(): step_data} if 'step_data' in locals() else {}
            }

    def _process_distributed_results(self, gathered_results, original_outputs, step_num):
        """Process results from distributed gathering"""
        
        processed = original_outputs.copy() if isinstance(original_outputs, dict) else {}
        processed['step'] = step_num
        processed['num_agents'] = len(gathered_results)
        processed['gathered_results'] = gathered_results
        
        if len(gathered_results) == 1:
            processed['training_mode'] = 'single_node'
            if step_num % 500 == 0:
                get_logger().info(f"Step {step_num}: Single-node training")
        else:
            processed['training_mode'] = 'distributed'
            
            losses = []
            batch_sizes = []
            vram_usage = []
            
            for agent_id, data in gathered_results.items():
                if isinstance(data, dict):
                    if 'loss' in data and isinstance(data['loss'], (int, float)):
                        losses.append(data['loss'])
                    if 'batch_size' in data:
                        batch_sizes.append(data['batch_size'])
                    if 'vram_gb' in data:
                        vram_usage.append(data['vram_gb'])
            
            if losses:
                processed['distributed_avg_loss'] = sum(losses) / len(losses)
                processed['distributed_loss_std'] = torch.std(torch.tensor(losses)).item()
            
            if batch_sizes:
                processed['total_batch_size'] = sum(batch_sizes)
            
            if vram_usage:
                processed['avg_vram_gb'] = sum(vram_usage) / len(vram_usage)
                processed['max_vram_gb'] = max(vram_usage)
            
            if step_num % 100 == 0:
                avg_loss = processed.get('distributed_avg_loss', 0.0)
                avg_vram = processed.get('avg_vram_gb', 0.0)
                get_logger().info(
                    f"Step {step_num}: {len(gathered_results)} agents, "
                    f"loss={avg_loss:.4f}, VRAM={avg_vram:.2f}GB"
                )
        
        return processed

    def save_lora_weights(self, save_path):
        """Save only LoRA weights"""
        if not self.enable_lora or self.lora_model is None:
            get_logger().warning("No LoRA model to save")
            return
        
        os.makedirs(save_path, exist_ok=True)
        self.lora_model.save_pretrained(save_path)
        get_logger().info(f"LoRA weights saved to: {save_path}")

    def load_lora_weights(self, load_path):
        """Load LoRA weights"""
        if not self.enable_lora:
            get_logger().warning("LoRA not enabled")
            return
        
        try:
            self.lora_model = PeftModel.from_pretrained(self.model, load_path)
            get_logger().info(f"LoRA weights loaded from: {load_path}")
        except Exception as e:
            get_logger().error(f"Failed to load LoRA weights: {e}")

    def merge_and_unload_lora(self):
        """Merge LoRA weights back into base model"""
        if not self.enable_lora or self.lora_model is None:
            get_logger().warning("No LoRA model to merge")
            return self.model
        
        get_logger().info("Merging LoRA weights into base model...")
        merged_model = self.lora_model.merge_and_unload()
        get_logger().info("LoRA weights merged successfully")
        return merged_model

    # Override communication methods
    def all_gather_object(self, obj):
        try:
            return self.robust_all_gather(obj, self._step_counter)
        except Exception as e:
            get_logger().error(f"all_gather_object failed: {e}")
            return {self.get_id(): obj}

    def get_id(self):
        try:
            if self._robust_backend:
                return self._robust_backend.get_id()
        except Exception as e:
            get_logger().warning(f"Failed to get ID: {e}")
        return f"trainer_{os.getpid()}"

    def set_communication_backend(self, backend):
        if backend is not None:
            self._robust_backend = create_robust_communication_wrapper(backend)
            get_logger().info("Communication backend updated")

    def _initialize_model(self, enable_gradient_checkpointing: bool = True):
        """Override to handle LoRA models properly"""
        if self._has_lora(self.model):
            get_logger().info("LoRA model detected - skipping device placement")
        else:
            self.model = self.model.to(device=self.device, dtype=self.dtype)
        
        if enable_gradient_checkpointing:
            self._enable_gradient_checkpointing()

    @torch.no_grad()
    def evaluate(self, state: GameState, data_manager: DataManager, reward_manager: RewardManager):
        """Evaluate with VRAM management"""
        if not self.judge_client:
            return
        
        # Clear cache trước eval
        self.vram_monitor.smart_clear(aggressive=True)
            
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
            prompt,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        input_ids = input_ids.to(self.model.device)
        
        # Generate with autocast
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = self.model.generate(input_ids, max_new_tokens=512)
        
        answer = self.processing_class.decode(outputs[0], skip_special_tokens=True)
        
        # Clear cache sau generation
        del outputs
        del input_ids
        self.vram_monitor.smart_clear(aggressive=True)
        
        self.judge_client.submit_answer(
            session_id=result["session_id"],
            round_number=state.round,
            user_answer=answer
        )

    @torch.no_grad()
    def play_prg_game_logits(self, prg_history_dict: dict) -> dict:
        """Play PRG game with VRAM management"""
        if not self.judge_client:
            return {'status': PRGGameStatus.ERROR}

        # Clear cache trước inference
        self.vram_monitor.smart_clear(aggressive=True)

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
        if not clue or not isinstance(choices, list) or not choices:
            return {'status': PRGGameStatus.ERROR}

        try:
            choices_str = ", ".join(choices)
            custom_prompt = f"{clue}\nPossible Answers: {choices_str}\nAnswer:"
            
            prompt = [
                {"role": "system", "content": PRG_SYSTEM_PROMPT_NO_THINKING},
                {"role": "user", "content": custom_prompt},
            ]
            input_ids = self.processing_class.apply_chat_template(
                prompt,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )

            input_ids = input_ids.to(self.model.device)
            choice_logits = self._get_choice_logits(input_ids, choices)
            choice_idx = torch.argmax(choice_logits).item()
            
            # Clear cache sau inference
            del input_ids
            del choice_logits
            self.vram_monitor.smart_clear(aggressive=True)
            
            return {
                "game_idx": game_id,
                "clue_idx": clue_id,
                "choice_idx": choice_idx,
                "choice": choices[choice_idx],
                "rounds_remaining": rounds_remaining,
                "status": PRGGameStatus.SUCCESS
            }

        except Exception as e:
            get_logger().info(f"Error while computing logits for choices: {e}")
            return {'status': PRGGameStatus.ERROR}

    def _get_choice_logits(self, input_ids: torch.Tensor, choices: List[str]) -> torch.Tensor:
        """Get choice logits with AGGRESSIVE VRAM management"""
        device = input_ids.device
        batch_size, prompt_len = input_ids.shape
        logits_list = []

        for i, choice in enumerate(choices):
            answer_str = f"<answer>{choice}</answer>"
            choice_ids = self.processing_class(
                answer_str,
                return_tensors="pt",
                add_special_tokens=False
            ).input_ids.to(device)

            seq = torch.cat([input_ids, choice_ids], dim=1)
            labels = seq.clone()
            labels[:, :prompt_len] = -100
            
            # Use autocast
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = self.model(input_ids=seq, labels=labels)
            
            total_log_prob = -outputs.loss * choice_ids.size(1)
            logits_list.append(total_log_prob)
            
            # Clear intermediates
            del choice_ids, seq, labels, outputs
            
            # Aggressive cleanup between choices
            if i > 0 and i % 2 == 0:  # Every 2 choices
                self.vram_monitor.smart_clear(aggressive=True)

        return torch.stack(logits_list)

    def cleanup(self):
        """Clean shutdown with AGGRESSIVE VRAM cleanup"""
        get_logger().info("Starting cleanup...")
        
        # Communication cleanup
        if self._robust_backend:
            try:
                if hasattr(self._robust_backend, 'backend'):
                    if hasattr(self._robust_backend.backend, 'shutdown'):
                        self._robust_backend.backend.shutdown()
                elif hasattr(self._robust_backend, 'shutdown'):
                    self._robust_backend.shutdown()
            except Exception as e:
                get_logger().warning(f"Communication cleanup error: {e}")
        
        # Aggressive VRAM cleanup
        if torch.cuda.is_available():
            self.vram_monitor.force_cleanup()
        
        get_logger().info("Cleanup completed")

    def __del__(self):
        """Ensure cleanup on deletion"""
        try:
            self.cleanup()
        except:
            pass
