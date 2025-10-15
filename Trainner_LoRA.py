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
    """Monitor and log VRAM usage"""
    
    def __init__(self, threshold_gb=0.5):
        self.threshold_gb = threshold_gb
        self.last_log_time = 0
        self.log_interval = 60  # Log every 60 seconds
        
    def get_vram_usage(self):
        """Get current VRAM usage in GB"""
        if not torch.cuda.is_available():
            return 0.0, 0.0
        
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return allocated, reserved
    
    def should_clear_cache(self):
        """Check if cache should be cleared"""
        allocated, _ = self.get_vram_usage()
        return allocated > self.threshold_gb
    
    def log_usage(self, force=False):
        """Log VRAM usage periodically"""
        current_time = time.time()
        if not force and (current_time - self.last_log_time) < self.log_interval:
            return
        
        allocated, reserved = self.get_vram_usage()
        get_logger().info(f"VRAM: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        self.last_log_time = current_time
    
    def smart_clear(self):
        """Smart cache clearing"""
        if self.should_clear_cache():
            torch.cuda.empty_cache()
            gc.collect()
            allocated, _ = self.get_vram_usage()
            get_logger().info(f"Cache cleared - VRAM: {allocated:.2f}GB")


class GRPOTrainerModule(GRPOLanguageTrainerModule, LoggerMixin):
    """
    Enhanced GRPO Trainer with LoRA, robust communication, and VRAM optimization.
    """

    def __init__(self, models: List[Any], **kwargs):
        """
        Initialize the GRPO trainer module with LoRA and robust communication.
        
        Args:
            models: List of models (can be empty for auto-initialization)
            **kwargs: Configuration parameters including:
                - lora_config: Dict or str preset ('ultra_low', 'low', 'balanced', 'high')
                - enable_lora: bool (default True)
                - vram_threshold_gb: float (default 3.5)
                - gradient_checkpointing: bool (default True)
                - model_id: str (default "Qwen/Qwen2.5-3B-Instruct")
        """
        # Initialize robust communication first
        self._init_robust_communication(kwargs)
        
        # LoRA configuration
        self.enable_lora = kwargs.get("enable_lora", True)
        self.lora_config_dict = self._get_lora_config(kwargs.get("lora_config", "balanced"))
        self.lora_model = None
        
        # VRAM monitoring
        vram_threshold = kwargs.get("vram_threshold_gb", 3.5)
        self.vram_monitor = VRAMMonitor(threshold_gb=vram_threshold)
        
        # Model initialization
        if models:
            for i, model in enumerate(models):
                models[i] = self._setup_model_with_lora(model, kwargs)
        else:
            model = self._create_quantized_model(kwargs)
            models = [model]
        
        super().__init__(models, **kwargs)
        
        # Judge client
        judge_base_url = kwargs.get("judge_base_url", None)
        self.judge_client = JudgeClient(judge_base_url) if judge_base_url else None
        
        # Communication tracking
        self._communication_errors = 0
        self._emergency_mode_enabled = False
        self._last_successful_gather = 0
        self._step_counter = 0
        
        # Log initial setup
        self._log_setup_info()

    def _get_lora_config(self, config_input):
        """Get LoRA configuration from input"""
        if isinstance(config_input, dict):
            return config_input
        elif isinstance(config_input, str):
            return LoRAConfig.get_config(config_input)
        else:
            return LoRAConfig.BALANCED

    def _create_quantized_model(self, kwargs):
        """Create a quantized model with LoRA"""
        model_id = kwargs.get("model_id", "Qwen/Qwen2.5-3B-Instruct")
        
        get_logger().info(f"Creating quantized model: {model_id}")
        
        # BitsAndBytes config for 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        # Apply LoRA if enabled
        if self.enable_lora:
            model = self._apply_lora_to_model(model)
        
        return model

    def _apply_lora_to_model(self, model):
        """Apply LoRA to a model"""
        get_logger().info("Applying LoRA configuration...")
        
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        # Create LoRA config
        lora_config = LoraConfig(**self.lora_config_dict)
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        self.lora_model = model
        
        # Log trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_pct = 100 * trainable_params / total_params
        
        get_logger().info(f"LoRA applied - Trainable: {trainable_params:,} ({trainable_pct:.2f}%)")
        get_logger().info(f"LoRA config: r={self.lora_config_dict['r']}, "
                         f"alpha={self.lora_config_dict['lora_alpha']}, "
                         f"modules={self.lora_config_dict['target_modules']}")
        
        return model

    def _setup_model_with_lora(self, model, kwargs):
        """Setup existing model with quantization check and LoRA"""
        is_quantized = self._is_model_quantized(model)
        
        if not is_quantized:
            get_logger().warning("Model is not quantized, reloading...")
            model = self._reload_with_quantization(model, kwargs)
        
        if self.enable_lora and not self._has_lora(model):
            get_logger().info("Adding LoRA to existing model...")
            model = self._apply_lora_to_model(model)
        
        return model

    def _has_lora(self, model):
        """Check if model already has LoRA"""
        return isinstance(model, PeftModel)

    def _log_setup_info(self):
        """Log setup information"""
        get_logger().info("=" * 60)
        get_logger().info("GRPO Trainer with LoRA - Setup Complete")
        get_logger().info("=" * 60)
        get_logger().info(f"LoRA Enabled: {self.enable_lora}")
        if self.enable_lora:
            get_logger().info(f"LoRA Rank: {self.lora_config_dict['r']}")
            get_logger().info(f"LoRA Alpha: {self.lora_config_dict['lora_alpha']}")
            get_logger().info(f"Target Modules: {self.lora_config_dict['target_modules']}")
        get_logger().info(f"VRAM Threshold: {self.vram_monitor.threshold_gb:.2f}GB")
        self.vram_monitor.log_usage(force=True)
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
        self.vram_monitor.log_usage()

    def train_step_with_communication(self, batch_data, step_num, loss_fn=None, optimizer=None):
        """Enhanced training step with built-in communication and VRAM management"""
        
        try:
            # VRAM check before forward pass
            self.vram_monitor.smart_clear()
            
            # Forward pass
            with torch.cuda.amp.autocast():
                outputs = self.forward(batch_data)
                loss = outputs.get('loss', None)
                
                if loss is None and loss_fn is not None:
                    loss = loss_fn(outputs, batch_data)
            
            # Prepare data for gathering
            step_data = {
                'step': step_num,
                'loss': loss.item() if loss is not None else 0.0,
                'agent_id': self._robust_backend.get_id(),
                'batch_size': len(batch_data) if hasattr(batch_data, '__len__') else 1,
                'timestamp': time.time()
            }
            
            if 'logits' in outputs:
                step_data['has_logits'] = True
            if 'rewards' in outputs:
                step_data['avg_reward'] = torch.mean(outputs['rewards']).item()
            
            # Distributed gathering
            gathered_results = self.robust_all_gather(step_data, step_num)
            
            # Process results
            processed_outputs = self._process_distributed_results(gathered_results, outputs, step_num)
            
            # Backward pass
            if loss is not None:
                loss.backward()
                
                # Gradient clipping
                if hasattr(self, 'model'):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step
                if optimizer is not None:
                    optimizer.step()
                    optimizer.zero_grad()
            
            # VRAM management
            if step_num % 10 == 0:
                self.vram_monitor.smart_clear()
            
            # Log VRAM periodically
            if step_num % 100 == 0:
                self.vram_monitor.log_usage()
            
            return processed_outputs
            
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
            
            for agent_id, data in gathered_results.items():
                if isinstance(data, dict):
                    if 'loss' in data and isinstance(data['loss'], (int, float)):
                        losses.append(data['loss'])
                    if 'batch_size' in data:
                        batch_sizes.append(data['batch_size'])
            
            if losses:
                processed['distributed_avg_loss'] = sum(losses) / len(losses)
                processed['distributed_loss_std'] = torch.std(torch.tensor(losses)).item()
            
            if batch_sizes:
                processed['total_batch_size'] = sum(batch_sizes)
            
            if step_num % 100 == 0:
                avg_loss = processed.get('distributed_avg_loss', 0.0)
                get_logger().info(f"Step {step_num}: Distributed - {len(gathered_results)} agents, loss={avg_loss:.4f}")
        
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

    def _is_model_quantized(self, model) -> bool:
        """Check if model is quantized"""
        if hasattr(model, 'is_quantized') and model.is_quantized:
            return True
        if hasattr(model, 'is_loaded_in_4bit') and model.is_loaded_in_4bit:
            return True
        
        if (hasattr(model, 'config') and 
            hasattr(model.config, 'quantization_config') and 
            model.config.quantization_config is not None):
            qconfig = model.config.quantization_config
            if hasattr(qconfig, 'load_in_4bit') and qconfig.load_in_4bit:
                return True
        
        int_params = sum(p.numel() for p in model.parameters() if 'int' in str(p.dtype).lower())
        total_params = sum(p.numel() for p in model.parameters())
        
        return total_params > 0 and int_params / total_params > 0.1

    def _reload_with_quantization(self, model, kwargs):
        """Reload model with quantization"""
        model_name = getattr(model, 'name_or_path', kwargs.get("model_id", "Qwen/Qwen2.5-3B-Instruct"))
        
        if hasattr(model, 'cpu'):
            model.cpu()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        
        return self._create_quantized_model({'model_id': model_name, **kwargs})

    def _initialize_model(self, enable_gradient_checkpointing: bool = True):
        """Override to handle quantized LoRA models"""
        is_quantized = self._is_model_quantized(self.model)
        
        if not is_quantized:
            self.model = self.model.to(device=self.device, dtype=self.dtype)
        
        if enable_gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            get_logger().info("Gradient checkpointing enabled")

    @torch.no_grad()
    def evaluate(self, state: GameState, data_manager: DataManager, reward_manager: RewardManager):
        """Evaluate with VRAM management"""
        if not self.judge_client:
            return
            
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
        outputs = self.model.generate(input_ids, max_new_tokens=512)
        answer = self.processing_class.decode(outputs[0], skip_special_tokens=True)
        
        # Clear cache after generation
        self.vram_monitor.smart_clear()
        
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
            
            # Clear cache after inference
            self.vram_monitor.smart_clear()
            
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
        """Get choice logits with VRAM management"""
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
            outputs = self.model(input_ids=seq, labels=labels)

            total_log_prob = -outputs.loss * choice_ids.size(1)
            logits_list.append(total_log_prob)
            
            # Clear cache between choices if needed
            if i > 0 and i % 5 == 0:
                self.vram_monitor.smart_clear()

        return torch.stack(logits_list)

    def cleanup(self):
        """Clean shutdown with VRAM cleanup"""
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
        
        # VRAM cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        get_logger().info("Cleanup completed")

    def __del__(self):
        """Ensure cleanup on deletion"""
        try:
            self.cleanup()
        except:
            pass
