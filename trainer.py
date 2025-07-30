from typing import Any, List
import requests
import torch
import torch.utils.data
import gc
from genrl.data import DataManager
from genrl.logging_utils.global_defs import get_logger
from genrl.logging_utils.ml_logger import LoggerMixin
from genrl.rewards import RewardManager
from genrl.state import GameState
from genrl.trainer.grpo_trainer import GRPOLanguageTrainerModule
from reasoning_gym.utils import SYSTEM_PROMPTS
from vllm import LLM


class GRPOTrainerModule(GRPOLanguageTrainerModule, LoggerMixin):
    """
    GRPO trainer with simple memory cleanup - no threading
    """
    def __init__(self, models: List[Any], **kwargs):
        super().__init__(models, **kwargs)
        self.judge_base_url = kwargs.get("judge_base_url", None)
        # Simple counter for cleanup
        self.step_counter = 0
        get_logger().info("PATCH: Memory-optimized trainer loaded (simple version)")

    @torch.no_grad()
    def evaluate(self, state: GameState, data_manager: DataManager, reward_manager: RewardManager):
        """Evaluation with immediate memory cleanup"""
        self.step_counter += 1
        
        base_url = self.judge_base_url
        if base_url:
            try:
                model_name = self.model.name_or_path
            except AttributeError:
                model_name = "none"
            try:
                request_data = {
                    "user_id": state.peer_id,
                    "round_number": state.round,
                    "model_name": model_name,
                }
                response = requests.post(f"{base_url}/request-question/", json=request_data)
                
                if response.status_code == 200:
                    result = response.json()
                    get_logger().debug(f'Received question: {result["question"]}')
                else:
                    get_logger().debug(f"Failed to receive question: {response.status_code}")
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
                
                # IMMEDIATE cleanup - most important part
                try:
                    del input_ids, outputs, prompt
                    gc.collect()
                except:
                    pass
                
                session_id = result["session_id"]
                submission_data = {
                    "session_id": session_id,
                    "round_number": state.round,
                    "user_answer": answer,
                }
                response = requests.post(f"{base_url}/submit-answer/", json=submission_data)
                
                if response.status_code == 200:
                    result = response.json()
                    get_logger().debug(f"Score: {result['score']}")
                    
                    # Light cleanup after successful evaluation
                    if self.step_counter % 10 == 0:
                        gc.collect()
                        get_logger().info(f"PATCH: Memory cleanup at step {self.step_counter}")
                    
                    return
                else:
                    get_logger().debug(f"Failed to submit answer: {response.status_code}")
                    return
                    
            except Exception as e:
                get_logger().debug(f"Failed to evaluate: {e}")
                return
        else:
            return
