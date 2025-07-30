import logging
import os
import sys
import time
import gc  # ✅ ADD: Memory cleanup
import threading  # ✅ ADD: Background cleanup
from collections import defaultdict
from pathlib import Path

from genrl.blockchain import SwarmCoordinator
from genrl.communication import Communication
from genrl.communication.hivemind.hivemind_backend import HivemindBackend
from genrl.data import DataManager
from genrl.game import BaseGameManager
from genrl.game.game_manager import DefaultGameManagerMixin
from genrl.logging_utils.global_defs import get_logger
from genrl.logging_utils.system_utils import get_system_info
from genrl.rewards import RewardManager
from genrl.roles import RoleManager
from genrl.state import GameState
from genrl.trainer import TrainerModule
from huggingface_hub import login, whoami

from rgym_exp.src.utils.name_utils import get_name_from_peer_id

# ✅ ADD: Colorful logging imports
try:
    from colorama import Fore, Style, init
    init(autoreset=True)  # Auto reset colors
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False
    # Fallback to no colors
    class MockFore:
        CYAN = GREEN = RED = YELLOW = MAGENTA = BLUE = ""
    class MockStyle:
        RESET_ALL = ""
    Fore = MockFore()
    Style = MockStyle()

try:
    import psutil  # ✅ ADD: Memory monitoring
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class SwarmGameManager(BaseGameManager, DefaultGameManagerMixin):
    """GameManager that orchestrates a game using a SwarmCoordinator."""

    def __init__(
        self,
        coordinator: SwarmCoordinator,
        max_stage: int,
        max_round: int,
        game_state: GameState,
        reward_manager: RewardManager,
        trainer: TrainerModule,
        data_manager: DataManager,
        communication: Communication,
        role_manager: RoleManager | None = None,
        run_mode: str = "train",
        log_dir: str = "logs",
        hf_token: str | None = None,
        hf_push_frequency: int = 20,
        **kwargs,
    ):

        super().__init__(
            max_stage=max_stage,
            max_round=max_round,
            game_state=game_state,
            reward_manager=reward_manager,
            trainer=trainer,
            data_manager=data_manager,
            communication=communication,
            role_manager=role_manager,
            run_mode=run_mode,
        )

        # ✅ ADD: Memory management initialization
        self.round_counter = 0
        self.agent_block_counter = 0
        self.last_memory_cleanup = time.time()
        self.memory_threshold_gb = 20  # Alert threshold
        self._start_memory_monitor()

        assert isinstance(self.communication, HivemindBackend)
        self.train_timeout = 60 * 60 * 24 * 31  # 1 month

        # Logging Setup
        self.peer_id = self.communication.get_id()
        self.state.peer_id = self.peer_id
        self.animal_name = get_name_from_peer_id(self.peer_id, True)
        format_msg = f"[{self.animal_name}] %(asctime)s %(levelname)s: %(message)s"
        logging.basicConfig(level=logging.INFO, format=format_msg)
        formatter = logging.Formatter(format_msg)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f"training_{self.animal_name}.log")
        )
        file_handler.setFormatter(formatter)
        _LOG = get_logger()
        _LOG.addHandler(file_handler)

        # Register peer_id and get current round from the chain
        self.coordinator = coordinator
        self.coordinator.register_peer(self.peer_id)
        round, _ = self.coordinator.get_round_and_stage()
        self.state.round = round
        self.communication.step_ = (
            self.state.round
        )  # initialize communication module to contract's round

        # --- VLLM INTEGRATION START ---
        # Safely get the model name first, then use it.
        model_name = "UnknownModel"
        
        # Check if we are in vLLM mode
        if hasattr(self.trainer, "use_vllm") and self.trainer.use_vllm:
            # In vLLM mode, use the name we saved in the trainer
            model_name = getattr(self.trainer, "model_name", "vLLM_Model")
        else:
            # In standard training mode, safely access the config attribute
            config_obj = getattr(getattr(self.trainer, "model", None), "config", None)
            if config_obj:
                model_name = getattr(config_obj, "_name_or_path", "UnknownModel")
        
        get_logger().info(f"Using Model: {model_name}")

        # enable push to HF if token was provided
        self.hf_token = hf_token
        if self.hf_token not in [None, "None"]:
            # This block should only run if we can actually push, which means we're NOT in vLLM mode.
            if not (hasattr(self.trainer, "use_vllm") and self.trainer.use_vllm):
                try:
                    username = whoami(token=self.hf_token)["name"]
                    model_name_suffix = model_name.split("/")[-1]
                    hub_model_id = f"{username}/{model_name_suffix}-Gensyn-Swarm-{self.animal_name}"
                    
                    self.trainer.args.hub_model_id = hub_model_id
                    self.trainer.args.push_to_hub = True
                    self.trainer.args.hub_token = self.hf_token
                    self.hf_push_frequency = hf_push_frequency
                    get_logger().info("Logging into Hugging Face Hub...")
                    login(self.hf_token)
                except Exception as e:
                    get_logger().warning(f"Could not set up Hugging Face push. Error: {e}")
            else:
                get_logger().info("Hugging Face push is disabled in vLLM mode.")
        # --- VLLM INTEGRATION END ---

        get_logger().info(
            f"🐱 Hello 🐈 [{get_name_from_peer_id(self.peer_id)}] 🦮 [{self.peer_id}]!"
        )
        get_logger().info(f"bootnodes: {kwargs.get('bootnodes', [])}")

        with open(os.path.join(log_dir, f"system_info.txt"), "w") as f:
            f.write(get_system_info())

        self.batched_signals = 0.0
        self.time_since_submit = time.time() #seconds
        self.submit_period = 1.0 #hours  # ✅ CÓ THỂ THAY ĐỔI: 1.0, 2.0, etc.
        self.submitted_this_round = False

        # ✅ ADD: Log memory optimization loaded
        get_logger().info(
            f"{Fore.GREEN}🚀 MANAGER PATCH: Memory-optimized manager initialized with colorful logging!{Style.RESET_ALL}"
        )

    # ✅ ADD: Memory monitoring methods
    def _start_memory_monitor(self):
        """Start background memory monitoring thread"""
        def memory_monitor():
            while True:
                try:
                    self._check_memory_pressure()
                    time.sleep(300)  # Check every 5 minutes
                except Exception as e:
                    get_logger().error(f"Memory monitor error: {e}")
                    time.sleep(600)  # Back off on error
                    
        monitor_thread = threading.Thread(target=memory_monitor, daemon=True)
        monitor_thread.start()

    def _check_memory_pressure(self):
        """Check and report memory pressure"""
        if not PSUTIL_AVAILABLE:
            return
            
        try:
            process = psutil.Process()
            memory_gb = process.memory_info().rss / 1024 / 1024 / 1024
            
            if memory_gb > self.memory_threshold_gb:
                get_logger().warning(
                    f"{Fore.YELLOW}⚠️ MANAGER PATCH: High memory usage: {memory_gb:.1f}GB{Style.RESET_ALL}"
                )
                self._aggressive_cleanup()
                
        except Exception as e:
            get_logger().error(f"Memory check failed: {e}")

    def _aggressive_cleanup(self):
        """Perform aggressive memory cleanup"""
        try:
            # Multiple garbage collection cycles
            for i in range(5):
                collected = gc.collect()
                if i == 0:
                    get_logger().info(
                        f"{Fore.CYAN}🧹 MANAGER PATCH: GC collected {collected} objects{Style.RESET_ALL}"
                    )
            
            # Clear accumulated rewards data if too large
            try:
                if hasattr(self, 'rewards') and len(self.rewards) > 100:
                    # Keep only recent 50 rounds
                    recent_rewards = dict(list(self.rewards.items())[-50:])
                    self.rewards.clear()
                    self.rewards.update(recent_rewards)
                    get_logger().info(
                        f"{Fore.CYAN}🧹 MANAGER PATCH: Cleaned old rewards data{Style.RESET_ALL}"
                    )
            except Exception:
                pass
                
            # Clear trainer cache if available
            try:
                if hasattr(self.trainer, '_cleanup_memory'):
                    self.trainer._cleanup_memory()
            except Exception:
                pass
                
        except Exception as e:
            get_logger().error(f"Aggressive cleanup failed: {e}")

    def _get_total_rewards_by_agent(self):
        rewards_by_agent = defaultdict(int)
        for stage in range(self.state.stage):
            rewards = self.rewards[stage]
            for agent_id, agent_rewards in rewards.items():
                for batch_id, batch_rewards in agent_rewards.items():
                    tot = 0
                    for generation_rewards in batch_rewards:
                        tot += sum(generation_rewards)
                    rewards_by_agent[agent_id] += tot

        return rewards_by_agent

    def _get_my_rewards(self, signal_by_agent):
        if len(signal_by_agent) == 0:
            return 0
        if self.peer_id in signal_by_agent:
            my_signal = signal_by_agent[self.peer_id]
        else:
            my_signal = 0
        my_signal = (my_signal + 1) * (my_signal > 0) + my_signal * (
            my_signal <= 0
        )
        return my_signal

    def _try_submit_to_chain(self, signal_by_agent):
        elapsed_time_hours = (time.time() - self.time_since_submit) / 3600
        if elapsed_time_hours > self.submit_period:
            try:
                # 🎯 ENHANCED LOGGING - Before submit
                get_logger().info(
                    f"{Fore.CYAN}🚀 [SUBMIT STARTING] Round: {self.state.round} | "
                    f"Points: {int(self.batched_signals)} | "
                    f"Agent: {self.animal_name}{Style.RESET_ALL}"
                )
                
                # Submit reward to blockchain
                self.coordinator.submit_reward(
                    self.state.round, 0, int(self.batched_signals), self.peer_id
                )
                
                # Determine winner
                if len(signal_by_agent) > 0:
                    max_agent, max_signal = max(signal_by_agent.items(), key=lambda x: x[1])
                    try:
                        winner_name = get_name_from_peer_id(max_agent, True) if max_agent != self.peer_id else self.animal_name
                    except:
                        winner_name = "unknown-agent"
                else:
                    max_agent = self.peer_id
                    winner_name = self.animal_name
                    max_signal = int(self.batched_signals)

                # Submit winners
                self.coordinator.submit_winners(self.state.round, [max_agent], self.peer_id)
                
                # 🎯 ENHANCED LOGGING - Success
                get_logger().info(
                    f"{Fore.GREEN}✅ [SUBMIT SUCCESS] 🎉 POINTS SUBMITTED! 🎉\n"
                    f"   💰 Points Sent: {int(self.batched_signals)}\n"
                    f"   🏆 Round Winner: {winner_name} ({max_signal} pts)\n"
                    f"   🕐 Next Submit: {self.submit_period} hours\n"
                    f"   🐾 Agent: {self.animal_name}{Style.RESET_ALL}"
                )
                
                # Reset counters
                submitted_points = int(self.batched_signals)
                self.batched_signals = 0.0
                self.time_since_submit = time.time()
                self.submitted_this_round = True
                
                # 🎯 Additional success log with stats
                get_logger().info(
                    f"{Fore.BLUE}📊 [STATS] Total Submitted: {submitted_points} | "
                    f"Round: {self.state.round} | "
                    f"Uptime: {elapsed_time_hours:.1f}h{Style.RESET_ALL}"
                )
                
            except Exception as e:
                # 🎯 ENHANCED ERROR LOGGING
                get_logger().error(
                    f"{Fore.RED}❌ [SUBMIT FAILED] 💥 SUBMISSION ERROR! 💥\n"
                    f"   🚨 Error: {str(e)}\n"
                    f"   💰 Points Lost: {int(self.batched_signals)}\n"
                    f"   🔄 Will Retry Next Cycle\n"
                    f"   🐾 Agent: {self.animal_name}{Style.RESET_ALL}"
                )
                
                get_logger().exception(
                    "Failed to submit to chain.\n"
                    "This is most likely transient and will recover.\n"
                    "There is no need to kill the program.\n"
                    "If you encounter this error, please report it to Gensyn by\n"
                    "filing a github issue here: https://github.com/gensyn-ai/rl-swarm/issues/ \n"
                    "including the full stacktrace."
                )
        else:
            # 🎯 WAITING LOG - Only show occasionally to avoid spam
            remaining_hours = self.submit_period - elapsed_time_hours
            remaining_minutes = remaining_hours * 60
            
            # Only log every 30 minutes when waiting
            if not hasattr(self, '_last_waiting_log'):
                self._last_waiting_log = 0
            
            if time.time() - self._last_waiting_log > 1800:  # 30 minutes
                get_logger().info(
                    f"{Fore.YELLOW}⏳ [WAITING] Next submit in: {remaining_minutes:.0f} minutes | "
                    f"Current points: {int(self.batched_signals)} | "
                    f"Agent: {self.animal_name}{Style.RESET_ALL}"
                )
                self._last_waiting_log = time.time()

    def _hook_after_rewards_updated(self):
        signal_by_agent = self._get_total_rewards_by_agent()
        old_signals = self.batched_signals
        self.batched_signals += self._get_my_rewards(signal_by_agent)
        
        # 🎯 LOG REWARD UPDATES
        reward_gained = self.batched_signals - old_signals
        if reward_gained > 0:
            get_logger().info(
                f"{Fore.GREEN}💰 [REWARD GAINED] +{reward_gained:.1f} points | "
                f"Total: {int(self.batched_signals)} | "
                f"Agent: {self.animal_name}{Style.RESET_ALL}"
            )
        
        self._try_submit_to_chain(signal_by_agent)
        
        # ✅ ADD: Light cleanup after rewards updated
        try:
            if hasattr(self, '_rewards_update_counter'):
                self._rewards_update_counter += 1
            else:
                self._rewards_update_counter = 1
                
            if self._rewards_update_counter % 20 == 0:
                gc.collect()
                
        except Exception as e:
            get_logger().error(f"Rewards cleanup failed: {e}")

    def _hook_after_round_advanced(self):
        # ✅ ADD: Increment round counter and cleanup
        self.round_counter += 1
        
        # 🎯 ENHANCED ROUND ADVANCEMENT LOG
        get_logger().info(
            f"{Fore.MAGENTA}🔄 [ROUND ADVANCED] 🚀 NEW ROUND STARTED! 🚀\n"
            f"   📈 Round: {self.state.round}\n"  
            f"   🎯 Total Rounds Completed: {self.round_counter}\n"
            f"   💰 Pending Points: {int(self.batched_signals)}\n"
            f"   🐾 Agent: {self.animal_name}{Style.RESET_ALL}"
        )
        
        self._save_to_hf()

        # Try to submit to chain again if necessary, but don't update our signal twice
        if not self.submitted_this_round:
            signal_by_agent = self._get_total_rewards_by_agent()
            self._try_submit_to_chain(signal_by_agent)
        
        # Reset flag for next round
        self.submitted_this_round = False

        # ✅ ADD: Memory cleanup every round
        try:
            gc.collect()
            
            # Aggressive cleanup every 10 rounds
            if self.round_counter % 10 == 0:
                self._aggressive_cleanup()
                get_logger().info(
                    f"{Fore.CYAN}🧹 [CLEANUP] Memory cleanup completed for round {self.round_counter}{Style.RESET_ALL}"
                )
                
        except Exception as e:
            get_logger().error(f"Round cleanup failed: {e}")

        # Block until swarm round advances
        self.agent_block()

    def _hook_after_game(self):
        self._save_to_hf()
        
        # ✅ ADD: Final cleanup after game
        try:
            get_logger().info(
                f"{Fore.GREEN}🎮 [GAME ENDED] Final cleanup after game | Agent: {self.animal_name}{Style.RESET_ALL}"
            )
            self._aggressive_cleanup()
        except Exception as e:
            get_logger().error(f"Final cleanup failed: {e}")

    def _save_to_hf(self):
        # This check also implicitly prevents pushes in vLLM mode because hf_token setup is skipped
        if (
            self.hf_token not in [None, "None"]
            and self.state.round % self.hf_push_frequency == 0
        ):
            get_logger().info(
                f"{Fore.BLUE}📤 [HF PUSH] Pushing model to Hugging Face Hub | Round: {self.state.round}{Style.RESET_ALL}"
            )
            try:
                repo_id = self.trainer.args.hub_model_id
                if repo_id is None:
                    repo_id = Path(self.trainer.args.output_dir).name

                self.trainer.model.push_to_hub(
                    repo_id=repo_id,
                    token=self.hf_token,
                    commit_message=f"rl-swarm: round {self.state.round}, agent {self.animal_name}",
                    tags=[
                        "rl-swarm",
                        "genrl-swarm",
                        "grpo",
                        "gensyn",
                        f"I am {self.animal_name}",
                    ],
                )
                
                get_logger().info(
                    f"{Fore.GREEN}✅ [HF SUCCESS] Model pushed successfully to {repo_id}{Style.RESET_ALL}"
                )
                
                # ✅ ADD: Cleanup after HF push
                gc.collect()
                
            except Exception as e:
                get_logger().error(
                    f"{Fore.RED}❌ [HF FAILED] Failed to push model: {str(e)}{Style.RESET_ALL}"
                )
                get_logger().exception(
                    "Failed to push model to the Hugging Face Hub. When you conclude training please try manually pushing it yourself using the instructions here: https://huggingface.co/docs/hub/en/models-uploading",
                    stack_info=True,
                )

    def agent_block(
        self, check_interval=5.0, log_timeout=10.0, max_check_interval=60.0 * 15
    ):
        start_time = time.monotonic()
        fetch_log_time = start_time
        check_backoff = (
            check_interval  # Exponential backoff for already finished rounds.
        )
        
        # ✅ ADD: Agent block counter for cleanup
        self.agent_block_counter += 1
        
        get_logger().info(
            f"{Fore.YELLOW}⏸️ [BLOCKING] Waiting for swarm round advancement... | Agent: {self.animal_name}{Style.RESET_ALL}"
        )
        
        while time.monotonic() - start_time < self.train_timeout:
            curr_time = time.monotonic()
            _ = self.communication.dht.get_visible_maddrs(latest=True)

            # Retrieve current round and stage.
            try:
                round_num, stage = self.coordinator.get_round_and_stage()
            except Exception as e:
                if curr_time - fetch_log_time > log_timeout:
                    get_logger().debug(
                        f"{Fore.YELLOW}🔍 Could not fetch round and stage: {e}. Next check in {check_interval}s.{Style.RESET_ALL}"
                    )
                    fetch_log_time = curr_time

                time.sleep(check_interval)
                continue

            if round_num >= self.state.round:
                get_logger().info(
                    f"{Fore.GREEN}🐝 [JOINING] Joining round: {round_num} | Agent: {self.animal_name}{Style.RESET_ALL}"
                )
                check_backoff = check_interval  # Reset backoff after successful round
                self.state.round = round_num  # advance to swarm's round.
                
                # ✅ ADD: Light cleanup before returning
                try:
                    if self.agent_block_counter % 50 == 0:
                        gc.collect()
                        get_logger().debug(
                            f"{Fore.CYAN}🧹 Agent block cleanup #{self.agent_block_counter}{Style.RESET_ALL}"
                        )
                except Exception:
                    pass
                    
                return
            else:
                get_logger().info(
                    f"{Fore.YELLOW}⏭️ Already finished round: {round_num}. Next check in {check_backoff}s.{Style.RESET_ALL}"
                )
                time.sleep(check_backoff)
                check_backoff = min(check_backoff * 2, max_check_interval)

            if round_num == self.max_round - 1:
                get_logger().info(
                    f"{Fore.MAGENTA}🏁 [FINAL ROUND] Reached maximum round: {self.max_round}{Style.RESET_ALL}"
                )
                return

        get_logger().info(
            f"{Fore.RED}⏰ [TIMEOUT] Training timed out after {self.train_timeout}s!{Style.RESET_ALL}"
        )
