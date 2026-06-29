import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from racing_env import RacingEnv

class RewardLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.current_reward  = 0

    def _on_step(self):
        self.current_reward += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_reward)
            self.current_reward = 0
        return True

def rolling_avg(data, window=20):
    result = []
    for i in range(len(data)):
        start = max(0, i - window)
        result.append(sum(data[start:i+1]) / (i - start + 1))
    return result

# ── Experiment 1: More timesteps (300k) ─────────
print("Experiment 1: Training for 300k timesteps...")
env      = RacingEnv(render_mode=None)
callback = RewardLogger()
model    = PPO("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=300_000, callback=callback)
env.close()
print("Done!")

# Save rewards for later comparison
import numpy as np
np.save("exp1_rewards.npy", callback.episode_rewards)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(callback.episode_rewards, alpha=0.3, color="blue", label="Raw reward")
plt.plot(rolling_avg(callback.episode_rewards), color="blue", label="Average reward")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Experiment 1 — 300k Timesteps")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("exp1.png")
plt.show()
print("Plot saved as exp1.png")