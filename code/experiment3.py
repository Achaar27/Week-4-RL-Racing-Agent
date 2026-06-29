import matplotlib.pyplot as plt
import numpy as np
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

# ── Experiment 3: Slower car ─────────────────────
# Slower car = easier to control = less crashes early on
print("Experiment 3: Slower car (300k timesteps)...")

env      = RacingEnv(render_mode=None, max_speed=3.0, accel=0.3)
callback = RewardLogger()
model    = PPO("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=300_000, callback=callback)
env.close()
print("Done!")

np.save("exp3_rewards.npy", callback.episode_rewards)

plt.figure(figsize=(10, 5))
plt.plot(callback.episode_rewards, alpha=0.3, color="red", label="Raw reward")
plt.plot(rolling_avg(callback.episode_rewards), color="red", label="Average reward")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Experiment 3 — Slower Car")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("exp3.png")
plt.show()
print("Plot saved as exp3.png")