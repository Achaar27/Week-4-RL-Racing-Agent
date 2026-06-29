import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from racing_env import RacingEnv

# this callback just records the reward after every episode
class RewardLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.current_reward = 0

    def _on_step(self):
        self.current_reward += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_reward)
            self.current_reward = 0
        return True

# rolling average to smooth the plot
def rolling_avg(data, window=20):
    result = []
    for i in range(len(data)):
        start = max(0, i - window)
        result.append(sum(data[start:i+1]) / (i - start + 1))
    return result

# train the agent
print("starting training...")
train_env = RacingEnv(render_mode=None)
callback = RewardLogger()
model = PPO("MlpPolicy", train_env, verbose=0)
model.learn(total_timesteps=100_000, callback=callback)
train_env.close()
print("training done!")

# plot how the reward changed over training
rewards = callback.episode_rewards

plt.figure(figsize=(10, 5))
plt.plot(rewards, alpha=0.3, color="blue", label="reward per episode")
plt.plot(rolling_avg(rewards), color="blue", label="average reward")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Learning Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("learning_curve.png")
plt.show()
print("plot saved")

# watch the trained agent drive
print("running trained agent...")
eval_env = RacingEnv(render_mode="human")

for episode in range(5):
    obs, _ = eval_env.reset()
    total_reward = 0
    done = False

    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        total_reward += reward
        done = terminated or truncated

    print(f"episode {episode + 1} reward: {total_reward:.2f}")

eval_env.close()