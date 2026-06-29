from racing_env import RacingEnv

env = RacingEnv(render_mode="human")
obs, _ = env.reset()

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
print("First observation:", obs)

for _ in range(500):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        print("Episode ended. Resetting...")
        obs, _ = env.reset()

env.close()