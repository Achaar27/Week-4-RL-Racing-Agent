import matplotlib.pyplot as plt
import numpy as np

def rolling_avg(data, window=20):
    result = []
    for i in range(len(data)):
        start = max(0, i - window)
        result.append(sum(data[start:i+1]) / (i - start + 1))
    return result

# Load all 3 experiment rewards
exp1 = np.load("exp1_rewards.npy")
exp2 = np.load("exp2_rewards.npy")
exp3 = np.load("exp3_rewards.npy")

plt.figure(figsize=(12, 6))

plt.plot(rolling_avg(exp1), color="blue",  label="Exp 1 — 300k timesteps (baseline)")
plt.plot(rolling_avg(exp2), color="green", label="Exp 2 — Stronger reward signal")
plt.plot(rolling_avg(exp3), color="red",   label="Exp 3 — Slower car")

plt.xlabel("Episode")
plt.ylabel("Average Total Reward")
plt.title("Experiment Comparison — All 3 Runs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("comparison.png")
plt.show()
print("Comparison plot saved as comparison.png")