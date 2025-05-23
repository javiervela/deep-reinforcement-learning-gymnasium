import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DQN_EVAL_CSV = "./data/DQN_eval_mean_reward.csv"
DQN_ROLLOUT_CSV = "./data/DQN_rollout_ep_rew_mean.csv"
PPO_EVAL_CSV = "./data/PPO_eval_mean_reward.csv"
PPO_ROLLOUT_CSV = "./data/PPO_rollout_ep_rew_mean.csv"

DQN_PLOT_FILE = "./plots/exercise1_dqn_plot.png"
PPO_PLOT_FILE = "./plots/exercise1_ppo_plot.png"

dqn_eval_df = pd.read_csv(DQN_EVAL_CSV)
dqn_rollout_df = pd.read_csv(DQN_ROLLOUT_CSV)
ppo_rollout_df = pd.read_csv(PPO_ROLLOUT_CSV)


# Plot DQN data
def plot_dqn_data():
    plt.figure(figsize=(6, 4))
    # plt.title("DQN Training and Evaluation")

    # Plot DQN rollout
    sns.lineplot(data=dqn_rollout_df, x="Step", y="Value", label="DQN Training")

    # Plot DQN evaluation
    sns.lineplot(
        data=dqn_eval_df, x="Step", y="Value", label="DQN Evaluation", linestyle="--"
    )

    plt.xlabel("Step")
    plt.ylabel("Mean Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(DQN_PLOT_FILE)


# Plot PPO data
def plot_ppo_data():
    plt.figure(figsize=(6, 4))
    # plt.title("PPO Training and Evaluation")

    # Plot PPO rollout
    sns.lineplot(data=ppo_rollout_df, x="Step", y="Value", label="PPO Training")

    # Plot PPO evaluation
    sns.lineplot(
        data=pd.read_csv(PPO_EVAL_CSV),
        x="Step",
        y="Value",
        label="PPO Evaluation",
        linestyle="--",
    )

    plt.xlabel("Step")
    plt.ylabel("Mean Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PPO_PLOT_FILE)


# Call the plotting functions
plot_dqn_data()
plot_ppo_data()
