import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DQN_EVAL_EP_LEN_CSV = "./data/ex2_DQN_eval_mean_ep_len.csv"
DQN_EVAL_REW_CSV = "./data/ex2_DQN_eval_mean_reward.csv"
DQN_ROLLOUT_EP_LEN_CSV = "./data/ex2_DQN_rollout_ep_len_mean.csv"
DQN_ROLLOUT_REW_CSV = "./data/ex2_DQN_rollout_ep_rew_mean.csv"

PPO_EVAL_EP_LEN_CSV = "./data/ex2_PPO_eval_mean_ep_len.csv"
PPO_EVAL_REW_CSV = "./data/ex2_PPO_eval_mean_reward.csv"
PPO_ROLLOUT_EP_LEN_CSV = "./data/ex2_PPO_rollout_ep_len_mean.csv"
PPO_ROLLOUT_REW_CSV = "./data/ex2_PPO_rollout_ep_rew_mean.csv"

A2C_EVAL_EP_LEN_CSV = "./data/ex2_A2C_eval_mean_ep_len.csv"
A2C_EVAL_REW_CSV = "./data/ex2_A2C_eval_mean_reward.csv"
A2C_ROLLOUT_EP_LEN_CSV = "./data/ex2_A2C_rollout_ep_len_mean.csv"
A2C_ROLLOUT_REW_CSV = "./data/ex2_A2C_rollout_ep_rew_mean.csv"

dqn_eval_rew_df = pd.read_csv(DQN_EVAL_REW_CSV)
dqn_rollout_rew_df = pd.read_csv(DQN_ROLLOUT_REW_CSV)
dqn_eval_len_df = pd.read_csv(DQN_EVAL_EP_LEN_CSV)
dqn_rollout_len_df = pd.read_csv(DQN_ROLLOUT_EP_LEN_CSV)

ppo_eval_rew_df = pd.read_csv(PPO_EVAL_REW_CSV)
ppo_rollout_rew_df = pd.read_csv(PPO_ROLLOUT_REW_CSV)
ppo_eval_len_df = pd.read_csv(PPO_EVAL_EP_LEN_CSV)
ppo_rollout_len_df = pd.read_csv(PPO_ROLLOUT_EP_LEN_CSV)

a2c_eval_rew_df = pd.read_csv(A2C_EVAL_REW_CSV)
a2c_rollout_rew_df = pd.read_csv(A2C_ROLLOUT_REW_CSV)
a2c_eval_len_df = pd.read_csv(A2C_EVAL_EP_LEN_CSV)
a2c_rollout_len_df = pd.read_csv(A2C_ROLLOUT_EP_LEN_CSV)

DQN_REW_PLOT_FILE = "./plots/exercise2_dqn_reward_plot.png"
DQN_LEN_PLOT_FILE = "./plots/exercise2_dqn_len_plot.png"
PPO_REW_PLOT_FILE = "./plots/exercise2_ppo_reward_plot.png"
PPO_LEN_PLOT_FILE = "./plots/exercise2_ppo_len_plot.png"
A2C_REW_PLOT_FILE = "./plots/exercise2_a2c_reward_plot.png"
A2C_LEN_PLOT_FILE = "./plots/exercise2_a2c_len_plot.png"


def plot_metric(
    rollout_df, eval_df, plot_file, ylabel, title, rollout_label, eval_label
):
    plt.figure(figsize=(6, 4))
    sns.lineplot(data=rollout_df, x="Step", y="Value", label=rollout_label)
    sns.lineplot(data=eval_df, x="Step", y="Value", label=eval_label, linestyle="--")
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()


# DQN plots
plot_metric(
    dqn_rollout_rew_df,
    dqn_eval_rew_df,
    DQN_REW_PLOT_FILE,
    "Mean Reward",
    "DQN: Reward vs Step",
    "DQN Training",
    "DQN Evaluation",
)
plot_metric(
    dqn_rollout_len_df,
    dqn_eval_len_df,
    DQN_LEN_PLOT_FILE,
    "Mean Episode Length",
    "DQN: Episode Length vs Step",
    "DQN Training",
    "DQN Evaluation",
)

# PPO plots
plot_metric(
    ppo_rollout_rew_df,
    ppo_eval_rew_df,
    PPO_REW_PLOT_FILE,
    "Mean Reward",
    "PPO: Reward vs Step",
    "PPO Training",
    "PPO Evaluation",
)
plot_metric(
    ppo_rollout_len_df,
    ppo_eval_len_df,
    PPO_LEN_PLOT_FILE,
    "Mean Episode Length",
    "PPO: Episode Length vs Step",
    "PPO Training",
    "PPO Evaluation",
)

# A2C plots
plot_metric(
    a2c_rollout_rew_df,
    a2c_eval_rew_df,
    A2C_REW_PLOT_FILE,
    "Mean Reward",
    "A2C: Reward vs Step",
    "A2C Training",
    "A2C Evaluation",
)
plot_metric(
    a2c_rollout_len_df,
    a2c_eval_len_df,
    A2C_LEN_PLOT_FILE,
    "Mean Episode Length",
    "A2C: Episode Length vs Step",
    "A2C Training",
    "A2C Evaluation",
)
