import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

A2C_EVAL_EP_LEN_CSV = "./data/ex3_A2C_eval_mean_ep_len.csv"
A2C_EVAL_REW_CSV = "./data/ex3_A2C_eval_mean_reward.csv"
A2C_ROLLOUT_EP_LEN_CSV = "./data/ex3_A2C_rollout_ep_len_mean.csv"
A2C_ROLLOUT_REW_CSV = "./data/ex3_A2C_rollout_ep_rew_mean.csv"

a2c_eval_rew_df = pd.read_csv(A2C_EVAL_REW_CSV)
a2c_rollout_rew_df = pd.read_csv(A2C_ROLLOUT_REW_CSV)
a2c_eval_len_df = pd.read_csv(A2C_EVAL_EP_LEN_CSV)
a2c_rollout_len_df = pd.read_csv(A2C_ROLLOUT_EP_LEN_CSV)

A2C_REW_PLOT_FILE = "./plots/exercise3_a2c_reward_plot.png"
A2C_LEN_PLOT_FILE = "./plots/exercise3_a2c_len_plot.png"


def plot_metric(
    rollout_df, eval_df, plot_file, ylabel, title, rollout_label, eval_label
):
    plt.figure(figsize=(6, 4))
    sns.lineplot(data=rollout_df, x="Step", y="Value", label=rollout_label)
    sns.lineplot(data=eval_df, x="Step", y="Value", label=eval_label, linestyle="--")
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()


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
