# Deep Reinforcement Learning with Gymnasium

This repository contains solutions and experiments for deep reinforcement learning (DRL) using the Gymnasium toolkit and Stable Baselines3.

> This project is an assignment for the course "Reinforcement Learning" at the Master in Artificial Intelligence Research at the Universidad Internacional Menéndez Pelayo (UIMP).

## Project Structure

```
├── data/           # CSV logs of training/evaluation metrics
├── logs/           # Saved models, tensorboard logs, and evaluations
├── notebook/       # Jupyter notebooks for each exercise and experiment
│   └── example/    # Example notebooks for reference
├── plots/          # Python scripts and generated plots for results
├── videos/         # Rendered videos of trained agents
├── pyproject.toml  # Poetry project configuration
├── poetry.lock     # Poetry lock file
└── README.md       # This file
```

## Main Features

- **Algorithms:** DQN, PPO, A2C (with hyperparameter optimization via Optuna)
- **Environments:** CartPole-v1, LunarLander-v3
- **Logging:** TensorBoard integration, CSV logs for metrics
- **Visualization:** Ready-to-use scripts and plots for training/evaluation
- **Reproducibility:** Saved models and evaluation results

## Installation

1. **Clone the repository:**
   ```sh
   git clone <repo-url>
   cd deep-reinforcement-learning-gymnasium
   ```
2. **Install Poetry:**
   ```sh
   pip install poetry
   ```
3. **Install dependencies:**
   ```sh
   poetry install
   ```
4. **Activate the environment:**
   ```sh
   poetry shell
   ```

## Usage

- **Jupyter Notebooks:**

  - Launch Jupyter Lab or Notebook:
    ```sh
    jupyter lab
    # or
    jupyter notebook
    ```
  - Open notebooks in the `notebook/` directory to run experiments, train agents, and visualize results.

- **Training & Evaluation:**

  - Notebooks are organized by exercise and algorithm (e.g., `exercise2_lunarlander_dqn.ipynb`).
  - Hyperparameter optimization notebooks (with Optuna) are also provided.

- **Plots:**

  - Run scripts in `plots/` to generate or update result plots:
    ```sh
    python plots/exercise1_plots.py
    python plots/exercise2_plots.py
    ```

- **TensorBoard:**

  - To visualize training logs:
    ```sh
    tensorboard --logdir logs/
    ```

- **Videos:**
  - Rendered agent videos are available in the `videos/` folder.

## Dependencies

- Python 3.13+
- gymnasium[box2d]
- stable-baselines3[extra]
- tensorboard
- ipykernel, ipywidgets
- numpy, pandas, matplotlib, seaborn
- optuna, tqdm, rich
- opencv-python

All dependencies are managed via Poetry (see `pyproject.toml`).

## Tips

- For best reproducibility, use the provided Poetry environment.
- Check the `data/` folder for CSV logs if you want to analyze results with your own scripts.
- Use the `logs/` folder to find saved models and evaluation results for further testing or deployment.

## References

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Optuna Documentation](https://optuna.org/)

## Author

Javier Vela (<javier.vela00@gmail.com>)
