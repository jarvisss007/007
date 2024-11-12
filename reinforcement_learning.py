from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

def train_rl_agent():
    """Train a reinforcement learning agent using Proximal Policy Optimization (PPO)."""
    try:
        # Use a simple environment like CartPole for demonstration
        env = make_vec_env('CartPole-v1', n_envs=1)

        # Instantiate the PPO model
        model = PPO('MlpPolicy', env, verbose=1)

        # Train the model
        model.learn(total_timesteps=10000)

        # Save the model
        model.save("ppo_trained_model")

        report = "Reinforcement Learning Model trained successfully on CartPole environment."
        return report

    except Exception as e:
        raise ValueError(f"Failed to train reinforcement learning agent: {e}")
