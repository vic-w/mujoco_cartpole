# train.py
import argparse
from stable_baselines3 import PPO
from robot import InvertedPendulumEnv

def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on InvertedPendulumEnv")
    parser.add_argument("--urdf", type=str, default="robot.urdf", help="Path to URDF file")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total training timesteps")
    parser.add_argument("--save_path", type=str, default="ppo_pendulum", help="Model save path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def main():
    args = parse_args()

    # Create environment (no rendering during training)
    env = InvertedPendulumEnv(args.urdf, render=False)

    # Build and train model
    model = PPO("MlpPolicy", env, verbose=1, seed=args.seed)
    model.learn(total_timesteps=args.timesteps)

    # Save model
    model.save(args.save_path)

    # Cleanup
    env.close()

if __name__ == "__main__":
    main()

