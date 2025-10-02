# test.py
import argparse
from stable_baselines3 import PPO
from robot import InvertedPendulumEnv

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO model")
    parser.add_argument("--urdf", type=str, default="robot.urdf", help="Path to URDF file")
    parser.add_argument("--model_path", type=str, default="ppo_pendulum", help="Path to saved model")
    parser.add_argument("--steps", type=int, default=1000, help="Evaluation steps")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy")
    return parser.parse_args()

def main():
    args = parse_args()

    # Create environment with rendering for visualization
    env = InvertedPendulumEnv(args.urdf, render=True)

    # Load model with attached env
    model = PPO.load(args.model_path, env=env)

    # Rollout
    obs, _ = env.reset()
    for _ in range(args.steps):
        action, _ = model.predict(obs, deterministic=args.deterministic)
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()

if __name__ == "__main__":
    main()

