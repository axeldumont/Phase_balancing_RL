import argparse
from stable_baselines3 import A2C, PPO, DQN
from environment import GymEnv
from utils import example_net, create_loads
import tensorrt
import os.path

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--steps', type=int, required=True, help='Number of steps to train the model', default=10000)
    parser.add_argument('--model', type=str, required=True, help='Name of the model', default='a2c', choices=['a2c', 'ppo', 'dqn'])
    parser.add_argument('--loads', type=int, required=True, help='Number of loads', default=5)
    parser.add_argument('--timesteps', type=int, required=True, help='Number of timesteps', default=10, choices=[10, 100])

    args = parser.parse_args()

    model_classes = {
        'a2c': A2C,
        'ppo': PPO,
        'dqn': DQN
    }
    type_train = 'small' if args.timesteps <= 10 else 'big'
    net = example_net(args.loads)
    if not os.path.exists(f'./loads/loads_{type_train}_{args.loads}.csv'):
        print(f'Creating loads_{type_train}_{args.loads}.csv')
        create_loads(args.loads, args.timesteps)
        
    env = GymEnv(net, f'./loads/loads_{type_train}_{args.loads}.csv')

    model_class = model_classes[args.model]
    model = model_class('MlpPolicy', env, verbose=0, tensorboard_log=f'./{args.model}_tensorboard/')

    # Example of training the model
    model.learn(total_timesteps=args.steps, tb_log_name='a2c')
    model.save(f'./models/{args.model}_model_{args.loads}_{args.timesteps}')

if __name__ == "__main__":
    main()