import numpy as np
import torch
import gym
import argparse
import os
import d4rl
import random
import json
import utils
import SVR
from torch.utils.tensorboard import SummaryWriter
import datetime
import time
from tqdm import trange
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
def eval_policy(policy, env_name, seed, mean, std, seed_offset=100, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + seed_offset)
	eval_env.action_space.seed(seed + seed_offset)
	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			state = (np.array(state).reshape(1,-1) - mean)/std
			action = policy.select_action(state)
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes
	d4rl_score = eval_env.get_normalized_score(avg_reward) * 100

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, D4RL score: {d4rl_score:.3f}")
	print("---------------------------------------")
	return d4rl_score


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--env", default="hopper-medium-v2")        # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--eval_freq", default=2e4, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--eval_episodes", default=10, type=int)
	parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument('--folder', default='train_rl')
	parser.add_argument("--no_normalize", action="store_true")
	parser.add_argument('--no_schedule', action="store_true")
	parser.add_argument('--snis', action="store_true")
	parser.add_argument("--alpha", default=0.1, type=float)
	parser.add_argument('--sample_std', default=0.2, type=float)
	args = parser.parse_args()

	print("---------------------------------------")
	print(f"Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	env = gym.make(args.env)
	work_dir = './runs/{}/{}/{}/alpha{}_seed{}_{}'.format(
     os.getcwd().split('/')[-1], args.folder, args.env, args.alpha, args.seed, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
	# Set seeds
	env.seed(args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	writer = SummaryWriter(work_dir)

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
	replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
 
	# minimal reward in all datasets of each environment
	if "hopper" in args.env:
		Q_min = -125
	elif "halfcheetah" in args.env:
		Q_min = -366
	elif "walker2d" in args.env:
		Q_min = -471
	elif "pen" in args.env:
		Q_min = -715

	if not args.no_normalize:
		mean,std = replay_buffer.normalize_states() 
	else:
		mean,std = 0,1

	bc_model_path='./SVR_bcmodels/bcmodel_'+args.env+'.pt'
	behav = SVR.Actor(state_dim, action_dim, max_action).to(device)
	behav.load_state_dict(torch.load(bc_model_path))
	behav.eval()

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"replay_buffer": replay_buffer,
		"discount": args.discount,
		"tau": args.tau,
		"policy_freq": args.policy_freq,
		"schedule": not args.no_schedule,
		"Q_min": Q_min,
		"snis": args.snis,
		"behav": behav,
		"alpha": args.alpha,
		"sample_std": args.sample_std,
	}

	policy = SVR.SVR(**kwargs)
	
	for t in trange(int(args.max_timesteps)):
		policy.train(args.batch_size, writer)
		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			print(f"Time steps: {t+1}")
			d4rl_score = eval_policy(policy, args.env, args.seed, mean, std, eval_episodes=args.eval_episodes)
			writer.add_scalar('eval/d4rl_score', d4rl_score, t)
	time.sleep( 10 )
