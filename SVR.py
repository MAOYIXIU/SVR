import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)
  
		self.l7 = nn.Linear(state_dim + action_dim, 256)
		self.l8 = nn.Linear(256, 256)
		self.l9 = nn.Linear(256, 1)

		self.l10 = nn.Linear(state_dim + action_dim, 256)
		self.l11 = nn.Linear(256, 256)
		self.l12 = nn.Linear(256, 1)

	def forward(self, state, action):
		sa = torch.cat([state, action], -1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
  
		q3 = F.relu(self.l7(sa))
		q3 = F.relu(self.l8(q3))
		q3 = self.l9(q3)

		q4 = F.relu(self.l10(sa))
		q4 = F.relu(self.l11(q4))
		q4 = self.l12(q4)
		return q1, q2, q3, q4


class SVR(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		replay_buffer,
		behav,
		Q_min,
		discount=0.99,
		tau=0.005,
		policy_freq=2,
		schedule=True,
		snis=False,
		num_sample=1,
		alpha=0.1,
		sample_std=0.2,
	):
		
		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
		self.critic_target = copy.deepcopy(self.critic)
        
		self.replay_buffer = replay_buffer
		self.max_action = max_action
		self.action_dim = action_dim
		self.discount = discount
		self.tau = tau
		self.policy_freq = policy_freq
		self.Q_min=Q_min
		self.behav = behav
		self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, int(int(1e6)/self.policy_freq))
		self.schedule = schedule
		self.snis = snis
		self.num_sample = num_sample
		self.alpha = alpha
		self.sample_std = sample_std
		self.inv_gauss_coef = 2 * (self.sample_std)**2
		self.total_it = 0


	def select_action(self, state):
		with torch.no_grad():
			self.actor.eval()
			state = torch.FloatTensor(state.reshape(1, -1)).to(device)
			action = self.actor(state).cpu().data.numpy().flatten()
			self.actor.train()
			return action

	def train(self, batch_size=256, writer=None):
		self.total_it += 1

		# Sample replay buffer 
		state, action, next_state, reward, not_done = self.replay_buffer.sample(batch_size)
		next_action = self.actor_target(next_state).detach()
		pi = self.actor(state).detach()
		u = pi.unsqueeze(1).repeat(1,self.num_sample,1)
		with torch.no_grad():
			noise = (torch.randn_like(u) * self.sample_std)
			u = (u + noise).clamp(-self.max_action, self.max_action)
			beta_prob = torch.exp(-torch.mean((self.behav(state)-action)**2, dim=1)/self.inv_gauss_coef)
			pi_prob = torch.exp(-torch.mean((pi-action)**2, dim=1)/self.inv_gauss_coef)
			isratio = torch.clamp(pi_prob / beta_prob, min=0.1)
			if self.snis:
				weight = isratio / isratio.mean() # for snis
			else:
				weight = isratio # for is
			weight = weight.unsqueeze(1)
		# Compute the target Q value
		with torch.no_grad():
			target_Q1, target_Q2, target_Q3, target_Q4 = self.critic_target(next_state, next_action)
			target_Q = torch.cat([target_Q1, target_Q2, target_Q3, target_Q4],dim=1)
			target_Q,_ = torch.min(target_Q,dim=1,keepdim=True)
			target_Q = reward + not_done * self.discount * target_Q

		# Get current Q estimates
		current_Q1, current_Q2, current_Q3, current_Q4 = self.critic(state, action)
		critic_loss =  F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) + F.mse_loss(current_Q3, target_Q) + F.mse_loss(current_Q4, target_Q)
		
		u_Q = []
		u_Q1, u_Q2, u_Q3, u_Q4 = self.critic(state.unsqueeze(1).repeat(1,self.num_sample,1), u)
		u_Q.append(u_Q1.mean(dim=1))
		u_Q.append(u_Q2.mean(dim=1))
		u_Q.append(u_Q3.mean(dim=1))
		u_Q.append(u_Q4.mean(dim=1))
		curr_Q = torch.cat([current_Q1,current_Q2,current_Q3,current_Q4], dim=1)
		u_Q = torch.cat(u_Q, dim=1)
		qmin = (torch.ones_like(curr_Q) * self.Q_min).detach()
		reg_diff = torch.clamp((u_Q-qmin)**2 - weight * (curr_Q-qmin)**2, min = -1e4)
		reg_loss = self.alpha * reg_diff.mean()
		if self.total_it % 10000 == 0:
			with torch.no_grad():
				writer.add_scalar('train/critic_loss', critic_loss.item(), self.total_it)
				writer.add_scalar('train/reg_loss', reg_loss.item(), self.total_it)
				unif = torch.distributions.uniform.Uniform(-1, 1).sample((batch_size, self.action_dim)).to(device)
				anoise1 = (action + torch.randn_like(action) * 0.1).clamp(-self.max_action, self.max_action)
				anoise5 = (action + torch.randn_like(action) * 0.5).clamp(-self.max_action, self.max_action)
				pinoise1 = (pi + torch.randn_like(action) * 0.1).clamp(-self.max_action, self.max_action)
				pinoise5 = (pi + torch.randn_like(action) * 0.5).clamp(-self.max_action, self.max_action)
				Q_pi1, Q_pi2, Q_pi3, Q_pi4 = self.critic(state, pi)
				Q_pi = torch.cat([Q_pi1, Q_pi2, Q_pi3, Q_pi4],dim=1)
				Q_unif1, Q_unif2, Q_unif3, Q_unif4 = self.critic(state, unif)
				Q_unif = torch.cat([Q_unif1, Q_unif2, Q_unif3, Q_unif4],dim=1)
				Q_anoise1_1, Q_anoise1_2, Q_anoise1_3, Q_anoise1_4 = self.critic(state, anoise1)
				Q_anoise1 = torch.cat([Q_anoise1_1, Q_anoise1_2, Q_anoise1_3, Q_anoise1_4],dim=1)
				Q_anoise5_1, Q_anoise5_2, Q_anoise5_3, Q_anoise5_4 = self.critic(state, anoise5)
				Q_anoise5 = torch.cat([Q_anoise5_1, Q_anoise5_2, Q_anoise5_3, Q_anoise5_4],dim=1)
				Q_pinoise1_1, Q_pinoise1_2, Q_pinoise1_3, Q_pinoise1_4 = self.critic(state, pinoise1)
				Q_pinoise1 = torch.cat([Q_pinoise1_1, Q_pinoise1_2, Q_pinoise1_3, Q_pinoise1_4],dim=1)
				Q_pinoise5_1, Q_pinoise5_2, Q_pinoise5_3, Q_pinoise5_4 = self.critic(state, pinoise5)
				Q_pinoise5 = torch.cat([Q_pinoise5_1, Q_pinoise5_2, Q_pinoise5_3, Q_pinoise5_4],dim=1)
				writer.add_scalar('Q/pi', Q_pi.mean().item(), self.total_it)
				writer.add_scalar('Q/a', curr_Q.mean().item(), self.total_it)
				writer.add_scalar('Q/unif', Q_unif.mean().item(), self.total_it)
				writer.add_scalar('Q/anoise0.1', Q_anoise1.mean().item(), self.total_it)
				writer.add_scalar('Q/anoise0.5', Q_anoise5.mean().item(), self.total_it)
				writer.add_scalar('Q/pinoise0.1', Q_pinoise1.mean().item(), self.total_it)
				writer.add_scalar('Q/pinoise0.5', Q_pinoise5.mean().item(), self.total_it)
		critic_loss += reg_loss
		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:
			# Compute actor loss
			pi = self.actor(state)
			if self.total_it <= 0:
				actor_loss = ((pi - action) * (pi - action)).mean()
			else:
				q1,q2,q3,q4 = self.critic(state, pi)
				Q = torch.cat([q1,q2,q3,q4], dim=1)
				Q,_ = torch.min(Q, dim=1)
				lmbda = 2.5 / Q.abs().mean().detach() # follow TD3BC
				actor_loss = -lmbda * Q.mean()

			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()
			if self.schedule:
				self.actor_lr_schedule.step()

			if self.total_it % 10000 == 0:
				writer.add_scalar('train/actor_loss', actor_loss.item(), self.total_it)
				if Q.mean().item() > 5e4:
					exit(0)
			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)