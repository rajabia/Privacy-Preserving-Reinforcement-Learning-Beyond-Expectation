from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
import numpy as np

import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import bisect

class obstacles():
	def __init__(self,position=None,penality= None,id_obs=None ):
		self.position=position
		self.penality= penality
		self.id=id_obs

class hitorstandcontinuous:
	def __init__(self, K=10, r=5,c=0.9):
		self.observation_space =  Box(low=0, high=float(K), shape=(K, K), dtype=np.float64)
		# actions: right left up down
		self.action_space = Discrete(4)
		self.cnt = 0
		#self.length: The Size of Trajectories
		self.length = 20
		#Size of Grids: K x K: [0,0]....[K-1,K-1]
		#Terminal point is  [K-1,K-1]
		self.num_states=K
		#self.state: The current State updates by taking a step
		self.state=np.array ([0,0])

		#Cost of taking an action in each state
		self.reward=np.zeros((K**2, 4))
		
		# The transition probability. with the probability of c 
		# the action is effective otherwise one of the neighbors state is 
		# selected as the next state
		self.c=c
		#TP : Matrix of transition probabilities
		self.TP = np.zeros((4,K**2,K**2))

		# obstacles locations. 
		self.obs=[]
		
		
		

	def add_obsticles(self,position, penality):
		if position[0]>self.num_states or position[0]>self.num_states:
			print("The position for obstacles is not acceptable")
			return
		if len(self.obs)==0:
			new_id=1
		else:
			new_id=self.obs[-1].id+1
		
		obs_new =  obstacles(position= position,penality=penality, id_obs=new_id)
		# obs_new.position=position
		self.obs.append(obs_new)

	def is_obstc(self, position):
		Obst=None
		for o in self.obs:
			if position[0]==o.position[0] and position[1]==o.position[1]:
				Obst=o
				break
		return Obst

	# Finding neighbors of current state
	def neighboars(self):
		Neigh=[]
		# Neigh.append(self.state)
		if np.floor(self.state[0])<self.num_states-1: 
			st=np.copy(self.state)
			st[0]=np.floor(st[0])+1+ np.random.uniform(0.0, 1.0,(1,))
			Neigh.append(st) 

		if np.floor(self.state[1])<self.num_states-1:
			st=np.copy(self.state)
			st[1]=np.floor(st[1])+1+ np.random.uniform(0.0, 1.0,(1,))
			Neigh.append(st)

		if np.floor(self.state[0])>=1:
			st=np.copy(self.state)
			st[0]=np.floor(st[0])- np.random.uniform(0.0, 1.0,(1,))
			Neigh.append(st)

		if np.floor(self.state[1])>=1:
			st=np.copy(self.state)
			st[0]=np.floor(st[0])-np.random.uniform(0.0, 1.0,(1,))
			Neigh.append(st) 

		return Neigh
		
	
	# Updating the state after taking action
	def step(self,action,gamma_cpt,sigma_cpt, eta_1, eta_2,N_max):
		self.cnt +=1
		# if self.state[0]>=self.num_states and self.state[1]:
		# 	return self.state, 0.0001,  True, None
		done=False 
		if self.cnt == self.length:
			done=True
		neighs=self.neighboars()
		st= state_number(self.state)
		p = np.zeros(len(neighs))
		V=[]
		X_0 = 100000
		X = np.zeros(N_max)
		for i,s in enumerate(neighs):
			s_indx=state_number(s)
			p[i] = self.TP[action,st,s_indx]
		
		V=policy_net(torch.from_numpy(np.array(neighs)).type(torch.FloatTensor))
		V=V.max(1)[0].detach().numpy()
		
		s_star=neighs[0]
		for i in range(N_max):
			idx = random.choices(np.arange(len(neighs)), weights = p, k=1)[0]
			s_prime= neighs[idx]
			X[i]=self.reward[st, action] + gamma_cpt* V[idx] +random.gauss(0,1)
			if X[i] < X_0:
				s_star = s_prime
				X_0 = X[i]
		rho_plus = 0
		rho_minus = 0
		X_sort = np.sort(X, axis = None)
		
		for i in range(0,N_max):
			z_1 = (N_max + i - 1)/N_max
			z_2 = (N_max - i)/N_max
			z_3 = i/N_max
			z_4 = (i-1)/N_max
			rho_plus = rho_plus + abs(max(0,X_sort[i]))**sigma_cpt * (z_1**eta_1/(z_1**eta_1 + (1-z_1)**eta_1)**(1/eta_1)-z_2**eta_1/(z_2**eta_1 + (1-z_2)**eta_1)**(1/eta_1))
			rho_minus = rho_minus + abs(min(0,X_sort[i]))**sigma_cpt * (z_3**eta_2/(z_3**eta_2 + (1-z_3)**eta_2)**(1/eta_2)-z_4**eta_2/(z_4**eta_2 + (1-z_4)**eta_2)**(1/eta_2))
		rho = rho_plus - rho_minus
		s_star[0]=np.floor(s_star[0])
		s_star[1]=np.floor(s_star[1])
		self.state=s_star+np.random.uniform(0.0, 1.0,(2,))
		
		return self.state, rho,  done, None

	# Initializing Trasition Matrix 
	def TP_calculate(self):
		# Finding neighbors of a given state
		def local_neigh(m,n):
			N_s=[]
			if m<self.num_states-1:
				N_s.append([m+1,n])
			if n<self.num_states-1:
				N_s.append([m,n+1])
			if m>=1:
				N_s.append([m-1,n])
			if n>=1:
				N_s.append([m,n-1])
			return N_s

		# if action aa = 0, i.e., moving towards right
		for i in range(self.num_states):
			for j in range(self.num_states):
				indx=i*self.num_states+ j 
				N_s=local_neigh(i,j)
				# Action = 0, Moving towards right
				if  [i+1,j] in N_s:
					indx_n=(i+1)*self.num_states+ j
					self.TP[0,indx, indx_n]= self.c
					for w in N_s:
						if w != [i+1,j]: 
							indx_n=(w[0])*self.num_states+ w[1]
							self.TP[0,indx, indx_n]= (1-self.c)/(len(N_s)-1)
				else:
					for w in N_s:
						indx_n=(w[0])*self.num_states+ w[1]
						self.TP[0,indx, indx_n]= (1)/(len(N_s))



				# Action = 1, Moving towards left
				if  [i-1,j] in N_s:
					indx_n=(i-1)*self.num_states+ j
					self.TP[1,indx, indx_n]= self.c
					for w in N_s:
						if w != [i-1,j]: 
							indx_n=(w[0])*self.num_states+ w[1]
							self.TP[1,indx, indx_n]= (1-self.c)/(len(N_s)-1)
				else:
					for w in N_s: 
						indx_n=(w[0])*self.num_states+ w[1]
						self.TP[1,indx, indx_n]= (1)/(len(N_s))



				# Action = 2, Moving towards up
				if  [i,j+1] in N_s:
					indx_n=i*self.num_states+ j+1
					self.TP[2,indx, indx_n]= self.c
					for w in N_s:
						if w != [i,j+1]: 
							indx_n=(w[0])*self.num_states+ w[1]
							self.TP[2,indx, indx_n]= (1-self.c)/(len(N_s)-1)
				else:
					for w in N_s: 
						indx_n=(w[0])*self.num_states+ w[1]
						self.TP[2,indx, indx_n]= (1)/(len(N_s))



				# Action = 3, Moving towards down
				if  [i,j-1] in N_s:
					indx_n=i*self.num_states+ j-1
					self.TP[3,indx, indx_n]= self.c
					for w in N_s:
						if w != [i,j-1]: 
							indx_n=(w[0])*self.num_states+ w[1]
							self.TP[3,indx, indx_n]= (1-self.c)/(len(N_s)-1)
				else:
					for w in N_s: 
						indx_n=(w[0])*self.num_states+ w[1]
						self.TP[3,indx, indx_n]= (1)/(len(N_s))


	# Updating the current state to a point in [0,1)
	# Calculating reward/cost
	def reset(self):
		self.cnt = 0
		self.state = np.random.uniform(0.0, 1.0,(2,))
		self.TP_calculate()
		len_of_SS=self.num_states**2
		r_temp = np.ones((len_of_SS,len_of_SS))
		for i in range(self.num_states):
			for j in range(self.num_states):
				for o in self.obs:
					c=o.position
					id_obs= c[0]*self.num_states+c[1]
					id_st= i*self.num_states+j
					r_temp[id_obs,id_st]= o.penality
				
				r_temp[ i*self.num_states+j,len_of_SS-1]=0


		self.R = np.ones((4,len_of_SS,len_of_SS))
		for a in range(4):
			self.R[a,:,:]=np.copy(r_temp)

		for i in range(len_of_SS):
			for a in range(4):
				self.reward[i,a]= np.dot(self.TP[a][i,:],self.R[a][i,:])

		
		return self.state

	def render(self):
		raise NotImplementedError

	def seed(self, seed_value):
		np.random.seed(seed)
		print('numpy seed is changed to {} globally'.format(seed))
		



def kk(x, y):
	return np.exp(-abs(x-y))

def rho(x, y):
	return np.exp(abs(x-y)) - np.exp(-abs(x-y))

class noisebuffer:
	def __init__(self, m, sigma, max_xy):
		self.buffer = []
		self.base = {}
		self.m = m
		self.sigma = sigma
		self.max_xy = max_xy
	
	def sample(self, st):
		buffer = self.buffer
		sigma = self.sigma
		if len(st.shape)>1:
			st=st.squeeze()
		
		
		s= np.sqrt(st[0]**2+st[1]**2)/np.sqrt(np.sqrt(self.max_xy[0]**2+self.max_xy[1]**2))
		
		if len(buffer) == 0:
			v0 = np.random.normal(0, sigma)
			v1 = np.random.normal(0, sigma)
			v2 = np.random.normal(0, sigma)
			v3 = np.random.normal(0, sigma)
			self.buffer.append((s, v0, v1,v2,v3))
			return (v0, v1, v2,v3)
		else:
			idx = bisect.bisect(buffer, (s, 0, 0))
			if len(buffer) == 1:
				if buffer[0][0] == s:
					return (buffer[0][1], buffer[0][2], buffer[0][3], buffer[0][4])
			else:
				if (idx <= len(buffer)-1) and (buffer[idx][0] == s):
					return (buffer[idx][1], buffer[idx][2], buffer[idx][3], buffer[idx][4])
				elif (idx >= 1) and (buffer[idx-1][0] == s):
					return (buffer[idx-1][1], buffer[idx-1][2],  buffer[idx-1][3], buffer[idx-1][4])
				elif (idx <= len(buffer)-2) and (buffer[idx+1][0] == s):
					return (buffer[idx+1][1], buffer[idx+1][2],  buffer[idx+1][3], buffer[idx+1][4])
			
		if s < buffer[0][0]:
			mean0 = kk(s, buffer[0][0]) * buffer[0][1]
			mean1 = kk(s, buffer[0][0]) * buffer[0][2]
			mean2 = kk(s, buffer[0][0]) * buffer[0][3]
			mean3 = kk(s, buffer[0][0]) * buffer[0][4]
			
			var0 = 1 - kk(s, buffer[0][0]) ** 2
			var1 = 1 - kk(s, buffer[0][0]) ** 2
			var2 = 1 - kk(s, buffer[0][0]) ** 2
			var3 = 1 - kk(s, buffer[0][0]) ** 2
			
			v0 = np.random.normal(mean0, np.sqrt(var0) * sigma)
			v1 = np.random.normal(mean1, np.sqrt(var1) * sigma)
			v2 = np.random.normal(mean2, np.sqrt(var2) * sigma)
			v3 = np.random.normal(mean3, np.sqrt(var3) * sigma)
			
			self.buffer.insert(0, (s, v0, v1, v2, v3))
			
		elif s > buffer[-1][0]:
			mean0 = kk(s, buffer[-1][0]) * buffer[0][1]
			mean1 = kk(s, buffer[-1][0]) * buffer[0][2]
			mean2 = kk(s, buffer[-1][0]) * buffer[0][3]
			mean3 = kk(s, buffer[-1][0]) * buffer[0][4]
			
			var0 = 1 - kk(s, buffer[-1][0]) ** 2
			var1 = var0
			var2 = var0
			var3 = var0
			
			v0 = np.random.normal(mean0, np.sqrt(var0) * sigma)
			v1 = np.random.normal(mean1, np.sqrt(var1) * sigma)
			v2 = np.random.normal(mean2, np.sqrt(var3) * sigma)
			v3 = np.random.normal(mean3, np.sqrt(var3) * sigma)
			
			self.buffer.insert(len(buffer), (s, v0, v1, v2, v3))
		else:
			idx = bisect.bisect(buffer, (s, None, None))
			sminus, eminus0, eminus1, eminus2, eminus3  = buffer[idx-1]
			splus, eplus0, eplus1, eplus2, eplus3 = buffer[idx]
			mean0 = (rho(splus, s)*eminus0 + rho(sminus, s)*eplus0) / rho(sminus, splus)
			mean1 = (rho(splus, s)*eminus1 + rho(sminus, s)*eplus1) / rho(sminus, splus)
			mean2 = (rho(splus, s)*eminus2 + rho(sminus, s)*eplus2) / rho(sminus, splus)
			mean3 = (rho(splus, s)*eminus3 + rho(sminus, s)*eplus3) / rho(sminus, splus)
			
			var0 = 1 - (kk(sminus, s)*rho(splus, s) + kk(splus, s)*rho(sminus, s)) / rho(sminus, splus)
			var1 = var0
			var2 = var0
			var3 = var0
			
			v0 = np.random.normal(mean0, np.sqrt(var0) * sigma)
			v1 = np.random.normal(mean1, np.sqrt(var1) * sigma)
			v2 = np.random.normal(mean2, np.sqrt(var1) * sigma)
			v3 = np.random.normal(mean3, np.sqrt(var1) * sigma)
			
			self.buffer.insert(idx, (s, v0, v1, v2, v3))
		return (v0, v1, v2, v3)

	def reset(self):
		self.buffer = []


class ReplayMemory(object):

	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.position = 0

	def push(self, *args):
		"""Saves a transition."""
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		self.memory[self.position] = Transition(*args)
		self.position = (self.position + 1) % self.capacity

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)


class DQN(nn.Module):

	def __init__(self, m, max_xy,DP=True, sigma=0.4,hidden=16):
		super(DQN, self).__init__()
		self.linear1 = nn.Linear(2, hidden)
		self.linear2 = nn.Linear(hidden, hidden)
		self.head = nn.Linear(hidden, m)
		self.nb = noisebuffer(m, sigma, max_xy)
		self.dp=DP


	# Called with either one element to determine next action, or a batch
	# during optimization. Returns tensor([[left0exp,right0exp]...]).
	def forward(self, s):

		x = F.relu(self.linear1(s))
		x = F.relu(self.linear2(x))
		x = self.head(x)
		if self.dp:
			eps = [self.nb.sample(state.numpy()) for state in s]
			eps = torch.from_numpy(np.array(eps).squeeze())
			return x + eps
		else:
			return x.type(torch.DoubleTensor)

def state_number(st ):
	i=np.floor(st[0])
	j=np.floor(st[1])
	temp=int(i*env.num_states+j)
	if temp>=100:
		print("***************", st, )
	return temp

# simulation for actor critic
def simulate_runs(Actor_Critic=True):
	policy_net.dp=False
	time_horizon = 400
	
	counter = 0
	current_state = [0,0]
	collision=0
	cost = 0
	next_state = [0,0]
	N_S=env.num_states**2

	#keeping number of collision per obstacles
	dict_coll={}
	for o in env.obs:
		dict_coll[o.id]=0

	while current_state[0]<env.num_states and current_state[1]<env.num_states and counter <= time_horizon:
		current_state = next_state
		V=policy_net(torch.from_numpy(np.array([current_state])).type(torch.FloatTensor))

		if Actor_Critic:
			softmax_func=nn.Softmax(dim=1)
			temp=softmax_func(V).detach().numpy().squeeze()
			current_action= random.choices(np.arange(0,4), weights= temp,k=1)[0]
		else: 
			V= V.max(1)[1].view(1, 1)
			current_action = int(V)

		current_state_index=state_number(current_state)
		next_state_idx = random.choices(np.arange(0,N_S), weights = env.TP[current_action][current_state_index,:], k = 1)[0]
		current_state[0]=int(np.floor(next_state_idx/env.num_states))
		current_state[1]=int(next_state_idx % env.num_states)
		cost = cost + gamma_cpt**counter*env.R[current_action][current_state_index,next_state_idx]

		#If hitting a obstacle ==> update # of collision
		ob=env.is_obstc(next_state)
		if not ob is None:
			dict_coll[ob.id]+=1
			collision = collision + 1

		counter = counter + 1

	return cost, collision, dict_coll

def select_action(state):
	global steps_done
	sample = random.random()
	eps_threshold = EPS_END + (EPS_START - EPS_END) * \
		math.exp(-1. * steps_done / EPS_DECAY)
	steps_done += 1
	
	if sample > eps_threshold:
		with torch.no_grad():
			# t.max(1) will return largest value for column of each row.
			# second column on max result is index of where max element was
			# found, so we pick action with the larger expected reward.
			
			return policy_net(state).max(1)[1].view(1, 1)
	else:
		return torch.tensor([[random.randrange(m)]], device=device, dtype=torch.long)


def optimize_model():
	if len(memory) < BATCH_SIZE:
		return
	transitions = memory.sample(BATCH_SIZE)
	
	# Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
	# detailed explanation). This converts batch-array of Transitions
	# to Transition of batch-arrays.
	batch = Transition(*zip(*transitions))
	

	# Compute a mask of non-final states and concatenate the batch elements
	# (a final state would've been the one after which simulation ended)
	non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
										  batch.next_state)), device=device, dtype=torch.uint8)
	non_final_next_states = torch.cat([s for s in batch.next_state
												if s is not None])
	
		
	state_batch = torch.cat(batch.state,axis=0).type(torch.FloatTensor)
	action_batch = torch.cat(batch.action)
	#import pdb; pdb.set_trace()
	reward_batch = torch.cat(batch.reward)

	# Compute Q(s_t, a) - the model computes Q(s_t), then we select the
	# columns of actions taken. These are the actions which would've been taken
	# for each batch state according to policy_net
	
	state_action_values = policy_net(state_batch).gather(1, action_batch)
	

	# Compute V(s_{t+1}) for all next states.
	# Expected values of actions for non_final_next_states are computed based
	# on the "older" target_net; selecting their best reward with max(1)[0].
	# This is merged based on the mask, such that we'll have either the expected
	# state value or 0 in case the state was final.
	batch_size=min(len(non_final_next_states),BATCH_SIZE )
	
	next_state_values = torch.zeros(batch_size, device=device).type(torch.DoubleTensor)
	
	next_state_values[non_final_mask] = policy_net(non_final_next_states).max(1)[0].detach()
	# Compute the expected Q values
	expected_state_action_values = (next_state_values * GAMMA) - reward_batch

	# Compute Huber loss
	loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

	# Optimize the model
	optimizer.zero_grad()
	loss.backward()
	for param in policy_net.parameters():
		param.grad.data.clamp_(-1, 1)
	optimizer.step()
	return loss.item()



def plot_average_var():
	from matplotlib import pyplot as pl
	X1=np.load('20Runs_DP_result1.00.npy', allow_pickle=True)
	X2=np.load('20Runs_No_DP_result.npy',  allow_pickle=True)
	X3=np.load('20Runs_DP_result5.00.npy',  allow_pickle=True)
	
	t=np.arange(1,np.array(X1).shape[1]+1,1)

	
	for j,x in enumerate(X3):
		x=[-1 if i is None else i for i in x]
		m=np.max(np.array(x))
		x=[m if i==-1 else i for i in x]
		X3[j,:]=np.array(x)
	X3=np.array(X3).astype(np.float64)
	Y3, Y3_e= np.mean(X3,axis=0).astype(np.float64),  np.std(X3,axis=0).astype(np.float64)
	pl.plot(t, Y3, 'k', color='#3F7F4C', label='DP-5')
	pl.fill_between(t, Y3-Y3_e, Y3+Y3_e,alpha=0.5, edgecolor='#3F7F4C', facecolor='#7EFF99')


	for j,x in enumerate(X1):
		x=[-1 if i is None else i for i in x]
		m=np.max(np.array(x))
		x=[m+random.gauss(0,1) if i==-1 else i for i in x]
		X1[j,:]=np.array(x)
	Y1= np.mean(X1,axis=0).astype(np.float64)
	X1=np.array(X1,dtype=float)
	Y1_e= np.std(np.array(X1),axis=0).astype(np.float64)
	pl.plot(t, Y1, 'k', color='#CC4F1B', label='DP-1')
	pl.fill_between(t, Y1-Y1_e, Y1+Y1_e,alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')


	for j,x in enumerate(X2):
		x=[-1 if i is None else i for i in x]
		m=np.max(np.array(x))
		x=[m if i==-1 else i for i in x]
		X2[j,:]=np.array(x)
	X2=np.array(X2, dtype=float)
	Y2, Y2_e= np.mean(X2,axis=0).astype(np.float64),  np.std(X2,axis=0).astype(np.float64)
	pl.plot(t, Y2, 'k', color='gray', label='NoDP')
	# pl.fill_between(t, Y2-Y2_e, Y2+Y2_e,alpha=0.5, edgecolor='#1B2ACC', facecolor='#089FFF')
	pl.fill_between(t, Y2-Y2_e, Y2+Y2_e,alpha=0.5, edgecolor='dimgray', facecolor='dimgray')

	pl.xlabel('Episode')
	pl.ylabel('Loss')
	pl.legend(loc="upper right")
	pl.savefig('Var_Together.png',bbox_inches='tight')



def run():
	
	all_episodic_rewards=[]
	cost, cost_ac=[],[]

	average_collision_per_obsticles=[]
	average_collision_per_obsticles_ac=[]

	while len(all_episodic_rewards) <20:
		policy_net = DQN(m,[env.num_states,env.num_states],DP=Diff_Priv,sigma=SIGMA).to(device)
		optimizer = optim.RMSprop(policy_net.parameters())
		memory = ReplayMemory(10000)
		episodic_rewards = []
		
		for i_episode in range(num_episodes):
			if i_episode % 100 == 0:
				print(i_episode)
			# Initialize the environment and state
			state = torch.from_numpy(env.reset())
			total_reward = 0
			for t in count():
				# Select and perform an action
				state=torch.reshape(state, (1, 2))
				# action = select_action(state.type(torch.FloatTensor) )
				action = torch.tensor([[random.randrange(m)]], device=device, dtype=torch.long)
				next_state, reward, done, info = env.step(action.item(),gamma_cpt,sigma_cpt, eta_1, eta_2,N_max)
				
				reward = torch.tensor([reward], device=device).type(torch.FloatTensor)
				next_state = torch.Tensor(next_state).unsqueeze(0)
				memory.push(state, action, next_state, reward)
				total_reward += float(reward.squeeze(0).data)
				state = next_state
				if done:
					break
					
			# Update the target network, copying all weights and biases in DQN
			l=optimize_model()
			episodic_rewards.append(l)
		all_episodic_rewards.append(episodic_rewards)
		


		a,b,a1,b1=0,0,0,0
		list_coll_ac=[]
		list_coll=[]
		for h in range(10):
			a_temp,b_temp,dict_coll=simulate_runs()
			a+=a_temp
			b+=b_temp
			list_coll_ac.append([dict_coll[k] for k in dict_coll])

			a_temp,b_temp,dict_coll = simulate_runs(Actor_Critic=False)
			a1+=a_temp
			b1+=b_temp
			list_coll.append([dict_coll[k] for k in dict_coll])

		cost_ac.append(a/10.0)
		cost.append(a1/10.0)

		ave_per_obst_ac=np.mean(np.array(list_coll_ac),axis=0)
		ave_per_obst=np.mean(np.array(list_coll),axis=0)
		
		print("General Simulation average Cost : %.4f and collision %.4f for 10 runs"%(a1/10.0,b1/10.0))
		print("Actor critic Simulation average Cost : %.4f and collision %.4f for 10 runs"%(a/10.0,b/10.0))
		print("General Simulation: Average of hitting each obstacles: ",ave_per_obst)
		print("Actor -Critic Simulation: Average of hitting each obstacles: ",ave_per_obst_ac)
		
		average_collision_per_obsticles.append(ave_per_obst)
		average_collision_per_obsticles_ac.append(ave_per_obst_ac)

	
	if Diff_Priv:
		savefileadd= 'DP_result%.2f'%SIGMA
	else:
		savefileadd= 'No_DP_result'

	if CPT_mod:
		np.save('20Runs_'+savefileadd+'.npy', all_episodic_rewards)
	print('*'*15 +"  Average for 20 runs   "+'*'*15 )
	print('General Simulation')
	print("Average Cost : %.4f "%(np.mean(cost)))
	print("Average obstacles hit for 20 runs is for each obstacles is  ", (np.mean(np.array(average_collision_per_obsticles),axis=0)))

	print('Actor_Critic Simulation')
	print("Average Cost:  %.4f"%(np.mean(cost_ac)))
	print("Average obstacles hit for 20 runs is for each obstacles is  ", (np.mean(np.array(average_collision_per_obsticles_ac),axis=0)))


if __name__ == '__main__':

	
	
	import argparse

	parser = argparse.ArgumentParser(description='Inputs for Privacy-Preserving Reinforcement Learning Beyond Expectation simulation')
	
	parser.add_argument('--DF', type=int,  default=0, help='0 for OFF 1 for On')#dest='Differential Privacy ON/OFF'
	parser.add_argument('--CPT', type=int,   default=0, help='0 for OFF 1 for On')#dest='CPT ON/OFF',
	parser.add_argument('--sigma', type=float,   default=5.0, help='any float number')#dest='Sigma value for DF',
	args = parser.parse_args()
	print(args)


	Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))
	BATCH_SIZE = 128
	GAMMA = 0.7
	#ES_** :Selecting the Action Parameters 
	#(weight decay for selecting optimized action ro randomly choosing an action)
	EPS_START = 0.9
	EPS_END = 0.05
	EPS_DECAY = 200
	TARGET_UPDATE = 1
	num_episodes = 400

	#DIFFERENTIAL PRIVACY MAIN PARAMETE : larger sigma ==> higher privacy level

	SIGMA= args.sigma

	#Trurn on/ Off DIFFERENTIAL PRIVACY  
	Diff_Priv= False if args.DF==0 else True

	#Trurn on/ Off Penalty for hitting obsticles
	CPT_mod=False if args.CPT==0 else True
	
	env = hitorstandcontinuous()
	m = env.action_space.n
	
	# Adding obsticles, we considered 4 obsticles with different costs of hitting
	#For example [9,8] with cost of 50
	if CPT_mod:
		env.add_obsticles([9,8], 50)
		env.add_obsticles([3,5], 5)
		env.add_obsticles([4,8], 25)
		env.add_obsticles([5,5], 10)
	else:
		env.add_obsticles([9,8], 1)
		env.add_obsticles([3,5], 1)
		env.add_obsticles([4,8], 1)
		env.add_obsticles([5,5], 1)

	seed = 3
	np.random.seed(seed)
	torch.manual_seed(seed)

	device = "cpu"
	policy_net = DQN(m,[env.num_states,env.num_states],DP=Diff_Priv,sigma=SIGMA).to(device)
	

	optimizer = optim.RMSprop(policy_net.parameters())
	memory = ReplayMemory(10000)


	steps_done = 0

	sigma_cpt = 0.88
	
	eta_1 = 0.61
	eta_2 = 0.69
	gamma_cpt = 0.9
	N_max=100

	run()

	args.DF= 0
	args.CPT=1
	run()
	for sigma in [1 , 5]:
		args.sigma=sigma
		args.DF=1
		run()
	plot_average_var()



	


	

