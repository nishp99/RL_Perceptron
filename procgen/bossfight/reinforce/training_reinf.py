import gym3
from gym3 import types_np
import numpy as np
import torch
import torch.optim as optim
import os
import sys
import utils
import pdb

from procgen import ProcgenGym3Env


def train(data_path, agent_health, policy, penalty=0, max_episodes=500000, tmax=100, Nagents=10, Nagents_gen=1000, lr=2e-4, discount=0.995, beta=0.01, save_num=110): #,agent_health as additional parameter
	print(f'tmax:{tmax}')
	print(save_num)
	file_path = os.path.join(data_path, f'data_dic.npy')
	model_path = os.path.join(data_path, f'model.pt')

	RIGHT = 7
	LEFT = 0

	device = utils.device
	print(device)
	# use appropriate policy network
	if policy == 'deep':
		policy = utils.Policy().to(device)
	elif policy == 'shallow':
		policy = utils.Shallow().to(device)
	elif policy == 'twolayer':
		policy = utils.Twolayer().to(device)

	optimizer = optim.Adam(policy.parameters(), lr)

	save_points = np.unique(np.round(np.logspace(0,np.log10(max_episodes),save_num))) # A vector of episodes to generalise at
	print(save_points)
	Nints = save_points.shape[0]
	save_ind = 0

	# we divide by two as two actions are taken every timestep
	episode_length = int(tmax/2 + 1)
	# Maximum episode length. N.B. this is currently hard-coded in the C++ code and cannot be changed by changing this constant
	print('about to make environment')
	#this implementation of vectorised environment could have errors later on in code due to differences to parallel_env.py
	env = ProcgenGym3Env(num=Nagents, env_name="bossfight", tmax=tmax, use_backgrounds=False, restrict_themes=True)
	# N.B. the agent_health argument is irrelevant--we do not use the returns computed by the environment/cpp code
	print('made environment')
	#create generalisation environment
	env_gen = ProcgenGym3Env(num=Nagents_gen, env_name="bossfight", tmax=tmax, use_backgrounds=False, restrict_themes=True)

	step = 0
	total_episodes = 0

	dic = dict()
	dic['training'] = np.zeros((int(max_episodes/1000), Nagents))
	dic['generalisation'] = np.zeros((save_num))

	#torch.save(policy.state_dict(), model_path)
	#np.save(file_path, dic)

	#rews = np.zeros((episode_length, Nagents))
	cumulative_rew = np.zeros(Nagents)
	state_list = []
	prob_list = []
	action_list = []

	random_vals = np.random.rand(episode_length, Nagents)

	while total_episodes <= max_episodes:
		#change to observe, take a do nothing step, then observe to reduce number of total observations within episode
		#can change this after, lets see how it works out first
		rew, obs, done = env.observe()
		cumulative_rew = utils.add(cumulative_rew, rew)
		if any(done):
			#done = done_1
			#obs_1, obs_2 = obs_2, obs_1
			pass
		else:
			env.act(np.zeros(Nagents)) #do nothing action in all environments
			rew_2, obs, done = env.observe()
			cumulative_rew = utils.add(cumulative_rew, rew_2)
			#rew += rew_2
			#done = done_2
			#print(f'training reward: {rew}')
		#rews[step, :] = np.copy(rew)

		#if episode complete
		if step > 0 and np.any(done):
			#print(f'training episode length: {step}')
			successful_episode = cumulative_rew > -agent_health

			#rewards calculated from successful/unsuccessful episodes
			end_rewards = successful_episode.astype(int)
			#print(f'training rewards: {end_rewards}')
			#this bit is all that is needed for without penalty, lets make this fast!
			rews = np.zeros((step, Nagents))
			rews[-1,:] = end_rewards

			"""# this is needed for penalties, can simplify for without penalties
			go through rews, using end_rewards as a mask check for failed episodes, find their indices
			loop indices, for each failed episode find nth (Nagent'th) instance of non-zero element
			set all elements before this and after this to zero
			set the nth element to penalty
			loop through successful episodes
			set all elements to zero except final element set to 1

			fail_indices = np.where(end_rewards == 0)[0]
			success_indices = np.where(end_rewards != 0)[0]

			for i in range(np.shape(fail_indices)[0]):
				count = 1
				for j in range(episode_length):
					if rews[j, fail_indices[i]] < 0:
						if count == agent_health:
							rews[:, fail_indices[i]] = np.zeros(episode_length)
							rews[j, fail_indices[i]] = penalty
							break
						else:
							count += 1

			for i in range(np.shape(success_indices)[0]):
				rews[:, success_indices[i]] = np.zeros(episode_length)
				rews[-1, success_indices[i]] = 1"""


			"""
			backward on loss
			"""
			L = -utils.surrogate(policy, prob_list, state_list, action_list, rews, discount, beta, episode_length)
			optimizer.zero_grad()
			L.backward()
			optimizer.step()
			del L

			total_episodes += 1
			#reset all values
			step = 0
			random_vals = np.random.rand(episode_length, Nagents)
			cumulative_rew = np.zeros(Nagents)
			beta *= 0.995 #reduces exploration in later runs
			state_list = []
			prob_list = []
			action_list = []

			#not necessary to save the weights if we calculate generalisation performance whilst training, lets calculate generalisation performance
			if any(save_points == total_episodes):
				#true_model_path = os.path.join(model_path, f'model{total_episodes}.pt')
				#torch.save(policy.state_dict(), true_model_path)
				rewards_mask = np.ones(Nagents_gen)
				rew, obs_gen, _ = env_gen.observe()
				env_gen.act(np.zeros(Nagents_gen))
				rew_2, obs_gen, done2 = env_gen.observe()
				counter = 0
				random_vals_gen = np.random.rand(episode_length, Nagents_gen)

				while not np.any(done2):
					batch_input = utils.preprocess_batch_gpu(obs_gen['rgb'])
					with torch.no_grad():
						probs = policy(batch_input).squeeze().cpu().numpy()
					"""#cpu
					batch_input = utils.preprocess_batch_cpu(obs_gen['rgb'])
					with torch.no_grad():
						probs = policy(batch_input).squeeze().numpy()"""
					acts = np.where(random_vals_gen[counter, :] < probs, RIGHT, LEFT)
					#acts = np.where(np.random.rand(Nagents_gen) < probs, RIGHT, LEFT)
					env_gen.act(acts)
					rew, obs_gen, done1 =  env_gen.observe()
					if np.any(done1):
						mask = np.where(rew < 0, 0, 1)
						rewards_mask *= mask
						break
					
					env_gen.act(np.zeros(Nagents_gen))
					rew_2, obs_gen, done2 = env_gen.observe()

					rew = rew + rew_2
					mask = np.where(rew < 0, 0, 1)
					rewards_mask *= mask
					counter += 1
				
				generalisation_success = np.mean(rewards_mask)
				dic['generalisation'][save_ind] = generalisation_success
				save_ind += 1

				print(f"Episode: {total_episodes}, survival: {generalisation_success}")

			if total_episodes % 1000 == 0:
				#dic['training'][int(total_episodes/1000), :] = successful_episode.astype(int)
				print(generalisation_success)
			#if total_episodes % 1 == 0:
				#torch.save(policy.state_dict(), model_path)
				np.save(file_path, dic)
				print(f"Iteration {total_episodes}")


		#require function to clean and batch frame, ready for input to network
		#need to turn into torch tensor and make channels the 2nd axis
		#then centre images (take away per channel mean values) and divide by 255?
		# (Nagents, 64, 64, 3)
		batch_input = utils.preprocess_batch_gpu(obs['rgb'])
		with torch.no_grad():
			probs = policy(batch_input).squeeze().cpu().numpy()
		"""#cpu
		batch_input = utils.preprocess_batch_cpu(obs['rgb'])
		with torch.no_grad():
			probs = policy(batch_input).squeeze().numpy()"""

		# probs will only be used as the pi_old
		# no gradient propagation is needed
		# so we move it to the cpu
		
		
		acts = np.where(random_vals[step, :] < probs, RIGHT, LEFT)
		#acts = np.where(np.random.rand(Nagents) < probs, RIGHT, LEFT)
		probs = np.where(acts == RIGHT, probs, 1.0 - probs)

		#store the results
		"""
		uncomment next line if using penalty
		"""
		#rews[step, :] = np.copy(rew)
		state_list.append(batch_input)
		prob_list.append(probs)
		action_list.append(acts)

		step += 1

		env.act(acts)  # Take actions in all envs
	return None
