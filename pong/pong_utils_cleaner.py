from parallelEnv import parallelEnv
# import matplotlib.pyplot as plt
import torch
import numpy as np
# from JSAnimation.IPython_display import display_animation
# from matplotlib import animation
import random as rand
from numba import jit, njit, uint8, types
import time

RIGHT = 4
LEFT = 5

five_d_uint8_C = uint8[:, :, :, :, :]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# preprocess a single frame
# crop image and downsample to 80x80
# stack two frames together as input
def preprocess_single(image, bkg_color=np.array([144, 72, 17])):
    img = np.mean(image[34:-16:2, ::2] - bkg_color, axis=-1) / 255.
    return img


# convert outputs of parallelEnv to inputs to pytorch neural net
# this is useful for batch processing especially on the GPU
"""def preprocess_part1(images, bkg_color=np.array([144, 72, 17])):
    #list_of_images = np.asarray(images)
    #print(f'part1 images shape: {images.shape}')
    #print(images.dtype)
    if len(images.shape) < 5:
        images = np.expand_dims(images, 1)
    #print(f'after expand_dims: {images.shape}')
    # subtract bkg and crop
    list_of_images_prepro = np.mean(images[:, :, 34:-16:2, ::2] - bkg_color,
                                    axis=-1) / 255.
    #print(list_of_images_prepro.dtpye)
    #print(f'after normalisation: {list_of_images_prepro.shape}')
    batch_input = np.swapaxes(list_of_images_prepro, 0, 1)
    return batch_input

def preprocess_part2(batch_input):
    return torch.from_numpy(batch_input).float().to(device)

def preprocess_batch(images):
    images = np.asarray(images)
    batch_input = preprocess_part1(images)
    return preprocess_part2(batch_input)"""

"""
jitted functions
"""
@jit(nopython=True)
def add(x,y):
    return x + y

@jit(nopython=True)
def multiply(x, y):
    return x * y

@jit(nopython=True)
def multiply4(w, x, y, z):
    return w * x * y * z

@jit(nopython=True)
def divide(x, y):
    return x / y

#initiate arrays for rewards, reward_mask, and times of death
@jit(nopython=True)
def create_arrays(tmax, n):
    rs = np.zeros((tmax, n))
    r_mask = np.ones(n)
    t_od = np.zeros(n)
    return rs, r_mask, t_od

#if ratio neq 0 we would include this, but we would have to modify it to return something
"""@jit(nopython=True)
def modifyreward(rewards_mask, mask, ratio, R, rewards, t):
    edited_reward = rewards_mask * (mask - 1) * ratio * R
    rewards_mask = rewards_mask * mask
    rewards[t, :] = np.copy(edited_reward)"""

@jit(nopython=True)
def assign_random_rewards(n, time_indices, rewards, rewards_mask, R):
    for i in range(n):
        rewards[time_indices[i], i] = rewards_mask[i] * R
    return rewards

@jit(nopython=True)
def update_reward_array(rewards, rewards_mask, R):
    #rewards[-1, :] = rewards_mask * R
    return rewards_mask * R

# preprocess_batch properly converts two frames into
# shape (n, 2, 80, 80), the proper input for the policy
# this is required when building CNN with pytorch
def preprocess_batch(images, bkg_color=np.array([144, 72, 17])):
    images = np.asarray(images)
    if len(images.shape) < 5:
        images = np.expand_dims(images, 1)
    images = mean_lastdim(images, bkg_color)
    #images = np.swapaxes(images, 0, 1)
    batch_input = torch.from_numpy(images).float().to(device)
    return batch_input

def preprocess_batch_cpu(images, bkg_color=np.array([144, 72, 17])):
    images = np.asarray(images)
    if len(images.shape) < 5:
        images = np.expand_dims(images, 1)
    images = mean_lastdim(images, bkg_color)
    #images = np.swapaxes(images, 0, 1)
    batch_input = torch.from_numpy(images).float()
    return batch_input

@jit(nopython=True)
def mean_lastdim(x, y):
    x = (x[:, :, 34:-16:2, ::2] - y) / 255
    tally = np.zeros((x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
    for i0 in range(x.shape[0]):
        for i1 in range(x.shape[1]):
            for i2 in range(x.shape[2]):
                for i3 in range(x.shape[3]):
                    tally[i0, i1, i2 ,i3] = np.mean(x[i0, i1, i2 ,i3, :])
    tally = np.swapaxes(tally, 0, 1)
    return tally

@jit(nopython=True)
def calculate_normalized_rewardbeta(rewards, discount, T):
    n_envs = rewards.shape[1]
    
    # Calculate discount factors
    discount_factors = discount ** np.arange(T)
    
    # Apply discount to rewards
    discounted_rewards = rewards * discount_factors[:, np.newaxis]
    
    # Calculate future rewards
    rewards_future = np.zeros_like(discounted_rewards)
    for env in range(n_envs):
        cumsum = 0
        for t in range(T - 1, -1, -1):
            cumsum += discounted_rewards[t, env]
            rewards_future[t, env] = cumsum
    
    # Calculate mean and standard deviation manually
    mean = np.zeros(T)
    std = np.zeros(T)
    for t in range(T):
        mean[t] = np.mean(rewards_future[t, :])
        std[t] = np.std(rewards_future[t, :]) + 1.0e-10
    
    # Normalize rewards
    rewards_normalized = (rewards_future - mean[:, np.newaxis]) / std[:, np.newaxis]
    
    return rewards_normalized

@jit(nopython=True)
def calculate_normalized_reward(rewards, discount, T):
    n_envs = rewards.shape[1]
    discounted_future_rewards = np.zeros_like(rewards)
    
    for env in range(n_envs):
        cumulative_reward = 0
        for t in range(T - 1, -1, -1):
            cumulative_reward = rewards[t, env] + discount * cumulative_reward
            discounted_future_rewards[t, env] = cumulative_reward
    
    # Calculate mean and standard deviation manually
    mean = np.zeros(T)
    std = np.zeros(T)
    for t in range(T):
        mean[t] = np.mean(discounted_future_rewards[t, :])
        std[t] = np.std(discounted_future_rewards[t, :]) + 1.0e-10
    
    # Normalize rewards
    rewards_normalized = (discounted_future_rewards - mean[:, np.newaxis]) / std[:, np.newaxis]
    
    return rewards_normalized

#the non jitted version
def calculate_normalized_reward_vectorise(rewards, discount):
    T, n_envs = rewards.shape
    
    # Calculate discounted future rewards in a vectorized manner
    discounted_future_rewards = np.zeros_like(rewards)
    cumulative_reward = np.zeros(n_envs)
    for t in range(T - 1, -1, -1):
        cumulative_reward = rewards[t, :] + discount * cumulative_reward
        discounted_future_rewards[t, :] = cumulative_reward
    
    # Calculate mean and standard deviation across environments
    mean = np.mean(discounted_future_rewards, axis=1)
    std = np.std(discounted_future_rewards, axis=1) + 1.0e-10
    
    # Normalize rewards
    rewards_normalized = (discounted_future_rewards - mean[:, np.newaxis]) / std[:, np.newaxis]
    
    return rewards_normalized

"""
Helper functions
"""
def optimize_actions(actions, device):
    return torch.from_numpy(np.asarray(actions)).to(device=device, dtype=torch.int8)

def optimize_float(data, device):
    return torch.from_numpy(np.asarray(data)).to(device=device, dtype=torch.float)

def optimize_actions_cpu(actions):
    return torch.from_numpy(np.asarray(actions))

def optimize_float_cpu(data):
    return torch.from_numpy(np.asarray(data))

# nrand = number of random steps before using the policy

# randomly choose preagent, reset environment, then run environment under action of preagent for random number of steps
def prerun_random(envs, num_envs, agents, num_agents, neutral_actions, nrand=5):
    index = np.random.randint(low=0, high=num_agents)
    pre_steps = rand.randint(10,55)
    agent = agents[index]

    # start all parallel agents
    envs.reset()
    envs.step(np.ones(num_envs, dtype=int))

    random_preruns = np.random.choice([4, 5], size=(num_envs, nrand))

    # perform nrand random steps
    for i in range(nrand):
        fr1, _, _, _, _ = envs.step(random_preruns[:, i])
        #fr1, re1, _, _, _ = envs.step(np.random.choice([RIGHT, LEFT], num_envs))
        fr2, _, _, _, _ = envs.step(neutral_actions)
        #fr2, re2, _, _, _ = envs.step(np.zeros(num_envs, dtype=int))

    random_thresholds = np.random.rand(num_envs, pre_steps)

    #take random number of steps using selected pretrained agent
    for i in range(pre_steps):
        #frame_input = preprocess_batch([fr1, fr2])
        frame_input = preprocess_batch_cpu([fr1, fr2])
        with torch.no_grad():
            #probabilities = agent(frame_input).squeeze().cpu().detach().numpy()
            #cpu
            probabilities = agent(frame_input).squeeze().numpy()
        #actions = np.where(np.random.rand(num_envs) < probabilities, RIGHT, LEFT)
        actions = np.where(random_thresholds[:, i] < probabilities, RIGHT, LEFT)

        fr1, _, _, _, _ = envs.step(actions)
        fr2, _, _, _, _ = envs.step(neutral_actions)
    
    return fr1, fr2
    

# collect trajectories for a parallelized environment envs
def collect_trajectories(envs, num_agents, num_envs, policy, R, ratio, training_neutral_actions, randrew = False, tmax=200, nrand=5, preagents=None):
    # initialize returning lists
    state_list = []
    prob_list = []
    action_list = []

    # initialize arrays
    #rewards, rewards_mask, time_od = create_arrays(tmax, num_envs)
    rewards_mask = np.ones(num_envs)
    time_od = np.zeros(num_envs)

    #define all the random arrays
    action_random_values = np.random.rand(num_envs, tmax)

    # prerun for random time with randomly selected pretrained agent (for stochastic starts)
    fr1, fr2 = prerun_random(envs, num_envs, preagents, num_agents, training_neutral_actions, nrand)

    # for t in range(tmax):
    for t in range(tmax):
        if not np.any(rewards_mask):
            done = True
            break
        # prepare the input
        #batch_input = preprocess_batch([fr1, fr2])
        batch_input = preprocess_batch_cpu([fr1, fr2])

        # probs will only be used as the pi_old
        # no gradient propagation is needed
        # so we move it to the cpu
        with torch.no_grad():
            #probs = policy(batch_input).squeeze().cpu().numpy()
            #cpu
            probs = policy(batch_input).squeeze().numpy()

        #sample action
        #action = np.where(np.random.rand(num_envs) < probs, RIGHT, LEFT)
        action = np.where(action_random_values[:, t] < probs, RIGHT, LEFT)

        # probabilities of taking actions actually taken
        probs = np.where(action == RIGHT, probs, 1.0 - probs)

        # we take one action and skip game forward
        fr1, re1, _, _, _= envs.step(action)
        fr2, re2, _, _, _ = envs.step(training_neutral_actions)

        # reward = reward accrued over those two previous steps
        reward = add(re1, re2)

        # mask tells us which environments 'died' during previous steps
        mask = np.where(reward < 0, 0, 1)

        # increase time_od
        time_od += rewards_mask

        # rewards mask tells is which environments are still alive
        rewards_mask = multiply(rewards_mask, mask)

        # do this if giving penalty for death
        """edited_reward = multiply4(rewards_mask,  mask - 1, ratio, R)
        rewards_mask = multiply(rewards_mask, mask)
        rewards[t, :] = np.copy(edited_reward)"""
        #time_od += rewards_mask
        
        #modifyreward(rewards_mask, mask, ratio, R, rewards, t)
        #we always set ratio = 0, so lets optimize for this

        # store the result
        state_list.append(batch_input)
        prob_list.append(probs)
        action_list.append(action)

        # stop if any of the trajectories is done
        # we want all the lists to be rectangular
        """if is_done.any():
            print('Done!')
            break"""

    # return pi_theta, states, actions, rewards, probability

    """
    we now convert the reward list to numpy array, then perform masking procedure, where any episodes with negative
    reward have all their rewards set to 0
    we give all other episodes (survived ones) rewards of zero throughout episode, except for very last timestep,
    where rewards of +R is given 
    process can be simplified by just creating this array from the reward mask, we will do that for now
    """
    # reward_list = np.asarray(reward_list)
    # rewards = np.zeros((len(action_list),n))

    # set reward array shape to match length of longest episode
    #set a semirandom time to receive reward for the surviving agents

    if np.any(rewards_mask):
        rewards = np.zeros((tmax, num_envs))
        if randrew:
            time_indices = tmax - 1 - np.random.randint(0, 5, num_envs)
            """env_indices = np.arange(n)
            rewards[time_indices, env_indices] = rewards_mask * R"""
            rewards = assign_random_rewards(num_envs, time_indices, rewards, rewards_mask, R)
        else:
            #jit
            rewards[-1, :] = update_reward_array(rewards, rewards_mask, R)
    else:
        rewards = np.zeros((len(prob_list), num_envs))

    return prob_list, state_list, \
           action_list, rewards, np.mean(time_od)


# convert states to probability, passing through the policy
def states_to_prob(policy, states):
    states = torch.stack(states)
    policy_input = states.view(-1, *states.shape[-3:])
    return policy(policy_input).view(states.shape[:-3])




# return sum of log-prob divided by T
# same thing as -policy_loss
def surrogate(policy, old_probs, states, actions, rewards,
              discount, beta, tmax):
    # discount = discount ** np.arange(len(rewards))
    """#discount = discount_powers(discount, tmax)

    # rewards = np.asarray(rewards) * discount[:, np.newaxis]
    #jit
    rewards = rewards * discount[:, np.newaxis]

    # convert rewards to future rewards
    #jit
    rewards_future = rewards[::-1].cumsum(axis=0)[::-1]

    #make new functions, then jit
    mean = np.mean(rewards_future, axis=1)
    std = np.std(rewards_future, axis=1) + 1.0e-10

    #jit
    rewards_normalized = (rewards_future - mean[:, np.newaxis]) / std[:, np.newaxis]"""

    # check if length of tmax, then use jitted function
    if rewards.shape[0] == tmax:
        rewards_normalized = calculate_normalized_reward(rewards, discount, tmax)
    else:
        rewards_normalized = calculate_normalized_reward_vectorise(rewards, discount)

    # convert everything into pytorch tensors and move to gpu if available
    """actions = np.asarray(actions)
    actions = torch.from_numpy(actions)
    actions = actions.to(torch.int8)
    actions = actions.to(device)

    old_probs = np.asarray(old_probs)
    old_probs = torch.from_numpy(old_probs)
    old_probs = old_probs.to(torch.float)
    old_probs = old_probs.to(device)

    rewards = np.asarray(rewards_normalized)
    rewards = torch.from_numpy(rewards)
    rewards = rewards.to(torch.float)
    rewards = rewards.to(device)"""
    """actions = optimize_actions(actions, device)
    old_probs = optimize_float(old_probs, device)
    rewards = optimize_float(rewards_normalized, device)"""

    #cpu
    actions = optimize_actions_cpu(actions)
    old_probs = optimize_float_cpu(old_probs)
    rewards = optimize_float_cpu(rewards_normalized)

    # convert states to policy (or probability)
    new_probs = states_to_prob(policy, states)
    new_probs = torch.where(actions == RIGHT, new_probs, 1.0 - new_probs)

    ratio = new_probs / old_probs

    # include a regularization term
    # this steers new_policy towards 0.5
    # add in 1.e-10 to avoid log(0) which gives nan
    entropy = -(new_probs * torch.log(old_probs + 1.e-10) + \
                (1.0 - new_probs) * torch.log(1.0 - old_probs + 1.e-10))

    return torch.mean(ratio * rewards + beta * entropy)


# clipped surrogate function
# similar as -policy_loss for REINFORCE, but for PPO
def clipped_surrogate(policy, old_probs, states, actions, rewards,
                      discount=0.995,
                      epsilon=0.1, beta=0.01):
    discount = discount ** np.arange(len(rewards))
    rewards = np.asarray(rewards) * discount[:, np.newaxis]

    # convert rewards to future rewards
    rewards_future = rewards[::-1].cumsum(axis=0)[::-1]

    mean = np.mean(rewards_future, axis=1)
    std = np.std(rewards_future, axis=1) + 1.0e-10

    rewards_normalized = (rewards_future - mean[:, np.newaxis]) / std[:, np.newaxis]

    # convert everything into pytorch tensors and move to gpu if available
    actions = torch.tensor(actions, dtype=torch.int8, device=device)
    old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)
    rewards = torch.tensor(rewards_normalized, dtype=torch.float, device=device)

    # convert states to policy (or probability)
    new_probs = states_to_prob(policy, states)
    new_probs = torch.where(actions == RIGHT, new_probs, 1.0 - new_probs)

    # ratio for clipping
    ratio = new_probs / old_probs

    # clipped function
    clip = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    clipped_surrogate = torch.min(ratio * rewards, clip * rewards)

    # include a regularization term
    # this steers new_policy towards 0.5
    # add in 1.e-10 to avoid log(0) which gives nan
    entropy = -(new_probs * torch.log(old_probs + 1.e-10) + \
                (1.0 - new_probs) * torch.log(1.0 - old_probs + 1.e-10))

    # this returns an average of all the entries of the tensor
    # effective computing L_sur^clip / T
    # averaged over time-step and number of trajectories
    # this is desirable because we have normalized our rewards
    return torch.mean(clipped_surrogate + beta * entropy)


import torch
import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):
    """
    def __init__(self):
        super(Policy, self).__init__()

        # 80x80 to outputsize x outputsize
        # outputsize = (inputsize - kernel_size + stride)/stride
        # (round up if not an integer)

        # conv1 : 80 x 80 -> 40 x 40
        self.conv1 = nn.Conv2d(2, 4, kernel_size=2, stride=2)
        # conv2 : 40 x 40 -> 20 x 20
        self.conv2 = nn.Conv2d(4, 8, kernel_size=2, stride=2)
        # conv3 : 20 x 20 -> 10 x 10
        self.conv3 = nn.Conv2d(8, 16, kernel_size=2, stride=2)
        # conv4 : 10 x 10 ->  5 x  5
        self.conv4 = nn.Conv2d(16, 32, kernel_size=2, stride=2)
        self.size = 32 * 5 * 5

        # 1 fully connected layer
        self.fc1 = nn.Linear(self.size, 64)
        self.fc2 = nn.Linear(64, 8)
        self.fc3 = nn.Linear(8, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = x.view(-1, self.size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.sig(self.fc3(x))
        return x"""

    def __init__(self):
        super(Policy, self).__init__()
        # 80x80x2 to 38x38x4
        # 2 channel from the stacked frame
        self.conv1 = nn.Conv2d(2, 4, kernel_size=6, stride=2, bias=False)
        # 38x38x4 to 9x9x32
        self.conv2 = nn.Conv2d(4, 16, kernel_size=6, stride=4)
        self.size = 9 * 9 * 16

        # two fully connected layer
        self.fc1 = nn.Linear(self.size, 256)
        self.fc2 = nn.Linear(256, 1)

        # Sigmoid to
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.size)
        x = F.relu(self.fc1(x))
        return self.sig(self.fc2(x))
