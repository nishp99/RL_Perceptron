# custom utilies for displaying animation, collecting rollouts and more
import os
import numpy as np
import torch
import torch.optim as optim
import gymnasium as gym
import ale_py
#from parallelEnv import parallelEnv
import pong_utils_cleaner
import time
import cProfile
import pstats
import io

device = pong_utils_cleaner.device
#check which device is being used.

def load_pretrained_agents(device):
    agents = []
    """
    Load a pretrained agent from the specified path.
    """
    path = os.path.join('t60eps15000n20trial0', 'model.pt')
    init_path = os.path.join('pong_stochastic', '202304-2213-5816')
    final_path = os.path.join('results', init_path)
    actual_path0 = os.path.join(final_path, path)
    agent0 = pong_utils_cleaner.Policy().to(device)
    agent0.load_state_dict(torch.load(actual_path0, map_location=device))
    agent0 = agent0.to('cpu')
    agents.append(agent0)

    path = os.path.join('t70eps15000n20trial0', 'model.pt')
    init_path = os.path.join('pong_stochastic', '202304-2213-5816')
    final_path = os.path.join('results', init_path)
    actual_path1 = os.path.join(final_path, path)
    agent1 = pong_utils_cleaner.Policy().to(device)
    agent1.load_state_dict(torch.load(actual_path1, map_location=device))
    agent1 = agent1.to('cpu')
    agents.append(agent1)

    path = os.path.join('t50eps30000n20trial0', 'model.pt')
    init_path = os.path.join('pong_stochastic', '202304-2315-5636')
    final_path = os.path.join('results', init_path)
    actual_path2 = os.path.join(final_path, path)
    agent2 = pong_utils_cleaner.Policy().to(device)
    agent2.load_state_dict(torch.load(actual_path2, map_location=device))
    agent2 = agent2.to('cpu')
    agents.append(agent2)

    path = os.path.join('t110eps30000n20trial0', 'model.pt')
    init_path = os.path.join('pong_stochastic', '202304-2315-5636')
    final_path = os.path.join('results', init_path)
    actual_path3 = os.path.join(final_path, path)
    agent3 = pong_utils_cleaner.Policy().to(device)
    agent3.load_state_dict(torch.load(actual_path3, map_location=device))
    agent3 = agent3.to('cpu')
    agents.append(agent3)

    path = os.path.join('t110eps30000n20trial1', 'model.pt')
    init_path = os.path.join('pong_stochastic', '202304-2315-5636')
    final_path = os.path.join('results', init_path)
    actual_path4 = os.path.join(final_path, path)
    agent4 = pong_utils_cleaner.Policy().to(device)
    agent4.load_state_dict(torch.load(actual_path4, map_location=device))
    agent4 = agent4.to('cpu')
    agents.append(agent4)

    path = os.path.join('t70eps60000n20trial3', 'model.pt')
    init_path = os.path.join('pong_stochastic', '202305-1713-1108')
    final_path = os.path.join('results', init_path)
    actual_path5 = os.path.join(final_path, path)
    agent5 = pong_utils_cleaner.Policy().to(device)
    agent5.load_state_dict(torch.load(actual_path5, map_location=device))
    agent5 = agent5.to('cpu')
    agents.append(agent5)

    path = os.path.join('t75eps10000n16', 'model.pt')
    init_path = os.path.join('pong_upgraded', '202304-0313-0328')
    final_path = os.path.join('results', init_path)
    actual_path6 = os.path.join(final_path, path)
    agent6 = pong_utils_cleaner.Policy().to(device)
    agent6.load_state_dict(torch.load(actual_path6, map_location=device))
    agent6 = agent6.to('cpu')
    agents.append(agent6)

    path = os.path.join('t103eps2500n8', 'model.pt')
    init_path = os.path.join('pong_upgraded', '202304-0310-4926')
    final_path = os.path.join('results', init_path)
    actual_path7 = os.path.join(final_path, path)
    agent7 = pong_utils_cleaner.Policy().to(device)
    agent7.load_state_dict(torch.load(actual_path7, map_location=device))
    agent7 = agent7.to('cpu')
    agents.append(agent7)

    path = os.path.join('t68eps10000n20trial0', 'model.pt')
    init_path = os.path.join('pong_randreward_gen', '202304-1117-5455')
    final_path = os.path.join('results', init_path)
    actual_path8 = os.path.join(final_path, path)
    agent8 = pong_utils_cleaner.Policy().to(device)
    agent8.load_state_dict(torch.load(actual_path8, map_location=device))
    agent8 = agent8.to('cpu')
    agents.append(agent8)

    path = os.path.join('t80eps10000n20trial0', 'model.pt')
    init_path = os.path.join('pong_randreward_gen', '202304-1117-5455')
    final_path = os.path.join('results', init_path)
    actual_path9 = os.path.join(final_path, path)
    agent9 = pong_utils_cleaner.Policy().to(device)
    agent9.load_state_dict(torch.load(actual_path9, map_location=device))
    agent9 = agent9.to('cpu')
    agents.append(agent9)

    return tuple(agents)


def train(episode, R, r, n, n_generalising, generalising_trials, test_spacing, tmax, path, repeat_action_prob=0.1, discount=1, randrew=True, preagent=False, generalising=False, curriculum=False, save_model=False):

    #print(os.system("nproc"))
    """print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    #print(os.system("nvidia-smi -a"))
    print(torch.cuda.memory_summary())
    print(torch.__version__)
    print(os.system("nvcc --version"))"""

    # Initialize policy and optimizer
    #policy = pong_utils_cleaner.Policy().to(device)
    # cpu version
    policy = pong_utils_cleaner.Policy()

    optimizer = optim.Adam(policy.parameters(), lr=1e-4)
    # Load pretrained agent if required
    pretrained_agents = load_pretrained_agents(device) if preagent else None
    num_agents = len(pretrained_agents)

    # initialize environment (for training and generalising, we use different number of parallel envs in each)
    gym.register_envs(ale_py)
    envs = gym.vector.SyncVectorEnv([
        lambda: gym.make('Pong-v4', repeat_action_probability=repeat_action_prob) for _ in range(n)
    ])
    envs_generalising = gym.vector.SyncVectorEnv([
        lambda: gym.make('Pong-v4', repeat_action_probability=repeat_action_prob) for _ in range(n_generalising)
    ])
    print('made_environments')

    # Create experiment folder (move to inside job loop)
    """path = os.path.join(experiment_path, folder_name)
    os.makedirs(path, exist_ok=True)"""
    file_path = os.path.join(path, 'dic.npy')
    #os.makedirs(file_path, exist_ok=True)
    model_path = os.path.join(path, 'model.pt')
    #os.makedirs(model_path, exist_ok=True)

    # Hyperparameters
    beta = .01

    # Dictionary to track generalisation progress
    dic = {
        #'r': np.zeros((episode, n))
        't': np.zeros((int(episode/test_spacing), n_generalising)),
        'eps': np.zeros(int(episode/test_spacing)),
        't_train': np.zeros(int(episode/test_spacing))
    }

    """
    Train the model using the specified parameters.
    """
    # pre generate the neutral actions
    generalising_neutral_actions = np.zeros(n_generalising, dtype=int)
    training_neutral_actions = np.zeros(n, dtype=int)
    nrand = 5

    for e in range(episode):
        # collect trajectories
        old_probs, states, actions, rewards, mean_time = \
            pong_utils_cleaner.collect_trajectories(envs, num_agents, n, policy, R, r, training_neutral_actions, randrew, tmax=tmax, preagents=pretrained_agents)

        #remove this?
        #total_rewards = np.sum(rewards, axis=0)

        # Update policy using surrogate loss
        L = -pong_utils_cleaner.surrogate(policy, old_probs, states, actions, rewards, discount, beta, tmax)
        optimizer.zero_grad()
        L.backward()
        optimizer.step()
        del L

        if e % test_spacing == 0:
            print(f'episode: {e}')
            time_od = np.zeros(n_generalising)
            for _ in range(generalising_trials):
                rewards_mask = np.ones(n_generalising)
                fr1, fr2 = pong_utils_cleaner.prerun_random(envs_generalising, n_generalising, pretrained_agents, num_agents, generalising_neutral_actions, nrand)
                while np.any(rewards_mask):
            
                    #batch_input = pong_utils_cleaner.preprocess_batch([fr1, fr2])
                    batch_input = pong_utils_cleaner.preprocess_batch_cpu([fr1, fr2])
                    with torch.no_grad():
                        #probs = policy(batch_input).squeeze().cpu().numpy()
                        #cpu
                        probs = policy(batch_input).squeeze().numpy()
                    
                    action = np.where(np.random.rand(n_generalising) < probs, 4, 5)
                    fr1, re1, _, _, _ = envs_generalising.step(action)
                    fr2, re2, _, _, _ = envs_generalising.step(generalising_neutral_actions)
                    
                    reward = pong_utils_cleaner.add(re1, re2)
                    mask = np.where(reward < 0, 0, 1)
                    rewards_mask = pong_utils_cleaner.multiply(rewards_mask, mask)
                    time_od = pong_utils_cleaner.add(time_od, rewards_mask)

            time_od = pong_utils_cleaner.divide(time_od, generalising_trials)
            print(time_od)
            dic['t_train'][int(e/test_spacing)] = mean_time
            dic['t'][int(e/test_spacing), :] = time_od
            dic['eps'][int(e/test_spacing)] = e

        # this reduces exploration in later runs
        beta *= .999

        # display some progress every 100 iterations
        if (e + 1) % 100 == 0:
            print(f"Episode: {e + 1}, Time: {time_od}")
            np.save(file_path, dic)

            if save_model and (e + 1) % 1000 == 0:
                torch.save(policy.state_dict(), model_path)

    #close environments
    envs.close()
    envs_generalising.close()
    
    np.save(file_path, dic)

    if save_model:
        torch.save(policy.state_dict(), model_path)
    
    torch.cuda.empty_cache()
    return 0
