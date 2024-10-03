import datetime
import os
from training_reinf import train

def main():
    #start timestamp with unique identifier for name
    run_timestamp = datetime.datetime.now().strftime('%Y%m-%d%H-%M%S')
    #os.mkdir(with name of unique identifier)

    #connect to results folder
    results_run = os.path.join('results', run_timestamp)
    os.makedirs(results_run, exist_ok=True)

    #make data folder inside timestamp

    outputpath = os.path.join(results_run, "outputs")
    os.makedirs(outputpath, exist_ok=True)

    #penalties = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    agent_healths = [1] 
    penalties = [0]
    discount = 0.999
    #discount = 1
    beta = 0.01
    #beta = 0
    tmax = 100
    Nagents = 16
    Nagents_gen = 1024
    max_episodes = 1000
    lr = 2e-3
    save_num=1
    #policy = 'twolayer'
    policy = 'shallow'
    #policy = 'deep'

    goods_path = os.path.join(results_run, f'poli{policy}agents{Nagents}eps{max_episodes}gamma{discount}beta{beta}')
    os.makedirs(goods_path, exist_ok=True)

    for agent_health in agent_healths:
        for penalty in penalties:
            data_path = os.path.join(goods_path, f'health{agent_health}pen{penalty}')
            os.makedirs(data_path, exist_ok=True)
            train(data_path = data_path, agent_health=agent_health, policy = policy, penalty=penalty, max_episodes=max_episodes, tmax=tmax, Nagents=Nagents, Nagents_gen=Nagents_gen, lr=lr, discount=discount, beta=beta, save_num=save_num)

if __name__ == '__main__':
    main()
