import datetime
import os
from training_reinf import train
import submitit



#start timestamp with unique identifier for name
run_timestamp = datetime.datetime.now().strftime('%Y%m-%d%H-%M%S')
#os.mkdir(with name of unique identifier)

node_type = os.path.join('gpu', run_timestamp)
os.makedirs(node_type, exist_ok=True)

#connect to results folder
results_run = os.path.join('results', node_type)
os.makedirs(results_run, exist_ok=True)

#make data folder inside timestamp

"""#os.path.join(results, unique identifier)
training_path = os.path.join(data, "training_data")
os.makedirs(training_path, exist_ok = True)

evaluation_path = os.path.join(data, "evaluation_data")
os.makedirs(evaluation_path, exist_ok = True)"""

outputpath = os.path.join(results_run, "outputs")
os.makedirs(outputpath, exist_ok=True)

executor = submitit.AutoExecutor(folder=outputpath)
executor.update_parameters(timeout_min=14400, mem_gb=6, gpus_per_node=1, cpus_per_task=16, slurm_array_parallelism=20, slurm_partition="gpu")

jobs = []

#penalties = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
agent_healths = [3,4]
penalties = [0]
discount = 0.999
#discount = 1
beta = 0.01
#beta = 0
tmax = 100
Nagents = 16
Nagents_gen = 1024
max_episodes = 4000000
lr = 2e-3
save_num=110
#policy = 'twolayer'
policy = 'shallow'
#policy = 'deep'

trials = 10

goods_path = os.path.join(results_run, f'poli{policy}agents{Nagents}eps{max_episodes}gamma{discount}beta{beta}')
os.makedirs(goods_path, exist_ok=True)

with executor.batch():
	for agent_health in agent_healths:
		for penalty in penalties:
			for i in range(trials):
				data_path = os.path.join(goods_path, f'health{agent_health}pen{penalty}trial{i}')
				os.makedirs(data_path, exist_ok=True)
				job = executor.submit(train, data_path = data_path, agent_health=agent_health, policy = policy, penalty=penalty, max_episodes=max_episodes, tmax=tmax, Nagents=Nagents, Nagents_gen=Nagents_gen, lr=lr, discount=discount, beta=beta, save_num=save_num)
				#jobs.append(job)
