#import gymnasium as gym
import numpy as np
import os
import submitit
import datetime

from training_cleaner import train
#from training_clean import train
#rom procgen import ProcgenEnv
#import update
#import entropy_update
#from update import *
#from entropy_update import *
import sys
import os
#sys.path.append('utils')

#start timestamp with unique identifier for name
run_timestamp = datetime.datetime.now().strftime('%Y%m-%d%H-%M%S')
#os.mkdir(with name of unique identifier)

#os.path.join(results, unique identifier)
results_path = os.path.join("results", "cpu_sync")
os.makedirs(results_path, exist_ok = True)

run_path = os.path.join(results_path, run_timestamp)
os.mkdir(run_path)

experiment_path = os.path.join(run_path, "outputs")
os.makedirs(experiment_path, exist_ok = True)

executor = submitit.AutoExecutor(folder=experiment_path)

#executor.update_parameters(timeout_min = 2000, mem_gb = 2, gpus_per_node = 1, cpus_per_task = 32, slurm_array_parallelism = 2, slurm_partition = "gpu")
executor.update_parameters(timeout_min = 14400, mem_gb = 64, cpus_per_task = 16, slurm_array_parallelism = 30, slurm_partition = "cpu")

jobs = []
#ts = [145, 159, 173, 189]
#ts = [98, 104, 110, 116, 122, 128, 134]

#ts = [30, 40, 50, 60, 65, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115]
ts = [30, 50, 70, 90, 110]
num_envs = [32]
trials = 20
#episodes = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]
episodes = [200000]
        
with executor.batch():
     for i in range(trials):
          for tmax in ts:
               for n in num_envs:
                    for episode in episodes:
                         folder_name = f't{tmax}eps{episode}n{n}trial{i}'
                         path = os.path.join(run_path, folder_name)
                         os.makedirs(path, exist_ok=True)
                         job = executor.submit(train, episode= episode, R = 2, r = 0, n = n, n_generalising=8, generalising_trials=20, test_spacing=100, tmax = tmax, path=path, randrew = True, preagent = True, generalising = True, curriculum = False, save_model = True)
                         jobs.append(job)
