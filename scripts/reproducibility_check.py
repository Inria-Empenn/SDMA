import os
import seaborn
import numpy

import utils
import data_generator 
import compute_MA_outputs
import plot_generated_data
import importlib

importlib.reload(compute_MA_outputs) # reupdate imported codes, useful for debugging
importlib.reload(plot_generated_data) # reupdate imported codes, useful for debugging
importlib.reload(data_generator) # reupdate imported codes, useful for debugging

results_dir = "results_in_generated_data"
if not os.path.exists(results_dir):
    os.mkdir(results_dir)

results_per_seed = {}
for random_seed in range(10):
    generated_data = {"Null":data_generator.generate_simulation(case="Null", seed=random_seed), 
                "Null correlated 80%": data_generator.generate_simulation(case="Null correlated", corr=0.8, seed=random_seed),
                "Null correlated 50%": data_generator.generate_simulation(case="Null correlated", corr=0.5, seed=random_seed),
                "Null correlated 20%": data_generator.generate_simulation(case="Null correlated", corr=0.2, seed=random_seed),
                "Non-null correlated 80%": data_generator.generate_simulation(case="Non-null correlated", corr=0.8, seed=random_seed),
                "Non-null heterogeneous\n voxels": data_generator.generate_simulation(case="Non-null heterogeneous voxels", seed=random_seed),
                "Non-null heterogeneous\n pipelines 20%": data_generator.generate_simulation(case="Non-null heterogeneous pipelines", anticorrelated_result_ratio=0.20, seed=random_seed),
                "Non-null heterogeneous\n pipelines 30%": data_generator.generate_simulation(case="Non-null heterogeneous pipelines", seed=random_seed),
                "Non-null heterogeneous\n pipelines 50%": data_generator.generate_simulation(case="Non-null heterogeneous pipelines", anticorrelated_result_ratio=0.5, seed=random_seed)
                }
    results_per_model = {}
    for simulation in generated_data.keys():
        contrast_estimates = generated_data[simulation]
        MA_outputs = compute_MA_outputs.get_MA_outputs(contrast_estimates)
        results_per_model[simulation] = MA_outputs
    results_per_seed[random_seed] = results_per_model


J =  generated_data['Null'].shape[1]
utils.plot_multiverse_PP(results_per_seed, J)

 




