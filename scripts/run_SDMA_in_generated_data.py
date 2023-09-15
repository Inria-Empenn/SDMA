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

generated_data = {"Null":data_generator.generate_simulation(case="Null"), 
                "Null correlated 80%": data_generator.generate_simulation(case="Null correlated", corr=0.8),
                "Null correlated 50%": data_generator.generate_simulation(case="Null correlated", corr=0.5),
                "Null correlated 20%": data_generator.generate_simulation(case="Null correlated", corr=0.2),
                "Non-null correlated 80%": data_generator.generate_simulation(case="Non-null correlated", corr=0.8),
                "Non-null heterogeneous\n voxels": data_generator.generate_simulation(case="Non-null heterogeneous voxels"),
                "Non-null heterogeneous\n pipelines 20%": data_generator.generate_simulation(case="Non-null heterogeneous pipelines", anticorrelated_result_ratio=0.20),
                "Non-null heterogeneous\n pipelines 30%": data_generator.generate_simulation(case="Non-null heterogeneous pipelines"),
                "Non-null heterogeneous\n pipelines 50%": data_generator.generate_simulation(case="Non-null heterogeneous pipelines", anticorrelated_result_ratio=0.5)
                }

# plot data (takes 30 sec)
plot_generated_data.plot_data(generated_data)

## DEBUGGING
def print_summary_results(MA_outputs):
    for model in MA_outputs.keys():
        print(model, " T-map (5 first): ", MA_outputs[model]['T_map'][:5])

for simulation in generated_data.keys():
    contrast_estimates = generated_data[simulation]
    MA_outputs = compute_MA_outputs.get_MA_outputs(contrast_estimates)
    K, J = contrast_estimates.shape
    utils.plot_PP(MA_outputs,contrast_estimates,simulation)
    if "Non-null" not in simulation:
        utils.plot_QQ(MA_outputs,contrast_estimates,simulation)
    utils.compare_contrast_estimates_plot(MA_outputs, simulation)
    ## DEBUGGING
    # print(simulation)
    # print_summary_results(MA_outputs)
    print("*** Simulation for {} is DONE".format(simulation))
    print(" "*100)

