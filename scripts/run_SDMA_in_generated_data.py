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

# plot data (takes 30 sec)

# plot_generated_data.plot_data()

def print_summary_results(MA_outputs):
    for model in MA_outputs.keys():
        print(model, " T-map (5 first): ", MA_outputs[model]['T_map'][:5])


for simulation in ["Null", "Null correlated", "Null correlated medium", "Null correlated low", "Non-null correlated", "Non-null heterogeneous"]:
    if simulation == "Null correlated medium":
        contrast_estimates = data_generator.generate_simulation(case="Null correlated", corr=0.5)
    elif simulation == "Null correlated low":
        contrast_estimates = data_generator.generate_simulation(case="Null correlated", corr=0.2)
    else:
        contrast_estimates = data_generator.generate_simulation(case=simulation)
    MA_outputs = compute_MA_outputs.get_MA_outputs(contrast_estimates)
    K, J = contrast_estimates.shape
    utils.plot_PP(MA_outputs,contrast_estimates,simulation)
    utils.plot_QQ(MA_outputs,contrast_estimates,simulation)
    utils.compare_contrast_estimates_plot(MA_outputs, simulation)
    print(simulation)
    print_summary_results(MA_outputs)
    print("*** Simulation for {} is DONE".format(simulation))
    print(" "*100)