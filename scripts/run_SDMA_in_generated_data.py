import os
import seaborn
import numpy

import utils
import compute_MA_outputs
import plot_generated_data
import importlib

importlib.reload(compute_MA_outputs) # reupdate imported codes, useful for debugging
importlib.reload(plot_generated_data) # reupdate imported codes, useful for debugging

results_dir = "results_in_generated_data"
if not os.path.exists(results_dir):
    os.mkdir(results_dir)

# plot data (takes 30 sec)

# plot_generated_data.plot_data()

simulation = "Null"
MA_outputs, contrast_estimates = compute_MA_outputs.get_MA_outputs(simulation)
K, J = contrast_estimates.shape

utils.plot_PP(MA_outputs,contrast_estimates,simulation)
utils.plot_QQ(MA_outputs,contrast_estimates,simulation)
utils.compare_contrast_estimates_plot(MA_outputs, simulation)