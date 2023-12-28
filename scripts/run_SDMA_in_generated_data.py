import os
import pandas
from datetime import datetime
import utils
import data_generator 
import compute_MA_outputs
import narps_visualisation
import importlib
from datetime import datetime
import numpy

importlib.reload(utils) # reupdate imported codes, useful for debugging
importlib.reload(compute_MA_outputs) # reupdate imported codes, useful for debugging


results_dir = "results_in_generated_data"
if not os.path.exists(results_dir):
    os.mkdir(results_dir)

dt = datetime.now()
if not os.path.exists("{}/log.txt".format(results_dir)):
    with open("{}/log.txt".format(results_dir), 'w') as f:
        f.write("Last ran the {}/{} at {}h{}".format(dt.day, dt.month, dt.hour, dt.minute))
else:
    with open("{}/log.txt".format(results_dir), 'a') as f:
        f.write("\nLast ran the {}/{} at {}h{}".format(dt.day, dt.month, dt.hour, dt.minute))



generated_data = {"Null":data_generator.generate_simulation(case="Null"), 
                "Null correlated 80%": data_generator.generate_simulation(case="Null correlated", corr=0.8),
                "Null correlated 50%": data_generator.generate_simulation(case="Null correlated", corr=0.5),
                "Null correlated 20%": data_generator.generate_simulation(case="Null correlated", corr=0.2),
                "Null mix": data_generator.generate_simulation(case="Null mix", corr=0.8),
                "Non-null correlated 80%": data_generator.generate_simulation(case="Non-null correlated", corr=0.8),
                "Non-null heterogeneous\n voxels": data_generator.generate_simulation(case="Non-null heterogeneous voxels"),
                "Non-null heterogeneous\n pipelines 20%": data_generator.generate_simulation(case="Non-null heterogeneous pipelines", anticorrelated_result_ratio=0.20),
                "Non-null heterogeneous\n pipelines 30%": data_generator.generate_simulation(case="Non-null heterogeneous pipelines"),
                "Non-null heterogeneous\n pipelines 50%": data_generator.generate_simulation(case="Non-null heterogeneous pipelines", anticorrelated_result_ratio=0.5),
                "Non-null heterogeneous\n pipelines (+3 indep)": data_generator.generate_simulation(case="Non-null heterogeneous pipelines (+3 indep)", anticorrelated_result_ratio=0.5)
                }

# plot data (takes 30 sec)
utils.plot_generated_data(generated_data, results_dir)

Poster_results = []
for simulation in generated_data.keys():
    contrast_estimates = generated_data[simulation]
    MA_outputs = compute_MA_outputs.get_MA_outputs(contrast_estimates)
    K, J = contrast_estimates.shape
    utils.plot_PP(MA_outputs,contrast_estimates,simulation, results_dir)
    if "Non-null" not in simulation:
        utils.plot_QQ(MA_outputs,contrast_estimates,simulation, results_dir)
    utils.compare_contrast_estimates_plot(MA_outputs, simulation, results_dir)
    print('Saving weights..')
    df_weights = pandas.DataFrame(columns=MA_outputs.keys())
    for row in range(K):
        for MA_model in MA_outputs.keys():
            df_weights[MA_model] = MA_outputs[MA_model]['weights']
    df_weights["Mean score"] = contrast_estimates.mean(axis=1)
    df_weights["Var"] = contrast_estimates.std(axis=1)
    utils.plot_weights(contrast_estimates, df_weights, simulation, results_dir)
    # plot residuals
    print("Computing residuals...")
    coefficients, residuals_maps = narps_visualisation.compute_betas(contrast_estimates)
    print("Building figure 6... betas (for residuals)")
    title = simulation
    if "\n" in title:
        title = title.replace('\n', '')
    title = title.replace(' ', '_')
    narps_visualisation.plot_betas(coefficients, title, results_dir, numpy.arange(K))
    print("*** Simulation for {} is DONE".format(simulation))
    print(" "*100)
    # for Poster
    if simulation in ["Null", "Null correlated 80%"]:
        Poster_results.append([MA_outputs,contrast_estimates,simulation])

# utils.plot_PP_Poster(Poster_results, results_dir)
# utils.PP_for_different_seeds(generated_data, results_dir)
utils.plot_PP_OHBM_abstract(Poster_results, results_dir)