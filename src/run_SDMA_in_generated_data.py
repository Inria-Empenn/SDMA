import os
import pandas
from datetime import datetime
import utils
import simulation_generator 
import compute_MA_outputs
import numpy

numpy.random.seed(42)

figures_dir = os.path.join("figures", "SIMULATIONS")



###########
# DISPLAY SIMULATIONS
###########

correlation, J, K = 0.8, 20000, 20 # basic

generated_data = {"Null":simulation_generator.generate_simulation(case="Null", K=K, J=J, corr=correlation), 
                "Null correlated 80%": simulation_generator.generate_simulation(case="Null correlated", corr=correlation, K=K, J=J),
                "Null mix": simulation_generator.generate_simulation(case="Null mix", corr=correlation, K=K, J=J)}
# Visualize simulation data
utils.plot_generated_data(generated_data, figures_dir, "simulations_visualisation")


###########
# RUN SDMA IN SIMULATIONS
###########

# (0.2, 0.5, and 0.8).{20,50,100\} and $J$ in \{5.000,  10.000, 20.000\}
# correlation, J, K
simulations = [
    (0.8, 20000, 20),
    (0.8, 20000, 50),
    (0.8, 20000, 100),
    (0.8, 10000, 20),
    (0.8, 10000, 50),
    (0.8, 10000, 100),
    (0.8, 5000, 20),
    (0.8, 5000, 50),
    (0.8, 5000, 100),
    (0.5, 20000, 20),
    (0.5, 20000, 50),
    (0.5, 20000, 100),
    (0.5, 10000, 20),
    (0.5, 10000, 50),
    (0.5, 10000, 100),
    (0.5, 5000, 20),
    (0.5, 5000, 50),
    (0.5, 5000, 100),
    (0.2, 20000, 20),
    (0.2, 20000, 50),
    (0.2, 20000, 100),
    (0.2, 10000, 20),
    (0.2, 10000, 50),
    (0.2, 10000, 100),
    (0.2, 5000, 20),
    (0.2, 5000, 50),
    (0.2, 5000, 100)
]

# Print the list of tuples

for sim in simulations:
    correlation, J, K = sim
    generated_data = {"Null":simulation_generator.generate_simulation(case="Null", K=K, J=J, corr=correlation), 
                    "Null correlated 80%": simulation_generator.generate_simulation(case="Null correlated", corr=correlation, K=K, J=J),
                    "Null mix": simulation_generator.generate_simulation(case="Null mix", corr=correlation, K=K, J=J)}
    outputs = []
    for simulation in generated_data.keys():
        contrast_estimates = generated_data[simulation]
        MA_outputs = compute_MA_outputs.get_MA_outputs(contrast_estimates)
        K, J = contrast_estimates.shape
        title = simulation + str(sim)
        outputs.append([MA_outputs,contrast_estimates,simulation])
    print('Plotting PP for ', sim)
    utils.plot_PP(outputs, figures_dir, correlation, J, K)
