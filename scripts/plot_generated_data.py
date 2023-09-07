import os
import seaborn
import numpy
import matplotlib.pyplot as plt

import data_generator 
import importlib
importlib.reload(data_generator) # reupdate imported codes, useful for debugging


def plot_data():
    generated_data = {"Null":data_generator.generate_simulation(case="Null"), 
                "Null correlated": data_generator.generate_simulation(case="Null correlated", corr=0.8),
                "Null correlated medium": data_generator.generate_simulation(case="Null correlated", corr=0.5),
                "Null correlated low": data_generator.generate_simulation(case="Null correlated", corr=0.2),
                "Non-null correlated": data_generator.generate_simulation(case="Non-null correlated", corr=0.8),
                "Non-null heterogeneous": data_generator.generate_simulation(case="Non-null heterogeneous", corr=0.8)
                }

    # #######################################
    # print("Plotting generated data")
    # #######################################
    print("Plotting data")
    plt.close('all')
    f, axs = plt.subplots(1, len(generated_data.keys()), figsize=(len(generated_data.keys())*6, 6)) 
    for index, title in enumerate(generated_data.keys()):
        contrast_estimates = generated_data[title]
        mean = numpy.round(numpy.mean(contrast_estimates), 2)
        var = numpy.round(numpy.var(contrast_estimates), 2)
        spat_mat = numpy.corrcoef(contrast_estimates.T)
        corr_mat = numpy.corrcoef(contrast_estimates)
        seaborn.heatmap(contrast_estimates[:, :50], center=0, vmin=contrast_estimates.min(), vmax=contrast_estimates.max(), cmap='coolwarm', ax=axs[index],cbar_kws={'shrink': 0.5})
        axs[index].title.set_text("{} data pipeline\nGenerated values (mean={}, var={})\nSpatial correlation={}\nPipelines correlation={}".format(title, mean, var, numpy.round(spat_mat.mean(), 2), numpy.round(corr_mat.mean(), 2)))
        axs[index].set_xlabel("J voxels", fontsize = 12)
        axs[index].set_ylabel("K pipelines", fontsize = 12)
    plt.tight_layout()
    plt.savefig("results_in_generated_data/data_visualisation.png")
    plt.close('all')
    print("Done plotting")

if __name__ == "__main__":
   print('This file is intented to be used as imported only')