import numpy
import scipy
import pandas
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn

def distribution_inversed(J):
    distribution_inversed = []
    for i in range(J):
        distribution_inversed.append(i/J)
    return distribution_inversed     

def minusLog10me(values):
    # prevent log10(0)
    return numpy.array([-numpy.log10(i) if i != 0 else 5 for i in values])

def plot_PP(MA_outputs, contrast_estimates,simulation):
     K, J = contrast_estimates.shape
     p_cum = distribution_inversed(J)
     x_lim_pplot = -numpy.log10(1/J)
     MA_estimators = list(MA_outputs.keys())

     f, axs = plt.subplots(1, len(MA_estimators), figsize=(len(MA_estimators)*2, 2), sharex=True) 
     for col, title in enumerate(MA_estimators):
          # store required variables
          #  T_map, p_values, ratio_significance, verdict, _ = MA_outputs[title].values() # dangerous because dictionnary are not ordered
          T_map = MA_outputs[title]["T_map"]
          p_values = MA_outputs[title]["p_values"]
          ratio_significance = MA_outputs[title]["ratio_significance"]
          verdict = MA_outputs[title]["verdict"]

          # reformat p and t to sort and plot
          df_obs = pandas.DataFrame(data=numpy.array([p_values, T_map]).T, columns=["p_values", "T_values"])
          df_obs = df_obs.sort_values(by=['p_values'])
          # explected t and p distribution
          t_expected = scipy.stats.norm.rvs(size=J, random_state=0)
          p_expected = 1-scipy.stats.norm.cdf(t_expected)
          df_exp = pandas.DataFrame(data=numpy.array([p_expected, t_expected]).T, columns=["p_expected", "t_expected"])
          df_exp = df_exp.sort_values(by=['p_expected'])
          # Assign values back
          p_expected = df_exp['p_expected'].values
          t_expected = df_exp['t_expected'].values

          p_obs_p_cum = minusLog10me(df_obs['p_values'].values) - minusLog10me(p_cum)

          # make pplot
          axs[col].set_xlabel("-log10 cumulative p")
          axs[col].title.set_text(title)
          axs[col].plot(minusLog10me(p_cum), p_obs_p_cum, color='y')
          if col == 0:
               axs[col].set_ylabel("{}\n\nobs p - cum p".format(simulation))
          else:
               axs[col].set_ylabel("")
          axs[col].axvline(-numpy.log10(0.05), ymin=-1, color='black', linewidth=0.5, linestyle='--')
          axs[col].axhline(0, color='black', linewidth=0.5, linestyle='--')

          # add theoretical confidence interval
          if "Non-null" not in simulation:
               ci = numpy.array([2*numpy.sqrt(p_c*(1-p_c)/J) for p_c in p_cum])
               p_obs_p_cum_ci_above = minusLog10me(numpy.array(p_cum)+ci) - minusLog10me(p_cum)
               p_obs_p_cum_ci_below = p_obs_p_cum_ci_above*-1
               axs[col].fill_between(minusLog10me(p_cum), p_obs_p_cum_ci_below, p_obs_p_cum_ci_above, color='b', alpha=.1)
               axs[col].set_xlim(0, x_lim_pplot)
               axs[col].set_ylim(-1, 1)
          else:
               axs[col].set_xlim(0, x_lim_pplot)
          color= 'green' if verdict == True else 'black'
          axs[col].text(2, 0.25, 'ratio={}%'.format(ratio_significance), color=color)

     plt.suptitle('P-P plots')
     plt.tight_layout()
     if "\n" in simulation:
          simulation = simulation.replace('\n', '')
     simulation = simulation.replace(' ', '_')

     plt.savefig("results_in_generated_data/pp_plot_{}.png".format(simulation))
     plt.close('all')
     print("** ENDED WELL **")



def plot_QQ(MA_outputs, contrast_estimates,simulation, which="p"):
     K, J = contrast_estimates.shape
     p_cum = distribution_inversed(J)
     x_lim_pplot = -numpy.log10(1/J)
     MA_estimators = list(MA_outputs.keys())

     f, axs = plt.subplots(1, len(MA_estimators), figsize=(len(MA_estimators)*2, 2), sharex=True) 
     for col, title in enumerate(MA_estimators):
          # store required variables
          #  T_map, p_values, ratio_significance, verdict, _ = MA_outputs[title].values() # dangerous because dictionnary are not ordered
          T_map = MA_outputs[title]["T_map"]
          p_values = MA_outputs[title]["p_values"]
          ratio_significance = MA_outputs[title]["ratio_significance"]
          verdict = MA_outputs[title]["verdict"]

          # make qqlot
          axs[col].title.set_text(title)
          if which=='p':
             sm.qqplot(p_values, scipy.stats.uniform, fit=True, line='r',ax=axs[col], markersize='2')
          else:
             sm.qqplot(T_map,  scipy.stats.norm, fit=True, line='r',ax=axs[col], markersize='2')
          if col == 0:
             axs[col].set_ylabel("{}\n\nSample Quantiles".format(title))
          else:
             axs[col].set_ylabel("")
     plt.suptitle('Q-Q plots')
     plt.tight_layout()
     if "\n" in simulation:
          simulation = simulation.replace('\n', '')
     simulation = simulation.replace(' ', '_')
     plt.savefig("results_in_generated_data/qq_plot_{}.png".format(simulation))
     plt.close('all')





def compare_contrast_estimates_plot(MA_outputs, simulation):
     MA_estimators = list(MA_outputs.keys())
     colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
     for ind, title in enumerate(MA_estimators):
          # store required variables
          #  T_map, p_values, ratio_significance, verdict, _ = MA_outputs[title].values() # dangerous because dictionnary are not ordered
          T_map = MA_outputs[title]["T_map"]
          p_values = MA_outputs[title]["p_values"]
          ratio_significance = MA_outputs[title]["ratio_significance"]
          verdict = MA_outputs[title]["verdict"]
          T_map.sort()
          plt.plot(range(0, len(T_map)), T_map, color=colors[ind], label=title)
     plt.legend(loc="lower right")
     plt.tight_layout()
     if "\n" in simulation:
          simulation = simulation.replace('\n', '')
     simulation = simulation.replace(' ', '_')
     plt.savefig("results_in_generated_data/MA_contrast_estimates_{}.png".format(simulation))
     plt.close('all')


def plot_multiverse_PP(results_per_seed, J):
     p_cum = distribution_inversed(J)
     x_lim_pplot = -numpy.log10(1/J)
     generated_data_types = list(results_per_seed[0].keys())
     MA_estimators = list(results_per_seed[0][generated_data_types[0]].keys())
     for generation in generated_data_types:
          f, axs = plt.subplots(1, len(MA_estimators), figsize=(len(MA_estimators)*2, 2), sharex=True) 
          for col, title in enumerate(MA_estimators):
               for seed in results_per_seed.keys():
                    # store required variables
                    # T_map, p_values, ratio_significance, verdict, _ = results_per_seed[seed][generation][title].values() # dangerous because dictionnary are not ordered
                    T_map = results_per_seed[seed][generation][title]["T_map"]
                    p_values = results_per_seed[seed][generation][title]["p_values"]
                    ratio_significance = results_per_seed[seed][generation][title]["ratio_significance"]
                    verdict = results_per_seed[seed][generation][title]["verdict"]
                    # reformat p and t to sort and plot
                    df_obs = pandas.DataFrame(data=numpy.array([p_values, T_map]).T, columns=["p_values", "T_values"])
                    df_obs = df_obs.sort_values(by=['p_values'])
                    # explected t and p distribution
                    t_expected = scipy.stats.norm.rvs(size=J, random_state=0)
                    p_expected = 1-scipy.stats.norm.cdf(t_expected)
                    df_exp = pandas.DataFrame(data=numpy.array([p_expected, t_expected]).T, columns=["p_expected", "t_expected"])
                    df_exp = df_exp.sort_values(by=['p_expected'])
                    # Assign values back
                    p_expected = df_exp['p_expected'].values
                    t_expected = df_exp['t_expected'].values

                    p_obs_p_cum = minusLog10me(df_obs['p_values'].values) - minusLog10me(p_cum)

                    # make pplot
                    axs[col].plot(minusLog10me(p_cum), p_obs_p_cum, color='y')
               axs[col].set_xlabel("-log10 cumulative p")
               axs[col].title.set_text(title)
               if col == 0:
                    axs[col].set_ylabel("{}\n\nobs p - cum p".format(generation))
               else:
                    axs[col].set_ylabel("")
               axs[col].axvline(-numpy.log10(0.05), ymin=-1, color='black', linewidth=0.5, linestyle='--')
               axs[col].axhline(0, color='black', linewidth=0.5, linestyle='--')

               # add theoretical confidence interval
               if "Non-null" not in generation:
                    ci = numpy.array([2*numpy.sqrt(p_c*(1-p_c)/J) for p_c in p_cum])
                    p_obs_p_cum_ci_above = minusLog10me(numpy.array(p_cum)+ci) - minusLog10me(p_cum)
                    p_obs_p_cum_ci_below = p_obs_p_cum_ci_above*-1
                    axs[col].fill_between(minusLog10me(p_cum), p_obs_p_cum_ci_below, p_obs_p_cum_ci_above, color='b', alpha=.1)
                    axs[col].set_xlim(0, x_lim_pplot)
                    axs[col].set_ylim(-1, 1)
               else:
                    axs[col].set_xlim(0, x_lim_pplot)

          plt.suptitle('P-P plots')
          plt.tight_layout()
          if "\n" in generation:
               generation = generation.replace('\n', '')
          generation = generation.replace(' ', '_')

          plt.savefig("results_in_generated_data/pp_plot_multiverse_{}.png".format(generation))
          plt.close('all')
          print("** PLOTTING multiverse in {} ENDED WELL **".format(generation))

def plot_weights(simulation, weights):
     print("Plotting weights for each MA model and pipeline")
     plt.close('all')
     f, ax = plt.subplots(1, 2, figsize=(len(weights.columns), 10), sharey=True) 
     seaborn.heatmap(weights[weights.columns[:-2]], center=0, vmin=0, vmax=1, cmap='coolwarm', square=True, ax=ax[0],cbar_kws={'shrink': 0.5})
     ax[0].title.set_text("Weights for each MA model and pipeline for simulation \n{}".format(simulation))
     ax[0].set_xlabel("MA models", fontsize = 12)
     ax[0].set_ylabel("Pipelines", fontsize = 12)
     seaborn.heatmap(weights[weights.columns[-2:]], center=0, cmap='coolwarm', square=True, ax=ax[1],cbar_kws={'shrink': 0.5})
     ax[1].title.set_text("Pipeline type")
     ax[1].set_xlabel("Mean Voxel", fontsize = 12)
     plt.tight_layout()
     if "\n" in simulation:
          simulation = simulation.replace('\n', '')
     simulation = simulation.replace(' ', '_')
     plt.savefig("results_in_generated_data/weights_in_{}.png".format(simulation))
     plt.close('all')
     print("Done plotting")

