import numpy
import scipy
import pandas
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn
import numpy as np
from community import community_louvain
import networkx as nx
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import compute_MA_outputs



def compute_contributions(pipeline_z_scores, W="SDMA", std_by_Stouffer=True):
     ones = numpy.ones((pipeline_z_scores.shape[0], 1))
     Q = numpy.corrcoef(pipeline_z_scores)
     W_sdma = (ones.T.dot(Q).dot(ones))**(-1/2) # scalar
     if W == "SDMA":
          contributions_SDMA_Stouffer = pipeline_z_scores * W_sdma
     else:
          contributions_SDMA_Stouffer = pipeline_z_scores * W # W=1
     Q_inv = numpy.linalg.inv(Q)
     W_gls= (ones.T.dot(Q_inv).dot(ones))**(-1/2) * (numpy.sum(Q_inv, axis=1)).reshape(-1, 1) # vector
     if std_by_Stouffer==True:
          contributions_GLS = pipeline_z_scores * W_gls / W_sdma
     else:
          contributions_GLS = pipeline_z_scores * W_gls 
     return contributions_SDMA_Stouffer, contributions_GLS

def plot_generated_data(generated_data, results_dir):
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
        # axs[index].title.set_text("{} data pipeline\nGenerated values (mean={}, var={})\nSpatial correlation={}\nPipelines correlation={}".format(title, mean, var, numpy.round(spat_mat.mean(), 2), numpy.round(corr_mat.mean(), 2)))
        axs[index].title.set_text("{} data pipeline\nCorr = {}".format(title, numpy.round(corr_mat.mean(), 2)))
        
        axs[index].set_xlabel("J voxels", fontsize = 12)
        axs[index].set_ylabel("K pipelines", fontsize = 12)
    plt.tight_layout()
    plt.savefig("{}/fig1.pdf".format(results_dir))
    plt.close('all')
    print("Done plotting")

def distribution_inversed(J):
    distribution_inversed = []
    for i in range(J):
        distribution_inversed.append(i/J)
    return distribution_inversed     

def minusLog10me(values):
    # prevent log10(0)
    return numpy.array([-numpy.log10(i) if i != 0 else 5 for i in values])

def plot_PP(MA_outputs, contrast_estimates,simulation, results_dir):
     K, J = contrast_estimates.shape
     p_cum = distribution_inversed(J)
     x_lim_pplot = -numpy.log10(1/J)
     MA_estimators = list(MA_outputs.keys())

     f, axs = plt.subplots(1, len(MA_estimators), figsize=(len(MA_estimators)*2.5, 3), sharey=True) 
     for col, title in enumerate(MA_estimators):
          # store required variables
          #  T_map, p_values, ratio_significance, verdict, _ = MA_outputs[title].values() # dangerous because dictionnary are not ordered
          T_map = MA_outputs[title]["T_map"]
          p_values = MA_outputs[title]["p_values"]
          ratio_significance = MA_outputs[title]["ratio_significance"]
          verdict = MA_outputs[title]["verdict"]

          # reformat p and t to sort and plot
          print(title, p_values.shape, T_map.shape)
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
          if color == 'black':
               if ratio_significance > 5:
                    color= 'red'
          axs[col].text(1.5, -0.7, '{}%'.format(ratio_significance), color=color)


     plt.suptitle('P-P plots')
     plt.tight_layout()
     if "\n" in simulation:
          simulation = simulation.replace('\n', '')
     simulation = simulation.replace(' ', '_')

     plt.savefig("{}/pp_plot_{}.png".format(results_dir, simulation))
     plt.close('all')
     print("** ENDED WELL **")



def plot_PP_Poster(Poster_results,results_dir):
     contrast_estimates = Poster_results[0][1]
     MA_outputs = Poster_results[0][0]
     K, J = contrast_estimates.shape
     p_cum = distribution_inversed(J)
     x_lim_pplot = -numpy.log10(1/J)
     MA_estimators = list(MA_outputs.keys())[1:]

     f, axs = plt.subplots(2, len(MA_estimators), figsize=(len(MA_estimators)*2.5, 5), sharey=True,sharex=True) 
     for row in range(2):
          for col, title in enumerate(MA_estimators):
               contrast_estimates = Poster_results[row][1]
               MA_outputs = Poster_results[row][0]
               simulation = Poster_results[row][2]
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

               if row == 0:
                    axs[row][col].title.set_text(title)
               else:
                    # make pplot
                    axs[row][col].set_xlabel("-log10 cumulative p")
               axs[row][col].plot(minusLog10me(p_cum), p_obs_p_cum, color='y')
               if col == 0:
                    axs[row][col].set_ylabel("{}\n\nobs p - cum p".format(simulation))
               else:
                    axs[row][col].set_ylabel("")
               axs[row][col].axvline(-numpy.log10(0.05), ymin=-1, color='black', linewidth=0.5, linestyle='--')
               axs[row][col].axhline(0, color='black', linewidth=0.5, linestyle='--')

               ci = numpy.array([2*numpy.sqrt(p_c*(1-p_c)/J) for p_c in p_cum])
               p_obs_p_cum_ci_above = minusLog10me(numpy.array(p_cum)+ci) - minusLog10me(p_cum)
               p_obs_p_cum_ci_below = p_obs_p_cum_ci_above*-1
               axs[row][col].fill_between(minusLog10me(p_cum), p_obs_p_cum_ci_below, p_obs_p_cum_ci_above, color='b', alpha=.1)
               axs[row][col].set_xlim(0, x_lim_pplot)
               axs[row][col].set_ylim(-1, 1)

               color= 'green' if verdict == True else 'black'
               if color == 'black':
                    if ratio_significance > 5:
                         color= 'red'
               axs[row][col].text(1.5, -0.7, '{}%'.format(ratio_significance), color=color)

               if row==0:
                    plt.tick_params(
                        axis='x',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        bottom=False)     # ticks along the bottom edge are off) 

     # plt.suptitle('P-P plots')
     plt.tight_layout()
     if "\n" in simulation:
          simulation = simulation.replace('\n', '')
     simulation = simulation.replace(' ', '_')

     plt.savefig("{}/pp_plot_POSTER.png".format(results_dir))
     plt.close('all')
     print("** ENDED WELL **")


def plot_PP_OHBM_abstract(Poster_results, results_dir):
     contrast_estimates = Poster_results[0][1]
     MA_outputs = Poster_results[0][0]
     K, J = contrast_estimates.shape
     p_cum = distribution_inversed(J)
     x_lim_pplot = -numpy.log10(1/J)
     MA_estimators = list(MA_outputs.keys())
     MA_estimators = list(MA_outputs.keys())[:]

     f, axs = plt.subplots(2, len(MA_estimators), figsize=(len(MA_estimators)*2.5, 5), sharey=True,sharex=True) 
     for row in range(2):
          for col, title in enumerate(MA_estimators):
               if title == "":
                    continue
               contrast_estimates = Poster_results[row][1]
               MA_outputs = Poster_results[row][0]
               simulation = Poster_results[row][2]
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

               if row == 0:
                    axs[row][col].title.set_text(title)
               else:
                    # make pplot
                    axs[row][col].set_xlabel("-log10 cumulative p", fontsize=12)
               axs[row][col].plot(minusLog10me(p_cum), p_obs_p_cum, color='y')
               if col == 0:
                    axs[row][col].set_ylabel("{}\n\nobs p - expt p".format(simulation), fontsize=12)
               else:
                    axs[row][col].set_ylabel("")
               axs[row][col].axvline(-numpy.log10(0.05), ymin=-1, color='black', linewidth=0.5, linestyle='--')
               axs[row][col].axhline(0, color='black', linewidth=0.5, linestyle='--')

               ci = numpy.array([2*numpy.sqrt(p_c*(1-p_c)/J) for p_c in p_cum])
               p_obs_p_cum_ci_above = minusLog10me(numpy.array(p_cum)+ci) - minusLog10me(p_cum)
               p_obs_p_cum_ci_below = p_obs_p_cum_ci_above*-1
               axs[row][col].fill_between(minusLog10me(p_cum), p_obs_p_cum_ci_below, p_obs_p_cum_ci_above, color='b', alpha=.1)
               axs[row][col].set_xlim(0, x_lim_pplot)
               axs[row][col].set_ylim(-1, 1)
               color= 'green' if verdict == True else 'black'
               if color == 'black':
                    if ratio_significance > 5:
                         color= 'red'
               axs[row][col].text(1.5, -0.7, '{}%'.format(ratio_significance), color=color)

               if row==0:
                    plt.tick_params(
                        axis='x',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        bottom=False)     # ticks along the bottom edge are off) 

     # plt.suptitle('P-P plots')
     plt.tight_layout()
     if "\n" in simulation:
          simulation = simulation.replace('\n', '')
     simulation = simulation.replace(' ', '_')

     # plt.savefig("{}/pp_plot_OHBM_ABSTRACT.png".format(results_dir))
     plt.savefig("{}/fig2.pdf".format(results_dir))
     plt.close('all')
     print("** ENDED WELL **")


def plot_PP_final_figure_2(Poster_results, results_dir, corr, J, K):
     contrast_estimates = Poster_results[0][1]
     MA_outputs = Poster_results[0][0]
     p_cum = distribution_inversed(J)
     x_lim_pplot = -numpy.log10(1/J)
     MA_estimators = list(MA_outputs.keys())
     MA_estimators = list(MA_outputs.keys())[:]

     f, axs = plt.subplots(3, len(MA_estimators), figsize=(len(MA_estimators)*2.5, 8), sharey=True,sharex=True) 
     for row in range(3):
          for col, title in enumerate(MA_estimators):
               if title == "":
                    continue
               contrast_estimates = Poster_results[row][1]
               MA_outputs = Poster_results[row][0]
               simulation = Poster_results[row][2]
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

               if row == 0:
                    axs[row][col].title.set_text(title)
               elif row == 2:
                    # make pplot
                    axs[row][col].set_xlabel("-log10 cumulative p", fontsize=12)
               axs[row][col].plot(minusLog10me(p_cum), p_obs_p_cum, color='y')
               if col == 0:
                    axs[row][col].set_ylabel("{}\n\nobs p - expt p".format(simulation), fontsize=12)
               else:
                    axs[row][col].set_ylabel("")
               axs[row][col].axvline(-numpy.log10(0.05), ymin=-1, color='black', linewidth=0.5, linestyle='--')
               axs[row][col].axhline(0, color='black', linewidth=0.5, linestyle='--')

               ci = numpy.array([2*numpy.sqrt(p_c*(1-p_c)/J) for p_c in p_cum])
               p_obs_p_cum_ci_above = minusLog10me(numpy.array(p_cum)+ci) - minusLog10me(p_cum)
               p_obs_p_cum_ci_below = p_obs_p_cum_ci_above*-1
               axs[row][col].fill_between(minusLog10me(p_cum), p_obs_p_cum_ci_below, p_obs_p_cum_ci_above, color='b', alpha=.1)
               axs[row][col].set_xlim(0, x_lim_pplot)
               axs[row][col].set_ylim(-1, 1)
               color= 'green' if verdict == True else 'black'
               if color == 'black':
                    if ratio_significance > 5:
                         color= 'red'
               axs[row][col].text(1.5, -0.7, '{}%'.format(numpy.round(ratio_significance, 2)), color=color)

               if row==0:
                    plt.tick_params(
                        axis='x',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        bottom=False)     # ticks along the bottom edge are off) 

     # plt.suptitle('P-P plots')
     
     if "\n" in simulation:
          simulation = simulation.replace('\n', '')
     simulation = simulation.replace(' ', '_')

     # plt.savefig("{}/pp_plot_OHBM_ABSTRACT.png".format(results_dir))
     plt.suptitle("{} voxels, {} pipelines, correlation between pipelines: {}".format(J, K, corr), fontsize=14)
     plt.tight_layout()
     plt.savefig("{}/fig2_J{}_K{}_Corr{}.pdf".format(results_dir,J, K, corr))
     plt.close('all')
     print("** ENDED WELL **")


def plot_QQ(MA_outputs, contrast_estimates,simulation, results_dir, which="p"):
     K, J = contrast_estimates.shape
     p_cum = distribution_inversed(J)
     x_lim_pplot = -numpy.log10(1/J)
     MA_estimators = list(MA_outputs.keys())

     f, axs = plt.subplots(1, len(MA_estimators), figsize=(len(MA_estimators)*2, 3), sharex=True) 
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
     plt.savefig("{}/qq_plot_{}.png".format(results_dir, simulation))
     plt.close('all')



def compare_contrast_estimates_plot(MA_outputs, simulation, results_dir):
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
     plt.savefig("{}/MA_contrast_estimates_{}.png".format(results_dir, simulation))
     plt.close('all')


def PP_for_different_seeds(generated_data, results_dir):
     results_per_seed = {}
     for random_seed in range(10):
          results_per_model = {}
          for simulation in generated_data.keys():
               contrast_estimates = generated_data[simulation]
               MA_outputs = compute_MA_outputs.get_MA_outputs(contrast_estimates)
               results_per_model[simulation] = MA_outputs
          results_per_seed[random_seed] = results_per_model
     J =  generated_data['Null'].shape[1]
     plot_multiverse_PP(results_per_seed, J, results_dir)

def plot_multiverse_PP(results_per_seed, J, results_dir):
     p_cum = distribution_inversed(J)
     x_lim_pplot = -numpy.log10(1/J)
     generated_data_types = list(results_per_seed[0].keys())
     MA_estimators = list(results_per_seed[0][generated_data_types[0]].keys())
     for generation in generated_data_types:
          f, axs = plt.subplots(1, len(MA_estimators), figsize=(len(MA_estimators)*2, 3), sharex=True) 
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

          plt.savefig("{}/pp_plot_multiverse_{}.png".format(results_dir, generation))
          plt.close('all')
          print("** PLOTTING multiverse in {} ENDED WELL **".format(generation))


def reorganize_according_to_new_indexing(matrix, partition, team_names=None):
     ''' Reorganized the covariance matrix according to the partition

     Parameters
     ----------
     matrix : correlation matrix (n_roi*n_roi)

     Returns
     ----------
     matrix reorganized

     '''
     # compute the best partition
     reorganized = numpy.zeros(matrix.shape).astype(matrix.dtype)
     labels = range(len(matrix))
     labels_new_order = []

     ## reorganize matrix abscissa wise
     i = 0
     # iterate through all created community
     for values in numpy.unique(list(partition.values())):
        # iterate through each ROI
        for key in partition:
            if partition[key] == values:
                reorganized[i] = matrix[key]
                labels_new_order.append(labels[key])
                i += 1
     # check positionning from original matrix to reorganized matrix
     # get index of first roi linked to community 0
     index_roi_com0_reorganized = list(partition.values()).index(0)
     # get nb of roi in community 0
     nb_com0 = numpy.unique(list(partition.values()), return_counts=True)[1][0]
     assert reorganized[0].sum() == matrix[index_roi_com0_reorganized].sum()

     if team_names==None:
          df_reorganized = pandas.DataFrame(index=labels_new_order, columns=labels, data=reorganized)
     else:
          team_names_new_order = []
          for ind in labels_new_order:
               team_names_new_order.append(team_names[ind])
          df_reorganized = pandas.DataFrame(index=team_names_new_order, columns=team_names, data=reorganized)
          
     ## reorganize matrix Ordinate wise
     df_reorganized = df_reorganized[df_reorganized.index]
     return df_reorganized


def hierarchical_clustering(Q, inv=False, team_names=None):
     #### hierarchical clustring for Q
     if not inv:
          dissimilarity = 1 - Q
          D = np.diag(np.diag(dissimilarity))
          dissimilarity -= D
          if not numpy.all(dissimilarity ==dissimilarity.T):
               dissimilarity = (dissimilarity+dissimilarity.T)/2
     else:
          dissimilarity = numpy.max(Q) - Q
          D = np.diag(np.diag(dissimilarity))
          dissimilarity -= D
          if not numpy.all(dissimilarity ==dissimilarity.T):
               dissimilarity = (dissimilarity+dissimilarity.T)/2
     Z = linkage(squareform(dissimilarity), 'complete')
     threshold = 0.8
     labels = fcluster(Z, threshold, criterion='distance')
     # Keep the indices to sort labels
     labels_order = numpy.argsort(labels)
     partition = {}
     for original, new in zip(numpy.arange(0, Q.shape[0]), labels_order):
          partition[original] = new 
     Q_organized = reorganize_according_to_new_indexing(Q, partition, team_names=team_names)
     Q_organized = (Q_organized+Q_organized.T)/2
     return Q_organized, labels_order


def plot_weights(matrix_KJ, weights, simulation, results_dir):
     Q = numpy.corrcoef(matrix_KJ)
     if not numpy.all(Q == Q.T):
          Q_sym = (Q+Q.T)/2

     Q_inversed = numpy.linalg.inv(Q)
     Q_inversed_sym = (Q_inversed+Q_inversed.T)/2

     #### hierarchical clustring for Q
     Q_organized, labels_order = hierarchical_clustering(Q_sym)
     Q_organized = (Q_organized+Q_organized.T)/2


     #### hierarchical clustring for Q inverse
     Q_inversed_organized, labels_order_Q_inverse = hierarchical_clustering(Q_inversed_sym, inv=True)
     Q_inversed_organized = (Q_inversed_organized+Q_inversed_organized.T)/2

     # build louvain community graph 
     G = nx.Graph(1-Q)  
     partition = community_louvain.best_partition(G, random_state=0)
     df_organized_louvain = reorganize_according_to_new_indexing(Q, partition)
     Q_organized_louvain = df_organized_louvain.values
     labels_louvain_order = df_organized_louvain.columns
     
     # build louvain community graph for inverse Q
     G = nx.Graph(numpy.max(Q_inversed) - Q_inversed)  # max(Qinv) - Qinv 
     partition = community_louvain.best_partition(G, random_state=0)
     df_organized_louvain = reorganize_according_to_new_indexing(Q_inversed, partition)
     Q_inv_organized_louvain = df_organized_louvain.values
     labels_inv_louvain_order = df_organized_louvain.columns

     data = weights[["SDMA GLS", "Mean score", "Var"]]
     plt.close('all')

     size_x_tot = 21
     f = plt.figure(figsize=(30, 5)) #, gridspec_kw={'wspace': 0})
     plt.suptitle(simulation)  
     colspan = 3  
     
     ax1 = plt.subplot2grid((1,size_x_tot), (0,0))
     seaborn.heatmap(numpy.array([data['SDMA GLS'].values]).T, center=0, yticklabels=data.index, cmap='coolwarm', square=True, xticklabels=['SDMA GLS'], fmt='.1f', ax=ax1, cbar=True, cbar_kws={'shrink': 0.25})    
     ax1.tick_params(axis='x', rotation=90)
     ax1.set_title('Weights')
     
     ax2 = plt.subplot2grid((1,size_x_tot), (0,1))
     seaborn.heatmap(numpy.array([data['Mean score'].values]).T, center=0, yticklabels=data.index, cmap='coolwarm', square=True, xticklabels=['Voxels Mean'], fmt='.1f', ax=ax2, cbar=True, cbar_kws={'shrink': 0.25})
     ax2.tick_params(axis='x', rotation=90)
     
     ax3 = plt.subplot2grid((1,size_x_tot), (0,2))
     seaborn.heatmap(numpy.array([data['Var'].values]).T, center=0, yticklabels=data.index, cmap='Reds', square=True, xticklabels=['Voxels Variance'], fmt='.1f', ax=ax3, cbar=True, cbar_kws={'shrink': 0.25})
     ax3.tick_params(axis='x', rotation=90)
     
     ax4 = plt.subplot2grid((1,size_x_tot), (0,3), colspan=colspan)
     seaborn.heatmap(Q, center=0, cmap='coolwarm', square=True, fmt='.1f', ax=ax4, cbar=True, cbar_kws={'shrink': 0.25})
     ax4.tick_params(axis='x', rotation=90)
     ax4.set_title('Q',y=-0.08,pad=-14)
    
     ax5 = plt.subplot2grid((1,size_x_tot), (0,3+ colspan), colspan=colspan)
     seaborn.heatmap(Q_inversed, center=0, cmap='coolwarm', square=True, fmt='.1f', ax=ax5, cbar=True, cbar_kws={'shrink': 0.25})
     ax5.tick_params(axis='x', rotation=90)
     ax5.set_title('Inverse Q',y=-0.08,pad=-14)
    
     ticks = numpy.arange(0, matrix_KJ.shape[0]) # for reordering labeling, needs to know index max

     ax6 = plt.subplot2grid((1,size_x_tot), (0,3+ colspan*2), colspan=colspan)
     seaborn.heatmap(Q_organized, center=0, cmap='coolwarm', square=True, fmt='.1f', ax=ax6, xticklabels=False, yticklabels=False, cbar=True, cbar_kws={'shrink': 0.25})
     ax6.set_xticks(ticks, labels_order)
     ax6.set_yticks(ticks, labels_order)
     ax6.tick_params(axis='x', rotation=90)
     ax6.set_title('Q organized',y=-0.08,pad=-14)

     ax7 = plt.subplot2grid((1,size_x_tot), (0,3+ colspan*3), colspan=colspan)
     seaborn.heatmap(Q_inversed_organized, center=0, cmap='coolwarm', square=True, fmt='.1f', ax=ax7, xticklabels=False, yticklabels=False, cbar=True, cbar_kws={'shrink': 0.25})
     ax7.set_xticks(ticks, labels_order_Q_inverse)
     ax7.set_yticks(ticks, labels_order_Q_inverse)
     ax7.tick_params(axis='x', rotation=90)
     ax7.set_title('Inverse Q organized',y=-0.08,pad=-14)

     ax8 = plt.subplot2grid((1,size_x_tot), (0,3+ colspan*4), colspan=colspan)
     seaborn.heatmap(Q_organized_louvain, center=0, cmap='coolwarm', square=True, fmt='.1f', ax=ax8, xticklabels=False, yticklabels=False, cbar=True, cbar_kws={'shrink': 0.25})
     ax8.set_xticks(ticks, labels_louvain_order)
     ax8.set_yticks(ticks, labels_louvain_order)
     ax8.tick_params(axis='x', rotation=90)
     ax8.set_title('Q Louvain organized',y=-0.08,pad=-14)

     ax9 = plt.subplot2grid((1,size_x_tot), (0,3+ colspan*5), colspan=colspan)
     seaborn.heatmap(Q_inv_organized_louvain, center=0, cmap='coolwarm', square=True, fmt='.1f', ax=ax9, xticklabels=False, yticklabels=False, cbar=True, cbar_kws={'shrink': 0.25})
     ax9.set_xticks(ticks, labels_inv_louvain_order)
     ax9.set_yticks(ticks, labels_inv_louvain_order)
     ax9.tick_params(axis='x', rotation=90)
     ax9.set_title('Q inversed Louvain organized',y=-0.08,pad=-14)


     plt.tight_layout()
     if "\n" in simulation:
          simulation = simulation.replace('\n', '')
     simulation = simulation.replace(' ', '_')
     plt.savefig("{}/weights_in_{}.png".format(results_dir, simulation))
     plt.close('all')
     print("Done plotting")



def figure_for_Narps_weights(results_dir, hyp, data, Q, name, ticks=None, labels_order=None):
     plt.close('all')
     colspan = 14
     size_x_tot = 20
     f = plt.figure(figsize=(size_x_tot+5, 15))
     if hyp =="":
          plt.suptitle("Weights for HCP")
     else:
          plt.suptitle("Weights for hypothesis ".format(hyp))

     if ticks is None:
          ax1 = plt.subplot2grid((1,size_x_tot), (0,0), colspan=2)
          seaborn.heatmap(numpy.array([data['SDMA GLS'].values]).T, center=0, yticklabels=data.index, cmap='coolwarm', square=True, xticklabels=['SDMA GLS'], fmt='.1f', ax=ax1, cbar=True, cbar_kws={'shrink': 0.25})    
          ax1.tick_params(axis='x', rotation=90)
          ax1.set_title('Weights')
          
          ax2 = plt.subplot2grid((1,size_x_tot), (0,2), colspan=2)
          seaborn.heatmap(numpy.array([data['Mean score'].values]).T, center=0, yticklabels=data.index, cmap='coolwarm', square=True, xticklabels=['Voxels Mean'], fmt='.1f', ax=ax2, cbar=True, cbar_kws={'shrink': 0.25})
          ax2.tick_params(axis='x', rotation=90)
          
          ax3 = plt.subplot2grid((1,size_x_tot), (0,4), colspan=2)
          seaborn.heatmap(numpy.array([data['Var'].values]).T, center=0, yticklabels=data.index, cmap='Reds', square=True, xticklabels=['Voxels Variance'], fmt='.1f', ax=ax3, cbar=True, cbar_kws={'shrink': 0.25})
          ax3.tick_params(axis='x', rotation=90)
     
          ax4 = plt.subplot2grid((1,size_x_tot), (0,6), colspan=colspan)
          seaborn.heatmap(Q, center=0, cmap='coolwarm', xticklabels=data.index, yticklabels=data.index, square=True, fmt='.1f', ax=ax4, cbar=True, cbar_kws={'shrink': 0.25})

     else:
          ax1 = plt.subplot2grid((1,size_x_tot), (0,0))
          seaborn.heatmap(numpy.array([data['SDMA GLS'].values[labels_order]]).T, center=0, yticklabels=False, cmap='coolwarm', square=True, xticklabels=['SDMA GLS'], fmt='.1f', ax=ax1, cbar=True, cbar_kws={'shrink': 0.25})    
          ax1.set_yticks(ticks+0.5, labels_order)
          ax1.tick_params(axis='x', rotation=90)
          ax1.set_title('Weights')
          
          ax2 = plt.subplot2grid((1,size_x_tot), (0,1))
          seaborn.heatmap(numpy.array([data['Mean score'].values[labels_order]]).T, center=0, yticklabels=False, cmap='coolwarm', square=True, xticklabels=['Voxels Mean'], fmt='.1f', ax=ax2, cbar=True, cbar_kws={'shrink': 0.25})
          ax2.set_yticks(ticks+0.5, labels_order)
          ax2.tick_params(axis='x', rotation=90)
          
          ax3 = plt.subplot2grid((1,size_x_tot), (0,2))
          seaborn.heatmap(numpy.array([data['Var'].values[labels_order]]).T, center=0, yticklabels=False, cmap='Reds', square=True, xticklabels=['Voxels Variance'], fmt='.1f', ax=ax3, cbar=True, cbar_kws={'shrink': 0.25})
          ax3.set_yticks(ticks+0.5, labels_order)
          ax3.tick_params(axis='x', rotation=90)

          ax4 = plt.subplot2grid((1,size_x_tot), (0,3), colspan=colspan)
          seaborn.heatmap(Q, center=0, cmap='coolwarm', square=True, fmt='.1f', ax=ax4, xticklabels=False, yticklabels=False, cbar=True, cbar_kws={'shrink': 0.25})
          ax4.set_xticks(ticks+0.5, labels_order)
          ax4.set_yticks(ticks+0.5, labels_order)
     ax4.tick_params(axis='x', rotation=90)
     ax4.set_title('{}'.format(name),y=-0.08,pad=-14)
     plt.tight_layout()
     if hyp =="":
          plt.savefig("{}/weights_{}.png".format(results_dir, name))
     else:
          plt.savefig("{}/hyp{}_weights_{}.png".format(results_dir, hyp, name))
     
     plt.close('all')


def plot_weights_Narps(results_dir, matrix_KJ, weights, hyp):
     team_names = list(weights.index)
     Q = numpy.corrcoef(matrix_KJ)
     Q_sym = (Q+Q.T)/2
     Q_inv = numpy.linalg.inv(Q)
     Q_inv_sym = (Q_inv+Q_inv.T)/2

     # #### hierarchical clustring for Q
     # Q_h, labels_order_Q_h = hierarchical_clustering(Q, team_names=team_names)
     # Q_h = (Q_h+Q_h.T)/2

     # #### hierarchical clustring for Q inverse
     # Q_h_inv, labels_order_Q_h_inv = hierarchical_clustering(numpy.linalg.inv(Q), inv=True, team_names=team_names)
     # Q_h_inv = (Q_h_inv+Q_h_inv.T)/2


     # # build louvain community graph for this specific hypothesis
     # G = nx.Graph(1-Q)  
     # partition = community_louvain.best_partition(G, random_state=0)
     # df_organized_louvain = reorganize_according_to_new_indexing(Q, partition)
     # Q_l = df_organized_louvain.values
     # labels_order_Q_l = df_organized_louvain.columns


     # # build louvain community graph for this specific hypothesis using Q inv
     # G = nx.Graph(numpy.max(Q_inv) - Q_inv)  # max(Qinv) - Qinv   
     # partition = community_louvain.best_partition(G, random_state=0)
     # df_organized_louvain = reorganize_according_to_new_indexing(Q_inv, partition)
     # Q_l_inv = df_organized_louvain.values
     # labels_order_Q_l_inv = df_organized_louvain.columns

     Q_inv = pandas.DataFrame(data=Q_inv, columns=team_names, index=team_names)
     Q = pandas.DataFrame(data=Q, columns=team_names, index=team_names)

     data = weights[["SDMA GLS", "Mean score", "Var"]]
     ticks = numpy.arange(0, matrix_KJ.shape[0]) # for reordering labeling, needs to know index max
     
     figure_for_Narps_weights(results_dir, hyp, data, Q, "Q", ticks=None, labels_order=None)
     figure_for_Narps_weights(results_dir, hyp, data, Q_inv, "Qinv", ticks=None, labels_order=None)
     # figure_for_Narps_weights(results_dir, hyp, data, Q_h, "Q_hierarchical", ticks=ticks, labels_order=labels_order_Q_h)
     # figure_for_Narps_weights(results_dir, hyp, data, Q_h_inv, "Q_hierarchical_inv", ticks=ticks, labels_order=labels_order_Q_h_inv)
     # figure_for_Narps_weights(results_dir, hyp, data, Q_l, "Q_louvain", ticks=ticks, labels_order=labels_order_Q_l)
     # figure_for_Narps_weights(results_dir, hyp, data, Q_l_inv, "Q_louvain_inv", ticks=ticks, labels_order=labels_order_Q_l_inv)

     print("Done plotting")

def paper_figure_simulations(generated_data, results_dir):
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
        axs[index].set_title("{} data pipeline\n Corr = {}".format(title, numpy.round(corr_mat.mean(), 2)), fontsize=18)
        axs[index].set_xlabel("J voxels", fontsize = 18)
        axs[index].set_ylabel("K pipelines", fontsize = 18)
        axs[index].tick_params(axis='y', labelrotation=0)
    plt.tight_layout()
    plt.savefig("{}/Fig1.png".format(results_dir))
    plt.close('all')
    print("Done plotting")
