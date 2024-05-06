
import os
import numpy
import nibabel
import time
from nilearn.input_data import NiftiMasker
import pandas
from datetime import datetime
import utils
import importlib
import random

importlib.reload(utils)

# Create folder
results_dir = "Tests_with_different_Q/results"
if not os.path.exists(results_dir):
    os.mkdir(results_dir)

# save log
dt = datetime.now()
if not os.path.exists("{}/log.txt".format(results_dir)):
    with open("{}/log.txt".format(results_dir), 'w') as f:
        f.write("Last ran the {}/{} at {}h{}".format(dt.day, dt.month, dt.hour, dt.minute))
else:
    with open("{}/log.txt".format(results_dir), 'a') as f:
        f.write("\nLast ran the {}/{} at {}h{}".format(dt.day, dt.month, dt.hour, dt.minute))

# save mask for inverse transform
participant_mask = nibabel.load("masking/mask_90.nii")
masker = NiftiMasker(
    mask_img=participant_mask)


hyp = 1

#get data from results_in_Narps_data
contrast_estimates_per_team = numpy.load('results_in_Narps_data/data/Hyp{}_resampled_maps.npy'.format(hyp), allow_pickle=True).item() # mind result dir
print("Contrast estimates successfully loaded")
contrast_estimates= masker.fit_transform(contrast_estimates_per_team.values())
team_names = list(contrast_estimates_per_team.keys())
time.sleep(2)


# for cluster analysis
pipeline_z_scores_per_team = numpy.load('results_in_Narps_data/data/Hyp{}_resampled_maps.npy'.format(hyp), allow_pickle=True).item() # mind result dir
pipeline_z_scores_per_team.pop("5496_VG39")# not included in narps study, thus to be removed

############################
# PREPARE DATA FOR 3 clusters RESULTS : correlated, anti-correlated, independent
############################

## get mean weight per cluster of teams 
## define cluster of team from Narps paper figure 2
correlated = ["AO86", "43FJ", "O21U", "3PQ2", "0JO0", "I9D6", "51PW", "94GU", "0ED6", "R5K7", "SM54", "B23O",
                "O03M","DC61", "X1Y5", "UI76", "2T7P", "2T6S", "27SS", "T54A", "1KB2", "08MQ", "V55J",
                "3TR7", "Q6O0", "E3B6", "L7J7", "9Q6R", "U26C", "50GV", "B5I6", "R9K3", "C88N", 
                "J7F9", "46CD", "C22U", "I52Y", "E6R3", "R7D1", "0C7Q", "6VV2", "98BT", "6FH5", "3C6G", "L3V8", "0I4U",
                "0H5E", "9U7M"]
anti_correlated = ["80GC", "1P0Y", "P5F3", "IZ20", "Q58J", "4TQ6", 'UK24']
independant = ["9T8E", "R42Q", "L9G5", "O6R6", "4SZ2"]

print("Getting full team names")
correlated_full_name = utils.get_full_name(correlated,contrast_estimates_per_team)
anti_correlated_full_name = utils.get_full_name(anti_correlated,contrast_estimates_per_team)
independant_full_name = utils.get_full_name(independant,contrast_estimates_per_team)

clusters = [correlated_full_name,
anti_correlated_full_name,
independant_full_name]

clusters_name = ["correlated",
"anti_correlated",
"independant"]



def compute_MA(Q, Q_name, contrast_estimates,team_names):
    print('*****Running ', Q_name, '*****')
    if not os.path.exists(os.path.join(results_dir, "{}".format(Q_name))):
        os.mkdir(os.path.join(results_dir, "{}".format(Q_name)))
    results_dir_hyp = os.path.join(results_dir, "{}".format(Q_name))

    MA_estimators_names = ["SDMA Stouffer", "GLS SDMA"]
    MA_outputs = {}  

    T_map, p_values, weights = utils.SDMA_Stouffer(contrast_estimates, Q)
    ratio_significance_raw = (p_values <= 0.05).sum() / len(p_values)
    ratio_significance = numpy.round(ratio_significance_raw * 100, 4)
    MA_outputs["SDMA Stouffer"] = {
       'T_map': T_map,
       'p_values': p_values,
       'ratio_significance': ratio_significance,
       'weights': weights
        }

    T_map, p_values, weights = utils.GLS_SDMA(contrast_estimates, Q)
    ratio_significance_raw = (p_values <= 0.05).sum() / len(p_values)
    ratio_significance = numpy.round(ratio_significance_raw * 100, 4)
    MA_outputs["GLS SDMA"] = {
       'T_map': T_map,
       'p_values': p_values,
       'ratio_significance': ratio_significance,
       'weights': weights
    } 
    # PLOT RESULTS

    utils.plot_brain(MA_outputs, hyp, results_dir_hyp, masker)

    print('Saving weights..')
    df_weights = pandas.DataFrame(columns=MA_outputs.keys(), index=team_names)
    K, J = contrast_estimates.shape
    for row in range(K):
        for MA_model in MA_outputs.keys():
            df_weights[MA_model] = MA_outputs[MA_model]['weights'].reshape(-1)
    df_weights["Mean score"] = contrast_estimates.mean(axis=1)
    df_weights["Var"] = contrast_estimates.std(axis=1)
    print("Building figure 4... weights")

    utils.plot_weights(results_dir_hyp, contrast_estimates, Q, df_weights)
    with open("{}/log.txt".format(results_dir), 'a') as f:
        f.write("\nRan first Q until the end {}/{} at {}h{}".format(dt.day, dt.month, dt.hour, dt.minute))


def compute_cluster_analysis(pipeline_z_scores_per_team, clusters, clusters_name, masker, Q_name):
    ##### 3 clusters : correlated, anti-correlated, independent
    results_dir_hyp = os.path.join(results_dir, "{}".format(Q_name))
    if not os.path.exists(os.path.join(results_dir_hyp, "3clusters_results")):
        os.mkdir(os.path.join(results_dir_hyp, "3clusters_results"))

    team_names = list(pipeline_z_scores_per_team.keys())
    pipeline_z_scores= masker.fit_transform(pipeline_z_scores_per_team.values())
    pipeline_z_scores_per_team = None # save RAM

    if Q_name == "Voxel_wise_centered_data":
        Q = utils.Q_from_std_data(pipeline_z_scores)
    elif Q_name == "Regress_out_mean_effect_data":
        Q = utils.Q_regress_out_mean_effect(pipeline_z_scores)
    elif Q_name == "Subsampled_voxels_data":
        Q = utils.Q_subsampled_voxels(pipeline_z_scores)
    elif Q_name == "Subsampled_voxels_data_small_subsample":
        Q = utils.Q_subsampled_voxels_small_subsample(pipeline_z_scores)
    else:
        print("!! NO QNAME FOUND..... !!")
        Q = "not found"
    utils.plot_clusters_brains(clusters, clusters_name, team_names, masker, pipeline_z_scores, results_dir_hyp, Q)
    with open("{}/log.txt".format(results_dir), 'a') as f:
        f.write("\nRan second (cluster) Q until the end {}/{} at {}h{}".format(dt.day, dt.month, dt.hour, dt.minute))


########################
# COMPUTE for Q = Voxel_wise_centered_data
########################

### COMPUTE SDMA + PLOT WEIGHTS
Q_name = "Voxel_wise_centered_data"
Q = utils.Q_from_std_data(contrast_estimates)
compute_MA(Q, Q_name, contrast_estimates, team_names)
compute_cluster_analysis(pipeline_z_scores_per_team, clusters, clusters_name, masker, Q_name)


########################
# COMPUTE for Q = Regress_out_mean_effect
########################
Q_name = "Regress_out_mean_effect_data"
Q = utils.Q_regress_out_mean_effect(contrast_estimates)
compute_MA(Q, Q_name, contrast_estimates, team_names)
compute_cluster_analysis(pipeline_z_scores_per_team, clusters, clusters_name, masker, Q_name)


########################
# COMPUTE for Q = Subsampled_voxels
########################
Q_name = "Subsampled_voxels_data"
Q = utils.Q_subsampled_voxels(contrast_estimates)
compute_MA(Q, Q_name, contrast_estimates, team_names)
compute_cluster_analysis(pipeline_z_scores_per_team, clusters, clusters_name, masker, Q_name)


########################
# COMPUTE for Q = Subsampled_voxels
########################
Q_name = "Subsampled_voxels_data_small_subsample"
Q = utils.Q_subsampled_voxels_small_subsample(contrast_estimates)
compute_MA(Q, Q_name, contrast_estimates, team_names)
compute_cluster_analysis(pipeline_z_scores_per_team, clusters, clusters_name, masker, Q_name)
