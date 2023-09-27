
import os
import numpy
import nibabel
import time
from nilearn.input_data import NiftiMasker
import extract_narps_data
import compute_MA_outputs
import narps_visualisation
import importlib
import pandas
import utils

importlib.reload(narps_visualisation)
importlib.reload(utils)
'''
hypotheses = {1: '+gain: equal indiff',
              2: '+gain: equal range',
              3: '+gain: equal indiff',
              4: '+gain: equal range',
              5: '-loss: equal indiff',
              6: '-loss: equal range',
              7: '+loss: equal indiff',
              8: '+loss: equal range',
              9: '+loss:ER>EI'}
'''

results_dir = "results_in_Narps_data"
if not os.path.exists(results_dir):
    os.mkdir(results_dir)

data_path = '/home/jlefortb/neurovault_narps_open_pipeline/orig/'
participant_mask = nibabel.load("masking/mask_90.nii")

# save mask for inverse transform
masker = NiftiMasker(
    mask_img=participant_mask)


#### NOT INCLUDED IN ANALYSIS 
# "4961_K9P0" only hyp 9 is weird
weird_maps = ["4951_X1Z4", "5680_L1A8", "5001_I07H", 
    "4947_X19V", "4961_K9P0", "4974_1K0E", "4990_XU70",
        "5001_I07H", "5680_L1A8"]


MA_estimators_names = ["Average",
    "Stouffer",
    "SDMA Stouffer",
    "Consensus \nSDMA Stouffer",
    "Consensus \nSDMA Stouffer \n using std inputs",
    "GLS \nSDMA Stouffer",
    "Consensus \nGLS \nSDMA Stouffer",
    "Consensus Average"]

similarity_mask_per_hyp = []

hyps = [1, 2, 5, 6, 7, 8, 9]#numpy.arange(1, 10, 1)
for hyp in hyps:
    print('*****Running hyp ', hyp, '*****')
    # check if resampled_maps already exists:
    try:
        resampled_maps = numpy.load('{}/Hyp{}_resampled_maps.npy'.format(results_dir, hyp), allow_pickle=True)
        print("resampled_maps successfully loaded")
    except:
        print("Data don't already exist thus starting resampling.")
        resampled_maps = extract_narps_data.resample_NARPS_unthreshold_maps(data_path, hyp, weird_maps, participant_mask, masker)
        print("Saving resampled NARPS unthreshold maps...")
        numpy.save("{}/Hyp{}_resampled_maps".format(results_dir, hyp), resampled_maps, allow_pickle=True, fix_imports=True)
    print("Starting Masking...")
    resampled_maps = masker.fit_transform(resampled_maps)
    print("Masking DONE")
    print("Z values extracted, shape=", resampled_maps.shape) # 61, 1537403 for hyp 1
    time.sleep(2)
    # compute several MA estimators for the obtained matrix
    try:
        MA_outputs = numpy.load('{}/Hyp{}_MA_estimates.npy'.format(results_dir, hyp),allow_pickle=True).item()
        print("MA_outputs successfully loaded")
    except:
        print("Data don't already exist thus recomputing MA estimates.")
        MA_outputs = compute_MA_outputs.get_MA_outputs(resampled_maps)
        print("Saving MA estimates...")
        numpy.save("{}/Hyp{}_MA_estimates".format(results_dir, hyp), MA_outputs, allow_pickle=True, fix_imports=True)
    print("Building figure 1...")
    narps_visualisation.plot_distributions(MA_outputs, hyp, MA_estimators_names, results_dir)
    print("Building figure 2...")
    narps_visualisation.plot_brains(MA_outputs, hyp, MA_estimators_names, results_dir, masker)
    print("Building figure 3...")
    narps_visualisation.plot_brain_nofdr(MA_outputs, hyp, MA_estimators_names, results_dir, masker)
    print("Building figure similarities/contrasts...")
    similarity_mask = narps_visualisation.plot_SDMA_results_divergence(MA_outputs, hyp, MA_estimators_names, results_dir, masker)
    similarity_mask_per_hyp.append(similarity_mask)
    print('Saving weights..')
    df_weights = pandas.DataFrame(columns=MA_outputs.keys())
    K, J = resampled_maps.shape
    for row in range(K):
        for MA_model in MA_outputs.keys():
            df_weights[MA_model] = MA_outputs[MA_model]['weights']
    df_weights["Mean score"] = resampled_maps.mean(axis=1)
    df_weights["Var"] = resampled_maps.std(axis=1)
    print("computing Q")
    Q = numpy.corrcoef(resampled_maps)
    utils.plot_weights_in_Narps(Q, hyp, df_weights)
    resampled_maps = None # empyting RAM memory

narps_visualisation.plot_hyp_similarities(similarity_mask_per_hyp, results_dir)
narps_visualisation.plot_expected_significant_rois(results_dir)
""" if necessary
# print resampled maps
from nilearn import plotting
import matplotlib.pyplot as plt

hyps = [1, 2, 5, 6, 7, 8, 9]#numpy.arange(1, 10, 1)
for hyp in hyps:
    print("******Plotting map for hyp ", hyp, "******")
    resampled_maps = numpy.load('{}/Hyp{}_resampled_maps.npy'.format(results_dir, hyp), allow_pickle=True)
    K = resampled_maps.shape[0]

    # 60 / 20 = 3 figures with 20 plots each
    for i_fig in range(4):
        plt.close('all')
        f, axs = plt.subplots(int(11), 2, figsize=(30, 35))
        if i_fig == 0:
            for pipeline_nb in range(11):
                print("Plotting map ", pipeline_nb)
                plotting.plot_stat_map(resampled_maps[pipeline_nb], cut_coords=(-21, 0, 9), figure=f, axes=axs[pipeline_nb, 0], title=pipeline_nb)
                plotting.plot_stat_map(resampled_maps[K - pipeline_nb - 1], cut_coords=(-21, 0, 9), figure=f, axes=axs[pipeline_nb, 1], title=K - pipeline_nb - 1)
            print('saving...')
            plt.suptitle("hyp {} part {}".format(hyp, i_fig),fontsize=20)
            plt.savefig("{}/Hyp{}_resampled_maps_part_{}".format(results_dir, hyp, i_fig))
            plt.close('all')
        elif i_fig == 1:
            for pipeline_nb in range(11, 22):
                row = pipeline_nb - 11
                print("Plotting map ", pipeline_nb)
                plotting.plot_stat_map(resampled_maps[pipeline_nb], cut_coords=(-21, 0, 9), figure=f, axes=axs[row, 0], title=pipeline_nb)
                plotting.plot_stat_map(resampled_maps[K - pipeline_nb - 1], cut_coords=(-21, 0, 9), figure=f, axes=axs[row, 1], title=K - pipeline_nb - 1)
            print('saving...')
            plt.suptitle("hyp {} part {}".format(hyp, i_fig),fontsize=20)
            plt.savefig("{}/Hyp{}_resampled_maps_part_{}".format(results_dir, hyp, i_fig))
            plt.close('all')
        elif i_fig == 2:
            for pipeline_nb in range(22, 31):
                row = pipeline_nb - 22
                print("Plotting map ", pipeline_nb)
                try:
                    plotting.plot_stat_map(resampled_maps[pipeline_nb], cut_coords=(-21, 0, 9), figure=f, axes=axs[row, 0], title=pipeline_nb)
                    plotting.plot_stat_map(resampled_maps[K - pipeline_nb - 1], cut_coords=(-21, 0, 9), figure=f, axes=axs[row, 1], title=K - pipeline_nb - 1)
                except:
                    pass
            print('saving...')
            plt.suptitle("hyp {} part {}".format(hyp, i_fig),fontsize=20)
            plt.savefig("{}/Hyp{}_resampled_maps_part_{}".format(results_dir, hyp, i_fig))
            plt.close('all')
    print('saving...')
    plt.suptitle("hyp {} part {}".format(hyp, i_fig),fontsize=20)
    plt.savefig("{}/Hyp{}_resampled_maps_part_{}".format(results_dir, hyp, i_fig))
    plt.close('all')
"""