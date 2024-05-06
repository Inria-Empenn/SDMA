
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
from datetime import datetime
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

# folder to store extracted resampled z maps
if not os.path.exists(os.path.join(results_dir, "data")):
    os.mkdir(os.path.join(results_dir, "data"))

# save log
dt = datetime.now()
if not os.path.exists("{}/log.txt".format(results_dir)):
    with open("{}/log.txt".format(results_dir), 'w') as f:
        f.write("Last ran the {}/{} at {}h{}".format(dt.day, dt.month, dt.hour, dt.minute))
else:
    with open("{}/log.txt".format(results_dir), 'a') as f:
        f.write("\nLast ran the {}/{} at {}h{}".format(dt.day, dt.month, dt.hour, dt.minute))

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
    "Consensus Average",
    "GLS SDMA",
    "Consensus GLS SDMA"]

MA_estimators_names = ["Stouffer",
    "SDMA Stouffer",
    "Consensus \nSDMA Stouffer",
    "Consensus Average",
    "SDMA GLS",
    "Consensus SDMA GLS"]

stop

similarity_mask_per_hyp = []

hyps = [1, 2, 5, 6, 7, 8, 9]#numpy.arange(1, 10, 1)
for hyp in hyps:
    print('*****Running hyp ', hyp, '*****')
    if not os.path.exists(os.path.join(results_dir, "hyp{}".format(hyp))):
        os.mkdir(os.path.join(results_dir, "hyp{}".format(hyp)))
    results_dir_hyp = os.path.join(results_dir, "hyp{}".format(hyp))
    # check if resampled_maps already exists:
    # try:
    #     resampled_maps_per_team = numpy.load('{}/data/Hyp{}_resampled_maps.npy'.format(results_dir, hyp), allow_pickle=True).item() # mind result dir
    #     print("resampled_maps successfully loaded")
    # except:
    #     print("Data don't already exist thus starting resampling.")
    resampled_maps_per_team = extract_narps_data.resample_NARPS_unthreshold_maps(data_path, hyp, weird_maps, participant_mask)
    print("Saving resampled NARPS unthreshold maps...")
    numpy.save("{}/data/Hyp{}_resampled_maps.npy".format(results_dir, hyp), resampled_maps_per_team, allow_pickle=True, fix_imports=True)
    time.sleep(2)
    # print("plotting brains...")
    # narps_visualisation.plot_nii_maps(resampled_maps_per_team, masker, hyp, os.path.join(results_dir, "data"), "resampled")
    print("Starting Masking...")
    resampled_maps= masker.fit_transform(resampled_maps_per_team.values())
    team_names = list(resampled_maps_per_team.keys())
    print("Masking DONE")
    print("Z values extracted, shape=", resampled_maps.shape) # 61, 1537403 for hyp 1

    # compute several MA estimators for the obtained matrix
    # try:
    #     MA_outputs = numpy.load('{}/data/Hyp{}_MA_estimates.npy'.format(results_dir, hyp),allow_pickle=True).item()
    #     print("MA_outputs successfully loaded")
    # except:
     # print("Data don't already exist thus recomputing MA estimates.")
    MA_outputs = compute_MA_outputs.get_MA_outputs(resampled_maps)
    print("Saving MA estimates...")
    numpy.save("{}/data/Hyp{}_MA_estimates".format(results_dir, hyp), MA_outputs, allow_pickle=True, fix_imports=True)
    print("Building figure 1... distributions")
    # narps_visualisation.plot_distributions(MA_outputs, hyp, MA_estimators_names, results_dir_hyp)
    print("Building figure 2... MA results on brains no fdr")
    narps_visualisation.plot_brain_nofdr(MA_outputs, hyp, MA_estimators_names, results_dir_hyp, masker)
    print("Building figure 3... similarities/contrasts...")
    # similarity_mask = narps_visualisation.plot_SDMA_results_divergence(MA_outputs, hyp, MA_estimators_names, results_dir_hyp, masker)
    # similarity_mask_per_hyp.append(similarity_mask)
    print('Saving weights..')
    # df_weights = pandas.DataFrame(columns=MA_outputs.keys(), index=team_names)
    # K, J = resampled_maps.shape
    # for row in range(K):
    #     for MA_model in MA_outputs.keys():
    #         df_weights[MA_model] = MA_outputs[MA_model]['weights']
    # df_weights["Mean score"] = resampled_maps.mean(axis=1)
    # df_weights["Var"] = resampled_maps.std(axis=1)
    # print("Building figure 4... weights")
    # utils.plot_weights_Narps(results_dir_hyp, resampled_maps, df_weights, hyp)
    # plot residuals
    # print("Computing residuals...")
    # coefficients, residuals_maps = narps_visualisation.compute_betas(resampled_maps)
    # print("Building figure 5... betas (for residuals)")
    # narps_visualisation.plot_betas(coefficients, hyp, results_dir_hyp, team_names)
    # print("Building figure 6... residuals")
    # residuals_maps_per_team = {}
    # for team, maps in zip(team_names, residuals_maps):
    #     residuals_maps_per_team[team] = masker.inverse_transform(maps)
    # narps_visualisation.plot_nii_maps(residuals_maps_per_team, masker, hyp, results_dir_hyp, "residuals")
    # resampled_maps, coefficients, residuals_maps, residuals_nii_maps  = None, None, None, None # empyting RAM memory

# narps_visualisation.plot_hyp_similarities(similarity_mask_per_hyp, results_dir)



