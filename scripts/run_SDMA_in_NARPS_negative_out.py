import os
import numpy
import nibabel
import time
from nilearn.input_data import NiftiMasker
import compute_MA_outputs
import narps_visualisation
import importlib
import pandas
import utils
from datetime import datetime

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

results_dir = "results_in_Narps_data_negative_out"
if not os.path.exists(results_dir):
    os.mkdir(results_dir)

# folder to store extracted resampled z maps
if not os.path.exists(os.path.join(results_dir, "data")):
    os.mkdir(os.path.join(results_dir, "data"))


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


negative_maps_team_name_per_hyp =[
    ['5164_Q58J',
      '4908_UK24',
      '5649_1P0Y',
      '4979_IZ20',
      '4967_P5F3',
      '4869_4TQ6',
      '4891_80GC'],
     [],
     ['4965_9U7M',
      '4883_6VV2',
      '4866_L7J7',
      '4984_B23O',
      '5637_46CD',
      '4932_AO86',
      '4972_O03M',
      '5619_R42Q',
      '4978_I9D6',
      '4807_0JO0',
      '4821_UI76',
      '4869_4TQ6',
      '4975_27SS',
      '4959_E6R3',
      '4891_80GC',
      '5675_SM54',
      '4881_2T6S'],
     ['4965_9U7M',
      '4883_6VV2',
      '4866_L7J7',
      '4984_B23O',
      '5637_46CD',
      '4932_AO86',
      '4972_O03M',
      '5619_R42Q',
      '4978_I9D6',
      '4807_0JO0',
      '4821_UI76',
      '4869_4TQ6',
      '4975_27SS',
      '4959_E6R3',
      '4891_80GC',
      '5675_SM54'],
     ['4932_AO86', '4963_DC61', '4881_2T6S'],
     ['4932_AO86', '4963_DC61'],
     ['4866_L7J7',
      '4824_43FJ',
      '4967_P5F3',
      '5496_VG39',
      '4807_0JO0',
      '4975_27SS',
      '5675_SM54']]



MA_estimators_names = ["Average",
    "Stouffer",
    "SDMA Stouffer",
    "Consensus \nSDMA Stouffer",
    "Consensus \nSDMA Stouffer \n using std inputs",
    "Consensus Average",
    "GLS SDMA",
    "Consensus GLS SDMA"]
similarity_mask_per_hyp = []

hyps = [1, 2, 5, 6, 7, 8, 9]#numpy.arange(1, 10, 1)
for ind, hyp in enumerate(hyps):
    print('*****Running hyp ', hyp, '*****')
    if not os.path.exists(os.path.join(results_dir, "hyp{}".format(hyp))):
        os.mkdir(os.path.join(results_dir, "hyp{}".format(hyp)))
    results_dir_hyp = os.path.join(results_dir, "hyp{}".format(hyp))
    # check if resampled_maps already exists:
    try:
        resampled_maps_per_team = numpy.load('results_in_Narps_data/data/Hyp{}_resampled_maps.npy'.format(hyp), allow_pickle=True).item()
        print("resampled_maps successfully loaded")
    except:
        print("Data don't already exist...")
    print("Starting Masking...")
    team_names = list(resampled_maps_per_team.keys())
    for tname in negative_maps_team_name_per_hyp[ind]:
        print("removing anticorrelated team: ", tname)
        del resampled_maps_per_team[tname]
    print("new length:", len(resampled_maps_per_team))
    print("DOUBLE CHECK team above")

    resampled_maps = masker.fit_transform(resampled_maps_per_team.values())
    team_names = list(resampled_maps_per_team.keys())
    print("Masking DONE")
    print("Z values extracted, shape=", resampled_maps.shape) # 61, 1537403 for hyp 1
    print("shape=", resampled_maps.shape, ", should be ", 61-len(negative_maps_team_name_per_hyp[ind])) 
    time.sleep(2)
    # compute several MA estimators for the obtained matrix
    print("computing MA estimates.")
    try:
        MA_outputs = numpy.load('{}/Hyp{}_MA_estimates.npy'.format(results_dir_hyp, hyp),allow_pickle=True).item()
        print("MA_outputs successfully loaded")
    except:
        print("Data don't already exist thus recomputing MA estimates.")
        MA_outputs = compute_MA_outputs.get_MA_outputs(resampled_maps)
        print("Saving MA estimates...")
        numpy.save("{}/Hyp{}_MA_estimates".format(results_dir_hyp, hyp), MA_outputs, allow_pickle=True, fix_imports=True)
    print("Building figure 1... distributions")
    narps_visualisation.plot_distributions(MA_outputs, hyp, MA_estimators_names, results_dir_hyp)
    print("Building figure 2... MA results on brains no fdr")
    narps_visualisation.plot_brain_nofdr(MA_outputs, hyp, MA_estimators_names, results_dir_hyp, masker)
    print("Building figure 3... similarities/contrasts...")
    similarity_mask = narps_visualisation.plot_SDMA_results_divergence(MA_outputs, hyp, MA_estimators_names, results_dir_hyp, masker)
    similarity_mask_per_hyp.append(similarity_mask)
    print('Saving weights..')
    df_weights = pandas.DataFrame(columns=MA_outputs.keys(), index=team_names)
    K, J = resampled_maps.shape
    for row in range(K):
        for MA_model in MA_outputs.keys():
            df_weights[MA_model] = MA_outputs[MA_model]['weights']
    df_weights["Mean score"] = resampled_maps.mean(axis=1)
    df_weights["Var"] = resampled_maps.std(axis=1)
    print("Building figure 5... weights")
    utils.plot_weights_Narps(results_dir_hyp, resampled_maps, df_weights, hyp)
    # plot residuals
    print("Computing residuals...")
    coefficients, residuals_maps = narps_visualisation.compute_betas(resampled_maps)
    print("Building figure 6... betas (for residuals)")
    narps_visualisation.plot_betas(coefficients, hyp, results_dir_hyp, team_names)
    print("Building figure 7... residuals")
    residuals_maps_per_team = {}
    for team, maps in zip(team_names, residuals_maps):
        residuals_maps_per_team[team] = masker.inverse_transform(maps)
    narps_visualisation.plot_nii_maps(residuals_maps_per_team, masker, hyp, results_dir_hyp, "residuals")
    resampled_maps, coefficients, residuals_maps, residuals_nii_maps  = None, None, None, None # empyting RAM memory

narps_visualisation.plot_hyp_similarities(similarity_mask_per_hyp, results_dir)

