
import os
import numpy
import nibabel
import time
from nilearn.input_data import NiftiMasker
import extract_narps_data
import compute_MA_outputs
import narps_visualisation
import importlib

importlib.reload(narps_visualisation)

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

results_dir = "results_in_Narps"
if not os.path.exists(results_dir):
    os.mkdir(results_dir)

data_path = '/home/jlefortb/neurovault_narps_open_pipeline/orig/'
participant_mask = nibabel.load("masking/mask_90.nii")

# save mask for inverse transform
masker = NiftiMasker(
    mask_img=participant_mask)
results_dir = "results_narpsdata"
if not os.path.exists(results_dir):
    os.mkdir(results_dir)

#### NOT INCLUDED IN ANALYSIS 
# "4961_K9P0" only hyp 9 is weird
weird_maps = ["4951_X1Z4", "5680_L1A8", "5001_I07H", 
    "4947_X19V", "4961_K9P0", "4974_1K0E", "4990_XU70",
        "5001_I07H", "5680_L1A8"]

MA_estimators_names = ["Average",
    "Stouffer",
    "Dependence-Corrected \nStouffer",
    "GLS Stouffer",
    "Consensus Stouffer",
    "Consensus Weighted \nStouffer",
    "Consensus GLS \nStouffer",
    "Consensus Average"]

hyps = [1, 2, 5, 6, 7, 8, 9]#numpy.arange(1, 10, 1)
for hyp in hyps:
    # check if resampled_maps already exists:
    try:
        resampled_maps = numpy.load('{}/Hyp{}_resampled_maps.npy'.format(results_dir, hyp))
        print("resampled_maps successfully loaded")
    except:
        print("Data don't already exist thus starting resampling.")
        resampled_maps = extract_narps_data.resample_NARPS_unthreshold_maps(data_path, hyp, weird_maps, participant_mask, masker)
        print("Saving resampled NARPS unthreshold maps...")
        numpy.save("{}/Hyp{}_resampled_maps".format(results_dir, hyp), resampled_maps, allow_pickle=True, fix_imports=True)
    print("Starting Masking...")
    resampled_maps = masker.fit_transform(resampled_maps)
    print("Masking DONE")
    print("*****")
    print("Z values extracted, shape=", resampled_maps.shape) # 61, 1537403 for hyp 1
    print("*****")
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
    resampled_maps = None # empyting RAM memory
