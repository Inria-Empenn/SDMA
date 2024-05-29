
import os
import numpy
import nibabel
from nilearn.input_data import NiftiMasker
import compute_MA_outputs
from utils import plot_significant_Pvalues_into_MNI


##################
# Apply SDMA in all NARPS hypothesis outcomes
##################

# path to partiticipants mask
participants_mask_path = os.path.join("results", "NARPS", "masking", "participants_mask.nii")
participant_mask = nibabel.load(participants_mask_path)
# path to resampled NARPS data
data_path = os.path.join("data", "NARPS")
# folder to store results
results_dir = os.path.join("results", "NARPS")
figures_dir = os.path.join("figures", "NARPS")

# hypotheses = {1: '+gain: equal indiff',
#               2: '+gain: equal range',
#               3: '+gain: equal indiff',
#               4: '+gain: equal range',
#               5: '-loss: equal indiff',
#               6: '-loss: equal range',
#               7: '+loss: equal indiff',
#               8: '+loss: equal range',
#               9: '+loss:ER>EI'}

# save mask for inverse transform
masker = NiftiMasker(
    mask_img=participant_mask)

#### NOT INCLUDED IN ANALYSIS 
# "4961_K9P0" only hyp 9 is weird
weird_maps = ["4951_X1Z4", "5680_L1A8", "5001_I07H", 
    "4947_X19V", "4961_K9P0", "4974_1K0E", "4990_XU70",
        "5001_I07H", "5680_L1A8"]

MA_estimators_names = ["Stouffer",
    "SDMA Stouffer",
    "Consensus \nSDMA Stouffer",
    "Consensus Average",
    "SDMA GLS",
    "Consensus SDMA GLS"]

# compute and plot significant p values for each hypothesis and SDMA methods
hyps = [1, 2, 5, 6, 7, 8, 9]
for hyp in hyps:
    print('*****Running SDMA in NARPS hyp ', hyp, '*****')
    results_dir_hyp = os.path.join(results_dir, "hyp{}".format(hyp), "SDMA")
    figures_dir_hyp = os.path.join(figures_dir, "hyp{}".format(hyp), "SDMA")
    
    # load data
    data_path = os.path.join("data", "NARPS")
    resampled_maps_per_team = numpy.load(os.path.join(data_path, "Hyp{}_resampled_maps.npy".format(hyp)), allow_pickle=True).item()

    resampled_maps = masker.fit_transform(resampled_maps_per_team.values())
    team_names = list(resampled_maps_per_team.keys())
    print("Data loading + masking DONE")

    print("Computing MA estimates...")
    MA_outputs = compute_MA_outputs.get_MA_outputs(resampled_maps)

    print("Saving MA estimates...")
    numpy.save(os.path.join(results_dir_hyp, "Hyp{}_MA_estimates".format(hyp)), MA_outputs, allow_pickle=True, fix_imports=True)

    print("Building figure... MA results mapped onto MNI brains")
    plot_significant_Pvalues_into_MNI(MA_outputs, hyp, MA_estimators_names, figures_dir_hyp, masker)
    
    resampled_maps, resampled_maps_per_team = None, None # empyting RAM memory




