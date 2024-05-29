
import os
import numpy
import nibabel
from nilearn.input_data import NiftiMasker
import compute_MA_outputs
from glob import glob
from utils import plot_significant_Pvalues_into_MNI

# path to partiticipants mask
participants_mask_path = os.path.join("results", "HCP", "masking", "participant_mask.nii.gz")
participant_mask = nibabel.load(participants_mask_path)
# path to resampled NARPS data
data_path = os.path.join("data", "HCP")
# folder to store results
results_dir = os.path.join("results", "HCP", "SDMA")
figures_dir = os.path.join("figures", "HCP", "SDMA")

MA_estimators_names = [
    "Stouffer",
    "SDMA Stouffer",
    "Consensus \nSDMA Stouffer",
    "Consensus Average",
    "SDMA GLS",
    "Consensus SDMA GLS"]

# save mask for inverse transform
masker = NiftiMasker(
    mask_img=participant_mask)

print('*****Running SDMA in HCP*****')

print("Data loading + masking...")
nifti_files = glob(os.path.join(data_path, "*.nii"))
resampled_maps = masker.fit_transform(nifti_files)
team_names = [file[61:-10] for file in nifti_files]

print("Data loading + masking DONE")
print("Computing MA estimates...")
MA_outputs = compute_MA_outputs.get_MA_outputs(resampled_maps)
print("Saving MA estimates...")
numpy.save(os.path.join(results_dir, "MA_estimates"), MA_outputs, allow_pickle=True, fix_imports=True)
print("Building figure... MA results mapped onto MNI brains")
plot_significant_Pvalues_into_MNI(MA_outputs, "", MA_estimators_names, figures_dir, masker)
resampled_maps = None # empyting RAM memory
