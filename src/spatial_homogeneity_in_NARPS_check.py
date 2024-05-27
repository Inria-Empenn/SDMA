import os
import math
import numpy
import nilearn.plotting
import nilearn.input_data
from nilearn import masking
from nilearn import image
import matplotlib.pyplot as plt
from nilearn.datasets import load_mni152_brain_mask
from nilearn.input_data import NiftiMasker
import pandas
import compute_MA_outputs
import nibabel as nib
from nilearn.datasets import fetch_atlas_aal
import seaborn

##################
# Check how fair is the assumption of same Q accross the brain for all hypotheses
##################


# path to partiticipants mask
participants_mask_path = "results/NARPS/masking/participants_mask.nii"
# path to resampled NARPS data
data_path = os.path.join("data", "NARPS")
# create folder to store results
results_dir = os.path.join("results", "NARPS")
figures_dir = os.path.join("figures", "NARPS")


# load mask made from participant zmaps + MNI brain mask
participants_mask = nib.load(participants_mask_path)

# load each mask in memory
frontal_path = os.path.join(results_dir, "masking", "Frontal_mask_AAL.nii")
occipital_path = os.path.join(results_dir, "masking", "Occipital_mask_AAL.nii")
parietal_path = os.path.join(results_dir, "masking", "Parietal_mask_AAL.nii")
temporal_path = os.path.join(results_dir, "masking", "Temporal_mask_AAL.nii")
insular_path = os.path.join(results_dir, "masking", "Hypocortical_mask_AAL.nii")
cingulum_path = os.path.join(results_dir, "masking", "Cingulum_mask_AAL.nii")
cerebellum_path = os.path.join(results_dir, "masking", "Cerebellum_mask_AAL.nii")
GM_path = os.path.join(results_dir, "masking", "GM_mask_AAL.nii")

WM_path = os.path.join(results_dir, "masking", "WM_mask.nii")

ROI_mask_paths = {
    "Frontal": frontal_path, 
    "Occipital": occipital_path, 
    "Parietal": parietal_path, 
    "Temporal": temporal_path, 
    "Insular": insular_path, 
    "Cingulum": cingulum_path, 
    "Cerebellum": cerebellum_path, 
    "WM": WM_path,
    "GM": GM_path,
    "Participants mask": participants_mask_path
    }



hyp = 1

# Load NARPS 
resampled_maps_per_team = numpy.load(os.path.join(data_path, "Hyp{}_resampled_maps.npy".format(hyp)), allow_pickle=True).item()
masker_GM_WM = NiftiMasker(
    mask_img=participants_mask)
masker_GM_WM.fit(resampled_maps_per_team.values())



MA_estimators_names = [
    "SDMA Stouffer",
    "Consensus \nSDMA Stouffer",
    "Consensus Average",
    "SDMA GLS",
    "Consensus SDMA GLS"
    ]

# storing results
outputs = {}

# reconstruct mask to make sure all voxels in the roi are in full brain as well
rebuilt_mask_GM = numpy.zeros(nib.load(ROI_mask_paths["Frontal"]).get_fdata().shape)
for roi_name in list(ROI_mask_paths.keys())[:-2]:
    roi_mask = nib.load(ROI_mask_paths[roi_name])
    if roi_name == "WM":
        rebuilt_mask_GM_WM = rebuilt_mask_GM + roi_mask.get_fdata()
        assert numpy.array_equal(numpy.unique(rebuilt_mask_GM_WM), [0, 1]), "rebuilt_mask_GM contains elements other than 0 or 1 : {}".format(numpy.unique(rebuilt_mask_GM_WM))

    else: 
        rebuilt_mask_GM += roi_mask.get_fdata()
    assert numpy.array_equal(numpy.unique(rebuilt_mask_GM), [0, 1]), "rebuilt_mask_GM contains elements other than 0 or 1 : {}".format(numpy.unique(rebuilt_mask_GM))

    

# for each ROI, compute SDMA analysis 
for roi_name in list(ROI_mask_paths.keys()):
    print("Starting Segmented analysis for: {}".format(roi_name))
    # storing data
    outputs[roi_name] = {}
    # get data within a ROI
    masker_roi = NiftiMasker(
                mask_img=nib.load(ROI_mask_paths[roi_name]))
    resampled_maps_in_ROI = masker_roi.fit_transform(resampled_maps_per_team.values())
    print("Compute MA estimates in {}".format(roi_name))
    MA_outputs = compute_MA_outputs.get_MA_outputs(resampled_maps_in_ROI)

    print("Saving results per SDMA")
    for row, SDMA_method in enumerate(MA_estimators_names):
        T_map = MA_outputs[SDMA_method]['T_map']
        T_brain = masker_roi.inverse_transform(T_map)
        if roi_name == "GM":
            # ensure all voxels of the brain are in the ROIs
            T_brain = nilearn.image.new_img_like(T_brain, T_brain.get_fdata()*rebuilt_mask_GM)
        elif roi_name == "Participants mask":
            # ensure all voxels of the brain are in the ROIs
            T_brain = nilearn.image.new_img_like(T_brain, T_brain.get_fdata()*rebuilt_mask_GM_WM)
        outputs[roi_name][SDMA_method] = [T_map, T_brain]



################################################################################
################ STEP 4 : DIFFERENCE B/W SEGMENTED ANALYSIS AND ORIGINAL #######
################################################################################

##################################################
# ASSEMBLING SEGMENTED RESULTS INTO ONE UNIQUE MAP
##################################################

def max_min_diff_per_roi(df, SDMA_method, image_of_differences, where):
    # saving absolute difference regionally
    # create new columns
    for roi_name in list(ROI_mask_paths.keys())[:-2]:
        roi_mask_nii = nib.load(ROI_mask_paths[roi_name])
        max_diff = numpy.round((image_of_differences.get_fdata()*roi_mask_nii.get_fdata()).max(), 2)
        min_diff = numpy.round((image_of_differences.get_fdata()*roi_mask_nii.get_fdata()).min(), 2)
        print(roi_name, "max diff=", max_diff, ", min diff=", min_diff)


SDMA_method = "SDMA Stouffer"
print("*** {} **** ".format(SDMA_method))
print("Extract T values data")
# get t_brain (t_values in 3d shape)
T_brain_Frontal = outputs["Frontal"][SDMA_method][1]
T_brain_Occipital = outputs["Occipital"][SDMA_method][1]
T_brain_Parietal = outputs["Parietal"][SDMA_method][1]
T_brain_Temporal = outputs["Temporal"][SDMA_method][1]
T_brain_Insular = outputs["Insular"][SDMA_method][1]
T_brain_Cingulum = outputs["Cingulum"][SDMA_method][1]
T_brain_Cerebellum = outputs["Cerebellum"][SDMA_method][1]
T_brain_WM = outputs["WM"][SDMA_method][1]


# rebuild full brain statistics from ROI segmented analysis
empty_3D = numpy.zeros(T_brain_Frontal.get_fdata().shape)
full_3D_GM = empty_3D + T_brain_Frontal.get_fdata()+ T_brain_Occipital.get_fdata()+ T_brain_Parietal.get_fdata()+ T_brain_Temporal.get_fdata()+ T_brain_Insular.get_fdata()+ T_brain_Cingulum.get_fdata()+ T_brain_Cerebellum.get_fdata()
full_3D_GM_WM = full_3D_GM + T_brain_WM.get_fdata()
# TO DO: make sure there is only 0 and 1 here
rebuilt_GM = nilearn.image.new_img_like(T_brain_Frontal, full_3D_GM)
rebuilt_GM_WM = nilearn.image.new_img_like(T_brain_Frontal, full_3D_GM_WM)

roi_name = 'Frontal'
# DIFFERENCE with GM
T_brain_GM = outputs["GM"][SDMA_method][1]

# "*rebuilt_mask_GM" to ensure all voxels from the brain are in the ROIs
differences_GM = empty_3D + rebuilt_GM.get_fdata() - T_brain_GM.get_fdata()*rebuilt_mask_GM
diff_GM_image = nilearn.image.new_img_like(T_brain_Frontal, differences_GM)
df=0
max_min_diff_per_roi(df, SDMA_method, diff_GM_image, "GM")

# DIFFERENCE with GM & WM
T_brain_GM_WM = outputs["Participants mask"][SDMA_method][1]
# "*rebuilt_mask_GM_WM" to ensure all voxels from the brain are in the ROIs
differences_GM_WM = empty_3D + rebuilt_GM_WM.get_fdata() - T_brain_GM_WM.get_fdata()*rebuilt_mask_GM_WM
diff_GM_WM_image = nilearn.image.new_img_like(T_brain_Frontal, differences_GM_WM)
max_min_diff_per_roi(df, SDMA_method, diff_GM_WM_image, "GM & WM")


image_of_differences = diff_GM_WM_image 
for roi_name in list(ROI_mask_paths.keys())[:-2]:
    roi_mask_nii = nib.load(ROI_mask_paths[roi_name])
    max_diff = numpy.round((image_of_differences.get_fdata()*roi_mask_nii.get_fdata()).max(), 2)
    min_diff = numpy.round((image_of_differences.get_fdata()*roi_mask_nii.get_fdata()).min(), 2)
    print(roi_name, "max diff=", max_diff, ", min diff=", min_diff)

for i, roi_name in enumerate(list(ROI_mask_paths.keys())[:-1]):
    masker_roi = NiftiMasker(
                mask_img=nib.load(ROI_mask_paths[roi_name]))
    masker_roi.fit(resampled_maps_per_team.values()) # fit on whole brain
    ROI_stats_in_ROI = masker_roi.transform(outputs[roi_name][SDMA_method][1])
    BRAIN_stats_in_ROI = masker_roi.transform(outputs["Participants mask"][SDMA_method][1])
    diff = ROI_stats_in_ROI - BRAIN_stats_in_ROI
    print(roi_name)
    print(diff.max())
    print(diff.min())





