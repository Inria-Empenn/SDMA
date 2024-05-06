# Check how fair is the assumption of same Q accross the brain for all hypotheses
import os
import glob
import numpy
import nibabel
import nilearn.plotting
import nilearn.input_data
from nilearn import masking
from nilearn import image
import warnings
import matplotlib.pyplot as plt
from nilearn.datasets import load_mni152_brain_mask
import seaborn
import importlib
from community import community_louvain
import networkx as nx
import utils
from nilearn.input_data import NiftiMasker
import pandas
import compute_MA_outputs
import narps_visualisation
from nilearn import plotting

importlib.reload(utils) # reupdate imported codes, useful for debugging


##################
# Create ROI masks using atlas Harvard Oxford
##################
from nilearn import datasets
from nilearn import image
import nibabel as nib
import math


atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
labels = atlas.labels

frontal_pole_idx = [1, 3, 4, 5, 6, 25, 27, 28, 33, 41]
occipital_pole_idx = [22, 23, 24, 31, 32, 36, 48]
parietal_pole_idx = [7, 17, 18, 19, 20 ,21, 26, 29, 30, 42, 43, 44, 45, 46]
temporal_pole_idx = [8, 9, 10, 11, 12, 13, 14 ,15, 16, 34, 35, 37, 38,39, 40]
insular_pole_idx = [2]

def nii_from_idx(atlas, idxs):
    mask = atlas.maps.get_fdata().copy()
    roi = numpy.zeros(mask.shape)
    for idx in idxs:
        roi[mask == idx ] = 1
    roi_nifti = nib.Nifti1Image(roi, affine=atlas.maps.affine)
    roi_nifti_mni = image.resample_to_img(
                        roi_nifti,
                        load_mni152_brain_mask(),
                        interpolation='nearest')
    return roi_nifti_mni


frontal_pole_nii = nii_from_idx(atlas, frontal_pole_idx)
occipital_pole_nii = nii_from_idx(atlas, occipital_pole_idx)
parietal_pole_nii = nii_from_idx(atlas, parietal_pole_idx)
temporal_pole_nii = nii_from_idx(atlas,temporal_pole_idx)
insular_pole_nii = nii_from_idx(atlas, insular_pole_idx)
whole_brain_nii = nii_from_idx(atlas, range(1,49))
mask_participant = nibabel.load('masking/mask_90.nii')

roi_masks = {"Frontal":frontal_pole_nii, "Occipital":occipital_pole_nii, "Parietal":parietal_pole_nii, 
    "Temporal":temporal_pole_nii , "Insular":insular_pole_nii, "Atlas Harvard-Oxford":whole_brain_nii, "Mask participant":mask_participant}

# ##################
# # COMPUTE FROBENIUS IN NARPS
# ##################


# f, axs = plt.subplots(5, 3, figsize=(25, 15))  
# coords = (20, 0, -23)
# nilearn.plotting.plot_roi(frontal_pole_nii, title='Frontal', axes=axs[0, 0], cut_coords=coords)
# nilearn.plotting.plot_roi(occipital_pole_nii, title='Occipital', axes=axs[1, 0], cut_coords=coords)
# nilearn.plotting.plot_roi(parietal_pole_nii, title='parietal', axes=axs[2, 0], cut_coords=coords)
# nilearn.plotting.plot_roi(temporal_pole_nii, title='temporal', axes=axs[3, 0], cut_coords=coords)
# nilearn.plotting.plot_roi(insular_pole_nii, title='insular', axes=axs[4, 0], cut_coords=coords)

# masks = [mask_participant, frontal_pole_nii]
# frontal_pole_nii_masked = masking.intersect_masks(masks, threshold=1, connected=False)
# nilearn.plotting.plot_roi(frontal_pole_nii_masked, title='Frontal_merged',axes=axs[0, 1], cut_coords=coords)

# masks = [mask_participant, occipital_pole_nii]
# occipital_pole_nii_masked = masking.intersect_masks(masks, threshold=1, connected=False)
# nilearn.plotting.plot_roi(occipital_pole_nii_masked, title='Occipital_merged', axes=axs[1, 1], cut_coords=coords)

# masks = [mask_participant, parietal_pole_nii]
# parietal_pole_nii_masked = masking.intersect_masks(masks, threshold=1, connected=False)
# nilearn.plotting.plot_roi(parietal_pole_nii_masked, title='Parietal_merged', axes=axs[2, 1], cut_coords=coords)

# masks = [mask_participant,temporal_pole_nii]
# temporal_pole_nii_masked = masking.intersect_masks(masks, threshold=1, connected=False)
# nilearn.plotting.plot_roi(temporal_pole_nii_masked, title='Temporal_merged', axes=axs[3, 1], cut_coords=coords)

# masks = [mask_participant,insular_pole_nii]
# insular_pole_nii_masked = masking.intersect_masks(masks, threshold=1, connected=False)
# nilearn.plotting.plot_roi(insular_pole_nii_masked, title='Insular_merged', axes=axs[4, 1], cut_coords=coords)

# nilearn.plotting.plot_roi(mask_participant, title='Mask participant', axes=axs[0, 2], cut_coords=coords)
# nilearn.plotting.plot_roi(atlas.maps, title='Atlas Harvard-Oxford', axes=axs[1, 2], cut_coords=coords)


# masks = [mask_participant, whole_brain_nii]
# whole_brain_nii_masked = masking.intersect_masks(masks, threshold=1, connected=False)
# nilearn.plotting.plot_roi(whole_brain_nii_masked, title='whole_brain_merged',axes=axs[2, 2], cut_coords=coords)

# axs[3, 2].axis('off')
# axs[4, 2].axis('off')
# plt.savefig("results_Q_assumptions/check_masking.pdf")
# plt.show()




# GET NARPS DATA hyp 1

hyp = 1
resampled_maps_per_team = numpy.load('results_in_Narps_data/data/Hyp{}_resampled_maps.npy'.format(hyp), allow_pickle=True).item() # mind result dir
masker = NiftiMasker(
    mask_img=mask_participant)
resampled_maps = masker.fit_transform(resampled_maps_per_team.values())


# # COmpute Frobenius
# def plot_frobenius_results(ROI_mask, label_ROI, coords=coords, mask_participant=mask_participant, mask_brain=whole_brain_nii_masked, resampled_maps=resampled_maps, resampled_maps_per_team=resampled_maps_per_team):
#     print("RUNNING ", label_ROI)
#     masker = NiftiMasker(
#                 mask_img=ROI_mask)
#     data = masker.fit_transform(resampled_maps_per_team.values())
#     Q = numpy.corrcoef(data)
#     df_Q = pandas.DataFrame(Q)
#     K = Q.shape[0]

#     masker_brain = NiftiMasker(
#             mask_img=mask_brain)
#     data_ref = masker_brain.fit_transform(resampled_maps_per_team.values())
#     Q_ref = numpy.corrcoef(data_ref)
#     df_Q_ref = pandas.DataFrame(Q_ref)
#     df_Q.to_excel("results_Q_assumptions/Q_frontal.xlsx")
#     df_Q_ref.to_excel("results_Q_assumptions/Q_brain.xlsx")
#     ones = numpy.ones((K, 1))
#     similarity_matrix = Q - Q_ref
#     print("min/max: ", similarity_matrix.min(), similarity_matrix.max())
#     rel_Q = numpy.ones(K).T.dot(Q).dot(numpy.ones(K))/K**2
#     print("***")
#     print(rel_Q)
#     rel_Q_ref = numpy.ones(K).T.dot(Q_ref).dot(numpy.ones(K))/K**2
#     print(rel_Q_ref)
#     print("***")
#     similarity_matrix_ratio = (rel_Q - rel_Q_ref)/rel_Q_ref*100
#     # Fro = numpy.linalg.norm(similarity_matrix, ord='fro')
#     Fro = math.sqrt(numpy.mean([elem**2 for row in similarity_matrix for elem in row]))

#     print(similarity_matrix_ratio, Fro)
#     return numpy.round(similarity_matrix_ratio, 2), numpy.round(Fro, 2)

# df = pandas.DataFrame(index=["Frontal", "Occipital", "Parietal", "Temporal", "Insular"], columns=["Qsi", "Fro (Q-Q_ref)"])

# df.loc["Frontal"] = plot_frobenius_results(frontal_pole_nii_masked, "Frontal")
# df.loc["Occipital"] = plot_frobenius_results(occipital_pole_nii_masked, "occipital")
# df.loc["Parietal"] = plot_frobenius_results(parietal_pole_nii_masked, "parietal")
# df.loc["Temporal"] = plot_frobenius_results(temporal_pole_nii_masked, "temporal")
# df.loc["Insular"] = plot_frobenius_results(insular_pole_nii_masked, "insular")

# df.to_excel("results_Q_assumptions/frobenius_scores_Narps_hyp{}.xlsx".format(hyp))

# ##############
# final figure
###############

# plt.close('all')
# f, axs = plt.subplots(3, 6, figsize=(30, 15)) 
# nilearn.plotting.plot_roi(mask_participant, axes=axs[0, 0], cut_coords=coords)
# axs[0, 0].set_title("Whole brain",fontsize=20)
# nilearn.plotting.plot_roi(frontal_pole_nii_masked, axes=axs[0, 1], cut_coords=coords)
# axs[0, 1].set_title("Frontal",fontsize=20)
# nilearn.plotting.plot_roi(occipital_pole_nii_masked, axes=axs[0, 2], cut_coords=coords)
# axs[0, 2].set_title("Occipital",fontsize=20)
# nilearn.plotting.plot_roi(parietal_pole_nii_masked, axes=axs[0, 3], cut_coords=coords)
# axs[0, 3].set_title("Parietal",fontsize=20)
# nilearn.plotting.plot_roi(temporal_pole_nii_masked, axes=axs[0, 4], cut_coords=coords)
# axs[0, 4].set_title("Temporal",fontsize=20)
# nilearn.plotting.plot_roi(insular_pole_nii_masked, axes=axs[0, 5], cut_coords=coords)
# axs[0, 5].set_title("Insular",fontsize=20)

# Q_ref = numpy.corrcoef(resampled_maps)
# axs[1, 0].set_title("Correlation matrix")
# seaborn.heatmap(Q_ref, center=0, cmap='coolwarm', robust=True, square=True, ax=axs[1, 0], cbar_kws={'shrink': 0.6})
# axs[2, 0].axis('off')

# def plot_Q_roi(ROI_mask, x, y, resampled_maps_per_team=resampled_maps_per_team, axs=axs, Q_ref=Q_ref):
#     print('runnin ', y)
#     masker = NiftiMasker(
#                 mask_img=ROI_mask)
#     data = masker.fit_transform(resampled_maps_per_team.values())
#     Q = numpy.corrcoef(data)
#     similarity_matrix = Q - Q_ref
#     Fro = numpy.linalg.norm(similarity_matrix, ord='fro')
#     seaborn.heatmap(Q, center=0, cmap='coolwarm', robust=True, square=True, ax=axs[x, y], cbar_kws={'shrink': 0.6})
#     axs[x, y].set_title("Correlation matrix")
#     seaborn.heatmap(similarity_matrix, center=0, cmap='coolwarm', robust=True, vmax=1, square=True, ax=axs[x+1, y], cbar_kws={'shrink': 0.6})
#     axs[x+1, y].set_title("Similarity matrix (Fro_norm={})".format(numpy.round(Fro, 2)))


# plot_Q_roi(frontal_pole_nii_masked, 1, 1)
# plot_Q_roi(occipital_pole_nii_masked, 1, 2)
# plot_Q_roi(parietal_pole_nii_masked, 1, 3)
# plot_Q_roi(temporal_pole_nii_masked, 1, 4)
# plot_Q_roi(insular_pole_nii_masked, 1, 5)

# plt.suptitle("NARPS hyp {}".format(hyp), size=25)
# plt.savefig("results_Q_assumptions/Figure_NARPS_hyp{}.pdf".format(hyp))
# plt.close('all')


##################################
# running SDMA in brain regions:
###################################
# SEGMENTED ANALYSIS

MA_estimators_names = [
    "SDMA Stouffer",
    "Consensus \nSDMA Stouffer",
    "Consensus Average",
    "SDMA GLS",
    "Consensus SDMA GLS"
    ]


hyps = [1, 2, 5, 6, 7, 8, 9]#numpy.arange(1, 10, 1)
for hyp in hyps:
    print('*****Running hyp ', hyp, '*****')
    resampled_maps_per_team = numpy.load('results_in_Narps_data/data/Hyp{}_resampled_maps.npy'.format(hyp), allow_pickle=True).item() # mind result dir
    masker = NiftiMasker(
        mask_img=mask_participant)
    resampled_maps = masker.fit_transform(resampled_maps_per_team.values())

    if not os.path.exists("results_Q_assumptions/hyp{}".format(hyp)):
        os.mkdir("results_Q_assumptions/hyp{}".format(hyp))
    results_dir_hyp = "results_Q_assumptions/hyp{}".format(hyp)

    Q_ref = numpy.corrcoef(resampled_maps)
    K = Q_ref.shape[0]
    ones = numpy.ones((K, 1))
    rel_Q_ref = numpy.ones(K).T.dot(Q_ref).dot(numpy.ones(K))/K**2

    for roi_name in roi_masks.keys():
        ROI_mask = roi_masks[roi_name]
        masker = NiftiMasker(
                    mask_img=ROI_mask)
        print("Starting Masking...")
        resampled_maps_roi = masker.fit_transform(resampled_maps_per_team.values())
        print("Z values extracted, shape=", resampled_maps_roi.shape) # 61, 1537403 for hyp 1
        print("Should be less than (61, 1537403)")
        print("compute Frobenius")
        Qi = numpy.corrcoef(resampled_maps_roi)
        similarity_matrix = Qi - Q_ref
        rel_Q = numpy.ones(K).T.dot(Qi).dot(numpy.ones(K))/K**2
        similarity_matrix_ratio = (rel_Q - rel_Q_ref)/rel_Q_ref*100
        Fro_normalized = math.sqrt(numpy.mean([elem**2 for row in similarity_matrix for elem in row]))    

        print("Compute MA estimates")
        MA_outputs = compute_MA_outputs.get_MA_outputs(resampled_maps_roi)
        print("Building figure")

        plt.close('all')
        f, axs = plt.subplots(len(MA_estimators_names) + 1, 1, figsize=(8, (len(MA_estimators_names)+1)*1.6))
        for row, title in enumerate(MA_estimators_names):
            T_map=MA_outputs[title]['T_map']
            p_values=MA_outputs[title]['p_values']
            # compute ratio of significant p-values
            ratio_significance = MA_outputs[title]['ratio_significance']
            # raw p values
            perc_sign_voxels = numpy.round(numpy.sum(p_values<=0.05)*100/len(p_values), 4)
            # back to 3D
            # p_brain_sign = masker.inverse_transform(p_stat)
            p_brain = masker.inverse_transform(p_values)
            t_brain = masker.inverse_transform(T_map)

            # apply threshold
            pdata = p_brain.get_fdata()
            tdata = t_brain.get_fdata()
            threshdata = (pdata <= 0.05)*tdata #0.05 is threshold significance
            threshimg = nibabel.Nifti1Image(threshdata, affine=t_brain.affine)
            long_title = title + ', {}%'.format(numpy.round(perc_sign_voxels, 2))
            if "\n" in long_title:
                long_title = long_title.replace('\n', '')
            plotting.plot_stat_map(threshimg, annotate=False, threshold=0.1,vmax=8, colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52), display_mode='z', cmap='Reds', axes=axs[row])
            axs[row].set_title(long_title)
        nilearn.plotting.plot_roi(ROI_mask, axes=axs[5], annotate=False, cut_coords=(-24, -10, 4, 18, 32, 52), display_mode='z')
        axs[5].set_title("Mask ROI {}".format(roi_name),fontsize=12)
        plt.suptitle('Hypothesis {}, ROI {}, Frob:{} ({}%)'.format(hyp, roi_name, numpy.round(Fro_normalized, 2), numpy.round(similarity_matrix_ratio, 2)))
        plt.savefig("{}/thresholded_map_HYP{}_ROI_{}.png".format(results_dir_hyp, hyp, roi_name))
        plt.close('all')
          














##################
# COMPUTE FROBENIUS IN HCP
##################
data_path = "/home/jlefortb/SDMA/hcp_data/preprocessed"
mask_participant = os.path.join(data_path, "mask.nii.gz")

atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
labels = atlas.labels

frontal_pole_idx = [1, 3, 4, 5, 6, 25, 27, 28, 33, 41]
occipital_pole_idx = [22, 23, 24, 31, 32, 36, 48]
parietal_pole_idx = [7, 17, 18, 19, 20 ,21, 26, 29, 30, 42, 43, 44, 45, 46]
temporal_pole_idx = [8, 9, 10, 11, 12, 13, 14 ,15, 16, 34, 35, 37, 38,39, 40]
insular_pole_idx = [2]

def nii_from_idx(atlas, idxs):
    mask = atlas.maps.get_fdata().copy()
    roi = numpy.zeros(mask.shape)
    for idx in idxs:
        roi[mask == idx ] = 1
    roi_nifti = nib.Nifti1Image(roi, affine=atlas.maps.affine)
    roi_nifti_mni = image.resample_to_img(
                        roi_nifti,
                        mask_participant,
                        interpolation='nearest')
    return roi_nifti_mni

frontal_pole_nii = nii_from_idx(atlas, frontal_pole_idx)
occipital_pole_nii = nii_from_idx(atlas, occipital_pole_idx)
parietal_pole_nii = nii_from_idx(atlas, parietal_pole_idx)
temporal_pole_nii = nii_from_idx(atlas,temporal_pole_idx)
insular_pole_nii = nii_from_idx(atlas, insular_pole_idx)

f, axs = plt.subplots(5, 3, figsize=(25, 15))  
coords = (20, 0, -23)
nilearn.plotting.plot_roi(frontal_pole_nii, title='Frontal', axes=axs[0, 0], cut_coords=coords)
nilearn.plotting.plot_roi(occipital_pole_nii, title='Occipital', axes=axs[1, 0], cut_coords=coords)
nilearn.plotting.plot_roi(parietal_pole_nii, title='parietal', axes=axs[2, 0], cut_coords=coords)
nilearn.plotting.plot_roi(temporal_pole_nii, title='temporal', axes=axs[3, 0], cut_coords=coords)
nilearn.plotting.plot_roi(insular_pole_nii, title='insular', axes=axs[4, 0], cut_coords=coords)

masks = [mask_participant, frontal_pole_nii]
frontal_pole_nii_masked = masking.intersect_masks(masks, threshold=1, connected=False)
nilearn.plotting.plot_roi(frontal_pole_nii_masked, title='Frontal_merged',axes=axs[0, 1], cut_coords=coords)

masks = [mask_participant, occipital_pole_nii]
occipital_pole_nii_masked = masking.intersect_masks(masks, threshold=1, connected=False)
nilearn.plotting.plot_roi(occipital_pole_nii_masked, title='Occipital_merged', axes=axs[1, 1], cut_coords=coords)

masks = [mask_participant, parietal_pole_nii]
parietal_pole_nii_masked = masking.intersect_masks(masks, threshold=1, connected=False)
nilearn.plotting.plot_roi(parietal_pole_nii_masked, title='Parietal_merged', axes=axs[2, 1], cut_coords=coords)

masks = [mask_participant,temporal_pole_nii]
temporal_pole_nii_masked = masking.intersect_masks(masks, threshold=1, connected=False)
nilearn.plotting.plot_roi(temporal_pole_nii_masked, title='Temporal_merged', axes=axs[3, 1], cut_coords=coords)

masks = [mask_participant,insular_pole_nii]
insular_pole_nii_masked = masking.intersect_masks(masks, threshold=1, connected=False)
nilearn.plotting.plot_roi(insular_pole_nii_masked, title='Insular_merged', axes=axs[4, 1], cut_coords=coords)

nilearn.plotting.plot_roi(mask_participant, title='Mask participant', axes=axs[0, 2], cut_coords=coords)
axs[1, 2].axis('off')
axs[2, 2].axis('off')
axs[3, 2].axis('off')
axs[4, 2].axis('off')
plt.savefig("results_Q_assumptions/check_masking_HCP.pdf")
plt.show()

# GET HCP DATA
masker = NiftiMasker(
    mask_img=mask_participant)
results_dir = "results_in_HCP_data"
resampled_maps = numpy.load('{}/data/resampled_maps.npy'.format(results_dir), allow_pickle=True) # mind result dir
team_names = numpy.load('{}/data/pipeline_names.npy'.format(results_dir), allow_pickle=True) # mind result dir
print("resampled_maps successfully loaded")
# fit masker
masker.fit(resampled_maps)
resampled_maps_per_team = masker.inverse_transform(resampled_maps)

# COmpute Frobenius
def plot_frobenius_results(ROI_mask, label_ROI, coords=coords, mask_participant=mask_participant, mask_brain=whole_brain_nii_masked, resampled_maps=resampled_maps, resampled_maps_per_team=resampled_maps_per_team):
    print("RUNNING ", label_ROI)
    masker = NiftiMasker(
                mask_img=ROI_mask)
    data = masker.fit_transform(resampled_maps_per_team.values())
    Q = numpy.corrcoef(data)
    df_Q = pandas.DataFrame(Q)
    K = Q.shape[0]

    masker_brain = NiftiMasker(
            mask_img=mask_brain)
    data_ref = masker_brain.fit_transform(resampled_maps_per_team.values())
    Q_ref = numpy.corrcoef(data_ref)
    df_Q_ref = pandas.DataFrame(Q_ref)
    df_Q.to_excel("results_Q_assumptions/Q_frontal.xlsx")
    df_Q_ref.to_excel("results_Q_assumptions/Q_brain.xlsx")
    ones = numpy.ones((K, 1))
    similarity_matrix = Q - Q_ref
    print("min/max: ", similarity_matrix.min(), similarity_matrix.max())
    rel_Q = numpy.ones(K).T.dot(Q).dot(numpy.ones(K))/K**2
    print("***")
    print(rel_Q)
    rel_Q_ref = numpy.ones(K).T.dot(Q_ref).dot(numpy.ones(K))/K**2
    print(rel_Q_ref)
    print("***")
    similarity_matrix_ratio = (rel_Q - rel_Q_ref)/rel_Q_ref*100
    # Fro = numpy.linalg.norm(similarity_matrix, ord='fro')
    Fro = math.sqrt(numpy.mean([elem**2 for row in similarity_matrix for elem in row]))

    print(similarity_matrix_ratio, Fro)
    return numpy.round(similarity_matrix_ratio, 2), numpy.round(Fro, 2)






df = pandas.DataFrame(index=["Frontal", "Occipital", "Parietal", "Temporal", "Insular"], columns=["Fro", "Fro_by2", "Fro_perc"])

df.loc["Frontal"] = plot_frobenius_results(frontal_pole_nii_masked, "Frontal")
df.loc["Occipital"] = plot_frobenius_results(occipital_pole_nii_masked, "occipital")
df.loc["Parietal"] = plot_frobenius_results(parietal_pole_nii_masked, "parietal")
df.loc["Temporal"] = plot_frobenius_results(temporal_pole_nii_masked, "temporal")
df.loc["Insular"] = plot_frobenius_results(insular_pole_nii_masked, "insular")

df.to_excel("results_Q_assumptions/frobenius_scores_HCP.xlsx")



##############
# final figure
###############

plt.close('all')
f, axs = plt.subplots(3, 6, figsize=(30, 15)) 
nilearn.plotting.plot_roi(mask_participant, axes=axs[0, 0], cut_coords=coords)
axs[0, 0].set_title("Whole brain",fontsize=20)
nilearn.plotting.plot_roi(frontal_pole_nii_masked, axes=axs[0, 1], cut_coords=coords)
axs[0, 1].set_title("Frontal",fontsize=20)
nilearn.plotting.plot_roi(occipital_pole_nii_masked, axes=axs[0, 2], cut_coords=coords)
axs[0, 2].set_title("Occipital",fontsize=20)
nilearn.plotting.plot_roi(parietal_pole_nii_masked, axes=axs[0, 3], cut_coords=coords)
axs[0, 3].set_title("Parietal",fontsize=20)
nilearn.plotting.plot_roi(temporal_pole_nii_masked, axes=axs[0, 4], cut_coords=coords)
axs[0, 4].set_title("Temporal",fontsize=20)
nilearn.plotting.plot_roi(insular_pole_nii_masked, axes=axs[0, 5], cut_coords=coords)
axs[0, 5].set_title("Insular",fontsize=20)

Q_ref = numpy.corrcoef(resampled_maps)
axs[1, 0].title.set_text("Correlation matrix")
seaborn.heatmap(Q_ref, center=0, cmap='coolwarm', robust=True, square=True, ax=axs[1, 0], cbar_kws={'shrink': 0.6})
axs[2, 0].axis('off')

def plot_Q_roi(ROI_mask, x, y, resampled_maps_per_team=resampled_maps_per_team, axs=axs, Q_ref=Q_ref):
    print('runnin ', y)
    masker = NiftiMasker(
                mask_img=ROI_mask)
    data = masker.fit_transform(resampled_maps_per_team)
    Q = numpy.corrcoef(data)
    similarity_matrix = Q - Q_ref
    Fro = numpy.linalg.norm(similarity_matrix, ord='fro')
    seaborn.heatmap(Q, center=0, cmap='coolwarm', robust=True, square=True, ax=axs[x, y], cbar_kws={'shrink': 0.6})
    axs[x, y].title.set_text("Correlation matrix")
    seaborn.heatmap(similarity_matrix, center=0, cmap='coolwarm', robust=True, square=True, ax=axs[x+1, y], cbar_kws={'shrink': 0.6})
    axs[x+1, y].title.set_text("Similarity matrix (Fro_norm={})".format(numpy.round(Fro, 2)))


plot_Q_roi(frontal_pole_nii_masked, 1, 1)
plot_Q_roi(occipital_pole_nii_masked, 1, 2)
plot_Q_roi(parietal_pole_nii_masked, 1, 3)
plot_Q_roi(temporal_pole_nii_masked, 1, 4)
plot_Q_roi(insular_pole_nii_masked, 1, 5)

plt.suptitle("HCP", size=25)
# plt.tight_layout()
plt.savefig("results_Q_assumptions/Figure_HCP.pdf")
plt.close('all')




