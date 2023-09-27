from nilearn import plotting
import matplotlib.pyplot as plt
import numpy
import nibabel
from statsmodels.stats.multitest import multipletests
from nilearn import masking

### VISUALIZATION NUMBERS
def plot_distributions(MA_outputs, hyp, MA_estimators_names, results_dir):
    plt.close('all')
    f, axs = plt.subplots(3, len(MA_estimators_names), figsize=(len(MA_estimators_names)*15/5, 5)) 
    for row in range(3):
        for col, title in enumerate(MA_estimators_names):
            T_map=MA_outputs[title]['T_map']
            p_values=MA_outputs[title]['p_values']
            # compute ratio of significant p-values
            ratio_significance = MA_outputs[title]['ratio_significance']

            # supplementary outcome for comparing with narps paper results
            fdr_results = multipletests(p_values, 0.05, 'fdr_tsbh')
            fdr_p_values = fdr_results[1]
            # compute ratio of significant p-values after fdr
            ratio_significance_raw_fdr = (fdr_p_values<=0.05).sum()/len(fdr_p_values)
            ratio_significance_fdr = numpy.round(ratio_significance_raw_fdr*100, 4)
            if row == 0:
                #t values
                axs[row, col].title.set_text(title)
                axs[row, col].hist(T_map, bins=200, color='green')
                if col ==0:
                    axs[row, col].set_ylabel("frequency")
                else:
                    axs[row, col].set(yticklabels=[])

                axs[row, col].set_xlabel("T value")

            elif row == 1:
                #pvalues
                axs[row, col].hist(p_values, bins=500, color='y')
                if col ==0:
                    axs[row, col].set_ylabel("frequency")
                else:
                    axs[row, col].set(yticklabels=[])
                axs[row, col].set_xlabel("p value")
                axs[row, col].axvline(0.05, ymin=0, color='black', linewidth=0.5, linestyle='--')
                axs[row, col].set_ylim(0, 12000)
                axs[row, col].text(0.10, 10000, 'ratio={}%'.format(ratio_significance))
            else:
                #fdr corrected p values
                axs[row, col].hist(fdr_p_values, bins=500, color='y')
                if col ==0:
                    axs[row, col].set_ylabel("frequency")
                else:
                    axs[row, col].set(yticklabels=[])
                axs[row, col].set_xlabel("p value (fdr)")
                axs[row, col].axvline(0.05, ymin=0, color='black', linewidth=0.5, linestyle='--')
                axs[row, col].set_ylim(0, 5000)
                axs[row, col].text(0.10, 3000, 'ratio (fdr) ={}%'.format(ratio_significance_fdr))

    plt.suptitle('Hypothesis {}'.format(hyp))
    plt.tight_layout()
    plt.savefig("{}/hyp{}_MA_outputs.png".format(results_dir, hyp))
    plt.close('all')


### VISUALIZATION BRAIN
def plot_brains(MA_outputs, hyp, MA_estimators_names, results_dir, masker):
    plt.close('all')
    f, axs = plt.subplots(len(MA_estimators_names), 2, figsize=(16, len(MA_estimators_names)*8/5))
    for row, title in enumerate(MA_estimators_names):
        T_map=MA_outputs[title]['T_map']
        p_values=MA_outputs[title]['p_values']
        # compute ratio of significant p-values
        ratio_significance = MA_outputs[title]['ratio_significance']

        if row==0:
            axs[row, 0].set_title("P values")
            axs[row, 1].set_title("P values FDR corrected")
        # raw p values
        perc_sign_voxels = numpy.round(numpy.sum(p_values<=0.05)*100/len(p_values), 4)
        assert ratio_significance == perc_sign_voxels

        # with fdr 
        fdr_results = multipletests(p_values, 0.05, 'fdr_tsbh')
        perc_sign_voxels_fdr = numpy.round(numpy.sum(fdr_results[0])*100/len(fdr_results[0]), 4)
        fdr_p_values = 1 - fdr_results[1]

        # back to 3D
        # p_brain_sign = masker.inverse_transform(p_stat)
        p_brain = masker.inverse_transform(p_values)
        p_brain_fdr = masker.inverse_transform(fdr_p_values)
        t_brain = masker.inverse_transform(T_map)

        # apply threshold
        pdata = p_brain.get_fdata()
        pdata_fdr = p_brain_fdr.get_fdata()
        tdata = t_brain.get_fdata()
        threshdata_fdr = (pdata_fdr > 0.95)*tdata #0.95 is threshold significance
        threshdata = (pdata <= 0.05)*tdata #0.05 is threshold significance
        threshimg_fdr = nibabel.Nifti1Image(threshdata_fdr, affine=t_brain.affine)
        threshimg = nibabel.Nifti1Image(threshdata, affine=t_brain.affine)

        long_title_fdr = title + ', {}%'.format(perc_sign_voxels_fdr)
        long_title = title + ', {}%'.format(perc_sign_voxels)

        # plotting.plot_stat_map(t_brain, title=long_title, colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap="coolwarm", ax=axs[row, 0])
        # plt.savefig("results_narpsdata/{}_hyp{}_t_map.png".format(title, hyp))
        # plt.close('all')
        # plotting.plot_stat_map(p_brain, threshold=0.1, title=long_title, colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap="Reds")
        # plt.savefig("results_narpsdata/{}_hyp{}_p_map.png".format(title, hyp))
        # plt.close('all')
        plotting.plot_stat_map(threshimg_fdr, annotate=False, title=long_title_fdr, threshold=0.1,vmax=8, colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap='Reds', axes=axs[row, 1])
        plotting.plot_stat_map(threshimg, annotate=False, title=long_title, threshold=0.1,vmax=8, colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap='Reds', axes=axs[row, 0])
        
        # plt.savefig("results_narpsdata/{}_hyp{}_p_sign_map.png".format(title, hyp))
        # plt.close('all')
    plt.suptitle('Hypothesis {}'.format(hyp))
    plt.savefig("{}/thresholded_map_hyp{}.png".format(results_dir, hyp))
    plt.close('all')

### VISUALIZATION BRAIN
def plot_brain_nofdr(MA_outputs, hyp, MA_estimators_names, results_dir, masker):
    plt.close('all')
    f, axs = plt.subplots(len(MA_estimators_names[1:]), 1, figsize=(8, len(MA_estimators_names[1:])*8/5))
    for row, title in enumerate(MA_estimators_names[1:]):
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
        long_title = title + ', {}%'.format(perc_sign_voxels)
        if "\n" in long_title:
            long_title = long_title.replace('\n', '')
        # plotting.plot_stat_map(t_brain, title=long_title, colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap="coolwarm", ax=axs[row, 0])
        # plt.savefig("results_narpsdata/{}_hyp{}_t_map.png".format(title, hyp))
        # plt.close('all')
        # plotting.plot_stat_map(p_brain, threshold=0.1, title=long_title, colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap="Reds")
        # plt.savefig("results_narpsdata/{}_hyp{}_p_map.png".format(title, hyp))
        # plt.close('all')
        plotting.plot_stat_map(threshimg, annotate=False, threshold=0.1,vmax=8, colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap='Reds', axes=axs[row])
        axs[row].set_title(long_title)
        
        # plt.savefig("results_narpsdata/{}_hyp{}_p_sign_map.png".format(title, hyp))
        # plt.close('all')
    plt.suptitle('Hypothesis {}'.format(hyp))
    plt.savefig("{}/thresholded_map_hyp{}_nofdr.png".format(results_dir, hyp))
    plt.close('all')


def plot_SDMA_results_divergence(MA_outputs, hyp, MA_estimators_names, results_dir, masker):
    threshimgs = []
    for row, title in enumerate(MA_estimators_names[2:]): # only SDMA
        p_values_raw =MA_outputs[title]['p_values']
        p_values = p_values_raw.copy()
        p_values[p_values <= 0.05] = -1 # -1 to avoid including p values of value 1 as significant
        p_values[p_values != -1] = 0
        p_values[p_values == -1] = 1
        # back to 3D
        threshimg = masker.inverse_transform(p_values)
        threshimgs.append(threshimg)
        # # debugging
        # print(p_values.sum()/p_values.__len__()*100)
    # plot similarities
    similarities_mask = masking.intersect_masks(threshimgs, threshold=1, connected=True)
    plt.close('all')
    plotting.plot_stat_map(similarities_mask, annotate=False, vmax=1, colorbar=False, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap='Greens')
    plt.suptitle("Similarities between SDMA models")
    plt.savefig("{}/similarity_maps_hyp{}.png".format(results_dir, hyp))
    plt.close('all')
    # plot contrast
    contrasts_mask = masking.intersect_masks(threshimgs, threshold=0, connected=True)
    plotting.plot_stat_map(contrasts_mask, annotate=False, vmax=1, colorbar=False, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap='Blues')
    plt.suptitle("Contrasts between SDMA models")
    plt.savefig("{}/contrast_maps_hyp{}.png".format(results_dir, hyp))
    plt.close('all')
    return similarities_mask

def plot_hyp_similarities(similarity_mask_per_hyp, results_dir):
    similarities_mask = masking.intersect_masks(similarity_mask_per_hyp, threshold=1, connected=True)
    plt.close('all')
    plotting.plot_stat_map(similarities_mask, annotate=False, vmax=1, colorbar=False, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap='Oranges')
    plt.suptitle("Similarities between hypothesis and models")
    plt.savefig("{}/similarity_maps_all_hyp.png".format(results_dir))
    plt.close('all')


# from nilearn.datasets import load_mni152_brain_mask
# from nilearn.datasets import fetch_atlas_aal
# from nilearn import masking
# from nilearn import image

# atlas_aal = fetch_atlas_aal()
# hyp789 = [
#     'Amygdala_R',
#     'Amygdala_L'
# ]
# hyp1256= [
#     'Precentral_L',
#     'Precentral_R',
#     'Frontal_Sup_L',
#     'Frontal_Sup_R',
#     'Frontal_Sup_Orb_L',
#     'Frontal_Sup_Orb_R',
#     'Frontal_Mid_L',
#     'Frontal_Mid_R',
#     'Frontal_Mid_Orb_L',
#     'Frontal_Mid_Orb_R',
#     'Frontal_Inf_Oper_L',
#     'Frontal_Inf_Oper_R',
#     'Frontal_Inf_Tri_L',
#     'Frontal_Inf_Tri_R',
#     'Frontal_Inf_Orb_L',
#     'Frontal_Inf_Orb_R',
#     'Precentral_L',
#      'Frontal_Sup_Medial_L',
#     'Frontal_Sup_Medial_R',
#     'Frontal_Med_Orb_L',
#     'Frontal_Med_Orb_R',
# ]
# hyp34= [
#     'Caudate_L',
#     'Caudate_R',
#     'Putamen_L',
#     'Putamen_R'
# ]
# indices_hyp789 = [atlas_aal.indices[i] for i in [atlas_aal.labels.index(roi) for roi in hyp789]]
# indices_hyp1256 = [atlas_aal.indices[i] for i in [atlas_aal.labels.index(roi) for roi in hyp1256]]
# indices_hyp34 = [atlas_aal.indices[i] for i in [atlas_aal.labels.index(roi) for roi in hyp34]]
# atlas_aal_nii = nibabel.load(atlas_aal.maps)
# # resample MNI gm mask space
# atlas_aal_nii = image.resample_to_img(
#                     atlas_aal_nii,
#                     load_mni152_brain_mask(),
#                     interpolation='nearest')

# # function to save PNG of mask
# def compute_save_display_mask(ROI_name, indices, results_dir):
#     # compute ROI mask
#     indexes_ROI = [numpy.where(atlas_aal_nii.get_fdata() == int(indice)) for indice in indices]
#     fake_ROI = numpy.zeros(atlas_aal_nii.get_fdata().shape)
#     for indexes in indexes_ROI:
#         fake_ROI[indexes] = 1
#     ROI_img = image.new_img_like(atlas_aal_nii, fake_ROI)
#     # Visualize the resulting image
#     plotting.plot_stat_map(ROI_img, title="{}".format(ROI_name), annotate=False, vmax=1, colorbar=False, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap='Blues')
#     plotting.plot_roi(ROI_img, title="{} regions of AAL atlas".format(ROI_name))
#     # plt.savefig('{}/{}_mask.png'.format(results_dir, ROI_name), dpi=300)
#     # plt.close('all')
#     plt.show()
# results_dir = "results_in_Narps_data"
# compute_save_display_mask('Ventral Striatum', indices_hyp789, results_dir)
# compute_save_display_mask('VMPFC', indices_hyp1256, results_dir)
# compute_save_display_mask('Amygdala', indices_hyp34, results_dir)