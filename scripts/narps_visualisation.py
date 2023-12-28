from nilearn import plotting
import matplotlib.pyplot as plt
import numpy
import nibabel
from statsmodels.stats.multitest import multipletests
from nilearn import masking
from sklearn.linear_model import LinearRegression
from nilearn import image
import seaborn
import pandas


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

    if hyp == "":
        plt.suptitle('HCP')
        plt.tight_layout()
        plt.savefig("{}/MA_outputs.png".format(results_dir))
        plt.close('all')

    else:
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
    #  for OHBM figures
    # MA_estimators_names = [MA_estimators_names[1], MA_estimators_names[2], MA_estimators_names[3],
    #         MA_estimators_names[5], MA_estimators_names[6]]
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
    if hyp == "":
        plt.suptitle('HCP')
        plt.savefig("{}/thresholded_map.png".format(results_dir))
        plt.close('all')
    else:
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
    if hyp =="":
        plt.savefig("{}/similarity_maps.png".format(results_dir))
    else:
        plt.savefig("{}/similarity_maps_hyp{}.png".format(results_dir, hyp))
    plt.close('all')
    # plot contrast
    contrasts_mask = masking.intersect_masks(threshimgs, threshold=0, connected=True)
    plotting.plot_stat_map(contrasts_mask, annotate=False, vmax=1, colorbar=False, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap='Blues')
    plt.suptitle("Contrasts between SDMA models")
    if hyp =="":
        plt.savefig("{}/contrast_maps.png".format(results_dir))
    else:
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


def compute_betas(resampled_maps):
    print("Compute betas for a pipeline and substract the sum of betas from the pipeline voxels value")
    residuals_maps = numpy.zeros(resampled_maps.shape) # K*J
    coefficients = []
    K = resampled_maps.shape[0]
    for k in range(K):
        print("computing pipeline ", k)
        Y = resampled_maps[k][:]
        resampled_maps_without_k = numpy.delete(resampled_maps, [k], axis=0)
        X = resampled_maps_without_k[:][:]
        reg = LinearRegression().fit(X.T, Y)
        # save betas
        coefficients.append(reg.coef_)
        # remove betas from pipeline voxels value
        residuals_maps[k][:] = resampled_maps[k][:] - numpy.sum(reg.coef_.reshape(-1, 1)*resampled_maps_without_k, axis=0)
    return coefficients, residuals_maps


def plot_betas(coefficients, hyp, results_dir, team_names):
    coefficients_to_plot=[]
    for ind, coefs in enumerate(coefficients):
        coefs = numpy.insert(coefs, ind, 0)
        coefficients_to_plot.append(coefs)
    coefficients_to_plot = numpy.array(coefficients_to_plot)
    coefficients_to_plot = pandas.DataFrame(data=coefficients_to_plot, columns=team_names, index=team_names)
    plt.close("all")
    f, ax = plt.subplots(figsize=(20, 20))
    seaborn.heatmap(coefficients_to_plot, center=0, cmap="coolwarm", square=True, fmt='.1f', cbar_kws={"shrink": 0.25}, figure=f, ax=ax)
    if hyp =="":
        plt.suptitle("betas",fontsize=12)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.savefig("{}/betas".format(results_dir))
    else:
        plt.suptitle("hyp {} betas".format(hyp),fontsize=12)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.savefig("{}/Hyp{}_betas".format(results_dir, hyp))
    plt.close("all")


def plot_nii_maps(resampled_maps, masker, hyp, results_dir, title):
    # 60 / 20 = 3 figures with 20 plots each 
    nii_maps = resampled_maps.values()
    numerical_maps = masker.fit_transform(nii_maps)
    team_names = list(resampled_maps.keys())
    max_range =  len(team_names)
    v_max = 10 if title == "resampled" else 5
    print("Max=",numerical_maps.max(), "Min=", numerical_maps.min(), "picked=", v_max)
    for i_fig in range(3):
        plt.close('all')
        if i_fig == 0:
            f, axs = plt.subplots(int(11), 2, figsize=(30, 35))
            for pipeline_nb in range(22):
                print("Plotting pipeline ", team_names[pipeline_nb], pipeline_nb, "/", len(nii_maps))
                row, col = [pipeline_nb, 0] if pipeline_nb <= 10 else [pipeline_nb - 11, 1]
                title_graph = "{}, ({}, {})".format(team_names[pipeline_nb], int(numpy.max((numerical_maps[pipeline_nb]))), int(numpy.min((numerical_maps[pipeline_nb]))))
                plotting.plot_stat_map(image.index_img(nii_maps, pipeline_nb), cut_coords=(-21, 0, 9), vmax=v_max, cmap="coolwarm", figure=f, axes=axs[row, col], title=title_graph)
            print('saving...')
            plt.suptitle("hyp {} part {}, {}".format(hyp, i_fig, title),fontsize=20)
            plt.savefig("{}/Hyp{}_{}_maps_part_{}".format(results_dir, hyp, title, i_fig))
            plt.close('all')
        elif i_fig == 1:
            f, axs = plt.subplots(int(11), 2, figsize=(30, 35))
            for pipeline_nb in range(22, 44):
                print("Plotting pipeline ", team_names[pipeline_nb], pipeline_nb, "/", len(nii_maps))
                row, col = [pipeline_nb-22, 0] if pipeline_nb-22 <= 10 else [pipeline_nb - 33, 1]
                title_graph = "{}, ({}, {})".format(team_names[pipeline_nb], int(numpy.max((numerical_maps[pipeline_nb]))), int(numpy.min((numerical_maps[pipeline_nb]))))
                plotting.plot_stat_map(image.index_img(nii_maps, pipeline_nb), cut_coords=(-21, 0, 9), vmax=v_max, cmap="coolwarm", figure=f, axes=axs[row, col], title=title_graph)
            print('saving...')
            plt.suptitle("hyp {} part {}, {}".format(hyp, i_fig, title),fontsize=20)
            plt.savefig("{}/Hyp{}_{}_maps_part_{}".format(results_dir, hyp, title, i_fig))
            plt.close('all')
        elif i_fig == 2:
            f, axs = plt.subplots(int(11), 2, figsize=(30, 35))
            for pipeline_nb in range(44, max_range): # max_range = nb of pipelines
                print("Plotting pipeline ", team_names[pipeline_nb], pipeline_nb, "/", len(nii_maps))
                row, col = [pipeline_nb-44, 0] if pipeline_nb-44 <= 10 else [pipeline_nb - 55, 1]
                try:
                    title_graph = "{}, ({}, {})".format(team_names[pipeline_nb], int(numpy.max((numerical_maps[pipeline_nb]))), int(numpy.min((numerical_maps[pipeline_nb]))))
                    plotting.plot_stat_map(image.index_img(nii_maps, pipeline_nb), cut_coords=(-21, 0, 9), vmax=v_max, cmap="coolwarm", figure=f, axes=axs[row, col], title=title_graph)
                except:
                    axs[row, col].set_axis_off()
                    print("no pipeline ", pipeline_nb)
            print('saving...')
            plt.suptitle("hyp {} part {}, {}".format(hyp, i_fig, title),fontsize=20)
            plt.savefig("{}/Hyp{}_{}_maps_part_{}".format(results_dir, hyp, title, i_fig))
            plt.close('all')


def plot_hcp_maps(resampled_maps, team_names, masker, results_dir, title):
    # 1 fig 24 plots in total
    numerical_maps = resampled_maps
    nii_maps = masker.inverse_transform(resampled_maps)
    max_range =  len(team_names)
    v_max = 10 if title == "resampled" else 1
    print("Max=",numerical_maps.max(), "Min=", numerical_maps.min(), "picked=", v_max)
    plt.close('all')
    f, axs = plt.subplots(12, 2, figsize=(30, 35))
    for pipeline_nb in range(max_range):
        print("Plotting pipeline ", team_names[pipeline_nb], ", => ", pipeline_nb, "/", max_range)
        row, col = [pipeline_nb, 0] if pipeline_nb <= 11 else [pipeline_nb - 12, 1]
        title_graph = "{}, ({}, {})".format(team_names[pipeline_nb], int(numpy.max((numerical_maps[pipeline_nb]))), int(numpy.min((numerical_maps[pipeline_nb]))))
        plotting.plot_stat_map(image.index_img(nii_maps, pipeline_nb), cut_coords=(-21, 0, 9), vmax=v_max, cmap="coolwarm", figure=f, axes=axs[row, col], title=title_graph)
    print('saving...')
    plt.suptitle(title,fontsize=20)
    plt.savefig("{}/{}_maps".format(results_dir, title))
    plt.close('all')
    

# from nilearn.datasets import load_mni152_brain_mask
# from nilearn.datasets import fetch_atlas_harvard_oxford
# import nilearn.plotting as plotting
# import numpy as np
# from nibabel import Nifti1Image

# atlas = fetch_atlas_harvard_oxford('sub-maxprob-thr0-1mm')
# # Define the index of the precentral gyrus in the atlas (in this case, index 3)
# roi_indexes = [5, 6, 16, 17] # striatum
# roi_indexes = [10, 20] # amygdale

# # Create a binary mask for the precentral gyrus
# roi_mask = np.zeros_like(atlas.maps.get_fdata())
# for roi_i in roi_indexes:
#     roi_mask[atlas.maps.get_fdata() == roi_i] = 1
# # Create a new Nifti image from the mask
# roi_nifti = Nifti1Image(roi_mask, affine=atlas.maps.affine)
# # Plot the roi using Nilearn
# plotting.plot_stat_map(roi_nifti, title="Amygdala", annotate=False, vmax=1, colorbar=False, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap='Blues')
# # plotting.plot_roi(roi_nifti, title=atlas.labels[roi_index], display_mode='ortho', cut_coords=(40, -20, 50))
# # Show the plot
# # plt.savefig('results_in_Narps_data/VMPFC_mask.png')
# plt.show()
# plt.close('all')




# atlas_aal = fetch_atlas_aal()
# hyp789 = [
#     'Amygdala_R',
#     'Amygdala_L'
# ]
# hyp1256= [
#     'Frontal_Sup_Orb_L',
#     'Frontal_Sup_Orb_R',
#     'Frontal_Mid_Orb_L',
#     'Frontal_Mid_Orb_R',
#     'Frontal_Inf_Orb_L',
#     'Frontal_Inf_Orb_R',
#     'Frontal_Med_Orb_L',
#     'Frontal_Med_Orb_R',
#     'Olfactory_L',
#     'Olfactory_R',

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
#     plt.savefig('{}/{}_AALmask.png'.format(results_dir, ROI_name), dpi=300)
#     plt.close('all')
#     plt.show()
# results_dir = "results_in_Narps_data"
# compute_save_display_mask('Amygdala', indices_hyp789, results_dir)
# compute_save_display_mask('VMPFC', indices_hyp1256, results_dir)
# compute_save_display_mask('Ventral Striatum', indices_hyp34, results_dir)