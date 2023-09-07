from nilearn import plotting
import matplotlib.pyplot as plt
import numpy
import nibabel
from statsmodels.stats.multitest import multipletests

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
    f, axs = plt.subplots(len(MA_estimators_names), 1, figsize=(8, len(MA_estimators_names)*8/5))
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
        long_title = title + ', {}%'.format(perc_sign_voxels)

        # plotting.plot_stat_map(t_brain, title=long_title, colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap="coolwarm", ax=axs[row, 0])
        # plt.savefig("results_narpsdata/{}_hyp{}_t_map.png".format(title, hyp))
        # plt.close('all')
        # plotting.plot_stat_map(p_brain, threshold=0.1, title=long_title, colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap="Reds")
        # plt.savefig("results_narpsdata/{}_hyp{}_p_map.png".format(title, hyp))
        # plt.close('all')
        plotting.plot_stat_map(threshimg, annotate=False, title=long_title, threshold=0.1,vmax=8, colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap='Reds', axes=axs[row])
        
        # plt.savefig("results_narpsdata/{}_hyp{}_p_sign_map.png".format(title, hyp))
        # plt.close('all')
    plt.suptitle('Hypothesis {}'.format(hyp))
    plt.savefig("{}/thresholded_map_hyp{}_nofdr.png".format(results_dir, hyp))
    plt.close('all')
