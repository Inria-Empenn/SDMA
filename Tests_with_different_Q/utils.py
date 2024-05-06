
import numpy
import scipy
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from nilearn import plotting
import nibabel
import pandas
import seaborn
from nilearn.signal import clean

def SDMA_Stouffer(contrast_estimates, Q):
    K = contrast_estimates.shape[0]
    ones = numpy.ones((K, 1))
    attenuated_variance = ones.T.dot(Q).dot(ones) / K**2
    # compute meta-analytic statistics
    T_map = numpy.mean(contrast_estimates, 0)/numpy.sqrt(attenuated_variance)
    T_map = T_map.reshape(-1)
    # compute p-values for inference
    p_values = 1 - scipy.stats.norm.cdf(T_map)
    p_values = p_values.reshape(-1)
    weights = numpy.zeros(K)
    return T_map, p_values, weights

def GLS_SDMA(contrast_estimates, Q):
    K = contrast_estimates.shape[0]
    Q_inv = numpy.linalg.inv(Q)
    ones = numpy.ones((K, 1))
    top = ones.T.dot(Q_inv).dot(contrast_estimates)
    down = ones.T.dot(Q_inv).dot(ones)
    T_map = top/numpy.sqrt(down)
    # Assuming variance is estimated on whole image
    # and assuming infinite df
    p_values = 1 - scipy.stats.norm.cdf(T_map)
    p_values = p_values.reshape(-1)
    weights = (ones.T.dot(Q_inv).dot(ones))**(-1/2) * numpy.sum(Q_inv, axis=1) 
    return T_map, p_values, weights


def Q_from_std_data(contrast_estimates):
    # compute Q from standardize data
    scaler = StandardScaler()
    std_contrast_estimate = scaler.fit_transform(contrast_estimates.T).T # scaling team wise and back to normal shape
    # z* = (z - z_mean) / s 
    # with s = image-wise var for pipeline k
    # with z_mean = image-wise mean for pipeline k
    # numpy.divide(numpy.subtract(contrast_estimates.T, contrast_estimates.mean(axis=1)), contrast_estimates.std(axis=1))
    Q = numpy.corrcoef(std_contrast_estimate)
    return Q

def Q_regress_out_mean_effect(contrast_estimates):
    mean_voxel_per_team = numpy.mean(contrast_estimates, axis=1)
    cleaned_contrast_estimates = clean(contrast_estimates, confounds=mean_voxel_per_team, detrend=False, standardize=False)
    Q = numpy.corrcoef(cleaned_contrast_estimates)
    return Q

def Q_subsampled_voxels(contrast_estimates):
    Qs = []
    print("running correlation matrix with subsampled voxels")
    for i in range(100):
        if i%20==0:
            print(i, "/100")
        sampled_voxels_indices = numpy.random.choice(contrast_estimates.shape[1], size=1000000, replace=False)
        if i != 0:
            assert numpy.any(sampled_voxels_indices != previous_sampled_voxels_indices)
        previous_sampled_voxels_indices = sampled_voxels_indices
        
        contrast_estimates_subsample = contrast_estimates[:, sampled_voxels_indices]
        Q_subsample = numpy.corrcoef(contrast_estimates_subsample)
        Qs.append(Q_subsample)
    Q = numpy.mean(Qs, axis=0)
    return Q

def Q_subsampled_voxels_small_subsample(contrast_estimates):
    Qs = []
    print("running correlation matrix with subsampled voxels")
    for i in range(100):
        if i%20==0:
            print(i, "/100")
        sampled_voxels_indices = numpy.random.choice(contrast_estimates.shape[1], size=500000, replace=False)
        if i != 0:
            assert numpy.any(sampled_voxels_indices != previous_sampled_voxels_indices)
        previous_sampled_voxels_indices = sampled_voxels_indices
        
        contrast_estimates_subsample = contrast_estimates[:, sampled_voxels_indices]
        Q_subsample = numpy.corrcoef(contrast_estimates_subsample)
        Qs.append(Q_subsample)
    Q = numpy.mean(Qs, axis=0)
    return Q

    

def get_full_name(endings, pipeline_z_scores_per_team):
    # returns a list of full team name given a list of team name endings
    full_name = []
    for team_name_ending in endings:
        found = 0
        for team_name in pipeline_z_scores_per_team.keys():
            if team_name_ending in team_name:
                print(team_name_ending, " associated with ", team_name)
                full_name.append(team_name)
                found = 1
        if found  == 0:
            print(team_name_ending, "Not found")
    print("****")
    print("should have {} entries and got {}".format(len(endings), len(full_name)))
    print("****")
    return full_name




def plot_brain(MA_outputs, hyp, results_dir, masker):
    MA_estimators_names = ["SDMA Stouffer", "GLS SDMA"]
    plt.close('all')
    f, axs = plt.subplots(len(MA_estimators_names), 1, figsize=(8, len(MA_estimators_names)+2))
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
        pdata = numpy.squeeze(p_brain.get_fdata())
        tdata = numpy.squeeze(t_brain.get_fdata())
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
    plt.savefig("{}/thresholded_map_hyp{}.png".format(results_dir, hyp))
    plt.close('all')


def figure_for_Narps_weights(results_dir, hyp, data, Q, name, ticks=None, labels_order=None):
     plt.close('all')
     colspan = 14
     size_x_tot = 20
     f = plt.figure(figsize=(size_x_tot+5, 15))
     if hyp =="":
          plt.suptitle("Weights for HCP")
     else:
          plt.suptitle("Weights for hypothesis ".format(hyp))

     if ticks is None:
          ax1 = plt.subplot2grid((1,size_x_tot), (0,0), colspan=2)
          seaborn.heatmap(numpy.array([data['GLS SDMA'].values]).T, center=0, yticklabels=data.index, cmap='coolwarm', square=True, xticklabels=['GLS SDMA'], fmt='.1f', ax=ax1, cbar=True, cbar_kws={'shrink': 0.25})    
          ax1.tick_params(axis='x', rotation=90)
          ax1.set_title('Weights')
          
          ax2 = plt.subplot2grid((1,size_x_tot), (0,2), colspan=2)
          seaborn.heatmap(numpy.array([data['Mean score'].values]).T, center=0, yticklabels=data.index, cmap='coolwarm', square=True, xticklabels=['Voxels Mean'], fmt='.1f', ax=ax2, cbar=True, cbar_kws={'shrink': 0.25})
          ax2.tick_params(axis='x', rotation=90)
          
          ax3 = plt.subplot2grid((1,size_x_tot), (0,4), colspan=2)
          seaborn.heatmap(numpy.array([data['Var'].values]).T, center=0, yticklabels=data.index, cmap='Reds', square=True, xticklabels=['Voxels Variance'], fmt='.1f', ax=ax3, cbar=True, cbar_kws={'shrink': 0.25})
          ax3.tick_params(axis='x', rotation=90)
     
          ax4 = plt.subplot2grid((1,size_x_tot), (0,6), colspan=colspan)
          seaborn.heatmap(Q, center=0, cmap='coolwarm', xticklabels=data.index, yticklabels=data.index, square=True, fmt='.1f', ax=ax4, cbar=True, cbar_kws={'shrink': 0.25})

     else:
          ax1 = plt.subplot2grid((1,size_x_tot), (0,0))
          seaborn.heatmap(numpy.array([data['GLS SDMA'].values[labels_order]]).T, center=0, yticklabels=False, cmap='coolwarm', square=True, xticklabels=['GLS SDMA'], fmt='.1f', ax=ax1, cbar=True, cbar_kws={'shrink': 0.25})    
          ax1.set_yticks(ticks+0.5, labels_order)
          ax1.tick_params(axis='x', rotation=90)
          ax1.set_title('Weights')
          
          ax2 = plt.subplot2grid((1,size_x_tot), (0,1))
          seaborn.heatmap(numpy.array([data['Mean score'].values[labels_order]]).T, center=0, yticklabels=False, cmap='coolwarm', square=True, xticklabels=['Voxels Mean'], fmt='.1f', ax=ax2, cbar=True, cbar_kws={'shrink': 0.25})
          ax2.set_yticks(ticks+0.5, labels_order)
          ax2.tick_params(axis='x', rotation=90)
          
          ax3 = plt.subplot2grid((1,size_x_tot), (0,2))
          seaborn.heatmap(numpy.array([data['Var'].values[labels_order]]).T, center=0, yticklabels=False, cmap='Reds', square=True, xticklabels=['Voxels Variance'], fmt='.1f', ax=ax3, cbar=True, cbar_kws={'shrink': 0.25})
          ax3.set_yticks(ticks+0.5, labels_order)
          ax3.tick_params(axis='x', rotation=90)

          ax4 = plt.subplot2grid((1,size_x_tot), (0,3), colspan=colspan)
          seaborn.heatmap(Q, center=0, cmap='coolwarm', square=True, fmt='.1f', ax=ax4, xticklabels=False, yticklabels=False, cbar=True, cbar_kws={'shrink': 0.25})
          ax4.set_xticks(ticks+0.5, labels_order)
          ax4.set_yticks(ticks+0.5, labels_order)
     ax4.tick_params(axis='x', rotation=90)
     ax4.set_title('{}'.format(name),y=-0.08,pad=-14)
     plt.tight_layout()
     if hyp =="":
          plt.savefig("{}/weights_{}.png".format(results_dir, name))
     else:
          plt.savefig("{}/hyp{}_weights_{}.png".format(results_dir, hyp, name))
     
     plt.close('all')


def plot_weights(results_dir, contrast_estimates, Q, weights):
     team_names = list(weights.index)
     Q_sym = (Q+Q.T)/2
     Q_inv = numpy.linalg.inv(Q)
     Q_inv_sym = (Q_inv+Q_inv.T)/2

     Q_inv = pandas.DataFrame(data=Q_inv, columns=team_names, index=team_names)
     Q = pandas.DataFrame(data=Q, columns=team_names, index=team_names)

     data = weights[["GLS SDMA", "Mean score", "Var"]]
     ticks = numpy.arange(0, contrast_estimates.shape[0]) # for reordering labeling, needs to know index max
     
     figure_for_Narps_weights(results_dir, 1, data, Q, "Q", ticks=None, labels_order=None)
     figure_for_Narps_weights(results_dir, 1, data, Q_inv, "Qinv", ticks=None, labels_order=None)
     print("Done plotting")


### extract voxels that are well defined.
### that is, independant cluster and anticorrelated cluster data points are separated
# cluster in hyp 1 and setting SDMA 0 and GLS 1:

def search_for_nicelly_defined_voxels(clusters, clusters_name, team_names, pipeline_z_scores):
    print("Looking for nicelly defined clusters..")
    indices_per_cluster = {}
    for ind, cluster in enumerate(clusters):
        indices = []
        for team in cluster:
            indices.append(team_names.index(team))
        indices_per_cluster[clusters_name[ind]] = indices

    voxels_nicelly_defined_corrup = []
    voxels_nicelly_defined_corrdown = []
    for voxel_ind in range(0, pipeline_z_scores.shape[1]):
        voxel_values = pipeline_z_scores[:, voxel_ind]
        correlated_min = min(voxel_values[indices_per_cluster['correlated']])
        anti_correlated_max = max(voxel_values[indices_per_cluster['anti_correlated']])
        correlated_max = max(voxel_values[indices_per_cluster['correlated']])
        anti_correlated_min = min(voxel_values[indices_per_cluster['anti_correlated']])
        if correlated_min >= anti_correlated_max:
            voxels_nicelly_defined_corrup.append(voxel_ind)
        if correlated_max <= anti_correlated_min:
            voxels_nicelly_defined_corrdown.append(voxel_ind)
    print("Found: ", len(voxels_nicelly_defined_corrdown), " nicelly defined voxels corrdown")
    print("and : ", len(voxels_nicelly_defined_corrup), " nicelly defined voxels corrup")
    return voxels_nicelly_defined_corrup, voxels_nicelly_defined_corrdown

def search_for_significant_voxels_within_nicelly_defined_voxel(voxels_nicelly_defined_corrup, voxels_nicelly_defined_corrdown, MA_outputs):
    voxels_of_interest_corrup = {'SDMA1_GLS0':[], 'SDMA1_GLS1':[], 'SDMA0_GLS1':[], 'not_significant':[]}
    voxels_of_interest_corrdown = {'SDMA1_GLS0':[], 'SDMA1_GLS1':[], 'SDMA0_GLS1':[], 'not_significant':[]}
    # corr up
    for voxel_nicelly_defined in voxels_nicelly_defined_corrup:
        if MA_outputs['SDMA Stouffer']['p_values'][voxel_nicelly_defined] <= 0.05:
            if MA_outputs['GLS SDMA']['p_values'][voxel_nicelly_defined] <= 0.05:
                voxels_of_interest_corrup['SDMA1_GLS1'].append(voxel_nicelly_defined)
            else:
                voxels_of_interest_corrup['SDMA1_GLS0'].append(voxel_nicelly_defined)
        elif MA_outputs['GLS SDMA']['p_values'][voxel_nicelly_defined] <= 0.05:
            voxels_of_interest_corrup['SDMA0_GLS1'].append(voxel_nicelly_defined)
        else:
            voxels_of_interest_corrup['not_significant'].append(voxel_nicelly_defined)
    # corr down 
    for voxel_nicelly_defined in voxels_nicelly_defined_corrdown:
        if MA_outputs['SDMA Stouffer']['p_values'][voxel_nicelly_defined] <= 0.05:
            if MA_outputs['GLS SDMA']['p_values'][voxel_nicelly_defined] <= 0.05:
                voxels_of_interest_corrdown['SDMA1_GLS1'].append(voxel_nicelly_defined)
            else:
                voxels_of_interest_corrdown['SDMA1_GLS0'].append(voxel_nicelly_defined)
        elif MA_outputs['GLS SDMA']['p_values'][voxel_nicelly_defined] <= 0.05:
            voxels_of_interest_corrdown['SDMA0_GLS1'].append(voxel_nicelly_defined)
        else:
            voxels_of_interest_corrdown['not_significant'].append(voxel_nicelly_defined)
    return voxels_of_interest_corrup, voxels_of_interest_corrdown


def get_cluster_indices(clusters, clusters_name, team_names):
    clusters_indices = {}
    for ind, cluster in enumerate(clusters):
        cluster_indices = []
        for team in cluster:
            cluster_indices.append(team_names.index(team))
        clusters_indices[clusters_name[ind]] = cluster_indices
    return clusters_indices

def compute_GLS_weights(pipeline_z_scores, Q, std_by_Stouffer=True):
    # compute weight for each pipeline
    ones = numpy.ones((pipeline_z_scores.shape[0], 1))
    Q_inv = numpy.linalg.inv(Q)
    W_sdma = (ones.T.dot(Q).dot(ones))**(-1/2) # scalar
    W_sdma = W_sdma.reshape(-1)
    if std_by_Stouffer == True:
        weight_pipelines_gls = (ones.T.dot(Q_inv).dot(ones))**(-1/2) / W_sdma * numpy.sum(Q_inv, axis=1) 
    else:
        weight_pipelines_gls = (ones.T.dot(Q_inv).dot(ones))**(-1/2) * numpy.sum(Q_inv, axis=1) 
    weight_pipelines_gls = weight_pipelines_gls.reshape(-1)
    return weight_pipelines_gls # length = nb of pipelines

def compute_contributions(pipeline_z_scores, Q, W="SDMA", std_by_Stouffer=True):
     ones = numpy.ones((pipeline_z_scores.shape[0], 1))
     W_sdma = (ones.T.dot(Q).dot(ones))**(-1/2) # scalar
     if W == "SDMA":
          contributions_SDMA_Stouffer = pipeline_z_scores * W_sdma
     else:
          contributions_SDMA_Stouffer = pipeline_z_scores * W # W=1
     Q_inv = numpy.linalg.inv(Q)
     W_gls= (ones.T.dot(Q_inv).dot(ones))**(-1/2) * (numpy.sum(Q_inv, axis=1)).reshape(-1, 1) # vector
     if std_by_Stouffer==True:
          contributions_GLS = pipeline_z_scores * W_gls / W_sdma
     else:
          contributions_GLS = pipeline_z_scores * W_gls 
     return contributions_SDMA_Stouffer, contributions_GLS



def plot_clusters_brains(clusters, clusters_name, team_names, masker, pipeline_z_scores, results_dir, Q):
    print("Check using the original computation of GLS contributions:")
    T_map, p_map, _ = GLS_SDMA(pipeline_z_scores, Q)
    T_map_GLS_nii = masker.inverse_transform(T_map)
    p_map = (p_map <= 0.05) * T_map
    p_map_GLS_nii = masker.inverse_transform(p_map)

    T_map, p_map, _ = SDMA_Stouffer(pipeline_z_scores, Q)
    T_map_SMDA_Stouffer_nii = masker.inverse_transform(T_map)
    p_map = (p_map <= 0.05) * T_map
    p_map_SDMA_Stouffer_nii = masker.inverse_transform(p_map)

    print("getting cluster indices...")
    clusters_indices = get_cluster_indices(clusters, clusters_name, team_names)
    print("get gls weight per pipeline...")
    weight_pipelines_gls = compute_GLS_weights(pipeline_z_scores, Q, std_by_Stouffer=False)

    print("get SDMA Stouffer and GLS contributions...")
    contributions_SDMA_Stouffer, contributions_GLS = compute_contributions(pipeline_z_scores, Q, W="SDMA", std_by_Stouffer=False)
    mean_contributions_SDMA_Stouffer_nii = masker.inverse_transform(numpy.mean(contributions_SDMA_Stouffer, axis=0))
    mean_contributions_GLS_nii = masker.inverse_transform(numpy.mean(contributions_GLS, axis=0))
    
    plt.close('all')
    f, axs = plt.subplots(len(clusters_name) + 2, 4, figsize=(16, len(clusters_name)*5), width_ratios=[0.1, 0.4, 0.4, 0.1])

    # plot mean SDMA Stouffer weight for this cluster
    ones = numpy.ones((pipeline_z_scores.shape[0], 1))
    SDMA_Stouffer_weight = (ones.T.dot(Q).dot(ones))**(-1/2) # scalar
    SDMA_Stouffer_weight = numpy.round(SDMA_Stouffer_weight, 4)
    axs[0, 0].imshow(numpy.array(SDMA_Stouffer_weight).reshape(-1, 1), cmap='coolwarm', aspect='equal', vmin=-0.5, vmax=0.5)
    axs[0, 0].text(0, 0, float(SDMA_Stouffer_weight), ha="center", va="center", color="black")
    axs[0, 0].axis('off')
    axs[0, 0].set_title('SDMA Stouffer weight')
    

    for row, name in enumerate(clusters_name):
        print("Drawing the mean weights + sum of contributions for cluster: ", name)
        this_cluster_indices = clusters_indices[name]
        this_cluster_contributions_SDMA_Stouffer = contributions_SDMA_Stouffer[this_cluster_indices]
        this_cluster_contributions_GLS = contributions_GLS[this_cluster_indices]

        # plot mean weight for this cluster
        mean_weight_of_this_cluster = numpy.round(weight_pipelines_gls[this_cluster_indices].mean(), 4)
        axs[row, 3].imshow(numpy.array(mean_weight_of_this_cluster).reshape(-1, 1), cmap='coolwarm', aspect='equal', vmin=-0.5, vmax=0.5)
        axs[row, 3].text(0, 0, mean_weight_of_this_cluster, ha="center", va="center", color="black")
        axs[row, 3].axis('off')
        if row == 0:
            axs[row, 3].set_title('mean GLS weight') 
        # take off axis where sdma weight is
        axs[row, 0].axis('off')



        # plot mean SDMA Stouffer contribution for this cluster
        sum_contributions_SDMA_Stouffer_this_cluster_nii = masker.inverse_transform(this_cluster_contributions_SDMA_Stouffer.sum(axis=0))
        plotting.plot_stat_map(sum_contributions_SDMA_Stouffer_this_cluster_nii,  
            annotate=False,  
            colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), 
            display_mode='z', cmap='coolwarm', axes=axs[row, 1])
        axs[row, 1].set_title("Sum contributions " + name, size=12)

        # plot mean GLS contribution for this cluster
        sum_contributions_GLS_this_cluster_nii = masker.inverse_transform(this_cluster_contributions_GLS.sum(axis=0))
        plotting.plot_stat_map(sum_contributions_GLS_this_cluster_nii,  
            annotate=False,  
            colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), 
            display_mode='z', cmap='coolwarm', axes=axs[row, 2])
        axs[row, 2].set_title("Sum contributions " + name, size=12)


    # plot whole brain SDMA Stouffer contribution computed from MA_estimator
    plotting.plot_stat_map(T_map_SMDA_Stouffer_nii,
        annotate=False,  
        colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), 
        display_mode='z', cmap='coolwarm', axes=axs[row+1, 1])
    axs[row+1, 1].set_title("T map", size=12)

    # plot whole brain GLS contribution computed from MA_estimator
    plotting.plot_stat_map(T_map_GLS_nii,
        annotate=False,  
        colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), 
        display_mode='z', cmap='coolwarm', axes=axs[row+1, 2])
    axs[row+1, 2].set_title("T map", size=12)

    # plot p value significant using SDMA Stouffer
    plotting.plot_stat_map(p_map_SDMA_Stouffer_nii,
        annotate=False, vmax=8,
        colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), 
        display_mode='z', cmap='Reds', axes=axs[row+2, 1])
    axs[row+2, 1].set_title("Significant T values", size=12)

    # plot p value significant using GLS
    plotting.plot_stat_map(p_map_GLS_nii,
        annotate=False, vmax=8,
        colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), 
        display_mode='z', cmap='Reds', axes=axs[row+2, 2])
    axs[row+2, 2].set_title("Significant T values", size=12)


    axs[row+1, 0].axis('off')
    axs[row+2, 0].axis('off')
    axs[row+1, 3].axis('off')
    axs[row+2, 3].axis('off')
    print("Saving plot")
    plt.savefig("{}/3clusters_results/3_cluster_weights.png".format(results_dir))
    # plt.show()