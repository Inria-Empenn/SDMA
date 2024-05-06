import numpy
import nibabel
from nilearn import plotting
from nilearn.input_data import NiftiMasker
import matplotlib.pyplot as plt
import utils
import MA_estimators
import importlib
import os
from matplotlib.patches import Patch
import random
import seaborn as sns


importlib.reload(utils)
importlib.reload(MA_estimators)


data_path = "/home/jlefortb/SDMA/hcp_data/preprocessed"
participant_mask = os.path.join(data_path, "mask.nii.gz")
# fit masker
masker = NiftiMasker(
    mask_img=participant_mask)
results_dir = "/home/jlefortb/SDMA/results_in_HCP_data"
 
resampled_maps = numpy.load('{}/data/resampled_maps.npy'.format(results_dir), allow_pickle=True) # mind result dir
team_names = numpy.load('{}/data/pipeline_names.npy'.format(results_dir), allow_pickle=True) # mind result dir
team_names = list(team_names)
print("resampled_maps successfully loaded")
# fit masker
masker.fit(resampled_maps)

if not os.path.exists(os.path.join(results_dir, "2clusters_results")):
	os.mkdir(os.path.join(results_dir, "2clusters_results"))



def get_cluster_indices(clusters, clusters_name, team_names):
	clusters_indices = {}
	for ind, cluster in enumerate(clusters):
		cluster_indices = []
		for team in cluster:
			cluster_indices.append(team_names.index(team))
		clusters_indices[clusters_name[ind]] = cluster_indices
	return clusters_indices


def compute_GLS_weights(pipeline_z_scores, std_by_Stouffer=True):
	# compute weight for each pipeline
	ones = numpy.ones((pipeline_z_scores.shape[0], 1))
	Q = numpy.corrcoef(pipeline_z_scores)
	Q_inv = numpy.linalg.inv(Q)
	W_sdma = (ones.T.dot(Q).dot(ones))**(-1/2) # scalar
	W_sdma = W_sdma.reshape(-1)
	if std_by_Stouffer == True:
		weight_pipelines_gls = (ones.T.dot(Q_inv).dot(ones))**(-1/2) / W_sdma * numpy.sum(Q_inv, axis=1) 
	else:
		weight_pipelines_gls = (ones.T.dot(Q_inv).dot(ones))**(-1/2) * numpy.sum(Q_inv, axis=1) 
	weight_pipelines_gls = weight_pipelines_gls.reshape(-1)
	return weight_pipelines_gls # length = nb of pipelines


def custom_swarmplot(data, ax, team_names, clusters_name, clusters):
	c=['orange', 'blue', 'red', "green", "yellow", "black", "grey", "purple", "lightgreen"]
	for i, value in enumerate(data):
		for ind, name in enumerate(clusters_name):
			if team_names[i] in clusters[ind]:
				ax.plot(0, value, 'o', markersize=6, alpha=0)
				ax.text(0, value, team_names[i], color=c[ind], fontsize=6, ha='center', va='bottom')
				continue



def plot_distance_from_highly_correlated_cluster(resampled_maps, team_names, results_dir, clusters_name, clusters, four_voxels_nicelly_defined):
	plt.close("all")
	print("getting cluster indices...")
	clusters_indices = get_cluster_indices(clusters, clusters_name, team_names)
	print("get highly correlated pipelines mean")
	highly_correlated_pipelines = [element for element in team_names if element.startswith("spm")]
	highly_correlated_pipelines_indices = get_cluster_indices([highly_correlated_pipelines], ["SPM"], team_names)
	print("get gls weight per pipeline...")
	weight_pipelines_gls = compute_GLS_weights(resampled_maps, std_by_Stouffer=False)
	print("get SDMA Stouffer and GLS contributions...")
	_, contributions_GLS = utils.compute_contributions(resampled_maps, std_by_Stouffer=False)
	correlated_cluster_indices = highly_correlated_pipelines_indices["SPM"]
	correlated_cluster_mean = resampled_maps[correlated_cluster_indices].mean()
	dist_pipelines_mean_from_correlated_mean = []
	for pipeline, weight in enumerate(weight_pipelines_gls):
		dist_pipeline_mean_from_correlated_mean = resampled_maps[pipeline, :].mean() - correlated_cluster_mean
		dist_pipelines_mean_from_correlated_mean.append(numpy.abs(dist_pipeline_mean_from_correlated_mean))

		# Create a figure and axis
	fig = plt.figure(figsize=(20, 8))

	# PLOT 1: SDMA STOUFFER CONTRIBUTIONS 
	ax0 = plt.subplot2grid((4,9), (0,0), rowspan=2, colspan=2)
	sns.regplot(x=dist_pipelines_mean_from_correlated_mean, y=weight_pipelines_gls,ci=None, ax=ax0, color='lightgrey')
	# ax0.plot(dist_pipelines_mean_from_correlated_mean, weight_pipelines_gls, 'o', markersize=6)
	ax0.set_xlabel("Distance from correlated pipelines")
	ax0.set_ylabel("GLS weights")

	ax6 = plt.subplot2grid((4,9), (1,4), rowspan=2, colspan=2)
	ax6.set_xlabel("Distance from correlated pipelines")
	ax6.set_ylabel("Z-scores")


	ax1 = plt.subplot2grid((4,9), (2,0), rowspan=2, colspan=2)
	sns.regplot(x=dist_pipelines_mean_from_correlated_mean, y=contributions_GLS.mean(axis=1),ci=None, ax=ax1, color='lightgrey')
	ax1.set_xlabel("Distance from correlated pipelines")
	ax1.set_ylabel("GLS contributions")

	c=['orange', 'blue', 'red', "green", "yellow", "black", "grey", "purple", "lightgreen"]
	for ind, dist in enumerate(dist_pipelines_mean_from_correlated_mean):
		for i, name in enumerate(clusters_name):
			if team_names[ind] in clusters[i]:
				ax1.plot(dist, contributions_GLS.mean(axis=1)[ind], 'o', markersize=6, color=c[i], alpha=0.5)
				ax0.plot(dist, weight_pipelines_gls[ind], 'o', markersize=6, color=c[i], alpha=0.5)
				ax6.plot(dist, resampled_maps.mean(axis=1)[ind], 'o', markersize=6, color=c[i], alpha=0.5)

	# Add legend to plot 3
	lines = []
	for i, name in enumerate(clusters_name):
		lines.append(Patch(color=c[i], label=name))
	ax1.legend(handles=lines, loc='lower right', prop={'size': 6})

	original_zscore_for_4voxels = []
	dist_pipelines_from_correlated_mean_for_4voxels = []
	contributions_GLS_for_4_voxels = []
	for voxel_significance_profil in four_voxels_nicelly_defined.keys():
		voxel_ind = four_voxels_nicelly_defined[voxel_significance_profil]
		contributions_GLS_for_4_voxels.append(contributions_GLS[:, voxel_ind])
		original_zscore_for_4voxels.append(resampled_maps[:, voxel_ind])
		dist_pipelines_from_correlated_mean_for_this_voxel = []
		correlated_cluster_mean_for_this_voxel = resampled_maps[correlated_cluster_indices][:, voxel_ind].mean()
		for pipeline, _ in enumerate(weight_pipelines_gls):
			dist_pipeline_from_correlated_mean_of_this_voxel = resampled_maps[pipeline, voxel_ind] - correlated_cluster_mean_for_this_voxel
			dist_pipelines_from_correlated_mean_for_this_voxel.append(dist_pipeline_from_correlated_mean_of_this_voxel)
		dist_pipelines_from_correlated_mean_for_4voxels.append(numpy.abs(dist_pipelines_from_correlated_mean_for_this_voxel))

	ax2 = plt.subplot2grid((4,9), (0,2), rowspan=1, colspan=2)
	sns.regplot(x=dist_pipelines_from_correlated_mean_for_4voxels[0], y=contributions_GLS_for_4_voxels[0],ci=None, ax=ax2, color='lightgrey')
	ax2.set_xlabel("Distance, {}".format(list(four_voxels_nicelly_defined.keys())[0]))
	ax2.set_ylabel("GLS contributions")
	for ind, dist in enumerate(dist_pipelines_from_correlated_mean_for_4voxels[0]):
		for i, name in enumerate(clusters_name):
			if team_names[ind] in clusters[i]:
				ax2.plot(dist, contributions_GLS_for_4_voxels[0][ind], 'o', markersize=6, color=c[i], alpha=0.5)

	ax7 = plt.subplot2grid((4,9), (0,6), rowspan=1, colspan=2)
	ax7.set_xlabel("Distance, {}".format(list(four_voxels_nicelly_defined.keys())[0]))
	ax7.set_ylabel("Z-scores")
	for ind, dist in enumerate(dist_pipelines_from_correlated_mean_for_4voxels[0]):
		for i, name in enumerate(clusters_name):
			if team_names[ind] in clusters[i]:
				ax7.plot(dist, original_zscore_for_4voxels[0][ind], 'o', markersize=6, color=c[i], alpha=0.5)


	ax3 = plt.subplot2grid((4,9), (1,2), rowspan=1, colspan=2)
	sns.regplot(x=dist_pipelines_from_correlated_mean_for_4voxels[1], y=contributions_GLS_for_4_voxels[1],ci=None, ax=ax3, color='lightgrey')
	ax3.set_xlabel("Distance, {}".format(list(four_voxels_nicelly_defined.keys())[1]))
	ax3.set_ylabel("GLS contributions")
	for ind, dist in enumerate(dist_pipelines_from_correlated_mean_for_4voxels[1]):
		for i, name in enumerate(clusters_name):
			if team_names[ind] in clusters[i]:
				ax3.plot(dist, contributions_GLS_for_4_voxels[1][ind], 'o', markersize=6, color=c[i], alpha=0.5)

	ax8 = plt.subplot2grid((4,9), (1,6), rowspan=1, colspan=2)
	ax8.set_xlabel("Distance, {}".format(list(four_voxels_nicelly_defined.keys())[1]))
	ax8.set_ylabel("Z-scores")
	for ind, dist in enumerate(dist_pipelines_from_correlated_mean_for_4voxels[1]):
		for i, name in enumerate(clusters_name):
			if team_names[ind] in clusters[i]:
				ax8.plot(dist, original_zscore_for_4voxels[1][ind], 'o', markersize=6, color=c[i], alpha=0.5)

	ax4 = plt.subplot2grid((4,9), (2,2), rowspan=1, colspan=2)
	sns.regplot(x=dist_pipelines_from_correlated_mean_for_4voxels[2], y=contributions_GLS_for_4_voxels[2],ci=None, ax=ax4, color='lightgrey')
	ax4.set_xlabel("Distance, {}".format(list(four_voxels_nicelly_defined.keys())[2]))
	ax4.set_ylabel("GLS contributions")
	for ind, dist in enumerate(dist_pipelines_from_correlated_mean_for_4voxels[2]):
		for i, name in enumerate(clusters_name):
			if team_names[ind] in clusters[i]:
				ax4.plot(dist, contributions_GLS_for_4_voxels[2][ind], 'o', markersize=6, color=c[i], alpha=0.5)

	ax9 = plt.subplot2grid((4,9), (2,6), rowspan=1, colspan=2)
	ax9.set_xlabel("Distance, {}".format(list(four_voxels_nicelly_defined.keys())[2]))
	ax9.set_ylabel("Z-scores")
	for ind, dist in enumerate(dist_pipelines_from_correlated_mean_for_4voxels[2]):
		for i, name in enumerate(clusters_name):
			if team_names[ind] in clusters[i]:
				ax9.plot(dist, original_zscore_for_4voxels[2][ind], 'o', markersize=6, color=c[i], alpha=0.5)

	ax5 = plt.subplot2grid((4,9), (3,2), rowspan=1, colspan=2)
	sns.regplot(x=dist_pipelines_from_correlated_mean_for_4voxels[3], y=contributions_GLS_for_4_voxels[3],ci=None, ax=ax5, color='lightgrey')
	ax5.set_xlabel("Distance, {}".format(list(four_voxels_nicelly_defined.keys())[3]))
	ax5.set_ylabel("GLS contributions")
	for ind, dist in enumerate(dist_pipelines_from_correlated_mean_for_4voxels[3]):
		for i, name in enumerate(clusters_name):
			if team_names[ind] in clusters[i]:
				ax5.plot(dist, contributions_GLS_for_4_voxels[3][ind], 'o', markersize=6, color=c[i], alpha=0.5)

	ax10 = plt.subplot2grid((4,9), (3,6), rowspan=1, colspan=2)
	ax10.set_xlabel("Distance, {}".format(list(four_voxels_nicelly_defined.keys())[3]))
	ax10.set_ylabel("Z-scores")
	for ind, dist in enumerate(dist_pipelines_from_correlated_mean_for_4voxels[3]):
		for i, name in enumerate(clusters_name):
			if team_names[ind] in clusters[i]:
				ax10.plot(dist, original_zscore_for_4voxels[3][ind], 'o', markersize=6, color=c[i], alpha=0.5)

	plt.tight_layout()
	plt.savefig("{}/{}clusters_results/contribution_per_distance.png".format(results_dir,len(clusters_name)))


def plot_distance_from_highly_correlated_cluster_absolute(resampled_maps, team_names, results_dir, clusters_name, clusters, four_voxels_nicelly_defined):
	plt.close("all")
	print("getting cluster indices...")
	clusters_indices = get_cluster_indices(clusters, clusters_name, team_names)
	print("get highly correlated pipelines mean")
	highly_correlated_pipelines = [element for element in team_names if element.startswith("spm")]
	highly_correlated_pipelines_indices = get_cluster_indices([highly_correlated_pipelines], ["SPM"], team_names)
	print("get gls weight per pipeline...")
	weight_pipelines_gls = compute_GLS_weights(resampled_maps, std_by_Stouffer=False)
	print("get SDMA Stouffer and GLS contributions...")
	_, contributions_GLS = utils.compute_contributions(resampled_maps, std_by_Stouffer=False)
	correlated_cluster_indices = highly_correlated_pipelines_indices["SPM"]
	correlated_cluster_mean = resampled_maps[correlated_cluster_indices].mean()
	dist_pipelines_mean_from_correlated_mean = []
	for pipeline, weight in enumerate(weight_pipelines_gls):
		dist_pipeline_mean_from_correlated_mean = resampled_maps[pipeline, :].mean() - correlated_cluster_mean
		dist_pipelines_mean_from_correlated_mean.append(numpy.abs(dist_pipeline_mean_from_correlated_mean))

		# Create a figure and axis
	fig = plt.figure(figsize=(8, 8))

	# PLOT 1: SDMA STOUFFER CONTRIBUTIONS 
	ax0 = plt.subplot2grid((4,3), (0,0), rowspan=2, colspan=2)
	sns.regplot(x=dist_pipelines_mean_from_correlated_mean, y=numpy.abs(weight_pipelines_gls),ci=None, ax=ax0, color='lightgrey')
	# ax0.plot(dist_pipelines_mean_from_correlated_mean, weight_pipelines_gls, 'o', markersize=6)
	ax0.set_xlabel("Distance from correlated pipelines")
	ax0.set_ylabel("GLS weights")


	ax1 = plt.subplot2grid((4,3), (2,0), rowspan=2, colspan=2)
	sns.regplot(x=dist_pipelines_mean_from_correlated_mean, y=numpy.abs(contributions_GLS.mean(axis=1)),ci=None, ax=ax1, color='lightgrey')
	ax1.set_xlabel("Distance from correlated pipelines")
	ax1.set_ylabel("GLS contributions")

	c=['orange', 'blue', 'red', "green", "yellow", "black", "grey", "purple", "lightgreen"]
	for ind, dist in enumerate(dist_pipelines_mean_from_correlated_mean):
		for i, name in enumerate(clusters_name):
			if team_names[ind] in clusters[i]:
				ax1.plot(dist, numpy.abs(contributions_GLS.mean(axis=1)[ind]), 'o', markersize=6, color=c[i], alpha=0.5)
				ax0.plot(dist, numpy.abs(weight_pipelines_gls[ind]), 'o', markersize=6, color=c[i], alpha=0.5)

	# Add legend to plot 3
	lines = []
	for i, name in enumerate(clusters_name):
		lines.append(Patch(color=c[i], label=name))
	ax1.legend(handles=lines, loc='lower right', prop={'size': 6})

	dist_pipelines_from_correlated_mean_for_4voxels = []
	contributions_GLS_for_4_voxels = []
	for voxel_significance_profil in four_voxels_nicelly_defined.keys():
		voxel_ind = four_voxels_nicelly_defined[voxel_significance_profil]
		contributions_GLS_for_4_voxels.append(contributions_GLS[:, voxel_ind])
		dist_pipelines_from_correlated_mean_for_this_voxel = []
		correlated_cluster_mean_for_this_voxel = resampled_maps[correlated_cluster_indices][:, voxel_ind].mean()
		for pipeline, _ in enumerate(weight_pipelines_gls):
			dist_pipeline_from_correlated_mean_of_this_voxel = resampled_maps[pipeline, voxel_ind] - correlated_cluster_mean_for_this_voxel
			dist_pipelines_from_correlated_mean_for_this_voxel.append(dist_pipeline_from_correlated_mean_of_this_voxel)
		dist_pipelines_from_correlated_mean_for_4voxels.append(numpy.abs(dist_pipelines_from_correlated_mean_for_this_voxel))

	ax2 = plt.subplot2grid((4,3), (0,2), rowspan=1, colspan=1)
	sns.regplot(x=dist_pipelines_from_correlated_mean_for_4voxels[0], y=numpy.abs(contributions_GLS_for_4_voxels[0]),ci=None, ax=ax2, color='lightgrey')
	ax2.set_xlabel("Distance, {}".format(list(four_voxels_nicelly_defined.keys())[0]))
	ax2.set_ylabel("GLS contributions")
	for ind, dist in enumerate(dist_pipelines_from_correlated_mean_for_4voxels[0]):
		for i, name in enumerate(clusters_name):
			if team_names[ind] in clusters[i]:
				ax2.plot(dist, numpy.abs(contributions_GLS_for_4_voxels[0][ind]), 'o', markersize=6, color=c[i], alpha=0.5)


	ax3 = plt.subplot2grid((4,3), (1,2), rowspan=1, colspan=1)
	sns.regplot(x=dist_pipelines_from_correlated_mean_for_4voxels[1], y=numpy.abs(contributions_GLS_for_4_voxels[1]),ci=None, ax=ax3, color='lightgrey')
	ax3.set_xlabel("Distance, {}".format(list(four_voxels_nicelly_defined.keys())[1]))
	ax3.set_ylabel("GLS contributions")
	for ind, dist in enumerate(dist_pipelines_from_correlated_mean_for_4voxels[1]):
		for i, name in enumerate(clusters_name):
			if team_names[ind] in clusters[i]:
				ax3.plot(dist, numpy.abs(contributions_GLS_for_4_voxels[1][ind]), 'o', markersize=6, color=c[i], alpha=0.5)


	ax4 = plt.subplot2grid((4,3), (2,2), rowspan=1, colspan=1)
	sns.regplot(x=dist_pipelines_from_correlated_mean_for_4voxels[2], y=numpy.abs(contributions_GLS_for_4_voxels[2]),ci=None, ax=ax4, color='lightgrey')
	ax4.set_xlabel("Distance, {}".format(list(four_voxels_nicelly_defined.keys())[2]))
	ax4.set_ylabel("GLS contributions")

	for ind, dist in enumerate(dist_pipelines_from_correlated_mean_for_4voxels[2]):
		for i, name in enumerate(clusters_name):
			if team_names[ind] in clusters[i]:
				ax4.plot(dist, numpy.abs(contributions_GLS_for_4_voxels[2][ind]), 'o', markersize=6, color=c[i], alpha=0.5)

	ax5 = plt.subplot2grid((4,3), (3,2), rowspan=1, colspan=1)
	sns.regplot(x=dist_pipelines_from_correlated_mean_for_4voxels[3], y=numpy.abs(contributions_GLS_for_4_voxels[3]),ci=None, ax=ax5, color='lightgrey')
	ax5.set_xlabel("Distance, {}".format(list(four_voxels_nicelly_defined.keys())[3]))
	ax5.set_ylabel("GLS contributions")
	for ind, dist in enumerate(dist_pipelines_from_correlated_mean_for_4voxels[3]):
		for i, name in enumerate(clusters_name):
			if team_names[ind] in clusters[i]:
				ax5.plot(dist, numpy.abs(contributions_GLS_for_4_voxels[3][ind]), 'o', markersize=6, color=c[i], alpha=0.5)

	plt.tight_layout()
	plt.savefig("{}/{}clusters_results/contribution_per_distance_absolute.png".format(results_dir,len(clusters_name)))



def plot_brains(clusters, clusters_name, team_names, masker, resampled_maps):
	print("Check using the original computation of GLS contributions:")
	T_map, p_map, _ = MA_estimators.GLS_SDMA(resampled_maps)
	T_map_GLS_nii = masker.inverse_transform(T_map)
	p_map = (p_map <= 0.05) * T_map
	p_map_GLS_nii = masker.inverse_transform(p_map)

	T_map, p_map, _ = MA_estimators.SDMA_Stouffer(resampled_maps)
	T_map_SMDA_Stouffer_nii = masker.inverse_transform(T_map)
	p_map = (p_map <= 0.05) * T_map
	p_map_SDMA_Stouffer_nii = masker.inverse_transform(p_map)

	print("getting cluster indices...")
	clusters_indices = get_cluster_indices(clusters, clusters_name, team_names)
	print("get gls weight per pipeline...")
	weight_pipelines_gls = compute_GLS_weights(resampled_maps, std_by_Stouffer=False)
	
	print("get SDMA Stouffer and GLS contributions...")
	contributions_SDMA_Stouffer, contributions_GLS = utils.compute_contributions(resampled_maps, W="SDMA", std_by_Stouffer=False)
	mean_contributions_SDMA_Stouffer_nii = masker.inverse_transform(numpy.mean(contributions_SDMA_Stouffer, axis=0))
	mean_contributions_GLS_nii = masker.inverse_transform(numpy.mean(contributions_GLS, axis=0))

	plt.close('all')
	f, axs = plt.subplots(len(clusters_name) + 4, 4, figsize=(16, len(clusters_name)*5), width_ratios=[0.1, 0.4, 0.4, 0.1])

	# variables for reconstructing global contribution from cluster contributions
	reconstructed_GLS_contributions = 0
	reconstructed_SDMA_Stouffer_contributions = 0

	# plot mean SDMA Stouffer weight for this cluster
	ones = numpy.ones((resampled_maps.shape[0], 1))
	Q = numpy.corrcoef(resampled_maps)
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

		# reconstruct global contribution from cluster contributions
		if row == 0:
			reconstructed_GLS_contributions = this_cluster_contributions_GLS.sum(axis=0)
			reconstructed_SDMA_Stouffer_contributions = this_cluster_contributions_SDMA_Stouffer.sum(axis=0)
		else:
			reconstructed_GLS_contributions += this_cluster_contributions_GLS.sum(axis=0)
			reconstructed_SDMA_Stouffer_contributions += this_cluster_contributions_SDMA_Stouffer.sum(axis=0)

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

	# TO DO: find a way to double check the following line, cheating for now
	print("reconstruct global contribution from cluster contributions")
	reconstructed_sum_GLS_contributions_nii = masker.inverse_transform(reconstructed_GLS_contributions)
	reconstructed_sum_SDMA_Stouffer_contributions_nii = masker.inverse_transform(reconstructed_SDMA_Stouffer_contributions)

	axs[row+2, 3].imshow(numpy.array(weight_pipelines_gls.mean()).reshape(-1, 1), cmap='coolwarm', aspect='equal', vmin=-0.5, vmax=0.5)
	axs[row+2, 3].text(0, 0, numpy.round(weight_pipelines_gls.mean(), 4), ha="center", va="center", color="black")
	axs[row+2, 3].axis('off')


# plot reconstructed brain SDMA Stouffer contribution
	plotting.plot_stat_map(reconstructed_sum_SDMA_Stouffer_contributions_nii,  
		annotate=False,  
		colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), 
		display_mode='z', cmap='coolwarm', axes=axs[row+1, 1])
	axs[row+1, 1].set_title("Sum reconstructed from clusters", size=12)

	# plot reconstructed brain GLS contribution
	plotting.plot_stat_map(reconstructed_sum_GLS_contributions_nii, 
		annotate=False, 
		 colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), 
		display_mode='z', cmap='coolwarm', axes=axs[row+1, 2])
	axs[row+1, 2].set_title("Sum reconstructed from clusters", size=12)

	# plot whole brain SDMA Stouffer contribution computed from utils.compute_contributions
	plotting.plot_stat_map(mean_contributions_SDMA_Stouffer_nii,
		annotate=False,  
		colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), 
		display_mode='z', cmap='coolwarm', axes=axs[row+2, 1])
	axs[row+2, 1].set_title("Mean global contributions", size=12)

	# plot whole brain GLS contribution computed from utils.compute_contributions
	plotting.plot_stat_map(mean_contributions_GLS_nii,
		annotate=False,  
		colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), 
		display_mode='z', cmap='coolwarm', axes=axs[row+2, 2])
	axs[row+2, 2].set_title("Mean global contributions", size=12)

	# plot whole brain SDMA Stouffer contribution computed from MA_estimator
	plotting.plot_stat_map(T_map_SMDA_Stouffer_nii,
		annotate=False,  
		colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), 
		display_mode='z', cmap='coolwarm', axes=axs[row+3, 1])
	axs[row+3, 1].set_title("T map", size=12)

	# plot whole brain GLS contribution computed from MA_estimator
	plotting.plot_stat_map(T_map_GLS_nii,
		annotate=False,  
		colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), 
		display_mode='z', cmap='coolwarm', axes=axs[row+3, 2])
	axs[row+3, 2].set_title("T map", size=12)

	# plot p value significant using SDMA Stouffer
	plotting.plot_stat_map(p_map_SDMA_Stouffer_nii,
		annotate=False, vmax=8,
		colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), 
		display_mode='z', cmap='Reds', axes=axs[row+4, 1])
	axs[row+4, 1].set_title("Significant T values", size=12)

	# plot p value significant using GLS
	plotting.plot_stat_map(p_map_GLS_nii,
		annotate=False, vmax=8,
		colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), 
		display_mode='z', cmap='Reds', axes=axs[row+4, 2])
	axs[row+4, 2].set_title("Significant T values", size=12)


	axs[row+1, 0].axis('off')
	axs[row+2, 0].axis('off')
	axs[row+3, 0].axis('off')
	axs[row+1, 3].axis('off')
	axs[row+3, 3].axis('off')
	axs[row+4, 0].axis('off')
	axs[row+4, 3].axis('off')
	print("Saving plot")



	plt.savefig("{}/{}clusters_results/per_cluster_weights_{}.png".format(results_dir,len(clusters_name), "{}clusters".format(len(clusters_name))))
	plot_distance_from_highly_correlated_cluster(resampled_maps, team_names, results_dir, clusters_name, clusters, eight_voxels_nicelly_defined)
	plot_distance_from_highly_correlated_cluster_absolute(resampled_maps, team_names, results_dir, clusters_name, clusters, eight_voxels_nicelly_defined)

	# plt.show()




def plot_voxels_per_cluster(masker, pipeline_z_scores, team_names, results_dir, clusters_name, clusters, eight_voxels_nicelly_defined):
	print("getting cluster indices...")
	clusters_indices = get_cluster_indices(clusters, clusters_name, team_names)
	print("get gls weight per pipeline...")
	weight_pipelines_gls = compute_GLS_weights(pipeline_z_scores)
	print("get SDMA Stouffer and GLS contributions...")
	contributions_SDMA_Stouffer, contributions_GLS = utils.compute_contributions(pipeline_z_scores, W=1)

	for voxel_significance_profil in eight_voxels_nicelly_defined.keys():
		voxel_index = eight_voxels_nicelly_defined[voxel_significance_profil]
		# plot each voxel:
		fake_ROI = numpy.zeros(pipeline_z_scores.shape[1])
		fake_ROI[voxel_index] = 1
		fake_ROI = masker.inverse_transform(fake_ROI)
	
		data_GLS_for_this_voxel = contributions_GLS[:, voxel_index]
		data_SDMA_Stouffer_for_this_voxel = contributions_SDMA_Stouffer[:, voxel_index]

		print("mean value of GLS contributions for this voxel:", data_SDMA_Stouffer_for_this_voxel.mean())
		print("mean value of SDMA Stouffer contributions for this voxel:", data_GLS_for_this_voxel.mean())
		
		plt.close('all')

		# Create a figure and axis
		fig = plt.figure(figsize=(14, 8))

		# PLOT 1: SDMA STOUFFER CONTRIBUTIONS 
		ax0 = plt.subplot2grid((2,13), (0,0), rowspan=2, colspan=2)
		custom_swarmplot(data_SDMA_Stouffer_for_this_voxel, ax0, team_names, clusters_name, clusters)
		# Customize the plot appearance
		ax0.set_xlabel('SDMA Stouffer')
		ax0.set_ylabel('Contributions SDMA (eq. Z values)')
		ax0.set_xlim([-0.01, 0.02])
		# Remove x-axis tick labels
		ax0.set_xticks([])

		# PLOT 2: GLS CONTRIBUTIONS 
		ax1 = plt.subplot2grid((2,13), (0,2), rowspan=2, colspan=2)
		custom_swarmplot(data_GLS_for_this_voxel, ax1, team_names, clusters_name, clusters)
		# Customize the plot appearance
		ax1.set_xlabel('GLS')
		ax1.set_ylabel('Contributions GLS')
		# ax1.set_title('{}'.format(condition))
		ax1.set_xlim([-0.01, 0.02])
		# Remove x-axis tick labels
		ax1.set_xticks([])

		# PLOT 3: GLS WEIGHTS
		ax2 = plt.subplot2grid((2,13), (0,4), rowspan=2, colspan=2)
		custom_swarmplot(weight_pipelines_gls, ax2, team_names, clusters_name, clusters)
		# Customize the plot appearance
		ax2.set_xlabel('GLS')
		ax2.set_ylabel('Weights GLS')
		# ax1.set_title('{}'.format(condition))
		ax2.set_xlim([-0.01, 0.02])
		# Remove x-axis tick labels
		ax2.set_xticks([])

		# Add legend to plot 3
		# Create Line2D objects with custom colors
		c=['orange', 'blue', 'red', "green", "yellow", "black", "grey", "purple", "lightgreen"]
		lines = []
		for i, name in enumerate(clusters_name):
			lines.append(Patch(color=c[i], label=name))
		ax2.legend(handles=lines, loc='upper right', prop={'size': 6})

		# PLOT 4: CONTRIBUTIONS per methods of all pipelines 
		ax3 = plt.subplot2grid((2,13), (0,6),  rowspan=1, colspan=3)
		i = 0
		for x1, x2 in zip(data_SDMA_Stouffer_for_this_voxel, data_GLS_for_this_voxel):
			for ind, name in enumerate(clusters_name):
				if team_names[i] in clusters[ind]:
					ax3.plot([0], [x1], 'o', color=c[ind])
					ax3.plot([1], [x2], 'o', color=c[ind])
					ax3.plot([0, 1], [x1, x2], '-', color=c[ind], alpha=0.2)
			i+=1
		ax3.set_xlabel('SDMA Stouffer vs GLS)')
		ax3.set_ylabel('Contributions')
		ax3.set_xticks([])


		# PLOT 5: mean pipeline CONTRIBUTION per methods 
		ax4= plt.subplot2grid((2,13), (1,6),  rowspan=1, colspan=3)
		for ind, cluster in enumerate(clusters):
			this_cluster_indices = clusters_indices[clusters_name[ind]]
			SDMA_Stouffer_clust_mean = data_SDMA_Stouffer_for_this_voxel[this_cluster_indices].mean(axis=0)
			GLS_SDMA_clust_mean = data_GLS_for_this_voxel[this_cluster_indices].mean(axis=0)
			ax4.plot([0], [SDMA_Stouffer_clust_mean], 'o', color=c[ind])
			ax4.plot([1], [GLS_SDMA_clust_mean], 'o', color=c[ind])
			ax4.plot([0, 1], [SDMA_Stouffer_clust_mean, GLS_SDMA_clust_mean], '-', color=c[ind], linewidth=2)
		ax4.plot([0], [data_SDMA_Stouffer_for_this_voxel.mean()], 'o', color='grey')
		ax4.plot([1], [data_GLS_for_this_voxel.mean()], 'o', color='grey')
		ax4.plot([0, 1], [data_SDMA_Stouffer_for_this_voxel.mean(), data_GLS_for_this_voxel.mean()], '--', color='grey', linewidth=1)
		ax4.set_xlabel('SDMA Stouffer vs GLS')
		ax4.set_ylabel('Mean contribution')
		ax4.set_xticks([])

		ax5= plt.subplot2grid((2,13), (0,10),  rowspan=2, colspan=3)
		# add visu voxel in brain
		fake_ROI = numpy.zeros(pipeline_z_scores.shape[1])
		fake_ROI[voxel_index] = 1
		fake_ROI = masker.inverse_transform(fake_ROI)
		plotting.plot_stat_map(fake_ROI, annotate=False, vmax=1,colorbar=False, cmap='Blues', axes=ax5)
		ax5.set_title(voxel_significance_profil, size=12)
		plt.tight_layout()
		title = "{}/{}clusters_results/cluster_for_voxel_{}.png".format(results_dir, len(clusters_name), voxel_index)
		plt.savefig(title)
		# plt.show()



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
		correlated_min = min(voxel_values[indices_per_cluster['SPM']])
		anti_correlated_max = max(voxel_values[indices_per_cluster['FSL']])
		correlated_max = max(voxel_values[indices_per_cluster['SPM']])
		anti_correlated_min = min(voxel_values[indices_per_cluster['FSL']])
		if correlated_min >= anti_correlated_max:
			voxels_nicelly_defined_corrup.append(voxel_ind)
		if correlated_max <= anti_correlated_min:
			voxels_nicelly_defined_corrdown.append(voxel_ind)
	print("Found: ", len(voxels_nicelly_defined_corrdown), " nicelly defined voxels corrdown")
	print("and : ", len(voxels_nicelly_defined_corrup), " nicelly defined voxels corrup")
	return voxels_nicelly_defined_corrup, voxels_nicelly_defined_corrdown


def search_for_significant_voxels_within_nicelly_defined_voxel(voxels_nicelly_defined_corrup, voxels_nicelly_defined_corrdown):
	voxels_of_interest_corrup = {'SDMA1_GLS0':[], 'SDMA1_GLS1':[], 'SDMA0_GLS1':[], 'not_significant':[]}
	voxels_of_interest_corrdown = {'SDMA1_GLS0':[], 'SDMA1_GLS1':[], 'SDMA0_GLS1':[], 'not_significant':[]}
	MA_outputs = numpy.load('{}/data/MA_estimates.npy'.format(results_dir),allow_pickle=True).item()
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


############################
# 2 clusters : correlated, independent
############################

random.seed(0)
spm = [element for element in team_names if element.startswith("spm")]
fsl = [element for element in team_names if element.startswith("fsl")]
clusters = [spm, fsl]

clusters_name = ["SPM",
	"FSL"]


# FIND VOXEL WELL DEFINED (ANTICORRELATED below CORRELATED)
voxels_nicelly_defined_corrup, voxels_nicelly_defined_corrdown = search_for_nicelly_defined_voxels(clusters, clusters_name, team_names, resampled_maps)
# FIND VOXEL SIGNICICANCE STATUT
voxels_of_interest_corrup, voxels_of_interest_corrdown = search_for_significant_voxels_within_nicelly_defined_voxel(voxels_nicelly_defined_corrup, voxels_nicelly_defined_corrdown)

random.seed(0)
# PICK ONE VOXEL PER SIGNIFICANCE STATUT
eight_voxels_nicelly_defined = {
			# corr up
			"Not significant corr>anticorr":random.choice(voxels_of_interest_corrup['not_significant']), 
			"SDMA1_GLS0 corr>anticorr":random.choice(voxels_of_interest_corrup['SDMA1_GLS0']), 
			"SDMA0_GLS1 corr>anticorr":random.choice(voxels_of_interest_corrup['SDMA0_GLS1']), 
			"SDMA1_GLS1 corr>anticorr":random.choice(voxels_of_interest_corrup['SDMA1_GLS1']),
			# corr down
			"Not significant corr<anticorr":random.choice(voxels_of_interest_corrdown['not_significant']), 
			"SDMA1_GLS0 corr<anticorr":random.choice(voxels_of_interest_corrdown['SDMA1_GLS0']), 
			"SDMA0_GLS1 corr<anticorr":random.choice(voxels_of_interest_corrdown['SDMA0_GLS1']), 
			"SDMA1_GLS1 corr<anticorr":random.choice(voxels_of_interest_corrdown['SDMA1_GLS1']),
			}


print("plotting 2 clusters")
plot_brains(clusters, clusters_name, team_names, masker, resampled_maps)
plot_voxels_per_cluster(masker, resampled_maps, team_names, results_dir, clusters_name, clusters, eight_voxels_nicelly_defined)

