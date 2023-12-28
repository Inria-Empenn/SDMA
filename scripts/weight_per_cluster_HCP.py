import numpy
import nibabel
import nilearn.datasets as ds
from nilearn import plotting
from nilearn import image
from nilearn.input_data import NiftiMasker
import matplotlib.pyplot as plt
import utils
import importlib
import os
from matplotlib.patches import Patch


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

data_SDMA_Stouffer, data_GLS = utils.compute_weights(resampled_maps)


outputs = numpy.load("{}/data/check_voxels_distribution_outputs.npy".format(results_dir), allow_pickle=True).item()
print("outputs successfully loaded")



def get_nii_per_cluster_team(clusters, clusters_name, resampled_maps, masker, team_names=team_names, data_SDMA_Stouffer=data_SDMA_Stouffer, data_GLS=data_GLS):
	print("Getting NII for each cluster team")
	nii_images = {}
	# compute mean for nii brain
	for ind, cluster in enumerate(clusters):
		cluster_indices = []
		for team in cluster:
			cluster_indices.append(team_names.index(team))
		print(cluster_indices)
		SDMA_Stouffer_cluster = data_SDMA_Stouffer[cluster_indices].mean(axis=0)
		GLS_SDMA_cluster = data_GLS[cluster_indices].mean(axis=0)
		SDMA_Stouffer_nii = masker.inverse_transform(SDMA_Stouffer_cluster)
		GLS_SDMA_nii = masker.inverse_transform(GLS_SDMA_cluster)
		nii_images[clusters_name[ind]] = [[SDMA_Stouffer_nii, GLS_SDMA_nii], resampled_maps[cluster_indices].mean(axis=0)]
	return nii_images




def plot_weight_for_a_voxel(v, voxel_number, clusters_name, results_dir="results_in_Narps_data", condition="_"):
	plt.close("all")
	# Create a figure and axis
	# add visu voxel in brain
	f, ax = plt.subplots(3, 1, figsize=(6, 8))
	c=['orange', 'blue', 'red', "green", "yellow", "black", "grey", "purple", "cian"]
	for ind, cluster in enumerate(clusters_name):
		ax[0].plot([0, 1], [numpy.mean(v[cluster][0]), numpy.mean(v[cluster][1])], marker='o', linestyle='dashed', label=cluster, color=c[ind])
		ax[1].hist(v[cluster][0], bins=50, color=c[ind], alpha=0.7)
		ax[2].hist(v[cluster][1], bins=50, color=c[ind], alpha=0.7)
	ax[1].set_xlabel('SStouffer')
	ax[2].set_xlabel('GLS')
	ax[0].set_xticks([0, 1], labels=["SDMA", "GLS"])
	ax[0].set_ylabel('Weight * raw Z')
	ax[0].legend()
	plt.suptitle("Voxel {} {}".format(voxel_number, condition))
	plt.tight_layout()	
	plt.savefig("{}/{}clusters_results/simple_plot_per_cluster_{}clusters_{}_{}.png".format(results_dir, len(clusters_name),len(clusters_name), voxel_number, condition))


def custom_swarmplot(data, ax, team_names, clusters_name, clusters):
	c=['orange', 'blue', 'red', "green", "yellow", "black", "grey", "purple", "lightgreen"]
	for i, value in enumerate(data):
		for ind, name in enumerate(clusters_name):
			if team_names[i] in clusters[ind]:
				ax.plot(0, value, 'o', markersize=6, alpha=0)
				ax.text(0, value, team_names[i], color=c[ind], fontsize=6, ha='center', va='bottom')
				continue


def plot_voxels_per_cluster(masker, voxel_index, resampled_maps, data_GLS, data_SDMA_Stouffer, condition, team_names, results_dir, clusters_name, clusters):
	plt.close('all')
	# plot each voxel:
	fake_ROI = numpy.zeros(resampled_maps.shape[1])
	fake_ROI[voxel_index] = 1
	fake_ROI = masker.inverse_transform(fake_ROI)
	specific_voxel_Z = resampled_maps[:, voxel_index]
	data_GLS = data_GLS[:, voxel_index]
	data_SDMA_Stouffer = data_SDMA_Stouffer[:, voxel_index]

	print("GLS:", data_SDMA_Stouffer.mean())
	print("SStouffer:", data_GLS.mean())
	
	plt.close('all')
	# Create a figure and axis
	fig = plt.figure(figsize=(10, 8))
	ax0 = plt.subplot2grid((2,7), (0,0), rowspan=2, colspan=2)
	custom_swarmplot(data_SDMA_Stouffer, ax0, team_names, clusters_name, clusters)
	# Customize the plot appearance
	ax0.set_xlabel('Data Point \nSDMA Stouffer')
	ax0.set_ylabel('Z Value')
	ax0.set_xlim([-0.01, 0.02])
	# Remove x-axis tick labels
	ax0.set_xticks([])


	ax1 = plt.subplot2grid((2,7), (0,2), rowspan=2, colspan=2)
	custom_swarmplot(data_GLS, ax1, team_names, clusters_name, clusters)
	# Customize the plot appearance
	ax1.set_xlabel('Data Point \nGLS')
	ax1.set_ylabel('Z Value')
	ax1.set_title('{}'.format(condition))
	ax1.set_xlim([-0.01, 0.02])
	# Remove x-axis tick labels
	ax1.set_xticks([])

	# Create Line2D objects with custom colors
	c=['orange', 'blue', 'red', "green", "yellow", "black", "grey", "purple", "lightgreen"]
	lines = []
	for i, name in enumerate(clusters_name):
		lines.append(Patch(color=c[i], label=name))
	ax1.legend(handles=lines, loc='upper right', prop={'size': 6})


	ax3 = plt.subplot2grid((2,7), (0,4),  colspan=3)
	i = 0
	for x1, x2 in zip(data_SDMA_Stouffer, data_GLS):
		for ind, name in enumerate(clusters_name):
			if team_names[i] in clusters[ind]:
				ax3.plot([0], [x1], 'o', color=c[ind])
				ax3.plot([1], [x2], 'o', color=c[ind])
				ax3.plot([0, 1], [x1, x2], '-', color=c[ind], alpha=0.2)
		i+=1

	# add mean voxel contribution
	ax4= plt.subplot2grid((2,7), (1,4),  colspan=3)
	for ind, cluster in enumerate(clusters):
		cluster_indices = []
		for team in cluster:
			cluster_indices.append(team_names.index(team))
		SDMA_Stouffer_clust_mean = data_SDMA_Stouffer[cluster_indices].mean(axis=0)
		GLS_SDMA_clust_mean = data_GLS[cluster_indices].mean(axis=0)
		ax4.plot([0, 1], [SDMA_Stouffer_clust_mean, GLS_SDMA_clust_mean], '-', color=c[ind], linewidth=2)

	# add visu voxel in brain
	# ax4= plt.subplot2grid((2,7), (1,4),  colspan=3)
	# plotting.plot_stat_map(fake_ROI, annotate=False, vmax=1, colorbar=False, cmap='Blues', axes=ax4)
	plt.tight_layout()
	title = "{}/{}clusters_results/{}_cluster_for_voxel_{}.png".format(results_dir,len(clusters_name), condition, voxel_index)
	plt.savefig(title)
	# plt.show()


def plot_brains(nii_images, clusters_name, title, masker):
	print("Plotting results")
	plt.close('all')
	f, axs = plt.subplots(len(nii_images.keys()), 3, figsize=(16, len(nii_images.keys())*2), width_ratios=[0.4, 0.4, 0.2])

	for row, name in enumerate(clusters_name):
		dist = nii_images[name][1]
		# We can set the number of bins with the *bins* keyword argument.
		axs[row, 2].hist(dist, bins=50)
		axs[row, 2].vlines(nii_images[name][1].mean(),0, 70000, colors="Red", linestyles='dashed')
		# axs[row, 2].plot(numpy.ones(len(nii_images[name][1])), nii_images[name][1], '.')		# return mean of raw values to check variability of cluster
		axs[row, 2].set_xlabel('Mean voxel values')
		if "independant_c" in name:
			plotting.plot_stat_map(nii_images[name][0][0], vmax=0.5, annotate=False, title=name+"+SDMA_Stouffer", threshold=0.1, colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap='coolwarm', axes=axs[row, 0])
			plotting.plot_stat_map(nii_images[name][0][1], vmax=0.5, annotate=False, title=name+"+GLS", threshold=0.1, colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap='coolwarm', axes=axs[row, 1])
		elif "independant" in name:
			plotting.plot_stat_map(nii_images[name][0][0], vmax=0.5, annotate=False, title=name+"+SDMA_Stouffer", threshold=0.1, colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap='coolwarm', axes=axs[row, 0])
			plotting.plot_stat_map(nii_images[name][0][1], vmax=0.5, annotate=False, title=name+"+GLS", threshold=0.1, colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap='coolwarm', axes=axs[row, 1])
		elif "slightly_correlated_c2" in name:
			plotting.plot_stat_map(nii_images[name][0][0], vmax=0.5, annotate=False, title=name+"+SDMA_Stouffer", threshold=0.1, colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap='coolwarm', axes=axs[row, 0])
			plotting.plot_stat_map(nii_images[name][0][1], vmax=0.5, annotate=False, title=name+"+GLS", threshold=0.1, colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap='coolwarm', axes=axs[row, 1])
		elif "slightly_correlated" in name:
			plotting.plot_stat_map(nii_images[name][0][0], vmax=0.5, annotate=False, title=name+"+SDMA_Stouffer", threshold=0.1, colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap='coolwarm', axes=axs[row, 0])
			plotting.plot_stat_map(nii_images[name][0][1], vmax=0.5, annotate=False, title=name+"+GLS", threshold=0.1, colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap='coolwarm', axes=axs[row, 1])
		elif "anti_correlated_c1" in name:
			plotting.plot_stat_map(nii_images[name][0][0], vmax=0.5, annotate=False, title=name+"+SDMA_Stouffer", threshold=0.1, colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap='coolwarm', axes=axs[row, 0])
			plotting.plot_stat_map(nii_images[name][0][1], vmax=0.5, annotate=False, title=name+"+GLS", threshold=0.1, colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap='coolwarm', axes=axs[row, 1])	
		elif "anti_correlated" in name:
			plotting.plot_stat_map(nii_images[name][0][0], vmax=0.5, annotate=False, title=name+"+SDMA_Stouffer", threshold=0.1, colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap='coolwarm', axes=axs[row, 0])
			plotting.plot_stat_map(nii_images[name][0][1], vmax=0.5, annotate=False, title=name+"+GLS", threshold=0.1, colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap='coolwarm', axes=axs[row, 1])
		elif "highly_correlated_c1" in name:
			plotting.plot_stat_map(nii_images[name][0][0], vmax=0.5, annotate=False, title=name+"+SDMA_Stouffer", threshold=0.1, colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap='coolwarm', axes=axs[row, 0])
			plotting.plot_stat_map(nii_images[name][0][1], vmax=0.5, annotate=False, title=name+"+GLS", threshold=0.1, colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap='coolwarm', axes=axs[row, 1])
		elif "highly_correlated_c3" in name:
			plotting.plot_stat_map(nii_images[name][0][0], vmax=0.5, annotate=False, title=name+"+SDMA_Stouffer", threshold=0.1, colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap='coolwarm', axes=axs[row, 0])
			plotting.plot_stat_map(nii_images[name][0][1], vmax=0.5, annotate=False, title=name+"+GLS", threshold=0.1, colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap='coolwarm', axes=axs[row, 1])
		else: #highly_correlated_c2
			plotting.plot_stat_map(nii_images[name][0][0], vmax=0.5, annotate=False, title=name+"+SDMA_Stouffer", threshold=0.1, colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap='coolwarm', axes=axs[row, 0])
			plotting.plot_stat_map(nii_images[name][0][1], vmax=0.5, annotate=False, title=name+"+GLS", threshold=0.1, colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap='coolwarm', axes=axs[row, 1])

	plt.savefig("{}/{}clusters_results/per_cluster_weights_{}.png".format(results_dir,len(clusters_name), title))
	# plt.show()







############################
# 2 clusters : correlated, independent
############################


spm = [element for element in team_names if element.startswith("spm")]
fsl = [element for element in team_names if element.startswith("fsl")]
clusters = [spm, fsl]

clusters_name = ["SPM",
	"FSL"]

nii_images = get_nii_per_cluster_team(clusters, clusters_name, resampled_maps, masker)
print("plotting 2 clusters")
plot_brains(nii_images, clusters_name, "2clusters", masker)


conditions = ["consensus0_gls1", "consensus1_gls0", "consensus1_gls1"]
for condition in [0, 1, 2]:
	for voxel in [0, -1]:
		v_ind = outputs[conditions[condition]][voxel]
		print("running condition {}, voxel {}".format(conditions[condition], v_ind))
		plot_voxels_per_cluster(masker, v_ind, resampled_maps, data_GLS, data_SDMA_Stouffer, conditions[condition], team_names, results_dir, clusters_name, clusters)
