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


importlib.reload(utils)

participant_mask = nibabel.load("masking/mask_90.nii")
# fit masker
masker = NiftiMasker(
    mask_img=participant_mask)
results_dir = "results_in_Narps_data"

hyp = 1
for c in range(2, 10):
	if not os.path.exists(os.path.join(results_dir, "hyp{}/{}clusters_results".format(hyp, c))):
		os.mkdir(os.path.join(results_dir, "hyp{}/{}clusters_results".format(hyp, c)))

resampled_maps_per_team = numpy.load('{}/data/Hyp{}_resampled_maps.npy'.format(results_dir, hyp), allow_pickle=True).item()
resampled_maps= masker.fit_transform(resampled_maps_per_team.values())
team_names = list(resampled_maps_per_team.keys())
data_SDMA_Stouffer, data_GLS = utils.compute_weights(resampled_maps)

try:
    outputs = numpy.load("{}/data/check_voxels_distribution_outputs.npy".format(results_dir), allow_pickle=True).item()
    print("outputs successfully loaded")
except:
	print("recharging outputs")


def get_full_name(endings):
	# returns a list of full team name given a list of team name endings
	full_name = []
	for team_name_ending in endings:
		found = 0
		for team_name in resampled_maps_per_team.keys():
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



def get_nii_per_cluster_team(clusters, clusters_name, resampled_maps, masker, team_names=team_names, data_SDMA_Stouffer=data_SDMA_Stouffer, data_GLS=data_GLS):
	print("Getting NII for each cluster team")
	nii_images = {}
	# compute mean for nii brain
	ones = numpy.ones((resampled_maps.shape[0], 1))
	Q = numpy.corrcoef(resampled_maps)
	W_sdma = (ones.T.dot(Q).dot(ones))**(-1/2) # scalar
	Q_inv = numpy.linalg.inv(Q)
	weights_pipeline_gls = numpy.sum(Q_inv, axis=1) * (ones.T.dot(Q_inv).dot(ones))**(-1/2)
	weights_pipeline_gls = weights_pipeline_gls.reshape(-1)
	for ind, cluster in enumerate(clusters):
		cluster_indices = []
		for team in cluster:
			cluster_indices.append(team_names.index(team))
		print("***********")
		print(weights_pipeline_gls.shape)
		print(clusters_name[ind])
		print("nb pipe in this group: ", len(cluster_indices))
		print("Resultat Camille:")
		print("value: ", weights_pipeline_gls[cluster_indices].sum())
		print("raw values :")
		print(weights_pipeline_gls[cluster_indices])
		print("***********")

		SDMA_Stouffer_cluster = data_SDMA_Stouffer[cluster_indices].mean(axis=0)
		GLS_SDMA_cluster = data_GLS[cluster_indices].mean(axis=0)
		SDMA_Stouffer_nii = masker.inverse_transform(SDMA_Stouffer_cluster)
		GLS_SDMA_nii = masker.inverse_transform(GLS_SDMA_cluster)
		nii_images[clusters_name[ind]] = [[SDMA_Stouffer_nii, GLS_SDMA_nii], resampled_maps[cluster_indices].mean(axis=0)]
	return nii_images

### AJOUT HEATMAP POID DE CAMILLE PRES DU RESLTAT CERVEAU CLUSTER
### CHECK VOXEL AVEC VALEUR UN PEU PRES IDENTIQUE PAR CLUSTER (plutot que significatif ou non)
### est ce automatisable ? (genre de voxel comme sur figure avec voxel unique: on voit independant, correlé, anticorrele bien separé)



def get_nii_per_cluster_team_verification(clusters, clusters_name, resampled_maps, masker, team_names=team_names, data_SDMA_Stouffer=data_SDMA_Stouffer, data_GLS=data_GLS):
	print("Getting NII for each cluster team")
	nii_images = {}
	# compute mean for nii brain
	cluster_contribs_sdma, cluster_contribs_gls  = [], []
	ones = numpy.ones((resampled_maps.shape[0], 1))
	Q = numpy.corrcoef(resampled_maps)
	W_sdma = (ones.T.dot(Q).dot(ones))**(-1/2) # scalar

	for ind, cluster in enumerate(clusters):
		cluster_indices = []
		for team in cluster:
			cluster_indices.append(team_names.index(team))
		print(cluster_indices)
		SDMA_Stouffer_cluster = data_SDMA_Stouffer[cluster_indices] * W_sdma
		GLS_SDMA_cluster = data_GLS[cluster_indices] * W_sdma
		cluster_contribs_sdma.append(SDMA_Stouffer_cluster)
		cluster_contribs_gls.append(GLS_SDMA_cluster)
	cluster_contribs_sdma_concat = numpy.concatenate((cluster_contribs_sdma))
	cluster_contribs_gls_concat = numpy.concatenate((cluster_contribs_gls))
	sdma_all_pipelines = numpy.sum(cluster_contribs_sdma_concat, axis=0) 
	gls_all_pipelines = numpy.sum(cluster_contribs_gls_concat, axis=0)
	SDMA_Stouffer_allpipelines_nii = masker.inverse_transform(sdma_all_pipelines)
	GLS_SDMA_allpipelines_nii = masker.inverse_transform(gls_all_pipelines)
	nii_images = [SDMA_Stouffer_allpipelines_nii, GLS_SDMA_allpipelines_nii]
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


def plot_voxels_per_cluster(masker, voxel_index, resampled_maps, data_GLS, data_SDMA_Stouffer, condition, team_names, hyp, results_dir, clusters_name, clusters):
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


	# # add visu voxel in brain
	# ax4= plt.subplot2grid((2,7), (1,4),  colspan=3)
	# plotting.plot_stat_map(fake_ROI, annotate=False, vmax=1, colorbar=False, cmap='Blues', axes=ax4)
	plt.tight_layout()
	title = "{}/hyp1/{}clusters_results/{}_cluster_for_voxel_{}.png".format(results_dir,len(clusters_name), condition, voxel_index)
	plt.savefig(title)
	# plt.show()


def plot_brains(nii_images, nii_reconstructed, clusters_name, title, masker):
	print("Plotting results")
	plt.close('all')
	f, axs = plt.subplots(len(nii_images.keys()) + 1, 3, figsize=(16, len(nii_images.keys())*3), width_ratios=[0.4, 0.4, 0.2])

	for row, name in enumerate(clusters_name):
		dist = nii_images[name][1]
		# We can set the number of bins with the *bins* keyword argument.
		axs[row, 2].hist(dist, bins=50)
		axs[row, 2].vlines(nii_images[name][1].mean(),0, 70000, colors="Red", linestyles='dashed')
		# axs[row, 2].plot(numpy.ones(len(nii_images[name][1])), nii_images[name][1], '.')		# return mean of raw values to check variability of cluster
		axs[row, 2].set_xlabel('Mean voxel values')
		if "independant_c" in name:
			plotting.plot_stat_map(nii_images[name][0][0],  annotate=False, title=name+"+SDMA_Stouffer", threshold=0.1, colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap='coolwarm', axes=axs[row, 0])
			plotting.plot_stat_map(nii_images[name][0][1],  annotate=False, title=name+"+GLS", threshold=0.1, colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap='coolwarm', axes=axs[row, 1])
		elif "independant" in name:
			plotting.plot_stat_map(nii_images[name][0][0],  annotate=False, title=name+"+SDMA_Stouffer", threshold=0.1, colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap='coolwarm', axes=axs[row, 0])
			plotting.plot_stat_map(nii_images[name][0][1],  annotate=False, title=name+"+GLS", threshold=0.1, colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap='coolwarm', axes=axs[row, 1])
		elif "slightly_correlated_c2" in name:
			plotting.plot_stat_map(nii_images[name][0][0],  annotate=False, title=name+"+SDMA_Stouffer", threshold=0.1, colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap='coolwarm', axes=axs[row, 0])
			plotting.plot_stat_map(nii_images[name][0][1],  annotate=False, title=name+"+GLS", threshold=0.1, colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap='coolwarm', axes=axs[row, 1])
		elif "slightly_correlated" in name:
			plotting.plot_stat_map(nii_images[name][0][0],  annotate=False, title=name+"+SDMA_Stouffer", threshold=0.1, colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap='coolwarm', axes=axs[row, 0])
			plotting.plot_stat_map(nii_images[name][0][1],  annotate=False, title=name+"+GLS", threshold=0.1, colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap='coolwarm', axes=axs[row, 1])
		elif "anti_correlated_c1" in name:
			plotting.plot_stat_map(nii_images[name][0][0],  annotate=False, title=name+"+SDMA_Stouffer", threshold=0.1, colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap='coolwarm', axes=axs[row, 0])
			plotting.plot_stat_map(nii_images[name][0][1],  annotate=False, title=name+"+GLS", threshold=0.1, colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap='coolwarm', axes=axs[row, 1])	
		elif "anti_correlated" in name:
			plotting.plot_stat_map(nii_images[name][0][0],  annotate=False, title=name+"+SDMA_Stouffer", threshold=0.1, colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap='coolwarm', axes=axs[row, 0])
			plotting.plot_stat_map(nii_images[name][0][1],  annotate=False, title=name+"+GLS", threshold=0.1, colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap='coolwarm', axes=axs[row, 1])
		elif "highly_correlated_c1" in name:
			plotting.plot_stat_map(nii_images[name][0][0],  annotate=False, title=name+"+SDMA_Stouffer", threshold=0.1, colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap='coolwarm', axes=axs[row, 0])
			plotting.plot_stat_map(nii_images[name][0][1],  annotate=False, title=name+"+GLS", threshold=0.1, colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap='coolwarm', axes=axs[row, 1])
		elif "highly_correlated_c3" in name:
			plotting.plot_stat_map(nii_images[name][0][0],  annotate=False, title=name+"+SDMA_Stouffer", threshold=0.1, colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap='coolwarm', axes=axs[row, 0])
			plotting.plot_stat_map(nii_images[name][0][1],  annotate=False, title=name+"+GLS", threshold=0.1, colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap='coolwarm', axes=axs[row, 1])
		else: #highly_correlated_c2
			plotting.plot_stat_map(nii_images[name][0][0],  annotate=False, title=name+"+SDMA_Stouffer", threshold=0.1, colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap='coolwarm', axes=axs[row, 0])
			plotting.plot_stat_map(nii_images[name][0][1],  annotate=False, title=name+"+GLS", threshold=0.1, colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap='coolwarm', axes=axs[row, 1])

	plotting.plot_stat_map(nii_reconstructed[0],  annotate=False, title="reconstructed SDMA_Stouffer", threshold=0.1, colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap='coolwarm', axes=axs[row+1, 0])
	plotting.plot_stat_map(nii_reconstructed[1],  annotate=False, title="reconstructed GLS", threshold=0.1, colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap='coolwarm', axes=axs[row+1, 1])
	plt.savefig("{}/hyp1/{}clusters_results/per_cluster_weights_{}.png".format(results_dir,len(clusters_name), title))
	# plt.show()







############################
# 2 clusters : correlated, independent
############################

## get mean weight per cluster of teams 
## define cluster of team from Narps paper figure 2
correlated = ["AO86", "43FJ", "O21U", "3PQ2", "0JO0", "I9D6", "51PW", "94GU", "0ED6", "R5K7", "SM54", "B23O",
				"O03M", "DC61", "X1Y5", "UI76", "2T7P", "2T6S", "27SS", "T54A", "1KB2", "08MQ", "V55J",
				"3TR7", "Q6O0", "E3B6", "L7J7", "K9P0", "X19V", "9Q6R", "U26C", "50GV", "B5I6", "R9K3", "C88N", 
				"J7F9", "46CD", "C22U", "I52Y", "E6R3", "R7D1", "0C7Q", "6VV2", "98BT", "6FH5", "3C6G", "L3V8", "0I4U",
				"0H5E", "9U7M", "80GC", "1P0Y", "P5F3", "IZ20", "Q58J", "4TQ6", 'UK24']
independant = ["9T8E", "R42Q", "XU70", "L9G5", "O6R6", "4SZ2", "I07H"]

print("Getting full team names")
correlated_full_name = get_full_name(correlated)
independant_full_name = get_full_name(independant)

clusters = [correlated_full_name,
	independant_full_name]

clusters_name = ["correlated",
	"independant"]

nii_images = get_nii_per_cluster_team(clusters, clusters_name, resampled_maps, masker)
nii_contrib = get_nii_per_cluster_team_verification(clusters, clusters_name, resampled_maps, masker, team_names=team_names, data_SDMA_Stouffer=data_SDMA_Stouffer, data_GLS=data_GLS)
print("plotting 2 clusters")
plot_brains(nii_images, nii_contrib, clusters_name, "2clusters", masker)

conditions = ["consensus0_gls1", "consensus1_gls0", "consensus1_gls1"]
for condition in [0, 1, 2]:
	# # for voxel in [0, -1]:
	# v_ind = 0
	# # v_ind = outputs[hyp][condition][voxel]
	# print("running hyp {}, condition {}, voxel {}".format(hyp, conditions[condition], v_ind))
	# plot_voxels_per_cluster(masker, v_ind, resampled_maps, data_GLS, data_SDMA_Stouffer, conditions[condition], team_names, hyp, results_dir, clusters_name, clusters)
	for voxel in [0, -1]:
		v_ind = outputs[hyp][condition][voxel]
		print("running hyp {}, condition {}, voxel {}".format(hyp, conditions[condition], v_ind))
		plot_voxels_per_cluster(masker, v_ind, resampled_maps, data_GLS, data_SDMA_Stouffer, conditions[condition], team_names, hyp, results_dir, clusters_name, clusters)

############################
# 3 clusters : correlated, anti-correlated, independent
############################

## get mean weight per cluster of teams 
## define cluster of team from Narps paper figure 2
correlated = ["AO86", "43FJ", "O21U", "3PQ2", "0JO0", "I9D6", "51PW", "94GU", "0ED6", "R5K7", "SM54", "B23O",
				"O03M","DC61", "X1Y5", "UI76", "2T7P", "2T6S", "27SS", "T54A", "1KB2", "08MQ", "V55J",
				"3TR7", "Q6O0", "E3B6", "L7J7", "K9P0", "X19V", "9Q6R", "U26C", "50GV", "B5I6", "R9K3", "C88N", 
				"J7F9", "46CD", "C22U", "I52Y", "E6R3", "R7D1", "0C7Q", "6VV2", "98BT", "6FH5", "3C6G", "L3V8", "0I4U",
				"0H5E", "9U7M"]
anti_correlated = ["80GC", "1P0Y", "P5F3", "IZ20", "Q58J", "4TQ6", 'UK24']
independant = ["9T8E", "R42Q", "XU70", "L9G5", "O6R6", "4SZ2", "I07H"]

print("Getting full team names")
correlated_full_name = get_full_name(correlated)
anti_correlated_full_name = get_full_name(anti_correlated)
independant_full_name = get_full_name(independant)

clusters = [correlated_full_name,
anti_correlated_full_name,
independant_full_name]

clusters_name = ["correlated",
"anti_correlated",
"independant"]

nii_images = get_nii_per_cluster_team(clusters, clusters_name,resampled_maps, masker)
plot_brains(nii_images, clusters_name, "3clusters", masker)

conditions = ["consensus0_gls1", "consensus1_gls0", "consensus1_gls1"]
for condition in [0, 1, 2]:
	# # for voxel in [0, -1]:
	# v_ind = 0
	# # v_ind = outputs[hyp][condition][voxel]
	# print("running hyp {}, condition {}, voxel {}".format(hyp, conditions[condition], v_ind))
	# plot_voxels_per_cluster(masker, v_ind, resampled_maps, data_GLS, data_SDMA_Stouffer, conditions[condition], team_names, hyp, results_dir, clusters_name, clusters)
	for voxel in [0, -1]:
		v_ind = outputs[hyp][condition][voxel]
		print("running hyp {}, condition {}, voxel {}".format(hyp, conditions[condition], v_ind))
		plot_voxels_per_cluster(masker, v_ind, resampled_maps, data_GLS, data_SDMA_Stouffer, conditions[condition], team_names, hyp, results_dir, clusters_name, clusters)

############################
# 4 clusters : slightly, highly, anti, independent
############################

## get mean weight per cluster of teams 
## define cluster of team from Narps paper figure 2
slightly_correlated = ["AO86", "43FJ", "O21U", "3PQ2", "0JO0", "I9D6", "51PW", "94GU", "0ED6", "R5K7", "SM54", "B23O",
				"O03M"]
highly_correlated = ["DC61", "X1Y5", "UI76", "2T7P", "2T6S", "27SS", "T54A", "1KB2", "08MQ", "V55J",
				"3TR7", "Q6O0", "E3B6", "L7J7", "K9P0", "X19V", "9Q6R", "U26C", "50GV", "B5I6", "R9K3", "C88N", 
				"J7F9", "46CD", "C22U", "I52Y", "E6R3", "R7D1", "0C7Q", "6VV2", "98BT", "6FH5", "3C6G", "L3V8", "0I4U",
				"0H5E", "9U7M"]
anti_correlated = ["80GC", "1P0Y", "P5F3", "IZ20", "Q58J", "4TQ6", 'UK24']
independant = ["9T8E", "R42Q", "XU70", "L9G5", "O6R6", "4SZ2", "I07H"]

print("Getting full team names")
slightly_correlated_full_name = get_full_name(slightly_correlated)
highly_correlated_full_name = get_full_name(highly_correlated)
anti_correlated_full_name = get_full_name(anti_correlated)
independant_full_name = get_full_name(independant)

clusters = [slightly_correlated_full_name,
highly_correlated_full_name,
anti_correlated_full_name,
independant_full_name]

clusters_name = ["slightly_correlated",
"highly_correlated",
"anti_correlated",
"independant"]

nii_images = get_nii_per_cluster_team(clusters, clusters_name, resampled_maps, masker)
plot_brains(nii_images, clusters_name, "4clusters", masker)

conditions = ["consensus0_gls1", "consensus1_gls0", "consensus1_gls1"]
for condition in [0, 1, 2]:
	# # for voxel in [0, -1]:
	# v_ind = 0
	# # v_ind = outputs[hyp][condition][voxel]
	# print("running hyp {}, condition {}, voxel {}".format(hyp, conditions[condition], v_ind))
	# plot_voxels_per_cluster(masker, v_ind, resampled_maps, data_GLS, data_SDMA_Stouffer, conditions[condition], team_names, hyp, results_dir, clusters_name, clusters)

	for voxel in [0, -1]:
		v_ind = outputs[hyp][condition][voxel]
		print("running hyp {}, condition {}, voxel {}".format(hyp, conditions[condition], v_ind))
		plot_voxels_per_cluster(masker, v_ind, resampled_maps, data_GLS, data_SDMA_Stouffer, conditions[condition], team_names, hyp, results_dir, clusters_name, clusters)

############################
# 5 clusters : slightly, highly_c1, highly_c2, anti, independent
############################

## get mean weight per cluster of teams 
## define cluster of team from Narps paper figure 2
slightly_correlated = ["AO86", "43FJ", "O21U", "3PQ2", "0JO0", "I9D6", "51PW", "94GU", "0ED6", "R5K7", "SM54", "B23O",
				"O03M"]
highly_correlated_c1 = ["DC61", "X1Y5", "UI76", "2T7P", "2T6S", "27SS", "T54A"]
highly_correlated_c2 = ["1KB2", "08MQ", "V55J", "3TR7", "Q6O0", "E3B6", "L7J7", "K9P0", "X19V", "9Q6R", "U26C", "50GV", "B5I6", "R9K3", "C88N", 
				"J7F9", "46CD", "C22U", "I52Y", "E6R3", "R7D1", "0C7Q", "6VV2", "98BT", "6FH5", "3C6G", "L3V8", "0I4U",
				"0H5E", "9U7M"]
anti_correlated = ["80GC", "1P0Y", "P5F3", "IZ20", "Q58J", "4TQ6", 'UK24']
independant = ["9T8E", "R42Q", "XU70", "L9G5", "O6R6", "4SZ2", "I07H"]


print("Getting full team names")
slightly_correlated_full_name = get_full_name(slightly_correlated)
highly_correlated_c1_full_name = get_full_name(highly_correlated_c1)
highly_correlated_c2_full_name = get_full_name(highly_correlated_c2)
anti_correlated_full_name = get_full_name(anti_correlated)
independant_full_name = get_full_name(independant)

clusters = [slightly_correlated_full_name,
highly_correlated_c1_full_name,
highly_correlated_c2_full_name,
anti_correlated_full_name,
independant_full_name]

clusters_name = ["slightly_correlated",
"highly_correlated_c1",
"highly_correlated_c2",
"anti_correlated",
"independant"]

nii_images = get_nii_per_cluster_team(clusters, clusters_name,resampled_maps, masker)
plot_brains(nii_images, clusters_name, "5clusters", masker)


conditions = ["consensus0_gls1", "consensus1_gls0", "consensus1_gls1"]
for condition in [0, 1, 2]:
	for voxel in [0, -1]:
		v_ind = outputs[hyp][condition][voxel]
		print("running hyp {}, condition {}, voxel {}".format(hyp, conditions[condition], v_ind))
		plot_voxels_per_cluster(masker, v_ind, resampled_maps, data_GLS, data_SDMA_Stouffer, conditions[condition], team_names, hyp, results_dir, clusters_name, clusters)

############################
# 6 clusters : slightly_c1, slightly_c2, highly_c1, highly_c2, anti, independent
############################

## get mean weight per cluster of teams 
## define cluster of team from Narps paper figure 2
slightly_correlated_c1 = ["AO86", "43FJ", "O21U", "3PQ2", "0JO0", "I9D6", "51PW", "94GU"]
slightly_correlated_c2 = ["0ED6", "R5K7", "SM54", "B23O", "O03M"]
highly_correlated_c1 = ["DC61", "X1Y5", "UI76", "2T7P", "2T6S", "27SS", "T54A"]
highly_correlated_c2 = ["1KB2", "08MQ", "V55J", "3TR7", "Q6O0", "E3B6", "L7J7", "K9P0", "X19V", "9Q6R", "U26C", "50GV", "B5I6", "R9K3", "C88N", 
				"J7F9", "46CD", "C22U", "I52Y", "E6R3", "R7D1", "0C7Q", "6VV2", "98BT", "6FH5", "3C6G", "L3V8", "0I4U",
				"0H5E", "9U7M"]
anti_correlated = ["80GC", "1P0Y", "P5F3", "IZ20", "Q58J", "4TQ6", 'UK24']
independant = ["9T8E", "R42Q", "XU70", "L9G5", "O6R6", "4SZ2", "I07H"]


print("Getting full team names")
slightly_correlated_c1_full_name = get_full_name(slightly_correlated_c1)
slightly_correlated_c2_full_name = get_full_name(slightly_correlated_c2)
highly_correlated_c1_full_name = get_full_name(highly_correlated_c1)
highly_correlated_c2_full_name = get_full_name(highly_correlated_c2)
anti_correlated_full_name = get_full_name(anti_correlated)
independant_full_name = get_full_name(independant)

clusters = [slightly_correlated_c1_full_name,
slightly_correlated_c2_full_name,
highly_correlated_c1_full_name,
highly_correlated_c2_full_name,
anti_correlated_full_name,
independant_full_name]

clusters_name = ["slightly_correlated_c1",
"slightly_correlated_c2",
"highly_correlated_c1",
"highly_correlated_c2",
"anti_correlated",
"independant"]

nii_images = get_nii_per_cluster_team(clusters, clusters_name,resampled_maps, masker)
plot_brains(nii_images, clusters_name, "6clusters", masker)


conditions = ["consensus0_gls1", "consensus1_gls0", "consensus1_gls1"]
for condition in [0, 1, 2]:
	for voxel in [0, -1]:
		v_ind = outputs[hyp][condition][voxel]
		print("running hyp {}, condition {}, voxel {}".format(hyp, conditions[condition], v_ind))
		plot_voxels_per_cluster(masker, v_ind, resampled_maps, data_GLS, data_SDMA_Stouffer, conditions[condition], team_names, hyp, results_dir, clusters_name, clusters)

############################
# 7 clusters : slightly_c1, slightly_c2, highly_c1, highly_c2, anti_c1, anti_c2, independent
############################

## get mean weight per cluster of teams 
## define cluster of team from Narps paper figure 2
slightly_correlated_c1 = ["AO86", "43FJ", "O21U", "3PQ2", "0JO0", "I9D6", "51PW", "94GU"]
slightly_correlated_c2 = ["0ED6", "R5K7", "SM54", "B23O", "O03M"]
highly_correlated_c1 = ["DC61", "X1Y5", "UI76", "2T7P", "2T6S", "27SS", "T54A"]
highly_correlated_c2 = ["1KB2", "08MQ", "V55J", "3TR7", "Q6O0", "E3B6", "L7J7", "K9P0", "X19V", "9Q6R", "U26C", "50GV", "B5I6", "R9K3", "C88N", 
				"J7F9", "46CD", "C22U", "I52Y", "E6R3", "R7D1", "0C7Q", "6VV2", "98BT", "6FH5", "3C6G", "L3V8", "0I4U",
				"0H5E", "9U7M"]
anti_correlated_c1 = ["80GC", "1P0Y", "P5F3"]
anti_correlated_c2 = ["IZ20", "Q58J", "4TQ6", 'UK24']
independant = ["9T8E", "R42Q", "XU70", "L9G5", "O6R6", "4SZ2", "I07H"]


print("Getting full team names")
slightly_correlated_c1_full_name = get_full_name(slightly_correlated_c1)
slightly_correlated_c2_full_name = get_full_name(slightly_correlated_c2)
highly_correlated_c1_full_name = get_full_name(highly_correlated_c1)
highly_correlated_c2_full_name = get_full_name(highly_correlated_c2)
anti_correlated_c1_full_name = get_full_name(anti_correlated_c1)
anti_correlated_c2_full_name = get_full_name(anti_correlated_c2)
independant_full_name = get_full_name(independant)

clusters = [slightly_correlated_c1_full_name,
slightly_correlated_c2_full_name,
highly_correlated_c1_full_name,
highly_correlated_c2_full_name,
anti_correlated_c1_full_name,
anti_correlated_c2_full_name,
independant_full_name]

clusters_name = ["slightly_correlated_c1",
"slightly_correlated_c2",
"highly_correlated_c1",
"highly_correlated_c2",
"anti_correlated_c1",
"anti_correlated_c2",
"independant"]

nii_images = get_nii_per_cluster_team(clusters, clusters_name,resampled_maps, masker)
plot_brains(nii_images, clusters_name, "7clusters", masker)

conditions = ["consensus0_gls1", "consensus1_gls0", "consensus1_gls1"]
for condition in [0, 1, 2]:
	for voxel in [0, -1]:
		v_ind = outputs[hyp][condition][voxel]
		print("running hyp {}, condition {}, voxel {}".format(hyp, conditions[condition], v_ind))
		plot_voxels_per_cluster(masker, v_ind, resampled_maps, data_GLS, data_SDMA_Stouffer, conditions[condition], team_names, hyp, results_dir, clusters_name, clusters)
############################
# 8 clusters : slightly_c1, slightly_c2, highly_c1, highly_c2, anti_c1, anti_c2, independent_c1, independent_c2
############################

## get mean weight per cluster of teams 
## define cluster of team from Narps paper figure 2
slightly_correlated_c1 = ["AO86", "43FJ", "O21U", "3PQ2", "0JO0", "I9D6", "51PW", "94GU"]
slightly_correlated_c2 = ["0ED6", "R5K7", "SM54", "B23O", "O03M"]
highly_correlated_c1 = ["DC61", "X1Y5", "UI76", "2T7P", "2T6S", "27SS", "T54A"]
highly_correlated_c2 = ["1KB2", "08MQ", "V55J", "3TR7", "Q6O0", "E3B6", "L7J7", "K9P0", "X19V", "9Q6R", "U26C", "50GV", "B5I6", "R9K3", "C88N", 
				"J7F9", "46CD", "C22U", "I52Y", "E6R3", "R7D1", "0C7Q", "6VV2", "98BT", "6FH5", "3C6G", "L3V8", "0I4U",
				"0H5E", "9U7M"]
anti_correlated_c1 = ["80GC", "1P0Y", "P5F3"]
anti_correlated_c2 = ["IZ20", "Q58J", "4TQ6", 'UK24']
independant_c1 = ["9T8E", "R42Q", "XU70"]
independant_c2 = ["L9G5", "O6R6", "4SZ2", "I07H"]


print("Getting full team names")
slightly_correlated_c1_full_name = get_full_name(slightly_correlated_c1)
slightly_correlated_c2_full_name = get_full_name(slightly_correlated_c2)
highly_correlated_c1_full_name = get_full_name(highly_correlated_c1)
highly_correlated_c2_full_name = get_full_name(highly_correlated_c2)
anti_correlated_c1_full_name = get_full_name(anti_correlated_c1)
anti_correlated_c2_full_name = get_full_name(anti_correlated_c2)
independant_c1_full_name = get_full_name(independant_c1)
independant_c2_full_name = get_full_name(independant_c2)

clusters = [slightly_correlated_c1_full_name,
slightly_correlated_c2_full_name,
highly_correlated_c1_full_name,
highly_correlated_c2_full_name,
anti_correlated_c1_full_name,
anti_correlated_c2_full_name,
independant_c1_full_name,
independant_c2_full_name]

clusters_name = ["slightly_correlated_c1",
"slightly_correlated_c2",
"highly_correlated_c1",
"highly_correlated_c2",
"anti_correlated_c1",
"anti_correlated_c2",
"independant_c1",
"independant_c2"]

nii_images = get_nii_per_cluster_team(clusters, clusters_name,resampled_maps, masker)
plot_brains(nii_images, clusters_name, "8clusters", masker)

conditions = ["consensus0_gls1", "consensus1_gls0", "consensus1_gls1"]
for condition in [0, 1, 2]:
	for voxel in [0, -1]:
		v_ind = outputs[hyp][condition][voxel]
		print("running hyp {}, condition {}, voxel {}".format(hyp, conditions[condition], v_ind))
		plot_voxels_per_cluster(masker, v_ind, resampled_maps, data_GLS, data_SDMA_Stouffer, conditions[condition], team_names, hyp, results_dir, clusters_name, clusters)
############################
# 9 clusters : slightly_c1, slightly_c2, highly_c1, highly_c2, highly_c3, anti_c1, anti_c2, independent_c1, independent_c2
############################

## get mean weight per cluster of teams 
## define cluster of team from Narps paper figure 2
slightly_correlated_c1 = ["AO86", "43FJ", "O21U", "3PQ2", "0JO0", "I9D6", "51PW", "94GU"]
slightly_correlated_c2 = ["0ED6", "R5K7", "SM54", "B23O", "O03M"]
highly_correlated_c1 = ["DC61", "X1Y5", "UI76", "2T7P", "2T6S", "27SS", "T54A"]
highly_correlated_c3 = ["1KB2", "08MQ", "V55J"]
highly_correlated_c2 = ["3TR7", "Q6O0", "E3B6", "L7J7", "K9P0", "X19V", "9Q6R", "U26C", "50GV", "B5I6", "R9K3", "C88N", 
				"J7F9", "46CD", "C22U", "I52Y", "E6R3", "R7D1", "0C7Q", "6VV2", "98BT", "6FH5", "3C6G", "L3V8", "0I4U",
				"0H5E", "9U7M"]
anti_correlated_c1 = ["80GC", "1P0Y", "P5F3"]
anti_correlated_c2 = ["IZ20", "Q58J", "4TQ6", 'UK24']
independant_c1 = ["9T8E", "R42Q", "XU70"]
independant_c2 = ["L9G5", "O6R6", "4SZ2", "I07H"]


print("Getting full team names")
slightly_correlated_c1_full_name = get_full_name(slightly_correlated_c1)
slightly_correlated_c2_full_name = get_full_name(slightly_correlated_c2)
highly_correlated_c1_full_name = get_full_name(highly_correlated_c1)
highly_correlated_c2_full_name = get_full_name(highly_correlated_c2)
highly_correlated_c3_full_name = get_full_name(highly_correlated_c3)
anti_correlated_c1_full_name = get_full_name(anti_correlated_c1)
anti_correlated_c2_full_name = get_full_name(anti_correlated_c2)
independant_c1_full_name = get_full_name(independant_c1)
independant_c2_full_name = get_full_name(independant_c2)

clusters = [slightly_correlated_c1_full_name,
slightly_correlated_c2_full_name,
highly_correlated_c1_full_name,
highly_correlated_c2_full_name,
highly_correlated_c3_full_name,
anti_correlated_c1_full_name,
anti_correlated_c2_full_name,
independant_c1_full_name,
independant_c2_full_name]

clusters_name = ["slightly_correlated_c1",
"slightly_correlated_c2",
"highly_correlated_c1",
"highly_correlated_c2",
"highly_correlated_c3",
"anti_correlated_c1",
"anti_correlated_c2",
"independant_c1",
"independant_c2"]

nii_images = get_nii_per_cluster_team(clusters, clusters_name,resampled_maps, masker)
plot_brains(nii_images, clusters_name, "9clusters", masker)

conditions = ["consensus0_gls1", "consensus1_gls0", "consensus1_gls1"]
for condition in [0, 1, 2]:
	for voxel in [0, -1]:
		v_ind = outputs[hyp][condition][voxel]
		print("running hyp {}, condition {}, voxel {}".format(hyp, conditions[condition], v_ind))
		plot_voxels_per_cluster(masker, v_ind, resampled_maps, data_GLS, data_SDMA_Stouffer, conditions[condition], team_names, hyp, results_dir, clusters_name, clusters)