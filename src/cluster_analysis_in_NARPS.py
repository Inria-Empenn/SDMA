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
import compute_MA_outputs

importlib.reload(utils)
importlib.reload(MA_estimators)

# path to partiticipants mask
participants_mask_path = os.path.join("results", "NARPS", "masking", "participants_mask.nii")
participant_mask = nibabel.load(participants_mask_path)
# path to resampled NARPS data
data_path = os.path.join("data", "NARPS")

# fit masker
masker = NiftiMasker(
    mask_img=participant_mask)

# Note that these analyses are ran only for hypothesis 1
# load data
path_to_resample_maps_hyp_1 = os.path.join("data", "NARPS", "Hyp1_resampled_maps.npy")
pipeline_z_scores_per_team = numpy.load(path_to_resample_maps_hyp_1, allow_pickle=True).item()
pipeline_z_scores_per_team.pop("5496_VG39")# not included in narps study, thus to be removed
team_names = list(pipeline_z_scores_per_team.keys())
pipeline_z_scores= masker.fit_transform(pipeline_z_scores_per_team.values())
MA_outputs = compute_MA_outputs.get_MA_outputs(pipeline_z_scores)

####################################################################
################### BEGINNING UTILS FUNCTIONS ######################
####################################################################
def get_full_name(endings, pipeline_z_scores_per_team=pipeline_z_scores_per_team):
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

def search_for_nicelly_defined_voxels(clusters, clusters_name, team_names, pipeline_z_scores):
	# extract voxels that are well defined.
	# that is, independant cluster and anticorrelated cluster data points are separated
	# cluster in hyp 1 and setting SDMA 0 and GLS 1:
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
			if MA_outputs['SDMA GLS']['p_values'][voxel_nicelly_defined] <= 0.05:
				voxels_of_interest_corrup['SDMA1_GLS1'].append(voxel_nicelly_defined)
			else:
				voxels_of_interest_corrup['SDMA1_GLS0'].append(voxel_nicelly_defined)
		elif MA_outputs['SDMA GLS']['p_values'][voxel_nicelly_defined] <= 0.05:
			voxels_of_interest_corrup['SDMA0_GLS1'].append(voxel_nicelly_defined)
		else:
			voxels_of_interest_corrup['not_significant'].append(voxel_nicelly_defined)
	# corr down	
	for voxel_nicelly_defined in voxels_nicelly_defined_corrdown:
		if MA_outputs['SDMA Stouffer']['p_values'][voxel_nicelly_defined] <= 0.05:
			if MA_outputs['SDMA GLS']['p_values'][voxel_nicelly_defined] <= 0.05:
				voxels_of_interest_corrdown['SDMA1_GLS1'].append(voxel_nicelly_defined)
			else:
				voxels_of_interest_corrdown['SDMA1_GLS0'].append(voxel_nicelly_defined)
		elif MA_outputs['SDMA GLS']['p_values'][voxel_nicelly_defined] <= 0.05:
			voxels_of_interest_corrdown['SDMA0_GLS1'].append(voxel_nicelly_defined)
		else:
			voxels_of_interest_corrdown['not_significant'].append(voxel_nicelly_defined)
	return voxels_of_interest_corrup, voxels_of_interest_corrdown

def plot_brains(clusters, clusters_name, team_names, masker, pipeline_z_scores):
	save_for_plotting_separately = {}
	print("Check using the original computation of GLS contributions:")
	T_map, p_map, _ = MA_estimators.SDMA_GLS(pipeline_z_scores)
	T_map_GLS_nii = masker.inverse_transform(T_map)
	p_map = (p_map <= 0.05) * T_map
	p_map_GLS_nii = masker.inverse_transform(p_map)

	T_map, p_map, _ = MA_estimators.SDMA_Stouffer(pipeline_z_scores)
	T_map_SMDA_Stouffer_nii = masker.inverse_transform(T_map)
	p_map = (p_map <= 0.05) * T_map
	p_map_SDMA_Stouffer_nii = masker.inverse_transform(p_map)

	print("getting cluster indices...")
	clusters_indices = get_cluster_indices(clusters, clusters_name, team_names)
	print("get gls weight per pipeline...")
	weight_pipelines_gls = compute_GLS_weights(pipeline_z_scores, std_by_Stouffer=False)

	print("get SDMA Stouffer and GLS contributions...")
	contributions_SDMA_Stouffer, contributions_GLS = compute_contributions(pipeline_z_scores, W="SDMA", std_by_Stouffer=False)
	mean_contributions_SDMA_Stouffer_nii = masker.inverse_transform(numpy.mean(contributions_SDMA_Stouffer, axis=0))
	mean_contributions_GLS_nii = masker.inverse_transform(numpy.mean(contributions_GLS, axis=0))
	
	plt.close('all')
	f, axs = plt.subplots(len(clusters_name) + 4, 4, figsize=(16, len(clusters_name)*5), width_ratios=[0.1, 0.4, 0.4, 0.1])

	# variables for reconstructing global contribution from cluster contributions
	reconstructed_GLS_contributions = 0
	reconstructed_SDMA_Stouffer_contributions = 0

	# plot mean SDMA Stouffer weight for this cluster
	ones = numpy.ones((pipeline_z_scores.shape[0], 1))
	Q = numpy.corrcoef(pipeline_z_scores)
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
		axs[row, 1].set_title("Contributions " + name + " subgroup", size=12)

		# plot mean GLS contribution for this cluster
		sum_contributions_GLS_this_cluster_nii = masker.inverse_transform(this_cluster_contributions_GLS.sum(axis=0))
		plotting.plot_stat_map(sum_contributions_GLS_this_cluster_nii,  
			annotate=False,  
			colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), 
			display_mode='z', cmap='coolwarm', axes=axs[row, 2])
		axs[row, 2].set_title("Sum contributions " + name + " subgroup", size=12)

		# Save each plot as unique figure to build the final manuscript figure 
		if len(clusters_name) == 3:
			save_for_plotting_separately["SDMA Stouffer Sum contributions {}".format(name)]= sum_contributions_SDMA_Stouffer_this_cluster_nii
			save_for_plotting_separately["SDMA GLS Sum contributions {}".format(name)]= sum_contributions_GLS_this_cluster_nii




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

	# plot whole brain SDMA Stouffer contribution computed from compute_contributions
	plotting.plot_stat_map(mean_contributions_SDMA_Stouffer_nii,
		annotate=False,  
		colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), 
		display_mode='z', cmap='coolwarm', axes=axs[row+2, 1])
	axs[row+2, 1].set_title("Mean global contributions", size=12)

	# plot whole brain GLS contribution computed from compute_contributions
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

	# Save each plot as unique figure to build the final manuscript figure 
	if len(clusters_name) == 3:
		save_for_plotting_separately["SDMA Stouffer sign p"]= p_map_SDMA_Stouffer_nii
		save_for_plotting_separately["SDMA GLS sign p"]= p_map_GLS_nii


	axs[row+1, 0].axis('off')
	axs[row+2, 0].axis('off')
	axs[row+3, 0].axis('off')
	axs[row+1, 3].axis('off')
	axs[row+3, 3].axis('off')
	axs[row+4, 0].axis('off')
	axs[row+4, 3].axis('off')
	print("Saving plot")
	saving_path = os.path.join("figures", "NARPS", "hyp1", "cluster_analysis", "{}clusters".format(len(clusters_name)), "per_cluster_weights_{}.png".format(len(clusters_name)))
	plt.savefig(saving_path)

	if len(clusters_name) ==3:
		# build figure 4
		plt.close('all')
		f, axs = plt.subplots(len(clusters_name) + 1, 2, figsize=(16, 7))

		plotting.plot_stat_map(save_for_plotting_separately["SDMA Stouffer Sum contributions correlated"],
		annotate=False,
		colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52),
		display_mode='z', cmap='coolwarm', axes=axs[0, 0])
		axs[0, 0].set_title("Contributions majority subgroup", size=12)

		plotting.plot_stat_map(save_for_plotting_separately["SDMA GLS Sum contributions correlated"],
		annotate=False,
		colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52),
		display_mode='z', cmap='coolwarm', axes=axs[0, 1])
		axs[0, 1].set_title("Contributions majority subgroup", size=12)

		plotting.plot_stat_map(save_for_plotting_separately["SDMA Stouffer Sum contributions anti_correlated"],
		annotate=False,
		colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52),
		display_mode='z', cmap='coolwarm', axes=axs[1, 0])
		axs[1, 0].set_title("Contributions opposite subgroup", size=12)

		plotting.plot_stat_map(save_for_plotting_separately["SDMA GLS Sum contributions anti_correlated"],
		annotate=False,
		colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52),
		display_mode='z', cmap='coolwarm', axes=axs[1, 1])
		axs[1, 1].set_title("Contributions opposite subgroup", size=12)

		plotting.plot_stat_map(save_for_plotting_separately["SDMA Stouffer Sum contributions independant"],
		annotate=False,
		colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52),
		display_mode='z', cmap='coolwarm', axes=axs[2, 0])
		axs[2, 0].set_title("Contributions unrelated subgroup", size=12)

		plotting.plot_stat_map(save_for_plotting_separately["SDMA GLS Sum contributions independant"],
		annotate=False,
		colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52),
		display_mode='z', cmap='coolwarm', axes=axs[2, 1])
		axs[2, 1].set_title("Contributions unrelated subgroup", size=12)

		plotting.plot_stat_map(save_for_plotting_separately["SDMA Stouffer sign p"],
		annotate=False, vmax=8,
		colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52),
		display_mode='z', cmap='Reds', axes=axs[3, 0])
		axs[3, 0].set_title("Significant T values", size=12)

		plotting.plot_stat_map(save_for_plotting_separately["SDMA GLS sign p"],
		annotate=False, vmax=8,
		colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52),
		display_mode='z', cmap='Reds', axes=axs[3, 1])
		axs[3, 1].set_title("Significant T values", size=12)

		saving_path = os.path.join("figures", "NARPS", "hyp1", "cluster_analysis", "3clusters", "Fig4.png")
		plt.savefig(saving_path, dpi=300)


def compute_contributions(pipeline_z_scores, W="SDMA", std_by_Stouffer=True):
     ones = numpy.ones((pipeline_z_scores.shape[0], 1))
     Q = numpy.corrcoef(pipeline_z_scores)
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

def plot_voxels_per_cluster(masker, pipeline_z_scores, team_names, clusters_name, clusters, eight_voxels_nicelly_defined):
	print("getting cluster indices...")
	clusters_indices = get_cluster_indices(clusters, clusters_name, team_names)
	print("get gls weight per pipeline...")
	weight_pipelines_gls = compute_GLS_weights(pipeline_z_scores)
	print("get SDMA Stouffer and GLS contributions...")
	contributions_SDMA_Stouffer, contributions_GLS = compute_contributions(pipeline_z_scores, W=1)

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
		saving_path = os.path.join("figures", "NARPS", "hyp1", "cluster_analysis", "{}clusters".format(len(clusters_name)), "cluster_for_voxel_{}.png".format(voxel_index))
		plt.savefig(saving_path)



def custom_swarmplot(data, ax, team_names, clusters_name, clusters):
	c=['orange', 'blue', 'red', "green", "yellow", "black", "grey", "purple", "lightgreen"]
	for i, value in enumerate(data):
		for ind, name in enumerate(clusters_name):
			if team_names[i] in clusters[ind]:
				ax.plot(0, value, 'o', markersize=6, alpha=0)
				ax.text(0, value, team_names[i], color=c[ind], fontsize=6, ha='center', va='bottom')
				continue

def remove_anticorr_pipelines_from_data_and_clusters(nb_anticorr_pipes_to_remove, pipeline_z_scores, team_names, masker):
	# compute GLS weight with increasing # of anticorrelated pipelines
	shrinked_pipeline_z_scores = pipeline_z_scores.copy()
	shrinked_team_names = team_names.copy()
	## define cluster of team from Narps paper figure 2
	correlated = ["AO86", "43FJ", "O21U", "3PQ2", "0JO0", "I9D6", "51PW", "94GU", "0ED6", "R5K7", "SM54", "B23O",
					"O03M","DC61", "X1Y5", "UI76", "2T7P", "2T6S", "27SS", "T54A", "1KB2", "08MQ", "V55J",
					"3TR7", "Q6O0", "E3B6", "L7J7", "9Q6R", "U26C", "50GV", "B5I6", "R9K3", "C88N", 
					"J7F9", "46CD", "C22U", "I52Y", "E6R3", "R7D1", "0C7Q", "6VV2", "98BT", "6FH5", "3C6G", "L3V8", "0I4U",
					"0H5E", "9U7M"]
	anti_correlated = ["80GC", "1P0Y", "P5F3", "IZ20", "Q58J", "4TQ6", 'UK24']
	independant = ["9T8E", "R42Q", "L9G5", "O6R6", "4SZ2"]

	print("Getting full team names")
	correlated_full_name = get_full_name(correlated)
	anti_correlated_full_name = get_full_name(anti_correlated)
	independant_full_name = get_full_name(independant)

	anticorr_pipelines_thrown = []
	for i in range(nb_anticorr_pipes_to_remove):
		anticorr_pipelines_thrown.append(anti_correlated_full_name.pop())
	assert anti_correlated.__len__() > 0, "Too many anticorr pipelines removed..."

	clusters = [correlated_full_name,
	anti_correlated_full_name,
	independant_full_name]

	clusters_name = ["correlated",
	"anti_correlated",
	"independant"]

	# restructure pipeline_z_scores
	for anticorr_pipeline_thrown in anticorr_pipelines_thrown:
		index_to_ditch = shrinked_team_names.index(anticorr_pipeline_thrown)
		print("size team_name = ", shrinked_team_names.__len__()," et z_scores = ", shrinked_pipeline_z_scores.shape)
		print("JETER: ", anticorr_pipelines_thrown)
		print("INDEX: ", index_to_ditch)
		print("JETER: ", shrinked_team_names[index_to_ditch])
		shrinked_pipeline_z_scores = numpy.delete(shrinked_pipeline_z_scores, index_to_ditch, axis=0)
		del shrinked_team_names[index_to_ditch]

	compute_GLS_weights_for_shrinked_nb_of_anticorrelated_pipelines(clusters, clusters_name, shrinked_team_names, masker, shrinked_pipeline_z_scores, nb_anticorr_pipes_to_remove)

def compute_GLS_weights_for_shrinked_nb_of_anticorrelated_pipelines(clusters, clusters_name, team_names, masker, shrinked_pipeline_z_scores, nb_anticorr_pipes_to_remove):
	print("Check using the original computation of GLS contributions:")
	T_map, p_map, _ = MA_estimators.SDMA_GLS(shrinked_pipeline_z_scores)
	T_map_GLS_nii = masker.inverse_transform(T_map)
	p_map = (p_map <= 0.05) * T_map
	p_map_GLS_nii = masker.inverse_transform(p_map)

	print("getting cluster indices...")
	clusters_indices = get_cluster_indices(clusters, clusters_name, team_names)
	print("get gls weight per pipeline...")
	weight_pipelines_gls = compute_GLS_weights(shrinked_pipeline_z_scores, std_by_Stouffer=False)

	print("get SDMA Stouffer and GLS contributions...")
	_, contributions_GLS = compute_contributions(shrinked_pipeline_z_scores, W="SDMA", std_by_Stouffer=False)
	mean_contributions_GLS_nii = masker.inverse_transform(numpy.mean(contributions_GLS, axis=0))
	
	plt.close('all')
	f, axs = plt.subplots(len(clusters_name) + 2, 2, figsize=(16, len(clusters_name)*5), width_ratios=[0.9, 0.1])


	for row, name in enumerate(clusters_name):
		print("Drawing the mean weights + sum of contributions for cluster: ", name)
		this_cluster_indices = clusters_indices[name]
		this_cluster_contributions_GLS = contributions_GLS[this_cluster_indices]

		# reconstruct global contribution from cluster contributions
		if row == 0:
			reconstructed_GLS_contributions = this_cluster_contributions_GLS.sum(axis=0)
		else:
			reconstructed_GLS_contributions += this_cluster_contributions_GLS.sum(axis=0)

		# plot mean weight for this cluster
		mean_weight_of_this_cluster = numpy.round(weight_pipelines_gls[this_cluster_indices].mean(), 4)
		axs[row, 1].imshow(numpy.array(mean_weight_of_this_cluster).reshape(-1, 1), cmap='coolwarm', aspect='equal', vmin=-0.5, vmax=0.5)
		axs[row, 1].text(0, 0, mean_weight_of_this_cluster, ha="center", va="center", color="black")
		axs[row, 1].axis('off')
		if row == 0:
			axs[row, 1].set_title('mean GLS weight') 
		# take off axis where sdma weight is
		axs[row, 1].axis('off')

		# plot mean GLS contribution for this cluster
		sum_contributions_GLS_this_cluster_nii = masker.inverse_transform(this_cluster_contributions_GLS.sum(axis=0))
		plotting.plot_stat_map(sum_contributions_GLS_this_cluster_nii,  
			annotate=False,  
			colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), 
			display_mode='z', cmap='coolwarm', axes=axs[row, 0])
		axs[row, 0].set_title("Sum contributions " + name, size=12)

	# plot whole brain GLS contribution computed from MA_estimator
	plotting.plot_stat_map(T_map_GLS_nii,
		annotate=False,  
		colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), 
		display_mode='z', cmap='coolwarm', axes=axs[row+1, 0])
	axs[row+1, 0].set_title("T map", size=12)

	# plot p value significant using GLS
	plotting.plot_stat_map(p_map_GLS_nii,
		annotate=False, vmax=8,
		colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), 
		display_mode='z', cmap='Reds', axes=axs[row+2, 0])
	axs[row+2, 0].set_title("Significant T values", size=12)

	axs[row+1, 1].axis('off')
	axs[row+2, 1].axis('off')

	print("Saving plot")
	plt.suptitle("Nb anticorrelated pipelines removed: {}".format(nb_anticorr_pipes_to_remove), fontsize="xx-large")
	saving_path = os.path.join("figures", "NARPS", "hyp1", "cluster_analysis", "{}clusters".format(len(clusters_name)), "effect_of_removing_{}anticorr_pipelines_on_per_cluster_weights.png".format(nb_anticorr_pipes_to_remove))
	plt.savefig(saving_path)
	# plt.show()


##############################################################
################### END UTILS FUNCTIONS ######################
##############################################################


############################
# Find voxel well defined (ANTICORRELATED below CORRELATED)
# this is important as we want to check the weight only for voxels that are representative of the respective clusters
############################

## get mean weight per cluster of teams 
## define cluster of team from Narps paper figure 2
correlated = ["AO86", "43FJ", "O21U", "3PQ2", "0JO0", "I9D6", "51PW", "94GU", "0ED6", "R5K7", "SM54", "B23O",
				"O03M","DC61", "X1Y5", "UI76", "2T7P", "2T6S", "27SS", "T54A", "1KB2", "08MQ", "V55J",
				"3TR7", "Q6O0", "E3B6", "L7J7", "9Q6R", "U26C", "50GV", "B5I6", "R9K3", "C88N", 
				"J7F9", "46CD", "C22U", "I52Y", "E6R3", "R7D1", "0C7Q", "6VV2", "98BT", "6FH5", "3C6G", "L3V8", "0I4U",
				"0H5E", "9U7M"]
anti_correlated = ["80GC", "1P0Y", "P5F3", "IZ20", "Q58J", "4TQ6", 'UK24']
independant = ["9T8E", "R42Q", "L9G5", "O6R6", "4SZ2"]

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

# FIND VOXEL WELL DEFINED (ANTICORRELATED below CORRELATED)
voxels_nicelly_defined_corrup, voxels_nicelly_defined_corrdown = search_for_nicelly_defined_voxels(clusters, clusters_name, team_names, pipeline_z_scores)
# FIND VOXEL SIGNICICANCE STATUT
voxels_of_interest_corrup, voxels_of_interest_corrdown = search_for_significant_voxels_within_nicelly_defined_voxel(voxels_nicelly_defined_corrup, voxels_nicelly_defined_corrdown,MA_outputs)

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


############################
# 2 clusters : correlated, independent
############################

## get mean weight per cluster of teams 
## define cluster of team from Narps paper figure 2
correlated = ["AO86", "43FJ", "O21U", "3PQ2", "0JO0", "I9D6", "51PW", "94GU", "0ED6", "R5K7", "SM54", "B23O",
				"O03M", "DC61", "X1Y5", "UI76", "2T7P", "2T6S", "27SS", "T54A", "1KB2", "08MQ", "V55J",
				"3TR7", "Q6O0", "E3B6", "L7J7", "9Q6R", "U26C", "50GV", "B5I6", "R9K3", "C88N", 
				"J7F9", "46CD", "C22U", "I52Y", "E6R3", "R7D1", "0C7Q", "6VV2", "98BT", "6FH5", "3C6G", "L3V8", "0I4U",
				"0H5E", "9U7M"]
				
independant = ["9T8E", "R42Q", "L9G5", "O6R6", "4SZ2", "80GC", "1P0Y", "P5F3", "IZ20", "Q58J", "4TQ6", 'UK24']

print("Getting full team names")
correlated_full_name = get_full_name(correlated)
independant_full_name = get_full_name(independant)

clusters = [correlated_full_name,
	independant_full_name]

clusters_name = ["correlated",
	"independant"]

print("plotting 2 clusters")
plot_brains(clusters, clusters_name, team_names, masker, pipeline_z_scores)
plot_voxels_per_cluster(masker, pipeline_z_scores, team_names, clusters_name, clusters, eight_voxels_nicelly_defined)



############################
# 3 clusters : correlated, anti-correlated, independent
############################

## get mean weight per cluster of teams 
## define cluster of team from Narps paper figure 2
correlated = ["AO86", "43FJ", "O21U", "3PQ2", "0JO0", "I9D6", "51PW", "94GU", "0ED6", "R5K7", "SM54", "B23O",
				"O03M","DC61", "X1Y5", "UI76", "2T7P", "2T6S", "27SS", "T54A", "1KB2", "08MQ", "V55J",
				"3TR7", "Q6O0", "E3B6", "L7J7", "9Q6R", "U26C", "50GV", "B5I6", "R9K3", "C88N", 
				"J7F9", "46CD", "C22U", "I52Y", "E6R3", "R7D1", "0C7Q", "6VV2", "98BT", "6FH5", "3C6G", "L3V8", "0I4U",
				"0H5E", "9U7M"]
anti_correlated = ["80GC", "1P0Y", "P5F3", "IZ20", "Q58J", "4TQ6", 'UK24']
independant = ["9T8E", "R42Q", "L9G5", "O6R6", "4SZ2"]

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

print("plotting 3 clusters")

plot_brains(clusters, clusters_name, team_names, masker, pipeline_z_scores)
stop
plot_voxels_per_cluster(masker, pipeline_z_scores, team_names, clusters_name, clusters, eight_voxels_nicelly_defined)
remove_anticorr_pipelines_from_data_and_clusters(1, pipeline_z_scores, team_names, masker)
remove_anticorr_pipelines_from_data_and_clusters(2, pipeline_z_scores, team_names, masker)
remove_anticorr_pipelines_from_data_and_clusters(3, pipeline_z_scores, team_names, masker)
remove_anticorr_pipelines_from_data_and_clusters(4, pipeline_z_scores, team_names, masker)




############################
# 4 clusters : slightly, highly, anti, independent
############################

## get mean weight per cluster of teams 
## define cluster of team from Narps paper figure 2
slightly_correlated = ["AO86", "43FJ", "O21U", "3PQ2", "0JO0", "I9D6", "51PW", "94GU", "0ED6", "R5K7", "SM54", "B23O",
				"O03M"]
highly_correlated = ["DC61", "X1Y5", "UI76", "2T7P", "2T6S", "27SS", "T54A", "1KB2", "08MQ", "V55J",
				"3TR7", "Q6O0", "E3B6", "L7J7", "9Q6R", "U26C", "50GV", "B5I6", "R9K3", "C88N", 
				"J7F9", "46CD", "C22U", "I52Y", "E6R3", "R7D1", "0C7Q", "6VV2", "98BT", "6FH5", "3C6G", "L3V8", "0I4U",
				"0H5E", "9U7M"]
anti_correlated = ["80GC", "1P0Y", "P5F3", "IZ20", "Q58J", "4TQ6", 'UK24']
independant = ["9T8E", "R42Q", "L9G5", "O6R6", "4SZ2"]

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

print("plotting 4 clusters")
plot_brains(clusters, clusters_name, team_names, masker, pipeline_z_scores)
plot_voxels_per_cluster(masker, pipeline_z_scores, team_names, clusters_name, clusters, eight_voxels_nicelly_defined)

############################
# 5 clusters : slightly, highly_c1, highly_c2, anti, independent
############################

## get mean weight per cluster of teams 
## define cluster of team from Narps paper figure 2
slightly_correlated = ["AO86", "43FJ", "O21U", "3PQ2", "0JO0", "I9D6", "51PW", "94GU", "0ED6", "R5K7", "SM54", "B23O",
				"O03M"]
highly_correlated_c1 = ["DC61", "X1Y5", "UI76", "2T7P", "2T6S", "27SS", "T54A"]
highly_correlated_c2 = ["1KB2", "08MQ", "V55J", "3TR7", "Q6O0", "E3B6", "L7J7", "9Q6R", "U26C", "50GV", "B5I6", "R9K3", "C88N", 
				"J7F9", "46CD", "C22U", "I52Y", "E6R3", "R7D1", "0C7Q", "6VV2", "98BT", "6FH5", "3C6G", "L3V8", "0I4U",
				"0H5E", "9U7M"]
anti_correlated = ["80GC", "1P0Y", "P5F3", "IZ20", "Q58J", "4TQ6", 'UK24']
independant = ["9T8E", "R42Q", "L9G5", "O6R6", "4SZ2"]


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


print("plotting 5 clusters")
plot_brains(clusters, clusters_name, team_names, masker, pipeline_z_scores)
plot_voxels_per_cluster(masker, pipeline_z_scores, team_names, clusters_name, clusters, eight_voxels_nicelly_defined)

############################
# 6 clusters : slightly_c1, slightly_c2, highly_c1, highly_c2, anti, independent
############################

## get mean weight per cluster of teams 
## define cluster of team from Narps paper figure 2
slightly_correlated_c1 = ["AO86", "43FJ", "O21U", "3PQ2", "0JO0", "I9D6", "51PW", "94GU"]
slightly_correlated_c2 = ["0ED6", "R5K7", "SM54", "B23O", "O03M"]
highly_correlated_c1 = ["DC61", "X1Y5", "UI76", "2T7P", "2T6S", "27SS", "T54A"]
highly_correlated_c2 = ["1KB2", "08MQ", "V55J", "3TR7", "Q6O0", "E3B6", "L7J7", "9Q6R", "U26C", "50GV", "B5I6", "R9K3", "C88N", 
				"J7F9", "46CD", "C22U", "I52Y", "E6R3", "R7D1", "0C7Q", "6VV2", "98BT", "6FH5", "3C6G", "L3V8", "0I4U",
				"0H5E", "9U7M"]
anti_correlated = ["80GC", "1P0Y", "P5F3", "IZ20", "Q58J", "4TQ6", 'UK24']
independant = ["9T8E", "R42Q", "L9G5", "O6R6", "4SZ2"]


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

print("plotting 6 clusters")
plot_brains(clusters, clusters_name, team_names, masker, pipeline_z_scores)
plot_voxels_per_cluster(masker, pipeline_z_scores, team_names, clusters_name, clusters, eight_voxels_nicelly_defined)


############################
# 7 clusters : slightly_c1, slightly_c2, highly_c1, highly_c2, anti_c1, anti_c2, independent
############################

## get mean weight per cluster of teams 
## define cluster of team from Narps paper figure 2
slightly_correlated_c1 = ["AO86", "43FJ", "O21U", "3PQ2", "0JO0", "I9D6", "51PW", "94GU"]
slightly_correlated_c2 = ["0ED6", "R5K7", "SM54", "B23O", "O03M"]
highly_correlated_c1 = ["DC61", "X1Y5", "UI76", "2T7P", "2T6S", "27SS", "T54A"]
highly_correlated_c2 = ["1KB2", "08MQ", "V55J", "3TR7", "Q6O0", "E3B6", "L7J7", "9Q6R", "U26C", "50GV", "B5I6", "R9K3", "C88N", 
				"J7F9", "46CD", "C22U", "I52Y", "E6R3", "R7D1", "0C7Q", "6VV2", "98BT", "6FH5", "3C6G", "L3V8", "0I4U",
				"0H5E", "9U7M"]
anti_correlated_c1 = ["80GC", "1P0Y", "P5F3"]
anti_correlated_c2 = ["IZ20", "Q58J", "4TQ6", 'UK24']
independant = ["9T8E", "R42Q", "L9G5", "O6R6", "4SZ2"]


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

print("plotting 7 clusters")
plot_brains(clusters, clusters_name, team_names, masker, pipeline_z_scores)
plot_voxels_per_cluster(masker, pipeline_z_scores, team_names, clusters_name, clusters, eight_voxels_nicelly_defined)

############################
# 8 clusters : slightly_c1, slightly_c2, highly_c1, highly_c2, anti_c1, anti_c2, independent_c1, independent_c2
############################

## get mean weight per cluster of teams 
## define cluster of team from Narps paper figure 2
slightly_correlated_c1 = ["AO86", "43FJ", "O21U", "3PQ2", "0JO0", "I9D6", "51PW", "94GU"]
slightly_correlated_c2 = ["0ED6", "R5K7", "SM54", "B23O", "O03M"]
highly_correlated_c1 = ["DC61", "X1Y5", "UI76", "2T7P", "2T6S", "27SS", "T54A"]
highly_correlated_c2 = ["1KB2", "08MQ", "V55J", "3TR7", "Q6O0", "E3B6", "L7J7", "9Q6R", "U26C", "50GV", "B5I6", "R9K3", "C88N", 
				"J7F9", "46CD", "C22U", "I52Y", "E6R3", "R7D1", "0C7Q", "6VV2", "98BT", "6FH5", "3C6G", "L3V8", "0I4U",
				"0H5E", "9U7M"]
anti_correlated_c1 = ["80GC", "1P0Y", "P5F3"]
anti_correlated_c2 = ["IZ20", "Q58J", "4TQ6", 'UK24']
independant_c1 = ["9T8E", "R42Q"]
independant_c2 = ["L9G5", "O6R6", "4SZ2"]


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


print("plotting 8 clusters")
plot_brains(clusters, clusters_name, team_names, masker, pipeline_z_scores)
plot_voxels_per_cluster(masker, pipeline_z_scores, team_names, clusters_name, clusters, eight_voxels_nicelly_defined)

############################
# 9 clusters : slightly_c1, slightly_c2, highly_c1, highly_c2, highly_c3, anti_c1, anti_c2, independent_c1, independent_c2
############################

## get mean weight per cluster of teams 
## define cluster of team from Narps paper figure 2
slightly_correlated_c1 = ["AO86", "43FJ", "O21U", "3PQ2", "0JO0", "I9D6", "51PW", "94GU"]
slightly_correlated_c2 = ["0ED6", "R5K7", "SM54", "B23O", "O03M"]
highly_correlated_c1 = ["DC61", "X1Y5", "UI76", "2T7P", "2T6S", "27SS", "T54A"]
highly_correlated_c3 = ["1KB2", "08MQ", "V55J"]
highly_correlated_c2 = ["3TR7", "Q6O0", "E3B6", "L7J7", "9Q6R", "U26C", "50GV", "B5I6", "R9K3", "C88N", 
				"J7F9", "46CD", "C22U", "I52Y", "E6R3", "R7D1", "0C7Q", "6VV2", "98BT", "6FH5", "3C6G", "L3V8", "0I4U",
				"0H5E", "9U7M"]
anti_correlated_c1 = ["80GC", "1P0Y", "P5F3"]
anti_correlated_c2 = ["IZ20", "Q58J", "4TQ6", 'UK24']
independant_c1 = ["9T8E", "R42Q"]
independant_c2 = ["L9G5", "O6R6", "4SZ2"]


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


print("plotting 9 clusters")
plot_brains(clusters, clusters_name, team_names, masker, pipeline_z_scores)
plot_voxels_per_cluster(masker, pipeline_z_scores, team_names, clusters_name, clusters, eight_voxels_nicelly_defined)
