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

importlib.reload(utils)

###################################
# extract voxel's index that is significant or not OR is significant in both in SDMA_Stouffer and GLS in frontal and occipital
###################################

outputs = {}
participant_mask = nibabel.load("masking/mask_90.nii")
# fit masker
masker = NiftiMasker(
    mask_img=participant_mask)
results_dir = "results_in_Narps_data"
# for hyp in [1, 2, 5, 6, 7, 8, 9]:

 
try:
    outputs = numpy.load("{}/data/check_voxels_distribution_outputs.npy".format(results_dir), allow_pickle=True).item()
    print("outputs successfully loaded")
except:
	print("recharging outputs")
	for hyp in [1, 2, 5, 6, 7, 8, 9]:
		# outputs[hyp] = {} 
		print("hyp ", hyp, " ***********")
		resampled_maps_per_team = numpy.load('{}/data/Hyp{}_resampled_maps.npy'.format(results_dir, hyp), allow_pickle=True).item()
		resampled_maps = masker.fit_transform(resampled_maps_per_team.values())
		# display following for specific hypothesis number
		results_hyp = numpy.load("{}/data/Hyp{}_MA_estimates.npy".format(results_dir, hyp), allow_pickle=True).item()
		# get significant voxels for SDMA_Stouffer
		consensus_p = results_hyp["SDMA Stouffer"]["p_values"]
		consensus_p_thresholded = (consensus_p <= 0.05)*1
		consensus_t = results_hyp["SDMA Stouffer"]["T_map"]
		back_to_brain_consensus = consensus_p_thresholded * consensus_t
		# consensus_threshimg = masker.inverse_transform(back_to_brain)

		# get significant voxels for GLS
		GLS_p = results_hyp["GLS SDMA"]["p_values"]
		GLS_p_thresholded = (GLS_p <= 0.05)*1
		GLS_t = results_hyp["GLS SDMA"]["T_map"]
		back_to_brain_GLS = GLS_p_thresholded * GLS_t
		# GLS_threshimg = masker.inverse_transform(back_to_brain)

		consensus0_gls1 = []
		consensus1_gls0 = []
		consensus1_gls1 = []
		for ind_voxel, voxel_value in enumerate(back_to_brain_consensus):
			if (voxel_value != 0)&(back_to_brain_GLS[ind_voxel]!= 0):
				consensus1_gls1.append(ind_voxel)
				print("1 1")
			elif voxel_value != 0:
				consensus1_gls0.append(ind_voxel)
				print("1 0")
			elif back_to_brain_GLS[ind_voxel]!= 0:
				consensus0_gls1.append(ind_voxel)
				print("0 1")
		outputs[hyp] = [consensus0_gls1, consensus1_gls0, consensus1_gls1]
	numpy.save("{}/data/check_voxels_distribution_outputs.npy".format(results_dir), outputs, allow_pickle=True, fix_imports=True)


###################################
# function to plot them
###################################

# Create a custom swarm plot
def custom_swarmplot(data, ax, team_names):
	for i, value in enumerate(data):
		ax.plot(0, value, 'o', markersize=6, alpha=0)
		ax.text(0, value, team_names[i], fontsize=6, ha='center', va='bottom')


def plot_voxels_per_team(masker, voxel_index, resampled_maps, condition, team_names, p_values, hyp, results_dir):
	plt.close('all')
	# plot each voxel:
	fake_ROI = numpy.zeros(resampled_maps.shape[1])
	fake_ROI[voxel_index] = 1
	fake_ROI = masker.inverse_transform(fake_ROI)
	
	specific_voxel_Z = resampled_maps[:, voxel_index]
	data_SDMA_Stouffer, data_GLS = utils.compute_weights(resampled_maps)
	data_GLS = data_GLS[:, voxel_index]
	data_SDMA_Stouffer = data_SDMA_Stouffer[:, voxel_index]

	print("GLS:", data_SDMA_Stouffer.mean())
	print("SStouffer:", data_GLS.mean())


	
	plt.close('all')
	# Create a figure and axis
	fig = plt.figure(figsize=(10, 8))
	ax0 = plt.subplot2grid((2,9), (0,0), rowspan=2, colspan=2)
	custom_swarmplot(specific_voxel_Z, ax0, team_names)
	# Customize the plot appearance
	ax0.set_xlabel('Data Point \nraw')
	ax0.set_ylabel('Z Value')
	# if p_values[0] <= 0.05:
	# 	ax0.hlines(p_values[0], 0, 1, color="red")
	# if p_values[1] <= 0.05:
	# 	ax0.hlines(p_values[1], 0, 1, color="blue")
	ax0.set_xlim([-0.01, 0.02])
	# Remove x-axis tick labels
	ax0.set_xticks([])


	ax1 = plt.subplot2grid((2,9), (0,2), rowspan=2, colspan=2)
	custom_swarmplot(data_SDMA_Stouffer, ax1, team_names)
	# Customize the plot appearance
	ax1.set_xlabel('Data Point \nSDMA Stouffer', color="red")
	ax1.set_ylabel('Z Value')
	ax1.set_title('{}'.format(condition))
	ax1.set_xlim([-0.01, 0.02])
	# Remove x-axis tick labels
	ax1.set_xticks([])

	ax2 = plt.subplot2grid((2,9), (0,4), rowspan=2, colspan=2)
	custom_swarmplot(data_GLS, ax2, team_names)
	# Customize the plot appearance
	ax2.set_xlabel('Data Point GLS', color="blue")
	ax2.set_title('{}'.format(condition))
	ax2.set_xlim([-0.01, 0.02])
	# Remove x-axis tick labels
	ax2.set_xticks([])

	# add visu voxel in brain
	ax3 = plt.subplot2grid((2,9), (0,6),  colspan=3)
	for x1, x2 in zip(data_SDMA_Stouffer, data_GLS):
		ax3.plot([0], [x1], 'o', color='red')
		ax3.plot([1], [x2], 'o', color='blue')
		ax3.plot([0, 1], [x1, x2], '-', color='black', alpha=0.2)
	ax3.plot([0, 1], [numpy.mean(data_SDMA_Stouffer), numpy.mean(data_GLS)], '-', color='yellow', linewidth=5)

	# add visu voxel in brain
	ax4= plt.subplot2grid((2,9), (1,6),  colspan=3)
	plotting.plot_stat_map(fake_ROI, annotate=False, vmax=1, colorbar=False, cmap='Blues', axes=ax4)
	plt.tight_layout()
	title = "{}/hyp{}/voxel_per_team/Hyp{}_{}_voxel{}_2methods.png".format(results_dir, hyp, hyp, condition, voxel_index)
	plt.savefig(title)
	# plt.show()





###################################
# plot automatically 2 of them per condition
###################################




conditions = ["consensus0_gls1", "consensus1_gls0", "consensus1_gls1"]
for hyp in outputs.keys():
	# folder to store figure of voxels
	if not os.path.exists(os.path.join(results_dir, "hyp{}/voxel_per_team".format(hyp))):
	    os.mkdir(os.path.join(results_dir, "hyp{}/voxel_per_team".format(hyp)))
	computed_p_values = numpy.load("{}/data/Hyp{}_MA_estimates.npy".format(results_dir, hyp), allow_pickle=True).item()
	print("*****", hyp, "*****")
	resampled_maps_per_team = numpy.load('{}/data/Hyp{}_resampled_maps.npy'.format(results_dir, hyp), allow_pickle=True).item()
	resampled_maps= masker.fit_transform(resampled_maps_per_team.values())
	team_names = list(resampled_maps_per_team.keys())

	for condition in [0, 1, 2]:
		for voxel in [0, -1]:
			v_ind = outputs[hyp][condition][voxel]
			print("running hyp {}, condition {}, voxel {}".format(hyp, conditions[condition], v_ind))
			p_values = [computed_p_values["SDMA Stouffer"]["p_values"][v_ind], computed_p_values["GLS SDMA"]["p_values"][v_ind]]
			plot_voxels_per_team(masker, v_ind, resampled_maps, conditions[condition], team_names, p_values, hyp, results_dir)

plot_voxels_per_team(masker, 0, resampled_maps, "voxel1", team_names, p_values, 1, results_dir)

