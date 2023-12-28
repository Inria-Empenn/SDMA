import numpy
import nibabel
import nilearn.datasets as ds
from nilearn import plotting
from nilearn import image
from nilearn.input_data import NiftiMasker
import matplotlib.pyplot as plt

###################################
# extract voxel's index that is significant or not OR is significant in both in SDMA_Stouffer and GLS in frontal and occipital
###################################

outputs = {}
participant_mask = nibabel.load("masking/mask_90.nii")


weird_maps = ["4951_X1Z4", "5680_L1A8", "5001_I07H", 
    "4947_X19V", "4961_K9P0", "4974_1K0E", "4990_XU70",
        "5001_I07H", "5680_L1A8"]

negative_maps_team_name_per_hyp =[
    ['5164_Q58J',
      '4908_UK24',
      '5649_1P0Y',
      '4979_IZ20',
      '4967_P5F3',
      '4869_4TQ6',
      '4891_80GC'],
     [],
     ['4965_9U7M',
      '4883_6VV2',
      '4866_L7J7',
      '4984_B23O',
      '5637_46CD',
      '4932_AO86',
      '4972_O03M',
      '5619_R42Q',
      '4978_I9D6',
      '4807_0JO0',
      '4821_UI76',
      '4869_4TQ6',
      '4975_27SS',
      '4959_E6R3',
      '4891_80GC',
      '5675_SM54',
      '4881_2T6S'],
     ['4965_9U7M',
      '4883_6VV2',
      '4866_L7J7',
      '4984_B23O',
      '5637_46CD',
      '4932_AO86',
      '4972_O03M',
      '5619_R42Q',
      '4978_I9D6',
      '4807_0JO0',
      '4821_UI76',
      '4869_4TQ6',
      '4975_27SS',
      '4959_E6R3',
      '4891_80GC',
      '5675_SM54'],
     ['4932_AO86', '4963_DC61', '4881_2T6S'],
     ['4932_AO86', '4963_DC61'],
     ['4866_L7J7',
      '4824_43FJ',
      '4967_P5F3',
      '5496_VG39',
      '4807_0JO0',
      '4975_27SS',
      '5675_SM54']]

# fit masker
masker = NiftiMasker(
    mask_img=participant_mask)
results_dir = "results_in_Narps_data_negative_out"

try:
    outputs = numpy.load("{}/check_voxels_distribution_outputs.npy".format(results_dir), allow_pickle=True).item()
    print("outputs successfully loaded")
except:
	for ind, hyp in enumerate([1, 2, 5, 6, 7, 8, 9]):
		outputs[hyp] = {} 
		print("hyp ", hyp, " ***********")
		resampled_maps_per_team = numpy.load('results_in_Narps_data/Hyp{}_resampled_maps.npy'.format(hyp), allow_pickle=True).item()
		print("resampled_maps successfully loaded")
		team_names = list(resampled_maps_per_team.keys())
		for tname in negative_maps_team_name_per_hyp[ind]:
			print("removing anticorrelated team: ", tname)
			del resampled_maps_per_team[tname]
		print("new length:", len(resampled_maps_per_team))
		team_names = list(resampled_maps_per_team.keys())
		resampled_maps = masker.fit_transform(resampled_maps_per_team.values())
		# display following for specific hypothesis number
		results_hyp = numpy.load("/home/jlefortb/SDMA/results_in_Narps_data_negative_out/Hyp{}_MA_estimates.npy".format(hyp), allow_pickle=True).item()

		# get significant voxels for SDMA_Stouffer
		consensus_p = results_hyp["SDMA Stouffer"]["p_values"]
		consensus_t = results_hyp["SDMA Stouffer"]["T_map"]
		consensus_p_brain = masker.inverse_transform(consensus_p)
		consensus_t_brain = masker.inverse_transform(consensus_t)
		# apply threshold
		consensus_p_data = consensus_p_brain.get_fdata()
		consensus_t_data = consensus_t_brain.get_fdata()
		consensus_threshdata = (consensus_p_data <= 0.05)*consensus_t_data #0.05 is threshold significance, *5 to magnify effect
		consensus_threshimg = nibabel.Nifti1Image(consensus_threshdata, affine=consensus_t_brain.affine)

		# get significant voxels for GLS
		GLS_p = results_hyp["GLS SDMA"]["p_values"]
		GLS_t = results_hyp["GLS SDMA"]["T_map"]
		GLS_p_brain = masker.inverse_transform(GLS_p)
		GLS_t_brain = masker.inverse_transform(GLS_t)
		# apply threshold
		GLS_p_data = GLS_p_brain.get_fdata()
		GLS_t_data = GLS_t_brain.get_fdata()
		GLS_threshdata = (GLS_p_data <= 0.05)*GLS_t_data #0.05 is threshold significance, *5 to magnify effect
		GLS_threshimg = nibabel.Nifti1Image(GLS_threshdata, affine=GLS_t_brain.affine)

		atlas_aal = ds.fetch_atlas_aal()
		for roi_label in [['Frontal_Med_Orb_L'], ['Calcarine_L']]: #['Frontal_Sup_Medial_R']
			indices = [atlas_aal.indices[i] for i in [atlas_aal.labels.index(roi) for roi in roi_label]]
			atlas_aal_nii = nibabel.load(atlas_aal.maps)
			# resample in participant mask space
			atlas_aal_nii = image.resample_to_img(
			                    atlas_aal_nii,
			                    participant_mask,
			                    interpolation='nearest')
			# # display ROI
			# fake_ROI = numpy.zeros(atlas_aal_nii.get_fdata().shape)
			# indexes_ROI = [numpy.where(atlas_aal_nii.get_fdata() == int(indice)) for indice in indices]
			# for indexes in indexes_ROI:
			#     fake_ROI[indexes] = 1
			# ROI_img = image.new_img_like(atlas_aal_nii, fake_ROI)
			# plotting.plot_stat_map(ROI_img, title="AAL", annotate=False, vmax=1, colorbar=False, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', cmap='Blues')
			# plt.show()

			# check automatically
			consensus0_gls1 = []
			consensus1_gls0 = []
			consensus1_gls1 = []
			indexes_ROI = numpy.array([numpy.where(atlas_aal_nii.get_fdata() == int(indice)) for indice in indices][0])
			for voxel_nb in range(len(indexes_ROI[0])):
				x, y, z = indexes_ROI[:, voxel_nb]
				voxel_coords= (numpy.array(x), numpy.array(y),numpy.array(z))
				if (GLS_threshimg.get_fdata()[voxel_coords] != 0)&(consensus_threshimg.get_fdata()[voxel_coords] != 0):
					# print("Both are significant in : ", voxel_coords)
					consensus1_gls1.append(voxel_coords)
					print("Found GLS, hyp, ", hyp, " Roi: ", roi_label)
				elif consensus_threshimg.get_fdata()[voxel_coords] != 0:
					# print("Consensus : ", voxel_coords)
					consensus1_gls0.append(voxel_coords)
					print("Found consensus, hyp, ", hyp, " Roi: ", roi_label)
				elif GLS_threshimg.get_fdata()[voxel_coords] != 0:
					# print("GLS : ", voxel_coords)
					consensus0_gls1.append(voxel_coords)
					print("Found BOTH, hyp, ", hyp, " Roi: ", roi_label)
			outputs[hyp][roi_label[0]] = [consensus0_gls1, consensus1_gls0, consensus1_gls1]
	numpy.save("{}/check_voxels_distribution_outputs.npy".format(results_dir), outputs, allow_pickle=True, fix_imports=True)



###################################
# function to plot them
###################################

# Create a custom swarm plot
def custom_swarmplot(data, ax, team_names):
	for i, value in enumerate(data):
		ax.plot(0, value, 'o', markersize=6, alpha=0)
		ax.text(0, value, team_names[i], fontsize=6, ha='center', va='bottom')

def plot_voxels_per_team(atlas_aal_nii, masker, voxel_coords, resampled_maps, condition, roi, team_names):
	plt.close('all')
	# plot each voxel:
	fake_ROI = numpy.zeros(atlas_aal_nii.get_fdata().shape)
	fake_ROI[voxel_coords] = 1
	# plot voxel
	ROI_img = image.new_img_like(atlas_aal_nii, fake_ROI)
	# return to 2D shape
	fake_roi_2D = numpy.squeeze(masker.fit_transform(ROI_img))
	# get index of significant voxel
	voxel_index = numpy.where(fake_roi_2D==1)[0][0]

	# Box plot: p_value per team for a specific voxel
	data = resampled_maps[:, voxel_index]

	ones = numpy.ones((resampled_maps.shape[0], 1))
	Q = numpy.corrcoef(resampled_maps)
	W = ones.T.dot(Q).dot(ones)
	data_SDMA_Stouffer = data/W**-1
	data_SDMA_Stouffer= data_SDMA_Stouffer.reshape(-1)
	W_gls= numpy.sum(Q**-1, axis=1).reshape(-1, 1)
	data_GLS = data.reshape(-1, 1) * W_gls
	data_GLS = data_GLS.reshape(-1)

	# Create a figure and axis
	fig = plt.figure(figsize=(6, 8))
	ax1 = plt.subplot2grid((1,4), (0,0))
	custom_swarmplot(data, ax1, team_names)
	# Customize the plot appearance
	ax1.set_xlabel('Data Point')
	ax1.set_ylabel('Z Value')
	ax1.set_title('{}'.format(condition))
	ax1.set_xlim([-0.01, 0.02])
	# Remove x-axis tick labels
	ax1.set_xticks([])

	# add visu voxel in brain
	ax2 = plt.subplot2grid((1,4), (0,1),  colspan=3)
	plotting.plot_stat_map(ROI_img, annotate=False, vmax=1, colorbar=False, cmap='Blues', axes=ax2)
	plt.tight_layout()
	title = "/home/jlefortb/SDMA/results_in_Narps_data_negative_out/Hyp{}_{}_{}_voxel{}.png".format(hyp, condition, roi, voxel_index)
	plt.savefig(title)
	# plt.show()

	### SECOND FIGURE:
	plt.close('all')
	# Create a figure and axis
	fig = plt.figure(figsize=(6, 8))
	ax1 = plt.subplot2grid((2,7), (0,0),  rowspan=2, colspan=2)
	custom_swarmplot(data_SDMA_Stouffer, ax1, team_names)
	# Customize the plot appearance
	ax1.set_xlabel('Data Point \nSDMA Stouffer')
	ax1.set_ylabel('Z Value')
	ax1.set_title('{}'.format(condition))
	ax1.set_xlim([-0.01, 0.02])
	# Remove x-axis tick labels
	ax1.set_xticks([])

	ax2 = plt.subplot2grid((2,7), (0,2),  rowspan=2, colspan=2)
	custom_swarmplot(data_GLS, ax2, team_names)
	# Customize the plot appearance
	ax2.set_xlabel('Data Point GLS')
	ax2.set_title('{}'.format(condition))
	ax2.set_xlim([-0.01, 0.02])
	# Remove x-axis tick labels
	ax2.set_xticks([])

	# add visu voxel in brain
	ax3 = plt.subplot2grid((2,7), (0,4),  colspan=3)
	for x1, x2 in zip(data_SDMA_Stouffer, data_GLS):
		ax3.plot([0, 1], [x1, x2],'ro-')

	# add visu voxel in brain
	ax4= plt.subplot2grid((2,7), (1,4),  colspan=3)
	plotting.plot_stat_map(ROI_img, annotate=False, vmax=1, colorbar=False, cmap='Blues', axes=ax4)
	plt.tight_layout()
	title = "/home/jlefortb/SDMA/results_in_Narps_data_negative_out/Hyp{}_{}_{}_voxel{}_2methods.png".format(hyp, condition, roi, voxel_index)
	plt.savefig(title)
	# plt.show()


###################################
# plot automatically 2 of them per condition
###################################
atlas_aal = ds.fetch_atlas_aal()
atlas_aal_nii = nibabel.load(atlas_aal.maps)
# resample in participant mask space
atlas_aal_nii = image.resample_to_img(
                    atlas_aal_nii,
                    participant_mask,
                    interpolation='nearest')

for ind, hyp in enumerate(outputs.keys()):
	print("*****", hyp, "*****")
	resampled_maps_per_team = numpy.load('results_in_Narps_data/Hyp{}_resampled_maps.npy'.format(hyp), allow_pickle=True).item()
	print("resampled_maps successfully loaded")
	team_names = list(resampled_maps_per_team.keys())
	for tname in negative_maps_team_name_per_hyp[ind]:
		print("removing anticorrelated team: ", tname)
		del resampled_maps_per_team[tname]
	resampled_maps = masker.fit_transform(resampled_maps_per_team.values())
	team_names = list(resampled_maps_per_team.keys())
	for roi in outputs[hyp].keys():
		for condition in [0, 1, 2]:
			for voxel in [0, -1]:
				print("running hyp {}, roi {}, condition {}, voxel {}".format(hyp, roi, ["Stouffer0_gls1", "Stouffer1_gls0", "Stouffer1_gls1"][condition], voxel))
				try:
					assert outputs[hyp][roi][condition][0] != outputs[hyp][roi][condition][-1]
				except:
					print("no significant voxels for this setting")
					continue
				plot_voxels_per_team(atlas_aal_nii, masker, outputs[hyp][roi][condition][voxel], resampled_maps, ["Stouffer0_gls1", "Stouffer1_gls0", "Stouffer1_gls1"][condition], roi, team_names)