from nilearn.datasets import load_mni152_gm_mask, load_mni152_brain_mask
from nilearn import masking
from nilearn import plotting
from nilearn import image
import glob, os
import matplotlib.pyplot as plt
import warnings
import numpy
import nibabel


##################
# gather all 3D brain data that will be used to compute a mask
##################
subjects = []
for path_to_sub in glob.glob("/home/jlefortb/neurovault_narps_open_pipeline/orig/*/hypo1_unthresh.nii.gz"):
	subjects.append(path_to_sub.split('/')[-2])
data_path = '/home/jlefortb/neurovault_narps_open_pipeline/orig/'


##################
# display all mask next to zmaps to check for weird masking
##################
plt.close('all')
for i_sub, subject in enumerate(subjects):
	unthreshold_maps = glob.glob(os.path.join(data_path, '{}/hypo*unthresh.nii.gz'.format(subject)))
	unthreshold_maps.sort()
	print('resampling sub ', subject, " ", i_sub, " / ", len(subjects))
	for ind, unthreshold_map in enumerate(unthreshold_maps):
		hyp = unthreshold_map.split('/')[-1][4]
		# create figure and build first map
		print('hyp ', hyp)
		if ind == 0:
			mask = masking.compute_background_mask(unthreshold_map)
			# resampled_map = image.resample_to_img(
		    #                 mask,
		    #                 load_mni152_brain_mask(),
		    #                 interpolation='nearest')
			f, axs = plt.subplots(len(unthreshold_maps), 2, figsize=(20, 35))
			# addmask
			plotting.plot_img(mask, cut_coords=(-21, 0, 9), figure=f, axes=axs[0, 0])
			title = "hyp " + hyp
			axs[0, 0].set_title(title,fontsize=40)
			# add raw zmaps
			plotting.plot_stat_map(nibabel.load(unthreshold_map), cut_coords=(-21, 0, 9), figure=f, axes=axs[0, 1])
		# add a map
		else:
			mask = masking.compute_background_mask(unthreshold_map)
			# resampled_map = image.resample_to_img(
		    #                 mask,
		    #                 load_mni152_brain_mask(),
		    #                 interpolation='nearest')
			# add mask
			plotting.plot_img(mask, cut_coords=(-21, 0, 9),  figure=f, axes=axs[ind, 0])
			title = "hyp " + hyp
			axs[ind, 0].set_title(title,fontsize=40)
			# add raw zmaps
			plotting.plot_stat_map(nibabel.load(unthreshold_map), cut_coords=(-21, 0, 9), figure=f, axes=axs[ind, 1])
		# save the figure
		if ind == len(unthreshold_maps)-1:
			print('saving...')
			plt.suptitle(subject,fontsize=50)
			plt.savefig("masking/debugmask/debugmask_{}.png".format(subject))
			plt.close('all')


# "4961_K9P0" only hyp 9 is weird
# weird_maps = ["4951_X1Z4", "5680_L1A8", "4888_L3V8", "4908_UK24", "5001_I07H", "4947_X19V", "4961_K9P0", "4966_3TR7", "4974_1K0E", "4990_XU70",
# 		"5001_I07H", "5680_L1A8"]
weird_maps = ["4951_X1Z4", "5680_L1A8", "5001_I07H", 
    "4947_X19V", "4961_K9P0", "4974_1K0E", "4990_XU70",
        "5001_I07H", "5680_L1A8"]
##################
# display WEIRD masks
##################
plt.close('all')
f, axs = plt.subplots(5, 4, figsize=(20, 25))
for i_sub, subject in enumerate(weird_maps):
	unthreshold_map = os.path.join(data_path, '{}/hypo9_unthresh.nii.gz'.format(subject))
	mask = masking.compute_background_mask(unthreshold_map)
	if i_sub < 5:
		col = 0
		row = i_sub
	else:
		col = 2
		row = i_sub - 5

	plotting.plot_img(mask, cut_coords=(-21, 0, 9), figure=f, axes=axs[row, col])
	title = subject
	axs[row, col].set_title(title,fontsize=15)
	# add raw zmaps
	plotting.plot_stat_map(nibabel.load(unthreshold_map), cut_coords=(-21, 0, 9), figure=f, axes=axs[row, col+1])
axs[4, 2].set_axis_off()
axs[4, 3].set_axis_off()
print('saving...')
plt.suptitle("Weird maps removed",fontsize=20)
plt.savefig("masking/weird_maps_removed.png")
plt.close('all')


##################
# COMPUTE MASK WITHOUT WEIRD TEAM MASK USING MNI GM MASK
##################
masks = []
for ind, subject in enumerate(subjects):
	print(ind, '/', len(subjects), subject)

	# zmaps to remove from mask because weird
	if subject in weird_maps:
		print(subject, ' got a weird map thus passed')
		continue

	for unthreshold_map in glob.glob(os.path.join(data_path, '{}/hypo*_unthresh.nii.gz'.format(subject))):
		# zmaps to remove from mask because weird
		mask = masking.compute_background_mask(unthreshold_map)
		resampled_mask = image.resample_to_img(
				mask,
				load_mni152_brain_mask(),
				interpolation='nearest')
		masks.append(resampled_mask)

##################
# COMPUTE MASK FOR DIFFERENT THRESHOLDS AND DISPLAY IT
##################
plt.close('all')
thresholds = numpy.arange(0.9, 1.01, 0.01)
f1, axs1 = plt.subplots(11, figsize=(8, 20))
f2, axs2 = plt.subplots(11, figsize=(8, 20))
for row, t in enumerate(thresholds):
	print(row, " thresh: ", t)
	participants_mask = masking.intersect_masks(masks, threshold=t, connected=True)
	plotting.plot_img(participants_mask, cut_coords=(-21, 0, 9),  figure=f2, axes=axs2[row])
	axs2[row].set_title('t={}'.format(int(t*100)),fontsize=15)
	# save the final mask
	if t == 0.9:
		nibabel.save(participants_mask, "masking/mask_{}.nii".format(int(t*100)))
	reshape_as_MNI_gm_mask = [participants_mask, load_mni152_brain_mask()]
	participants_mask = masking.intersect_masks(reshape_as_MNI_gm_mask, threshold=1, connected=True)
	plotting.plot_img(participants_mask, cut_coords=(-21, 0, 9),  figure=f1, axes=axs1[row])
	axs1[row].set_title('t={}'.format(int(t*100)),fontsize=15)
	# save the final mask
	if t == 0.9:
		nibabel.save(participants_mask, "masking/mask_{}_noMNI.nii".format(int(t*100)))

plt.show()


f1.savefig("masking/participant_masks_90-100.png") 
f2.savefig("masking/participant_masks_90-100_noMNI.png") 
plt.close('all')






# ##################
# # COMPUTE MASK USING NARPS RESULTS
# ##################

# # save narps results as nii img
# thresh = 0.95
# hypnums = [1, 2, 5, 6, 7, 8, 9]
# for i, hyp in enumerate(hypnums):
# 	print(hyp)
# 	pmap = '/home/jlefortb/narps_open_pipelines/IBMA/results_consensus_analysis/hypo{}_1-fdr.nii.gz'.format(hyp)
# 	tmap = '/home/jlefortb/narps_open_pipelines/IBMA/results_consensus_analysis/hypo{}_t.nii.gz'.format(hyp)
# 	pimg = nibabel.load(pmap)
# 	timg = nibabel.load(tmap)
# 	pdata = pimg.get_fdata()
# 	tdata = timg.get_fdata()[:, :, :, 0]
# 	threshdata = (pdata > thresh)*tdata
# 	print("Should be ", (pdata > thresh).sum(), " voxels")
# 	threshimg = nibabel.Nifti1Image(threshdata, affine=timg.affine)
# 	nibabel.save(threshimg, "masking/hyp{}_narps.nii.gz".format(hyp))
# 	narps_mask = masking.compute_background_mask(threshimg)
# 	nibabel.save(narps_mask, "masking/hyp{}_narps_mask.nii.gz".format(hyp))
# 	print("and is ", narps_mask.get_fdata().sum(), " voxels")
