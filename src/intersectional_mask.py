from nilearn.datasets import load_mni152_brain_mask
from nilearn import masking
from nilearn import image
import glob, os
import nibabel


##################
# gather NARPS team results that will be used to compute a mask
##################
data_path = os.path.join("data", "NARPS")
results_dir = os.path.join("results", "NARPS")
subjects = []
for path_to_sub in glob.glob(os.path.join(data_path, '*', "hypo1_unthresh.nii.gz")):
	subjects.append(path_to_sub.split('/')[-2])

# "4961_K9P0" only hyp 9 is weird
not_included_maps = ["4951_X1Z4", "5680_L1A8", "5001_I07H",
					"4947_X19V", "4961_K9P0", "4974_1K0E", "4990_XU70",
					"5001_I07H", "5680_L1A8"]


##################
# COMPUTE MASK WITHOUT -NOT INCLUDED TEAM- MASK USING MNI GM MASK
##################
masks = []
for ind, subject in enumerate(subjects):
	print(ind +1, '/', len(subjects), subject)

	# zmaps to remove from mask because weird
	if subject in not_included_maps:
		print(subject, ' Maps not included')
		continue

	for unthreshold_map in glob.glob(os.path.join(data_path, subject, 'hypo*_unthresh.nii.gz')):
		# zmaps to remove from mask because weird
		mask = masking.compute_background_mask(unthreshold_map)
		resampled_mask = image.resample_to_img(
				mask,
				load_mni152_brain_mask(),
				interpolation='nearest')
		masks.append(resampled_mask)

participants_mask = masking.intersect_masks(masks, threshold=0.9, connected=True)
nibabel.save(participants_mask, os.path.join(results_dir, "masking", "participants_mask.nii"))
