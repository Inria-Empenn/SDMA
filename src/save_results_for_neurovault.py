import os
import numpy
import nibabel
from nilearn.input_data import NiftiMasker
from glob import glob

#### NARPS ####

participants_mask_path = os.path.join("results", "NARPS", "masking", "participants_mask.nii")
participant_mask = nibabel.load(participants_mask_path)

masker = NiftiMasker(
    mask_img=participant_mask)

MA_estimators_names = ["Stouffer",
    "SDMA Stouffer",
    "Consensus \nSDMA Stouffer",
    "Consensus Average",
    "SDMA GLS",
    "Consensus SDMA GLS"]

results_dir = os.path.join("results", "NARPS")

hyps = [1, 2, 5, 6, 7, 8, 9]
for hyp in hyps:
	results_dir_hyp = os.path.join(results_dir, "hyp{}".format(hyp), "SDMA")

	# load data
	print("***")
	print("loading data hyp", hyp)
	data_path = os.path.join("data", "NARPS")
	resampled_maps_per_team = numpy.load(os.path.join(data_path, "Hyp{}_resampled_maps.npy".format(hyp)), allow_pickle=True).item()
	resampled_maps = masker.fit_transform(resampled_maps_per_team.values())

	MA_outputs = numpy.load(os.path.join(results_dir_hyp, "Hyp{}_MA_estimates.npy".format(hyp)), allow_pickle=True).item()
	for title in MA_estimators_names[1:]:
		print("saving nifti for ", title)
		T_map=MA_outputs[title]['T_map']
		p_values=MA_outputs[title]['p_values']
		ratio_significance = MA_outputs[title]['ratio_significance']
		p_brain = masker.inverse_transform(p_values)
		t_brain = masker.inverse_transform(T_map)
		# apply threshold
		pdata = p_brain.get_fdata()
		tdata = t_brain.get_fdata()
		threshdata = (pdata <= 0.05)*tdata #0.05 is threshold significance
		threshimg = nibabel.Nifti1Image(threshdata, affine=t_brain.affine)
		long_title = title + '_hyp{}.nii'.format(hyp)
		if "\n" in long_title:
			long_title = long_title.replace('\n', '')
		long_title = long_title.replace(' ', '_')
		saving_path = os.path.join("manuscript", "neurovault", long_title)

		nibabel.save(threshimg, saving_path)


#### HCP ####

# path to partiticipants mask
participants_mask_path = os.path.join("results", "HCP", "masking", "participant_mask.nii.gz")
participant_mask = nibabel.load(participants_mask_path)
# path to resampled HCP data
data_path = os.path.join("data", "HCP")
# create folder to store results
results_dir = os.path.join("results", "HCP")

masker = NiftiMasker(
    mask_img=participant_mask)

MA_estimators_names = ["Stouffer",
    "SDMA Stouffer",
    "Consensus \nSDMA Stouffer",
    "Consensus Average",
    "SDMA GLS",
    "Consensus SDMA GLS"]

nifti_files = glob(os.path.join(data_path, "*.nii"))
resampled_maps = masker.fit_transform(nifti_files)

MA_outputs = numpy.load(os.path.join(results_dir, "SDMA", "MA_estimates.npy"), allow_pickle=True).item()
for title in MA_estimators_names[1:]:
	print("saving nifti for ", title)
	T_map=MA_outputs[title]['T_map']
	p_values=MA_outputs[title]['p_values']
	ratio_significance = MA_outputs[title]['ratio_significance']
	p_brain = masker.inverse_transform(p_values)
	t_brain = masker.inverse_transform(T_map)
	# apply threshold
	pdata = p_brain.get_fdata()
	tdata = t_brain.get_fdata()
	threshdata = (pdata <= 0.05)*tdata #0.05 is threshold significance
	threshimg = nibabel.Nifti1Image(threshdata, affine=t_brain.affine)
	long_title = title
	if "\n" in long_title:
		long_title = long_title.replace('\n', '')
	long_title = long_title.replace(' ', '_')
	saving_path = os.path.join("manuscript", "neurovault", "HCP", long_title)
	nibabel.save(threshimg, saving_path)