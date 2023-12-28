import numpy
import nibabel
import time
from nilearn.input_data import NiftiMasker
import extract_narps_data
import narps_visualisation
import importlib
import utils
from nilearn import plotting
import matplotlib.pyplot as plt

importlib.reload(narps_visualisation)
importlib.reload(utils)

negative_maps_index_per_hyp = [[1, 11, 23, 24, 31, 46, 50],
								[],
								[4, 17, 19, 22, 27, 32, 37, 40, 41, 42, 44, 46, 47, 48, 50, 55, 58],
								[4, 17, 19, 27, 32, 37, 40, 41, 42, 44, 46, 47, 48, 50, 55],
								[32, 45, 58],
								[32, 45],
								[19, 26, 31, 36, 42, 47, 55]
								]


pipeline_weird_weights = [
		[31, 58],
		[0, 7, 59],
		[19],
		[19],
		[58],
		[32],
		[19]]
pipeline_weird_weights_flatten = [31,58,0,7,59,19,19,58,32,19]
pipeline_weird_weights_neg_out = [
		[51],
		[24],
		[20],
		[21],
		[24],
		[24],
		[31]]
pipeline_weird_weights_neg_out_flatten = [51,24,20,21,24,24,31]

results_dir = "results_in_Narps_data"

data_path = '/home/jlefortb/neurovault_narps_open_pipeline/orig/'
participant_mask = nibabel.load("masking/mask_90.nii")

# save mask for inverse transform
masker = NiftiMasker(
    mask_img=participant_mask)

hyps = [1, 2, 5, 6, 7, 8, 9]

map_to_plot = []
map_to_plot_neg_out = []
for ind, hyp in enumerate(hyps):
    print('*****Running hyp ', hyp, '*****')
    # check if resampled_maps already exists:
    try:
        resampled_maps = numpy.load('{}/Hyp{}_resampled_maps.npy'.format(results_dir, hyp), allow_pickle=True)
        print("resampled_maps successfully loaded")
    except:
    	print("relauch resampling first")
    print("Starting Masking...")
    resampled_maps = masker.fit_transform(resampled_maps)
    resampled_maps_neg_out = numpy.delete(resampled_maps, negative_maps_index_per_hyp[ind], axis=0)
    time.sleep(2)
    print("Masking DONE")
    for idx_map_to_plot in pipeline_weird_weights[ind]:
    	print("saving pipeline #", idx_map_to_plot)
    	map_to_plot.append(resampled_maps[idx_map_to_plot])
    for idx_map_to_plot in pipeline_weird_weights_neg_out[ind]:
    	print("saving neg out pipeline #", idx_map_to_plot)
    	map_to_plot_neg_out.append(resampled_maps_neg_out[idx_map_to_plot])

# #### PLOTTING weird maps
plt.close('all')
hyp_order = [1,1,2,2,2,5,6,7,8,9]

for ind, map_ in enumerate(map_to_plot):
	print("plotting pipeline #", pipeline_weird_weights_flatten[ind])
	map_= masker.inverse_transform(map_)
	title = "pipe #{}, hyp{}".format(pipeline_weird_weights_flatten[ind], hyp_order[ind])
	# plotting.plot_stat_map(map_, colorbar=True, cut_coords=(-21, 0, 9), title=title)
	plotting.plot_stat_map(map_, colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', title=title)
	plt.savefig("{}/weird_weight_hyp{}_pipe{}_long_form.png".format(results_dir, hyp_order[ind], pipeline_weird_weights_flatten[ind]))
plt.close('all')

plt.close('all')
hyp_order = [1, 2, 5, 6, 7, 8, 9]
for ind, map_ in enumerate(map_to_plot_neg_out):
	print("plotting pipeline #", pipeline_weird_weights_neg_out_flatten[ind])
	map_= masker.inverse_transform(map_)
	title = "pipe #{}, hyp{}".format(pipeline_weird_weights_neg_out_flatten[ind], hyp_order[ind])
	# plotting.plot_stat_map(map_, colorbar=True, cut_coords=(-21, 0, 9), title=title)
	plotting.plot_stat_map(map_, colorbar=True, cut_coords=(-24, -10, 4, 18, 32, 52, 64), display_mode='z', title=title)
plt.savefig("{}/weird_weight_per_hyp_neg_out.png".format(results_dir))
plt.close('all')