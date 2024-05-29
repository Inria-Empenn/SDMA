import os
import glob
from nilearn import image
from os.path import join as opj
import numpy
import nibabel

def resample_NARPS_unthreshold_maps(data_path, hyp, subjects_removed_list, mask):
    print("Extracting data from hypothesis ", hyp)
    # extract subject list
    subjects_list = []
    for path_to_sub in glob.glob(opj(data_path, "*/hypo{}_unthresh.nii.gz".format(hyp))):
        subjects_list.append(path_to_sub.split('/')[-2])
    # Resample unthreshold maps for a given hypothesis + mask
    resampled_maps = {}
    for i_sub, subject in enumerate(subjects_list):
        if i_sub%10==0:
            print('resample image ', i_sub, '/', len(subjects_list))
        # zmaps to remove from mask because weird
        if subject in subjects_removed_list:
            print(subject, ' got a weird map thus not included')
            continue
        unthreshold_map = os.path.join(data_path, '{}/hypo{}_unthresh.nii.gz'.format(subject, hyp))
        ## DEBUGGING
        # print('MAP shape: ', nibabel.load(unthreshold_map).get_fdata().shape)
        # print('MNI shape: ', mask.get_fdata().shape)
        # resample MNI
        resampled_map = image.resample_to_img(
                    unthreshold_map,
                    mask,
                    interpolation='nearest')
        assert resampled_map.get_fdata().shape == mask.get_fdata().shape
        resampled_maps[subject] = resampled_map
        # if i_sub%20==0: # debugging every 20 maps
        #     plt.close('all')
        #     plotting.plot_stat_map(nibabel.load(unthreshold_map), cut_coords=(-21, 0, 9))
        #     plotting.plot_stat_map(resampled_map, cut_coords=(-21, 0, 9))
        #     plt.show()
        resampled_map = None # emptying RAM memory
    print("Resample DONE")
    return resampled_maps


# path to partiticipants mask
participants_mask_path = os.path.join("results" , "NARPS", "masking", "participants_mask.nii")
participant_mask = nibabel.load(participants_mask_path)

# data_path = '/home/jlefortb/neurovault_narps_open_pipeline/orig/'
data_path = os.path.join("data", "NARPS")
# path to save resampled NARPS data
output_path = os.path.join("data", "NARPS")


#### NOT INCLUDED IN ANALYSIS 
# "4961_K9P0" only hyp 9 is weird
weird_maps = ["4951_X1Z4", "5680_L1A8", "5001_I07H", 
    "4947_X19V", "4961_K9P0", "4974_1K0E", "4990_XU70",
        "5001_I07H", "5680_L1A8"]

for hyp in [1, 2, 5, 6, 7, 8, 9]:
    resampled_maps_per_team = resample_NARPS_unthreshold_maps(data_path, hyp, weird_maps, participant_mask)
    print("Saving resampled NARPS unthreshold maps...")
    numpy.save(os.path.join(output_path, "Hyp{}_resampled_maps.npy".format(hyp)), resampled_maps_per_team, allow_pickle=True, fix_imports=True)

