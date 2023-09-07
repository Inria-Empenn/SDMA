import os
import glob
from nilearn import image
from os.path import join as opj

def resample_NARPS_unthreshold_maps(data_path, hyp, subjects_removed_list, mask):
    print("Extracting data from hypothesis ", hyp)
    # extract subject list
    subjects_list = []
    for path_to_sub in glob.glob(opj(data_path, "*/hypo{}_unthresh.nii.gz".format(hyp))):
        subjects_list.append(path_to_sub.split('/')[-2])
    # Resample unthreshold maps for a given hypothesis + mask
    resampled_maps = []
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
        # check in narps if it was done this way !! => yes, see "suivi jeremy" doc
        assert resampled_map.get_fdata().shape == mask.get_fdata().shape
        # print("Testing equality (if nothing is displayed before ***, matrices are equal)")
        # numpy.testing.assert_array_equal(resampled_map.affine, mask.affine)
        # print("***")
        resampled_maps.append(resampled_map)
        # if i_sub%20==0: # debugging every 20 maps
        #     plt.close('all')
        #     plotting.plot_stat_map(nibabel.load(unthreshold_map), cut_coords=(-21, 0, 9))
        #     plotting.plot_stat_map(resampled_map, cut_coords=(-21, 0, 9))
        #     plt.show()
        resampled_map = None # emptying RAM memory
    print("Resample DONE")
    return resampled_maps

if __name__ == "__main__":
   print('This file is intented to be used as imported only')