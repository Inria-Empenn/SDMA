import os
import numpy
import nibabel
import time
from nilearn.input_data import NiftiMasker
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


results_dir = "covariance_check"
if not os.path.exists(results_dir):
    os.mkdir(results_dir)

data_path = '/home/jlefortb/neurovault_narps_open_pipeline/orig/'
participant_mask = nibabel.load("masking/mask_90.nii")

# save mask for inverse transform
masker = NiftiMasker(
    mask_img=participant_mask)


#### NOT INCLUDED IN ANALYSIS 

# load narps data hyp 1
resampled_maps_per_team = numpy.load('results_in_Narps_data/data/Hyp1_resampled_maps.npy', allow_pickle=True).item() # mind result dir
print("resampled_maps successfully loaded")
time.sleep(2)
# print("plotting brains...")
# narps_visualisation.plot_nii_maps(resampled_maps_per_team, masker, hyp, os.path.join(results_dir, "data"), "resampled")
print("Starting Masking...")
resampled_maps= masker.fit_transform(resampled_maps_per_team.values())
team_names = list(resampled_maps_per_team.keys())
print("Masking DONE")
print("Z values extracted, shape=", resampled_maps.shape) # 61, 1537403 for hyp 1

scaler = StandardScaler()
resampled_maps_std = scaler.fit_transform(resampled_maps.T).T

print(resampled_maps_std.mean(axis=1)) # [0, 0, 0, ..., 0]
print(resampled_maps_std.std(axis=1)) # [1, 1, 1, ..., 1]

corr_mat = numpy.corrcoef(resampled_maps_std)
cov_mat = numpy.cov(resampled_maps_std)

resampled_maps_mean_substracted = resampled_maps_std - numpy.mean(resampled_maps, axis=0, keepdims=True) #shape=1, 1537403
corr_mat_substract_mean = numpy.corrcoef(resampled_maps_mean_substracted)
cov_mat_substract_mean = numpy.cov(resampled_maps_mean_substracted)
diag_cov = numpy.diag(cov_mat)
diag_cov_sub_mean = numpy.diag(cov_mat_substract_mean)


plt.close('all')
fig = plt.figure(figsize=(12, 15))

ax0 = plt.subplot2grid((4,5), (0,0), rowspan=2, colspan=2)
sns.heatmap(corr_mat, center=0, cmap="coolwarm", square=True, fmt='.1f', cbar_kws={"shrink": 0.25}, figure=fig, ax=ax0)
ax0.title.set_text('Correlation')


ax1 = plt.subplot2grid((4,5), (2,0), rowspan=2, colspan=2)
sns.heatmap(corr_mat_substract_mean, center=0, cmap="coolwarm", square=True, fmt='.1f', cbar_kws={"shrink": 0.25}, figure=fig, ax=ax1)

ax1.title.set_text('Correlation mean_substracted')


ax2 = plt.subplot2grid((4,5), (0,2), rowspan=2, colspan=2)
sns.heatmap(cov_mat, center=0, cmap="coolwarm", square=True, fmt='.1f', cbar_kws={"shrink": 0.25}, figure=fig, ax=ax2)
ax2.title.set_text('Covariance')


ax3 = plt.subplot2grid((4,5), (2,2), rowspan=2, colspan=2)
sns.heatmap(cov_mat_substract_mean, center=0, cmap="coolwarm", square=True, fmt='.1f', cbar_kws={"shrink": 0.25}, figure=fig, ax=ax3)
ax3.title.set_text('Covariance mean_substracted')

ax4 = plt.subplot2grid((4,5), (1,4), rowspan=1, colspan=1)
ax4.boxplot(diag_cov,   labels=["Diag cov"])
ax4.set_yticks([0.99, 1.01])

ax5 = plt.subplot2grid((4,5), (2,4), rowspan=1, colspan=1)
ax5.boxplot(diag_cov_sub_mean,   labels=["Diag cov mean substracted"])


plt.tight_layout()
plt.savefig("{}/matrices.png".format(results_dir))
plt.show()


