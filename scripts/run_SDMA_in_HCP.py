
import os
import numpy
from nilearn.input_data import NiftiMasker
import compute_MA_outputs
import narps_visualisation
import importlib
from datetime import datetime
import utils
from glob import glob
import pandas

importlib.reload(narps_visualisation)
importlib.reload(utils)


results_dir = "/home/jlefortb/SDMA/results_in_HCP_data"
if not os.path.exists(results_dir):
    os.mkdir(results_dir)

# folder to store extracted resampled z maps
if not os.path.exists(os.path.join(results_dir, "data")):
    os.mkdir(os.path.join(results_dir, "data"))

# save log
dt = datetime.now()
if not os.path.exists("{}/log.txt".format(results_dir)):
    with open("{}/log.txt".format(results_dir), 'w') as f:
        f.write("Last ran the {}/{} at {}h{}".format(dt.day, dt.month, dt.hour, dt.minute))
else:
    with open("{}/log.txt".format(results_dir), 'a') as f:
        f.write("\nLast ran the {}/{} at {}h{}".format(dt.day, dt.month, dt.hour, dt.minute))

MA_estimators_names = ["Average",
    "Stouffer",
    "SDMA Stouffer",
    "Consensus \nSDMA Stouffer",
    "Consensus \nSDMA Stouffer \n using std inputs",
    "Consensus Average",
    "GLS SDMA",
    "Consensus GLS SDMA"]


print('*****loading data*****')
data_path = "/home/jlefortb/SDMA/hcp_data/preprocessed"
intersection_mask = os.path.join(data_path, "mask.nii.gz")
# save mask for inverse transform
masker = NiftiMasker(
    mask_img=intersection_mask)

try:
    resampled_maps = numpy.load('{}/data/resampled_maps.npy'.format(results_dir), allow_pickle=True) # mind result dir
    team_names = numpy.load('{}/data/pipeline_names.npy'.format(results_dir), allow_pickle=True) # mind result dir
    print("resampled_maps successfully loaded")
    # fit masker
    masker.fit(resampled_maps)
except:
    print("Data don't already exist thus starting resampling.")
    nifti_files = glob(os.path.join(data_path, "*.nii"))
    resampled_maps = masker.fit_transform(nifti_files)
    team_names = [file[61:-10] for file in nifti_files]
    print("Saving resampled NARPS unthreshold maps...")
    numpy.save("{}/data/resampled_maps.npy".format(results_dir), resampled_maps, allow_pickle=True, fix_imports=True)
    numpy.save("{}/data/pipeline_names.npy".format(results_dir), team_names, allow_pickle=True, fix_imports=True)


hyp = ""

print("plotting brains...")
narps_visualisation.plot_hcp_maps(resampled_maps, team_names, masker, os.path.join(results_dir, "data"), "resampled")

# compute several MA estimators for the obtained matrix
try:
    MA_outputs = numpy.load('{}/data/MA_estimates.npy'.format(results_dir),allow_pickle=True).item()
    print("MA_outputs successfully loaded")
except:
    print("Data don't already exist thus recomputing MA estimates.")
    MA_outputs = compute_MA_outputs.get_MA_outputs(resampled_maps)
    print("Saving MA estimates...")
    numpy.save("{}/data/MA_estimates".format(results_dir), MA_outputs, allow_pickle=True, fix_imports=True)
print("Building figure 1... distributions")
narps_visualisation.plot_distributions(MA_outputs, hyp, MA_estimators_names, results_dir)
print("Building figure 2... MA results on brains no fdr")
narps_visualisation.plot_brain_nofdr(MA_outputs, hyp, MA_estimators_names, results_dir, masker)
print("Building figure 3... similarities/contrasts...")
similarity_mask = narps_visualisation.plot_SDMA_results_divergence(MA_outputs, hyp, MA_estimators_names, results_dir, masker)
print('Saving weights..')
df_weights = pandas.DataFrame(columns=MA_outputs.keys(), index=team_names)
K, J = resampled_maps.shape
for row in range(K):
    for MA_model in MA_outputs.keys():
        df_weights[MA_model] = MA_outputs[MA_model]['weights']
df_weights["Mean score"] = resampled_maps.mean(axis=1)
df_weights["Var"] = resampled_maps.std(axis=1)
print("Building figure 4... weights")
utils.plot_weights_Narps(results_dir, resampled_maps, df_weights, hyp)
# plot residuals
print("Computing residuals...")
coefficients, residuals_maps = narps_visualisation.compute_betas(resampled_maps)
print("Building figure 5... betas (for residuals)")
narps_visualisation.plot_betas(coefficients, hyp, results_dir, team_names)
print("Building figure 6... residuals")
residuals_maps_per_team = {}
for team, maps in zip(team_names, residuals_maps):
    residuals_maps_per_team[team] = masker.inverse_transform(maps)
narps_visualisation.plot_hcp_maps(masker.fit_transform(list(residuals_maps_per_team.values())), list(residuals_maps_per_team.keys()), masker, os.path.join(results_dir, "data"), "residuals")
resampled_maps, coefficients, residuals_maps, residuals_nii_maps  = None, None, None, None # empyting RAM memory

# narps_visualisation.plot_hyp_similarities(similarity_mask_per_hyp, results_dir)



