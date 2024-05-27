import os
import math
import numpy
import nilearn.plotting
import nilearn.input_data
from nilearn import masking
from nilearn import image
import matplotlib.pyplot as plt
from nilearn.datasets import load_mni152_brain_mask
from nilearn.input_data import NiftiMasker
import pandas
import compute_MA_outputs
import nibabel as nib
from nilearn.datasets import fetch_atlas_aal
import seaborn

##################
# Check how fair is the assumption of same Q accross the brain for all hypotheses
##################


# path to partiticipants mask
participants_mask_path = os.path.join("results" , "NARPS", "masking", "participants_mask.nii")
# path to resampled NARPS data
data_path = os.path.join("data", "NARPS")
# create folder to store results
results_dir = os.path.join("results", "NARPS")
figures_dir = os.path.join("figures", "NARPS")


############################################################
################ STEP 1 : MASKING ##########################
############################################################

##################
# Create AAL masks
# and AAL ROIs
##################



# Load AAL atlas
atlas_aal = fetch_atlas_aal()

# Define ROI in each mask
frontal = ['Frontal_Inf_Oper_L',
    'Frontal_Inf_Oper_R',
    'Frontal_Inf_Orb_L',
    'Frontal_Inf_Orb_R',
    'Frontal_Inf_Tri_L',
    'Frontal_Inf_Tri_R',
    'Frontal_Med_Orb_L',
    'Frontal_Med_Orb_R',
    'Frontal_Mid_L',
    'Frontal_Mid_Orb_L',
    'Frontal_Mid_Orb_R',
    'Frontal_Mid_R',
    'Frontal_Sup_L',
    'Frontal_Sup_Medial_L',
    'Frontal_Sup_Medial_R',
    'Frontal_Sup_Orb_L',
    'Frontal_Sup_Orb_R',
    'Frontal_Sup_R',
    'Rectus_L',
    'Rectus_R',
    ]

occipital =[
    'Occipital_Sup_L',
    'Occipital_Sup_R',
    'Occipital_Mid_L',
    'Occipital_Mid_R',
    'Occipital_Inf_L',
    'Occipital_Inf_R',
    'Calcarine_L',
    'Calcarine_R',
    ]

parietal =[
    'Parietal_Sup_L',
    'Parietal_Sup_R',
    'Parietal_Inf_L',
    'Parietal_Inf_R',
    'Precuneus_L',
    'Precuneus_R',
    'Rolandic_Oper_L',
    'Rolandic_Oper_R',
    'Supp_Motor_Area_L',
    'Supp_Motor_Area_R',
    'SupraMarginal_L',
    'SupraMarginal_R',
    'Paracentral_Lobule_L',
    'Paracentral_Lobule_R',
    'Precentral_L',
    'Precentral_R',
    'Postcentral_L',
    'Postcentral_R',
    "Cuneus_L",
    "Cuneus_R"
    ]

temporal = [
    'Temporal_Inf_L',
    'Temporal_Inf_R',
    'Temporal_Mid_L',
    'Temporal_Mid_R',
    'Temporal_Pole_Mid_L',
    'Temporal_Pole_Mid_R',
    'Temporal_Pole_Sup_L',
    'Temporal_Pole_Sup_R',
    'Temporal_Sup_L',
    'Temporal_Sup_R',
    'Olfactory_L',
    'Olfactory_R',
    'Lingual_L',
    'Lingual_R',
    'Fusiform_L',
    'Fusiform_R',
    'Heschl_L',
    'Heschl_R',
    ]

cingulum = [ 'Cingulum_Ant_L',
    'Cingulum_Ant_R',
    'Cingulum_Mid_L',
    'Cingulum_Mid_R',
    'Cingulum_Post_L',
    'Cingulum_Post_R',
    ]

cerebellum = [
    'Cerebelum_Crus1_L',
    'Cerebelum_Crus1_R',
    'Cerebelum_Crus2_L',
    'Cerebelum_Crus2_R',
    'Cerebelum_3_L',
    'Cerebelum_3_R',
    'Cerebelum_4_5_L',
    'Cerebelum_4_5_R',
    'Cerebelum_6_L',
    'Cerebelum_6_R',
    'Cerebelum_7b_L',
    'Cerebelum_7b_R',
    'Cerebelum_8_L',
    'Cerebelum_8_R',
    'Cerebelum_9_L',
    'Cerebelum_9_R',
    'Cerebelum_10_L',
    'Cerebelum_10_R',
    'Vermis_10',
    'Vermis_1_2',
    'Vermis_3',
    'Vermis_4_5',
    'Vermis_6',
    'Vermis_7',
    'Vermis_8',
    'Vermis_9'
    ]

hypocortical = [ 'Thalamus_L',
    'Thalamus_R',
    'Amygdala_L',
    'Amygdala_R',
    'Angular_L',
    'Angular_R',
    'Caudate_L',
    'Caudate_R',
    'Putamen_L',
    'Putamen_R',
    'ParaHippocampal_L',
    'ParaHippocampal_R',
    'Pallidum_L',
    'Pallidum_R',
    'Insula_L',
    'Insula_R',
    'Hippocampus_L',
    'Hippocampus_R',
    ]

# Get index of each roi to be included in each atlas
indices_frontal = [atlas_aal.indices[i] for i in [atlas_aal.labels.index(roi) for roi in frontal]]
indices_occipital = [atlas_aal.indices[i] for i in [atlas_aal.labels.index(roi) for roi in occipital]]
indices_parietal = [atlas_aal.indices[i] for i in [atlas_aal.labels.index(roi) for roi in parietal]]
indices_temporal = [atlas_aal.indices[i] for i in [atlas_aal.labels.index(roi) for roi in temporal]]
indices_cerebellum = [atlas_aal.indices[i] for i in [atlas_aal.labels.index(roi) for roi in cerebellum]]
indices_hypocortical = [atlas_aal.indices[i] for i in [atlas_aal.labels.index(roi) for roi in hypocortical]]
indices_cingulum = [atlas_aal.indices[i] for i in [atlas_aal.labels.index(roi) for roi in cingulum]]
indices_aal = [atlas_aal.indices[i] for i in [atlas_aal.labels.index(roi) for roi in atlas_aal.labels]]

brain_AAL = nib.load(atlas_aal.maps)
# resample MNI gm mask space
brain_AAL = image.resample_to_img(
                        brain_AAL,
                        load_mni152_brain_mask(),
                        interpolation='nearest')

# load mask made from participant zmaps + MNI brain mask
participants_mask = nib.load(participants_mask_path)

# function to save PNG of mask
def create_ROI_mask(ROI_name, indices, participants_mask=participants_mask, brain_AAL=brain_AAL):
    # get indexes of all voxels from a specific ROI
    indexes_ROI = [numpy.where(brain_AAL.get_fdata() == int(indice)) for indice in indices]
    # create empty brain
    empty_brain_matrix = numpy.zeros(brain_AAL.get_fdata().shape)
    # fill the empty brain with 1 in voxels belonging to the ROI
    for indexes in indexes_ROI:
        empty_brain_matrix[indexes] = 1
    # transform empty_brain_matrix into a 3D brain 
    ROI_img = nilearn.image.new_img_like(brain_AAL, empty_brain_matrix)
    # shape ROI_mask from mask_participant to ensure all voxels are present
    masks = [participants_mask, ROI_img]
    ROI_img = masking.intersect_masks(masks, threshold=1, connected=False)
    print("saving... ",ROI_name)
    nib.save(ROI_img, os.path.join(results_dir, "masking", "{}_mask_AAL.nii".format(ROI_name)))
    # Visualize the resulting image
    nilearn.plotting.plot_roi(ROI_img, title="{} region from the AAL atlas".format(ROI_name))
    plt.savefig(os.path.join(figures_dir,"masking", "{}_mask_AAL.png".format(ROI_name)), dpi=300)
    plt.close('all')
    # empty memory
    ROI_img, masks, empty_brain_matrix, indexes_ROI = None, None,None,None

# get mask for each roi, save the nii and the png
create_ROI_mask('Frontal', indices_frontal)
create_ROI_mask('Occipital', indices_occipital)
create_ROI_mask('Parietal', indices_parietal)
create_ROI_mask('Temporal', indices_temporal)
create_ROI_mask('Cerebellum', indices_cerebellum)
create_ROI_mask('Hypocortical', indices_hypocortical)
create_ROI_mask('Cingulum', indices_cingulum)
create_ROI_mask('GM', indices_aal)

# load each mask in memory
frontal_path = os.path.join(results_dir, "masking", "Frontal_mask_AAL.nii")
occipital_path = os.path.join(results_dir, "masking", "Occipital_mask_AAL.nii")
parietal_path = os.path.join(results_dir, "masking", "Parietal_mask_AAL.nii")
temporal_path = os.path.join(results_dir, "masking", "Temporal_mask_AAL.nii")
insular_path = os.path.join(results_dir, "masking", "Hypocortical_mask_AAL.nii")
cingulum_path = os.path.join(results_dir, "masking", "Cingulum_mask_AAL.nii")
cerebellum_path = os.path.join(results_dir, "masking", "Cerebellum_mask_AAL.nii")
GM_path = os.path.join(results_dir, "masking", "GM_mask_AAL.nii")

# create a ROI of participants mask MINUS AAL mask, to get the remaining voxels not included in AAL
AAL_nii = nib.load(GM_path)
empty_3D = numpy.zeros(AAL_nii.get_fdata().shape)
full_3D = empty_3D + participants_mask.get_fdata() -  AAL_nii.get_fdata()
remaining_voxels_fullmask = nilearn.image.new_img_like(participants_mask, full_3D)
# get the main component of the ROI (not every single voxel)
remaining_voxels_nii = nilearn.masking.compute_background_mask(remaining_voxels_fullmask, opening=4, connected=True)
# remove voxels that are in AAL as well
voxels_already_in_AAL = remaining_voxels_nii.get_fdata() + AAL_nii.get_fdata()
remaining_voxels_nii.get_fdata()[voxels_already_in_AAL==2] = 0
remaining_voxels_nii = nilearn.image.new_img_like(remaining_voxels_nii, remaining_voxels_nii.get_fdata())
nib.save(remaining_voxels_nii, os.path.join(results_dir, "masking", "WM_mask.nii"))
WM_path = os.path.join(results_dir, "masking", "WM_mask.nii")
# empty memory
AAL_nii, empty_3D, full_3D, remaining_voxels_fullmask, remaining_voxels_nii = None, None,None,None,None

ROI_mask_paths = {
    "Frontal": frontal_path, 
    "Occipital": occipital_path, 
    "Parietal": parietal_path, 
    "Temporal": temporal_path, 
    "Insular": insular_path, 
    "Cingulum": cingulum_path, 
    "Cerebellum": cerebellum_path, 
    "WM": WM_path,
    "GM": GM_path,
    "Participants mask": participants_mask_path
    }




###################
## VISUALIZE MASKING
###################

plt.close("all")
f, axs = plt.subplots(8, 3, figsize=(25, 15))  
coords = (20, 0, -23)

# Plot original AAL ROI
for ind, roi_name in enumerate(list(ROI_mask_paths.keys())[:-2]):
    nilearn.plotting.plot_roi(nib.load(ROI_mask_paths[roi_name]), title=roi_name, axes=axs[ind, 0], cut_coords=coords)

# Plot merging of original AAL ROI + participants mask
for ind, roi_name in enumerate(list(ROI_mask_paths.keys())[:-2]):
    masks = [participants_mask, nib.load(ROI_mask_paths[roi_name])]
    intersect_mask = masking.intersect_masks(masks, threshold=1, connected=False) # make sure all voxels were present in the NARPS team results
    nilearn.plotting.plot_roi(intersect_mask, title='{} merged'.format(roi_name),axes=axs[ind, 1], cut_coords=coords)
# Plot participants mask + merged original AAL atlas + participants mask
nilearn.plotting.plot_roi(participants_mask, title='Participants mask', axes=axs[0, 2], cut_coords=coords)
nilearn.plotting.plot_roi(nib.load(GM_path), title='Atlas AAL', axes=axs[1, 2], cut_coords=coords)
masks = [participants_mask, nib.load(GM_path)]
intersect_mask = masking.intersect_masks(masks, threshold=1, connected=False)
nilearn.plotting.plot_roi(intersect_mask, title='AAL merged',axes=axs[2, 2], cut_coords=coords)
intersect_mask, masks = None, None
# create full brain from AAL ROIs
empty_3D = numpy.zeros(nib.load(GM_path).get_fdata().shape)
full_3D = empty_3D
for roi_name in list(ROI_mask_paths.keys())[:-2]:
    full_3D += nib.load(ROI_mask_paths[roi_name]).get_fdata()
rebuilt_segmented_mask = nilearn.image.new_img_like(nib.load(GM_path), full_3D)
nilearn.plotting.plot_roi(rebuilt_segmented_mask, title='ROI_merged_together',axes=axs[3, 2], cut_coords=coords)



axs[4, 2].axis('off')
axs[5, 2].axis('off')
axs[6, 2].axis('off')
axs[7, 2].axis('off')

os.path.join(results_dir, "masking", "WM_mask.nii")
plt.savefig(os.path.join(figures_dir, "masking", "check_masking_AAL.png"))
plt.close("all")

# empty memory
intersect_mask, masks,empty_3D, full_3D, rebuilt_segmented_mask = None, None,None,None,None


for hyp in [1, 2, 5, 6, 7, 8, 9]:
    print("******** ")
    print("RUNNING RESULTS FOR HYP ", hyp)
    print("******** ")
    results_dir = os.path.join("results", "NARPS", "hyp{}".format(hyp))
    figures_dir = os.path.join("figures", "NARPS", "hyp{}".format(hyp))


    ##############################################################
    ################ STEP 2 : FROBENIUS ##########################
    ##############################################################


    # Create the MultiIndex for the columns
    columns = pandas.MultiIndex.from_tuples(
        [("GM as Qref", "Qsi (%)"), ("GM as Qref", "Norm. F"),
        ("GM+WM as Qref", "Qsi (%)"), ("GM+WM as Qref", "Norm. F"), 
        ("Mean diff in corr", "GM"), ("Mean diff in corr", "GM+WM"),
        ("Roi size (%)", ""),
        ("Diff (max,min) with GM", "SDMA Stouffer"),
        ("Diff (max,min) with GM", "Consensus \nSDMA Stouffer"),
        ("Diff (max,min) with GM", "Consensus Average"),
        ("Diff (max,min) with GM", "SDMA GLS"),
        ("Diff (max,min) with GM", "Consensus SDMA GLS"),
        ("Diff (max,min) with GM & WM", "SDMA Stouffer"),
        ("Diff (max,min) with GM & WM", "Consensus \nSDMA Stouffer"),
        ("Diff (max,min) with GM & WM", "Consensus Average"),
        ("Diff (max,min) with GM & WM", "SDMA GLS"),
        ("Diff (max,min) with GM & WM", "Consensus SDMA GLS")
        ])
    df = pandas.DataFrame(index=[list(ROI_mask_paths.keys())[:-2]], columns=columns)

    # Load NARPS 
    resampled_maps_per_team = numpy.load(os.path.join(data_path, "Hyp{}_resampled_maps.npy".format(hyp)), allow_pickle=True).item()


    # function to compute normalized Frobenius norm and Qsi
    def Frobenius(Q_ref, Q_roi):
        # COMPUTE FROBENIUS
        K = Q_roi.shape[0]
        similarity_matrix = Q_roi - Q_ref
        rel_Q_roi = numpy.ones(K).T.dot(Q_roi).dot(numpy.ones(K))/K**2
        rel_Q_ref = numpy.ones(K).T.dot(Q_ref).dot(numpy.ones(K))/K**2
        Qsi = (rel_Q_roi - rel_Q_ref)/rel_Q_ref*100
        normalized_Frobenius = math.sqrt(numpy.mean([elem**2 for row in similarity_matrix for elem in row]))
        return Qsi, normalized_Frobenius

    # GM+WM AS Q_ref 
    masker_GM_WM = NiftiMasker(
        mask_img=participants_mask)
    resampled_maps = masker_GM_WM.fit_transform(resampled_maps_per_team.values())
    Q_ref_GM_WM = numpy.corrcoef(resampled_maps)
    resampled_maps, masker_GM_WM = None, None # empty memory

    # GM ONLY AS Q_ref 
    masker_GM = NiftiMasker(
                    mask_img=nib.load(ROI_mask_paths["GM"]))
    GM_data = masker_GM.fit_transform(resampled_maps_per_team.values())
    Q_ref_GM = numpy.corrcoef(GM_data)
    GM_data, masker_GM = None, None # empty memory

    # for each ROI, compute Frobenius 
    for roi_name in list(ROI_mask_paths.keys())[:-2]:
        print("Starting Frobenius for: {}".format(roi_name))
        # get data within a ROI
        masker_roi = NiftiMasker(
                    mask_img=nib.load(ROI_mask_paths[roi_name]))
        extracted_data = masker_roi.fit_transform(resampled_maps_per_team.values())

        df[("Roi size (%)", "")].loc[roi_name] = numpy.round(nib.load(ROI_mask_paths[roi_name]).get_fdata().sum()*100/(nib.load(ROI_mask_paths["Participants mask"]).get_fdata().sum()), 2)
        # correlation matrix within the ROI
        Q_roi = numpy.corrcoef(extracted_data)
        df[("Mean diff in corr", "GM")].loc[roi_name] = numpy.round(numpy.mean(Q_roi) - numpy.mean(Q_ref_GM), 2)
        df[("Mean diff in corr", "GM+WM")].loc[roi_name] = numpy.round(numpy.mean(Q_roi) - numpy.mean(Q_ref_GM_WM), 2)

        # GM+WM AS Q_ref 
        Qsi, normalized_Frobenius = Frobenius(Q_ref_GM_WM, Q_roi)
        Qsi = numpy.round(Qsi, 2)
        normalized_Frobenius = numpy.round(normalized_Frobenius, 2)
        # save Frobenius results in DF
        df[("GM+WM as Qref", "Qsi (%)")].loc[roi_name] = Qsi
        df[("GM+WM as Qref", "Norm. F")].loc[roi_name] = normalized_Frobenius

        # GM AS Q_ref 
        Qsi, normalized_Frobenius = Frobenius(Q_ref_GM, Q_roi)
        Qsi = numpy.round(Qsi, 2)
        normalized_Frobenius = numpy.round(normalized_Frobenius, 2)
        # save Frobenius results in DF
        df[("GM as Qref", "Qsi (%)")].loc[roi_name] = Qsi
        df[("GM as Qref", "Norm. F")].loc[roi_name] = normalized_Frobenius

        # overwrite at each turn
        df.to_excel(os.path.join(results_dir, "spatial_homogeneity", "Frobenius_score_NARPS.xlsx"))
    extracted_data, masker_roi = None, None # empty memory




    ##############################################################
    ################ BONUS STEP : CORRELATION ####################
    ##############################################################

    from community import community_louvain
    import networkx as nx

    def reorganize_according_to_new_indexing(matrix, partition, team_names=None):
         ''' Reorganized the covariance matrix according to the partition

         Parameters
         ----------
         matrix : correlation matrix (n_roi*n_roi)

         Returns
         ----------
         matrix reorganized

         '''
         # compute the best partition
         reorganized = numpy.zeros(matrix.shape).astype(matrix.dtype)
         labels = range(len(matrix))
         labels_new_order = []

         ## reorganize matrix abscissa wise
         i = 0
         # iterate through all created community
         for values in numpy.unique(list(partition.values())):
            # iterate through each ROI
            for key in partition:
                if partition[key] == values:
                    reorganized[i] = matrix[key]
                    labels_new_order.append(labels[key])
                    i += 1
         # check positionning from original matrix to reorganized matrix
         # get index of first roi linked to community 0
         index_roi_com0_reorganized = list(partition.values()).index(0)
         # get nb of roi in community 0
         nb_com0 = numpy.unique(list(partition.values()), return_counts=True)[1][0]
         assert reorganized[0].sum() == matrix[index_roi_com0_reorganized].sum()

         if team_names==None:
              df_reorganized = pandas.DataFrame(index=labels_new_order, columns=labels, data=reorganized)
         else:
              team_names_new_order = []
              for ind in labels_new_order:
                   team_names_new_order.append(team_names[ind])
              df_reorganized = pandas.DataFrame(index=team_names_new_order, columns=team_names, data=reorganized)
              
         ## reorganize matrix Ordinate wise
         df_reorganized = df_reorganized[df_reorganized.index]
         return df_reorganized



    # compute Q in brain and ROI and reorganize them
    print("RUNNING -Correlation and reorganised correlation-")

    Q_ref = Q_ref_GM_WM.copy()
    # sort Q_ref:
    organised_ind = numpy.argsort(Q_ref, axis=0)
    Q_ref_organized = numpy.take_along_axis(Q_ref, organised_ind, axis=0)

    # sort Q_ref using Louvain
    G = nx.Graph(numpy.abs(Q_ref))  
    partition = community_louvain.best_partition(G, random_state=0)
    df_Q_ref_organized_louvain = reorganize_according_to_new_indexing(Q_ref, partition)


    plt.close('all')
    f, axs = plt.subplots(10, 3, figsize=(6, 30)) 
    # first row correlation matrices 
    seaborn.heatmap(Q_ref, center=0, cmap='coolwarm', robust=True, square=True, ax=axs[0, 0], cbar=False)
    axs[0, 0].set_title("Q_GM_WM" + " Q",fontsize=12)

    # second row reorganized correlation matrices
    seaborn.heatmap(Q_ref_organized, center=0, cmap='coolwarm', robust=True, square=True, ax=axs[0, 1], cbar=False)
    axs[0, 1].set_title("Q_GM_WM" + " sorted Q",fontsize=12)
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])

    # third row reorganized correlation matrices LOUVAIN
    seaborn.heatmap(df_Q_ref_organized_louvain, center=0, cmap='coolwarm', robust=True, square=True, ax=axs[0, 2], cbar=False)
    axs[0, 2].set_title("Q_GM_WM" + " Q Louvain",fontsize=12)


    for i, roi_name in enumerate(list(ROI_mask_paths.keys())[:-1]):
        print("Computing Q for ", roi_name)
        # get data within a ROI
        masker_roi = NiftiMasker(
                    mask_img=nib.load(ROI_mask_paths[roi_name]))
        extracted_data = masker_roi.fit_transform(resampled_maps_per_team.values())
        Q_roi = numpy.corrcoef(extracted_data)
        Q_roi_organized = numpy.take_along_axis(Q_roi, organised_ind, axis=0)
        df_Q_roi_organized_louvain = reorganize_according_to_new_indexing(Q_roi, partition)

        seaborn.heatmap(Q_roi, center=0, cmap='coolwarm', robust=True, square=True, ax=axs[1+i, 0], cbar=False)
        axs[1+i, 0].set_title(roi_name + " Q",fontsize=12)
        seaborn.heatmap(Q_roi_organized, center=0, cmap='coolwarm', robust=True, square=True, ax=axs[1+i, 1], cbar=False)
        axs[1+i, 1].set_title(roi_name + " sorted Q",fontsize=12)
        axs[1+i, 1].set_xticks([])
        axs[1+i, 1].set_yticks([])
        seaborn.heatmap(df_Q_roi_organized_louvain, center=0, cmap='coolwarm', robust=True, square=True, ax=axs[1+i, 2], cbar=False)
        axs[1+i, 2].set_title(roi_name + " Q Louvain",fontsize=12)
    extracted_data, masker_roi = None, None # empty memory
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "spatial_homogeneity", "Q_reorg_Q_among_ROI.png"))
    plt.close('all')



    #######################################################################
    ################ STEP 3 : SEGMENTED ANALYSIS ##########################
    #######################################################################

    ###################################
    # running SDMA methods in all ROIs
    ###################################

    MA_estimators_names = [
        "SDMA Stouffer",
        "Consensus \nSDMA Stouffer",
        "Consensus Average",
        "SDMA GLS",
        "Consensus SDMA GLS"
        ]

    # storing results
    outputs = {}

    # reconstruct mask to make sure all voxels in the roi are in full brain as well
    rebuilt_mask_GM = numpy.zeros(nib.load(ROI_mask_paths["Frontal"]).get_fdata().shape)
    for roi_name in list(ROI_mask_paths.keys())[:-2]:
        roi_mask = nib.load(ROI_mask_paths[roi_name])
        if roi_name == "WM":
            rebuilt_mask_GM_WM = rebuilt_mask_GM + roi_mask.get_fdata()
            assert numpy.array_equal(numpy.unique(rebuilt_mask_GM_WM), [0, 1]), "rebuilt_mask_GM contains elements other than 0 or 1 : {}".format(numpy.unique(rebuilt_mask_GM_WM))

        else: 
            rebuilt_mask_GM += roi_mask.get_fdata()
        assert numpy.array_equal(numpy.unique(rebuilt_mask_GM), [0, 1]), "rebuilt_mask_GM contains elements other than 0 or 1 : {}".format(numpy.unique(rebuilt_mask_GM))

        

    # for each ROI, compute SDMA analysis 
    for roi_name in list(ROI_mask_paths.keys()):
        print("Starting Segmented analysis for: {}".format(roi_name))
        # storing data
        outputs[roi_name] = {}
        # get data within a ROI
        masker_roi = NiftiMasker(
                    mask_img=nib.load(ROI_mask_paths[roi_name]))
        resampled_maps_in_ROI = masker_roi.fit_transform(resampled_maps_per_team.values())
        print("Compute MA estimates in {}".format(roi_name))
        MA_outputs = compute_MA_outputs.get_MA_outputs(resampled_maps_in_ROI)
        resampled_maps_in_ROI = None # empty memory

        print("Saving results per SDMA")
        for row, SDMA_method in enumerate(MA_estimators_names):
            T_map = MA_outputs[SDMA_method]['T_map']
            T_brain = masker_roi.inverse_transform(T_map)
            if roi_name == "GM":
                # ensure all voxels of the brain are in the ROIs
                T_brain = nilearn.image.new_img_like(T_brain, T_brain.get_fdata()*rebuilt_mask_GM)
            elif roi_name == "Participants mask":
                # ensure all voxels of the brain are in the ROIs
                T_brain = nilearn.image.new_img_like(T_brain, T_brain.get_fdata()*rebuilt_mask_GM_WM)
            outputs[roi_name][SDMA_method] = [T_map, T_brain]
    T_brain, T_map = None, None # empty memory



    ################################################################################
    ################ STEP 4 : DIFFERENCE B/W SEGMENTED ANALYSIS AND ORIGINAL #######
    ################################################################################

    ##################################################
    # ASSEMBLING SEGMENTED RESULTS INTO ONE UNIQUE MAP
    ##################################################

    def max_min_diff_per_roi(df, SDMA_method, image_of_differences, where):
        # saving absolute difference regionally
        # create new columns
        for roi_name in list(ROI_mask_paths.keys())[:-2]:
            roi_mask_nii = nib.load(ROI_mask_paths[roi_name])
            max_diff = numpy.round((image_of_differences.get_fdata()*roi_mask_nii.get_fdata()).max(), 2)
            min_diff = numpy.round((image_of_differences.get_fdata()*roi_mask_nii.get_fdata()).min(), 2)
            print(roi_name, "max diff=", max_diff, ", min diff=", min_diff)
            df[("Diff (max,min) with {}".format(where), "{}".format(SDMA_method))].loc[roi_name] = [[max_diff, min_diff]]    
        df.to_excel(os.path.join(results_dir, "spatial_homogeneity", "Frobenius_score_NARPS.xlsx"))


    for SDMA_method in MA_estimators_names:
        print("*** {} **** ".format(SDMA_method))
        print("Extract T values data")
        # get t_brain (t_values in 3d shape)
        T_brain_Frontal = outputs["Frontal"][SDMA_method][1]
        T_brain_Occipital = outputs["Occipital"][SDMA_method][1]
        T_brain_Parietal = outputs["Parietal"][SDMA_method][1]
        T_brain_Temporal = outputs["Temporal"][SDMA_method][1]
        T_brain_Insular = outputs["Insular"][SDMA_method][1]
        T_brain_Cingulum = outputs["Cingulum"][SDMA_method][1]
        T_brain_Cerebellum = outputs["Cerebellum"][SDMA_method][1]
        T_brain_WM = outputs["WM"][SDMA_method][1]


        # rebuild full brain statistics from ROI segmented analysis
        empty_3D = numpy.zeros(T_brain_Frontal.get_fdata().shape)
        full_3D_GM = empty_3D + T_brain_Frontal.get_fdata()+ T_brain_Occipital.get_fdata()+ T_brain_Parietal.get_fdata()+ T_brain_Temporal.get_fdata()+ T_brain_Insular.get_fdata()+ T_brain_Cingulum.get_fdata()+ T_brain_Cerebellum.get_fdata()
        full_3D_GM_WM = full_3D_GM + T_brain_WM.get_fdata()
        # TO DO: make sure there is only 0 and 1 here
        rebuilt_GM = nilearn.image.new_img_like(T_brain_Frontal, full_3D_GM)
        rebuilt_GM_WM = nilearn.image.new_img_like(T_brain_Frontal, full_3D_GM_WM)


        # DIFFERENCE with GM
        T_brain_GM = outputs["GM"][SDMA_method][1]
        
        # "*rebuilt_mask_GM" to ensure all voxels from the brain are in the ROIs
        differences_GM = empty_3D + rebuilt_GM.get_fdata() - T_brain_GM.get_fdata()*rebuilt_mask_GM
        diff_GM_image = nilearn.image.new_img_like(T_brain_Frontal, differences_GM)
        max_min_diff_per_roi(df, SDMA_method, diff_GM_image, "GM")

        # DIFFERENCE with GM & WM
        T_brain_GM_WM = outputs["Participants mask"][SDMA_method][1]
        # "*rebuilt_mask_GM_WM" to ensure all voxels from the brain are in the ROIs
        differences_GM_WM = empty_3D + rebuilt_GM_WM.get_fdata() - T_brain_GM_WM.get_fdata()*rebuilt_mask_GM_WM
        diff_GM_WM_image = nilearn.image.new_img_like(T_brain_Frontal, differences_GM_WM)
        max_min_diff_per_roi(df, SDMA_method, diff_GM_WM_image, "GM & WM")
        
        print("Create figure")
        # plot results
        plt.close('all')
        f, axs = plt.subplots(7, figsize=(12, 30)) 

        max_abs_brain = numpy.abs((T_brain_GM_WM.get_fdata()*rebuilt_mask_GM_WM)).max()

        scale_diff_GM = numpy.abs(diff_GM_image.get_fdata()).max() - numpy.abs(diff_GM_image.get_fdata()).max()*0.1 # 10% less than original value for visibiity
        max_diff_GM = numpy.round((diff_GM_image.get_fdata()).max(), 2)
        min_diff_GM = numpy.round((diff_GM_image.get_fdata()).min(), 2)
        scale_diff_GM_WM = numpy.abs(diff_GM_WM_image.get_fdata()).max() - numpy.abs(diff_GM_WM_image.get_fdata()).max()*0.1 # 10% less than original value for visibiity
        max_diff_GM_WM = numpy.round((diff_GM_WM_image.get_fdata()).max(), 2)
        min_diff_GM_WM = numpy.round((diff_GM_WM_image.get_fdata()).min(), 2)



        cut_coords=(-24, -10, 4, 18, 32, 52)

        nilearn.plotting.plot_stat_map(T_brain_GM, colorbar=True, axes=axs[0], cmap="coolwarm", vmax=max_abs_brain, cut_coords=cut_coords, display_mode='z')
        axs[0].set_title("GM results (AAL mask)",fontsize=20)
        nilearn.plotting.plot_stat_map(rebuilt_GM, colorbar=True, axes=axs[1], cmap="coolwarm", vmax=max_abs_brain, cut_coords=cut_coords, display_mode='z')
        axs[1].set_title("Rebuild segmented analysis from GM",fontsize=20)
        nilearn.plotting.plot_stat_map(diff_GM_image, colorbar=True, axes=axs[2], cmap="coolwarm", vmax=scale_diff_GM, cut_coords=cut_coords, display_mode='z')
        axs[2].set_title("Differences GM (max={}, min={})".format(max_diff_GM, min_diff_GM),fontsize=20)

        axs[3].axis('off')

        nilearn.plotting.plot_stat_map(T_brain_GM_WM, colorbar=True, axes=axs[4], cmap="coolwarm", vmax=max_abs_brain, cut_coords=cut_coords, display_mode='z')
        axs[4].set_title("GM WM results (original with participant mask)",fontsize=20)
        nilearn.plotting.plot_stat_map(rebuilt_GM_WM, colorbar=True, axes=axs[5], cmap="coolwarm", vmax=max_abs_brain, cut_coords=cut_coords, display_mode='z')
        axs[5].set_title("Rebuild segmented analysis from GM WM",fontsize=20)
        nilearn.plotting.plot_stat_map(diff_GM_WM_image, colorbar=True, axes=axs[6], cmap="coolwarm", vmax=scale_diff_GM_WM, cut_coords=cut_coords, display_mode='z')
        axs[6].set_title("Differences GM WM (max={}, min={})".format(max_diff_GM_WM, min_diff_GM_WM),fontsize=20)
       
        plt.suptitle('{}'.format(SDMA_method), fontsize=25)
        plt.savefig(os.path.join(figures_dir, "spatial_homogeneity", "segmented_analysis_{}.png".format(SDMA_method)))
        plt.close('all')

    # empty memory
    T_brain_Frontal,T_brain_Occipital,T_brain_Parietal ,T_brain_Temporal,T_brain_Insular,T_brain_Cingulum ,T_brain_Cerebellum,T_brain_WM = None, None, None, None,None,None,None,None,

    
    plt.close('all')
    for SDMA_method in MA_estimators_names:
        f, axs = plt.subplots(9, 2, figsize=(10, 30)) 
        for i, roi_name in enumerate(list(ROI_mask_paths.keys())[:-1]):
            print("Plotting {} {}".format(SDMA_method, roi_name))
            masker_roi = NiftiMasker(
                        mask_img=nib.load(ROI_mask_paths[roi_name]))
            masker_roi.fit(resampled_maps_per_team.values()) # fit on whole brain
            ROI_stats_in_ROI = masker_roi.transform(outputs[roi_name][SDMA_method][1]).reshape(-1)
            BRAIN_stats_in_ROI = masker_roi.transform(outputs["Participants mask"][SDMA_method][1]).reshape(-1)
            diff = ROI_stats_in_ROI - BRAIN_stats_in_ROI

            axs[i, 0].scatter(range(len(diff)), diff, color="lightblue", marker='x')
            if i == 8:
                axs[i, 0].set_xlabel('Voxel')
            axs[i, 0].set_ylabel('Z value')
            axs[i, 0].axhline(y=0, color='black', linestyle='--', linewidth=2)
            if "GLS" in SDMA_method:
                if "Consensus" in SDMA_method:
                    axs[i, 0].axhline(y=40, color='black', linestyle='--', linewidth=0.5)
                    axs[i, 0].axhline(y=-40, color='black', linestyle='--', linewidth=0.5)
                    axs[i, 0].set_title("{}".format(roi_name))
                    axs[i, 0].set_ylim(-80, 80)

                else:
                    axs[i, 0].axhline(y=6.5, color='black', linestyle='--', linewidth=0.5)
                    axs[i, 0].axhline(y=-6.5, color='black', linestyle='--', linewidth=0.5)
                    axs[i, 0].set_title("{}".format(roi_name))
                    axs[i, 0].set_ylim(-13, 13)

            else:
                axs[i, 0].axhline(y=2.5, color='black', linestyle='--', linewidth=0.5)
                axs[i, 0].axhline(y=-2.5, color='black', linestyle='--', linewidth=0.5)
                axs[i, 0].set_title("{}".format(roi_name))
                axs[i, 0].set_ylim(-5, 5)

            diff_3D = diff.reshape(1, -1)
            diff_3D = masker_roi.inverse_transform(diff_3D)

            nilearn.plotting.plot_stat_map(diff_3D, draw_cross=False, annotate=False, colorbar=True, axes=axs[i, 1], cmap="coolwarm", vmax=2, cut_coords=cut_coords[1:-1], display_mode='z')
            if i == 0:
                axs[i, 1].set_title("Visualisation differences in brain")
        plt.suptitle("{} Z values \n (Stats ROI minus Stats brain in that ROI)".format(SDMA_method))

        plt.savefig(os.path.join(figures_dir, "spatial_homogeneity", "scatter_differences_for_{}.png".format(SDMA_method)))
        plt.close('all')





