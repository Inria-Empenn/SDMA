# Check how fair is the assumption of same Q accross the brain for all hypotheses
import os
import glob
import numpy
import nibabel
import nilearn.plotting
import nilearn.input_data
from nilearn import masking
from nilearn import image
import warnings
import matplotlib.pyplot as plt
from nilearn.datasets import load_mni152_brain_mask
import seaborn
import importlib
from community import community_louvain
import networkx as nx

import utils

importlib.reload(utils) # reupdate imported codes, useful for debugging

##################
# SETTING UP NECESSARY INFO
##################

data_path = '/home/jlefortb/neurovault_narps_open_pipeline/orig/'
results_dir = 'results_Q_assumptions'
if not os.path.exists(results_dir):
    os.mkdir(results_dir)
if not os.path.exists('results_Q_assumptions/temp'):
    os.mkdir('results_Q_assumptions/temp')

#### NOT INCLUDED IN ANALYSIS 
# "4961_K9P0" only hyp 9 is weird
weird_maps = ["4951_X1Z4", "5680_L1A8", "5001_I07H",
              "4947_X19V", "4961_K9P0", "4974_1K0E", "4990_XU70",
              "5001_I07H", "5680_L1A8"]

subjects = []
for path_to_sub in glob.glob("/home/jlefortb/neurovault_narps_open_pipeline/orig/*/hypo1_unthresh.nii.gz"):
    subject = path_to_sub.split('/')[-2]
    if subject in weird_maps:
        print(subject, ' got a weird map thus not included')
        continue
    else:
        subjects.append(subject)

hyp_nums = [1, 2, 5, 6, 7, 8, 9]

##################
# Create AAL masks
##################
from nilearn.datasets import fetch_atlas_aal
atlas_aal = fetch_atlas_aal()
frontal = ['Frontal_Sup_L',
 'Frontal_Sup_R',
 'Frontal_Sup_Orb_L',
 'Frontal_Sup_Orb_R',
 'Frontal_Mid_L',
 'Frontal_Mid_R',
 'Frontal_Mid_Orb_L',
 'Frontal_Mid_Orb_R',
 'Frontal_Inf_Oper_L',
 'Frontal_Inf_Oper_R',
 'Frontal_Inf_Tri_L',
 'Frontal_Inf_Tri_R',
 'Frontal_Inf_Orb_L',
 'Frontal_Inf_Orb_R',
 'Frontal_Sup_Medial_L',
 'Frontal_Sup_Medial_R',
 'Frontal_Med_Orb_L',
 'Frontal_Med_Orb_R']

occipital =[
 'Occipital_Sup_L',
 'Occipital_Sup_R',
 'Occipital_Mid_L',
 'Occipital_Mid_R',
 'Occipital_Inf_L',
 'Occipital_Inf_R'
]
parietal =[
 'Parietal_Sup_L',
 'Parietal_Sup_R',
 'Parietal_Inf_L',
 'Parietal_Inf_R',
]
temporal = [
 'Temporal_Sup_L',
 'Temporal_Sup_R',
 'Temporal_Pole_Sup_L',
 'Temporal_Pole_Sup_R',
 'Temporal_Mid_L',
 'Temporal_Mid_R',
 'Temporal_Pole_Mid_L',
 'Temporal_Pole_Mid_R',
 'Temporal_Inf_L',
 'Temporal_Inf_R'
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
 'Cerebelum_10_R'
 ]

indices_frontal = [atlas_aal.indices[i] for i in [atlas_aal.labels.index(roi) for roi in frontal]]
indices_occipital = [atlas_aal.indices[i] for i in [atlas_aal.labels.index(roi) for roi in occipital]]
indices_parietal = [atlas_aal.indices[i] for i in [atlas_aal.labels.index(roi) for roi in parietal]]
indices_temporal = [atlas_aal.indices[i] for i in [atlas_aal.labels.index(roi) for roi in temporal]]
indices_cerebellum = [atlas_aal.indices[i] for i in [atlas_aal.labels.index(roi) for roi in cerebellum]]
indices_aal = [atlas_aal.indices[i] for i in [atlas_aal.labels.index(roi) for roi in atlas_aal.labels]]

atlas_aal_nii = nibabel.load(atlas_aal.maps)
# resample MNI gm mask space
atlas_aal_nii = image.resample_to_img(
                        atlas_aal_nii,
                        load_mni152_brain_mask(),
                        interpolation='nearest')

# function to save PNG of mask
def compute_save_display_mask(ROI_name, indices):
    # compute ROI mask
    indexes_ROI = [numpy.where(atlas_aal_nii.get_fdata() == int(indice)) for indice in indices]
    fake_ROI = numpy.zeros(atlas_aal_nii.get_fdata().shape)
    for indexes in indexes_ROI:
        fake_ROI[indexes] = 1
    ROI_img = nilearn.image.new_img_like(atlas_aal_nii, fake_ROI)
    # shape ROI_mask from mask_participant to ensure all voxels are present
    mask_participant = nibabel.load('masking/mask_90.nii') # load mask made from participant zmaps + MNI brain mask
    masks = [mask_participant, ROI_img]
    ROI_img = masking.intersect_masks(masks, threshold=1, connected=True)
    print("saving... ",ROI_name)
    nibabel.save(ROI_img, "{}/temp/{}_mask.nii".format(results_dir, ROI_name))
    # Visualize the resulting image
    nilearn.plotting.plot_roi(ROI_img, title="{} regions of AAL atlas".format(ROI_name))
    plt.savefig('{}/{}_mask.png'.format(results_dir, ROI_name), dpi=300)
    plt.close('all')


compute_save_display_mask('Frontal_aal', indices_frontal)
compute_save_display_mask('occipital_aal', indices_occipital)
compute_save_display_mask('parietal_aal', indices_parietal)
compute_save_display_mask('temporal_aal', indices_temporal)
compute_save_display_mask('cerebellum_aal', indices_cerebellum)
compute_save_display_mask('brain_aal', indices_aal)



##############
# create each masker with the AAL roi masked with the mask made with all subjects to ensure data is present
##############

# Compute Q matrices per hypothesis and save them

masks_for_masker = glob.glob('{}/temp/*mask.nii'.format(results_dir))
masks_for_masker.append('masking/mask_90.nii')
names = [mask_for_masker.split('/')[-1][:-4] for mask_for_masker in masks_for_masker]

for hyp in hyp_nums:
    unthreshold_maps = [os.path.join(data_path, sub, 'hypo{}_unthresh.nii.gz'.format(hyp)) for sub in subjects]
    unthreshold_maps.sort()
    
    # need to resample to get same affine for each
    unthreshold_maps_resampled = []
    for ind, file in enumerate(unthreshold_maps):
        print("Doing {}/{} for hyp : {}".format(ind, len(unthreshold_maps), hyp))
        # create resampled file
        # ignore nilearn warnings
        # these occur on some of the unthresholded images
        # that contains NaN values
        # we probably don't want to set those to zero
        # because those would enter into interpolation
        # and then would be treated as real zeros later
        # rather than "missing data" which is the usual
        # intention
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            resampled_gm = nilearn.image.resample_to_img(
                file,
                load_mni152_brain_mask(),
                interpolation='nearest')
        try:
            unthreshold_maps_resampled.append(resampled_gm)
        except:
            print('pb with', file)

    # compute Q with all other masks
    for ind, mask_for_masker in enumerate(masks_for_masker):
        print("masking...", names[ind])
        masker = nilearn.input_data.NiftiMasker(
            mask_img=mask_for_masker)
        data = masker.fit_transform(unthreshold_maps_resampled)
        print("Computing Q and saving...")
        Q = numpy.corrcoef(data)      
        numpy.save('{}/temp/Q_{}_hyp{}'.format(results_dir, names[ind], hyp), Q)




##############
# Plot Q matrices per hypothesis 
##############
# organized + raw matrices
partitioning = {}
for hyp in hyp_nums:
    print(hyp)
    correlation_matrices = glob.glob('{}/temp/Q_*_hyp{}.npy'.format(results_dir, hyp))
    correlation_matrices.sort()
    # put the participant mask at index 0 to fit louvain and sorting according
    # to participant mask and not frontal mask (originaly at index 0)
    new_order = [3, 1, 0, 4, 5, 6, 2]
    correlation_matrices = [correlation_matrices[ind] for ind in new_order]

    # load reference matrix (correlation matrix with participant mask) for similarity computation
    matrix_reference_path = '{}/temp/Q_mask_90_hyp{}.npy'.format(results_dir, hyp)
    matrix_reference = numpy.load(matrix_reference_path)


    f, axs = plt.subplots(4, 8, figsize=(25, 15))  
    for ind, matrice in enumerate(correlation_matrices):
        matrix = numpy.load(matrice)
        if ind == 0:
            organised_ind = numpy.argsort(matrix, axis=0)
            # build louvain community graph for this specific hypothesis
            G = nx.Graph(numpy.abs(matrix))  
            partition = community_louvain.best_partition(G, random_state=0)
            partitioning['hyp{}_partition'.format(hyp)]=[partition, G, matrix]
        matrix_organized_louvain = utils.reorganize_with_louvain_community(matrix, partition)
        matrix_organized = numpy.take_along_axis(matrix, organised_ind, axis=0)
        if ind < 4:
            row = ind
            col = 0
        else:
            row = ind - 4
            col = 1

        if matrice.split('/')[-1] == "Q_mask_90_hyp{}.npy".format(hyp):
            name_roi = "participant_mask"
        else:
            name_roi = matrice.split('/')[-1][2:-18]
        title = name_roi + ' ' + str(numpy.round(numpy.mean(numpy.load(matrice))*100, 1))
        title_organized = name_roi


        if matrice != matrix_reference_path:
            # similarity_matrix = sklearn.metrics.pairwise.cosine_similarity(matrix, matrix_reference)
            similarity_matrix = matrix - matrix_reference
            similarity_matrix_ratio = matrix/matrix.shape[0]**2 / matrix_reference/matrix.shape[0]**2
            similarity_matrix_perc_diff = ((matrix/matrix.shape[0]**2 - matrix_reference/matrix.shape[0]**2)/matrix_reference/matrix.shape[0]**2)*100
            # Frobenius Norm => (Sum(abs(value)**2))**1/2
            Fro = numpy.linalg.norm(similarity_matrix, ord='fro')
            Fro_div2 = numpy.linalg.norm(similarity_matrix/2, ord='fro')
            Fro_ratio = numpy.linalg.norm(similarity_matrix_ratio, ord='fro')
            Fro_perc_diff = numpy.linalg.norm(similarity_matrix_perc_diff, ord='fro')

            title_similarity = (name_roi 
                + '\n{}|{}|{}|{}%'.format(numpy.round(Fro, 1), numpy.round(Fro_div2, 1), numpy.round(Fro_ratio, 1), numpy.round(Fro_perc_diff , 1)))
            seaborn.heatmap(similarity_matrix, center=0, cmap='coolwarm', robust=True, square=True, ax=axs[row, col+6], cbar_kws={'shrink': 0.6})
            axs[row, col+6].title.set_text(title_similarity)


        seaborn.heatmap(matrix, center=0, cmap='coolwarm', robust=True, square=True, ax=axs[row, col], cbar_kws={'shrink': 0.6})
        seaborn.heatmap(matrix_organized, center=0, cmap='coolwarm', robust=True, square=True, ax=axs[row, col+2], cbar_kws={'shrink': 0.6})
        seaborn.heatmap(matrix_organized_louvain, center=0, cmap='coolwarm', robust=True, square=True, ax=axs[row, col+4], cbar_kws={'shrink': 0.6})


        axs[row, col].title.set_text(title)
        axs[row, col+2].title.set_text(title_organized)
        axs[row, col+4].title.set_text(title_organized)
    axs[0, 6].axis('off') # get rid of reference mask used for similarity matrix

    axs[0, 6].text(0.1, 0.7, 'frobenius score as:') 
    axs[0, 6].text(0.1, 0.6, '    a|b|c|d') 
    axs[0, 6].text(0.1, 0.5, 'a: Qi -Qb') 
    axs[0, 6].text(0.1, 0.4, 'b: (Qi-Qb)/2') 
    axs[0, 6].text(0.1, 0.3, 'c: (Qi/K**2)/(Qb/K**2)') 
    axs[0, 6].text(0.1, 0.2, 'd: ((Qi/K**2)-(Qb/K**2))/(Qb/K**2)') 


    if hyp == 9:
        axs[-2, 5].axis('off') # get rid of matrice using mask from narps (it's empty)
        axs[-2, 1].axis('off') # get rid of matrice using mask from narps (it's empty)
        axs[-2, 3].axis('off') # get rid of matrice using mask from narps (it's empty)
        axs[-2, 7].axis('off') # get rid of matrice using mask from narps (it's empty)
    axs[-1, 5].axis('off') # get rid of matrice using mask from narps (it's empty)
    axs[-1, 1].axis('off') # get rid of matrice using mask from narps (it's empty)
    axs[-1, 3].axis('off') # get rid of matrice using mask from narps (it's empty)
    axs[-1, 7].axis('off') # get rid of matrice using mask from narps (it's empty)


    plt.suptitle('hyp  {}'.format(hyp), size=16, fontweight='bold')
    f.subplots_adjust(top=0.78) 
    plt.figtext(0.1,0.95,"Original", va="center", ha="center", size=12, fontweight='bold')
    plt.figtext(0.35,0.95,"Sorted : Intensity", va="center", ha="center", size=12, fontweight='bold')
    plt.figtext(0.6,0.95,"Sorted : Louvain", va="center", ha="center", size=12, fontweight='bold')
    plt.figtext(0.87,0.95,"Similarity matrix", va="center", ha="center", size=12, fontweight='bold')
    line = plt.Line2D((.75,.75),(.1,.9), color="k", linewidth=3)
    f.add_artist(line)
    plt.tight_layout()
    plt.savefig('{}/hyp_{}.png'.format(results_dir, hyp), dpi=300)
    plt.close('all')





def build_both_graph_heatmap(matrix, G, partition, saving_name, title_graph, title_heatmap, subjects, hyp):
    from community import community_louvain
    import networkx as nx
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    shapes = 'so^>v<dph8'

    f, axs = plt.subplots(1, 2, figsize=(30, 15)) 
    # draw the graph
    pos = nx.spring_layout(G, seed=0)
    # color the nodes according to their partition
    colors = ['blue', 'yellow', 'green', 'red', 'darkviolet', 'orange', "yellowgreen", 'lime', 'crimson', 'aqua']
    nx.draw_networkx_edges(G, pos, ax=axs[0], alpha=0.06)#, min_source_margin=, min_target_margin=)
    inv_map = {k: subjects[k] for k, v in partition.items()}
    for node, color in partition.items():
        nx.draw_networkx_nodes(G, pos, [node], ax=axs[0], node_size=900,
                               node_color=[colors[color]], margins=-0.01, alpha=0.35)
        #Now only add labels to the nodes you require (the hubs in my case)
        nx.draw_networkx_labels(G,pos,inv_map, ax=axs[0], font_size=10, font_color='black')
    axs[0].set_title(title_graph, fontsize=16)

    legend_labels = []
    for com_nb in range(max(partition.values())+1):
        patch = mpatches.Patch(color=colors[com_nb], label='Community {}'.format(com_nb))
        legend_labels.append(patch)
    axs[0].legend(handles=legend_labels, loc='lower left', handleheight=0.2)

    # draw heatmap
    matrix_organized_louvain = utils.reorganize_with_louvain_community(matrix, partition)
    labels = [subjects[louvain_index] + "_c{}".format(partition[louvain_index]) for louvain_index in matrix_organized_louvain.columns]
    seaborn.heatmap(matrix_organized_louvain, center=0, cmap='coolwarm', robust=True, square=True, ax=axs[1], cbar_kws={'shrink': 0.6})
    axs[1].set_title(title_heatmap, fontsize=16)
    N_team = matrix_organized_louvain.columns.__len__()
    axs[1].set_xticks(range(N_team), labels=labels, rotation=90, fontsize=7)
    axs[1].set_yticks(range(N_team), labels=labels, fontsize=7)
    plt.suptitle("Hypothesis {}".format(hyp), fontsize=20)
    plt.savefig(saving_name, dpi=300)
    # plt.show()
    plt.close('all')



# build a matrix which summarize all hypothese louvain community into one community
matrix_graph = numpy.zeros((len(subjects), len(subjects)))
# teams per partition
for key_i in partitioning.keys():
    print('\n***** Doing ****')
    print(key_i)
    hyp = key_i[3]
    print('*****')
    matrix = partitioning[key_i][2]
    G = partitioning[key_i][1]
    partition = partitioning[key_i][0]
    # draw both heatmap and graph
    title_graph = "Community of Narps team"
    title_heatmap = "Heatmap (Louvain organized) correlation matrix of Narps team"
    saving_name = '{}/graph_and_heatmap_hyp{}.png'.format(results_dir, hyp)
    build_both_graph_heatmap(matrix, G, partition, saving_name, title_graph, title_heatmap, subjects, hyp)

    # build summary matrix for alltogether matrix
    # nb_partitions = numpy.unique(list(partitioning[key_i].values()))[-1]
    for key_j in partitioning[key_i][0].keys():
        community_key_j = partitioning[key_i][0][key_j]
        for team in range(len(partitioning[key_i][0].keys())):
            if team == key_j:
                continue
            if partitioning[key_i][0][team] == community_key_j:
                print(partitioning[key_i][0][team], " == ", community_key_j, ' thus adding 1 at row: ', subjects[team], " col: ", subjects[key_j])
                matrix_graph[team][key_j] += 1

# all together graph
G = nx.Graph(matrix_graph, seed=0)
# compute the best partition
partition = community_louvain.best_partition(G, random_state=0)
title_graph = "Community of Narps team"
title_heatmap = "Heatmap (Louvain organized) based on occurence \nof belonging to the same community across each NARPS hypothesis"
saving_name = '{}/graph_and_heatmap_all_hyp.png'.format(results_dir)
build_both_graph_heatmap(matrix_graph, G, partition, saving_name, title_graph, title_heatmap, subjects, "All")






##############
# BUILD FINAL FIGURES FOR PAPER
##############

# compute frobenius score for each hypothesis
for hyp in hyp_nums:
    print(hyp)
    correlation_matrices = glob.glob('{}/temp/Q_*_hyp{}.npy'.format(results_dir, hyp))
    correlation_matrices.sort()
    # put the participant mask at index 0 to fit louvain and sorting according
    # to participant mask and not frontal mask (originaly at index 0)
    new_order = [3, 1, 0, 4, 5, 6, 2]
    correlation_matrices = [correlation_matrices[ind] for ind in new_order]

    # load reference matrix (correlation matrix with participant mask) for similarity computation
    matrix_reference_path = '{}/temp/Q_mask_90_hyp{}.npy'.format(results_dir, hyp)
    matrix_reference = numpy.load(matrix_reference_path)


    for ind, matrice in enumerate(correlation_matrices):
        matrix = numpy.load(matrice)
        if matrice.split('/')[-1] == "Q_mask_90_hyp{}.npy".format(hyp):
            name_roi = "participant_mask"
        else:
            name_roi = matrice.split('/')[-1][2:-18]
        
        if matrice != matrix_reference_path:
            similarity_matrix_perc_diff = ((matrix/matrix.shape[0]**2 - matrix_reference/matrix.shape[0]**2)/matrix_reference/matrix.shape[0]**2)*100
            similarity_matrix_ratio = ((matrix - matrix_reference)/matrix_reference)*100
            # Frobenius Norm => (Sum(abs(value)**2))**1/2
            Fro_ratio = numpy.linalg.norm(similarity_matrix_ratio, ord='fro')
            Fro_ratio_diff = numpy.linalg.norm(similarity_matrix_perc_diff, ord='fro')
            
            print("hyp {}, with atlas {}, Frobenius norm = {}%, diff = {}%".format(hyp, name_roi, Fro_ratio,Fro_ratio_diff))
