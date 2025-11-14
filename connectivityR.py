import importlib.util
import os
import numpy as np
import nibabel as nb

def load_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    # if spec is None:
        # print(f"Cannot find module {module_name} at {file_path}")
        # raise ImportError(f"Cannot find module {module_name} at {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Update the paths to the actual locations of your scripts
if os.name == 'nt':
    connT = load_module_from_file('make_local_connectivity_tcorr', r"C:\github\CAPS\cluster_roi_master\make_local_connectivity_tcorr.py")
    connS = load_module_from_file('make_local_connectivity_scorr', r"C:\github\CAPS\cluster_roi_master\make_local_connectivity_scorr.py")
    groupMean = load_module_from_file('group_mean_binfile_parcellation', r"C:\github\CAPS\cluster_roi_master\group_mean_binfile_parcellation.py")
elif os.name == 'posix':
    connT = load_module_from_file('make_local_connectivity_tcorr', r"/media/sf_github/CAPS/cluster_roi_master/make_local_connectivity_tcorr.py")
    connS = load_module_from_file('make_local_connectivity_scorr', r"/media/sf_github/CAPS/cluster_roi_master/make_local_connectivity_scorr.py")
    groupMean = load_module_from_file('group_mean_binfile_parcellation', r"/media/sf_github/CAPS/cluster_roi_master/group_mean_binfile_parcellation.py")
    
else: # /home/raulh87/mnt/a471/userdata/raulh87/github/CAPS/cluster_roi_master
    connT = load_module_from_file('make_local_connectivity_tcorr', r"/home/raulh87/mnt/a471/userdata/raulh87/github/CAPS/cluster_roi_master/make_local_connectivity_tcorr.py")
    connS = load_module_from_file('make_local_connectivity_scorr', r"/home/raulh87/mnt/a471/userdata/raulh87/github/CAPS/cluster_roi_master/make_local_connectivity_scorr.py")
    groupMean = load_module_from_file('group_mean_binfile_parcellation', r"/home/raulh87/mnt/a471/userdata/raulh87/github/CAPS/cluster_roi_master/group_mean_binfile_parcellation.py")

def corr(datafolder, sub_N, run_N, mask_name, corr_type, dataset, task, specie, threshold=0.5, session='', rnd=False, rnd_N=0):
    # Construct the input file path
    input_folder = os.path.join(datafolder, dataset, 'normalized', f'{specie}-sub-{str(sub_N).zfill(2)}')
    if session:
        input_folder += f'_ses-{session}'
    input_file = os.path.join(
        input_folder,
        f'{specie}-sub-{str(sub_N).zfill(2)}_task-{task}_run-{str(run_N).zfill(2)}.nii.gz')

    # Construct the output file path
    if rnd: # same folder but inside the folder rnd, added padded zeros
        output_folder = os.path.join(datafolder, dataset, 'hierarchical_clustering', 'rnd', f'{specie}-sub-{str(sub_N).zfill(3)}')
        output_file = os.path.join(output_folder, f'{specie}-sub-{str(sub_N).zfill(3)}_{str(rnd_N).zfill(4)}')
    else:
        output_folder = os.path.join(datafolder, dataset, 'hierarchical_clustering', f'{specie}-sub-{str(sub_N).zfill(3)}')
        output_file = os.path.join(output_folder, f'{specie}-sub-{str(sub_N).zfill(3)}')
    if session:
        output_file += f'_ses-{session}'
    output_file += f'_task-{task}_run-{str(run_N).zfill(2)}-{corr_type}corr.npy'

    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        print('Creating folder:', output_folder)
        os.makedirs(output_folder)

    print('Output file:', output_file)
    if os.name == 'nt':
        maskFile = os.path.join(datafolder, dataset, 'masks', f'{mask_name}.nii.gz')
    else:
        maskFile = os.path.join(datafolder, dataset, 'masks', f'{mask_name}.nii.gz')
        #userdata/raulh87
    print('Mask file:', maskFile)

    # Running the actual connectivity
    if corr_type == 't':
        connT.make_local_connectivity_tcorr(input_file, maskFile, output_file, threshold)
    elif corr_type == 's':
        connS.make_local_connectivity_scorr(input_file, maskFile, output_file, threshold)

def group_mean(datafolder, subs_possible, runs_possible, corr_type, number_of_clusters, mask_name, dataset, task, specie, session=''):
    corr_conn_files = []
    for sub_N in subs_possible:
        for run_N in runs_possible:
            input_folder = os.path.join(datafolder, dataset, 'hierarchical_clustering', f'{specie}-sub-{str(sub_N).zfill(3)}')
            input_file = os.path.join(input_folder, f'{specie}-sub-{str(sub_N).zfill(3)}')
            if session:
                input_file += f'_ses-{session}'
            input_file += f'_task-{task}_run-{str(run_N).zfill(2)}-{corr_type}corr.npy'
            corr_conn_files.append(input_file)

    maskFile = os.path.join(datafolder, dataset, 'masks', f'{mask_name}.nii.gz')
    mask = nb.load(maskFile)
    mask_data = mask.get_fdata()
    n_voxels = np.count_nonzero(mask_data)
    file_out = os.path.join(
        datafolder, dataset, 'hierarchical_clustering',
        f'{specie}-group-{corr_type}corr-vox{n_voxels}-{mask_name}.npy'
    )

    groupMean.group_mean_binfile_parcellate(corr_conn_files, file_out, number_of_clusters, n_voxels)
    return corr_conn_files

               



def pyClusterROI_corr(project_dict, sub_N, run_N, mask_name, corr_type, threshold, atlas_version='2mm'):
    # This never worked. I went back to connectivityR.corr
    # print warning deprecated
    Warning('This function never worked, use others in connectivityR.corr instead')

    # Calculate temporal/spatial correlation, uses the pyClusterROI toolbox from Cameron Craddock (https://github.com/ccraddock/cluster_roi)
    # project_dict: dictionary containing the project information
    # sub_N: subject number
    # run_N: run number
    # mask_name: name of the mask, the mask will be loaded from the associated atlas folder indicated in project_dict
    # corr_type: type of correlation ('t' for temporal or 's' for spatial)
    # threshold: threshold for the correlation matrix
    dataset = project_dict['Dataset']
    session = project_dict['Session']
    task = project_dict['Task']
    specie = project_dict['Specie']
    datafolder = project_dict['Datafolder']
    atlas = project_dict['Atlas_type']
    # input_folder has the normalized data
    input_folder = datafolder + os.sep + dataset + os.sep + 'normalized' + os.sep + specie + '-sub-' + str(sub_N).zfill(3)
    input_file = input_folder 
    if session != '':
        input_file = input_file + '_ses-' + session
    input_file = input_file + os.sep + specie + '-sub-' + str(sub_N).zfill(3) + '_task-' + task + '_run-' + str(run_N).zfill(2) + '.nii.gz'
    # output_folder has the results of the clustering
    output_folder = datafolder + os.sep + dataset + os.sep + 'hierarchical_clustering' + os.sep + specie + '-sub-' + str(sub_N).zfill(3) + os.sep + specie + '-sub-' + str(sub_N).zfill(3) + os.sep
    output_file = output_folder 
    if session != '':
        output_file = output_file + '_ses-' + session
    output_file = output_file + '_task-' + task + '_run-' + str(run_N).zfill(2) + '-' + mask_name + '-' + corr_type + 'corr.npy'
    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Check if mask_name has the extension .nii.gz, if it does assign to maskfile
    if mask_name.endswith('.nii.gz'):
        maskfile = mask_name
    else: # build the path to the mask file based on the path for the atlas
        # get the directory of the current script
        script_dir = os.getcwd()
        if project_dict['Specie'] == 'H':
            specieS = 'Hum'
        elif project_dict['Specie'] == 'D':
            specieS = 'Dog'

        # get the path to the mask in the atlas folder
        maskfile = os.path.join(script_dir, '..', 'dog_brain_toolkit', 'Atlas', specieS, atlas , atlas + atlas_version, mask_name + '.nii.gz')
        maskfile = os.path.normpath(maskfile)



    # # run make_local_connectivity based on the type of function 
    # if corr_type == 't':
    #     make_local_connectivity_tcorr(input_file, maskfile, output_file, threshold)
    # elif corr_type == 's':
    #     make_local_connectivity_scorr(input_file, maskfile, output_file, threshold)
    # else:
    #     print('Error: invalid correlation type, valid options are "t" for temporal or "s" for spatial')
    
