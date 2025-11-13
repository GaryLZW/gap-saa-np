#!/usr/bin/env python

from gaphelpers.write_slurm import write_dft_slurm, get_adsorbate_info
from gaphelpers.hyperparameter import get_atomization_energy
from sklearn.decomposition import KernelPCA
from dscribe.descriptors import SOAP
from sklearn.cluster import KMeans
from ase.io import read, write
from ase import neighborlist
import numpy as np, pandas as pd, os, sys
from pathlib import Path
from shutil import copytree
import glob, pickle


def build_SOAP_vector(list_of_atoms, elements):

    species = elements#["H", "C", "O", "Rh"]
    rcut = 4.1
    nmax = 9
    lmax = 3

    # Setting up the SOAP descriptor
    soap = SOAP(
        species=species,
        periodic=True,
        r_cut=rcut,
        n_max=nmax,
        l_max=lmax,
        sigma = 0.23,
        average="inner"
    )

    return soap.create(list_of_atoms)
#	return np.array([mini for mini in soap.create(list_of_atoms)])    


def kpca_kmeans(submitdir, kmeanscluster=5, only_chemisorption=False, code="aims", free_atom_e=None):
    """
    This function takes converged minima ensemble from parallelized minima hopping and sample five representative structures from there.
    To this end, conformer ensemble is embedded into 2D kPCA space using SOAP.
    From 2D kPCA space, K-means clustering is used to partition them into five groups and the lowest Eads from each group is selected.

    Parameters :
    ------------
    submitdir : string
        Directory where submission script is located.
    kmeancluster : int
        Number of samples to be extracted.
    free_atom_e : dict

    Return :
    --------
    representatives : dict
        index of conformer in minima.traj as key and corresponding structure as value
    """

    minima = read(submitdir+ "/a_parallel_minhop" + "/minima.traj@:")

    if len(minima) < 5:
        return {i: atoms for i, atoms in enumerate(minima)}

    else:
        # if only_chemisorption:
        #     physisorption, only_H_bond, minima = filter_physisorption(minima, filter_hydrogen_bond = True, verbose = True)
        #     if len(minima) == 0:
        #         print("There are only physisorbed structures")
        #         minima  =  read(submitdir+ "/a_parallel_minhop" + "/minima.traj@:")
        #
        #     elif len(minima) < kmeanscluster:
        #         return {i: atoms for i, atoms in enumerate(minima)}
        
        elements = minima[0].get_chemical_symbols()
        # minima_soap = soap.create(minima)
        minima_soap = build_SOAP_vector(minima, elements=elements)

        AE = np.array([get_atomization_energy(atoms, atoms.get_potential_energy(apply_constraint=False), free_atom_e=free_atom_e) for atoms in minima])
        kernel_pca = KernelPCA(n_components=2, kernel="poly")
        minima_soap_kernel_pca = kernel_pca.fit_transform(minima_soap)
        clustering = KMeans(n_clusters=kmeanscluster, random_state=1).fit(minima_soap_kernel_pca)
        print("clustering done!")

        # Make dataframe
        minima_dict = {}
        for i, atoms in enumerate(minima):
            minima_dict[i] = {"ase" : atoms,
                              "AE" : get_atomization_energy(atoms, atoms.get_potential_energy(apply_constraint=False), code=code, free_atom_e=free_atom_e),
                              "label" : clustering.labels_[i]
                             }
        df = pd.DataFrame.from_dict(minima_dict).transpose()
        df["AE"] = df.AE.astype("float")


        # Sample representative points
        representatives = [df.loc[df.label == g].AE.idxmin() for g in np.unique(clustering.labels_)]
        print("Representative index : ", representatives)

        # minima_dict = {i: minima[i] for i in representatives}

        return {i: minima[i] for i in representatives}



def filter_physisorption(minima_list, filter_hydrogen_bond = True, verbose = False):

    physisorption = []
    chemisorption = []
    only_H_bond = []
    
    atom_dict = {i : symb for i, symb in enumerate(minima_list[0].get_chemical_symbols()) if symb in ['C', 'H', 'O']}
    adsorbateidx = sorted(list(atom_dict.keys()))[0]
    
    for i, atoms in enumerate(minima_list):
        cutOff = neighborlist.natural_cutoffs(atoms) # generate covalent radii for each elements in ase object

        neighbor = neighborlist.NeighborList(cutOff, self_interaction=False, bothways=True)
        neighbor.update(atoms)

        adjacency = neighbor.get_connectivity_matrix(sparse=False)
        total_binding_number = np.count_nonzero(adjacency[adsorbateidx:,:adsorbateidx])
        if total_binding_number == 0:
            physisorption.append(atoms)
            continue
        
        if filter_hydrogen_bond:
            Hydrogen_indices = [index for index, symbol in atom_dict.items() if symbol == "H"]
            Num_H_bonds = 0
            for H_index in Hydrogen_indices:
                Num_H_bonds += np.count_nonzero(adjacency[H_index,:adsorbateidx])

            if total_binding_number == Num_H_bonds:
                only_H_bond.append(atoms)
            else:
                chemisorption.append(atoms)
                
        else:
            chemisorption.append(atoms)
                
    if verbose:
        print(f"Total minima  : {len(minima_list)}")
        print(f"Physisorption : {len(physisorption)}")
        print(f"Only H-bonded : {len(only_H_bond)}")
        print(f"chemisorpiton : {len(chemisorption)}")
        
    return physisorption, only_H_bond, chemisorption


def cluster_sample(submitdir, singlepoint=False, submit=True, kmeanscluster=10, only_chemisorption=True,
                   code="aims", f_name="b_DFT_relaxation", free_atom_e=None, path_to_workflow=None):
    """
    Perform DFT single point calculation / geometry relaxation on sampled representative structures.
    Ultimately DFT relaxation is needed, but single point calculation is required when the accuracy of
    converged GAP need to be confirmed.

    Using the sampled structure, this function submits DFT calculation after modifing submission script.
    For specific environment this function might be modified.

    Parameters :
    ------------
    smiles : string
        Smiles string of current adsorbate. Required just for naming purposes.
    submitdir : string
        Directory where submission script is located.
    surface : string
        Surface (facet) of the model. Required just for naming purposes.
    singlepoint : bool
        Whether to perform single point or geometry relaxation.
    submit : bool
        Wheter to submit job
    free_atom_e : dict
    path_to_workflow : str
    """

    minima_dict = kpca_kmeans(submitdir, only_chemisorption=only_chemisorption, kmeanscluster=kmeanscluster, free_atom_e=free_atom_e)

    if singlepoint:

        f_name = "/c_DFT_singlepoint4cluster"

        job_ids = []
        # if using_opt_adsorbate:
        #     idx, formula, smiles = get_adsorbate_info(using_opt_adsorbate=using_opt_adsorbate, adsorbate_file=adsorbate_file)
        # else:
        #     idx, formula, smiles = get_adsorbate_info(smiles)
        for i, index in enumerate(minima_dict.keys()):
            Path("/".join([submitdir, f_name, f"{i}_structure"] )).mkdir(parents = True, exist_ok = True)
            os.chdir("/".join([submitdir, f_name, f"{i}_structure"] ))
            with open(f"structure_{index}.pickle", "wb") as h:
                 pickle.dump(minima_dict[index], h, protocol=pickle.HIGHEST_PROTOCOL)

            write_dft_slurm(f"{i}-min{index}-struc", code=code,
                            target_file_name=f"structure_{index}.pickle", path_to_workflow=path_to_workflow)
            #os.system('sed -i "7s/.*/#SBATCH --ntasks=40/" submit.sh')
            #if code == "aims":
            #    os.system(f'sed -i "36s/.*/ase_submit.py -sp structure_{index}.pickle/" submit.sh')
            #elif code == "qe":
            #    os.system(f'sed -i "37s/.*/ase_submit.py -qe -sp structure_{index}.pickle/" submit.sh')


            if submit:
                num = os.popen("sbatch submit.sh").read()
                job_ids.append(num.strip().split()[-1])

        return job_ids

    else:

#	if only_chemisorption:
#			f_name = "/b_DFT_relaxation_filtered"
#		else:
#			f_name = "/b_DFT_relaxation"

        Path(f"{submitdir}/{f_name}").mkdir(parents=True, exist_ok=True)
        os.chdir(f"{submitdir}/{f_name}")

        for i, conf_index in enumerate(minima_dict.keys()):
            Path("/".join([submitdir, f_name, f"{i}_structure"] )).mkdir(parents = True, exist_ok = True)
            os.chdir("/".join([submitdir, f_name, f"{i}_structure"] ))
            with open(f"structure_{conf_index}.pickle", "wb") as h:
                pickle.dump(minima_dict[conf_index], h, protocol=pickle.HIGHEST_PROTOCOL)

            write_dft_slurm(f"{i}-struc_{conf_index}", code=code, time="24:00:00",
                            target_file_name=f"structure_{conf_index}.pickle", path_to_workflow=path_to_workflow,)

            #os.system('sed -i "7s/.*/#SBATCH --ntasks=40/" submit.sh')
            #os.system('sed -i "10s/.*/#SBATCH --time=24:00:00/" submit.sh')
            if code == "aims":
                os.system(f'sed -i "35s/.*/ase_submit.py -rm -ts -dr structure_{conf_index}.pickle/" submit.sh')
            elif code == "qe":
                os.system(f'sed -i "37s/.*/ase_submit.py -qe -rm -dr structure_{conf_index}.pickle /" submit.sh')

            if submit:
                os.system(f'sbatch submit.sh')


