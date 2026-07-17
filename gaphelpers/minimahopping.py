#!/usr/bin/env python

from ase.io import read, write
from ase.io.aims import read_aims_output
from collections import deque
import glob, random, pickle
from pathlib import Path
from shutil import copyfile, rmtree
import argparse, os, numpy as np, pandas as pd
import time, igraph as ig
from write_slurm import write_dft_slurm, get_adsorbate_info
from encode import make_trainingset_file, parse_output, add_dummy_cell
from cluster import build_SOAP_vector, cluster_sample
from get_argparse import get_minhop_args
from gap_command import get_SOAP_param_dict
from hyperparameter import get_atomization_energy

from wfl.configset import ConfigSet, OutputSpec
from wfl.descriptors.quippy import from_any_to_Descriptor as calc_descriptors
from wfl.select.by_descriptor import greedy_fps_conf_global
#####################
from ase.optimize.minimahopping import MinimaHopping
from ase.constraints import FixAtoms, Hookean

try:
    from quippy.potential import Potential
except:
    pass
from ase import neighborlist
from ase.optimize.minimahopping import MHPlot
from ase import Atoms
from ase.build import add_adsorbate
from ase.io import Trajectory
import ase

from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops
from io import StringIO
from os.path import dirname, abspath
import matplotlib
from sklearn.metrics import root_mean_squared_error as rmse

matplotlib.use("agg")
####################
from ase.optimize import MDMin
import subprocess
from rdkit2ase import rdkit2ase, ase2rdkit
from mace.calculators import mace_mp
from ase.optimize import BFGS
import random
from ase.cluster import Octahedron, wulff_construction


def test_short_distance(structure, mindist=0.3):
    """
    Function for examining if given structure contains too short distance between atoms.
    mindist is set to 0.3 Angs which safely avoids any error from FHI-AIMS.

    Parameters :
    ------------
    structure : ase Atoms object
        Structure to be examined.
    mindist : float
        Minimum allowed distance between atoms.

    Return :
    Boolean

    """
    distancematrix = structure.get_all_distances()
    distancematrix[np.diag_indices(distancematrix.shape[0])] = 1
    if np.min(distancematrix) < mindist:
        return False
    else:
        return True


def sample_training_set(iteration, submit=True, forcemask=False, collect_job_id=True, parallel=False,
                        sampling_method="stratified_random", calculate_all_minima=False, code="aims",
                        path_to_workflow=None, universal_soap_yaml=None):
    """
    This function samples 5 geometries from previous minima hopping and submit single point DFT calculation.
    Three geometries are from minima including one global and two local minima.
    Two other geometries are randomly sampled from MD trajectory
    If sampled geometry have too short distance, then it resampled.

    parameters :
    ------------
    iteration : integer
        Which iteration are you in for GAP fitting
    submit : bool
        Whether jobs to be submitted right after sampling
    forcemask : bool
        Whether forces of lowermost three metal layers are excluded from GAP fitting
    collect_job_id : bool
        For SLURM job submission with dependency, job IDs are printed for sleeping next iteration fitting.
    sampling_method : str
        Sampling strategy for training set: Either "FPS" or "Recipe" can be used.

    """
    cwd = os.getcwd()
    print(cwd)
    os.chdir(cwd + f"/iter_{iteration}/2_Minhop")
    job_ids = []

    minimas = read("minima.traj@:")

    ################
    # This part is added for rebuttal: to calculate DFT reference for all minima
    # structures found in serial minima hopping
    if calculate_all_minima:
        f_name = cwd + f"/iter_{iteration}/2_Minhop/"
        Path(f_name).mkdir(parents=True, exist_ok=True)
        traj = Trajectory(f_name + "/structure.traj", mode="w")
        for idx, atoms in enumerate(minimas):
            traj.write(atoms)
        traj.close()

        if submit:
            os.chdir(f_name)
            if np.all([os.path.exists(f"{idx}_structure/stdout.log") for idx, atoms in enumerate(minimas)]):
                print("There is a output file in each folder. Continue")
            else:
                write_dft_slurm("minima_calculation", forcemask=forcemask, code=code, path_to_workflow=path_to_workflow)
                num = os.popen("sbatch submit.sh").read()
                job_ids.append(num.strip().split()[-1])

        os.chdir(cwd + f"/iter_{iteration}/2_Minhop")
    ################

    Path(cwd + f"/iter_{iteration}/3_DFT_minhop").mkdir(parents=True, exist_ok=True)
    os.system(f"rsync -az {cwd}/iter_{iteration}/input_training_data_iter_{iteration}.xyz " +
              f"{cwd}/iter_{iteration}/3_DFT_minhop/input_training_data_iter_{iteration + 1}.xyz")

    new_trainingset = deque()
    minima_dict = {minima.get_potential_energy(apply_constraint=False): minima for minima in minimas}

    mdtraj = deque()
    if parallel:
        for i in range(parallel):
            mdtrajfiles = sorted([file for file in glob.glob(f"{str(i).zfill(2)}/md*.traj")])
            for traj in mdtrajfiles:
                mdtraj.extendleft(read(os.getcwd() + "/" + traj + "@:"))
    else:
        mdtrajfiles = sorted([file for file in glob.glob(f"md*.traj")])
        for traj in mdtrajfiles:
            mdtraj.extendleft(read(traj + "@:"))

    if sampling_method == "stratified_random":

        # get one global minimum + two local minima
        while len(new_trainingset) < 3:

            # get global minimum
            if len(new_trainingset) == 0:
                energy = min(minima_dict.keys())
                if test_short_distance(minima_dict[energy]):
                    new_trainingset.append(minima_dict.pop(energy))
                else:
                    minima_dict.pop(energy)

            # get two local minimum
            else:
                energy = random.sample(list(minima_dict), 1)[0]
                if test_short_distance(minima_dict[energy]):
                    new_trainingset.append(minima_dict.pop(energy))
                else:
                    minima_dict.pop(energy)

        # Get two random structure from MD trajectory
        random.seed(3)
        random.shuffle(mdtraj)
        while len(new_trainingset) < 5:
            mdstruc = mdtraj.pop()
            if test_short_distance(mdstruc):
                new_trainingset.append(mdstruc)

        write(cwd + f"/iter_{iteration}/3_DFT_minhop/new_training_set_iter{iteration}.xyz", new_trainingset,
              format="extxyz")


    elif sampling_method.lower() == "full_random":

        print("full random sampling is triggered!!")
        mdtraj.extendleft(minimas)
        random.seed(3)
        random.shuffle(mdtraj)
        while len(new_trainingset) < 5:
            newstruc = mdtraj.pop()
            if test_short_distance(newstruc):
                new_trainingset.append(newstruc)

        write(cwd + f"/iter_{iteration}/3_DFT_minhop/new_training_set_iter{iteration}.xyz", new_trainingset,
              format="extxyz")


    elif sampling_method.lower() == "full_fps":
        print("farthest point sampling triggered!!")
        mdtraj.extendleft(minimas)

        strucs = ConfigSet(mdtraj)
        output = OutputSpec(files=cwd + f"/iter_{iteration}/3_DFT_minhop/desc_{iteration}.xyz")
        desc_dicts = get_SOAP_param_dict(mdtraj, universal_soap_yaml=universal_soap_yaml)
        md_desc = calc_descriptors(inputs=strucs, outputs=output, descs=desc_dicts,
                                   key='desc', per_atom=False)

        fps_out = OutputSpec(files=cwd + f"/iter_{iteration}/3_DFT_minhop/new_training_set_iter{iteration}.xyz")
        new_trainingsets = greedy_fps_conf_global(inputs=md_desc,
                                                  outputs=fps_out,
                                                  num=7, at_descs_info_key='desc',
                                                  keep_descriptor_info=False)

        # In order to avoid error induced by too close atoms, 7 structures are sampled (slightly more than
        # desired number of samples) and only when those qualify short distance test are included.
        for newstruc in new_trainingsets:
            if test_short_distance(newstruc):
                new_trainingset.append(newstruc)
            if len(new_trainingset) == 5:
                break

    elif sampling_method.lower() == "stratified_fps":

        path = f"{cwd}/iter_{iteration}/3_DFT_minhop"
        md_strucs = ConfigSet(mdtraj)
        md_output = OutputSpec(files=f"{path}/md_desc_{iteration}.xyz")

        minima_strucs = ConfigSet(minimas)
        minima_output = OutputSpec(files=f"{path}/minima_desc_{iteration}.xyz")

        # Calculate descriptor and save xyz file respectively for minima and md trajectory
        desc_dicts = get_SOAP_param_dict(minimas, universal_soap_yaml=universal_soap_yaml)
        md_desc = calc_descriptors(inputs=md_strucs, outputs=md_output, descs=desc_dicts,
                                   key='desc', per_atom=False)
        minima_desc = calc_descriptors(inputs=minima_strucs, outputs=minima_output, descs=desc_dicts,
                                       key='desc', per_atom=False)

        fps_out1 = OutputSpec(files=f"{path}/fps_minima_{iteration}.xyz")
        fps_out2 = OutputSpec(files=f"{path}/fps_mdtraj_{iteration}.xyz")

        # Select three from minima and two from MD trajectory using FPS.
        new_trainingset_minima = greedy_fps_conf_global(inputs=minima_desc,
                                                        outputs=fps_out1,
                                                        num=3, at_descs_info_key='desc',
                                                        keep_descriptor_info=False)
        new_trainingset_mdtraj = greedy_fps_conf_global(inputs=md_desc,
                                                        outputs=fps_out2,
                                                        num=2, at_descs_info_key='desc',
                                                        keep_descriptor_info=False)

        os.system(
            f"cat {path}/fps_minima_{iteration}.xyz {path}/fps_mdtraj_{iteration}.xyz >> {path}/new_training_set_iter{iteration}.xyz")
        new_trainingset = [st for st in new_trainingset_minima] + [st for st in new_trainingset_mdtraj]

    #	Path(cwd + f"/iter_{iteration}/3_DFT_minhop").mkdir(parents=True, exist_ok = True)
    #	os.system(f"rsync -az {cwd}/iter_{iteration}/input_training_data_iter_{iteration}.xyz " +
    #			  f"{cwd}/iter_{iteration}/3_DFT_minhop/input_training_data_iter_{iteration+1}.xyz")

    #	write(cwd + f"/iter_{iteration}/3_DFT_minhop/new_training_set_iter{iteration}.xyz", new_trainingset, format="extxyz")
    f_name = cwd + f"/iter_{iteration}/3_DFT_minhop"
    Path(f_name).mkdir(parents=True, exist_ok=True)
    traj = Trajectory(f_name + "/structure.traj", mode="w")
    for idx, atoms in enumerate(list(new_trainingset)):
        traj.write(atoms)
    traj.close()

    if submit and collect_job_id:

        os.chdir(f_name)
        if np.all([os.path.exists(f"{idx}_structure/stdout.log") for idx, atoms in enumerate(list(new_trainingset))]):
            print("There is a output file in each folder. Continue")
        else:
            write_dft_slurm(f"iter{iteration}", forcemask=forcemask, code=code, path_to_workflow=path_to_workflow)

        num = os.popen("sbatch submit.sh").read()
        job_ids.append(num.strip().split()[-1])
    else:
        os.chdir(f_name)
        if np.all([os.path.exists(f"{idx}_structure/stdout.log") for idx, atoms in enumerate(list(new_trainingset))]):
            print("There is a output file in each folder. Continue")
        else:
            write_dft_slurm(f"iter{iteration}", forcemask=forcemask, code=code, path_to_workflow=path_to_workflow)
            os.system(f"sbatch submit.sh")

    if collect_job_id:
        return job_ids


def check_slurm_completion(job_ids, timelimit=1200 * 60, sleep=60):
    """
    This function takes submitted job's IDs as list and check whether they're
    completed every 10 seconds.
    When jobs are all completed, this retures True.

    parameters :
    ------------
    job_ids : list
        SLURM job IDs as list of string.
    timelimit : int
        If whole DFT single point calculations exceeds this limit this function
        terminates automatically.
    sleep : int
        In which every seconds job status will be examined.

    Return :
    --------
    Bool as True

    """

    jobarray = dict(zip(job_ids, [False] * len(job_ids)))

    start = time.time()
    while not all(jobarray.values()):

        if time.time() - start > timelimit:
            print(f"Time limit of {timelimit / 60} min reached ")
            break

        for jobid in jobarray.keys():
            statusline = os.popen(f"squeue --me | grep {jobid}").read()
            if not bool(statusline) or statusline.strip().split()[4] == "CG":
                jobarray[jobid] = True

        time.sleep(sleep)

    return True


def prepare_initial_config(parent_metal, dopant, lattice, surface_indices, surface_energies, size, one_out_of_n,
                           peratomsigma=False, forcemask=True, submit=True, lattice_param=3.85, code="aims",
                           free_atom_e=None, path_to_workflow=None):
    """
    When iteration step is 0, then single point DFT result is required.
    This function generates adsoprtion surface and submit DFT calculation.
    When job is done, it returns xyz file and
    """
    cwd = os.getcwd()
    Path(cwd + "/z_init").mkdir(parents=True, exist_ok=True)
    subprocess.run(["cp", "../restart_training_data.xyz", "./z_init/input_training_data_iter_0.xyz"])
    init_structs = []
    job_ids = []
    #calc = mace_mp(model="medium", dispersion=False, default_dtype="float64", device='cpu')
    for i in range(5):

        nanoparticle = create_dilute_alloy_np(parent_metal, dopant, lattice, lattice_param, surface_indices,
                                              surface_energies, size, one_out_of_n, randomseed=i * 50 + 1000)

        init_structs.append(nanoparticle.copy())
        # Add a minima optimized with MACE-MP-medium
        # opt_adsorption = newadsorption.copy()
        # cons0 = FixAtoms(indices=[atom.index for atom in opt_adsorption if atom.symbol not in ['C', 'H', 'O']])
        # opt_adsorption.set_constraint([cons0, ])
        # opt_adsorption.calc = calc
        # opt = BFGS(opt_adsorption)
        # opt.run(fmax=0.05)
        # init_structs.append(opt_adsorption.copy())
    f_name = cwd + "/z_init"
    Path(f_name).mkdir(parents=True, exist_ok=True)
    traj = Trajectory(f_name + "/structure.traj", mode="w")
    for idx, atoms in enumerate(init_structs):
        traj.write(atoms)
    traj.close()

    if submit:
        os.chdir(f_name)
        if np.all([os.path.exists(f"{idx}_structure/stdout.log") for idx, atoms in enumerate(init_structs)]):
            print("There is a output file in each folder. Continue")
            make_trainingset_file(-1, cwd, peratomsigma=peratomsigma, forcemask=forcemask, update="initial",
                                  code=code, free_atom_e=free_atom_e)
        else:
            write_dft_slurm("init_config", forcemask=forcemask, code=code, path_to_workflow=path_to_workflow)

            num = os.popen("sbatch submit.sh").read()
            job_ids.append(num.strip().split()[-1])

        if check_slurm_completion(job_ids):
            os.chdir(cwd)
            make_trainingset_file(-1, cwd, peratomsigma=peratomsigma, forcemask=forcemask, update="initial", code=code,
                                  free_atom_e=free_atom_e)

        return True


def prepare_isolated_atom(parent_metal, dopant, lattice, surface_indices, surface_energies, size, one_out_of_n,
                          forcemask=True, submit=True, lattice_param=3.85, code="aims", path_to_workflow=None):
    """
    When iteration step is 0, then single point DFT result is required.
    This function generates adsoprtion surface and submit DFT calculation.
    When job is done, it returns xyz file and
    """
    cwd = os.getcwd()
    Path(cwd + "/isolated-atom").mkdir(parents=True, exist_ok=True)
    job_ids = []

    nanoparticle = create_dilute_alloy_np(parent_metal, dopant, lattice, lattice_param,
                                          surface_indices, surface_energies, size, one_out_of_n)

    elements = set(nanoparticle.get_chemical_symbols())
    ele_energy = {}

    for ele in elements:
        f_name = cwd + f"/isolated-atom/{ele}"
        Path(f_name).mkdir(parents=True, exist_ok=True)
        write(f_name + f"/free-atom-{ele}.traj", Atoms(ele))

    if submit:
        for ele in elements:
            f_name = cwd + f"/isolated-atom/{ele}"
            os.chdir(f_name)
            if os.path.exists(f'stdout.log'):
                this_atom = read_aims_output(f"stdout.log")
                ele_energy[ele] = this_atom.get_potential_energy()
            else:
                write_dft_slurm("free_atom", forcemask=forcemask, code=code,
                                target_file_name=f"free-atom-{ele}.traj", path_to_workflow=path_to_workflow,
                                time="00:20:00", nodes=1, cores=1, mem=4000)
                num = os.popen("sbatch submit.sh").read()
                job_ids.append(num.strip().split()[-1])

        if job_ids != [] and check_slurm_completion(job_ids):
            for idx, ele in enumerate(elements):
                f_name = cwd + f"/isolated-atom/{ele}"
                this_atom = read_aims_output(f_name + "/" + "stdout.log")
                ele_energy[ele] = this_atom.get_potential_energy()

    os.chdir(cwd)

    return ele_energy


def graphfromsmiles(smiles):
    """
    Generate igraph object from smiles
    Convert 2D smiles into 3D geometry using embedding and subsequently optimize with MMFF
    From MMFF optimized geometry, bondlengths, etc are saved as graph edge attributes.
    This will be used for graph matching from ASE read adsorbate's molecular graph.

    Parameters :
    ------------
    smiles : str
        SMILES string for adsorbate

    Return :
    --------
    molgraph : igraph object
        graph of molecule with bondtype, bondlengths, etc, are encoded as edge attributes
    """

    #     smiles = "C"
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m)
    AllChem.MMFFOptimizeMolecule(m)
    position = m.GetConformer().GetPositions()

    # initiate molecular graph
    molgraph = ig.Graph.Adjacency(rdmolops.GetAdjacencyMatrix(m), mode="undirected")
    molgraph.vs["Symbols"] = [m.GetAtomWithIdx(i).GetSymbol() for i in range(m.GetNumAtoms())]
    molgraph.vs["AtomicNums"] = [m.GetAtomWithIdx(i).GetAtomicNum() for i in range(m.GetNumAtoms())]

    for bond in m.GetBonds():
        bondidxtuple = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

        molgraph.es[molgraph.get_eid(*bondidxtuple)]["bondtype"] = str(bond.GetBondType())
        molgraph.es[molgraph.get_eid(*bondidxtuple)]["atomsymbols"] = tuple(
            [m.GetAtomWithIdx(i).GetSymbol() for i in bondidxtuple])
        molgraph.es[molgraph.get_eid(*bondidxtuple)]["bondlength"] = np.linalg.norm(
            position[bondidxtuple[0]] - position[bondidxtuple[1]])

    return molgraph


def graphfromase(slabi, ):  # metal = "Rh"):
    """
    For given surface slab with adsorbate, this first calculate adjacency matrix and extract
    adsorbate's molecular graph as adjacency matrix.
    This is returned as igraph object

    Parameters :
    ------------
    slabi : ase object
        metal surface with adsorbate on top.
    metal : string
        metal element symbol.

    Return :
    --------
    ase_graph : igraph object
        adsorbate molecule's graph object
    """
    #     slabi = read(path)
    cutOff = neighborlist.natural_cutoffs(slabi)  # generate covalent radii for each elements in ase object
    # neighbor =
    neighbor = neighborlist.NeighborList(cutOff, self_interaction=False, bothways=True)
    neighbor.update(slabi)

    atom_dict = {i: symb for i, symb in enumerate(slabi.get_chemical_symbols()) if symb in ['C', 'O', 'H']}
    # print(atom_dict)
    adsorbateidx = sorted(list(atom_dict.keys()))[0]
    adsorbateidx

    adjacency = neighbor.get_connectivity_matrix(sparse=False)[adsorbateidx:, adsorbateidx:]
    ase_graph = ig.Graph.Adjacency(adjacency, mode="undirected")
    ase_graph.vs["Symbols"] = [atom.symbol for atom in slabi[sorted(atom_dict.keys())]]
    ase_graph.vs["AtomicNums"] = [atom.number for atom in slabi[sorted(atom_dict.keys())]]
    ase_graph.vs["AtomIndex"] = sorted(list(atom_dict.keys()))

    return ase_graph


def graphfromAdsfile(ads, ):
    """
    For given adsorbate files, this first calculate adjacency matrix and extract
    adsorbate's molecular graph as adjacency matrix.
    This is returned as igraph object

    Parameters :
    ------------
    ads : ase object
        adsorbate in gas phase.

    Return :
    --------
    ase_graph : igraph object
        adsorbate molecule's graph object
    """
    cutOff = neighborlist.natural_cutoffs(ads)  # generate covalent radii for each elements in ase object
    # neighbor =
    neighbor = neighborlist.NeighborList(cutOff, self_interaction=False, bothways=True)
    neighbor.update(ads)
    adsorbateidx = 0
    adjacency = neighbor.get_connectivity_matrix(sparse=False)[adsorbateidx:, adsorbateidx:]
    molgraph = ig.Graph.Adjacency(adjacency, mode="undirected")
    molgraph.vs["Symbols"] = [atom.symbol for atom in ads]
    molgraph.vs["AtomicNums"] = [atom.number for atom in ads]

    m = ase2rdkit(ads, suggestions=[])

    for bond in m.GetBonds():
        bondidxtuple = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

        molgraph.es[molgraph.get_eid(*bondidxtuple)]["bondtype"] = str(bond.GetBondType())
        molgraph.es[molgraph.get_eid(*bondidxtuple)]["atomsymbols"] = tuple(
            [m.GetAtomWithIdx(i).GetSymbol() for i in bondidxtuple])
        molgraph.es[molgraph.get_eid(*bondidxtuple)]["bondlength"] = ads.get_distance(bondidxtuple[0], bondidxtuple[1])

    return molgraph


def check_adsorbate_isomorphism(smiles, slab, return_mapping=True, using_opt_adsorbate=False, adsorbate_file=None):
    """
    Compare two molecular graph from smiles and from slab structure, and
    encode expected bondlength and other properties to slab structure.
    This informaiton will be further used to generate Hookean constraints.

    Parameters :
    ------------
    smiles : str
        SMILES string for adsorbate
    slab : ase object
        metal surface with adsorbate on top.

    Return :
    ase_graph : igraph object
        with bond length saved as attribute, adsorbate graph is returned is isomorphism match
        was successful. If failed, False is returned.
    --------

    """

    if using_opt_adsorbate:
        ads = read(adsorbate_file)
        molgraph = graphfromAdsfile(ads)
    else:
        molgraph = graphfromsmiles(smiles)

    ase_graph = graphfromase(slab)

    isomorphic, map12, map21 = molgraph.isomorphic_vf2(ase_graph,
                                                       color1=molgraph.vs["AtomicNums"],
                                                       color2=ase_graph.vs["AtomicNums"],
                                                       return_mapping_12=True, return_mapping_21=True)

    mapping = dict(zip(map12, map21))

    if isomorphic and return_mapping:
        for smiles_edge in molgraph.es:
            ase_graph.es[ase_graph.get_eid(*tuple([mapping[idx] for idx in smiles_edge.tuple]))].update_attributes(
                smiles_edge.attributes())

        return ase_graph

    elif isomorphic and not return_mapping:
        return True

    else:
        raise NameError("error: not isomorphic!")

        return False


def prevent_volatilation(slab, padding=2.5, verbose=True, k=20):
    """
    Prevent volatilation of molecule.
    This calculates longest distance from center of mass atom and add it to maximum metal layer with padding (e.g.Angs)
    Ax + By + Cz + D = 0
    D = metal_z + max_dist_cm + padding

    """
    molindices = [atom.index for atom in slab if atom.symbol in ['C', 'H', 'O']]
    mol = slab[min(molindices):]
    cm_index = np.argmin(np.array([np.linalg.norm(atom.position - mol.get_center_of_mass()) for atom in mol]))
    farthest_index = np.argmax(np.array([np.linalg.norm(atom.position - mol.get_center_of_mass()) for atom in mol]))
    distance_cm = mol.get_distance(cm_index, farthest_index)
    metal_z = max([atom.z for atom in slab if atom.index not in molindices])

    d = (metal_z + distance_cm + padding) * -1

    if verbose:
        print(f"cm_index : {molindices[cm_index]}")
        print(f"farthest_index : {molindices[farthest_index]}")
        print(f"distance_cm : {distance_cm}")
        print(f"d : {d}")

    return Hookean(a1=molindices[cm_index], a2=(0, 0, 1, d), k=k)


def apply_constraints(atoms, **kwargs):
    """
    Apply constraints to the system.

    Parameters :
    ------------
    atoms : ase object
        system where constraints will be imposed.
    relax_metal : bool
        whether to relax upper two metal layer. If False, all metal atoms are fixed.
    constrain_Hookean : bool
        Imposed Hookean constraint on adsorbate molecule.
    method : str
        Hookean constraint is generated based on the SMILES string of adsorbate molecule.
        If file, base on pre-optimized adsorbate geometry
    volatilation : bool
        whetehr to impose Hookean constraint with respect to surface.
    adsorbate_file: str

    """
    relax_metal = kwargs.get("relax_metal", False)
    constrain_Hookean = kwargs.get("constrain_Hookean", True)
    method = kwargs.get("method", "smiles")
    volatilation = kwargs.get("volatilation", True)
    # using_opt_adsorbate = kwargs.get("using_opt_adsorbate", False)
    adsorbate_file = kwargs.get("adsorbate_file", None)

    cons = []
    default_rcuts = []
    del atoms.constraints  # Delete constraints first
    # Get distances
    cutOff = neighborlist.natural_cutoffs(atoms)
    neighbor = neighborlist.NeighborList(cutOff, self_interaction=False, bothways=False)
    neighbor.update(atoms)
    matrix = neighbor.get_connectivity_matrix(sparse=False)

    NumRh = len([atom for atom in atoms if atom.symbol not in ['C', 'H', 'O']])

    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if (matrix[i, j] == 1 and i >= NumRh and j >= NumRh):
                d = atoms.get_distances(i, j)
                default_rcuts.append([i, j, d[0]])

    cons = []
    if relax_metal:
        height_set = sorted(set([atom.z for atom in atoms if atom.symbol not in ['C', 'H', 'O']]))
        # if len(height_set) == 4:
        #	uppermost_layer_height = height_set[-2]
        # elif len(height_set) == 12:
        #	uppermost_layer_height = height_set[-6]
        uppermost_layer_height = height_set[int(-0.5 * len(height_set))]

        cons.append(FixAtoms(indices=[atom.index for atom in atoms if atom.z < uppermost_layer_height]))
    else:
        cons.append(FixAtoms(indices=[atom.index for atom in atoms if atom.symbol not in ['C', 'H', 'O']]))

    if constrain_Hookean and method == "smiles":
        smiles = kwargs["smiles"]
        mapped_graph = check_adsorbate_isomorphism(smiles, atoms)

        if mapped_graph:
            for edge in mapped_graph.es:
                bond_pair = [mapped_graph.vs[vid]["AtomIndex"] for vid in edge.tuple]
                cons.append(Hookean(a1=bond_pair[0], a2=bond_pair[1],
                                    rt=1.05 * edge["bondlength"], k=10))

    elif constrain_Hookean and method == "file":
        mapped_graph = check_adsorbate_isomorphism(smiles=None, slab=atoms, using_opt_adsorbate=True,
                                                   adsorbate_file=adsorbate_file)

        if mapped_graph:
            for edge in mapped_graph.es:
                bond_pair = [mapped_graph.vs[vid]["AtomIndex"] for vid in edge.tuple]
                cons.append(Hookean(a1=bond_pair[0], a2=bond_pair[1],
                                    rt=1.05 * edge["bondlength"], k=10))

    elif constrain_Hookean and method != "smiles":
        for r in default_rcuts:
            cons.append(Hookean(a1=r[0], a2=r[1], rt=1.05 * r[2], k=25.))
    else:
        # cons = constraints
        pass

    if volatilation:
        cons.append(prevent_volatilation(atoms))

    atoms.set_constraint(cons)


def get_surface_index(atoms, lattice_constant, set_tag=False):
    """
    Get indices for surface atoms of a metal nanoparticle based on maximum radius to centre of mass.
    """
    radius = [np.linalg.norm(np.asarray([atom.position - atoms.get_center_of_mass()])) for atom in atoms]
    cutoff = np.max(radius) - lattice_constant / 2.0
    surf_index = [atom.index for atom in atoms if radius[atom.index] > cutoff]

    if set_tag:
        for atom in atoms:
            atom.tag = 1
        for i in surf_index:
            atoms[i].tag = 0

    return surf_index

def create_dilute_alloy_np(
        parent_metal: str = "Cu",
        dopant: str = "Ru",
        lattice: str = "fcc",
        lattice_constant=3.575,
        surface_indices=[(1, 1, 1), (1, 1, 0), (1, 0, 0)],
        surface_energies=[1.447, 1.660, 1.584],
        size=150,
        one_out_of_n=9,
        randomseed=46867,
):
    """
    Create a dilute alloy nanoparticle with a fixed concentration of dopant at the surface.

    Parameters:
    -----------
    parent_metal: str. Element symbol of the parent metal.
    lattice: str. Lattice type of the parent metal.
    one_out_of_n: int. 1/n concentration of dopant on the surface.
    """
    atoms = wulff_construction(symbol=parent_metal, surfaces=surface_indices, energies=surface_energies,
                               size=size, structure=lattice, latticeconstant=lattice_constant)
    surf_i = get_surface_index(atoms, lattice_constant, set_tag=True)

    #r_max = np.max([np.linalg.norm(np.asarray([atom.position - atoms.get_center_of_mass()])) for atom in atoms])
    #atoms.set_cell([2 * (r_max + 50),] * 3)
    #atoms.set_center_of_mass([r_max + 50, ] * 3)
    random.seed(randomseed)
    sample_index = random.sample(surf_i, int(len(surf_i) / one_out_of_n))
    for i in sample_index:
        atoms[i].symbol = dopant

    return atoms

def create_adsorbate_surface(
        mol,
        surface="111",
        height=1.5,
        metal="Rh",
        lattice_param=3.85,
        vacuum=30,
        adsorbate_json=str(Path(__file__).parent.resolve()) + "/n3_mol_mod.json",
        randseed=0,
        using_opt_surface=False,
        surface_file=None,
        using_opt_adsorbate=False,
        adsorbate_file=None,
):
    """
    Create adsorbate with metal surface.
    Molecular adsorbates information is stored in JSON file.
    One can search molecule by index (int.) or SMILES (string)
    Position of the adsorbate is randomly assigned in x,y coordinate.

    Parameters:
    -----------
    mol : integer or string
        Molecules's index in JSON file or corresponding SMILES string
    height : float
        Height above metal surface. Height is measured from the distance
        between lowest z coordinate of adsorbate molecule and metal surface.
    surface : string or int
        Metal surfaces' facet currently either "111" or "211" for Rh
    metal : string
        Catalyst metal to be tested.
    lattice_param : integer
        Lattice parameter for metal.
    adsorbate_json : string
        Path for adsorbate file to be used
    using_opt_surface: boolean
        Flag for using a pre-optimized surface
    surface_file:
        File of the pre-optimized surface
    """

    mdf = pd.read_json(os.path.expanduser(adsorbate_json))
    mdf["asemol"] = [read(StringIO(xyz), format="xyz") for xyz in mdf.unrelaxed_xyz]

    if isinstance(mol, int):
        mol = mdf.asemol[mol]
    elif isinstance(mol, str):
        mol = mdf.loc[mdf.smiles == mol].asemol.item()
    if using_opt_adsorbate:
        mol = read(adsorbate_file)

    if using_opt_surface:
        atoms = read(surface_file)
    else:
        if surface == "111" or surface == 111:
            atoms = ase.build.fcc111(metal, size=(3, 3, 4), a=lattice_param, vacuum=vacuum)
        elif surface == "211" or surface == 211:
            atoms = ase.build.fcc211(metal, size=(3, 3, 4), a=lattice_param, vacuum=vacuum)

    cutOff = neighborlist.natural_cutoffs(atoms)

    np.random.seed(randseed)
    vs = ["x", "-x", "y", "-y", "z", "-z"]
    mol.rotate(a=90 * np.random.random(), v=vs[np.random.randint(0, 6)], center="COM")
    min_atom_index = np.argmin(np.array([atom.z - cutOff[atom.index] for atom in mol]))
    add_adsorbate(atoms, mol, position=np.matmul(np.array(atoms.get_cell())[:2, :2].T, np.random.rand(2, 1)).ravel(),
                  height=1.5, mol_index=min_atom_index)

    return atoms


def prepare_input(parent_metal, dopant, lattice, surface_indices, surface_energies, size, one_out_of_n,
                  iteration=None, tmpdir=None, randseed=0, parallel=1, lattice_param=3.85):
    """
    Prepare adsorate + surface structure

    Parameters :
    ------------
    smiles : str
        SMILES string for adsorbate
    facet : str or int
        facet of surface (e.g.) 111 or 211
    iteration : int
        iteration number of GAP fitting
    tmpdir : str
        directory for scratch folder
    randseed : int
        Random seed for random initialization of adsorption struture.
    parallel : int
        Number of randomized initial structures to be generated for parallel minima hopping.
        If parallel=1, it's serial minima hopping for GAP fitting.
        But parallel > 1, parallel minima hopping is prepared.
    """
    if tmpdir != None:
        Path(tmpdir + f"/iter_{iteration}/2_Minhop").mkdir(parents=True, exist_ok=True)
        os.chdir(tmpdir + f"/iter_{iteration}/2_Minhop")

        if iteration == 0:
            reference_dat = read(tmpdir + f'/iter_{iteration}/input_training_data_iter_{iteration}.xyz@:')
            write(tmpdir + f"/iter_0/2_Minhop/adsorption.traj", reference_dat[-1])
        else:
            if parallel == 1:
                nanoparticle = create_dilute_alloy_np(parent_metal, dopant, lattice, lattice_param, surface_indices,
                                                      surface_energies, size, one_out_of_n, randomseed=randseed)
                add_dummy_cell(nanoparticle)
                write(tmpdir + f"/iter_{iteration}/2_Minhop/adsorption.traj", nanoparticle)
            else:
                for i in range(parallel):
                    Path(tmpdir + f"/iter_{iteration}/2_Minhop/{str(i).zfill(2)}").mkdir(parents=True, exist_ok=True)
                    nanoparticle = create_dilute_alloy_np(parent_metal, dopant, lattice, lattice_param, surface_indices,
                                                          surface_energies, size, one_out_of_n, randomseed=i)
                    add_dummy_cell(nanoparticle)
                    write(tmpdir + f"/iter_{iteration}/2_Minhop/{str(i).zfill(2)}/adsorption.traj", nanoparticle)

    else:
        if parallel == 1:
            nanoparticle = create_dilute_alloy_np(parent_metal, dopant, lattice, lattice_param, surface_indices,
                                                  surface_energies, size, one_out_of_n, randomseed=randseed)
            add_dummy_cell(nanoparticle)
            write("adsorption.traj", nanoparticle)
        else:
            for i in range(parallel):
                Path(f"{str(i).zfill(3)}").mkdir(parents=True, exist_ok=True)
                nanoparticle = create_dilute_alloy_np(parent_metal, dopant, lattice, lattice_param, surface_indices,
                                                      surface_energies, size, one_out_of_n, randomseed=i)
                add_dummy_cell(nanoparticle)
                write(f"{str(i).zfill(3)}/adsorption.traj", nanoparticle)


def check_GAP_convergence(statistics,
                          submitdir,
                          E_threshold=8,  # meV/atom
                          F_threshold=0.15,  # eV/A
                          alpha=0.3,
                          adjust=True,
                          sampling_method="recipe",
                          calc_all_minima=False,
                          convergence="light"
                          ):
    """
    Check GAP convergence using Exponential moving average.

    Parameters :
    ------------
    statistics : dict
        Error statistics including energy and force RMSD and MAE.
    submitdir : str
        submission directory
    E_threshold : float
        Energy threshold for GAP convergence. default 8 meV/atom
    F_threshold : float
        Force threshold for GAP convegernce. default 0.15 eV/AA
    alpha : float
        hyperparameter for exponential moving average.
    adjust : bool
        Whether to apply adjustment for exponential moving average.
    sampling_method : str
        sampling method can be either "recipe" or "FPS"

    Return :
    --------
    Boolean
    """
    #	print("statistics : ", statistics)
    atoms = read(submitdir + "/input_training_data_iter_0.xyz")
    # df = pd.DataFrame(statistics, index=[0])

    if isinstance(statistics, dict):
        df = pd.DataFrame.from_dict(statistics).transpose()

    elif isinstance(statistics, str) and calc_all_minima:
        df = pd.read_csv(statistics)

    #	if sampling_method == "recipe" or sampling_method == "intermediate":
    #	if not calc_all_minima:
    df["E_RMSD"] = df["RMSD_E_val_minima"] / atoms.get_global_number_of_atoms() * 1000
    df["F_RMSD"] = df["RMSD_F_val_minima"]

    df["E_EMA"] = df["E_RMSD"].ewm(alpha=alpha, adjust=adjust).mean()
    df["E_SMA"] = df["E_RMSD"].rolling(5, min_periods=1).mean()
    df["F_EMA"] = df["F_RMSD"].ewm(alpha=alpha, adjust=adjust).mean()
    df["F_SMA"] = df["F_RMSD"].rolling(5, min_periods=1).mean()

    #	elif sampling_method == "fps" or sampling_method == "random":
    #	else:
    #		print("Validation on every minima found from minima hopping")
    #		df["E_RMSD"] = df["RMSD_E_val"]/atoms.get_global_number_of_atoms()*1000
    #		df["F_RMSD"] = df["RMSD_F_val"]

    #	if not calc_all_minima:
    #		df["E_EMA"] = df["E_RMSD"].ewm(alpha=alpha, adjust=adjust).mean()
    #		df["E_SMA"] = df["E_RMSD"].rolling(5, min_periods = 1).mean()
    #		df["F_EMA"] = df["F_RMSD"].ewm(alpha=alpha, adjust=adjust).mean()
    #		df["F_SMA"] = df["F_RMSD"].rolling(5, min_periods = 1).mean()

    iteration = df.index.values[-1]
    print(f'E_EMA : {df.loc[iteration]["E_EMA"]:0.3f}')
    print(f'F_EMA : {df.loc[iteration]["F_EMA"]:0.3f}')

    if convergence == "tight":
        if df.loc[iteration]["E_EMA"] < 30 and df.loc[iteration]["F_EMA"] < 0.15:
            print("GAP converged with tight convergence criteria")
            return True
        else:
            return False

    elif convergence == "very_tight":
        if df.loc[iteration]["E_EMA"] < E_threshold and df.loc[iteration]["F_EMA"] < F_threshold:
            print("GAP converged with very tight convergence criteria")
            return True
        else:
            return False

    elif convergence == "light":
        if df.loc[iteration]["E_EMA"] < E_threshold or df.loc[iteration]["F_EMA"] < F_threshold:
            print("GAP converged with light convergence criteria")
            return True
        else:
            return False


def check_GAP_convergence_after_parallel_MH(iteration, submitdir, calculate=True, code="aims", free_atom_e=None, path_to_workflow=None):
    if calculate:
        job_ids = cluster_sample(submitdir=submitdir, singlepoint=True, code=code,
                                 free_atom_e=free_atom_e, path_to_workflow=path_to_workflow)
        if job_ids is None:
            print("DFT jobs should already finished")
        elif check_slurm_completion(job_ids):
            print("DFT jobs all finished")
    else:
        pass
    
    potential_file = submitdir + f"/iter_{iteration - 1}/GAP_2b_soap_iter_{iteration - 1}/GAP_2b_soap_iter_{iteration - 1}.xml"
    pot = Potential(param_filename=potential_file)

    DFT, GAP = [], []
    import fnmatch
    minima_folders = fnmatch.filter(os.listdir(f"{submitdir}/c_DFT_singlepoint4cluster"), '*_structure')
    for i in range(len(minima_folders)):
        # DFT result
        pwd = f"{submitdir}/c_DFT_singlepoint4cluster/{i}_structure"
        os.chdir(pwd)

        if code == "aims":
            atoms = parse_output(free_atom_e=free_atom_e)

        elif code == "qe":
            atoms = read("espresso.pwo")
            atoms.info["F_el"] = atoms.get_potential_energy()
            atoms.info["F_AE"] = get_atomization_energy(atoms, atoms.info["F_el"], code=code)

        DFT.append(get_atomization_energy(atoms, atoms.info["F_el"], code=code, free_atom_e=free_atom_e))

        # GAP result
        #with open(f"{pwd}/" + glob.glob("*.pickle")[0], 'rb') as handle:
        #    atoms = pickle.load(handle)
        structures = read(f"{submitdir}/c_DFT_singlepoint4cluster/structure.traj@:")
        
        atoms = structures[i]
        atoms.calc = pot
        GAP.append(get_atomization_energy(atoms, atoms.get_potential_energy(apply_constraint=False), code=code,
                                          free_atom_e=free_atom_e))
        error = rmse(DFT, GAP, )
    print(f"RMSD for selected {len(minima_folders)} minima is {error:0.2f} eV")

    if error < 1:
        return True
    else:
        return False


def print_minhop_parameter(hop):
    print("=" * 40)
    print("Minima Hopping parameters")
    print("=" * 40)
    for key in hop._default_settings.keys():
        print(f"{key:<16s} :  {getattr(hop, '_' + key)}")
    print("=" * 40)


def minimahopping(
        iteration,
        potential_file,
        tmpdir=None,
        statistics={},
        verbose=True,
        **kwargs
):
    """
    Master function for minima hopping.

    Parameters :
    ------------
    iteration : int
        iteration number of GAP fitting
    smiles : str
         SMILES string for adsorbate
    potential_file : str
        path of potential file
    facet : int or str
        facet of surface (e.g.) 111 or 211
    tmpdir : str
        directory for scratch folder
    statistics : dict
        Error statistics including energy and force RMSD and MAE.
    verbose : bool
        print process explicitly.
    Ediff0 : float
        Ediff0 for minima hopping : Energy difference criteria. Default : 0.75 eV
    T0 : int
        Initial temperature of minima hopping. Default : 2000 K
    hops : int
        Number of hops for minima hopping. Default: 20
    parallel : bool
        Whether to use parallel minima hopping.
    initial_struc_file : str
        file name of initial structure file.
    randseed : int
        random seed for initializaiton of adsorption structure.
    relax_metal : bool
        Whether to relax metal layers in slab.
    maxtemp_scale : float
        When to terminate parallel minima hopping with temperature criteria.
        Default : 2 means When MD temperature reaches 2*T0 = T, simulation is terminated.
    timestep : float
        timestep for MD in minima hopping. in fs.
    using_opt_surface : bool
    surface_file : str
    using_opt_adsorbate : bool
    adsorbate_file : str,
    """
    Ediff0 = kwargs.get("Ediff0", 0.75)
    T0 = kwargs.get("T0", 2000)
    hops = kwargs.get("hops", 20)
    parallel = kwargs.get("parallel", False)
    initial_struc_file = kwargs.get("initial_struc_file", "adsorption.traj")
    randseed = kwargs.get("randseed", 0)
    relax_metal = kwargs.get("relax_metal", False)
    maxtemp_scale = kwargs.get("maxtemp_scale", 2)
    timestep = kwargs.get("timestep", 0.5)
    parent_metal = kwargs.get("parent_metal", "Cu")
    dopant = kwargs.get("dopant", "Ru")
    lattice = kwargs.get("lattice", "fcc")
    surface_indices = kwargs.get("surface_indices", [(1, 1, 1), (1, 1, 0), (1, 0, 0)])
    surface_energies = kwargs.get("surface_energies", [1.447, 1.660, 1.584])
    size = kwargs.get("size", 150)
    one_out_of_n = kwargs.get("one_out_of_n", 9)
    lattice_param = kwargs.get("lattice_param", 3.575)

    if verbose:
        print("start minima hopping")

    # Read in Structure
    atoms = read(initial_struc_file, index="-1")

    # Set Constraints # No need anymore?
    # if using_opt_adsorbate:
    #     apply_constraints(atoms, relax_metal=relax_metal, method='file', adsorbate_file=adsorbate_file)
    # else:
    #     apply_constraints(atoms, relax_metal=relax_metal, smiles=smiles)

    if verbose:
        print(atoms.constraints)
        print("start reading potential file")

    # Set Calculator/ Potential
    if not parallel:

        if iteration < 2 or statistics[iteration - 1]["RMSD_F_val"] > 3:  # eV/AA
            calc = mace_mp(model="medium", dispersion=False, default_dtype="float64", device='cpu')
            atoms.calc = calc
            if verbose:
                print("Low accuracy! Use mace as calculator to insure stability.")
        else:
            pot = Potential(param_filename=potential_file)
            atoms.calc = pot
            if verbose:
                print("potential file has been read!")
    else:
        
        pot = Potential(param_filename=potential_file)
        atoms.calc = pot
        if verbose:
                print("potential file has been read!")


    # Minima Hopping
    if not parallel:

        if iteration < 2 or statistics[iteration - 1]["RMSD_F_val"] > 3:  # eV/AA
            hops = 10
            fmax = 10
            print(f"Force RMSD is larger than 1 eV/A : totalsteps are set {hops}, optimizer in minhop is set as mdmin")
            print("Accuracy too low! Use mace as calculator to insure stability.")
            low_accuracy = True
            hop = MinimaHopping(atoms, Ediff0=Ediff0, T0=T0, timestep=timestep, minima_traj="minima.traj",
                                optimizer=MDMin, fmax=fmax)

        else:
            fmax = 0.05
            print(f"Force RMSD is less than 1 eV/A : totalsteps are set {hops}")
            low_accuracy = False
            hop = MinimaHopping(atoms, Ediff0=Ediff0, T0=T0, timestep=timestep, minima_traj="minima.traj")

        if verbose:
            print_minhop_parameter(hop)

        hop(totalsteps=hops, maxtemp=maxtemp_scale * T0)

        count = 0
        while len(read("minima.traj@:")) < 3:
            print("Minima from minhop are less than 3! Ten more hops are added")
            os.system("rm -f hop.log md* qn* ")

            prepare_input(parent_metal, dopant, lattice, surface_indices, surface_energies, size, one_out_of_n,
                          iteration=iteration, tmpdir=tmpdir, randseed=iteration + count + 100,
                          lattice_param=lattice_param)

            atoms = read(initial_struc_file, index="-1")

            # if using_opt_adsorbate:
            #     apply_constraints(atoms, relax_metal=relax_metal, method='file', adsorbate_file=adsorbate_file)
            # else:
            #     apply_constraints(atoms, relax_metal=relax_metal, smiles=smiles)

            if iteration < 2 or statistics[iteration - 1]["RMSD_F_val"] > 3:  # eV/AA
                calc = mace_mp(model="medium", dispersion=False, default_dtype="float64", device='cpu')
                atoms.calc = calc
                if verbose:
                    print("Low accuracy! Use mace as calculator to insure stability.")
            else:
                pot = Potential(param_filename=potential_file)
                atoms.calc = pot
                if verbose:
                    print("potential file has been read!")
            print("initial structure for minima hopping changed!")

            if low_accuracy:
                hop_re = MinimaHopping(atoms, Ediff0=6, T0=T0, timestep=timestep,
                                       optimizer=MDMin, fmax=fmax, minima_traj="minima.traj")
            else:
                hop_re = MinimaHopping(atoms, Ediff0=6, T0=T0, timestep=timestep, minima_traj="minima.traj")

            hops += 10
            hop_re(totalsteps=hops)
            if verbose:
                print_minhop_parameter(hop_re)

            count += 1
    else:
        hop = MinimaHopping(atoms, Ediff0=Ediff0, T0=T0, timestep=timestep, minima_traj="../minima.traj")
        if verbose:
            print_minhop_parameter(hop)

        hop(totalsteps=80, maxtemp=maxtemp_scale * T0)

    mhplot = MHPlot()
    mhplot.save_figure('MinHop_summary.png')

    if verbose:
        print(f"Minima hopping using GAP_2b_soap_iter_{iteration}.xml finished!!")


def Run_parallel_minima_hopping(iteration, submitdir, path_to_workflow, parallel=40, rerun=False, lattice_param=3.85,
                                relax_metal="", parent_metal="Cu", dopant="Ru", lattice="fcc",
                                surface_indices=[(1, 1, 1), (1, 1, 0), (1, 0, 0)],
                                surface_energies=[1.447, 1.660, 1.584], size=150, one_out_of_n=9, t0=2000):
    if relax_metal:
        relax_metal = " -rm"
    else:
        relax_metal = ""

    if not rerun:
        print("Start parallel minima hopping")
        Path(submitdir + f"/a_parallel_minhop").mkdir(parents=True, exist_ok=True)
        os.chdir(submitdir + f"/a_parallel_minhop")

        prepare_input(parent_metal, dopant, lattice, surface_indices, surface_energies, size, one_out_of_n,
                      parallel=parallel, lattice_param=lattice_param,)
        potential_file = submitdir + f"/iter_{iteration - 1}/GAP_2b_soap_iter_{iteration - 1}/GAP_2b_soap_iter_{iteration - 1}.xml"

        with open("cmd.lst", "w") as f:
            for i in range(parallel):
                f.write(
                    f'cd ./{str(i).zfill(3)}; srun --mem=4GB --exclusive -N 1 -n 1 python {os.path.dirname(os.path.realpath(__file__))}/minimahopping.py -p "../GAP_2b_soap_iter_{iteration - 1}.xml"{relax_metal} -para -v -t0 {t0} > stdout.log \n')

        copyfile(f"{potential_file}", f"{os.getcwd()}/GAP_2b_soap_iter_{iteration - 1}.xml")

    else:
        print("Start parallel minima hopping AGAIN!")
        os.rename(f"{submitdir}/a_parallel_minhop", f"{submitdir}/y_parallel_minhop_prev")
        Path(submitdir + f"/a_parallel_minhop").mkdir(parents=True, exist_ok=True)
        os.chdir(submitdir + f"/a_parallel_minhop")

        prepare_input(parent_metal, dopant, lattice, surface_indices, surface_energies, size, one_out_of_n,
                      parallel=parallel, lattice_param=lattice_param, )
        potential_file = submitdir + f"/iter_{iteration}/GAP_2b_soap_iter_{iteration}/GAP_2b_soap_iter_{iteration}.xml"

        with open("cmd.lst", "w") as f:

            for i in range(parallel):
                f.write(
                    f'cd ./{str(i).zfill(3)}; srun --mem=4GB --exclusive -N 1 -n 1 python {os.path.dirname(os.path.realpath(__file__))}/minimahopping.py -p "../GAP_2b_soap_iter_{iteration}.xml"{relax_metal} -para -v -t0 {t0} > stdout.log \n')

        copyfile(f"{potential_file}", f"{os.getcwd()}/GAP_2b_soap_iter_{iteration}.xml")

    #	copyfile(f"{os.path.dirname(os.path.realpath(__file__))}/GNUparallel.sh", f"{os.getcwd()}/GNUparallel.sh" )
    copyfile(f"{dirname(dirname(abspath(__file__)))}/GNUparallel.sh",
             f"{os.getcwd()}/GNUparallel.sh")
    # if using_opt_adsorbate:
    #     idx, formula, smiles = get_adsorbate_info(using_opt_adsorbate=using_opt_adsorbate,
    #                                               adsorbate_file=adsorbate_file)
    # else:
    #     idx, formula, smiles = get_adsorbate_info(smiles)
    os.system(f'sed -i "/^#SBATCH -J/s/.*/#SBATCH -J Minhop_M{size}_{dopant}{parent_metal}/" GNUparallel.sh')
    os.system(f'sed -i "/^#SBATCH --ntasks=40/s/.*/#SBATCH --ntasks={parallel}/" GNUparallel.sh')
    os.system(f'sed -i "/^#SBATCH --mem=180000M/s/.*/#SBATCH --mem={3000*parallel}M/" GNUparallel.sh')
    os.system(f'sed -i "/^export PYTHONPATH=/s|.*|export PYTHONPATH={path_to_workflow}:\$PYTHONPATH|" GNUparallel.sh')
    job_ids = os.popen("sbatch GNUparallel.sh").read().strip().split()[-1:]

    if check_slurm_completion(job_ids, timelimit=300 * 60, sleep=30):
        print("parallel minhop finished!")


if __name__ == "__main__":
    args = get_minhop_args()

    if args.parallel != 1:
        minima_traj = "../minima.traj"
    else:
        minima_traj = "minima.traj"

    minimahopping(
        iteration=args.iteration,
        potential_file=args.potential,
        hops=args.hops,
        tmpdir=args.tmpdir,
        minima_traj=minima_traj,
        relax_metal=args.relaxmetal,
        randseed=args.randseed,
        parallel=args.parallel,
        maxtemp_scale=args.maxtemp_scale,
        T0=args.initial_temperature
    )
