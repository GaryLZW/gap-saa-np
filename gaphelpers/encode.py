from ase.io import read, write
from ase.io.aims import read_aims_output
import os, numpy as np
from collections import Counter
import argparse
from numpy.linalg import norm
from gaphelpers.hyperparameter import get_atomization_energy
from gaphelpers.get_argparse import get_encode_argparse
import subprocess



def get_force_atom_sigma(force_array, f_percentage = 0.01, F_min = 0.01):
    """
    Get force atom sigme for each atoms rather than using identical value of force sigma.
    force atom sigma is determined by taking norm of force array and multiplying with certain value.
    For simplicity, this function has not been used in this work.

    Parameters :
    ------------
    force_array : numpy array
        forces of each atoms in the system.
    f_percentage : float
        fractional number multiplied to force norm.
    F_min : float
        Minimum Force norm if norm is above this threshold norm is used.

    Return :
    --------
    force_atom_sigmas : numpy array
        force atom sigma values for each atoms determined with heuristics.
    """

    force_atom_sigmas = []

    for atom_force in force_array:
        Fi_norm = norm(atom_force)

        if Fi_norm > F_min:
            atom_sigma = f_percentage * Fi_norm
        else:
            atom_sigma = f_percentage * F_min
        force_atom_sigmas.append(atom_sigma)

    return np.asarray(force_atom_sigmas)



def mask_forces(atoms):
    """
    This function is used to mask force component of metal slabs of lower half by height.

    Parameters :
    ------------
    atoms : ASE object
        metal surface with adsorbate on top.

    Return :
    --------
    force_mask : numpy array
        array of Booleans indicating which atoms's force will be ignored in GAP fitting.
    """

# mask forces for half height
    height_set = sorted(set([atom.z for atom in atoms if atom.symbol not in ['C', 'H', 'O']]))
    #if len(height_set) == 4:
    #	uppermost_layer_height = height_set[-1]
    #elif len(height_set) == 12:
    #	uppermost_layer_height = height_set[-3]
    uppermost_layer_height = height_set[int(-0.5 * len(height_set))]
    force_mask = []
    for atom in atoms:
        if atom.z < uppermost_layer_height:
            force_mask.append("T")
        else:
            force_mask.append("F")

    return np.asarray(force_mask)



def parse_output(outputfile="stdout.log", free_atom_e=None):
    """
    This function reads FHI-aims output file and record required information to ASE atoms object.
    This assumes that outputfile is located at the same directory where this fucntion is executed and
    single point calculation has been performed.

    Uncorrected electronic total energy (E_uncorr), corrected electronic total energy, T -> 0 (E_corr),
    electronic free energy (F_el) are recored.
    Also Atomization energy (F_AE) and atomization per atom (F_AE_N) is calculated based on F_el.

    Parameters :
    ------------
    outputfile : string
        filename of FHI-aims stdout file.

    Return :
    --------
    atoms : ASE object
        adsorption structure with DFT energy info saved in info
    """
    print(os.getcwd())
    atoms = read_aims_output(outputfile)
    lines = os.popen(f'grep -A3 "Energy and forces in a compact form:" stdout.log').read()
    atoms.info["F_el"] = float(lines.split("\n")[3].split()[5])
    atoms.info["E_uncorr"] = float(lines.split("\n")[1].split()[5])
    atoms.info["E_corr"] = float(lines.split("\n")[2].split()[5])
    atoms.info["F_AE"] = get_atomization_energy(atoms, atoms.info["F_el"], free_atom_e=free_atom_e) if len(atoms) > 1 else 0.0
    atoms.info["F_AE_N"] = atoms.info["F_AE"] / len(atoms) if len(atoms) > 1 else 0.0

    return atoms

def add_dummy_cell(atoms):
    r_max = np.max([np.linalg.norm(np.asarray([atom.position - atoms.get_center_of_mass()])) for atom in atoms])
    atoms.set_cell([2 * (r_max + 50), ] * 3)
    atoms.set_center_of_mass([r_max + 50, ] * 3)

def write_xyz(
        iteration,
        peratomsigma=False,
        forcemask=True,
        initial=False,
        code="aims",
        free_atom_e=None,
        ):
    """
    This function reads calculated DFT output and append corresponding structure to the existing training set.

    Parameters :
    ------------
    iteration : int
        GAP fitting iteration
    peratomsigma : bool
        Whether to use peratomsigma
    forcemask : bool
        Whether to use forcemask for 3 lower metal atoms.
    initial : bool
        True if iteration=0 else False.
    free_atom_e : dict

    """
    if code == "aims":
        atoms = parse_output(free_atom_e=free_atom_e)
    else:
        atoms = read("espresso.pwo")
        atoms.info["F_el"] = atoms.get_potential_energy()
        atoms.info["F_AE"] = get_atomization_energy(atoms, atoms.info["F_el"], code="qe")

    if initial:
        atoms.info["source"] = "mdtraj"
    else:
        if int(os.getcwd().split("/")[-1].split("_")[0]) < 3:
            atoms.info["source"] = "minima"
        elif int(os.getcwd().split("/")[-1].split("_")[0]) >= 3:
            atoms.info["source"] = "mdtraj"


    if peratomsigma:
        atoms.arrays["force_atom_sigma"] = get_force_atom_sigma(atoms.get_forces())
    else:
        atoms.arrays.pop("force_atom_sigma", None)


    if forcemask:
        atoms.arrays["force_mask"] = mask_forces(atoms)
    else:
        atoms.arrays.pop("force_mask", None)

    atoms.arrays["forces_dft"] = atoms.get_forces()
    if len(atoms) == 1 or len(atoms) == 2:
        atoms.set_cell([(50,0,0),(0,50,0),(0,0,50)])
    else: # Add large dummy box to make quippy happy
        #r_max = np.max([np.linalg.norm(np.asarray([atom.position - atoms.get_center_of_mass()])) for atom in atoms])
        #atoms.set_cell([2 * (r_max + 50), ] * 3)
        #atoms.set_center_of_mass([r_max + 50, ] * 3)
        add_dummy_cell(atoms)

    write(f"../input_training_data_iter_{iteration+1}.xyz", atoms, format="extxyz", append="w")


def make_trainingset_file(iteration, submitdir, peratomsigma = False, forcemask=True, update="iterative", code="aims",
                          free_atom_e=None):
    """
    This function makes training set file from DFT calculation

    Parameters :
    ------------
    iteration : int
        GAP fitting iteration
    peratomsigma : bool
        Whether to use peratomsigma
    forcemask : bool
        Whether to use forcemask for 3 lower metal atoms.
    update : str
        Either 'iterative' or 'initial' or 'patch'

    """
    if update == "iterative":
        for i in range(5):
            f_name = "/".join([submitdir, f"iter_{iteration}", "3_DFT_minhop", f"{i}_structure"])
            os.chdir(f_name)
            write_xyz(iteration, peratomsigma=peratomsigma, forcemask=forcemask, code=code, free_atom_e=free_atom_e)

    elif update == "initial":
        for i in range(5):
            f_name = "/".join([submitdir, "z_init", f"{i}_structure"])
            os.chdir(f_name)
            write_xyz(iteration, peratomsigma=peratomsigma, forcemask=forcemask, initial=True, code=code,
                      free_atom_e=free_atom_e)
        training_file_dir = "/".join([submitdir, "z_init"])
        subprocess.run(["cp", training_file_dir+f"/input_training_data_iter_{iteration+1}.xyz", submitdir])
    elif update == "patch":
        for i in range(5):
            f_name = f"{submitdir}/c_DFT_singlepoint4cluster/{i}_structure"
            os.chdir(f_name)
            write_xyz(iteration, peratomsigma = peratomsigma, forcemask = forcemask, code=code, free_atom_e=free_atom_e)


if __name__ == "__main__":

    args = get_encode_argparse()
    iteration = int(args.iteration)

    atoms = read_aims_output("stdout.log")
    lines = os.popen('grep -A3 "Energy and forces in a compact form:" stdout.log').read()
    atoms.info["F_el"] = float(lines.split("\n")[3].split()[5])
    atoms.info["E_uncorr"] = float(lines.split("\n")[1].split()[5])
    atoms.info["E_corr"] = float(lines.split("\n")[2].split()[5])
    atoms.info["F_AE"] = get_atomization_energy(atoms, atoms.info["F_el"]) if len(atoms) > 1 else 0.0
    atoms.info["F_AE_N"] = atoms.info["F_AE"] / len(atoms) if len(atoms) > 1 else 0.0

    if int(os.getcwd().split("/")[-1].split("_")[0]) < 3:
        atoms.info["source"] = "minima"
    elif int(os.getcwd().split("/")[-1].split("_")[0]) >= 3:
        atoms.info["source"] = "mdtraj"


    if args.peratomsigma:
        atoms.arrays["force_atom_sigma"] = get_force_atom_sigma(atoms.get_forces())
    else:
        atoms.arrays.pop("force_atom_sigma", None)


    if args.forcemask:
        atoms.arrays["force_mask"] = mask_forces(atoms)
    else:
        atoms.arrays.pop("force_mask", None)


    atoms.arrays["forces_dft"] = atoms.get_forces()
    if len(atoms) == 1 or len(atoms) == 2:
        atoms.set_cell([(50,0,0),(0,50,0),(0,0,50)])


    write(f"../input_training_data_iter_{iteration+1}.xyz", atoms, format="extxyz", append="w")

