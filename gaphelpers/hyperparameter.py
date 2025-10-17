import numpy as np
import ase
from collections import deque
from ase import neighborlist
from collections import Counter


def flatten_forces(forces):
    """
	Function to flat all forces arrays.
	"""
    force_flat = deque()
    for force_array in forces:
        force_flat.extend(list(force_array.flatten()))

    return np.asarray(force_flat)


def get_atomization_energy(atoms, potential_energy, code="aims", free_atom_e=None):
    """
		Calculates the atomization energy of
		that specific atoms object.

	Parameters:
	-------
	atoms : ase.Atoms
		Structure for which the atomization
		energy is calculated.
    free_atom_e: dict
	Return : 
	potential_energy : float 
		atomization energy of the system. 	
	"""
    if code == "aims":
        # Atomic energies calculated with FHI-AIMS
        # RPBE + light setting
        # atomic_energies = {
        # "H" : -13.7212718188775,
        # "C" : -1030.41241330161,
        # "O" : -2045.24380597435,
        # "Rh": -131618.235069883
        # }

        # Calculated with setting chose by user
        atomic_energies = free_atom_e
    else:
        # Atomic energies calculated with QE
        # BEEF-vdw + KE cutoff 500 eV
        atomic_energies = {
            "H": -13.231142200133974,
            "C": -162.96858286157322,
            "O": -438.3769369597124,
            "Rh": -3088.9744248069533
        }

    atoms_per_struc = Counter(atoms.get_chemical_symbols())
    for symb, atomic_energy in atomic_energies.items():
        potential_energy -= (atoms_per_struc[symb] * atomic_energy)

    return potential_energy


def get_Num_bonds(slab):
    """
	Calculates number of chemical bond in the system for 2-body potential fitting. 

	Parameters : 
	------------
	slab : ase object
		slab as ASE atoms object

	Return : 
	--------
	Nbonds : int
		Number of bonds in the system. 
	"""
    cutOff = neighborlist.natural_cutoffs(slab)
    neighbor = neighborlist.NeighborList(cutOff, self_interaction=False, bothways=True)
    neighbor.update(slab)
    Nbonds = np.sum(neighbor.get_connectivity_matrix(sparse=False)) / 2

    return Nbonds


def get_hyperparams(
        reference_dat,
        stage,
        quip_results=None,
        percentage=0.001,  # Accuracy of the fit
        default_params=(0.034, 0.001, 0.05),
        free_atom_e=None):
    """
	Function that approximates the hyperparameter:
	delta, sigma_E and sigma_F.
	Above three hyperparameters are determined based on implemented heuristics.

	Parameters:
	-------
	reference_dat : N-np.array or list
		Fit property for which delta should be calculated.
	stage : int
		Stage either 1 or 2 : 
		if 1, 2-body potential is fitted. 
		if 2, then 2-body + Manybody SOAP potential is fitted. 
	quip_results : N-np.array or list
		Predicted results as array of ase.Atoms object from 2-Body potential
	percentage : float
		floating number to be multiplied to standard deviation of fit property
	default_params : tuple
		List of default hyperparameters for iteration = 0. 
		[delta, sigma_E, sigma_F]
		These are used when evaluation of standard deivation is impossible when there're
		only single structure for traning set. 

	Returns:
	--------
	delta : float
		delta.
	sigma_E : float
		Energy sigma, which is 0.1 percent of the standard deviation.
	sigma_F : float
		Force sigma, np.sqrt(sigma_E).

	"""
    AE_ref = np.asarray([atoms.info["F_AE"] for atoms in reference_dat])
    Natom = np.asarray([atoms.get_global_number_of_atoms() for atoms in reference_dat])
    Nbond = np.asarray([get_Num_bonds(atoms) for atoms in reference_dat])

    if len(AE_ref) < 15:
        if stage == 1:
            return 0.1551, 0.001, 0.05
        elif stage == 2:
            return 0.0318, 0.001, 0.05

    if stage == 1:

        sigma_E = np.std((AE_ref) / Nbond, ddof=1) * percentage
        sigma_F = np.sqrt(sigma_E)
        delta_2B = np.std((AE_ref) / Nbond, ddof=1)

        return delta_2B, sigma_E, sigma_F

    elif stage == 2 and quip_results is not None:

        AE_2B = np.asarray(
            [get_atomization_energy(atoms, atoms.get_potential_energy(), free_atom_e=free_atom_e) for atoms in
             quip_results])

        sigma_E = np.std((AE_ref - AE_2B) / Natom, ddof=1) * percentage
        sigma_F = np.sqrt(sigma_E)
        delta_2B_soap = np.std((AE_ref - AE_2B) / Natom, ddof=1)

        return delta_2B_soap, sigma_E, sigma_F
