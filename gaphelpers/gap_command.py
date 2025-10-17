from ase.io import read, write
import numpy as np
import yaml
from universalSOAP.hypers import SOAP_hypers
from itertools import combinations_with_replacement
from sklearn.metrics import mean_absolute_error as mae, root_mean_squared_error as rmse
from hyperparameter import get_atomization_energy, flatten_forces, get_hyperparams
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("agg")
from pathlib import Path
from shutil import copyfile
import os
from collections import deque

def get_fit_command(
    energy_sigma,
    force_sigma,
    stage,
    iteration,
    forcemask = False,
    glue_baseline = str(Path(__file__).parent.resolve()) + "/glue_baseline.xml",
    code = "aims",
    free_atom_e=None,
):
    """
    Quick and dirty function that generates the general gap_fit
    command without the gap parameter. In this case
    only parameters are inputs that can be varied
    which change during our fit procedure.

    Parameters :
    ------------
    energy_sigma : float
        energy sigma for regularization
    force_sigma : float
        force sigma for regularization
    statge : int
        if 1 2-body potential is fitted and if 2 many-body potential is build based of residual error of 2-body potential
    iteration : int
        Iteration number of GAP fitting
    forcemask : bool
        whether to use force mask for lowest three metal layer.
    glue_baseline : str
        path for baseline potential.
    free_atom_e : dict

    Return :
    --------
    gap_fit_comm : str
        gap fitting command as string.
    """

    train_filename = f'input_training_data_iter_{iteration}.xyz'

    if stage == 1:
        gp_file = f'GAP_2b_iter_{iteration}.xml'
    if stage == 2:
        gp_file = f'GAP_2b_soap_iter_{iteration}.xml'

    gap_fit_comm = "! gap_fit"

    gap_fit_comm += " default_sigma={"
    gap_fit_comm += f"{energy_sigma} {force_sigma} 0 0"
    gap_fit_comm += "}"
    gap_fit_comm += f" do_copy_at_file=F"
    gap_fit_comm += f" sparse_separate_file=F"

    gap_fit_comm += f" gp_file={gp_file}"
#	gap_fit_comm += f" e0=0 e0_method=average"
    if code == "aims":
        atom_e_str = ''
        for x in free_atom_e.keys():
            if x == list(free_atom_e.keys())[-1]:
                atom_e_str += f"{x}:{free_atom_e[x]}"
            else:
                atom_e_str += f"{x}:{free_atom_e[x]}:"

        gap_fit_comm += " e0={"+atom_e_str+"}" #" e0={H:-13.7212718188775:C:-1030.41241330161:O:-2045.24380597435:Rh:-131618.235069883}"
    elif code == "qe":
        gap_fit_comm += " e0={H:-13.231438493358885:C:-162.96959390424973:O:-438.377323853479:Rh:-3088.9768277322223}"
    gap_fit_comm += f" core_param_file={glue_baseline}"
    gap_fit_comm += " core_ip_args={IP Glue}"
    gap_fit_comm += f" at_file={train_filename}"
    gap_fit_comm += f" energy_parameter_name=F_el"
    gap_fit_comm += f" force_parameter_name=forces_dft"
    if forcemask:
        gap_fit_comm += f" force_mask_parameter_name=force_mask"

    return gap_fit_comm



def get_gap_command(stage, 
                    iteration,
                    deltas,
                    tmpdir,
                    forcemask=False,
                    multiple_universal_soap=False,
                    universal_soap_yaml=None,
                    ):
    """
    This function specifies setting related to representation used in GAP

    Parameters :
    ------------
    stage : int
        if 1 2-body potential is fitted and if 2 many-body potential is build based of residual error of 2-body potential
    iteration : int
        iteration number of GAP fitting
    deltas : tuple or list
        delta values which weighs 2-body and many-body contributions
    tmpdir : str
        directory for scratch folder
    forcemask : bool
        whether to use force mask for metal.
    multiple_universal_soap : bool
        whether to use multiple SOAP as give by universal SOAP.
        Using multiple soap has minimum advantage as opposed to significantly increased computational cost for GAP fitting.
    univeral_soap_yaml : str
        path for univeral soap YAML file
    """
    training_file = tmpdir + f"/iter_{iteration}/input_training_data_iter_{iteration}.xyz"
    refdata = read(training_file + "@:")

    atomic_number_cho = [6, 1, 8]  # Atomic number for C H O
    atom_pairs = set()
    atomic_numbers_list = list({struc.get_atomic_numbers().tobytes() : struc.get_atomic_numbers() for idx, struc in enumerate(refdata)}.values())
    for a in atomic_numbers_list:
        uniques, counts = np.unique(a, return_counts = True)
        atom_pairs.update(list(combinations_with_replacement(uniques, 2)))

        for i in np.where(counts == 1)[0]:
            remove = uniques[i]
            atom_pairs.remove((remove, remove))

    atom_pairs = list(atom_pairs)

    command = " gap={"
    for idx, atom_pair in enumerate(atom_pairs, 1):
        command += f"distance_Nb order=2 cutoff=5.0 delta={deltas[0]}"
        command += " covariance_type=ard_se"
        command += " n_sparse=15"
        command += " theta_uniform=1.0"
        command += " sparse_method=RANDOM"
        command += " add_species=F"
        command += " Z={{"+"{0} {1}".format(*atom_pair) + "}}"
        if stage == 1 and len(atom_pairs) == idx:
            command += "}"
            return command
        elif len(atom_pairs) != idx:
            command += " : "

    if multiple_universal_soap:
        print("Using Universal SOAP")
        length_scales = yaml.safe_load(open(universal_soap_yaml))
        universal_soap = SOAP_hypers(uniques,
                                     length_scales = length_scales,
                                     spacing = 1.5,
                                     no_extra_inner=True,
                                     no_extra_outer=True,)

        for Z in universal_soap.keys():
            for soap in universal_soap[Z]:

                command += f" :"
                command += f" soap"
                command += f" cutoff={soap['cutoff']} l_max=3 n_max=9"
                command += f" atom_sigma={soap['atom_gaussian_width']}"
                command += f" cutoff_transition_width={soap['cutoff_transition_width']}"
                command += f" covariance_type=dot_product"
                command += f" delta={deltas[1]}"
                command += f" zeta=4"
                command += f" add_species=F"
                command += f" n_species={len(universal_soap.keys())}"
                command += f" Z={Z}"
                command += " species_Z={{" + " ".join(str(Z) for Z in universal_soap.keys()) + "}}"
                if forcemask:
                    command += " n_sparse={}".format(2000 if Z in atomic_number_cho else 1000)
                else:
                    command += f" n_sparse=2000"

                command += f" sparse_method=cur_points"

    else:
        length_scales = yaml.safe_load(open(universal_soap_yaml))
        universal_soap = SOAP_hypers(uniques,
                                    length_scales = length_scales,
                                    spacing = 1.5,
                                    no_extra_inner=True,
                                    no_extra_outer=True
                                       )
        # Only single soap is selected by removing one with larger cutoff
        for Z in universal_soap:
            toremove = np.argmax(np.asarray([soapdict["cutoff"] for soapdict in universal_soap[Z]]))
            universal_soap[Z].remove(universal_soap[Z][toremove])

        for Z, soap in universal_soap.items():
            soap = soap[0]
            command += f" :"
            command += f" soap"
            command += f" cutoff={soap['cutoff']} l_max=3 n_max=9"
            command += f" atom_sigma={soap['atom_gaussian_width']}"
            command += f" cutoff_transition_width={soap['cutoff_transition_width']}"
            command += f" covariance_type=dot_product"
            command += f" delta={deltas[1]}"
            command += f" zeta=4"
            command += f" add_species=F"
            command += f" n_species={len(universal_soap.keys())}"
            command += f" Z={Z}"
            command += " species_Z={{" + " ".join(str(Z) for Z in universal_soap.keys()) + "}}"
            if forcemask:
                command += " n_sparse={}".format(2000 if Z in atomic_number_cho else 1000)
            else:
                command += f" n_sparse=2000"

            command += f" sparse_method=cur_points"


    if stage == 2:
        command += "}"
    return command



def get_SOAP_param_dict(
        structures,
        delta = 0.4,
        universal_soap_yaml=None,
        ):

    length_scales = yaml.safe_load(open(universal_soap_yaml))

    # refdata = read(training_file + "@:")
    refdata = structures

    atom_pairs = set()
    atomic_numbers_list = list({struc.get_atomic_numbers().tobytes() : struc.get_atomic_numbers() for idx, struc in enumerate(refdata)}.values())
    for a in atomic_numbers_list:
        uniques, counts = np.unique(a, return_counts = True)

    universal_soap = SOAP_hypers(uniques,
                                length_scales = length_scales,
                                spacing = 1.5,
                                no_extra_inner=True,
                                no_extra_outer=True,
                                   )
    for Z in universal_soap:
        toremove = np.argmax(np.asarray([soapdict["cutoff"] for soapdict in universal_soap[Z]]))
        universal_soap[Z].remove(universal_soap[Z][toremove])

    desc_dicts = []
    for Z in universal_soap.keys():
        desc_dicts.append({
                "Z" : Z,
                "add_species" : False,
                "atom_sigma" : universal_soap[Z][0]["atom_gaussian_width"],
                "central_weight" : 1.0,
                "covariance_type" : "dot_product",
                "cutoff" : universal_soap[Z][0]["cutoff"],
                "cutoff_transition_width" : universal_soap[Z][0]["cutoff_transition_width"],
                "delta" : delta,
                "f0" : 0.0,
                "l_max" : 3,
                "n_max" : 9,
                "n_sparse" : 2000,
                "sparse_method" : "cur_point",
                "n_species" : len(universal_soap.keys()),
                "print_sparse_index" : True,
                "soap": True,
                "species_Z" : [ele for ele in universal_soap.keys()],
                "zeta" : 4,
                "average" : True
        })

    return desc_dicts



def plot_accuracy(iteration, reference_dat, quip_results, param_file, code="aims", free_atom_e=None):
    """
    Plot Parity plot for DFT and GAP for energy and forces

    Parameters :
    ------------
    iteration : int
        iteration number of GAP fitting
    reference_dat : list
        list of ase atoms object with DFT reference
    quip_results : list
        list of ase atoms object with GAP prediction
    param_file : str
        string indicating either 2-body or many-body

    Return :
    --------
    statistics_dict : dict
        dictionary containing RMSD and MAE of energy and force components.
        This is training error where structures to be tested are already in training set.
    """
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(10, 5), facecolor='w')
    statistics_dict = {}

    # plot energy
    E_ref = np.asarray([atoms.info["F_AE"]  for atoms in reference_dat])
    #E_gap = np.asarray([atoms.info["energy"] for atoms in quip_results])
    E_gap = np.asarray([get_atomization_energy(atoms, atoms.get_potential_energy(), code=code, free_atom_e=free_atom_e) for atoms in quip_results] )
    ax1.scatter(E_ref, E_gap)
    ax1.plot([np.min(E_ref), np.max(E_ref)],
             [np.min(E_ref), np.max(E_ref)],'-.', color='k')
    ax1.set_xlabel("AE (DFT) [eV]", fontsize=14)
    ax1.set_ylabel("AE (GAP) [eV]", fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=14)

    statistics_dict["RMSD_E_train"] = rmse(E_ref, E_gap)
    statistics_dict["MAE_E_train"] = mae(E_ref, E_gap)

    ax1.annotate(f"RMSD : {statistics_dict['RMSD_E_train']:0.3f} eV",
        (0.05,0.9), xycoords='axes fraction', fontsize=14)
    ax1.annotate(f"MAE : {statistics_dict['MAE_E_train']:0.3f} eV",
        (0.05,0.82), xycoords='axes fraction', fontsize=14)
    ax1.set_title(f"Energy {param_file}iter_{iteration}", fontsize = 14)

    # plot forces
    F_ref = flatten_forces([atoms.arrays["forces_dft"] for atoms in reference_dat])
    F_gap = flatten_forces([atoms.arrays["force"] for atoms in quip_results])
    ax2.scatter(F_ref, F_gap)
    ax2.plot([np.min(F_ref), np.max(F_ref)],
             [np.min(F_ref), np.max(F_ref)],'-.', color='k')
    ax2.set_xlabel("Force (DFT) [eV/$\AA$]", fontsize=14)
    ax2.set_ylabel("Force (GAP) [eV/$\AA$]", fontsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=14)

    statistics_dict["RMSD_F_train"] = rmse(F_ref, F_gap)
    statistics_dict["MAE_F_train"] = mae(F_ref, F_gap)

    ax2.annotate(f"RMSD : {statistics_dict['RMSD_F_train']:0.3f} eV/$\AA$",
        (0.05,0.9), xycoords='axes fraction', fontsize=14)
    ax2.annotate(f"MAE : {statistics_dict['MAE_F_train']:0.3f} eV/$\AA$",
        (0.05,0.82), xycoords='axes fraction', fontsize=14)
    ax2.set_title(f"Force {param_file}iter_{iteration}", fontsize = 14)

    plt.tight_layout()
    plt.savefig('E_correlation.png', dpi=300)


    return statistics_dict


def get_validation(iteration, submitdir, calculate=True,
                   potential_file="previous", calc_all_minima=False, code="aims", free_atom_e=None):
    """
    Get validation error on structures not in the training set.

    Parameters :
    ------------
    iteration : int
        iteration number of GAP fitting
    submitdir : str
        submission directory
    calculate : bool
        whether validation quip will be executed
    potential_file : str
        file name of potential file to be used.
    free_atom_e : dict

    Return :
    --------
    statistics_dict : dict
        dictionary containing validation results of current GAP iteration.
        Among keywords added, 'RMSD_E_val_minima' and 'RMSD_F_val_minima' is used for convergence criterion.
    """

    if potential_file == "previous":
        potential_file = f"../GAP_2b_soap_iter_{iteration}/GAP_2b_soap_iter_{iteration}.xml"

    print(f"Newly earned structures will be validated using the potential : {potential_file}")
    os.chdir(submitdir + f"/iter_{iteration}/3_DFT_minhop/")
    quip_comm = f"! quip E=T F=T atoms_filename=new_training_set_iter{iteration}.xyz param_filename={potential_file} | grep AT | sed 's/AT//' > quip_new_training_data_iter_{iteration}.xyz"
    if calculate:
        os.system(quip_comm)

    if calc_all_minima:
        # DFT data
        dft_result = read(f"{submitdir}/iter_{iteration}/2_Minhop/input_training_data_iter_{iteration+1}.xyz@:")
        dft_E_val = [atoms.info["F_AE"] for atoms in dft_result ]
        dft_F_val = [flatten_forces(atoms.arrays["forces_dft"]) for atoms in dft_result ]
#		dft_F_val = np.array([flatten_forces(atoms.arrays["forces_dft"]) for atoms in dft_result ]).ravel()

        # GAP data
        gap_result = read(f"{submitdir}/iter_{iteration}/2_Minhop/minima.traj@:")
        gap_E_val = [get_atomization_energy(atoms, atoms.get_potential_energy(apply_constraint=False), code=code, free_atom_e=free_atom_e) for atoms in gap_result]
        gap_F_val = [flatten_forces(atoms.get_forces(apply_constraint=False)) for atoms in gap_result]
#		gap_F_val = np.array([flatten_forces(atoms.get_forces(apply_constraint=False)) for atoms in gap_result]).ravel()

    else:
        # DFT data
        # Evaluating force using current implementation is not recommended.
        # It might as well changed as right above with "ravel" function.
        dft_result = read(f"input_training_data_iter_{iteration+1}.xyz@-5:")
        dft_E_val = [atoms.info["F_AE"] for atoms in dft_result ]
        dft_F_val = [flatten_forces(atoms.arrays["forces_dft"]) for atoms in dft_result ]

        # GAP data
        gap_result = read(f"quip_new_training_data_iter_{iteration}.xyz@:")
        gap_E_val = [get_atomization_energy(atoms, atoms.get_potential_energy(), code=code, free_atom_e=free_atom_e) for atoms in gap_result]
        gap_F_val = [flatten_forces(atoms.arrays["force"]) for atoms in gap_result]

    statistics_dict = {}

#	statistics_dict["RMSD_E_val"] = rmse(dft_E_val, gap_E_val, )
#	statistics_dict["RMSD_F_val"] = rmse(dft_F_val, gap_F_val, )
#	statistics_dict["MAE_E_val"] = mae(dft_E_val, gap_E_val)
#	statistics_dict["MAE_F_val"] = mae(dft_F_val, gap_F_val)
    if calc_all_minima:
        statistics_dict["RMSD_E_val_minima"] = rmse(dft_E_val, gap_E_val, )
        statistics_dict["RMSD_F_val_minima"] = rmse(dft_F_val, gap_F_val, )
        statistics_dict["RMSD_E_val"] = statistics_dict["RMSD_E_val_minima"]
        statistics_dict["RMSD_F_val"] = statistics_dict["RMSD_F_val_minima"]
        statistics_dict["MAE_E_val_minima"] = mae(dft_E_val, gap_E_val)
        statistics_dict["MAE_F_val_minima"] = mae(dft_F_val, gap_F_val)

    else:
        dft_E_val_minima = [atoms.info["F_AE"] for atoms in dft_result if atoms.info["source"] == "minima" ]
        dft_F_val_minima = [flatten_forces(atoms.arrays["forces_dft"]) for atoms in dft_result if atoms.info["source"] == "minima" ]
        # Error for all five samples
        statistics_dict["RMSD_E_val"] = rmse(dft_E_val, gap_E_val, )
        statistics_dict["RMSD_F_val"] = rmse(dft_F_val, gap_F_val, )

        # Error for only 3 minima
        statistics_dict["RMSD_E_val_minima"] = rmse(dft_E_val_minima, gap_E_val[:3], )
        statistics_dict["RMSD_F_val_minima"] = rmse(dft_F_val_minima, gap_F_val[:3], )
        statistics_dict["MAE_E_val_minima"] = mae(dft_E_val_minima, gap_E_val[:3])
        statistics_dict["MAE_F_val_minima"] = mae(dft_F_val_minima, gap_F_val[:3])

    print("Validation completed !!")

    return statistics_dict


def get_commands(
                iteration,
                energy_sigma,
                force_sigma,
                stage,
                tmpdir,
                deltas,
                param_file,
                forcemask,
                multiple_universal_soap,
                code = "aims",
                free_atom_e=None,
                universal_soap_yaml=None,
                 ):
    """
    This function generates gap fitting command

    Parameters :
    ------------
    iteration : int
        iteration number of GAP fitting
    energy_sigma : float
        energy sigma for regularization
    force_sigma : float
        force sigma for regularization
    stage : int
        if 1 2-body potential is fitted and if 2 many-body potential is build based of residual error of 2-body potential
    tmpdir : str
        directory for scratch folder
    deltas : tuple or list
        delta values which weighs 2-body and many-body contributions
    param_file : str
        file name of potential file to be used.
    forcemask : bool
        whether to use force mask for lowest three metal layer.
    multiple_universal_soap : bool
        whether to use multiple SOAP as give by universal SOAP.
    free_atom_e : dict

    Return :
    --------
    f_gap_fit_comm : str
        command line string for GAP fitting.
    quip_comm : str
        command line string for Validation
    """
    train_filename = f'input_training_data_iter_{iteration}.xyz'

    gap_fit_comm = get_fit_command(
            energy_sigma,
            force_sigma,
            stage=stage,
            iteration=iteration,
            forcemask=forcemask,
            code=code,
            free_atom_e=free_atom_e,
    )

    gap_fit_comm += get_gap_command(stage=stage,
                                    iteration=iteration,
                                    deltas=deltas,
                                    tmpdir=tmpdir,
                                    forcemask=forcemask,
                                    multiple_universal_soap=multiple_universal_soap,
                                    universal_soap_yaml=universal_soap_yaml,
                                    )

    params_out = f'params_stage_{stage}_iter_{iteration}.out'
    file_err = f'file_err_stage_{stage}_iter_{iteration}.out'
    f_gap_fit_comm = f'{gap_fit_comm} 1>{params_out} 2>{file_err}'

    quip_comm = f"! quip E=T F=T atoms_filename={train_filename} param_filename={param_file}iter_{iteration}.xml | grep AT | sed 's/AT//' > quip_train_{param_file}iter_{iteration}.xyz"

    # Generate results_folder
    f_name = f'{param_file}iter_{iteration}'
    Path(f_name).mkdir(parents=True, exist_ok = True)

    os.system(f"rsync -az {tmpdir}/iter_{iteration}/{train_filename} {tmpdir}/iter_{iteration}/{f_name}/{train_filename}")

    return f_gap_fit_comm, quip_comm


def gap_fitting(iteration,
                tmpdir,
                forcemask,
                multiple_universal_soap = False,
                percentage = 0.001,
                verbose = True,
                code = "aims",
                free_atom_e=None,
                universal_soap_yaml=None,
                ):
    """
    This function gathers all the hyperparameters using heuristics and carrys out GAP fitting and its validation.
    In first stage 2-body potential is fitted and many-body potential is fitted in seconde stage.

    Parameters :
    ------------
    iteration : int
        Iteration number of GAP fitting
    tmpdir : str
        directory for scratch folder
    forcemask : bool
        whether to use force mask for lowest three metal layer.
    multiple_universal_soap : bool
        whether to use multiple SOAP as give by universal SOAP.
    percentage : float
        fractional number multiplied to energy sigma
    verbose : bool
        Wheter to explicitly print the commands
    free_atom_e : dict

    Return :
    --------
    plot_accuracy : func
        Parity plot is made for training set error.
    """
    deltas = deque()
    stages = 2

    if verbose:
        print("--------------------------------------------")
        print(f"Iterative GAP fitting : iter #{iteration}")
        print("--------------------------------------------")

    for stage in range(1, stages+1):

        os.chdir( tmpdir + f'/iter_{iteration}' )

        if verbose:
            print("Stage : ", stage)
        if stage == 1:
            param_file = 'GAP_2b_'
            quip_results = None
        else:
            param_file = 'GAP_2b_soap_'

        # Read input data
        reference_dat = read(tmpdir + f'/iter_{iteration}/input_training_data_iter_{iteration}.xyz@:')

        delta, energy_sigma, force_sigma = get_hyperparams(reference_dat,
                                                           stage,
                                                           quip_results=quip_results,
                                                           percentage=percentage, free_atom_e=free_atom_e)
        deltas.append(delta)

        f_gap_fit_comm, quip_comm = get_commands(iteration,
                                                 energy_sigma,
                                                 force_sigma,
                                                 stage,
                                                 tmpdir,
                                                 deltas,
                                                 param_file,
                                                 forcemask = forcemask,
                                                 multiple_universal_soap = multiple_universal_soap,
                                                 code = code,
                                                 free_atom_e=free_atom_e,
                                                 universal_soap_yaml=universal_soap_yaml)

        # Perform fits
        os.chdir(f'{param_file}iter_{iteration}')
        if verbose:
            print(f_gap_fit_comm)
        os.system(f_gap_fit_comm)
        if verbose:
            print(quip_comm)
        os.system(quip_comm)
        quip_results=read(f"quip_train_{param_file}iter_{iteration}.xyz@:")

        if verbose:
            print(f'{param_file[:-1]} fitting done\n')

    # Plottings
    return plot_accuracy(iteration, reference_dat, quip_results, param_file, code=code, free_atom_e=free_atom_e)


