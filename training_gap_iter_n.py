#!/bin/sh
''''exec python -u -- "$0" ${1+"$@"} # '''
# vi: syntax=python

# This script can be used to train a GAP potential
# This is modified by Hyunwook @ 28th Jan 2022 on Raven

from ase.io import read, write
import os, sys, pandas as pd, numpy as np, glob, pickle
from shutil import copyfile, rmtree
#from gaphelpers.sample_training_set import sample_training_set, check_slurm_completion, prepare_initial_config
from gaphelpers.gap_command import gap_fitting, get_validation, plot_accuracy
from gaphelpers.minimahopping import minimahopping, prepare_input, check_GAP_convergence, Run_parallel_minima_hopping, check_GAP_convergence_after_parallel_MH, sample_training_set, check_slurm_completion, prepare_initial_config, prepare_isolated_atom
from gaphelpers.get_argparse import get_argparse
from gaphelpers.encode import make_trainingset_file, parse_output
from gaphelpers.write_slurm import get_adsorbate_info, write_dft_slurm
#from gaphelpers.hyperparameter import get_atomization_energy
from gaphelpers.cluster import cluster_sample
from pathlib import Path
import time
#from sklearn.metrics import mean_squared_error as rmse

args = get_argparse()
################################
submitdir = os.getcwd()
scratchfolder = submitdir +"/"+ "myGapScratch"
universal_soap_yaml = f"{args.gaphelper}/length_scales_VASP_auto_length_scales.yaml"

# make temp folder
if args.noscratch:
    tmpdir = submitdir
else:
    tmpdir = scratchfolder
Path(tmpdir).mkdir(parents=True, exist_ok = True)

################################

# Automatic determination of starting iteration
# if there's no proceeded training iteration
#last_iter = sorted([fn for fn in next(os.walk(submitdir))[1] if fn.startswith("iter_")], key =lambda s:int(s.split("_")[1]))[-1]
#last_iter = int(last_iter.split("_")[1])
if len([fn for fn in next(os.walk(submitdir))[1] if fn.startswith("iter_")]) == 0:
    start_iter = int(args.iteration)
else:
    last_iter = sorted([fn for fn in next(os.walk(submitdir))[1] if fn.startswith("iter_")], key =lambda s:int(s.split("_")[1]))[-1]
    last_iter = int(last_iter.split("_")[1])

    if last_iter != int(args.iteration) or last_iter == 0:
        print(f"last iteration was {last_iter}. Resume workflow from there. ")
        start_iter = last_iter + 1
    else:
        start_iter = int(args.iteration)

if args.enditeration == None:
    end_iter = start_iter
else:
    end_iter = int(args.enditeration)

### This part sets the input params for building np ###
miller_index = []
surf_e_list = []
#for index in args.surface_energy:
#    miller_index.append(index)
#    surf_e_list.append(args.surface_energy[index])

np_params = {"parent_metal": args.parent,
             "dopant": args.dopant,
             "lattice": "fcc",
             "surface_indices": [(1, 1, 1), (1, 1, 0), (1, 0, 0)],
             "surface_energies": [1.447, 1.660, 1.584],
             "size": args.size,
             "one_out_of_n": args.one_out_of_n}

if args.quantum_espresso:
    lattice_param, code = 3.6, "qe"
else:
    lattice_param, code = 3.575, "aims"
###########################################################

isolated_atom_energy = prepare_isolated_atom(**np_params, lattice_param=lattice_param, forcemask=args.force_mask,
                                             code=code, path_to_workflow=args.gaphelper)

if start_iter == 0 and not os.path.isfile(submitdir + "/input_training_data_iter_0.xyz"):

    prepare_initial_config(**np_params, forcemask=args.force_mask, lattice_param=lattice_param, code=code,
                           free_atom_e=isolated_atom_energy, path_to_workflow=args.gaphelper)

if start_iter != 0 and os.path.isfile(submitdir + "/statistics.csv"):
    statistics = pd.read_csv(submitdir + "/statistics.csv", index_col=0).transpose().to_dict()
#elif calc_all_minima:
#	statistics = pd.read_csv(submitdir + "/all_minima_statistics.csv", index_col=0).transpose().to_dict()
else:
    statistics = {}

timing = {}

# ===========================
# Start Iterative GAP fitting
# ===========================
for iteration in range(start_iter, end_iter+1):

    if iteration >= 5 and check_GAP_convergence(statistics, submitdir, sampling_method=args.sampling,
                                                calc_all_minima=args.calc_all_minima, convergence=args.convergence):
        break

    os.chdir(tmpdir)
    Path(tmpdir + f'/iter_{iteration}').mkdir(parents=True, exist_ok=True)
    if iteration == 0:
        copyfile(submitdir + f'/input_training_data_iter_0.xyz',
                 tmpdir + f'/iter_{iteration}/input_training_data_iter_{iteration}.xyz')
    else:
        os.system(f"rsync -az {submitdir}/iter_{iteration-1}/3_DFT_minhop/input_training_data_iter_{iteration}.xyz " +
                  f"{tmpdir}/iter_{iteration}/input_training_data_iter_{iteration}.xyz")

    # GAP fitting
    #os.environ["OMP_NUM_THREADS"] = "36"
    f_name = f"{tmpdir}/iter_{iteration}/GAP_2b_soap_iter_{iteration}"
    if os.path.isdir(f"{f_name}") and os.path.isfile(f"{f_name}/GAP_2b_soap_iter_{iteration}.xml"):
        # Replacement of segmentation error
        print("GAP_fitting is already done! Resuming to minima hopping")
        reference_dat = read(tmpdir + f'/iter_{iteration}/input_training_data_iter_{iteration}.xyz@:')
        quip_results = read(tmpdir + f"/iter_{iteration}/GAP_2b_soap_iter_{iteration}/quip_train_GAP_2b_soap_iter_{iteration}.xyz@:")
        statistics[iteration] = plot_accuracy(iteration, reference_dat, quip_results, 'GAP_2b_soap_', code=code, free_atom_e=isolated_atom_energy)

    else:
        start = time.time()
        statistics[iteration] = gap_fitting(iteration, tmpdir, forcemask=args.force_mask,
                                            multiple_universal_soap = args.multiple_universal_soap, code=code,
                                            free_atom_e=isolated_atom_energy, universal_soap_yaml=universal_soap_yaml,
                                            )
        timing["gapfit"] = time.time() - start


    # Minima hopping
    #os.environ["OMP_NUM_THREADS"] = "1"
    start = time.time()
    prepare_input(**np_params, iteration=iteration, tmpdir=tmpdir, randseed=iteration, lattice_param=lattice_param)

    minimahopping(iteration,
                  f"../GAP_2b_soap_iter_{iteration}/GAP_2b_soap_iter_{iteration}.xml",
                  tmpdir=tmpdir,
                  randseed=iteration,
                  statistics=statistics,
                  timestep=0.5,
                  relax_metal=args.relaxmetal,
                  **np_params,
                  lattice_param=lattice_param,
                  T0=args.initial_temperature)
    timing["minhop"] = time.time() - start

    # DFT calculation
    os.chdir(submitdir)
    if tmpdir != submitdir:
        os.system(f"rsync -az {tmpdir}/iter_{iteration} {submitdir}")
        rmtree(tmpdir + f"/iter_{iteration}")

    start = time.time()
    job_ids = sample_training_set(iteration, forcemask=args.force_mask, sampling_method=args.sampling,
                                  parallel=args.parallel, calculate_all_minima=args.calc_all_minima,
                                  code=code, path_to_workflow=args.gaphelper, universal_soap_yaml=universal_soap_yaml)

    print("DFT job submitted.")
    if check_slurm_completion(job_ids):

        if args.calc_all_minima:
            for j in range(len(read(f"{submitdir}/iter_{iteration}/2_Minhop/minima.traj@:"))):
                print(j, end = " ")
                os.chdir(f"{submitdir}/iter_{iteration}/2_Minhop/{j}_structure")
                atoms = parse_output(free_atom_e=isolated_atom_energy)
                atoms.arrays["forces_dft"] = atoms.get_forces()
                write(f"{submitdir}/iter_{iteration}/2_Minhop/input_training_data_iter_{iteration+1}.xyz",
                      atoms, format="extxyz", append="w")

        print("DFT job all finished")

    make_trainingset_file(iteration, submitdir, forcemask=args.force_mask, code=code, free_atom_e=isolated_atom_energy)

    timing["DFT"] = time.time() - start


    # Validation (GAP Evaluation)
    start = time.time()
    validation_error = get_validation(iteration, submitdir, calc_all_minima = args.calc_all_minima, code=code, free_atom_e=isolated_atom_energy)
    if iteration in statistics.keys():
        statistics[iteration] = {**statistics[iteration], **validation_error}
    else:
        statistics[iteration] = validation_error

    timing["validation"] = time.time() - start

    print(statistics[iteration])

    if os.path.isfile(submitdir + "/statistics.csv"):
        pd.DataFrame.from_dict({iteration : statistics[iteration]}).transpose().to_csv(submitdir + "/statistics.csv", mode="a", header=False)
    else:
        pd.DataFrame.from_dict({iteration : statistics[iteration]}).transpose().to_csv(submitdir + "/statistics.csv")

    for key, val in timing.items():
        print(f"iter #{iteration} {key} : {val:.2f} s ")
    print(f"iter #{iteration} total : {sum(list(timing.values())):.2f} s" )




# =======================
# Parallel Minima hopping
# =======================
Run_parallel_minima_hopping(iteration, submitdir, path_to_workflow=args.gaphelper, parallel=180,
                            lattice_param=lattice_param, relax_metal=args.relaxmetal,
                            t0=args.initial_temperature, **np_params)



# ===============
# DFT Relaxation
# ===============

# Check GAP convergence; collect minima with kPCA and k-means clustering
if not check_GAP_convergence_after_parallel_MH(submitdir, code=code,
                                               free_atom_e=isolated_atom_energy, path_to_workflow=args.gaphelper):
    print(f"Error for selected five minima is too high. ")
#
#	# In case you don't know what's the iteration number
#	iteration = int(sorted(glob.glob("iter_*"), key=lambda x:int(x.split("_")[-1]))[-1].split("_")[-1]) + 1
#	# copy xyz file to current folder
#	os.system(f"rsync -az {submitdir}/iter_{iteration-1}/3_DFT_minhop/input_training_data_iter_{iteration}.xyz "
#			  f"{submitdir}/c_DFT_singlepoint4cluster/input_training_data_iter_{iteration}.xyz") 
#	make_trainingset_file(iteration, submitdir, forcemask=args.force_mask, update="patch", code=code)
#		
#	os.chdir(tmpdir)
#	Path( tmpdir + f'/iter_{iteration}').mkdir(parents=True, exist_ok = True)
#	os.system(f"rsync -az {submitdir}/c_DFT_singlepoint4cluster/input_training_data_iter_{iteration}.xyz " +
#			   f"{tmpdir}/iter_{iteration}/input_training_data_iter_{iteration}.xyz")
#	statistics[iteration] = gap_fitting(iteration, tmpdir, forcemask=args.force_mask,
#										multiple_universal_soap = args.multiple_universal_soap, code=code)
#	pd.DataFrame.from_dict({iteration : statistics[iteration]}).transpose().to_csv(submitdir + "/statistics.csv", mode="a", header=False)
#
#
#	Run_parallel_minima_hopping(iteration, submitdir, args.minimahopping, args.facet, rerun = True, lattice_param=lattice_param, vacuum=vacuum, relax_metal=args.relaxemetal)
#	cluster_sample(args.minimahopping, submitdir=submitdir, only_chemisorption=True, code=code)	
#
#else:


#cluster_sample(args.minimahopping, submitdir=submitdir, only_chemisorption=True,  code = code, free_atom_e=isolated_atom_energy, using_opt_adsorbate=file_params["using_opt_adsorbate"], adsorbate_file=args.adsorbate, path_to_workflow=args.gaphelper)
#cluster_sample(args.minimahopping, submitdir=submitdir, singlepoint = True, only_chemisorption=True,  code = code, free_atom_e=isolated_atom_energy, using_opt_adsorbate=file_params["using_opt_adsorbate"], adsorbate_file=args.adsorbate, path_to_workflow=args.gaphelper)

