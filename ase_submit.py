#!/usr/bin/env python
# If using density matrix initialization, remember to change path2slabdm to your path
# for storing pre-calculated slab density matrices
from ase.calculators.aims import Aims, AimsProfile
from ase.io import read
from ase.io.aims import read_aims_output
import os
import argparse, pickle
from gaphelpers.kgrid import get_k_grid
from elsi_restart.combine_elsi import combine_slab_ads_dm
import subprocess
import fnmatch


def get_dft_args():
    parser = argparse.ArgumentParser(description = "Give in input")
    parser.add_argument("-dr","--DFTrelaxation", nargs="?",type=str, const="adsorption.traj", help = "Perform geometry relaxation with DFT: max force 0.05 eV/A as default.")
    parser.add_argument("-sp","--DFTsinglepoint", nargs="?",type=str, const="adsorption.traj", help = "Perform single point calculation with DFT. default filename: adsorption.traj")
    #parser.add_argument("-dm","--densitymatrix", action="store_true", help="Perform single point calculation with density matrix initialization")
    parser.add_argument("-mh","--minimahopping", action="store_true", help="Perform minima hopping with GAP. Default filename: adsorption.traj")
    parser.add_argument("-smi","--smiles", nargs="?", const=None, default=None, type=str, help="Impose Hookean constraint to molecule. Rcuts are inferred from given SMILES.")
    parser.add_argument("-rm","--relaxmetal", action="store_true", help="Relax upper two metal layers. If not used, all metal slab is constrained")
    parser.add_argument("-p", "--potential", nargs="?",type=str, const="GAP_0.xml", help="filename of GAP potential")
#	parser.add_argument("-ic", "--inherent_constraint", action="store_true", help="Use constraint as stored in *.traj file" )
    parser.add_argument("-ts", "--twostep", action="store_true", help="Use constraint as stored in *.traj file and toggle two step DFT relaxation" )
    parser.add_argument("-sd", "--submitdir", nargs="?",type=str,default="", help="path submit directory")
    parser.add_argument("-r", "--resume", action="store_true", help="Resume DFT relaxation (esp. with two step relaxation)")
    parser.add_argument("-slab", "--slab", action="store_true", help="Path to pre-optimized suface model")
    parser.add_argument("-ads", "--adsorbate", action="store_true", help="Path to pre-optimized adsorbate geometry")

    return parser.parse_args()


args = get_dft_args()

file_params = {"using_opt_surface":False,
               "surface_file":None,
               "using_opt_adsorbate":False,
               "adsorbate_file":None}

if os.path.exists(args.slab):
    file_params["using_opt_surface"] = True
    file_params["surface_file"] = args.slab
if os.path.exists(args.adsorbate):
    file_params["using_opt_adsorbate"] = True
    file_params["adsorbate_file"] = args.adsorbate

aimsbin = "/apps/local/projects/scw1057/software/fhi-aims/bin/aims.231208.scalapack.mpi.x"
species_dir = "/apps/local/projects/scw1057/software/fhi-aims/species_defaults/defaults_2020/light"
mpiexe = "time srun"
#cpu_command = "--nodes=$SLURM_NNODES --ntasks=$SLURM_NTASKS -d mpirun"
outfile = f"{args.submitdir}/stdout.log" if args.submitdir else "stdout.log"
aims_command = "{} {}".format(mpiexe, aimsbin)
sampling_density = 0.019 #Sampling density for k points
profile=AimsProfile(command=aims_command, default_species_directory=species_dir)

dft_params = {"override_warning_libxc": ".true.",
              "xc": 'libxc MGGA_X_MBEEF+GGA_C_PBE_SOL',
              "spin": 'none',
              "relativistic": 'atomic_zora scalar',
              "sc_accuracy_forces": 1e-4,
              "sc_accuracy_etot": 1e-5,
              "sc_accuracy_eev": 1e-3,
              "sc_accuracy_rho": 1e-4
              }


# if clauses managing different type of calculations

# if args.DFTrelaxation:
#
#     filename, extension = os.path.splitext(args.DFTrelaxation)
#     if extension == ".pickle":
#         with open(args.DFTrelaxation, "rb") as h:
#             atoms = pickle.load(h)
#     else:
#         atoms = read(args.DFTrelaxation, index="-1")
#
#
# #	if not args.twostep:
#     if file_params["using_opt_adsorbate"]:
#         apply_constraints(atoms, relax_metal=args.relaxmetal, method="file", constrain_Hookean=True)
#     else:
#         apply_constraints(atoms, relax_metal=args.relaxmetal, smiles=args.smiles, constrain_Hookean=True)
#
#     print(atoms.constraints)
#
#     #aimsbin ="/u/hjung/Softwares/FHIaims2021/build/aims.210716_1.scalapack.mpi.x"
#     #aimsbin = "/apps/local/projects/scw1057/software/fhi-aims/bin/aims.231208.scalapack.mpi.x"
#     #species_dir = "/apps/local/projects/scw1057/software/fhi-aims/species_defaults/defaults_2020/light"
#     #mpiexe = "srun"
#     #outfile = f"{args.submitdir}/stdout.log" if args.submitdir else "stdout.log"
#     #aims_command = "{} {} > {}".format(mpiexe, aimsbin, outfile)
#     #print(aims_command)
#
#     atoms.set_pbc((True, True, False))
#
#     kgrid = get_k_grid(atoms, sampling_density, verbose=True, simple_reciprocal_space_parameters=False)
#
#     calc = Aims(
#         profile=profile,
#         k_grid=kgrid,
#         **dft_params,
#         compute_forces=".true."
#     )
#     calc.template.outputname = outfile
#
#     if args.twostep and args.resume:
#
#         atoms.calc = calc
#         os.rename("bfgs_opt2.traj", "bfgs_opt2_prev.traj")
#         opt = FIRE(atoms, trajectory = "bfgs_opt2_re.traj")
#         opt.run(fmax = 0.05)
#         os.system("ase gui bfgs_opt2_prev.traj bfgs_opt2_re.traj -o bfgs_opt2.traj")
#         os.system("ase gui bfgs_opt1.traj bfgs_opt2.traj -o bfgs_opt.traj")
#
#     elif args.twostep:
#
#         atoms.calc = calc
#         opt = FIRE(atoms, trajectory = "bfgs_opt1.traj")
#
#         opt.run(fmax = 0.2)
#
#         print("DFT Relaxation with Hookean constraint finished")
#         # Optimization without Hookean Constraint
#         newcons = [constraint for constraint in atoms.constraints if str(constraint).startswith("FixAtoms")]
#         atoms.set_constraint(newcons)
#         print(atoms.constraints)
#
#         opt = FIRE(atoms, trajectory = "bfgs_opt2.traj")
#         opt.run(fmax = 0.05)
#         os.system("ase gui bfgs_opt1.traj bfgs_opt2.traj -o bfgs_opt.traj")
#
#     else:
#         atoms.calc = calc
#         opt = FIRE(atoms, trajectory = "bfgs_opt.traj")
#         opt.run(fmax = 0.05)


if args.DFTsinglepoint:

    filename, extension = os.path.splitext(args.DFTsinglepoint)
    if extension == ".pickle":
        with open(args.DFTsinglepoint, "rb") as h:
            atoms = pickle.load(h)
    else:
        structures = read(args.DFTsinglepoint, index=":")

        if len(structures) > 1:
            atoms = structures[-1]
        else:
            atoms = structures[0]

        calc = Aims(profile=profile,
                    **dft_params,
                    )
        calc.template.outputname = outfile
        atoms.calc = calc
        if os.path.exists(outfile):
            atoms = read_aims_output(outfile)

        atoms.get_potential_energy(force_consistent=True)

# elif args.minimahopping:
#
#
#     # Read in Structure
#     atoms = read("adsorption.traj", index="-1")
#
#     # Set Constraints
#     if file_params["using_opt_adsorbate"]:
#         apply_constraints(atoms, relax_metal=args.relaxmetal, method="file", constrain_Hookean=True)
#     else:
#         apply_constraints(atoms, relax_metal=args.relaxmetal, smiles=args.smiles, constrain_Hookean=True)
#
#     print(atoms.constraints)
#
#     # Set Calculator/ Potential
#     print("start reading potential file")
#     #pot_file = args.potential
#     pot = Potential(param_filename = args.potential)
#     atoms.calc = pot
#     print(f"Potential : {args.potential}")
#     print("potential file has been read!")
#
#     # Minima Hopping
#     hop = MinimaHopping(atoms, Ediff0=0.75, T0=2000., minima_traj="../minima.traj")
#     hop(totalsteps = 20)
#
#     mhplot = MHPlot(E0=True)
#     mhplot.save_figure('MinHop_summary.png')

