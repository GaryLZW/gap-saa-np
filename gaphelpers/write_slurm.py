import pandas as pd
import argparse, os
from pathlib import Path
from get_argparse import get_argparse
from ase.io import read


def get_adsorbate_info(mol=None, 
                    mol_json= str(Path(__file__).parent.resolve()) + "/n3_mol_mod.json",
                       using_opt_adsorbate=False, adsorbate_file=None,):

    mdf = pd.read_json(mol_json)
    if using_opt_adsorbate:
        ads = read(adsorbate_file)
        idx = 666
        formula = ads.get_chemical_formula(mode='reduce',)
        smiles = 'dull'
    elif isinstance(mol, int):
        idx = mol
        formula = mdf.loc[mol].formula
        smiles = mdf.loc[mol].smiles


    elif mol.isnumeric():
        mol = int(mol)
        idx = mol
        formula = mdf.loc[mol].formula
        smiles = mdf.loc[mol].smiles

    elif isinstance(mol, str):
        idx = mdf.loc[mdf.smiles == mol].index.item()
        formula = mdf.loc[mdf.smiles == mol].formula.item()
        smiles = mol

    return idx, formula, smiles


def write_slurm(
            end_iter,
            parent_metal,
            dopant,
            size,
            one_out_of_n,
            iteration=0,
            forcemask=False,
            no_universal_soap=True,
            ncpus = 1,
            slurm_file_name = "submit.sh",
            mol_json = str(Path(__file__).parent.resolve()) + "/n3_mol_mod.json",
            code = "aims",
            path_to_workflow=None,
            ):
    """
    This function write slurm submit file for Global optimization
    Default file name is submit.sh

    Parameters :
    ------------
    end_iter : int
        Maximum iteration for GAP fitting. (Usually less than 30 is more than enough in our study)
    mol : int or str
        Either integer if this is already included in dataset in JSON file, else SMILES string is required.
    facet : string
        Surface facet for surface either 111 or 211 (for now)
    iteration : int
        Iteration number
    forcemask : bool
        whether to use force mask for lowest three metal layer.
    multiple_universal_soap : bool
        whether to use multiple SOAP as give by universal SOAP
    ncpus : interger
        Number of cpu to be used when GAP fitting. Triggers multithreaded fitting.
    slurm_file_name : str
        name for submission bash file
    mol_json : str
        path for molecular dataset JSON file.
        131 molecules of [CHO]3 from 'ACS Omega 2019, 4, 3370−3379' were tested for this work
        Information on these molecules were gathered for simplicity.
        However, presence of this JSON file is not mandatory.
    using_opt_adsorbate : bool 
    adsorbate_file : str
    using_opt_surface : bool
    surface_file : str
    path_to_workflow : str
    """


    # if using_opt_adsorbate:
    #     idx, formula, smiles = get_adsorbate_info(using_opt_adsorbate=using_opt_adsorbate, adsorbate_file=adsorbate_file)
    # else:
    #     idx, formula, smiles = get_adsorbate_info(mol)

    universal_soap = "-mus " if no_universal_soap else ""
    forcemask = "-fm " if forcemask else ""
    code = "-qe" if code == "qe" else ""
    mem = 24000
    ncpus = 1

    slurm_str = f"""#!/bin/bash -l
#SBATCH -o ./tjob.out.%j
#SBATCH -e ./tjob.err.%j
#SBATCH -D ./
#SBATCH -J GAP_{parent_metal}{dopant}_wulff
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={ncpus}	
#SBATCH --mem={mem}
#SBATCH --time=48:00:00
"""

    slurm_str += f"""

module purge

module load Python/3.10.4-GCCcore-11.3.0

. /shared/home1/$USER/quip/bin/activate


#module load anaconda/2020.02

export PYTHONPATH={path_to_workflow}:$PYTHONPATH
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
ulimit -s unlimited 
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
#eval "$(conda shell.bash hook)"
#conda activate quip

"""

    slurm_str += f"""
python3 {path_to_workflow}/training_gap_iter_n.py -i {iteration} -e {end_iter} -parent {parent_metal} -dop {dopant} -size {size} -conc {one_out_of_n} -gap {path_to_workflow} {universal_soap}{forcemask} {code}  | tee -a stdout.log 
"""

    with open(slurm_file_name, "w") as f:
        f.write(slurm_str)


def write_dft_slurm(jobname, 
                    target_file_name="structure.traj",
                    slurm_file_name="submit.sh",
                    forcemask=False,
                    code="aims",
                    path_to_workflow=None,
                    time="02:00:00",
                    nodes=2,
                    cores=384,
                    mem="200G",
):
    forcemask = "-fm" if forcemask else ""
    code_name = "-qe" if code == "qe" else ""

    if code == "qe":
        add_string = 'export ASE_ESPRESSO_COMMAND="srun /u/hjung/Softwares/QE/qe-7.0/bin/pw.x -in PREFIX.pwi > PREFIX.pwo"\n'
#		add_string += 'export ESPRESSO_PSEUDO="/u/hjung/Softwares/QE/qe-7.0/pseudo/"'
    else:
        add_string = ""

    slurm_str = f"""#!/bin/bash -l
#SBATCH -o ./tjob.out.%j
#SBATCH -e ./tjob.err.%j
#SBATCH -D ./
#SBATCH -J {jobname}
#SBATCH --nodes={nodes}
#SBATCH --ntasks={cores}
#SBATCH --ntasks-per-core=1
#SBATCH --mem={mem}
#SBATCH --time={time}

module purge

module load Python/3.10.4-GCCcore-11.3.0
. /shared/home1/$USER/quip/bin/activate

module load intel/2025b

export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
ulimit -s unlimited
export OMP_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export MKL_NUM_THREADS=1

{add_string}

export PYTHONPATH={path_to_workflow}:$PYTHONPATH

##p_job=$(pwd -L | cut -d "/" -f 3-)
#p_job=$(pwd -P)
#randdir=`tr -dc A-Za-z0-9 </dev/urandom | head -c 13 ; echo ''`
#p_scratch="`echo $p_job |cut -d '/' -f 4-`/$randdir/"
#p_scratch="$p_job/$randdir/"

#mkdir -p $p_scratch

#cp * $p_scratch
#cd $p_scratch

python3 {path_to_workflow}/ase_submit.py -sp {target_file_name}

# custom post-command stuff
#cp -r *.traj espresso.* *.log $p_job
#cd $p_job
"""
    with open(slurm_file_name, "w") as f:
        f.write(slurm_str)


if __name__ == "__main__":
    args = get_argparse()

#	numdir = len(next(os.walk('.'))[1])
    f_name = f"saa_{args.dopant}{args.parent}_wulff"
    Path(f_name).mkdir(parents=True, exist_ok=True)
    if os.path.exists("restart_training_data.xyz"):
        os.system(f"cp restart_training_data.xyz {f_name}")
    os.chdir(f_name)
    
    write_slurm(
            iteration = args.iteration,
            end_iter = args.enditeration,
            parent_metal=args.parent,
            dopant=args.dopant,
            size=args.size,
            one_out_of_n=args.one_out_of_n,
            path_to_workflow=args.gaphelper,
            )	
    if args.submit:
        os.system("sbatch submit.sh")

