#!/bin/bash -l
 
# stderr and stdout
#SBATCH -D ./
#SBATCH -J Minhop_M6_CCO
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --ntasks-per-core=1
#SBATCH --time=6:00:00
#SBATCH --mem=180000M
#SBATCH -o ./tjob.out.%j
#SBATCH -e ./tjob.err.%j
 
 
echo "#--- Job started at `date`"
 
# unloading modules
module purge
# loading user modules
#module load parallel/20190122
#module load mkl/2020/2
#module load compiler/intel/2020/0
#module load anaconda/2020.02
#module load python/3.10.4
module load Python/3.10.4-GCCcore-11.3.0
module load parallel/20250722-GCCcore-14.3.0

# this one echos each command (only after module scripts!)
#ulimit -u 10000

# environment variables
#eval "$(conda shell.bash hook)"
#conda activate quip
. /shared/home1/$USER/quip/bin/activate
export PYTHONPATH=/shared/home1/$USER/gap-saa-np:$PYTHONPATH
# the actual binary that is run
 
# custom pre-command stuff
#p_job=`echo "$PWD" |sed 's/^\/cobra//'`
#randdir=`tr -dc A-Za-z0-9 </dev/urandom | head -c 13 ; echo ''`
#p_scratch="./myGapScratch/`echo $p_job |cut -d '/' -f 3-`/$randdir/"
#mkdir -p $p_scratch
 
#cp -r * $p_scratch
#cd $p_scratch

 
# run, Forest, run,...
parallel -X --delay 0.2 --joblog task.log --progress --resume -j $SLURM_NTASKS < cmd.lst
 
 
# custom post-command stuff
#cp -r * $p_job
#cd $p_job
 
# remove the requested files
# nothing to clean
 
echo "#--- Job ended at `date`"

