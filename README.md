# Machine-Learning Driven Global Optimization of Surface Adsorbate Geometries
authored by : Hyunwook Jung, Lena Sauerland, Sina Stocker, Karsten Reuter, Johannes T. Margraf

Fritz-Haber-Institut der Max-Planck-Gesellschaft Faradayweg 4-6 D-14195 Berlin Germany

# Overview
This workflow package aims for accelerating global optimization of surface adsorbate structure with active training of machine-learning potential where Gaussian Approximation Potential (GAP) is used herein. Surrogate GAP model is fitted on-the-fly starting from single structure and iterative training scheme automatically selects required training set structures and hyperparameters  

This workflow entails DFT calculation where FHI-aims is used in current implementation. Therefore this requires FHI-aims to be installed but extension to other software is also possible. In addition to DFT code, this utilize GAP within QUIP interface, parallel minima hopping using GNU parallel, RDkit, universal SOAP, DScribe. Therefore corresponding settings are required. 



# Settings:
1. scratchfolder : If you are using scratchfolder, you can specify scratch folder to use. otherwise, You can avoid using separate scratch folder by adding optional keyword argument `-ns` or `--noscratch` when executing `training_gap_iter_n.py`
2. DFT submission : DFT single-point or relaxation is carried out by executing `ase_submit.py` file. This file can be placed such as binary directory `/bin` to make it executalbe. Otherwise, user should modify the file location in `write_slurm.py` in `gaphelpers` folder. 
3. The number of CPUs and specific settings in SLURM submission file should be adapted depending on specific computational environment of user. 

# Dependency: 
- [igraph](https://igraph.org/)
- [GNU parallel](https://www.gnu.org/software/parallel)
- [RDKit](https://www.rdkit.org)
- [QUIP](https://github.com/libAtoms/QUIP)
- [universal SOAP](https://github.com/libAtoms/universalSOAP)
- [DScribe](https://singroup.github.io/dscribe/latest/)


# License:
This package is released under the MIT License.

# Example for usage

Added by Zhongwei Lu

Prepare relevant packages.
This is an example using anaconda on hawk.
```
module load anaconda/2020.02
conda create -n quip python=3.10.15 #Use python 3.10
source activate # Or eval "$(conda shell.bash hook)"
conda activate quip
```

Regular venv should also work (Example on Falcon)
```
module load Python/3.10.4-GCCcore-11.3.0
python3 -m venv quip
```


```
pip install ase==3.25.0
pip install rdkit
pip install igraph
pip install universalSOAP
pip install dscribe
pip install pandas
pip install pyyaml
pip install rdkit2ase
pip install wfl
```
Numpy 1.x is needed at the time of writing.

Remember to change the path to the length scale files, which is in the `universalSOAP` package, in `gaphelpers/gap_command.py`, the `universal_soap_yaml` parameter.

Install quip from pip
```
pip install quippy-ase
```

Or compile from source. Sequence for loading modules seems to matter.
```
module purge
module load anaconda/2020.02 compiler/intel/2020/2 mkl/2020/2 compiler/gnu/8/1.0 
source activate # Or eval "$(conda shell.bash hook)"
conda activate quip

git clone --recursive https://github.com/libAtoms/QUIP.git
export QUIP_ARCH=linux_x86_64_ifort_icc_serial
make config
#Linking options -L/apps/compilers/intel/2020.2/compilers_and_libraries/linux/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core

make

make install-quippy
```


These lines in `quippy/potential.py` are commented
```
    #def _get_name(self):
    #    return self.name_

    #@property
    #def name(self):
    #    return self._get_name()

    #@name.setter
    #def name(self, name):
    #    self.name_ = name
```

Version of `f90wrap` needs to be consistent with version of `quippy-ase`. For `quippy-ase` 0.10.x you need `f90wrap` 0.3.x, for `quippy-ase 0.9.x` you need `f90wrap` 0.2.x.

## Important scripts
`gaphelpers/write_slurm.py`: manage submission script for Slurm. Change `module load` commands according to the supercomputer environment

`training_gap_iter_n.py`: manage GAP training and sampling of GAP minima. You can change scratch folder.

`ase_submit.py`: manage settings for DFT calculations. Change the path to your aims binary, the aims command, and species directory accordingly.

`gaphelpers/gap_command.py`: manage commands for gap fitting, e.g. `sparse_method` can be set here.

`gaphelpers/minimahopping.py`: change waiting time limit for DFT calculation including queuing time in `check_slurm_completion`.  

## Example bash script for runing the workflow

This is performed on a scratch(or work) folder rather than home.
```
#Runing gap sampling of adsorption complex

module purge
module load anaconda/2020.02
source activate # Or eval "$(conda shell.bash hook)"
conda activate quip

export PYTHONPATH=/home/$USER/gap_workflow_surface:$PYTHONPATH
export GAPWF=/home/$USER/gap_workflow_surface


python3 $GAPWF/gaphelpers/write_slurm.py --iteration 0 --enditeration 80 -mh C#C[O] --facet 100 -slab /scratch/$USER/gapTest/slab-Nb.traj -ads /scratch/$USER/gapTest/ch3co.traj -gap $GAPWF -submit 
```
Here we are using trajectory files as input. The smiles string here is just to prevent `missing positonal argument` error.



## Use mpi for gap\_fit (On-going work)

Example for hawk

Create a venv

```
module load compiler/gnu/9/2.0 compiler/intel/2018/2 mpi/openmpi/3.1.5 mkl/2018/2 python/3.9.2

python3 -m venv mpi-quip
. mpi-quip/bin/activate

pip install --upgrade pip
pip install numpy==1.23.4
pip install ase==3.25.0

```

Compile the mpi-version of QUIP and quippy. More info at https://github.com/libAtoms/QUIP.

```
git clone --recursive https://github.com/libAtoms/QUIP.git
```

Actual compilation
```
export QUIP_ARCH=linux_x86_64_ifort_gcc_openmpi

make config
#Linking options -L/apps/compilers/intel/2018.2/compilers_and_libraries/linux/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lmkl_blacs_intelmpi_lp64 -lmkl_scalapack_lp64
#MPI: To use the MPI parallelisation of gap_fit, you have to add your system library to the linking options, e.g. -lscalapack or -lscalapack-openmpi, enable GAP support, enable QR decomposition, and enable ScaLAPACK.

make

make install-quippy
```

Failed Compilation
```
module load compiler/intel/2020/2  mpi/intel/2020/2 mkl/2020/2
module load python/3.9.2

python3 -m venv quip-mpi
. quip-mpi/bin/activate

pip install --upgrade pip
pip install numpy==1.23.4
pip install ase==3.25.0
```



```
export QUIP_ARCH=linux_x86_64_ifort_icc_mpi

make config
#Linking options -L/opt/intel/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lmkl_blacs_intelmpi_lp64 -lmkl_scalapack_lp64
#MPI: To use the MPI parallelisation of gap_fit, you have to add your system library to the linking options, e.g. -lscalapack or -lscalapack-openmpi, enable GAP support, enable QR decomposition, and enable ScaLAPACK.

make

make install-quippy
```
Compilation failed at:
```
./src.linux-x86_64-3.9/./src.linux-x86_64-3.9/fortranobject.c(707): error: expected an expression
      for (int i = 0; i < rank; ++i) {
           ^

./src.linux-x86_64-3.9/./src.linux-x86_64-3.9/fortranobject.c(707): error: identifier "i" is undefined
      for (int i = 0; i < rank; ++i) {
                      ^

compilation aborted for ./src.linux-x86_64-3.9/./src.linux-x86_64-3.9/fortranobject.c (code 2)
```

