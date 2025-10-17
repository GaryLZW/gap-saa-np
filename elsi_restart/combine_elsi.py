#!/usr/bin/env python

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from elsi_restart.read_elsi import read_elsi_to_csc, get_n_electrons_from_elsi, write_csc_to_elsi
import scipy.sparse as sp
import fnmatch


def combine_two_dm(dm1, dm2):
    
    nnz_value = np.concatenate((dm1.data, dm2.data))
    
    indptr2 = dm2.indptr + dm1.indptr[-1]
    indptr1 = np.delete(dm1.indptr, -1)
    col_ptr = np.concatenate((indptr1, indptr2))
    
    row_index = np.concatenate((dm1.indices, (dm2.indices + dm1.shape[1])))
    
    n_basis = dm1.shape[1] + dm2.shape[1]
    #print('nnz', len(nnz_value), ' size', dm1.size, ' ',dm2.size)
    #print(len(col_ptr), ' last column pointer', col_ptr[-1] )
    #print('n_basis', n_basis, ' row_ind', len(row_index))
    return sp.csc_matrix((nnz_value, row_index, col_ptr), shape=(n_basis, n_basis))


def combine_slab_ads_dm(path_to_slab=None, n_spin=1):
    cwd = os.getcwd()
    path_to_ads = "."  #'/scratch/c.c22015584/ch3co-dm/100'

    n_kpts = int(len(fnmatch.filter(os.listdir(cwd), '*.csc')) / n_spin)
    first_dm = 'D_spin_01_kpt_000001.csc'
    n_electron = get_n_electrons_from_elsi(path_to_ads+'/'+first_dm) + get_n_electrons_from_elsi(path_to_slab+'/'+first_dm)

    for iS in range(1, n_spin+1):
        for iK in range(1, n_kpts+1):
            csc_filename = f'D_spin_{iS:0>2}_kpt_{iK:0>6}.csc'
            # Read into csc
            csc1 = read_elsi_to_csc(path_to_slab + '/' + csc_filename)
            csc2 = read_elsi_to_csc(path_to_ads + '/' + csc_filename)

            csc_combined = combine_two_dm(csc1, csc2)
            write_csc_to_elsi(csc_combined, csc_filename, n_electron)



'''
# test write_function
path_1 = './elsi_write/'
n_spin = 2
n_kpts = 26
for iS in range(1, n_spin+1):
    for iK in range(1, n_kpts+1):
        test_file = f'D_spin_{iS:0>2}_kpt_{iK:0>6}.csc'
        csc = read_elsi_to_csc(path_1+test_file)
        write_csc_to_elsi(csc, test_file, get_n_electrons_from_elsi(path_1+test_file))
'''
