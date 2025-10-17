#!/usr/bin/env python
# This file is from FHI-aims utilities.elsi_matrix
# Example python code for reading an ELSI matrix file and how to handle the resulting object further.
import struct
import numpy as np
import scipy.sparse
import scipy.sparse as sp


def read_elsi_to_csc(filename):
    mat = open(filename, "rb")
    data = mat.read()
    mat.close()
    i8 = "l"
    i4 = "i"

    # Get header
    start = 0
    end = 128
    header = struct.unpack(i8 * 16, data[start:end])
    #print(header)

    # Number of basis functions (matrix size)
    n_basis = header[3]

    # Total number of non-zero elements
    nnz = header[5]

    # Get column pointer
    start = end
    end = start + n_basis * 8
    col_ptr = struct.unpack(i8 * n_basis, data[start:end])
    col_ptr += (nnz + 1,)
    col_ptr = np.array(col_ptr)

    # Get row index
    start = end
    end = start + nnz * 4
    row_idx = struct.unpack(i4 * nnz, data[start:end])
    row_idx = np.array(row_idx)

    # Get non-zero value
    start = end

    if header[2] == 0:
        # Real case
        end = start + nnz * 8
        nnz_val = struct.unpack("d" * nnz, data[start:end])
    else:
        # Complex case
        end = start + nnz * 16
        nnz_val = struct.unpack("d" * nnz * 2, data[start:end])
        nnz_val_real = np.array(nnz_val[0::2])
        nnz_val_imag = np.array(nnz_val[1::2])
        nnz_val = nnz_val_real + 1j * nnz_val_imag

    nnz_val = np.array(nnz_val)

    # Change convention
    for i_val in range(nnz):
        row_idx[i_val] -= 1

    for i_col in range(n_basis + 1):
        col_ptr[i_col] -= 1

    return sp.csc_matrix((nnz_val, row_idx, col_ptr), shape=(n_basis, n_basis))


def write_csc_to_elsi(csc: scipy.sparse.csc_matrix, filename, n_electrons=0):
    mat = open(filename, "wb")
    i8 = "l"
    i4 = "i"

    if type(csc.data[0]) is np.complex128:
        is_a_complex_m = 1
    else:
        is_a_complex_m = 0

    n_basis = csc.shape[0]
    nnz = csc.size
    #print('size:', csc.size, ' nnz:', nnz)

    # Write header
    header = np.zeros(16, dtype=np.int_)
    header[0] = 170915  # FILE_VERSION in ELSI_CONSTANT
    header[1] = -910910  # UNSET in ELSI_CONSTANT
    header[2] = is_a_complex_m
    header[3] = n_basis
    header[4] = n_electrons
    header[5] = nnz
    header[6] = -910910
    header[7] = -910910
    header[8:] = -910910
    #start = 0
    #end = 128
    #struct.pack_into(i8 * 16, mat, start, *header)
    b_header = struct.pack(i8 * 16, *header)

    # Pack column pointer
    #start = end
    #end = start + n_basis * 8
    col_ptr = csc.indptr + 1  # change convention
    col_ptr = np.delete(col_ptr, -1)
    #struct.pack_into(i8 * n_basis, mat, start, *col_ptr)
    b_col_ptr = struct.pack(i8 * n_basis, *col_ptr)

    # Pack row index
    #start = end
    #end = start + nnz * 4
    row_idx = csc.indices + 1  # change convention
    #struct.pack(i4 * nnz, mat, start, *row_idx)
    b_row_idx = struct.pack(i4 * nnz, *row_idx)

    # Pack non-zero values
    #start = end

    if is_a_complex_m:
        # end = start + nnz*16
        nnz_value = np.zeros(nnz * 2)
        nnz_value_real = np.real(csc.data)
        nnz_value_imaginary = np.imag(csc.data)
        pointer_to_real = [True, False] * nnz
        nnz_value[pointer_to_real] = nnz_value_real
        nnz_value[np.invert(pointer_to_real)] = nnz_value_imaginary
        #struct.pack("d" * nnz * 2, mat, start, *nnz_value)
        b_nnz_value = struct.pack("d" * nnz * 2, *nnz_value)

    else:
        # end = start + nnz * 8
        nnz_value = csc.data
        #struct.pack_into("d" * nnz, mat, start, *nnz_value)
        b_nnz_value = struct.pack("d" * nnz, *nnz_value)

    b_data = b_header + b_col_ptr + b_row_idx + b_nnz_value
    mat.write(b_data)

    mat.close()


def get_n_electrons_from_elsi(filename):
    mat = open(filename, "rb")
    data = mat.read(128)
    mat.close()
    i8 = "l"

    # Get header
    start = 0
    end = 128
    header = struct.unpack(i8 * 16, data[start:end])

    return header[4]
