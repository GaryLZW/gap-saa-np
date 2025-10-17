def get_k_grid(model, sampling_density, verbose=False, simple_reciprocal_space_parameters=True):
    import math
    import numpy as np

    dimensions = sum(model.pbc)

    if dimensions == 0:
        print("You are studying a molecular system. This has no periodicity, and therefore no k-grid is necessary. "
              "Returning null")
        return None

    # These are lattice vectors
    lattice_v = np.array(model.get_cell())
    # These are lattice parameters
    lattice_param = np.array([np.linalg.norm(v) for v in lattice_v])

    # Check if the model is periodic and with vacuum along a certain axis. There could be vacuum if
    # the lattice parameter is 5 angstrom longer than the range of atomic positions along an axis,
    check_vacuum_and_periodic = np.array((lattice_param - np.ptp(model.get_positions(), axis=0)) > 5) & model.pbc
    if sum(check_vacuum_and_periodic):
        print("There could be vacuum in these axes", np.array(['x', 'y', 'z'])[check_vacuum_and_periodic],
              ", but they are also periodic."
              "\nIf you don't want the model to be treated as periodic in these dimensions,",
              "set pbc for these axes to false, or check if k point sampling is actually 1 in these dimensions")

    if simple_reciprocal_space_parameters:
        # Simplified reciprocal lattice parameters
        reciprocal_param = 2 * math.pi / lattice_param
    else:
        # volume of the cell
        volume = np.dot(lattice_v[0], np.cross(lattice_v[1], lattice_v[2]))
        # These are reciprocal lattice vectors.
        # For definition, see section 2.4 of https://www.physics-in-a-nutshell.com/article/15/the-reciprocal-lattice
        reciprocal_v = [np.cross(lattice_v[(i + 1) % 3], lattice_v[(i + 2) % 3]) * 2 * math.pi / volume
                        for i in range(len(lattice_v))]
        # These are reciprocal lattice parameters
        reciprocal_param = np.array([np.linalg.norm(r_v) for r_v in reciprocal_v])

    k_grid_density = 1 / (sampling_density * 2 * math.pi)
    k_grid = k_grid_density * reciprocal_param
    # Convert k_grid to integer
    k_grid = np.array([math.ceil(k) for k in k_grid])
    # Remove k-sampling if direction is not periodic in any dimension
    k_grid[np.invert(model.pbc)] = 1

    if verbose:
        print("Based on lattice xyz dimensions", "x", round(lattice_param[0], 3), "y", round(lattice_param[1], 3),
              "z", round(lattice_param[2], 3))
        print("and", "one k-point per", str(sampling_density), "* 2π Å^-1",
              "sampling density, the k-grid chosen for periodic calculation is",
              str(k_grid) + ".")
        if not simple_reciprocal_space_parameters:
            print("Please note you are using the strict definition of reciprocal lattice vector here. "
                  "This would generate a slightly denser k-grid than using simple reciprocal space parameters in "
                  "cases where a non-orthogonal cell is used as input.")

    return tuple(k_grid)
