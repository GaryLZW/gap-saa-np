import argparse

# argparse
def get_argparse():

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument(
        "-i",
        '--iteration',
        default=0,
        type=int,
        help='Number of training iteration.'
    )
    parser.add_argument(
        "-e",
        '--enditeration',
        type=int,
        help='Terminating iteration point'
    )
    parser.add_argument(
        "-mol_index",
        '--mol_index',
        type=int,
        help='Mol index'
    )
    parser.add_argument(
            "-mh",
            "--minimahopping",
            nargs="?",
            type=str,
            help="Give SMILES string for constraints in minimahopping. Default filename: adsorption.traj"
    )
    parser.add_argument(
            "-rm",
            "--relaxmetal",
            action="store_true",
            help="Relax upper half of metal slab. If not used, all metal slab is constrained"
            )
    parser.add_argument(
            "-f",
            "--facet",
            #choices=["111","211"],
            type=str,
            help="Surface facet for adsorption. For now either 111 or 211 for Rh. Just a notation if using file input."
    )
    parser.add_argument(
            "-fm",
            "--force_mask",
            action="store_true",
            help="Mask force of lower half atom layers"
    )
    parser.add_argument(
            "-ns",
            "--noscratch",
            action="store_true",
            help="Don't use scratch folder and write on submitting directory"
    )
    parser.add_argument(
            "-mus",
            "--multiple_universal_soap",
            action="store_false",
            help="If option used, multiple Universal SOAP is used. Default: Single universal SOAP"
    )
    parser.add_argument(
            "-para",
            "--parallel",
            action="store_true",
            help="Test: parallel mode for GAP training and minima hopping"
    )
    parser.add_argument(
            "-sampling",
            "--sampling",
            nargs="?",
            default="stratified_random",
            const="staratified_random",
            choices=["stratified_random", "full_fps", "stratified_fps", "full_random"],
            help="Choose sampling strategy for training set structures"
    )
    parser.add_argument(
            "-calc_all_minima",
            "--calc_all_minima",
            action="store_true",
            help="If used, single point calculations for all minima structure found in minima hopping are submitted"
    )
    parser.add_argument(
            "-qe",
            "--quantum_espresso",
            action="store_true",
            help="Use QE+BEEF-vdw for DFT calculation."
    )

    parser.add_argument(
            "-conv",
            "--convergence",
            nargs="?",
            default="very_tight",
            const="very_tight",
            choices=["light", "tight", "very_tight"],
            help="GAP convergence : E_EMA<8 mev/atom AND F_EMA<0.15ev/AA (very tight)"
    )
    parser.add_argument(
            "-submit",
            "--submit",
            action="store_true",
            help="Submit job"
    )
    
    parser.add_argument(
            "-parent",
            "--parent",
            type=str,
            default="Cu",
            help="Name of parent metal"
    )
    
    parser.add_argument(
            "-dop",
            "--dopant",
            type=str,
            default="Ru",
            help="Name of dopant."
    )
    parser.add_argument(
            "-gap",
            "--gaphelper",
            type=str,
            help="Path to gaphelper code"
    )

    args = parser.parse_args()

    return args


def get_minhop_args():

    parser = argparse.ArgumentParser(description = "Arguments for Minima hopping")
    parser.add_argument(
        "-i",
        '--iteration',
        default=0,
        type=int,
        help='Number of training iteration.'
    )
    parser.add_argument(
            "-mh",
            "--minimahopping",
            nargs="?",
            type=str,
            help="Give SMILES string for constraints in minimahopping. Default filename: adsorption.traj"
    )
    parser.add_argument(
            "-rm",
            "--relaxmetal",
            action="store_true",
            help="Relax upper two metal layers. If not used, all metal slab is constrained"
            )
    parser.add_argument(
            "-p",
            "--potential",
            nargs="?",
            type=str,
            const="GAP_0.xml",
            help="filename of GAP potential"
            )
    parser.add_argument(
            "-hops",
            "--hops",
            nargs="?",
            type=int,
            const=20,
            default=20,
            help="Number of hops"
            )
    parser.add_argument(
            "-f",
            "--facet",
            #choices=["111","211"],
            type=str,
            help="Surface facet for adsorption. For now either 111 or 211 for Rh. "
    )
    parser.add_argument(
            "-tmp",
            "--tmpdir",
            type=str,
            help="Surface facet for adsorption. For now either 111 or 211 for Rh. "
    )
    parser.add_argument(
            "-para",
            "--parallel",
            action="store_true",
            help="Parallelized Minima hopping. Shares minima list: ../minima.traj"
    )
    parser.add_argument(
            "-v",
            "--volatilation",
            action="store_true",
            help="Impose Hookean constraint to prevent volatilation"
    )
    parser.add_argument(
            "-mt",
            "--maxtemp_scale",
            default = 2.0,
            type=float,
            help="Max temperature for Minima hopping; default=2.0 (e.g. maxtemp = maxtemp_scale*T0)"
    )
    parser.add_argument(
            "-rs",
            "--randseed",
            default = 0,
            type=int,
            help="Random seed for initial structure generation"
    )
    parser.add_argument(
            "-slab",
            "--slab",
            type=str,
            help="Path to pre-optimized suface model"
    )
    
    parser.add_argument(
            "-ads",
            "--adsorbate",
            type=str,
            help="Path to pre-optimized adsorbate geometry"
    )

    args = parser.parse_args()

    return args

def get_encode_argparse():

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument(
        "-i",
        '--iteration',
        default=0,
        type=int,
        help='Number of training iteration.'
    )
    parser.add_argument(
        "-pas",
        '--peratomsigma',
        action="store_true",
        help='Do not use per_atom_sigma for force components'
    )
    parser.add_argument(
        "-fm",
        '--forcemask',
        action="store_true",
        help='Mask force elements for lower three metal layers'
    )
    parser.add_argument(
        "-c",
        '--correct',
        action="store_true",
        help='Correction function'
    )
    args = parser.parse_args()

    return args


