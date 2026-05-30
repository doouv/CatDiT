"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import warnings
from collections import Counter

import numpy as np
from matminer.featurizers.composition.composite import ElementProperty
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

warnings.filterwarnings("ignore", category=UserWarning)  # raised when converting to Crystal object


CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
CompFP = ElementProperty.from_preset("magpie")


class Catalyst:
    """Catalyst object that holds information about a structure and its validity, including
    PyMatGen Structure object, composition, and fingerprints, modified based on Crystal object.

    Adapted from: https://github.com/txie-93/cdvae
    """

    def __init__(self, cat_array_dict):
        self.frac_coords = cat_array_dict["frac_coords"]
        self.atom_types = cat_array_dict["atom_types"]
        self.tags = cat_array_dict["tags"]
        self.sample_idx = cat_array_dict["sample_idx"]
        self.lengths = cat_array_dict["lengths"].squeeze()
        assert self.lengths.ndim == 1
        self.angles = cat_array_dict["angles"].squeeze()
        assert self.lengths.ndim == 1
        self.dict = cat_array_dict

        self.get_structure()
        if self.constructed:
            self.get_composition()
            self.get_validity()
            # self.get_symmetry()
            self.get_fingerprints()
        else:
            self.struct_valid = False

    def get_structure(self):
        if min(self.lengths) < 0:
            self.constructed = False
            self.invalid_reason = "non_positive_lattice"
        if (
            np.isnan(self.lengths).any()
            or np.isnan(self.angles).any()
            or np.isnan(self.frac_coords).any()
        ):
            self.constructed = False
            self.invalid_reason = "nan_value"
        # this catches validity failures down the line
        elif (1 > self.atom_types).any() or (self.atom_types > 104).any():
            self.constructed = False
            self.invalid_reason = f"{self.atom_types=} are not with range"
        else:
            try:
                self.structure = Structure(
                    lattice=Lattice.from_parameters(
                        *(self.lengths.tolist() + self.angles.tolist())
                    ),
                    species=self.atom_types,
                    coords=self.frac_coords,
                    coords_are_cartesian=False,
                )
                self.structure.add_site_property("tags", self.tags)
                self.constructed = True
                if self.structure.volume < 0.1:
                    self.constructed = False
                    self.invalid_reason = "unrealistically_small_lattice"
            except TypeError:
                self.constructed = False
                self.invalid_reason = f"{self.atom_types=} are not possible"
            except Exception:
                self.constructed = False
                self.invalid_reason = "construction_raises_exception"

    def get_composition(self):
        bulk_atoms = self.atom_types[self.tags <=1 ]
        elem_counter = Counter(bulk_atoms)
        composition = [(elem, elem_counter[elem]) for elem in sorted(elem_counter.keys())]
        elems, counts = list(zip(*composition))
        counts = np.array(counts)
        counts = counts / np.gcd.reduce(counts)
        self.elems = elems
        self.comps = tuple(counts.astype("int").tolist())

    # def check_vacuum_layer(self):
    #     z_positions = self.structure.cart_coords[:, 2]
    #     cell_height = self.lengths[2]
    #     slab_thickness = z_positions.max() - z_positions.min()
    #     self.vacuum_thickness = cell_height - slab_thickness
    #     self.has_vacuum = self.vacuum_thickness >= 10.0

    def get_validity(self):
        if self.constructed:
            self.struct_valid = structure_validity(self.structure)
        else:
            self.struct_valid = False

    # def get_symmetry(self):
    #     """
    #     Adapted from: https://github.com/materialsproject/pymatgen/blob/v2025.5.28/src/pymatgen/core/surface.py#L283-L304
    #     """
    #     sga = SpacegroupAnalyzer(self.structure, symprec=0.1)
    #     symm_ops = sga.get_point_group_operations()
    #     self.symmetry = (
    #         sga.is_laue()
    #         or any(op.translation_vector[2] != 0 for op in symm_ops)
    #         or any(np.all(op.rotation_matrix[2] == np.array([0, 0, -1])) for op in symm_ops)
    #     )

    def get_fingerprints(self):
        elem_counter = Counter(self.atom_types)
        comp = Composition(elem_counter)
        self.comp_fp = CompFP.featurize(comp)
        try:
            site_fps = [
                CrystalNNFP.featurize(self.structure, i) for i in range(len(self.structure))
            ]
        except Exception:
            # counts crystal as invalid if fingerprint cannot be constructed.
            self.struct_valid = False
            self.comp_fp = None
            self.struct_fp = None
            return
        self.struct_fp = np.array(site_fps).mean(axis=0)

    def __repr__(self):
        return f"Catalyst (struct_valid={self.struct_valid}, atoms={self.atom_types}, lengths={self.lengths}, angles={self.angles}, idx={self.sample_idx})"




def structure_validity(catalyst, cutoff=0.5):
    dist_mat = catalyst.distance_matrix
    # Pad diagonal with a large number
    dist_mat = dist_mat + np.diag(np.ones(dist_mat.shape[0]) * (cutoff + 10.0))
    if dist_mat.min() < cutoff or catalyst.volume < 0.1:
        return False
    else:
        return True


chemical_symbols = [
    # 0
    "X",
    # 1
    "H", "He",
    # 2
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    # 3
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    # 4
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    # 5
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    # 6
    "Cs", "Ba",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
    # 7
    "Fr", "Ra",
    "Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md", "No","Lr",
    "Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og",
]