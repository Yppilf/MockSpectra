from molmass import Formula # type: ignore
import prodimopy.hitran as ht # type: ignore
import prodimopy.run_slab as runs # type: ignore
import pandas as pd
from typing import List, Optional
import os, pickle

pd.set_option('future.no_silent_downcasting', True)

class Molecule:
    def __init__(self, moleculeName: str, isotopes: List[int] = [1]) -> None:
        """Generates molecule data used for generating slab models.

        Parameters:
        moleculeName    (str)        (required)  Molecular formula of the desired molecule.
        isotopes        (list[int])  (optional)  List of isotopes to be included. Default=[1].

        Returns:
        None"""
        self.molecule: str = moleculeName
        self.isotopologue: List[int] = isotopes
        self.filepath: str = f"data/hitran/{moleculeName}.par"
        self.QTpath: str = "data/QTpy/"

        f = Formula(moleculeName)
        self.mol_mass: float = f.mass

    def get_moldata(self, lower: float, upper: float) -> None:
        """Reads the HITRAN file containing all the emission lines of the given molecule.

        Parameters:
        lower   (float)  (required)  The lower wavelength limit in microns.
        upper   (float)  (required)  The upper wavelength limit in microns.

        Returns:
        None"""
        self.mol_data = ht.read_hitran(self.filepath,
            self.molecule,
            self.isotopologue,
            lowerLam=lower, higherLam=upper)
        
        include = ['lambda','A','E_u','E_l','global_u','global_l','local_u','local_l','g_u','g_l']  # Include only these columns, rest are irrelevant for this project
        self.mol_data[include]

    def generateSlab(
        self, column_density: float, temperature: float, lower_wavelength: float, upper_wavelength: float, cache_folder: Optional[str] = None
    ) -> None:
        """Generates a 0D slab model using prodimopy for the given molecule at various parameters.

        Parameters:
        column_density      (float)  (required)  The gas column density in molecules per cm^2.
        temperature         (float)  (required)  Temperature in Kelvin of the molecule.
        lower_wavelength    (float)  (required)  The lower wavelength from where to run the slab model.
        upper_wavelength    (float)  (required)  The upper wavelength from where to run the slab model.
        cache_folder        (string) (optional)  The folder in which generated slabs are cached

        Returns:
        None"""

        if cache_folder and self.load_from_cache(cache_folder, column_density, temperature, lower_wavelength, upper_wavelength):
            return
        
        self.data = runs.run_0D_slab(Ng    = column_density,            # The gas column density in molecules per cm2
                                Tg         = temperature,               # The gas temperature
                                vturb      = 2.0,                       # Delta nu_D or the width parameter in km/s
                                molecule   = self.molecule,             # name of the molecule
                                mol_mass   = self.mol_mass,             # molecular mass in amu
                                HITRANfile = self.filepath,             # Path of the HITRAN data file
                                QTpath     = self.QTpath,               # path of the partition sums
                                isotopolog = self.isotopologue,         # list of isotopologues to include      
                                wave_mol   = [lower_wavelength,upper_wavelength],         # wavelength region in which the lines are to be picked up
                                mode       = 'line_by_line',            # "line-by-line" calculation, or include mutual "overlap"
                                output     = 'return',                  # "return" data or write to "file" or do "both"
                                )
        
        if cache_folder:
            self.save_to_cache(cache_folder, column_density, temperature, lower_wavelength, upper_wavelength)

    def get_cache_filename(self, cache_folder: str, column_density: float, temperature: float, lower_wavelength: float, upper_wavelength: float) -> str:
        """Generate a filename for cached data based on molecule settings."""
        return os.path.join(cache_folder, f"{self.molecule}_cd{column_density}_T{temperature}_wl{lower_wavelength}-{upper_wavelength}.pkl")

    def load_from_cache(self, cache_folder: str, column_density: float, temperature: float, lower_wavelength: float, upper_wavelength: float) -> bool:
        """Load cached slab data if available."""
        cache_file = self.get_cache_filename(cache_folder, column_density, temperature, lower_wavelength, upper_wavelength)
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                self.data = pickle.load(f)
            return True
        return False

    def save_to_cache(self, cache_folder: str, column_density: float, temperature: float, lower_wavelength: float, upper_wavelength: float) -> None:
        """Save generated slab data to cache."""
        os.makedirs(cache_folder, exist_ok=True)
        cache_file = self.get_cache_filename(cache_folder, column_density, temperature, lower_wavelength, upper_wavelength)
        with open(cache_file, 'wb') as f:
            pickle.dump(self.data, f)
   
    def to_dict(self) -> dict:
        """Convert Molecule instance to dictionary."""
        return {
            "molecule": self.molecule,
            "isotopologue": self.isotopologue
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Molecule":
        """Create a Molecule instance from a dictionary. This method is not meant to be used standalone, but in combination with SlabModel I/O"""
        instance = cls(
            moleculeName=data["molecule"],
            isotopes=data["isotopologue"],
        )
        instance.mol_mass = Formula(data["molecule"]).mass
        return instance