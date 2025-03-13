import numpy as np
import copy, os, pickle, json
from convolver import Convolver
from molecule import Molecule
from debrisDisk import DebrisDisk
from typing import List, Optional
from scipy.interpolate import interp1d
from tqdm import tqdm

class SlabModel:
    def __init__(self, molecules: List[Molecule], 
                 convolver: Convolver, 
                 densities: List[float], 
                 temperatures: List[float], 
                 emitting_radii: List[float], 
                 distance: float, 
                 normalize: bool = False, 
                 debris_disk: Optional[DebrisDisk] = None,
                 cache_folder: Optional[str] = None) -> None:
        """Generates a slab model by summing molecular contributions and optionally adding a debris disk.
        
        Parameters:
        molecules       (list[Molecule])    (required)  List of Molecule objects that will be used in the slab model
        convolver       (Convolver)         (required)  Convolver object used to adapt slab model to instrument expectation
        densities       (list[float])       (required)  List of column densities in cm^-2 in the same order as the molecules
        temperatures    (list[float])       (required)  List of temperatures in Kelvin in the same order as the molecules
        emitting_radii  (list[float])       (required)  List of radii for emitting areas of each species in AU in the same order as the molecules
        distance        (float)             (required)  Distance between observer and object in parsec
        normalize       (bool)              (optional)  Whether to normalize the final spectrum such that max(flux)=1. Default=false
        debris_disk     (DebrisDisk)        (optional)  The debris disk to add as a typical noise profile of the instrument. Default=None
        cache_folder    (string)            (optional)  The folder used to store and retrieve intermittent files for faster data generation. Default=None
        
        Returns:
        None"""
        self.molecules = molecules
        self.convolver = convolver
        self.densities = densities
        self.temperatures = temperatures
        self.emitting_radii = emitting_radii
        self.distance = distance
        self.normalize = normalize
        self.debris_disk = debris_disk
        self.cache_folder = cache_folder
        self.wavelength = None
        self.flux = None
        self.molecule_fluxes = {}  # Store individual molecule spectra

    def generate_spectrum(self, verbose=False) -> None:
        """Generate the slab model spectrum by summing molecular contributions and resampling as needed.
        Adds debris disk if supplied. Normalizes if self.normalize=True.

        Parameters:
        verbose (bool)  (optional)  Whether to output progress updates in the form of progress bars. Default=False

        Returns:
        None"""
        wavelengths_list = []
        intensities_list = []
        self.molecule_fluxes.clear()

        for i in tqdm(range(len(self.molecules)), desc="Molecules", disable=not verbose):
            mol = self.molecules[i]
            dens = self.densities[i]
            temp = self.temperatures[i]
            rad = self.emitting_radii[i]

            cached_data = None
            if self.cache_folder:
                cached_data = self.load_from_cache(self.cache_folder, mol.molecule, dens, temp, rad, self.distance, self.convolver.wavelength_overlap, self.convolver.resolving_power)

            if cached_data:
                mol_wavelengths, mol_intensities = cached_data
            else:
                mol_wavelengths = []
                mol_intensities = []
                for channel in self.convolver.data:
                    mol.generateSlab(dens, temp, channel["wl"], channel["wu"], cache_folder=self.cache_folder)
                    mol = self.convolver.convolveData(mol, channel["wl"], channel["wu"], emittingRadius=rad, distance=self.distance)
                    mol_wavelengths.append(mol.convWavelength)
                    mol_intensities.append(mol.convIntensity)

                mol_wavelengths = np.concatenate(mol_wavelengths)
                mol_intensities = np.concatenate(mol_intensities)

                if self.cache_folder:
                    self.save_to_cache(self.cache_folder, mol.molecule, dens, temp, rad, self.distance, self.convolver.wavelength_overlap, self.convolver.resolving_power, mol_wavelengths, mol_intensities)

            wavelengths_list.append(mol_wavelengths)
            intensities_list.append(mol_intensities)
            self.molecule_fluxes[mol.molecule] = (mol_wavelengths, mol_intensities)

        # Define a common wavelength grid (high-resolution)
        common_wavelengths = np.unique(np.concatenate(wavelengths_list))

        # Interpolate molecular intensities onto the common grid and sum
        total_intensity = np.zeros_like(common_wavelengths)
        for mol_name, (wl, intensity) in self.molecule_fluxes.items():
            interp_intensity = np.interp(common_wavelengths, wl, intensity, left=0, right=0)
            total_intensity += interp_intensity
            self.molecule_fluxes[mol_name] = (common_wavelengths, interp_intensity)  # Store interpolated fluxes

        # Resample debris disk onto the same wavelength grid
        if self.debris_disk is not None:
            debris_interp = interp1d(self.debris_disk.wavelength, self.debris_disk.flux, bounds_error=False, fill_value=0)
            total_intensity += debris_interp(common_wavelengths)

        # Normalize if required
        if self.normalize and np.max(total_intensity) > 0:
            normalizationFactor = np.max(total_intensity)
            total_intensity /= normalizationFactor
            for mol_name in self.molecule_fluxes:
                self.molecule_fluxes[mol_name] = (common_wavelengths, self.molecule_fluxes[mol_name][1] / normalizationFactor)

        self.wavelength = np.array(common_wavelengths)
        self.flux = total_intensity

    def get_cache_filename(self, cache_folder: str, molecule: str, column_density: float, temperature: float, emitting_radius: float, distance: float, wavelength_overlap: str, resolving_power: str) -> str:
        """Generate a filename for cached data based on molecule settings.
        
        Parameters:
        cache_folder        (string)        (required)  The folder where intermediate cache files are stored
        molecule            (string)        (required)  The molecule name of the intermediate file
        column_density      (float)         (required)  The column density in cm^-2 of the molecule
        temperature         (float)         (required)  The temperature in K of the molecule
        emitting_radius     (float)         (required)  The radius of the emitting area of the molecule
        distance            (float)         (required)  The distance between source and observer in pc
        wavelength_overlap  (string)        (required)  The wavelength overlap removal method of the convolver used
        resolving_power     (string)        (required)  The resolving power value determining method used in the convolver
        
        Returns:
        (string)    The file path to the cache folder"""
        return os.path.join(cache_folder, f"{molecule}_conv_cd{column_density}_T{temperature}_er{emitting_radius}_d{distance}_wo{wavelength_overlap}_rp{resolving_power}.pkl")

    def load_from_cache(self, cache_folder: str, molecule: str, column_density: float, temperature: float, emitting_radius: float, distance: float, wavelength_overlap: str, resolving_power: str):
        """Load cached convolved data if available.
        
        Parameters:
        cache_folder        (string)        (required)  The folder where intermediate cache files are stored
        molecule            (string)        (required)  The molecule name of the intermediate file
        column_density      (float)         (required)  The column density in cm^-2 of the molecule
        temperature         (float)         (required)  The temperature in K of the molecule
        emitting_radius     (float)         (required)  The radius of the emitting area of the molecule
        distance            (float)         (required)  The distance between source and observer in pc
        wavelength_overlap  (string)        (required)  The wavelength overlap removal method of the convolver used
        resolving_power     (string)        (required)  The resolving power value determining method used in the convolver
        
        Returns:
        Pickle file or None"""
        cache_file = self.get_cache_filename(cache_folder, molecule, column_density, temperature, emitting_radius, distance, wavelength_overlap, resolving_power)
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)  # Returns (wavelengths, intensities)
        return None

    def save_to_cache(self, cache_folder: str, molecule: str, column_density: float, temperature: float, emitting_radius: float, distance: float, wavelength_overlap: str, resolving_power: str, wavelengths: np.ndarray, intensities: np.ndarray) -> None:
        """Save generated convolved data to cache.
        
        Parameters:
        cache_folder        (string)        (required)  The folder where intermediate cache files are stored
        molecule            (string)        (required)  The molecule name of the intermediate file
        column_density      (float)         (required)  The column density in cm^-2 of the molecule
        temperature         (float)         (required)  The temperature in K of the molecule
        emitting_radius     (float)         (required)  The radius of the emitting area of the molecule
        distance            (float)         (required)  The distance between source and observer in pc
        wavelength_overlap  (string)        (required)  The wavelength overlap removal method of the convolver used
        resolving_power     (string)        (required)  The resolving power value determining method used in the convolver
        wavelengths         (list[float])   (required)  The list of wavelengths to save in the cache file
        intensities         (list[float])   (required)  The intensities corresponding to the wavelengths to save in the cache file
        
        Returns:
        None"""
        os.makedirs(cache_folder, exist_ok=True)
        cache_file = self.get_cache_filename(cache_folder, molecule, column_density, temperature, emitting_radius, distance, wavelength_overlap, resolving_power)
        with open(cache_file, 'wb') as f:
            pickle.dump((wavelengths, intensities), f)

    def summary(self) -> None:
        """Prints a structured summary of the slab model.
        
        Parameters:
        
        Returns:
        None"""
        header = (
            f"Slab Model Summary\n"
            f"{'=' * 65}\n"
            f"{'Molecule':<10} | {'Density (cm⁻²)':<15} | {'Temperature (K)':<15} | {'Emitting Radius (AU)':<20}\n"
            f"{'-' * 65}"
        )

        rows = []
        for i in range(len(self.molecules)):
            mol = self.molecules[i]
            dens = self.densities[i]
            temp = self.temperatures[i]
            rad = self.emitting_radii[i]

            rows.append(f"{mol.molecule:<10} | {dens:<15.2e} | {temp:<15} | {rad:<20.3f}")

        # Additional details
        footer = (
            f"{'-' * 65}\n"
            f"Distance: {self.distance} pc\n"
            f"Debris Disk Set: {isinstance(self.debris_disk, DebrisDisk)}\n"
            f"Normalized: {self.normalize}\n"
            f"{'=' * 65}\n"
            f"Convolver:\n"
            f"Wavelength Overlap: {self.convolver.wavelength_overlap}\n"
            f"Resolving Power: {self.convolver.resolving_power}\n"
        )

        # Combine all parts
        summary_text = "\n".join([header] + rows + [footer])

        print(summary_text)

    def export_model(self, filepath: str) -> None:
        """Exports the current SlabModel instance to a file.

        Parameters:
        filepath (str): The path where the model should be saved.

        Returns:
        None"""
        model_data = {
            "molecules": [mol.to_dict() for mol in self.molecules],
            "densities": self.densities,
            "temperatures": self.temperatures,
            "emitting_radii": self.emitting_radii,
            "distance": self.distance,
            "normalize": self.normalize,
            "debris_disk": self.debris_disk.to_dict() if self.debris_disk else None,
            "cache_folder": self.cache_folder,
            "convolver": self.convolver.to_dict(),
        }

        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=4)


    @staticmethod
    def import_model(filepath: str) -> "SlabModel":
        """Imports a SlabModel instance from a file.

        Parameters:
        filepath (str): The path of the saved model file.

        Returns:
        SlabModel: A reconstructed SlabModel instance.
        """
        with open(filepath, 'r') as f:
            model_data = json.load(f)

        # Reconstruct objects
        molecules = [Molecule.from_dict(mol_data) for mol_data in model_data["molecules"]]
        convolver = Convolver.from_dict(model_data["convolver"]) 
        convolver.readResolvingModel()
        debris_disk = DebrisDisk.from_dict(model_data["debris_disk"]) if model_data["debris_disk"] else None

        return SlabModel(
            molecules=molecules,
            convolver=convolver,
            densities=model_data["densities"],
            temperatures=model_data["temperatures"],
            emitting_radii=model_data["emitting_radii"],
            distance=model_data["distance"],
            normalize=model_data["normalize"],
            debris_disk=debris_disk,
            cache_folder=model_data["cache_folder"]
        )
        