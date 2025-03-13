from slabModel import SlabModel
from convolver import Convolver
from debrisDisk import DebrisDisk
from molecule import Molecule
import numpy as np
import h5py, os, multiprocessing, itertools, random
from typing import List
from functools import partial
from tqdm import tqdm
from scipy.stats import qmc

class DataCube:
    def __init__(self, convolver : Convolver, debris_disk : DebrisDisk, distances : List[float]) -> None:
        """Initializes the DataCube class.
        
        Parameters:
        convolver   (Convolver)     (required)  The convolver class used in every slab model
        debris_disk (DebrisDisk)    (required)  The debris disk to add to the slab model
        distances   (list[float])   (required)  List of distances at which to generate the slab models
        
        Returns:
        None"""
        self.convolver = convolver
        self.debris_disk = debris_disk
        self.molecules = []
        self.distances = distances
        self.permutations_file = None

    def add_molecule(self, mol_name : str, temperatures : List[float], densities : List[float], emitting_radii : List[float]) -> None:
        """Adds a molecule to the datacube to use for forming slab models
        
        Parameters:
        mol_name        (string)         (required)  The name of the molecule, such as "H2O", "CO2"
        temperatures    (list[float])    (required)  The temperature range in Kelvin at which to form the molecule.
        densities       (list[float])    (required)  The column density range in cm^-2 at which to vary the molecule.
        emitting_radii  (list[float])    (required)  The radii range of the emitting area in AU at which to vary the molecule.
        
        Returns:
        None"""
        obj = {
            "molecule": Molecule(mol_name),
            "temperatures": temperatures,
            "densities": densities,
            "emitting_radii": emitting_radii
        }
        self.molecules.append(obj)

    def genPermutations(self, output_file : str) -> None:
        """Generates a list of all current permutations and saves them to a file
        Note this function will only be applicable to small datasets, otherwise memory issues will arise.
        
        Parameters:
        output_file (string)    (required)  The filepath where to store the permutations
        
        Returns:
        None"""
        # Collect all parameter lists
        temp_ranges = [mol["temperatures"] for mol in self.molecules]
        density_ranges = [mol["densities"] for mol in self.molecules]
        radius_ranges = [mol["emitting_radii"] for mol in self.molecules]

        # Generate all combinations of (T, N, R) across molecules
        molecule_combinations = list(itertools.product(*temp_ranges, *density_ranges, *radius_ranges))
        
        # Add distances to generate full permutations
        full_permutations = list(itertools.product(molecule_combinations, self.distances))

        # Save to HDF5 file
        with h5py.File(output_file, "w") as hdf:
            group = hdf.create_group("permutations")
            for index, (mol_params, distance) in enumerate(full_permutations):
                dataset_name = f"{index}"
                combined_data = np.array([*mol_params, distance], dtype=np.float64)
                group.create_dataset(dataset_name, data=combined_data)

        self.permutations_file = output_file

    def latinHypercubeSamplePermutations(self, output_file: str, num_samples: int) -> None:
        """Latin Hypercube Sampling for diverse coverage."""

        num_params = len(self.molecules) * 3 + 1  # (T, N, R) for each molecule + distance
        sampler = qmc.LatinHypercube(d=num_params)
        samples = qmc.scale(sampler.random(num_samples), [0] * num_params, [1] * num_params)

        with h5py.File(output_file, "w") as hdf:
            group = hdf.create_group("permutations")
            dataset = group.create_dataset("data", shape=(num_samples, num_params), dtype=np.float64)

            for i, sample in enumerate(samples):
                sampled_params = []
                index = 0
                for mol in self.molecules:
                    temp = mol["temperatures"][int(sample[index] * len(mol["temperatures"]))]
                    density = mol["densities"][int(sample[index + 1] * len(mol["densities"]))]
                    radius = mol["emitting_radii"][int(sample[index + 2] * len(mol["emitting_radii"]))]
                    sampled_params.extend([temp, density, radius])
                    index += 3
                
                distance = self.distances[int(sample[-1] * len(self.distances))]
                sampled_params.append(distance)
                dataset[i] = np.array(sampled_params, dtype=np.float64)

        self.permutations_file = output_file



    def gen_dataset(self, num_cores: int, output_folder: str, cache_folder: str, normalize: bool = False) -> None:
        """Generates the entire dataset and saves them to the output folder
        
        Parameters:
        num_cores       (int)       (required)  The number of CPU cores to use for generating slab models. Note ~2Gb RAM per CPU core for the average slab model. Test this number with a single SlabModel before trying to generate a dataset
        output_folder   (string)    (required)  The folder where to save the slab models to.
        cache_folder    (string)    (required)  The folder where temporary files are stored. All slab models generated use the same cache folder to be able to interchange files
        normalize       (bool)      (optional)  Whether to normalize the intensity values for each slab models. Default=False
        
        Returns:
        None"""
        if not self.permutations_file:
            raise ValueError("No permutation file found. Run `gen_permutations()` first.")

        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(cache_folder, exist_ok=True)

        print(f"Loading permutations from {self.permutations_file}...")

        with h5py.File(self.permutations_file, "r") as hdf:
            permutation_keys = list(hdf["permutations"].keys())

        print(f"Total permutations to process: {len(permutation_keys)}")
        print(f"Using {num_cores} CPU cores.")

        # Use multiprocessing with a progress bar
        process_permutation_partial = partial(
            self._process_permutation,
            output_folder=output_folder,
            cache_folder=cache_folder,
            normalize=normalize
        )

        with multiprocessing.Pool(num_cores) as pool, tqdm(total=len(permutation_keys)) as pbar:
            for _ in pool.imap_unordered(process_permutation_partial, permutation_keys):
                pbar.update(1)  # Update progress bar after each completed task

    def _process_permutation(self, permutation_key: str, output_folder: str, cache_folder: str, normalize: bool) -> None:
        """Processes a single permutation, generates the slab model, and saves it to disk."""
        with h5py.File(self.permutations_file, "r") as hdf:
            permutation_data = hdf["permutations"][permutation_key][...]

        # Extract distance (last value)
        distance = permutation_data[-1]

        # Extract molecule-specific parameters correctly
        num_molecules = len(self.molecules)
        temperatures = []
        densities = []
        emitting_radii = []
        molecules_list = []

        for i, mol in enumerate(self.molecules):
            temperatures.append(permutation_data[i])
            densities.append(permutation_data[num_molecules + i])
            emitting_radii.append(permutation_data[2 * num_molecules + i])
            molecules_list.append(mol["molecule"])  # Extract molecule objects

        # Generate slab model
        slab = SlabModel(
            molecules=molecules_list,
            convolver=self.convolver,
            densities=densities,
            temperatures=temperatures,
            emitting_radii=emitting_radii,
            distance=distance,
            normalize=normalize,
            debris_disk=self.debris_disk,
            cache_folder=cache_folder
        )
        slab.generate_spectrum(verbose=False)

        # Save to disk
        output_filename = os.path.join(output_folder, f"slab_{permutation_key}.json")
        slab.export_model(output_filename)
