from convolver import Convolver
from debrisDisk import DebrisDisk
from dataCube import DataCube  # Ensure this imports your DataCube class

# Define test parameters
output_file = "test_permutations.h5"
output_folder = "./test_output"
cache_folder = "./test_cache"
num_cores = 2  # Use a small number of cores for testing

# Create required components
convolver = Convolver("./data/convolver_data/JWST_MIRI_MRS.json", wavelength_overlap="minimal", resolving_power="mean")
convolver.readResolvingModel()   

debris_disk = DebrisDisk("./data/noise_profile/JWST_MIRI_MRS.csv")
debris_disk.remove_continuum(50)

# Create DataCube instance
datacube = DataCube(convolver, debris_disk, distances=[155])  # Single test distance

# Add a test molecule with a small range of values
datacube.add_molecule(
    mol_name="CO",
    temperatures=[100.0, 200.0],  # Two test temperatures
    densities=[1e15, 1e16],  # Two test densities
    emitting_radii=[0.5, 1]  # Two test emitting radii
)

datacube.add_molecule(
    mol_name="H2O",
    temperatures=[100.0, 200.0],  # Two test temperatures
    densities=[1e17, 1e18],  # Two test densities
    emitting_radii=[0.5, 1]  # Two test emitting radii
)

# Generate permutations and save to file
datacube.genPermutations(output_file)

# Generate the dataset
datacube.gen_dataset(
    num_cores=num_cores,
    output_folder=output_folder,
    cache_folder=cache_folder,
    normalize=False
)

print(f"Test dataset generated in {output_folder}")
