# Dataset for deep learning project
from convolver import Convolver
from debrisDisk import DebrisDisk
from dataCube import DataCube  # Ensure this imports your DataCube class

# Define test parameters
output_file = "dataset_permutations.h5"
output_folder = "/scratch/s4950836/ppd_mockspectra"
cache_folder = "/scratch/s4950836/ppd_cache"
num_cores = 16  # Use a small number of cores for testing
num_samples = 2000

# Create required components
convolver = Convolver("./data/convolver_data/JWST_MIRI_MRS.json", wavelength_overlap="minimal", resolving_power="mean")
convolver.readResolvingModel()   

debris_disk = DebrisDisk("./data/noise_profile/JWST_MIRI_MRS.csv")
debris_disk.remove_continuum(50)

# Create DataCube instance
datacube = DataCube(convolver, debris_disk, distances=[100, 150])

datacube.add_molecule(
    mol_name="H2O",
    temperatures=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500],
    densities=[1e5, 1e17, 1e18, 1e19, 1e20, 1e21],
    emitting_radii=[0.1, 0.5]
)

datacube.add_molecule(
    mol_name="CH4",
    temperatures=[100, 200, 300, 400, 500, 600, 700, 800],
    densities=[1e5, 1e15, 1e16, 1e17, 1e18, 1e19],
    emitting_radii=[0.1, 0.5]
)

datacube.add_molecule(
    mol_name="CO",
    temperatures=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500],
    densities=[1e5, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22],
    emitting_radii=[0.1, 0.5]
)

datacube.add_molecule(
    mol_name="NH3",
    temperatures=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    densities=[1e5, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20],
    emitting_radii=[0.1, 0.5]
)

datacube.add_molecule(
    mol_name="CO2",
    temperatures=[200, 300, 400, 500, 600, 700, 800, 900, 1000],
    densities=[1e5, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21],
    emitting_radii=[0.1, 0.5]
)

datacube.add_molecule(
    mol_name="SO2",
    temperatures=[300, 400, 500, 600, 700, 800, 900, 1000, 1100],
    densities=[1e5, 1e15, 1e16, 1e17, 1e18, 1e19],
    emitting_radii=[0.1, 0.5]
)

# Noise mimicing molecules
datacube.add_molecule(
    mol_name="H2S",
    temperatures=[300, 900],
    densities=[1e5, 1e16, 1e19],
    emitting_radii=[0.1, 0.5]
)

datacube.add_molecule(
    mol_name="C2H2",
    temperatures=[300, 900],
    densities=[1e5, 1e16, 1e19],
    emitting_radii=[0.1, 0.5]
)

datacube.add_molecule(
    mol_name="HCN",
    temperatures=[300, 900],
    densities=[1e5, 1e16, 1e19],
    emitting_radii=[0.1, 0.5]
)

# Generate permutations and save to file
print("Generating permutations")
datacube.latinHypercubeSamplePermutations(output_file, num_samples)

# Generate the dataset
print("Generating dataset")
datacube.gen_dataset(
    num_cores=num_cores,
    output_folder=output_folder,
    cache_folder=cache_folder,
    normalize=False
)

print(f"Dataset generated in {output_folder}")
