import numpy as np
import matplotlib.pyplot as plt
from convolver import Convolver
from molecule import Molecule
from debrisDisk import DebrisDisk
from slabModel import SlabModel
from visualization import Visualization
import os
from matplotlib.pyplot import figure,show

# This script is for showing the functionalities of the module applied on an example case.
# Because of usage on the cluster, the images are not shown directly but saved to the ./demo folder.

# Create if not exists the demo folder
demo_folder = "./demo"
os.makedirs(demo_folder, exist_ok=True)

# Open a debris disk as a noise profile, resembling of a debris disk spectrum from JWST MIRI MRS
print("Loading debris disk spectrum...")
debris_disk = DebrisDisk("./data/noise_profile/JWST_MIRI_MRS.csv")

# Plot the debris disk with the continuum highlighted, with a window size of 50
print("Plotting debris disk with continuum...")
debris_disk.remove_continuum(50)
viz = Visualization()
viz.plotDebrisDisk(debris_disk, "./demo/debrisdisk_raw.png")

# Plot the debris disk with the continuum subtracted, with standard deviations capped at +-3 and highlighted by horizontal lines
print("Plotting capped and subtracted continuum...")
debris_disk.filter_flux(sigma_multiplier=3)     # Cap the noise spectrum at 3 sigma peaks
viz.plotCappedStdDisk(debris_disk, max_std=3, output_file="./demo/debrisdisk_subtracted.png")

# Create a convolver object loaded on JWST MIRI MRS
print("Creating convolver...")
conv = Convolver("./data/convolver_data/JWST_MIRI_MRS.json", wavelength_overlap="minimal", resolving_power="mean")
conv.readResolvingModel()   

# Plot the convolver settings
print("Plotting chosen convolver settings...")
viz.plotConvolverSettings(conv, "./demo/convolver.png")

# Create a slab model with different molecules, densities, emitting radii and temperatures 
print("Creating slab model...")
mol_list = ["CO2", "C2H2", "HCN", "H2O", "OH", "SO2"]       # names of the molecules included
densities = [2.2e18, 4.6e17, 4.6e17, 3.2e18, 1e18, 1e16]    # Column density of each molecule
temperatures = [400, 500, 875, 625, 1075, 600]              # Temperature of each molecule
emittingRadii = [0.11, 0.05, 0.06, 0.15, 0.06, 0.06]        # Emitting radius of each molecule
molecules = [Molecule(mol) for mol in mol_list]             # Assume only most abundant isotope
distance = 155
slab = SlabModel(molecules, conv, densities, temperatures, emittingRadii, distance, normalize=True, debris_disk=debris_disk, cache_folder="./demo_cache")
slab.generate_spectrum(verbose=True)

# Plot the slab model, highlighting contributions of different molecules separately as well as the total spectrum
print("Plotting slab model...")
viz.plotFullSlabModel(slab, "./demo/slabModel.png")

# Plot a spliced slab model, better highlighting the features of the slab model
viz.plotSplicedSlabModel(slab, 0.5, "./demo/splicedSlabModel.png")

# Export slab model to file
slab.export_model("./demo/saved_slab.json")

# Load slab model from file
loaded_model = SlabModel.import_model("./demo/saved_slab.json")

# Give description of the slab model
loaded_model.summary()



