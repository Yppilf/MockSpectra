# Mock spectra generation
This module generates mock spectra of protoplanetary disks using prodimopy.

## Debris disk
The DebrisDisk class is used to read and prepare a debris disk, which can be added as a typical noise profile of your observing instrument.
The implementation of the class is given in debrisDisk.py, accompanied with documentation per function in the docstrings, including typehints.

## Convolver
The Convolver class is used to resample a generated spectrum to mimic the behavior of a desired instrument. 
It requires an input file specifying resolving powers in different wavelength ranges, which allows overlap.
The implementation is given in convolver.py

## Molecule
The Molecule class uses ProDiMoPy to generate a 0d slab model for a single molecule. This uses data from the HitRan database.
By default only the first isotope from HitRan is considered, but this can be specified. 
Currently only 0D models are implemented. ProDiMoPy also offers possible 1D slab models taking into account radial changes, but that is not implemented.
The implementation of the Molecule class is given in molecule.py

## Slab Model
The SlabModel class combines the previous classes into a combined slab model, where the different species are combined and a debris disk noise profile can be added.
The implementation is given in slabModel.py

## Visualization
In the Visualization class there are methods to plot spectra and the debris disk, to gain insight in the obtained results.
The implementation can be found in visualization.py

## dataCube.py
The DataCube class is used to generate a large dataset of slab models with varying parameters. The implementation is found in dataCube.py. We have the options to generate the full permutation list, or to sample N spectra from the parameter space. The sampling only takes into account the minimum and maximum values of the parameter space, not the intermediate values specified. It uses Latin Hypercube Sampling to ensure full parameter space coverage.