import numpy as np
from matplotlib.pyplot import figure,show
from debrisDisk import DebrisDisk
from slabModel import SlabModel
from convolver import Convolver
import matplotlib.pyplot as plt

class Visualization:
    def __init__(self):
        pass

    def plotDebrisDisk(self, debris_disk : DebrisDisk, output_file : str) -> None:
        fig = figure()
        frame = fig.add_subplot()

        frame.plot(debris_disk.raw_wavelength, debris_disk.raw_flux, label="Raw debris disk")

        if len(debris_disk.continuum) > 0:
            frame.plot(debris_disk.wavelength, debris_disk.continuum, label="SMA continuum")

        frame.set_xlabel("Wavelength (microns)", fontsize=16)
        frame.set_ylabel("Flux (mJy)", fontsize=16)
        frame.set_title("Raw debris disk with highlighted continuum", fontsize=18)
        frame.legend()
        fig.tight_layout()
        fig.savefig(output_file)

    def plotCappedStdDisk(self, debris_disk : DebrisDisk, max_std : int, output_file : str) -> None:
        sigma_levels = debris_disk.calculate_sigma_levels(max_std)

        fig = figure()
        frame = fig.add_subplot()
        frame.plot(debris_disk.wavelength, debris_disk.flux, label="Debris disk")
        for n in range(1, max_std + 1):
            pos_sigma, neg_sigma = sigma_levels[n]
            plt.axhline(y=pos_sigma, color='black', linestyle='--')
            plt.axhline(y=neg_sigma, color='black', linestyle='--')
            plt.text(debris_disk.wavelength[-1], pos_sigma, f"{n}$\sigma$", color='black')
            plt.text(debris_disk.wavelength[-1], neg_sigma, f"-{n}$\sigma$", color='black')

        frame.set_xlabel("Wavelength (microns)", fontsize=16)
        frame.set_ylabel("Flux (mJy)", fontsize=16)
        frame.set_title("Continuum subtracted debris disk", fontsize=18)
        frame.legend()
        fig.tight_layout()
        fig.savefig(output_file)

    def plotConvolverSettings(self, convolver : Convolver, output_file : str) -> None:
        fig = figure()
        frame = fig.add_subplot()

        for channel in convolver.data:
            frame.plot([channel["wl"], channel["wu"]], [channel["resolving_power"], channel["resolving_power"]], color="r")
            frame.plot([channel["wl"], channel["wl"]], [channel["resolving_power"] - 5, channel["resolving_power"] + 5], color="r", linewidth=2)
            frame.plot([channel["wu"], channel["wu"]], [channel["resolving_power"] - 5, channel["resolving_power"] + 5], color="r", linewidth=2)
            plt.text((channel["wu"]-channel["wl"])/4 + channel["wl"], channel["resolving_power"]-80, channel["channel_name"], color='black')
        frame.set_xlabel("Wavelength (microns)", fontsize=16)
        frame.set_ylabel("Resolving power", fontsize=16)
        frame.set_title("Resolving power versus wavelength", fontsize=18)
        fig.tight_layout()
        fig.savefig(output_file)

    def plotFullSlabModel(self, slab_model : SlabModel, output_file : str) -> None:
        fig = figure()
        frame = fig.add_subplot()

        # Plot total flux
        frame.plot(slab_model.wavelength, slab_model.flux, label="Total flux", linewidth=2, color="black")

        # Plot individual molecules
        sorted_molecules = sorted(slab_model.molecule_fluxes.items(), key=lambda x: np.max(x[1][1]))[::-1]
        for mol_name, (mol_wavelength, mol_flux) in sorted_molecules:
            frame.plot(mol_wavelength, mol_flux, label=mol_name)

        frame.set_xlabel("Wavelength (microns)", fontsize=16)
        frame.set_ylabel("Flux", fontsize=16)
        frame.set_title("Slab Model with Individual Molecule Contributions", fontsize=18)
        frame.legend()
        fig.tight_layout()
        fig.savefig(output_file)

    def plotSplicedSlabModel(self, slab_model : SlabModel, zoom_interval : int, output_file : str) -> None:
        sorted_molecules = sorted(slab_model.molecule_fluxes.items(), key=lambda x: np.max(x[1][1]))[::-1]
        min_wl, max_wl = np.min(slab_model.wavelength), np.max(slab_model.wavelength)
        num_subplots = int(np.ceil((max_wl - min_wl) / zoom_interval))

        num_rows = int(np.floor(np.sqrt(num_subplots)))
        num_cols = int(np.ceil(num_subplots / num_rows))

        fig = figure(figsize=(num_cols * 4, num_rows * 3))
        frame = fig.subplots(nrows=num_rows, ncols=num_cols)
        frame = frame.flatten()

        valid_axes = []

        for i in range(num_subplots):
            wl_min = min_wl + i * zoom_interval
            wl_max = wl_min + zoom_interval

            mask = (slab_model.wavelength >= wl_min) & (slab_model.wavelength < wl_max)
            zoom_wavelength = slab_model.wavelength[mask]
            zoom_flux = slab_model.flux[mask]

            if len(zoom_wavelength) == 0:
                frame[i].set_visible(False)  # Hide empty plots
                continue

            valid_axes.append(frame[i])
            
            frame[i].plot(zoom_wavelength, zoom_flux, label="Total Flux", linewidth=2, color="black")

            # Plot molecule contributions (sorted)
            for mol_name, (mol_wavelength, mol_flux) in sorted_molecules:
                mol_mask = (mol_wavelength >= wl_min) & (mol_wavelength < wl_max)
                frame[i].plot(mol_wavelength[mol_mask], mol_flux[mol_mask], label=mol_name)

            frame[i].tick_params(axis='both', labelsize=12)

        for i in range(num_subplots, num_cols*num_rows):
             frame[i].set_visible(False)

        fig.supxlabel("Wavelength (microns)", fontsize=18)
        fig.supylabel("Flux", fontsize=18)
        fig.suptitle("Spliced Slab Model with Individual Molecule Contributions", fontsize=20)

        # Create a single legend to the right
        handles, labels = valid_axes[0].get_legend_handles_labels()  # Get legend from one subplot
        fig.subplots_adjust(right=0.75)  # Make space for the legend on the right
        fig.legend(handles, labels, loc='center right', fontsize=14)

        # Adjust layout to prevent overlapping labels
        fig.tight_layout()
        plt.subplots_adjust(left=0.05, right=0.85)
        fig.savefig(output_file)