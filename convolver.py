import json
from scipy.constants import astronomical_unit as au
from scipy.constants import parsec as pc
import numpy as np
from molecule import Molecule

class Convolver:
    def __init__(self, filename : str, wavelength_overlap: str = "lower", resolving_power: str = "mean") -> None:
        """Convolver class with optional caching and configurable settings.
        
        Parameters:
        filename            (string)    (required)  The filepath to the JSON file containing the data
        wavelength_overlap  (string)    (optional)  What method to apply for overlapping wavelength ranges of channels. Default=lower
            Options:
                lower   - Always takes the maximum of the lower channel range, continuing the second channel from that point
                upper   - Always takes the maximum of the higher channel range, using the lower channel until that point
                optimal - Takes the wavelength range of the channel with highest resolving power, cutting the other channel at that point
                minimal - Takes the wavelength range of the channel with the lowest resolving power, cutting the other channel at that point
        resolving_power     (string)    (optional)  The method of determining the resolving power used. Default=mean
            Options:
                mean    - Takes the average between upper and lower limit of resolving power of the channel
                max     - Takes the upper limit
                min     - Takes the lower limit
        
        Returns:
        None"""
        self.filename = filename
        self.wavelength_overlap = wavelength_overlap
        self.resolving_power = resolving_power
        self.data = None

    def readResolvingModel(self) -> None:
        """Read a JSON resolving model into memory
        
        Parameters:
        

        Returns:
        None"""
        # To whomever might need to read this, sorry
        with open(self.filename) as json_data:
            d = json.load(json_data)

        # Determine used resolving power
        for channel in d:
            u = channel["resolving_power_upper"]
            l = channel["resolving_power_lower"]

            if self.resolving_power=="mean":
                channel["resolving_power"] = (u+l)/2
            elif self.resolving_power=="max":
                channel["resolving_power"] = u
            elif self.resolving_power=="min":
                channel["resolving_power"] = l

        # Determine used wavelength range
        for i,channel in enumerate(d):
            if self.wavelength_overlap=="lower":
                if i==0:
                    channel["wl"] = channel["wavelength_lower"]
                else:
                    channel["wl"] = d[i-1]["wu"]
                channel["wu"] = channel["wavelength_upper"]
            elif self.wavelength_overlap=="upper":
                if i==len(d)-1:
                    channel["wu"] = channel["wavelength_upper"]
                else:
                    channel["wu"] = d[i+1]["wavelength_lower"]
                channel["wl"] = channel["wavelength_lower"]
            elif self.wavelength_overlap=="optimal":
                if i==0:
                    channel["wl"] = channel["wavelength_lower"]
                    if channel["resolving_power"] > d[i+1]["resolving_power"]:
                        channel["wu"] = channel["wavelength_upper"]
                    else:
                        channel["wu"] = d[i+1]["wavelength_lower"]
                elif i==len(d)-1:
                    channel["wu"] = channel["wavelength_upper"]
                    if channel["resolving_power"] > d[i-1]["resolving_power"]:
                        channel["wl"] = channel["wavelength_lower"]
                    else:
                        channel["wl"] = d[i-1]["wavelength_upper"]
                else:
                    if channel["resolving_power"] > d[i-1]["resolving_power"]:
                        channel["wl"] = channel["wavelength_lower"]
                    else:
                        channel["wl"] = d[i-1]["wavelength_upper"]

                    if channel["resolving_power"] > d[i+1]["resolving_power"]:
                        channel["wu"] = channel["wavelength_upper"]
                    else:
                        channel["wu"] = d[i+1]["wavelength_lower"]
            elif self.wavelength_overlap=="minimal":
                if i==0:
                    channel["wl"] = channel["wavelength_lower"]
                    if channel["resolving_power"] < d[i+1]["resolving_power"]:
                        channel["wu"] = channel["wavelength_upper"]
                    else:
                        channel["wu"] = d[i+1]["wavelength_lower"]
                elif i==len(d)-1:
                    channel["wu"] = channel["wavelength_upper"]
                    if channel["resolving_power"] < d[i-1]["resolving_power"]:
                        channel["wl"] = channel["wavelength_lower"]
                    else:
                        channel["wl"] = d[i-1]["wavelength_upper"]
                else:
                    if channel["resolving_power"] < d[i-1]["resolving_power"]:
                        channel["wl"] = channel["wavelength_lower"]
                    else:
                        channel["wl"] = d[i-1]["wavelength_upper"]

                    if channel["resolving_power"] < d[i+1]["resolving_power"]:
                        channel["wu"] = channel["wavelength_upper"]
                    else:
                        channel["wu"] = d[i+1]["wavelength_lower"]

        self.data = d
                
    def convolveData(self, molecule : Molecule, 
                     lower : float, 
                     upper : float, 
                     emittingRadius : float, 
                     distance : float) -> Molecule:
        """Applies resolving power calculations on the generated spectrum by the molecule.
        Assumes the entire range of the molecule to be within one channel.
        
        Parameters:
        molecule        (Molecule)  (required)  Molecule object for which to convolve the data
        lower           (float)     (required)  Lower wavelength for convolving in microns
        upper           (float)     (required)  Upper wavelength for convolving in microns
        emittingRadius  (float)     (required)  Radius in AU of emitting area
        distance        (float)     (required)  Distance in parsec from object to observer
        
        Returns:
        Molecule"""

        data = molecule.data

        for channel in self.data:
            if lower >= channel["wl"] and lower < channel["wu"]:
                res = channel["resolving_power"]
                break

        data.convolve(R=res,lambda_0=lower,lambda_n=upper,verbose=False)
        molecule.convWavelength = data.convWavelength
        solid_angle = np.pi*(emittingRadius*au)**2/(distance*pc)**2   # Solid angle = emitting area / distance^2
        molecule.convIntensity = data.convLTEflux*1e26*solid_angle
        return molecule
    
    def to_dict(self) -> dict:
        """Convert Convolver instance to dictionary."""
        return {
            "filename": self.filename,
            "wavelength_overlap": self.wavelength_overlap,
            "resolving_power": self.resolving_power,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Convolver":
        """Create a Convolver instance from a dictionary."""
        return cls(
            filename=data["filename"],
            wavelength_overlap=data["wavelength_overlap"],
            resolving_power=data["resolving_power"],
        )