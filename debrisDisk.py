import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from typing import Tuple, Optional

class DebrisDisk:
    def __init__(self, filename: Optional[str] = None, 
                 wavelength: Optional[np.ndarray] = None, 
                 flux: Optional[np.ndarray] = None):
        """Initializes a DebrisDisk object. Data can be loaded from a file or provided directly.

        Parameters:
        filename    (string)        (optional)  Path to a CSV or NPZ file containing debris disk data.  Default = None
        wavelength  (list[float])   (optional)  Wavelength values (if not loading from a file).         Default = None
        flux        (list[float])   (optional)  Flux values corresponding to the provided wavelengths.  Default = None"""
        self.continuum = []
        if filename:
            self.load_data(filename)
        elif wavelength is not None and flux is not None:
            self.wavelength = wavelength
            self.flux = flux
            self.flux = self.flux.astype(float)
            self.raw_wavelength = self.wavelength
            self.raw_flux = self.flux
        else:
            raise ValueError("Either provide a filename or both wavelength and flux arrays.")
    
    def load_data(self, filename: str) -> None:
        """Loads debris disk data from a csv or npz file.
        
        Parameters:
        filename    (string)    (required)  Path to a csv or npz file containing debris disk data.

        Returns:
        None"""
        try:
            if filename.endswith(".csv"):
                data = pd.read_csv(filename, sep=",")
                self.wavelength = data['wavelength'].values
                self.flux = data['flux'].values
                self.raw_wavelength = self.wavelength
                self.raw_flux = self.flux
            elif filename.endswith(".npz"):
                data = np.load(filename)
                self.wavelength = data['wavelengths']
                self.flux = data['intensities']
                self.raw_wavelength = self.wavelength
                self.raw_flux = self.flux
            else:
                raise ValueError("Unsupported file format. Use csv or npz.")
        except Exception as e:
            raise IOError(f"Error loading data from {filename}: {e}")
    
    def remove_continuum(self, window_size: int) -> None:
        """Removes the continuum using a moving average filter.
        
        Parameters:
        window_size (int)   (required) Size of the moving average window.

        Returns:
        None"""
        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")
        
        smoothed_flux = self.moving_average(self.flux, window_size)
        trim_size = window_size // 2
        self.wavelength = self.wavelength[trim_size:-trim_size+1]
        self.continuum = smoothed_flux
        self.flux = self.flux[trim_size:-trim_size+1] - smoothed_flux
    
    @staticmethod
    def moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
        """Computes the moving average of a given dataset.
        
        Parameters:
        data        (list[float])   (required)  Input data array
        window_size (int)           (required) Size of the moving average window.

        Returns:
        list[float]     Smoothed data"""
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    
    def offset_flux(self, method: str="median") -> None:
        """Offsets the flux to center it around zero using either median or mean.
        
        Parameters:
        method  (string)    (optional)  Offset method.Default = "median"
            Options:
            - "median"  offsets by the median of the flux
            - "mean"    offsets by the mean of the flux
        
        Returns:
        None"""
        if method == "median":
            offset = np.median(self.flux)
        elif method == "mean":
            offset = np.mean(self.flux)
        else:
            raise ValueError("Offset method must be 'median' or 'mean'.")
        self.flux -= offset
    
    def interpolate_flux(self, target_wavelengths: np.ndarray) -> np.ndarray:
        """Interpolates the flux to match target wavelengths.
        
        Parameters:
        target_wavelengths  (list[float])   (required) Wavelengths to interpolate to.

        Returns:
        (list[float])   Interpolated flux values."""
        try:
            interp_func = interp1d(self.wavelength, self.flux, kind='linear', fill_value="extrapolate")
            return interp_func(target_wavelengths)
        except Exception as e:
            raise ValueError(f"Error interpolating flux: {e}")
    
    def compute_noise_thresholds(self, sigma_multiplier: float=3.0) -> Tuple[float, float]:
        """Computes noise thresholds based on standard deviation.
        
        Parameters:
        sigma_multiplier    (float)     (optional)   Multiplier for standard deviation. Default=3.0

        Returns:
        tuple[float,float]  Lower and upper noise thresholds."""
        std_dev = np.std(self.flux)
        mean_flux = np.mean(self.flux)
        return mean_flux - sigma_multiplier * std_dev, mean_flux + sigma_multiplier * std_dev
    
    def filter_flux(self, sigma_multiplier: float=3.0) -> None:
        """Clips flux values to be within computed noise thresholds.
        
        Parameters:
        sigma_multiplier    (float)     (optional)   Multiplier for standard deviation. Default=3.0

        Returns:
        None"""
        lower, upper = self.compute_noise_thresholds(sigma_multiplier-1)
        self.flux = np.clip(self.flux, lower, upper)
    
    def calculate_sigma_levels(self, max_sigma: int=3) -> dict:
        """Computes sigma levels from -max_sigma to +max_sigma.
        
        max_sigma   (int)   (optional)  Maximum sigma level to compute. Default=3

        Returns:
        (dict)  Dictionary mapping sigma levels to flux values."""
        std_dev = np.std(self.flux)
        mean_flux = np.mean(self.flux)
        return {n: (mean_flux + n * std_dev, mean_flux - n * std_dev) for n in range(1, max_sigma + 1)}
    
    def to_dict(self) -> dict:
        """Convert DebrisDisk instance to dictionary."""
        return {
            "wavelength": self.wavelength.tolist(),
            "flux": self.flux.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DebrisDisk":
        """Create a DebrisDisk instance from a dictionary."""
        instance = cls(
            wavelength=np.array(data["wavelength"]),
            flux=np.array(data["flux"]),
        )
        return instance
