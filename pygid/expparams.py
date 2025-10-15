from typing import Any
import numpy as np
import os
import collections
from dataclasses import dataclass
import fabio
import warnings

from . import pixel_dict


@dataclass
class ExpParams:
    """
        A class to store experimental parameters for X-ray scattering experiments.
        These parameters describe the experimental setup, such as wavelength, detector
        size, position and orientation.

        Attributes
        ----------
        poni_path : str
            The path to the PONI file with experimental parameters.
        mask_path : str, optional
            The path to a mask file. The extension can be '.npy', '.edf', '.tiff' or'.tif'.
            Masked pixes should be 1 or True
        mask : np.ndarray, optional
            A binary mask where 1 (True) indicates a masked area and 0 (False) indicates a region of interest (ROI).
        count_range : List, optional
            Minimum and maximum values for the dynamic mask
        SDD : float, optional
            The sample-to-detector distance, measured in meters. Can be read from poni-file.
        wavelength : float, optional
            The wavelength of the X-rays used in the experiment, measured in meters. Can be read from poni-file.
        rot1, rot2, rot3 : float, optional
            Rotation angles for the experimental setup, representing rotation around different axes
            (e.g., the 3 Euler angles or rotation matrices). Can be read from poni-file.
        poni1, poni2 : float, optional
            The PONI calibration values that describe the position of the direct beam position before rotations.
            Can be read from poni-file.
        centerX, centerY : float, optional
            The coordinates (in pixels) representing the driect beam position on the detector.
        px_size : float, optional
            The pixel size in real space (in meters).
            Should be given if poni_path is not provided or couldn't read from poni-fie.
        flipud : bool, optional
            A flag to indicate if the image should be flipped upside down (default is False).
        fliplr : bool, optional
            A flag to indicate if the image should be flipped left-right (default is False).
        transp : bool, optional
            A flag indicating if the image should be transposed (default is False).
        img_dim : float, optional
            The dimensions of the image (e.g., width and height in pixels).
        k : float, optional
            A wave vector. Calculated automatically from wavelength.
        ai: float or list, optional
            Angle of incidence in GID experiments (in degrees) (default is 0).
        scan: str, optional
            A command that was called for the measurement of the angular scan.
            Should consist of either 5 values or 3:
            "ascan om 0.0400 0.1000 12 3"
            "0.0400 0.1000 12" (start, end, number of points -1)
            ai will be calculated from scan if provided.
    Example:
        --------
        exp_config = ExpParams( poni_path = 'LaB6.poni',
                        mask_path = 'mask.npy',
                        fliplr = True, flipud = True, transp = False, count_range = [0,1000],  ai = 0)
       """

    poni_path: str = None
    mask_path: str = None
    mask: np.array = None
    flipud: bool = False
    fliplr: bool = False
    transp: bool = False
    px_size: float = None
    img_dim: float = None
    n: Any = None
    SDD: float = None
    wavelength: float = None
    rot1: float = None
    rot2: float = None
    rot3: float = 0
    poni1: float = None
    poni2: float = None
    centerX: float = None
    centerY: float = None
    k: float = None
    count_range: list = None
    ai: float = None
    scan: str = None

    def __post_init__(self):
        """Post-initialization hook to prepare experimental parameters.

            This method performs several setup operations after dataclass initialization:
            - Parses and constructs the incident angle (`ai`) list from the scan string.
            - Loads a detector mask from either a `.npy` or image-based mask file.
            - Reads experimental geometry parameters from a PONI file if provided.
            - Updates experimental parameters based on image dimensions.
            - Performs a final parameter consistency check.

            Raises:
                ValueError: If a specified mask file cannot be found or read.
            """
        # Parse and generate the list of incident angles if scan information is provided
        if self.scan is not None:
            parts = self.scan.split()
            if len(parts) != 3:
                start, end, num_points = float(parts[2]), float(parts[3]), int(parts[4])
            else:
                start, end, num_points = float(parts[0]), float(parts[1]), int(parts[2])
            self.ai = np.round(np.linspace(start, end, num_points + 1), 4).tolist()
            print(f"ai list calculated: {self.ai}")

        # Load the detector mask (supports .npy or Fabio-compatible image formats)
        if self.mask_path is not None:
            extension = os.path.splitext(self.mask_path)[1]
            if extension == '.npy':
                self.mask = np.load(self.mask_path).astype(bool)  # .astype('float64')
            else:
                try:
                    self.mask = fabio.open(self.mask_path).data.astype(bool)
                except:
                    raise ValueError("Mask file not found.")
        # Load geometry calibration parameters from the PONI file
        if self.poni_path is not None:
            self.read_from_file(self.poni_path)

        # Update experimental parameters if image dimensions are defined
        if self.img_dim is not None:
            self._exp_params_update_()
        # Validation to ensure all parameters are consistent and correctly set
        self.check_params()

    def check_params(self):
        """
                Validation to ensure all parameters are consistent and correctly set
        """
        if self.px_size is None:
            raise ValueError('px_size is not set and Detector was not recognized. Please provide.')
        if (self.centerX is None or self.centerY is None) and (self.poni1 is None or self.poni2 is None):
            raise ValueError('Center position and PONI are not set.')
        if self.wavelength is None:
            raise ValueError('wavelength is not provided.')
        if self.SDD is None:
            raise ValueError('SDD is not provided.')
        if self.rot1 is None:
            warnings.warn("rot1 is not provided. rot1 = 0 will be used",
                          category=UserWarning)
            self.rot1 = 0
        if self.rot2 is None:
            warnings.warn("rot2 is not provided. rot2 = 0 will be used", category=UserWarning)
            self.rot2 = 0
        if self.ai is None:
            warnings.warn("Angle of incidence (ai) and scan are not provided. ai = 0 will be used",
                          category=UserWarning)
            self.ai = 0

    def read_from_file(self, filename):
        """
        Reads detector geometry and experiment configuration from a poni-file.

        Parameters
        ----------
        filename : str
            poni-file path
        """

        data = collections.OrderedDict()
        with open(filename) as opened_file:
            for line in opened_file:
                if line.startswith("#") or (":" not in line):
                    continue
                words = line.split(":", 1)
                key = words[0].strip().lower()
                try:
                    value = words[1].strip()
                except:
                    pass
                data[key] = value
        self.read_from_dict(data)

    def read_from_dict(self, config):

        """
        Reads detector geometry and experiment configuration from a dictionary.

        Parameters
        ----------
        config : dict
            Dictionary containing keys

        Notes
        -----
        If pixel size is not explicitly provided, attempts to infer it from a known detector name
        using `pixel_dict.pixel_dict`.
        """

        if "distance" in config:
            self.SDD = float(config["distance"])
        elif "dist" in config:
            self.SDD = float(config["dist"])
        if "poni1" in config:
            self.poni1 = float(config["poni1"])
        if "poni2" in config:
            self.poni2 = float(config["poni2"])
        if "rot1" in config:
            self.rot1 = float(config["rot1"])
        if "rot2" in config:
            self.rot2 = float(config["rot2"])
        if "rot3" in config:
            self.rot3 = float(config["rot3"])
        if "wavelength" in config:
            self.wavelength = float(config["wavelength"]) * 1e10
        if self.px_size is None:
            if "pixel1" in config:
                self.px_size = int(config["pixel1"])
            else:
                if "Detector" in config:
                    self.Detector = str(config["Detector"])
                    self.px_size = pixel_dict.pixel_dict["Detector"]
                if "detector" in config:
                    self.Detector = str(config["detector"])
                    self.px_size = pixel_dict.pixel_dict[self.Detector]

    def _calc_dq_(self):
        """
        Calculates resolution based on the pixel size.
        """
        dq = 4 * np.pi / self.wavelength * np.sin(0.5 * np.arctan(self.px_size / self.SDD))
        return dq

    def _calc_k_(self):
        """
        Calculates the wavevector from the given wavelength.
        """
        if self.k is None:
            self.k = 2 * np.pi / self.wavelength

    def _exp_params_update_(self):
        """
        Calculates experimental parameters based on the flipping keys.
        """
        self._cal_rot_()
        if self.poni1 is None or self.poni2 is None:
            self._calc_poni_()
        elif self.centerY is None or self.centerX is None:
            self._calc_center_()

    def _calc_poni_(self):
        """
        Calculates poni1 and poni2 from the beam center.

        """

        if self.transp:
            self.centerY, self.centerX = self.centerX, self.centerY
        if self.flipud:
            self.centerY = self.img_dim[0] - self.centerY
        if self.fliplr:
            self.centerX = self.img_dim[1] - self.centerX

        self.poni1, self.poni2 = (self.centerY * self.px_size - self.SDD * np.tan(self.rot2) / np.cos(self.rot1),
                                  self.centerX * self.px_size + self.SDD * np.tan(self.rot1))

    def _calc_center_(self):
        """
        Calculates beam center from poni1 and poni2.
        """
        if self.transp:
            self.poni1, self.poni2 = self.poni2, self.poni1

        if self.flipud:
            self.poni1 = self.img_dim[0] * self.px_size - self.poni1
        if self.fliplr:
            self.poni2 = self.img_dim[1] * self.px_size - self.poni2

        self.centerY, self.centerX = ((self.SDD * np.tan(self.rot2) / np.cos(self.rot1) + self.poni1) / self.px_size,
                                      (-self.SDD * np.tan(self.rot1) + self.poni2) / self.px_size)

    def _cal_rot_(self):
        """
        Calculates detector rotation angles based on the flipping keys.
        """
        if self.transp:
            self.rot1, self.rot2 = -self.rot2, -self.rot1
        if self.flipud:
            self.rot2 = -self.rot2
        if self.fliplr:
            self.rot1 = -self.rot1
