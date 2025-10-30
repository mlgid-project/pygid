from . import ExpParams
import numpy as np
import numexpr as ne
from typing import Optional, Any
from dataclasses import dataclass
import pickle
import joblib, os, re
import logging
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s"
)

@dataclass
class CorrMaps:
    """
       A data class to store correction matrices.

       Attributes
       ----------
       flat_field : np.ndarray, optional
           The flat field correction matrix used to compensate for detector imperfections.
       pol_corr_matrix : np.ndarray, optional
           The polarization correction matrix.
       solid_angle_corr_matrix : np.ndarray, optional
           The solid angle correction matrix.
       air_attenuation_corr_matrix : np.ndarray, optional
           The air attenuation correction matrix.
       sensor_attenuation_corr_matrix : np.ndarray, optional
           The sensor attenuation correction matrix .
       absorption_corr_matrix : np.ndarray, optional
           The sample absorption correction matrix.
       lorentz_corr_matrix : np.ndarray, optional
           The Lorentz factor correction matrix.

       Methods
       -------
       This class does not define any methods, it is primarily used as a container for storing
       the correction matrices.
       """

    flat_field: Optional[np.array] = None
    dark_current: Optional[np.array] = None
    pol_corr_matrix: Optional[np.array] = None
    solid_angle_corr_matrix: Optional[np.array] = None
    air_attenuation_corr_matrix: Optional[np.array] = None
    sensor_attenuation_corr_matrix: Optional[np.array] = None
    absorption_corr_matrix: Optional[np.array] = None
    lorentz_corr_matrix: Optional[np.array] = None


@dataclass
class CoordMaps:
    """
    A data class for calculating q- and angular ranges, correction matrices, and remapped pixel positions.

    Attributes
    ----------
    params : ExpParams, optional
        Experimental parameters.
    hor_positive : bool, default=False
        Flag to use only positive horizontal axis values relative to the direct beam.
    vert_positive : bool, default=False
        Flag to use only positive vertical axis values relative to the direct beam.
    img_dim : np.ndarray, optional
        Dimensions of the raw image frame.
    q_xy_range, q_z_range, q_x_range, q_y_range : optional
        Q-space ranges for GID or transmission experiments.
    radial_range, angular_range : optional
        q- and angle- ranges.
    dq : float, optional
        Step size in q-space.
    dang : float, default=0.3
        Step size in azimuthal angle.
    corr_matrices : CorrMaps, optional
        Holds correction matrices. Defaults to None.
    make_pol_corr, make_solid_angle_corr, make_air_attenuation_corr,
    make_sensor_attenuation_corr, make_absorption_corr, make_lorentz_corr : bool, default=False
        Flags to calculate specific correction matrices.
    pol_type : float, default=0.98
        Polarization parameter: 0.98 for synchrotrons, 0.5 for unpolarized tubes.
    air_attenuation_coeff, sensor_attenuation_coeff, sample_attenuation_coeff : float, optional
        Linear coefficients for air, sensor, and sample attenuation corrections (in 1/m).
    sensor_thickness, sample_thickness : float, optional
        Thickness of the sensor and sample (in meters).
    powder_dim : float, optional
        Dimension of powder for Lorentz correction: 2 or 3.
    dark_current, flat_field : np.ndarray, optional
        Arrays for dark current and flat field corrections.
    path_to_save : str, optional
        Path where coordinate map will be saved (.pkl format).
    path_to_load : str, optional
        Path from which coordinate map will be loaded (.pkl format).
    sub_matrices : Any, optional
        Automatically created if `params` contains a list of incident angles.
    ai : float, optional
        Incident angle (in degrees). Each matrix should have only one ai;
        otherwise, sub_matrices are created automatically.

    Example
    -------
    matrix1 = CoordMaps(
        params=params,
        hor_positive=True,
        vert_positive=True,
        dang=0.1,
        make_pol_corr=True,
        make_solid_angle_corr=True
    )
    """

    params: ExpParams = None
    img_dim: Optional[np.array] = None
    hor_positive: bool = False
    vert_positive: bool = False
    q_xy_range: tuple = None
    q_z_range: tuple = None
    q_x_range: tuple = None
    q_y_range: tuple = None
    dq: float = None
    radial_range: tuple = None
    angular_range: tuple = None
    dang: float = 0.3
    corr_matrices: CorrMaps = None
    make_pol_corr: bool = False
    make_solid_angle_corr: bool = False
    make_air_attenuation_corr: bool = False
    make_sensor_attenuation_corr: bool = False
    make_absorption_corr: bool = False
    make_lorentz_corr: bool = False
    pol_type: float = 0.98
    air_attenuation_coeff: float = None
    sensor_attenuation_coeff: float = None
    sensor_thickness: float = None
    sample_attenuation_coeff: float = None
    sample_thickness: float = None
    powder_dim: float = None
    dark_current: Optional[np.array] = None
    flat_field: Optional[np.array] = None
    path_to_save: str = None
    path_to_load: str = None
    sub_matrices: Any = None
    logger: Any = None
    ai: float = None

    def __post_init__(self):
        """
            Post-initialization for CoordMaps.

            - If `path_to_load` is provided, loads saved instance attributes and updates the current instance.
            - Ensures `params` and `ai` are set.
            - If `ai` is a list, automatically creates `sub_matrices` for each incident angle.
        """

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        # Load saved instance if a path is provided
        if self.path_to_load is not None:
            loaded_attrs = self.load_instance(self.path_to_load).__dict__
            if 'path_to_save' in loaded_attrs:
                del loaded_attrs['path_to_save']
            if 'path_to_load' in loaded_attrs:
                del loaded_attrs['path_to_load']
            self.__dict__.update(loaded_attrs)
            del loaded_attrs
            return

        # Ensure experimental parameters are provided
        if self.params is None:
            raise AttributeError('No ExpParams class instance provided.')

        # Set incident angle if not provided
        if self.ai is None:
            self.ai = self.params.ai

        # If multiple incident angles are provided, create sub-matrices
        if isinstance(self.ai, list) or isinstance(self.ai, np.ndarray):
            attrs = self.__dict__.copy()
            if 'path_to_save' in attrs:
                del attrs['path_to_save']
            if 'path_to_load' in attrs:
                del attrs['path_to_load']

            self.sub_matrices = []
            for angle in self.ai:
                attrs["ai"] = angle
                self.sub_matrices.append(CoordMaps(**attrs))

        if self.path_to_save is not None:
            self.save_instance()

    def save_instance(self):
        """
            Save the current CoordMaps instance to disk using joblib.
            The instance is saved to `self.path_to_save` in .pkl format.

            """
        if self.path_to_save is None:
            return
        joblib.dump(self, self.path_to_save)
        self.logger.info(f"CoordMaps instance saved in {self.path_to_save}")

    @classmethod
    def load_instance(cls, path_to_load):
        """
            Load a saved instance of the class from disk using joblib.
        """
        return joblib.load(path_to_load)

    def _coordmaps_update_(self):
        """
            Update coordinate maps, pixel coordinates, q- and angular ranges,
            and correction matrices based on current experimental parameters.
        """
        # Calculate wavevector k
        self.params._calc_k_()

        # Compute dq if not defined
        if self.dq is None:
            self.dq = self.params._calc_dq_()

        # Update image dimensions
        self.img_dim = self.params.img_dim

        # Compute pixel coordinates relative to direct beam
        self.y = np.arange(self.img_dim[0]) * self.params.px_size - self.params.poni1
        self.x = np.arange(self.img_dim[1]) * self.params.px_size - self.params.poni2

        # Determine angular and q-ranges
        self._find_ang_ranges_()
        self._find_q_ranges_()

        # Calculate correction matrices
        self._calc_corrs_()

    def _find_q_ranges_(self):
        """
            Calculate q-ranges in XY and Z directions and q_abs.

            Updates self.q_xy_range and self.q_z_range based on the current
            incident angle and experimental parameters.
        """
        # calculate ranges if they are not already defined
        if self.q_xy_range is None or self.q_z_range is None:
            # Convert pixel positions to lab-frame q-vectors if not done yet
            if not hasattr(self, 'q_lab_from_p'):
                self.q_lab_from_p = self._p_to_q_lab_(calc_type="corner")

            # Convert lab-frame q to sample-frame q and compute ranges
            q_xy_range, q_z_range = self._find_ranges_q_giwaxs_(
                *self._q_smpl_to_q_giwaxs_(
                    self._q_lab_to_q_smpl_(
                        self.q_lab_from_p,
                        ai=self.ai
                    )
                )
            )
            if self.q_xy_range is None:
                self.q_xy_range = q_xy_range
            if self.q_z_range is None:
                self.q_z_range = q_z_range

            self.q_xy_range = list(self.q_xy_range)
            self.q_z_range = list(self.q_z_range)

            # Apply positivity constraints if requested
            self.q_xy_range[0] = self.q_xy_range[0] if not self.hor_positive else np.maximum(self.q_xy_range[0], 0)
            self.q_z_range[0] = self.q_z_range[0] if not self.vert_positive else np.maximum(self.q_z_range[0], 0)

        # Only calculate if q_min or q_max are not already defined

        if self.radial_range is None:
            if not hasattr(self, 'q_lab_from_p'):
                self.q_lab_from_p = self._p_to_q_lab_(calc_type="corner")
            q_min, q_max = self._find_q_abs_(
                self._q_lab_to_q_smpl_(
                    self.q_lab_from_p,
                    ai=self.ai
                )
            )
            self.radial_range = (q_min, q_max)

    def _find_ang_ranges_(self):
        """
            Calculate angular range (ang_min, ang_max) for the sample.

            Updates self.angular_range based on
            pixel positions and experimental parameters. Applies horizontal
            or vertical positivity constraints if specified.
        """
        # Only calculate if ang_min or ang_max are not already defined
        if self.angular_range is not None:
            return

        if not hasattr(self, 'q_lab_from_p'):
            self.q_lab_from_p = self._p_to_q_lab_(calc_type="frame")
        else:
            # Recalculate if q has certain shapes (corner cases)
            if len(self.q) in [4, 6, 8]:
                self.q_lab_from_p = self._p_to_q_lab_(calc_type="frame")
        ang_min, ang_max = self._find_ang_()
        # Apply horizontal/vertical positivity constraints
        if self.hor_positive:
            ang_min = max(ang_min, -90)
            ang_max = min(ang_max, 90)
        if self.vert_positive:
            ang_min = max(ang_min, 0)
            ang_max = min(ang_max, 180)
        # Update angular range
        self.angular_range = (ang_min, ang_max)


    def _find_ang_(self):
        """
        Computes the angular range (phi) from the q-vector components.

        Returns
        -------
        tuple
            (min_phi, max_phi) in degrees
        """
        q = self.q
        # Calculate azimuthal angle φ relative to qy axis, converted to degrees
        phi = 180 - np.arctan2(q[..., 2], np.sqrt(q[..., 1] ** 2 + q[..., 0] ** 2) * np.sign(q[..., 1])) / np.pi * 180
        self.phi = phi
        return np.min(phi), np.max(phi)

    def _p_to_q_lab_(self, calc_type="corner"):
        """
            Converts detector pixel positions to scattering vectors q in the laboratory frame.

            Parameters
            ----------
            calc_type : str, optional
                Determines the region used for calculation:
                - "corner": only detector corners (for q-range estimation)
                - "frame": detector edge (for angular range estimation)
                - "full": full pixel grid (for complete q-mapping)

            Returns
            -------
            np.ndarray
                3D array of q-vectors in the laboratory frame.
        """

        # Define detector coordinates in meters relative to beam center (poni)
        # d1 — horizontal axis, d2 — vertical axis
        SDD = self.params.SDD
        d1 = -np.arange(self.img_dim[1]) * self.params.px_size + self.params.poni2
        d2 = np.arange(self.img_dim[0]) * self.params.px_size - self.params.poni1
        self.d1, self.d2 = d1, d2
        # Build detector mesh grid in lab coordinates
        Py, Pz = np.meshgrid(d1, d2)
        # -----------------------------------------------
        # Select detector region depending on calculation type
        # -----------------------------------------------
        if calc_type == "corner":
            # Use only detector corners for fast estimation
            corner_x = np.array([SDD, SDD, SDD, SDD])
            corner_y = np.array(
                [Py[0, 0], Py[0, -1], Py[-1, -1], Py[-1, 0]])
            corner_z = np.array(
                [Pz[0, 0], Pz[0, -1], Pz[-1, -1], Pz[-1, 0]])
            P = np.stack([corner_x, corner_y, corner_z], axis=1)

            # Add two extra points along detector axes if beam center crosses the detector
            if np.min(d1) * np.max(d1) < 0:
                P = np.append(P, [[SDD, 0, Pz[0, 0]]], axis=0)
                P = np.append(P, [[SDD, 0, Pz[0, -1]]], axis=0)
            if np.min(d2) * np.max(d2) < 0:
                P = np.append(P, [[SDD, Py[0, 0], 0]], axis=0)
                P = np.append(P, [[SDD, Py[0, -1], 0]], axis=0)

            self.P = P
        elif calc_type == "frame":
            # Use detector edge (frame) to get angular range
            Px = SDD * np.ones_like(Pz)
            top_x = Px[0, :]
            bottom_x = Px[-1, :]
            left_x = Px[1:-1, 0]
            right_x = Px[1:-1, -1]

            top_y = Py[0, :]
            bottom_y = Py[-1, :]
            left_y = Py[1:-1, 0]
            right_y = Py[1:-1, -1]

            top_z = Pz[0, :]
            bottom_z = Pz[-1, :]
            left_z = Pz[1:-1, 0]
            right_z = Pz[1:-1, -1]

            edge_x = np.concatenate([top_x, right_x, bottom_x[::-1], left_x[::-1]])
            edge_y = np.concatenate([top_y, right_y, bottom_y[::-1], left_y[::-1]])
            edge_z = np.concatenate([top_z, right_z, bottom_z[::-1], left_z[::-1]])

            P = np.stack([edge_x, edge_y, edge_z], axis=1)
        else:
            # Use the full detector area for correction matrices
            Px = SDD * np.ones_like(Pz)
            P = np.stack([Px, Py, Pz], axis=-1)

        # Apply detector rotation according to experiment geometry
        R3 = rotation_matrix(-self.params.rot1, axis='z')
        R2 = rotation_matrix(self.params.rot2, axis='y')
        R1 = rotation_matrix(self.params.rot3, axis='x')
        t_reshaped = P @ R3.T @ R2.T @ R1.T
        t = t_reshaped.reshape(P.shape)
        # Compute q-vectors from the scattered beam direction
        t_abs = np.sqrt(np.sum(t ** 2, axis=-1))
        t_abs = t_abs[..., np.newaxis]
        k = self.params.k
        kf = ne.evaluate("k * t / t_abs")
        q = kf - np.array([k, 0, 0])
        q_abs = np.sqrt(np.sum(q ** 2, axis=-1))
        self.q = q
        # For full detector calculation, store additional fields
        if calc_type == "full":
            self.kf = kf
            self.cos_2th = ne.evaluate("1 - (q_abs / k) ** 2 / 2")
        return q

    def _calc_corrs_(self):
        """
        Initializes and computes correction matrices for the detector image.

        The method sets up a CorrMaps instance and, depending on user flags,
        calculates optional correction matrices such as polarization, solid angle,
        absorption, or attenuation.

        Notes
        -----
        - The method uses the full q-map when at least one correction requiring
          angular dependence is enabled.
        - Flat-field and dark-current arrays are passed directly to CorrMaps.
        """

        # Initialize CorrMaps container if not provided
        if self.corr_matrices is None:
            self.corr_matrices = CorrMaps()
            self.corr_matrices.flat_field = self.flat_field
            self.corr_matrices.dark_current = self.dark_current

            # Compute full q-mapping if any correction depends on scattering geometry
            if (self.make_lorentz_corr or self.make_absorption_corr or self.make_sensor_attenuation_corr or
                    self.make_air_attenuation_corr or self.make_solid_angle_corr or self.make_pol_corr):
                self.q_lab_from_p = self._p_to_q_lab_(calc_type="full")

            # Polarization correction
            if self.make_pol_corr and self.pol_type is not None:
                self.corr_matrices.pol_corr_matrix = calc_pol_corr_matrix(kf=self.kf, pol_type=self.pol_type)

            # Solid angle correction
            if self.make_solid_angle_corr:
                self.corr_matrices.solid_angle_corr_matrix = calc_solid_angle_corr_matrix(self.cos_2th)

            # air attenuation correction
            if self.make_air_attenuation_corr:
                if self.air_attenuation_coeff is None:
                    self.logger.info("air_attenuation_coeff was not defined, air_attenuation_corr_matrix was not calculated")
                else:
                    self.corr_matrices.air_attenuation_corr_matrix = calc_air_attenuation_corr_matrix(self.cos_2th,
                                                                                                      self.air_attenuation_coeff,
                                                                                                      self.params.SDD)
            # sensor attenuation correction
            if self.make_sensor_attenuation_corr:
                if self.sensor_attenuation_coeff is None or self.sensor_thickness is None:
                    self.logger.info(
                        "sensor_attenuation_coeff or sensor_thickness was not defined, sensor_attenuation_corr_matrix was not calculated")
                else:
                    self.corr_matrices.sensor_attenuation_corr_matrix = calc_sensor_attenuation_corr_matrix(
                        self.cos_2th,
                        self.sensor_attenuation_coeff,
                        self.sensor_thickness,
                        self.params.SDD)
            # absorption correction
            if self.make_absorption_corr:
                if self.sample_attenuation_coeff is None or self.sample_thickness is None:
                    self.logger.info(
                        "sample_attenuation_coeff or sample_thickness was not defined, absorption_corr_matrix was not calculated")
                else:
                    self.corr_matrices.absorption_corr_matrix = calc_absorption_corr_matrix(self.kf, self.ai,
                                                                                            self.sample_attenuation_coeff,
                                                                                            self.sample_thickness)
            # lorentz correction
            if self.make_lorentz_corr:
                if self.powder_dim is None:
                    self.logger.info("powder_dim was not defined, lorentz_corr_matrix was not calculated")
                else:
                    self.corr_matrices.lorentz_corr_matrix = calc_lorentz_corr_matrix(self.kf, self.ai, self.powder_dim)


        else:
            self.corr_matrices = CorrMaps()
            # compute corrections that depend on the ai
            if (self.make_lorentz_corr or self.make_absorption_corr):
                self.q_lab_from_p = self._p_to_q_lab_(calc_type="full")

            if self.make_absorption_corr:
                if self.sample_attenuation_coeff is None or self.sample_thickness is None:
                    self.logger.info("sample_attenuation_coeff or sample_thickness was not defined, absorption_corr_matrix was not calculated")
                else:
                    self.corr_matrices.absorption_corr_matrix = calc_absorption_corr_matrix(self.kf, self.ai,
                                                                                            self.sample_attenuation_coeff,
                                                                                            self.sample_thickness)
            if self.make_lorentz_corr:
                if self.powder_dim is None:
                    self.logger.info("powder_dim was not defined, lorentz_corr_matrix was not calculated")
                else:
                    self.corr_matrices.lorentz_corr_matrix = calc_lorentz_corr_matrix(self.kf, self.ai,
                                                                                      self.powder_dim)

    def _calc_recip_giwaxs_(self, q_xy_range=None, q_z_range=None, dq=None):
        """
            Calculates cylindrical reciprocal-space coordinate maps for GIWAXS geometry.

            Parameters
            ----------
            q_xy_range : list or tuple, optional
                In-plane q range (default: self.q_xy_range).
            q_z_range : list or tuple, optional
                Out-of-plane q range (default: self.q_z_range).
            dq : float, optional
                Step size in reciprocal space (default: self.dq).
        """
        if q_xy_range is None:
            q_xy_range = self.q_xy_range
        if q_z_range is None:
            q_z_range = self.q_z_range
        dq = self.dq if dq is None else dq
        self.dq = dq

        self.p_x_gid, self.p_y_gid = self._q_lab_to_p_(
            self._q_smpl_to_q_lab_(
                self._q_giwaxs_to_q_smpl_(
                    *self._make_q_giwaxs_cart_grid_(q_xy_range, q_z_range, self.dq),
                    ai=self.ai
                ),
                ai=self.ai
            )
        )

    def _calc_pol_giwaxs_(self, radial_range=None, angular_range=None, dang=None, dq=None):
        """
            Computes pixel coordinates corresponding to a GIWAXS polar grid.

            Parameters
            ----------
            radial_range : tuple[float, float], optional
                Radial q-range (default: self.radial_range ).
            angular_range : tuple[float, float], optional
                Angular range in degrees (default: self.angular_range).
            dang : float, optional
                Angular step in degrees (default: self.dang).
            dq : float, optional
                Radial q-step (default: self.dq).
        """
        radial_range = self.radial_range if radial_range is None else radial_range
        angular_range = self.angular_range if angular_range is None else angular_range
        dang = self.dang if dang is None else dang
        self.dang = dang
        dq = self.dq if dq is None else dq
        self.dq = dq

        self.p_x_smpl_pol, self.p_y_smpl_pol = self._q_lab_to_p_(
            self._q_smpl_to_q_lab_(
                self._q_giwaxs_to_q_smpl_(
                    *self._make_q_giwaxs_polar_grid_(radial_range, dq, angular_range,
                                                     dang),
                    ai=self.ai
                ),
                ai=self.ai
            )
        )

    def _calc_pseudopol_giwaxs_(self, q_gid_rad_range=None, q_gid_azimuth_range=None, dang=None, dq=None):
        """
        Computes pixel coordinates for a pseudo-polar GIWAXS grid.

        This function generates a grid in reciprocal space using pseudo-polar
        coordinates (q_rad, q_azimuth) and maps it back to detector pixel positions.

        Parameters
        ----------
        q_gid_rad_range : tuple[float, float], optional
            Radial q-range in GIWAXS geometry. Defaults to self.q_xy_range.
        q_gid_azimuth_range : tuple[float, float], optional
            Azimuthal q-range in GIWAXS geometry. Defaults to self.q_z_range.
        dang : float, optional
            Step size in azimuthal angle (degrees). Defaults to self.dang.
        dq : float, optional
            Step size in radial q (Å⁻¹). Defaults to self.dq.
        """

        dang = self.dang if dang is None else dang
        self.dang = dang
        dq = self.dq if dq is None else dq
        self.dq = dq

        self.p_x_smpl_pseudopol, self.p_y_smpl_pseudopol = self._q_lab_to_p_(
            self._q_smpl_to_q_lab_(
                self._q_giwaxs_to_q_smpl_(
                    *self._make_q_giwaxs_pseudopolar_grid_(q_gid_rad_range, q_gid_azimuth_range, self.dq, self.dang),
                    ai=self.ai
                ),
                ai=self.ai
            )
        )

    def _calc_recip_ewald_(self, q_x_range=None, q_y_range=None, dq=None):
        """
            Computes pixel coordinates for a Cartesian reciprocal-space grid
            based on the Ewald sphere construction (for the transmission geometry).

            This method determines the reciprocal-space coverage of the detector
            in the laboratory frame (q_x, q_y) and generates the corresponding
            detector pixel positions.

            Parameters
            ----------
            q_x_range : tuple[float, float], optional
                Range of q_x values (Å⁻¹). Defaults to self.q_x_range.
            q_y_range : tuple[float, float], optional
                Range of q_y values (Å⁻¹). Defaults to self.q_y_range.
            dq : float, optional
                Step size in reciprocal-space sampling (Å⁻¹). Defaults to self.dq.
            """
        if not hasattr(self, 'q_lab_from_p'):
            self.q_lab_from_p = self._p_to_q_lab_(calc_type="corner")

        if self.q_x_range is None or self.q_y_range is None:
            self.q_x_range, self.q_y_range = self._find_ranges_q_ewald_(
                self.q_lab_from_p
            )
            self.q_x_range = list(self.q_x_range)
            self.q_y_range = list(self.q_y_range)

            self.q_x_range[0] = self.q_x_range[0] if not self.hor_positive else 0
            self.q_y_range[0] = self.q_y_range[0] if not self.vert_positive else 0

        q_x_range = self.q_x_range if q_x_range is None else q_x_range
        q_y_range = self.q_y_range if q_y_range is None else q_y_range
        dq = self.dq if dq is None else dq
        self.dq = dq

        self.p_x_ewald, self.p_y_ewald = self._q_lab_to_p_(
            self._make_q_ewald_cart_grid_(q_x_range, q_y_range, self.dq)
        )

    def _calc_pol_ewald_(self, radial_range=None, angular_range=None, dang=None, dq=None):
        """
            Computes pixel coordinates for a polar grid on the Ewald sphere (for the transmission geometry).

            This method generates a polar reciprocal-space grid (q, φ)
            and maps it onto the detector plane within the Ewald geometry.

            Parameters
            ----------
            radial_range : tuple[float, float], optional
                Range of radial q values (Å⁻¹). Defaults to self.radial_range.
            angular_range : tuple[float, float], optional
                Range of azimuthal angles (degrees). Defaults to self.angular_range.
            dang : float, optional
                Step size in azimuthal angle (degrees). Defaults to self.dang.
            dq : float, optional
                Step size in radial q (Å⁻¹). Defaults to self.dq.
            """
        radial_range = self.radial_range if radial_range is None else radial_range
        angular_range = self.angular_range if angular_range is None else angular_range
        dang = self.dang if dang is None else dang
        self.dang = dang
        dq = self.dq if dq is None else dq
        self.dq = dq

        self.p_x_lab_pol, self.p_y_lab_pol = self._q_lab_to_p_(
            self._make_q_ewald_polar_grid_(radial_range, dq, angular_range,
                                           dang)
        )

    def _calc_pseudopol_ewald_(self, q_rad_range=None, q_azimuth_range=None, dang=None, dq=None):
        """
            Computes pixel coordinates for a pseudo-polar grid on the Ewald sphere (for the transmission geometry).

            This method constructs a pseudo-polar reciprocal-space grid
            (q_rad, q_azimuth) for the Ewald geometry and converts it
            into detector pixel coordinates.

            Parameters
            ----------
            q_rad_range : tuple[float, float], optional
                Range of radial q values (Å⁻¹). Defaults to self.q_xy_range.
            q_azimuth_range : tuple[float, float], optional
                Range of azimuthal q values (Å⁻¹). Defaults to self.q_z_range.
            dang : float, optional
                Step size in azimuthal angle (degrees). Defaults to self.dang.
            dq : float, optional
                Step size in radial q (Å⁻¹). Defaults to self.dq.
        """
        dang = self.dang if dang is None else dang
        self.dang = dang
        dq = self.dq if dq is None else dq
        self.dq = dq

        self.p_x_lab_pseudopol, self.p_y_lab_pseudopol = self._q_lab_to_p_(
            self._make_q_ewald_pseudopolar_grid_(q_rad_range, q_azimuth_range, self.dq, self.dang)
        )

    def _q_lab_to_q_smpl_(self, q_lab, ai=0):
        """
            Converts reciprocal-space vectors from the laboratory frame (q_lab)
            to the sample frame (q_smpl) using the incidence angle.

            Parameters
            ----------
            q_lab : ndarray
                Reciprocal-space vectors in the laboratory frame (Å⁻¹).
            ai : float, optional
                Incidence angle of the sample (degrees). Defaults to 0.

            Returns
            -------
            q_smpl : ndarray
                Reciprocal-space vectors in the sample frame (Å⁻¹).
            """
        ai = np.deg2rad(ai)
        R_ai = rotation_matrix(ai, axis='y')
        q_smpl = q_lab @ R_ai.T
        return q_smpl

    def _q_smpl_to_q_lab_(self, q_smpl, ai=0):
        """
            Converts reciprocal-space vectors from the sample frame (q_smpl)
            to the laboratory frame (q_lab) using the incidence angle.

            Parameters
            ----------
            q_smpl : ndarray
                Reciprocal-space vectors in the sample frame (Å⁻¹).
            ai : float, optional
                Incidence angle of the sample (degrees). Defaults to 0.

            Returns
            -------
            q_lab : ndarray
                Reciprocal-space vectors in the laboratory frame (Å⁻¹).
            """
        ai = np.deg2rad(ai)
        R_ai = rotation_matrix(-ai, axis='y')
        q_lab = q_smpl @ R_ai.T
        return q_lab

    def _q_lab_to_p_(self, q_lab):
        """
            Projects reciprocal-space vectors in the laboratory frame (q_lab)
            onto the detector plane, producing pixel coordinates.

            Parameters
            ----------
            q_lab : ndarray
                Reciprocal-space vectors in the laboratory frame (Å⁻¹).

            Returns
            -------
            p_x, p_y : ndarray
                Detector pixel coordinates corresponding to q_lab.
            """
        SDD = self.params.SDD
        k = self.params.k
        k_refl = q_lab + np.array([k, 0, 0])

        R3 = rotation_matrix(self.params.rot1, axis='z')
        R2 = rotation_matrix(-self.params.rot2, axis='y')
        R1 = rotation_matrix(self.params.rot3, axis='x')

        p_refl = k_refl @ R1.T @ R2.T @ R3.T

        alpha = SDD / p_refl[..., 0]
        alpha = alpha[..., np.newaxis]
        p = alpha * p_refl

        p_y = (p[..., 2] + self.params.poni1) / self.params.px_size
        p_x = -(p[..., 1] - self.params.poni2) / self.params.px_size
        return p_x.astype(np.float32), p_y.astype(np.float32)

    def _q_smpl_to_q_giwaxs_(self, q_smpl):
        """
            Converts reciprocal-space vectors from the sample frame (q_smpl)
            to the GIWAXS coordinate system (q_xy, q_z).

            Parameters
            ----------
            q_smpl : ndarray
                Reciprocal-space vectors in the sample frame (Å⁻¹).

            Returns
            -------
            q_xy_giwaxs : ndarray
                In-plane scattering component (Å⁻¹).
            q_z_giwaxs : ndarray
                Out-of-plane scattering component (Å⁻¹).
            """
        q_x = q_smpl[..., 0]
        q_y = q_smpl[..., 1]
        q_z = q_smpl[..., 2]
        q_xy_giwaxs = ne.evaluate('-sqrt(q_x ** 2 + q_y ** 2) * q_y / abs(q_y)')
        q_z_giwaxs = q_z
        return q_xy_giwaxs, q_z_giwaxs

    def _q_smpl_to_q_ewald_(self, q_smpl):
        """
            Converts reciprocal-space vectors from the sample frame (q_smpl)
            to Ewald-sphere coordinates (q_y, q_z).

            Parameters
            ----------
            q_smpl : ndarray
                Reciprocal-space vectors in the sample frame (Å⁻¹).

            Returns
            -------
            q_y : ndarray
                Horizontal scattering component (Å⁻¹).
            q_z : ndarray
                Vertical scattering component (Å⁻¹).
        """
        q_y = q_smpl[..., 1]
        q_z = q_smpl[..., 2]
        return q_y, q_z

    def _q_giwaxs_to_q_smpl_(self, q_xy_giwaxs, q_z_giwaxs, ai=0):
        """
            Converts reciprocal-space vectors from the GIWAXS coordinate system
            (q_xy, q_z) to the sample frame (q_smpl).

            Parameters
            ----------
            q_xy_giwaxs : ndarray
                In-plane GIWAXS scattering component (Å⁻¹).
            q_z_giwaxs : ndarray
                Out-of-plane GIWAXS scattering component (Å⁻¹).
            ai : float, optional
                Incidence angle of the sample (degrees). Defaults to 0.

            Returns
            -------
            q_smpl : ndarray
                Reciprocal-space vectors in the sample frame (Å⁻¹).
        """
        ai = np.deg2rad(ai)
        k = self.params.k
        q_x = ne.evaluate("(-(q_xy_giwaxs**2 + q_z_giwaxs**2) / (2 * k) + q_z_giwaxs * sin(ai)) / cos(ai)")
        q_y = ne.evaluate("-(q_xy_giwaxs / abs(q_xy_giwaxs)) * sqrt(q_xy_giwaxs**2 - q_x**2)")
        q_z = q_z_giwaxs
        q_smpl = np.stack([q_x, q_y, q_z], axis=-1)
        return q_smpl

    def _find_q_abs_(self, q):
        """
            Determines the minimum and maximum absolute |q| values
            from the provided reciprocal-space array.

            Parameters
            ----------
            q : ndarray
                Reciprocal-space vectors (Å⁻¹).

            Returns
            -------
            q_min : float
                Minimum absolute scattering vector magnitude (Å⁻¹).
            q_max : float
                Maximum absolute scattering vector magnitude (Å⁻¹).
        """
        q_max = np.max(np.linalg.norm(q, axis=-1))
        if len(q) >= 8:
            q_min = 0
        else:
            q_min = np.min(np.linalg.norm(q, axis=-1))
        return q_min, np.max(np.linalg.norm(q, axis=-1))

    def _find_ranges_q_giwaxs_(self, q_xy_giwaxs, q_z_giwaxs):
        """
            Computes the minimum and maximum q-ranges for GIWAXS data
            in both in-plane (q_xy) and out-of-plane (q_z) directions.

            Parameters
            ----------
            q_xy_giwaxs : ndarray
                In-plane GIWAXS scattering component (Å⁻¹).
            q_z_giwaxs : ndarray
                Out-of-plane GIWAXS scattering component (Å⁻¹).

            Returns
            -------
            q_xy_range : tuple(float, float)
                (min, max) range for q_xy (Å⁻¹).
            q_z_range : tuple(float, float)
                (min, max) range for q_z (Å⁻¹).
            """
        q_xy_giwaxs_min, q_xy_giwaxs_max = np.nanmin(q_xy_giwaxs), np.nanmax(q_xy_giwaxs)
        q_z_giwaxs_min, q_z_giwaxs_max = np.nanmin(q_z_giwaxs), np.nanmax(q_z_giwaxs)
        return (q_xy_giwaxs_min, q_xy_giwaxs_max), (q_z_giwaxs_min, q_z_giwaxs_max)

    def _find_ranges_q_ewald_(self, q):
        """
            Computes the q_y and q_z ranges for Ewald-sphere representation
            based on the input reciprocal-space vectors.

            Parameters
            ----------
            q : ndarray
                Reciprocal-space vectors (Å⁻¹).

            Returns
            -------
            q_y_range : tuple(float, float)
                (min, max) range for q_y (Å⁻¹).
            q_z_range : tuple(float, float)
                (min, max) range for q_z (Å⁻¹).
            """
        Q_x = q[..., 0]
        Q_y = q[..., 1]
        Q_z = q[..., 2]
        k = self.params.k
        QQ = -2 * k * Q_x
        Q_1 = Q_y * np.sqrt(QQ / (QQ - Q_x ** 2))
        Q_2 = Q_z * np.sqrt(QQ / (QQ - Q_x ** 2))

        q_y_ewald_min, q_y_ewald_max = -np.nanmax(Q_1), -np.nanmin(Q_1)
        q_z_ewald_min, q_z_ewald_max = np.nanmin(Q_2), np.nanmax(Q_2)
        return (q_y_ewald_min, q_y_ewald_max), (q_z_ewald_min, q_z_ewald_max)

    def _make_q_giwaxs_cart_grid_(self, q_xy_range=(0, 5), q_z_range=(0, 5), dq=0.01):
        """
            Generates a Cartesian grid of q-values for GIWAXS experiments.

            Parameters
            ----------
            q_xy_range : tuple, optional
                Min and max values for the in-plane scattering vector (Å⁻¹). Default is (0, 5).
            q_z_range : tuple, optional
                Min and max values for the out-of-plane scattering vector (Å⁻¹). Default is (0, 5).
            dq : float, optional
                Step size for q-grid (Å⁻¹). Default is 0.01.

            Returns
            -------
            Q_xy : ndarray
                2D grid of in-plane q-values (Å⁻¹).
            Q_z : ndarray
                2D grid of out-of-plane q-values (Å⁻¹).
        """
        q_xy = np.arange(q_xy_range[0], q_xy_range[1], dq)
        q_z = np.arange(q_z_range[0], q_z_range[1], dq)
        self.q_z = q_z
        self.q_xy = q_xy
        Q_xy, Q_z = np.meshgrid(q_xy, q_z)
        return Q_xy, Q_z

    def _make_q_ewald_cart_grid_(self, q_x_range=(0, 5), q_y_range=(0, 5), dq=0.01):
        """
            Generates a Cartesian grid of q-values in the laboratory frame for Ewald-sphere representation.

            Parameters
            ----------
            q_x_range : tuple, optional
                Min and max q_x values (Å⁻¹). Default is (0, 5).
            q_y_range : tuple, optional
                Min and max q_y values (Å⁻¹). Default is (0, 5).
            dq : float, optional
                Step size for the q-grid (Å⁻¹). Default is 0.01.

            Returns
            -------
            q_lab : ndarray
                3D array of q-vectors in the laboratory frame (Å⁻¹), shape (..., 3).
        """
        q_1 = np.arange(q_x_range[0], q_x_range[1], dq)
        q_2 = np.arange(q_y_range[0], q_y_range[1], dq)
        Q_1, Q_2 = np.meshgrid(q_1, q_2)
        k = self.params.k
        QQ = Q_1 ** 2 + Q_2 ** 2
        Q_x = -QQ / (2 * k)
        Q_z = ne.evaluate('sqrt(QQ- Q_x**2)/sqrt(1 + (Q_1/Q_2)**2)') * np.sign(Q_2)
        Q_y = -ne.evaluate('sqrt(QQ - Q_x**2)/sqrt(1 + (Q_2/Q_1)**2)') * np.sign(Q_1)
        self.q_x = q_1
        self.q_y = q_2
        q_lab = np.stack([Q_x, Q_y, Q_z], axis=-1)
        return q_lab

    def _make_q_giwaxs_polar_grid_(self, q_range=[0, 4], dq=0.01, ang_range=[0, 90], dang=0.01):
        """
            Generates a polar grid of q-values for GIWAXS experiments.

            Parameters
            ----------
            q_range : list, optional
                Min and max radial q-values (Å⁻¹). Default is [0, 4].
            dq : float, optional
                Step size for radial q (Å⁻¹). Default is 0.01.
            ang_range : list, optional
                Min and max azimuthal angles (degrees). Default is [0, 90].
            dang : float, optional
                Step size for azimuthal angle (degrees). Default is 0.01.

            Returns
            -------
            Q_xy : ndarray
                In-plane q-values in Cartesian coordinates (Å⁻¹).
            Q_z : ndarray
                Out-of-plane q-values in Cartesian coordinates (Å⁻¹).
            """
        q_pol = np.arange(q_range[0], q_range[1], dq)
        ang_pol = np.arange(ang_range[0], ang_range[1], dang)
        self.q_gid_pol = q_pol
        self.ang_gid_pol = ang_pol

        ang_pol_rad = np.deg2rad(ang_pol)
        Q_pol, ANG_pol = np.meshgrid(q_pol, ang_pol_rad)
        Q_xy, Q_z = Q_pol * np.cos(ANG_pol), Q_pol * np.sin(ANG_pol)
        return Q_xy, Q_z

    def _make_q_ewald_polar_grid_(self, q_range=[0, 4], dq=0.01, ang_range=[0, 90], dang=0.01):
        """
            Generates a polar grid of q-vectors in the laboratory frame for Ewald-sphere experiments.

            Parameters
            ----------
            q_range : list, optional
                Min and max radial q-values (Å⁻¹). Default is [0, 4].
            dq : float, optional
                Step size for radial q (Å⁻¹). Default is 0.01.
            ang_range : list, optional
                Min and max azimuthal angles (degrees). Default is [0, 90].
            dang : float, optional
                Step size for azimuthal angle (degrees). Default is 0.01.

            Returns
            -------
            q_lab : ndarray
                3D array of q-vectors in the laboratory frame (Å⁻¹), shape (..., 3).
            """
        q_pol = np.arange(q_range[0], q_range[1], dq)
        ang_pol = np.arange(ang_range[0], ang_range[1], dang)
        self.q_pol = q_pol
        self.ang_pol = ang_pol
        k = self.params.k

        ang_pol_rad = np.deg2rad(ang_pol)
        Q_pol, ANG_pol = np.meshgrid(q_pol, ang_pol_rad)
        Q_x = -Q_pol * Q_pol / (2 * k)
        Q_r = np.sqrt(Q_pol ** 2 - Q_x ** 2)
        Q_y, Q_z = -Q_r * np.cos(ANG_pol), Q_r * np.sin(ANG_pol)
        q_lab = np.stack([Q_x, Q_y, Q_z], axis=-1)
        return q_lab

    def _make_q_giwaxs_pseudopolar_grid_(self, q_gid_rad_range=None, q_gid_azimuth_range=None, dq=0.01, dang=0.01):
        """
            Generates a pseudopolar grid of q-vectors for GIWAXS experiments.

            Parameters
            ----------
            q_gid_rad_range : list or None
                Optional min and max radial q-values (Å⁻¹). If None, defaults to object attributes.
            q_gid_azimuth_range : list or None
                Optional min and max azimuthal q-values (degrees). If None, uses angular range from object.
            dq : float, optional
                Step size for radial q (Å⁻¹). Default is 0.01.
            dang : float, optional
                Step size for azimuthal angle (degrees). Default is 0.01.

            Returns
            -------
            Q_xy : ndarray
                In-plane q-values in Cartesian coordinates (Å⁻¹).
            Q_z : ndarray
                Out-of-plane q-values in Cartesian coordinates (Å⁻¹).
        """
        q_rad = np.arange(self.radial_range[0], self.radial_range[1], dq)
        ang_range = self.angular_range
        if not hasattr(self, 'phi'):
            self._find_ang_()
        q = self.q
        q_abs = np.sqrt(q[..., 1] ** 2 + q[..., 0] ** 2 + q[..., 2] ** 2)
        phi = np.arctan2(q[..., 2], np.sqrt(q[..., 1] ** 2 + q[..., 0] ** 2) * np.sign(-q[..., 1]))
        phi[phi > np.radians(self.angular_range[1])] = np.nan
        phi[phi < np.radians(self.angular_range[0])] = np.nan
        self.q_rad, self.phi = q_rad, phi
        q_phi = q_abs * phi
        if q_gid_rad_range is not None:
            q_rad = np.arange(q_gid_rad_range[0], q_gid_rad_range[1], dq)

        if q_gid_azimuth_range is not None:
            q_azimuth = np.linspace(q_gid_azimuth_range[0], q_gid_azimuth_range[1],
                                    int((ang_range[1] - ang_range[0]) / dang))
        else:
            q_azimuth = np.linspace(0, np.nanmax(q_phi), int((ang_range[1] - ang_range[0]) / dang))

        self.q_gid_azimuth = q_azimuth
        self.q_gid_rad = q_rad

        Q_rad, Q_azimuth = np.meshgrid(q_rad, q_azimuth)
        ANG = ne.evaluate("Q_azimuth / Q_rad")
        Q_rad[ANG > np.radians(ang_range[1])] = np.nan
        Q_rad[ANG < np.radians(ang_range[0])] = np.nan

        Q_xy, Q_z = Q_rad * np.cos(ANG), Q_rad * np.sin(ANG)
        return Q_xy, Q_z

    def _make_q_ewald_pseudopolar_grid_(self, q_rad_range=[0, 4], q_azimuth_range=[0, 90], dq=0.01, dang=0.01):
        """
            Generates a pseudopolar grid of q-vectors in the laboratory frame for Ewald-sphere experiments.

            Parameters
            ----------
            q_rad_range : list, optional
                Min and max radial q-values (Å⁻¹). Default is [0, 4].
            q_azimuth_range : list, optional
                Min and max azimuthal angles (degrees). Default is [0, 90].
            dq : float, optional
                Step size for radial q (Å⁻¹). Default is 0.01.
            dang : float, optional
                Step size for azimuthal angle (degrees). Default is 0.01.

            Returns
            -------
            q_lab : ndarray
                3D array of q-vectors in the laboratory frame (Å⁻¹), shape (..., 3).
            """
        if len(self.q) in [4, 6, 8]:
            self._p_to_q_lab_(calc_type="frame")

        q = self.q
        q_abs = np.linalg.norm(q, axis=-1)
        q_rad = np.arange(self.radial_range[0], self.radial_range[1], dq)
        ang_range = self.angular_range
        phi = np.arctan2(q[..., 2], -q[..., 1])
        phi[phi > np.radians(ang_range[1])] = np.nan
        phi[phi < np.radians(ang_range[0])] = np.nan

        self.phi = phi
        q_phi = q_abs * phi

        if q_rad_range is not None:
            q_rad = np.arange(q_rad_range[0], q_rad_range[1], dq)

        if q_azimuth_range is not None:
            q_azimuth = np.linspace(q_azimuth_range[0], q_azimuth_range[1], int((ang_range[1] - ang_range[0]) / dang))
        else:
            q_azimuth = np.linspace(0, np.nanmax(q_phi), int((ang_range[1] - ang_range[0]) / dang))

        self.q_azimuth = q_azimuth
        self.q_rad = q_rad

        Q_rad, Q_azimuth = np.meshgrid(q_rad, q_azimuth)
        ANG = ne.evaluate("Q_azimuth / Q_rad")
        Q_rad[ANG > np.radians(ang_range[1])] = np.nan
        Q_rad[ANG < np.radians(ang_range[0])] = np.nan

        k = self.params.k
        Q_x = -Q_rad * Q_rad / (2 * k)
        Q_r = np.sqrt(Q_rad ** 2 - Q_x ** 2)
        Q_y, Q_z = -Q_r * np.cos(ANG), Q_r * np.sin(ANG)

        q_lab = np.stack([Q_x, Q_y, Q_z], axis=-1)
        return q_lab


def calc_pol_corr_matrix(kf, pol_type):
    """
        Calculates the polarization correction matrix.

        Parameters
        ----------
        kf : ndarray
            Scattered wavevector components in lab frame, shape (..., 3).
        pol_type : float or str
            Polarization type. Can be 'tube', 'synchrotron', or a float between 0 and 1.

        Returns
        -------
        pol_corr_matrix : ndarray
            Polarization correction factors normalized to max value.
        """
    k1 = kf[..., 0]
    k2 = kf[..., 1]
    k3 = kf[..., 2]
    cos_gamma_2 = ne.evaluate('k1**2/(k1**2+k2**2)')
    cos_delta_2 = ne.evaluate('(k1**2+k2**2)/(k1**2+k2**2+k3**2)')
    pol_corr_matrix_hor = ne.evaluate('1 - cos_delta_2 * (1 - cos_gamma_2)')
    pol_corr_matrix_vert = ne.evaluate('cos_delta_2')
    if pol_type == "tube":
        pol_type = 0.5
    elif pol_type == "synchrotron":
        pol_type = 0.98
    if not isinstance(pol_type, float) and not isinstance(pol_type, int):
        raise TypeError('pol_type must be float')
    if pol_type > 1 or pol_type < 0:
        raise ValueError('pol_type must be between 0 and 1')
    pol_corr_matrix = ne.evaluate('1 / (pol_corr_matrix_hor*pol_type + pol_corr_matrix_vert*(1-pol_type))')
    norm = np.nanmax(pol_corr_matrix)
    pol_corr_matrix = ne.evaluate("pol_corr_matrix / norm")
    return pol_corr_matrix


def calc_solid_angle_corr_matrix(cos_2th):
    """
        Calculates the solid angle correction matrix.

        Parameters
        ----------
        cos_2th : ndarray
            Cosine of the 2θ scattering angle.

        Returns
        -------
        solid_angle_corr_matrix : ndarray
            Correction factors proportional to cos^3(2θ).
        """
    solid_angle_corr_matrix = ne.evaluate("cos_2th**3")
    return solid_angle_corr_matrix


def calc_air_attenuation_corr_matrix(cos_2th, air_attenuation_coeff, SDD):
    """
        Calculates air attenuation correction matrix.

        Parameters
        ----------
        cos_2th : ndarray
            Cosine of 2θ angles.
        air_attenuation_coeff : float
            Linear attenuation coefficient of air (1/m).
        SDD : float
            Sample-to-detector distance (m).

        Returns
        -------
        air_attenuation_corr_matrix : ndarray
            Normalized air attenuation correction factors.
        """
    air_attenuation_corr_matrix = ne.evaluate("exp(- air_attenuation_coeff * SDD / cos_2th)")
    norm = np.nanmax(air_attenuation_corr_matrix)
    air_attenuation_corr_matrix = ne.evaluate("air_attenuation_corr_matrix / norm")
    return air_attenuation_corr_matrix


def calc_sensor_attenuation_corr_matrix(cos_2th, sensor_attenuation_coeff, sensor_thickness, SDD):
    """
        Calculates sensor attenuation correction matrix.

        Parameters
        ----------
        cos_2th : ndarray
            Cosine of 2θ angles.
        sensor_attenuation_coeff : float
            Linear attenuation coefficient of sensor (1/m).
        sensor_thickness : float
            Detector sensor thickness (m).
        SDD : float
            Sample-to-detector distance (m).

        Returns
        -------
        sensor_attenuation_corr_matrix : ndarray
            Normalized sensor attenuation correction factors.
        """
    sensor_attenuation_corr_matrix = ne.evaluate("1 - exp(-sensor_attenuation_coeff * sensor_thickness / cos_2th)")
    norm = np.nanmax(sensor_attenuation_corr_matrix)
    sensor_attenuation_corr_matrix = ne.evaluate("sensor_attenuation_corr_matrix / norm")
    return sensor_attenuation_corr_matrix


def calc_absorption_corr_matrix(kf, ai, sample_attenuation_coeff, sample_thickness):
    """
        Calculates absorption correction matrix for the sample.

        Parameters
        ----------
        kf : ndarray
            Scattered wavevector components in lab frame.
        ai : float
            Incident angle in degrees.
        sample_attenuation_coeff : float
            Linear attenuation coefficient of sample (1/m).
        sample_thickness : float
            Sample thickness (m).

        Returns
        -------
        absorption_corr_matrix : ndarray
            Absorption correction factors.
        """
    k1 = kf[..., 0]
    k2 = kf[..., 1]
    k3 = kf[..., 2]
    ai = np.deg2rad(ai)
    cos_delta = ne.evaluate('sqrt((k1**2+k2**2)/(k1**2+k2**2+k3**2))')
    delta = ne.evaluate('arccos(cos_delta)') * np.sign(k3)
    ka = ne.evaluate("(1/sin(ai))+(1/sin(delta - ai))")
    absorption_corr_matrix = ne.evaluate("(1 - exp(-sample_attenuation_coeff * sample_thickness * ka))/(sin(ai) * ka)")
    absorption_corr_matrix = ne.evaluate("(1 - exp(-sample_attenuation_coeff * sample_thickness * ka)) / sin(ai) * ka")
    return absorption_corr_matrix


def calc_lorentz_corr_matrix(kf_lab, ai, powder_dim=2):
    """
        Calculates Lorentz correction matrix for 2D or 3D powder diffraction.

        Parameters
        ----------
        kf_lab : ndarray
            Scattered wavevector in lab frame.
        ai : float
            Incident angle in degrees.
        powder_dim : int, optional
            2 for 2D powder, 3 for 3D powder. Default is 2.

        Returns
        -------
        lorentz_corr_matrix : ndarray
            Normalized Lorentz correction factors.
        """
    if powder_dim == 2:
        ai = np.deg2rad(ai)
        R_ai = rotation_matrix(ai, axis='y')
        kf_smpl = kf_lab @ R_ai.T
        k1 = kf_smpl[..., 0]
        k2 = kf_smpl[..., 1]
        k3 = kf_smpl[..., 2]
        sin_gamma_smpl_2 = ne.evaluate('k2**2/(k1**2+k2**2)')
        lorentz_corr_matrix = ne.evaluate('1/sqrt(sin_gamma_smpl_2)')
    elif powder_dim == 3:
        k1 = kf_lab[..., 0]
        k2 = kf_lab[..., 1]
        k3 = kf_lab[..., 2]
        sin_2th = ne.evaluate('sqrt(k2**2+k3**2)/sqrt(k1**2+k2**2+k3**2)')
        lorentz_corr_matrix = ne.evaluate('1 / sin_2th')
    else:
        raise ValueError(f'powder_dim should be 2 or 3, not {powder_dim}')
    norm = np.nanmax(lorentz_corr_matrix)
    lorentz_corr_matrix = ne.evaluate("lorentz_corr_matrix / norm")
    return lorentz_corr_matrix


def generate_q_values(q_min, q_max, q_res, dq):
    """
        Generates a 1D array of q-values with either fixed number of points or fixed step.

        Parameters
        ----------
        q_min : float
            Minimum q-value.
        q_max : float
            Maximum q-value.
        q_res : int or None
            Number of points. If None, use dq.
        dq : float
            Step size if q_res is None.

        Returns
        -------
        ndarray
            1D array of q-values.
        """
    if q_res is not None:
        return np.linspace(q_min, q_max, q_res)
    else:
        return np.arange(np.round(q_min / dq) * dq,
                         np.round(q_max / dq) * dq, dq)


def create_2d_vector_array(q_x, q_y, q_z):
    """
        Creates a 2D array of 3D vectors with given q-components along each axis.

        Parameters
        ----------
        q_x, q_y, q_z : ndarray
            1D arrays of q-components along x, y, z.

        Returns
        -------
        vector_array : ndarray
            3D array of shape (len(q_z), len(q_x), 3) with vectors stacked along last dimension.
        """
    vector_array = np.zeros((len(q_z), len(q_x), 3))

    for i, z in enumerate(q_z):
        vector_array[i, :, 0] = q_x
        vector_array[i, :, 1] = q_y
        vector_array[i, :, 2] = z

    return vector_array


def rotation_matrix(angle, axis='x'):
    """
        Returns a 3x3 rotation matrix for a given angle and axis.

        Parameters
        ----------
        angle : float
            Rotation angle in radians.
        axis : str, optional
            Axis of rotation ('x', 'y', or 'z'). Default is 'x'.

        Returns
        -------
        ndarray
            3x3 rotation matrix.
    """
    if axis == 'x':
        return np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])
    elif axis == 'y':
        return np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
    elif axis == 'z':
        return np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
