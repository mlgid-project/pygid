from . import ExpParams
import numpy as np
import numexpr as ne
from typing import Optional, Any
from dataclasses import dataclass
import pickle
import joblib, os, re
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)

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
        A data class that calculates q- and angular range, corection matices and remaped pixels positions

        Attributes
        ----------
        params : ExpParams, optional
            Experimaental parametres
        ai : float, optional
            Incident angle (in degrees). Takes one value from the list in params. Each matrix should have only one ai,
            otherwise sub_matrices will be created.
        q_max : np.ndarray, optional
            Maximum value for the q-vector. Can be provided or calulates from raw image frame.
        hor_positive : bool, optional
            Flag indicating if only positive values of horizontal axis should be used relative to the direct beam position
            (default is False).
        vert_positive : bool, optional
            Flag indicating if only positive values of vertical axis should be used relative to the direct beam position
            (default is False).
        ang_max : float, optional
            Maximum azimuthal angle (in degrees, CCW from the horizon). Default is 90.
        ang_min : float, optional
            Minimum azimuthal angle (in degrees, default is 0).
        q_xy_range : Any, optional
            Range for the q-values in the xy-plane. Used for GID experments.
        q_z_range : Any, optional
            Range for the q-values in the z-direction. Used for GID experments.
        q_x_range : Any, optional
            Range for the q-values in the x-direction. Used for transmission experments.
        q_y_range : Any, optional
            Range for the q-values in the y-direction. Used for transmission experments.
        dq : float, optional
            Step size in q-space.
        dang : float, optional
            Step size in angle (default is 0.3).
        corr_matrices : CorrMaps, optional
            An instance of CorrMaps class that holds correction matrices. Defaults to None.
        make_pol_corr : bool, optional
            Flag to calculate polarization correction matrix. Defaults to False.
        make_solid_angle_corr : bool, optional
            Flag to calculate solid angle correction matrix. Defaults to False.
        make_air_attenuation_corr : bool, optional
            Flag to calculate air attenuation correction matrix. Defaults to False.
        make_sensor_attenuation_corr : bool, optional
            Flag to calculate sensor attenuation correction matrix. Defaults to False.
        make_absorption_corr : bool, optional
            Flag to calculate absorption correction matrix. Defaults to False.
        make_lorentz_corr : bool, optional
            Flag to calculate Lorentz correction matrix. Defaults to False.
        pol_type : str, optional
            Type of polar correction matrix: 'synchrotron' (default) or 'tube '.
        air_attenuation_coeff : float, optional
            Linear coefficient for air attenuation correction (in 1/m).
        sensor_attenuation_coeff : float, optional
            Linear coefficient for sensor attenuation correction (in 1/m).
        sensor_thickness : float, optional
            Thickness of the detector sensor (in m).
        sample_attenuation_coeff : float, optional
            Linear coefficient for sample attenuation correction (in 1/m).
        sample_thickness : float, optional
            Thickness of the sample (in m).
        powder_dim : float, optional
            Dimension of powder for Lorentz correction: 2 or 3.
        dark_current : np.array, optional
            Array for dark current values, if available.
        flat_field : np.array, optional
            Array for flat field correction values, if available.
        path_to_save : str, optional
            Path where coordinate map will be saved. Path format should be '.pkl'
        path_to_load : str, optional
            Path from which coordinate map will be loaded. Path format should be '.pkl'
        sub_matrices : Any, optional
            Sub-matrices creates automatically if params consist of list of incident angles.

    Example:
        --------
        matrix1 = CoordMaps(params = params, hor_positive = True,  vert_positive = True, dang = 0.1,
                    make_pol_corr= True, make_solid_angle_corr = True)
    """

    params: ExpParams = None
    ai: float = None
    q_max: Optional[np.array] = None
    q_min: Optional[np.array] = None
    img_dim: Optional[np.array] = None
    hor_positive: bool = False
    vert_positive: bool = False
    ang_min: float = None
    ang_max: float = None
    q_xy_range: Any = None
    q_z_range: Any = None
    q_x_range: Any = None
    q_y_range: Any = None
    dq: float = None
    dang: float = 0.3
    corr_matrices: CorrMaps = None
    make_pol_corr: bool = False
    make_solid_angle_corr: bool = False
    make_air_attenuation_corr: bool = False
    make_sensor_attenuation_corr: bool = False
    make_absorption_corr: bool = False
    make_lorentz_corr: bool = False
    pol_type: str = 'synchrotron'
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

    def __post_init__(self):
        if self.path_to_load is not None:
            loaded_attrs = self.load_instance(self.path_to_load).__dict__
            if 'path_to_save' in loaded_attrs:
                del loaded_attrs['path_to_save']
            if 'path_to_load' in loaded_attrs:
                del loaded_attrs['path_to_load']
            self.__dict__.update(loaded_attrs)
            del loaded_attrs
            return
        if self.params is None:
            raise AttributeError('No ExpParams class instance provided.')
        if self.ai is None:
            self.ai = self.params.ai

        if isinstance(self.ai, list):
            attrs = self.__dict__.copy()
            if 'path_to_save' in attrs:
                del attrs['path_to_save']
            if 'path_to_load' in attrs:
                del attrs['path_to_load']

            self.sub_matrices = []
            for angle in self.ai:
                attrs["ai"] = angle
                self.sub_matrices.append(CoordMaps(**attrs))

    def save_instance(self):
        if self.path_to_save is not None:
            joblib.dump(self, self.path_to_save)
            print(f"CoordMaps were saved in {self.path_to_save}")

    @classmethod
    def load_instance(cls, path_to_load):
        return joblib.load(path_to_load)

    def _coordmaps_update_(self):
        self.params._calc_k_()
        if self.dq is None:
            self.dq = self.params._calc_dq_()

        self.img_dim = self.params.img_dim
        self.y = np.arange(self.img_dim[0]) * self.params.px_size - self.params.poni1
        self.x = np.arange(self.img_dim[1]) * self.params.px_size - self.params.poni2

        self._find_ang_ranges_()
        self._find_q_ranges_()
        self._calc_corrs_()





    def _find_q_ranges_(self):
        if self.q_xy_range is None or self.q_z_range is None:
            if not hasattr(self, 'q_lab_from_p'):
                self.q_lab_from_p = self._p_to_q_lab_(calc_type="corner")
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

            self.q_xy_range[0] = self.q_xy_range[0] if not self.hor_positive else np.maximum(self.q_xy_range[0], 0)
            self.q_z_range[0] = self.q_z_range[0] if not self.vert_positive else np.maximum(self.q_z_range[0], 0)

        if self.q_max is None or self.q_min is None:
            if not hasattr(self, 'q_lab_from_p'):
                self.q_lab_from_p = self._p_to_q_lab_(calc_type="corner")
            q_min, q_max = self._find_q_abs_(
                self._q_lab_to_q_smpl_(
                    self.q_lab_from_p,
                    ai=self.ai
                )
            )
            if self.q_min is None:
                self.q_min = q_min
            if self.q_max is None:
                self.q_max = q_max
            self.radial_range = [self.q_min, self.q_max]

    def _find_ang_ranges_(self):
        if self.ang_max is None or self.ang_min is None:
            if not hasattr(self, 'q_lab_from_p'):
                self.q_lab_from_p = self._p_to_q_lab_(calc_type="frame")
            else:
                if len(self.q) in [4, 6, 8]:
                    self.q_lab_from_p = self._p_to_q_lab_(calc_type="frame")

            ang_min, ang_max = self._find_ang_()
            if self.ang_min is None:
                self.ang_min = ang_min
            if self.ang_max is None:
                self.ang_max = ang_max

            if self.hor_positive:
                self.ang_min = max(self.ang_min, -90)
                self.ang_max = min(self.ang_max, 90)
            if self.vert_positive:
                self.ang_min = max(self.ang_min, 0)
                self.ang_max = min(self.ang_max, 180)

            self.angular_range = [ang_max, ang_min]

    def _find_ang_(self):

        q = self.q
        phi = 180 - np.arctan2(q[..., 2], np.sqrt(q[..., 1] ** 2 + q[..., 0] ** 2) * np.sign(q[..., 1]))/np.pi*180
        self.phi = phi
        return np.min(phi), np.max(phi)

    def _p_to_q_lab_(self, calc_type="corner"):
        SDD = self.params.SDD
        d1 = -np.arange(self.img_dim[1]) * self.params.px_size + self.params.poni2
        d2 = np.arange(self.img_dim[0]) * self.params.px_size - self.params.poni1

        self.d1, self.d2 = d1, d2

        Py, Pz = np.meshgrid(d1, d2)

        if calc_type == "corner":

            corner_x = np.array([SDD, SDD, SDD, SDD])
            corner_y = np.array(
                [Py[0, 0], Py[0, -1], Py[-1, -1], Py[-1, 0]])
            corner_z = np.array(
                [Pz[0, 0], Pz[0, -1], Pz[-1, -1], Pz[-1, 0]])

            P = np.stack([corner_x, corner_y, corner_z], axis=1)

            if np.min(d1)*np.max(d1)<0:
                P = np.append(P, [[SDD, 0, Pz[0, 0]]], axis=0)
                P = np.append(P, [[SDD, 0, Pz[0, -1]]], axis=0)
            if np.min(d2)*np.max(d2)<0:
                P = np.append(P, [[SDD, Py[0, 0], 0]], axis=0)
                P = np.append(P, [[SDD, Py[0, -1], 0]], axis=0)

            self.P = P
        elif calc_type == "frame":
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
            Px = SDD * np.ones_like(Pz)
            P = np.stack([Px, Py, Pz], axis=-1)
        R3 = rotation_matrix(-self.params.rot1, axis='z')
        R2 = rotation_matrix(self.params.rot2, axis='y')
        R1 = rotation_matrix(self.params.rot3, axis='x')
        t_reshaped = P @ R3.T @ R2.T @ R1.T
        t = t_reshaped.reshape(P.shape)
        t_abs = np.sqrt(np.sum(t ** 2, axis=-1))
        t_abs = t_abs[..., np.newaxis]
        k = self.params.k
        kf = ne.evaluate("k * t / t_abs")
        q = kf - np.array([k, 0, 0])
        q_abs = np.sqrt(np.sum(q ** 2, axis=-1))
        if calc_type == "full":
            self.kf = kf
            self.cos_2th = ne.evaluate("1 - (q_abs / k) ** 2 / 2")
        self.q = q
        return q

    def _calc_corrs_(self):
        if self.corr_matrices is None:
            self.corr_matrices = CorrMaps()
            self.corr_matrices.flat_field = self.flat_field
            self.corr_matrices.dark_current = self.dark_current

            if (self.make_lorentz_corr or self.make_absorption_corr or self.make_sensor_attenuation_corr or
                    self.make_air_attenuation_corr or self.make_solid_angle_corr or self.make_pol_corr):
                self.q_lab_from_p = self._p_to_q_lab_(calc_type="full")

            if self.make_pol_corr and self.pol_type is not None:
                self.corr_matrices.pol_corr_matrix = calc_pol_corr_matrix(kf=self.kf, pol_type=self.pol_type)

            if self.make_solid_angle_corr:
                self.corr_matrices.solid_angle_corr_matrix = calc_solid_angle_corr_matrix(self.cos_2th)

            if self.make_air_attenuation_corr:
                if self.air_attenuation_coeff is None:
                    print("air_attenuation_coeff was not defined, air_attenuation_corr_matrix was not calculated")
                else:
                    self.corr_matrices.air_attenuation_corr_matrix = calc_air_attenuation_corr_matrix(self.cos_2th,
                                                                                                      self.air_attenuation_coeff,
                                                                                                      self.params.SDD)
            if self.make_sensor_attenuation_corr:
                if self.sensor_attenuation_coeff is None or self.sensor_thickness is None:
                    print(
                        "sensor_attenuation_coeff or sensor_thickness was not defined, sensor_attenuation_corr_matrix was not calculated")
                else:
                    self.corr_matrices.sensor_attenuation_corr_matrix = calc_sensor_attenuation_corr_matrix(
                        self.cos_2th,
                        self.sensor_attenuation_coeff,
                        self.sensor_thickness,
                        self.params.SDD)

            if self.make_absorption_corr:
                if self.sample_attenuation_coeff is None or self.sample_thickness is None:
                    print(
                        "sample_attenuation_coeff or sample_thickness was not defined, absorption_corr_matrix was not calculated")
                else:
                    self.corr_matrices.absorption_corr_matrix = calc_absorption_corr_matrix(self.kf, self.ai,
                                                                                          self.sample_attenuation_coeff,
                                                                                          self.sample_thickness)

            if self.make_lorentz_corr:
                if self.powder_dim is None:
                    print("powder_dim was not defined, lorentz_corr_matrix was not calculated")
                else:
                    self.corr_matrices.lorentz_corr_matrix = calc_lorentz_corr_matrix(self.kf, self.ai, self.powder_dim)


        else:
            self.corr_matrices = CorrMaps()
            if (self.make_lorentz_corr or self.make_absorption_corr):
                self.q_lab_from_p = self._p_to_q_lab_(calc_type="full")

            if self.make_absorption_corr:
                if self.sample_attenuation_coeff is None or self.sample_thickness is None:
                    print(
                        "sample_attenuation_coeff or sample_thickness was not defined, absorption_corr_matrix was not calculated")
                else:
                    self.corr_matrices.absorption_corr_matrix = calc_absorption_corr_matrix(self.kf, self.ai,
                                                                                          self.sample_attenuation_coeff,
                                                                                          self.sample_thickness)
            if self.make_lorentz_corr:
                if self.powder_dim is None:
                    print("powder_dim was not defined, lorentz_corr_matrix was not calculated")
                else:
                    self.corr_matrices.lorentz_corr_matrix = calc_lorentz_corr_matrix(self.kf, self.ai,
                                                                                      self.powder_dim)

    def _calc_recip_giwaxs_(self, q_xy_range=None, q_z_range=None, dq = None):
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

    def _calc_pol_giwaxs_(self, radial_range=None, angular_range=None, dang = None, dq = None):
        radial_range = (self.q_min, self.q_max) if radial_range is None else radial_range
        angular_range = (self.ang_min, self.ang_max) if angular_range is None else angular_range
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

    def _calc_pseudopol_giwaxs_(self, q_gid_rad_range=None, q_gid_azimuth_range = None, dang = None, dq = None):
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

    def _calc_recip_ewald_(self, q_x_range=None, q_y_range=None, dq = None):

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

    def _calc_pol_ewald_(self, radial_range=None, angular_range=None, dang = None, dq = None):

        radial_range = (self.q_min, self.q_max) if radial_range is None else radial_range
        angular_range = (self.ang_min, self.ang_max) if angular_range is None else angular_range
        dang = self.dang if dang is None else dang
        self.dang = dang
        dq = self.dq if dq is None else dq
        self.dq = dq

        self.p_x_lab_pol, self.p_y_lab_pol = self._q_lab_to_p_(
            self._make_q_ewald_polar_grid_(radial_range, dq, angular_range,
                                           dang)
        )

    def _calc_pseudopol_ewald_(self,  q_rad_range=None, q_azimuth_range = None, dang = None, dq = None):

        dang = self.dang if dang is None else dang
        self.dang = dang
        dq = self.dq if dq is None else dq
        self.dq = dq

        self.p_x_lab_pseudopol, self.p_y_lab_pseudopol = self._q_lab_to_p_(
            self._make_q_ewald_pseudopolar_grid_(q_rad_range, q_azimuth_range, self.dq, self.dang)
        )

    def _q_lab_to_q_smpl_(self, q_lab, ai=0):
        ai = np.deg2rad(ai)
        R_ai = rotation_matrix(ai, axis='y')
        q_smpl = q_lab @ R_ai.T
        return q_smpl

    def _q_smpl_to_q_lab_(self, q_smpl, ai=0):
        ai = np.deg2rad(ai)
        R_ai = rotation_matrix(-ai, axis='y')
        q_lab = q_smpl @ R_ai.T
        return q_lab

    def _q_lab_to_p_(self, q_lab):
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
        q_x = q_smpl[..., 0]
        q_y = q_smpl[..., 1]
        q_z = q_smpl[..., 2]
        q_xy_giwaxs = ne.evaluate('-sqrt(q_x ** 2 + q_y ** 2) * q_y / abs(q_y)')
        q_z_giwaxs = q_z
        return q_xy_giwaxs, q_z_giwaxs

    def _q_smpl_to_q_ewald_(self, q_smpl):
        q_y = q_smpl[..., 1]
        q_z = q_smpl[..., 2]
        return q_y, q_z

    def _q_giwaxs_to_q_smpl_(self, q_xy_giwaxs, q_z_giwaxs, ai=0):
        ai = np.deg2rad(ai)
        k = self.params.k
        q_x = ne.evaluate("(-(q_xy_giwaxs**2 + q_z_giwaxs**2) / (2 * k) + q_z_giwaxs * sin(ai)) / cos(ai)")
        q_y = ne.evaluate("-(q_xy_giwaxs / abs(q_xy_giwaxs)) * sqrt(q_xy_giwaxs**2 - q_x**2)")
        q_z = q_z_giwaxs
        q_smpl = np.stack([q_x, q_y, q_z], axis=-1)
        return q_smpl

    def _find_q_abs_(self, q):
        q_max = np.max(np.linalg.norm(q, axis=-1))
        if len(q) >= 8:
            q_min = 0
        else:
            q_min = np.min(np.linalg.norm(q, axis=-1))
        return q_min, np.max(np.linalg.norm(q, axis=-1))

    def _find_ranges_q_giwaxs_(self, q_xy_giwaxs, q_z_giwaxs):
        q_xy_giwaxs_min, q_xy_giwaxs_max = np.nanmin(q_xy_giwaxs), np.nanmax(q_xy_giwaxs)
        q_z_giwaxs_min, q_z_giwaxs_max = np.nanmin(q_z_giwaxs), np.nanmax(q_z_giwaxs)
        return (q_xy_giwaxs_min, q_xy_giwaxs_max), (q_z_giwaxs_min, q_z_giwaxs_max)

    def _find_ranges_q_ewald_(self, q):
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
        q_xy = np.arange(q_xy_range[0], q_xy_range[1], dq)
        q_z = np.arange(q_z_range[0], q_z_range[1], dq)
        self.q_z = q_z
        self.q_xy = q_xy
        Q_xy, Q_z = np.meshgrid(q_xy, q_z)
        return Q_xy, Q_z

    def _make_q_ewald_cart_grid_(self, q_x_range=(0, 5), q_y_range=(0, 5), dq=0.01):

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
        q_pol = np.arange(q_range[0], q_range[1], dq)
        ang_pol = np.arange(ang_range[0], ang_range[1], dang)
        self.q_gid_pol = q_pol
        self.ang_gid_pol = ang_pol

        ang_pol_rad = np.deg2rad(ang_pol)
        Q_pol, ANG_pol = np.meshgrid(q_pol, ang_pol_rad)
        Q_xy, Q_z = Q_pol * np.cos(ANG_pol), Q_pol * np.sin(ANG_pol)
        return Q_xy, Q_z

    def _make_q_ewald_polar_grid_(self, q_range=[0, 4], dq=0.01, ang_range=[0, 90], dang=0.01):
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

    def _make_q_giwaxs_pseudopolar_grid_(self, q_gid_rad_range = None, q_gid_azimuth_range = None, dq=0.01, dang=0.01):

        q_rad = np.arange(self.q_min, self.q_max, dq)
        ang_range = [self.ang_min, self.ang_max]
        if not hasattr(self,'phi'):
            self._find_ang_()
        q = self.q
        q_abs = np.sqrt(q[..., 1] ** 2 + q[..., 0] ** 2 + q[..., 2] ** 2)
        phi = np.arctan2(q[..., 2], np.sqrt(q[..., 1] ** 2 + q[..., 0] ** 2) * np.sign(-q[..., 1]))
        phi[phi > np.radians(self.ang_max)] = np.nan
        phi[phi < np.radians(self.ang_min)] = np.nan
        self.q_rad, self.phi = q_rad, phi
        q_phi = q_abs * phi
        if q_gid_rad_range is not None:
            q_rad = np.arange(q_gid_rad_range[0], q_gid_rad_range[1], dq)

        if q_gid_azimuth_range is not None:
            q_azimuth = np.linspace(q_gid_azimuth_range[0], q_gid_azimuth_range[1], int((ang_range[1] - ang_range[0]) / dang))
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

    def _make_q_ewald_pseudopolar_grid_(self, q_rad_range = [0, 4], q_azimuth_range=[0, 90], dq=0.01, dang=0.01):
        if len(self.q) in [4,6,8]:
            self._p_to_q_lab_(calc_type="frame")

        q = self.q
        q_abs = np.linalg.norm(q, axis=-1)
        q_rad = np.arange(self.q_min, self.q_max, dq)
        ang_range = [self.ang_min, self.ang_max]
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
    k1 = kf[..., 0]
    k2 = kf[..., 1]
    k3 = kf[..., 2]
    cos_gamma_2 = ne.evaluate('k1**2/(k1**2+k2**2)')
    cos_delta_2 = ne.evaluate('(k1**2+k2**2)/(k1**2+k2**2+k3**2)')
    if pol_type == 'synchrotron':
        pol_corr_matrix_hor = ne.evaluate('1 - cos_delta_2 * (1 - cos_gamma_2)')
        pol_corr_matrix_vert = ne.evaluate('cos_delta_2')
        pol_corr_matrix = ne.evaluate('1 / (pol_corr_matrix_hor*0.98 + pol_corr_matrix_vert*0.02)')
    elif pol_type == 'tube':
        pol_corr_matrix = ne.evaluate('2 / (1 + cos_gamma_2 * cos_delta_2)')
    else:
        raise ValueError(f'pol_type should be "synchrotron" or "tube", not {pol_type}')
    norm = np.nanmax(pol_corr_matrix)
    pol_corr_matrix = ne.evaluate("pol_corr_matrix / norm")
    return pol_corr_matrix


def calc_solid_angle_corr_matrix(cos_2th):
    solid_angle_corr_matrix = ne.evaluate("cos_2th**3")
    return solid_angle_corr_matrix


def calc_air_attenuation_corr_matrix(cos_2th, air_attenuation_coeff, SDD):
    air_attenuation_corr_matrix = ne.evaluate("exp(- air_attenuation_coeff * SDD / cos_2th)")
    norm = np.nanmax(air_attenuation_corr_matrix)
    air_attenuation_corr_matrix = ne.evaluate("air_attenuation_corr_matrix / norm")
    return air_attenuation_corr_matrix


def calc_sensor_attenuation_corr_matrix(cos_2th, sensor_attenuation_coeff, sensor_thickness, SDD):
    sensor_attenuation_corr_matrix = ne.evaluate("1 - exp(-sensor_attenuation_coeff * sensor_thickness / cos_2th)")
    norm = np.nanmax(sensor_attenuation_corr_matrix)
    sensor_attenuation_corr_matrix = ne.evaluate("sensor_attenuation_corr_matrix / norm")
    return sensor_attenuation_corr_matrix


def calc_absorption_corr_matrix(kf, ai, sample_attenuation_coeff, sample_thickness):
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
    if q_res is not None:
        return np.linspace(q_min, q_max, q_res)
    else:
        return np.arange(np.round(q_min / dq) * dq,
                         np.round(q_max / dq) * dq, dq)


def create_2d_vector_array(q_x, q_y, q_z):
    vector_array = np.zeros((len(q_z), len(q_x), 3))

    for i, z in enumerate(q_z):
        vector_array[i, :, 0] = q_x
        vector_array[i, :, 1] = q_y
        vector_array[i, :, 2] = z

    return vector_array


def rotation_matrix(angle, axis='x'):
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
