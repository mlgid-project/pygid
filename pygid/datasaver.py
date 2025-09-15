from __future__ import annotations
from typing import TYPE_CHECKING
import os
import yaml
from dataclasses import dataclass
from typing import Any
import numpy as np
import h5py
import re
from datetime import datetime
import warnings
# from silx.io.h5py_utils import open_item

if TYPE_CHECKING:
    from . import Conversion


class ExpMetadata:
    """
    The class to store sample and experment metadata.

    Attributes
    ----------
    All attributes are variable that will be saved in sample gruop in h5-file
    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __setattr__(self, name, value):
        self.__dict__[name] = value

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"


class SampleMetadata:
    def __init__(self, *, path_to_save=None, path_to_load=None, data=None):
        self.path_to_save = path_to_save
        self.path_to_load = path_to_load
        self.data = data or {}

        if self.path_to_load:
            self.load(self.path_to_load)
        if self.path_to_save:
            self.save(self.path_to_save)

    def __repr__(self):
        return f"{self.__class__.__name__}(data={self.data})"

    def save(self, filepath=None):
        filepath = filepath or self.path_to_save
        if filepath is None:
            raise ValueError("Filepath is not defined for saving.")

        ext = os.path.splitext(filepath)[1].lower()

        if ext in [".yaml", ".yml"]:
            with open(filepath, 'w') as f:
                yaml.dump({"data": self.data}, f, sort_keys=False, default_flow_style=False)
        else:
            with open(filepath, 'w') as f:
                for key, value in self.data.items():
                    f.write(f"{key}={value}\n")

        print("Saved sample metadata to", filepath)

    def load(self, filepath=None):
        filepath = filepath or self.path_to_load
        if filepath is None:
            raise ValueError("Filepath is not defined for loading.")

        ext = os.path.splitext(filepath)[1].lower()

        if ext in [".yaml", ".yml"]:
            with open(filepath, 'r') as f:
                content = yaml.safe_load(f)
                self.data = content.get("data", {})
        else:
            with open(filepath, 'r') as f:
                self.data = {}
                for line in f:
                    if '=' in line:
                        key, value = line.strip().split('=', 1)
                        self.data[key] = self._parse_value(value)

    def _parse_value(self, value):
        try:
            return eval(value, {"__builtins__": {}})
        except Exception:
            return value


@dataclass
class DataSaver:
    """
    A class to save processed data.

    Attributes:
    -----------
    sample : Conversion
        An instance of the `Conversion` class, representing the sample data to be saved.
    path_to_save : str, optional
        The path inclufing file name where the data should be saved. Default is None. The file format should be .h5
    h5_group : str, optional
        The specific group in the HDF5 file where the data will be stored. Default is {file name}_0000.
    overwrite_file : bool, optional
        Whether to overwrite an existing file. Default is True.
    metadata : Metadata, optional
        'Metadata' class instance related to the samples or experiment. Default is None.

    """
    sample: Conversion
    path_to_save: str = None
    h5_group: str = None
    overwrite_file: bool = True
    overwrite_group: bool = False
    exp_metadata: ExpMetadata = None
    smpl_metadata: SampleMetadata = None
    img_container_detect: Any = None
    img_container_fit: Any = None
    matched_data: Any = None
    unit_cell_data: Any = None

    def __post_init__(self):
        self.matrix = self.sample.matrix[0]
        self.params = self.matrix.params

        if hasattr(self.sample, 'path'):
            self.original_path = self.sample.path

        if hasattr(self.sample, 'ai_list'):
            self.ai_list = self.sample.ai_list
        else:
            raise ValueError("conversion process is not correct. ai_list was not calculated")

        if hasattr(self.sample, 'converted_frame_num'):
            self.frame_num = self.sample.converted_frame_num
        else:
            raise ValueError("conversion process is not correct. converted_frame_num was not calculated")

        if self.path_to_save is None:
            self.path_to_save = "result.h5"
        if self.h5_group is None:
            self.h5_group = os.path.basename(os.path.splitext(self.path_to_save)[0])
            self.h5_group = modify_string('entry', first_modification=True)

        keys = [
            "img_gid_q", "img_q", "img_gid_pol",
            "img_pol", "img_gid_pseudopol", "img_pseudopol",
            "rad_cut_gid", "azim_cut_gid", "horiz_cut_gid",
            "rad_cut", "azim_cut"
        ]

        name, data = next(
            ((key, getattr(self.sample, key)) for key in keys if hasattr(self.sample, key)),
            (None, None)
        )

        self._save_data_(name, data)
        delattr(self.sample, name)

    def _save_data_(self, name, data):
        """
        Saves converted images to HDF5 file.
        """
        folder_to_save = os.path.dirname(self.path_to_save)
        file_name = self.path_to_save

        if not os.path.exists(folder_to_save) and folder_to_save != "":
            os.makedirs(folder_to_save)

        if os.path.exists(file_name) and self.overwrite_file:
            mode = 'w'
        else:
            mode = 'a'
        with h5py.File(file_name, mode) as root:
        # with open_item(file_name, "/", mode=mode, retry_timeout=1200, retry_period=0.1) as root:
            root.attrs["NX_class"] = "NXroot"
            if not self.overwrite_group and self.h5_group in root:
                root.attrs["NX_class"] = "NXroot"
                new_h5_group = change_h5_group(root, self.h5_group, name, data[0].shape)
                if self.h5_group != new_h5_group:
                    warnings.warn(
                        f"Image shape or type mismatch: cannot append to dataset in {self.h5_group}. Using new group {new_h5_group}.",
                        category=UserWarning
                    )
                    self.h5_group = new_h5_group

            else:
                if self.h5_group in root:
                    del root[self.h5_group]

            if self.h5_group not in root:
                _make_groups_(root, self.h5_group)
                save_single_data(root[f"/{self.h5_group}"], 'definition', 'NXsas')
                save_single_data(root[f"/{self.h5_group}"], 'title', str(self.h5_group))
                save_expparams(root, self.h5_group, self.params)
            save_single_data(root[f"/{self.h5_group}/instrument"], 'angle_of_incidence', self.ai_list, extend_list=True)
            save_single_data(root[f"/{self.h5_group}/data"], 'frame_num', self.frame_num, extend_list=True)

            if self.exp_metadata is None:
                self.exp_metadata = ExpMetadata(filename=self.original_path)
            if not hasattr(self.exp_metadata, "filename"):
                self.exp_metadata.filename = self.original_path
            save_exp_metadata(root, self.exp_metadata, self.h5_group)

            if self.smpl_metadata is not None:
                save_smpl_metadata(root, self.smpl_metadata, self.h5_group)

            if data is not None:
                create_dataset(root, self.h5_group, name, data)

            save_matrix(root, self.h5_group, self.matrix, name)
            fill_process_group(root, self.h5_group, self.matrix)
            fill_analysis_group(root, self.h5_group, len(data), self.img_container_detect, self.img_container_fit,
                                self.matched_data, self.unit_cell_data)
            print(f"Saved in {self.path_to_save} in group {self.h5_group}")
        return


def save_matrix(root, h5_group, matrix, img_name):
    """
    Saves coordinate maps data to HDF5 file.

    Parameters
    ----------
    root : h5py.File
        The root object of the opened HDF5 file where the matrix will be saved.
    h5_group : str
        The name of the group within the HDF5 file under which the matrix data will be stored.
    matrix : numpy.ndarray
        The matrix data to be saved.
    """
    keys_map = {
        "img_gid_q": ["q_xy", "q_z"],
        "img_q": ["q_x", "q_y"],
        "img_gid_pol": ["q_gid_pol", "ang_gid_pol"],
        "img_pol": ["q_pol", "ang_pol"],
        "img_gid_pseudopol": ["q_gid_rad", "q_gid_azimuth"],
        "img_pseudopol": ["q_rad", "q_azimuth"],
        "rad_cut": ["q_pol"],
        "rad_cut_gid": ["q_gid_pol"],
        "azim_cut": ["ang_pol"],
        "azim_cut_gid": ["ang_gid_pol"],
        "horiz_cut_gid": ["q_xy"]
    }

    keys = keys_map.get(img_name, [])

    coords_dict = {key: getattr(matrix, key) for key in keys if hasattr(matrix, key)}
    keys = list(coords_dict.keys())
    # print("keys", "keys")
    for name in coords_dict:
        data = coords_dict[name]
        save_single_data(root[f"{h5_group}/data"], name,
                         np.array(data, dtype=np.float64), attrs={'interpretation': 'axis', 'units': '1/Angstrom'})
    if len(keys) == 2:
        root[f"{h5_group}/data"].attrs.update({'signal': img_name, 'axes': ["frame_num", keys[1], keys[0]]})
    else:
        root[f"{h5_group}/data"].attrs.update({'signal': img_name, 'axes': ["frame_num", keys[0]]})


def modify_string(s, first_modification=True):
    """
    Modifies the input string by appending or incrementing a 4-digit numerical suffix. The number is incremented by 1.
    Parameters
    ----------
    s : str
        The input string to be modified.
    first_modification : bool, optional
        A flag indicating whether this is the first modification attempt. If True,
        the string is returned unchanged even if it ends with a 4-digit number.
    """
    match = re.search(r'(\d{4})$', s)
    if match:
        if first_modification:
            return s
        number = int(match.group(1)) + 1
        return s[:-4] + f"{number:04d}"
    else:
        return s + "_0000"


def read_dataset_size(root, h5_group):
    """
        Reads the shape of specified datasets within a given HDF5 group.

        Parameters
        ----------
        root : h5py.File or h5py.Group
            The root object of the opened HDF5 file.
        h5_group : str
            The name of the group within the HDF5 file containing the datasets.
        names : list of str
            The names of datasets whose shapes are to be read.
    """

    datashape_dict = {}

    keys = [
        "img_gid_q", "img_q", "img_gid_pol",
        "img_pol", "img_gid_pseudopol", "img_pseudopol",
        "rad_cut", "azim_cut",
        "rad_cut_gid", "azim_cut_gid", "horiz_cut_gid",

    ]

    for key in keys:
        dataset_name = f'/{h5_group}/data/{key}'
        if dataset_name in root:
            datashape_dict[key] = root[dataset_name][0].shape
    return datashape_dict


def change_h5_group(root, h5_group, name, data_shape):
    """
       Checks if the shapes of the datasets in the given HDF5 group match the shapes of the provided images.
       If there is a mismatch, modifies the group name and recursively checks the next group.

       Parameters
       ----------
       root : h5py.File or h5py.Group
           The root object of the opened HDF5 file.
       h5_group : str
           The name of the group within the HDF5 file to check.
       images : dict
           conveted images to be saved
    """

    data_old = read_dataset_size(root, h5_group)
    change = any(
        key != name or data_shape_old != data_shape
        for key, data_shape_old in data_old.items()
    )
    if change:
        new_h5_group = modify_string(h5_group, first_modification=False)
        return change_h5_group(root, new_h5_group, name, data_shape)
    return h5_group


def create_dataset(root, h5_group, name, data):
    """
    Creates a dataset in the specified HDF5 group and writes the provided data.

    Parameters
    ----------
    root : h5py.File
        The root or group object of the HDF5 file.
    h5_group : str
        The name of the group in which to create the dataset.
    name : str
        The name of the dataset to create.
    data : array-like
        The data to be stored in the dataset.
    """

    dataset_name = f'/{h5_group}/data/{name}'
    data = np.array(data)

    if dataset_name in root:
        dataset = root[dataset_name]
        if dataset.chunks is None:
            raise TypeError(f"The dataset '{dataset_name}' must be chunked to allow resizing.")

        current_size = dataset.shape[0]
        new_size = current_size + data.shape[0]
        dataset.resize(new_size, axis=0)
        dataset[current_size:new_size] = data

    else:
        maxshape = (None,) + data.shape[1:]
        print()
        root[f'/{h5_group}/data'].create_dataset(
            name=name,
            data=data,
            maxshape=maxshape,
            chunks=True)


def ensure_group_exists(root, group_name, attrs=None):
    """
        Ensures that the specified group exists in the HDF5 file. If the group does not exist, it is created.
        Optionally updates the group's attributes.

        Parameters
        ----------
        root : h5py.File or h5py.Group
            The root or parent group of the HDF5 file.
        group_name : str
            The name of the group to check or create.
        attrs : dict, optional
            A dictionary of attributes to assign to the group if it is created.
        """
    if group_name not in root:
        root.create_group(group_name)
        if attrs:
            root[group_name].attrs.update(attrs)


def _make_groups_(root, h5_group="entry"):
    """
    Creates required groups in the HDF5 file under the specified base group.

    Parameters
    ----------
    root : h5py.File
        The root or parent group of the HDF5 file.
    h5_group : str, optional
        The name of the top-level group under which subgroups will be created.
    """

    root.attrs["default"] = h5_group
    root.create_group(f'/{h5_group}').attrs.update({"NX_class": "NXentry", "EX_required": "true", "default": "data"})
    ensure_group_exists(root, f'/{h5_group}/instrument', {'NX_class': 'NXinstrument', 'EX_required': 'true'})
    ensure_group_exists(root, f'/{h5_group}/instrument/source', {'NX_class': 'NXsource', 'EX_required': 'true'})
    ensure_group_exists(root, f'/{h5_group}/instrument/monochromator',
                        {'NX_class': 'NXmonochromator', 'EX_required': 'true'})
    ensure_group_exists(root, f'/{h5_group}/instrument/detector',
                        {'NX_class': 'NXdetector', 'EX_required': 'true'})
    ensure_group_exists(root, f'/{h5_group}/sample', {'NX_class': 'NXsample', 'EX_required': 'true'})
    ensure_group_exists(root, f'/{h5_group}/data',
                        {'NX_class': 'NXdata', 'EX_required': 'true', 'signal': 'img_gid_q'})
    ensure_group_exists(root, f'/{h5_group}/process', {'NX_class': 'NXprocess', 'EX_required': 'true'})


def save_single_data(root, dataset_name, data, extend_list=False, attrs=None):
    """
    Saves a single dataset to the specified location in the HDF5 file.

    Parameters
    ----------
    root : h5py.File
        The root or parent group of the HDF5 file.
    dataset_name : str
        Full path to the dataset within the HDF5 file.
    data : str or numeric
        Data to be stored in the dataset.
    type : str, optional
        NeXus class attribute to assign to the dataset. Default is 'NX_CHAR'.
    """
    if attrs is None:
        attrs = {'type': 'NX_CHAR'}
    if data is not None:
        if dataset_name in root:
            if not extend_list:
                del root[dataset_name]
            else:
                try:
                    old_data = root[dataset_name][()].decode('utf-8')
                except:
                    old_data = root[dataset_name][()]
                del root[dataset_name]

                if isinstance(old_data, np.ndarray):
                    old_data = old_data.tolist()
                old_data = [old_data] if not isinstance(old_data, list) else old_data
                if isinstance(old_data, list):
                    if isinstance(old_data[0], bytes):
                        old_data = [item.decode('utf-8') for item in old_data]
                if isinstance(data, list) or isinstance(data, np.ndarray):
                    for i in data:
                        old_data.append(i)
                else:
                    old_data.append(data)
                data = old_data

        root.create_dataset(
            name=dataset_name, data=data, maxshape=None,
        ).attrs.update(attrs)


def save_single_metadata(root, metadata, dataset_name, data_name, nx_type="NX_CHAR", required=False, extend_list=False):
    """
    Saves a single metadata entry to the specified dataset location in an HDF5 file.

    Parameters
    ----------
    root : h5py.File
        The root or parent group of the HDF5 file.
    metadata : Metadata
        Metadata class instance containing metadata values.
    dataset_name : str
        Full path to the group or dataset where metadata should be stored.
    data_name : str
        Name of the metadata field to save.
    nx_type : str, optional
        NeXus type for the attribute. Default is "NX_CHAR".
    required : bool, optional
        If True and the key is missing in metadata, an empty dataset will be created.
    """
    if hasattr(metadata, data_name) or required:
        if hasattr(metadata, data_name):
            data = getattr(metadata, data_name)
        else:
            data = str(data_name)
        if data_name in root and not extend_list:
            return
        if data is not None:
            if dataset_name in root:
                if not extend_list:
                    del root[dataset_name]
                else:
                    try:
                        old_data = root[dataset_name][()].decode('utf-8')
                    except:
                        old_data = root[dataset_name][()]
                    del root[dataset_name]

                    if isinstance(old_data, np.ndarray):
                        old_data = old_data.tolist()
                    old_data = [old_data] if not isinstance(old_data, list) else old_data
                    if isinstance(old_data, list):
                        if isinstance(old_data[0], bytes):
                            old_data = [item.decode('utf-8') for item in old_data]
                    if isinstance(data, list) or isinstance(data, np.ndarray):
                        for i in data:
                            old_data.append(i)
                    else:
                        old_data.append(data)
                    data = old_data

            root.create_dataset(
                name=dataset_name, data=data, maxshape=None,
            ).attrs.update({'EX_required': 'true'})


def save_exp_metadata(root, exp_metadata=None, h5_group="entry"):
    """
    Saves multiple metadata entries to the specified group in an HDF5 file.

    Parameters
    ----------
    root : h5py.File
        The root or parent group of the HDF5 file.
    metadata : dict, optional
         Metadata class instance containing metadata values.
    h5_group : str, optional
        The name of the group within the HDF5 file where metadata will be stored. Default is "entry".
    """

    if h5_group + '/instrument/source' not in root:
        root.require_group(f"/{h5_group}/instrument/source")

    if not hasattr(exp_metadata, "source_probe"):
        exp_metadata.source_probe = "x-ray"

    save_single_metadata(root[f"/{h5_group}/instrument"], exp_metadata, 'name', 'instrument_name', required=True)
    save_single_metadata(root[f"/{h5_group}/instrument/source"], exp_metadata, 'type', 'source_type', required=True)
    save_single_metadata(root[f"/{h5_group}/instrument/source"], exp_metadata, 'name', 'source_name', required=False)
    save_single_metadata(root[f"/{h5_group}/instrument/source"], exp_metadata, 'probe', 'source_probe', required=True)
    save_single_metadata(root[f"/{h5_group}/instrument/monochromator"], exp_metadata, 'wavelength_spread',
                         'wavelength_spread', required=False)
    save_single_metadata(root[f"/{h5_group}/instrument/detector"], exp_metadata, 'name', 'detector_name',
                         required=False)
    save_single_metadata(root[f"/{h5_group}"], exp_metadata, 'start_time', 'start_time', "NX_DATE_TIME", required=True)
    save_single_metadata(root[f"/{h5_group}"], exp_metadata, 'end_time', 'end_time', "NX_DATE_TIME", required=True)
    save_single_metadata(root[f"/{h5_group}/data"], exp_metadata, 'filename', 'filename', required=False,
                         extend_list=True)
    # add frame_idx, epoch, mon, watt0, transmission here:
    save_single_metadata(root[f"/{h5_group}/instrument"], exp_metadata, 'frame_idx', 'frame_idx', required=False,
                         extend_list=True)
    save_single_metadata(root[f"/{h5_group}/instrument"], exp_metadata, 'epoch', 'epoch', required=False,
                         extend_list=True)
    save_single_metadata(root[f"/{h5_group}/instrument"], exp_metadata, 'mon', 'mon', required=False,
                         extend_list=True)
    save_single_metadata(root[f"/{h5_group}/instrument"], exp_metadata, 'watt0', 'watt0', required=False,
                         extend_list=True)
    save_single_metadata(root[f"/{h5_group}/instrument"], exp_metadata, 'transmission', 'transmission', required=False,
                         extend_list=True)

    saved_attr = ['instrument_name', 'source_type', 'source_probe', 'source_name', 'wavelength_spread',
                  'source_name', 'start_time', 'end_time', 'detector_name', 'detector', 'source',
                  'filename', 'frame_idx', 'epoch', 'mon']
    for attr_name in exp_metadata.__dict__:
        if attr_name not in saved_attr:
            save_single_metadata(root[f"/{h5_group}/instrument"], exp_metadata, attr_name, attr_name, extend_list=False)


def save_smpl_metadata(root, smpl_metadata=None, h5_group="entry"):
    if smpl_metadata is None or not isinstance(smpl_metadata, SampleMetadata):
        raise ValueError("Valid SampleMetadata instance must be provided.")

    h5_path = f"/{h5_group}/sample"
    group = root.require_group(h5_path)

    def write_dict_to_group(h5grp, data):
        for key, value in data.items():
            if isinstance(value, dict):
                subgrp = h5grp.require_group(key)
                write_dict_to_group(subgrp, value)
            else:
                if key in h5grp:
                    del h5grp[key]
                try:
                    h5grp.create_dataset(key, data=value)
                except TypeError:
                    h5grp.create_dataset(key, data=str(value))

    write_dict_to_group(group, smpl_metadata.data)
    if "name" not in root[h5_path]:
        warnings.warn("SampleMetadata does not contain a 'name' attribute.")


def save_expparams(root, h5_group, params):
    """
        Saves experimental parameters to a specified group in an HDF5 file.

        Parameters
        ----------
        root : h5py.File
            The root or parent group of the HDF5 file.
        h5_group : str
            The name of the group within the HDF5 file where the experimental parameters will be stored.
        params : dict
            ExpParams class instance containing the experimental parameters to be saved.
        """
    save_single_data(root[f"/{h5_group}/instrument/monochromator"], 'wavelength', params.wavelength * 1e-10,
                     attrs={'type': 'NX_FLOAT',
                            'units': 'm'})
    save_single_data(root[f"/{h5_group}/instrument/detector"], 'distance', params.SDD,
                     attrs={'type': 'NX_FLOAT',
                            'units': 'm'})
    save_single_data(root[f"/{h5_group}/instrument/detector"], 'x_pixel_size', params.px_size,
                     attrs={'type': 'NX_FLOAT',
                            'units': 'm'}
                     )
    save_single_data(root[f"/{h5_group}/instrument/detector"], 'y_pixel_size', params.px_size,
                     attrs={'type': 'NX_FLOAT',
                            'units': 'm'}
                     )
    save_single_data(root[f"/{h5_group}/instrument/detector"], 'polar_angle', -params.rot2,
                     attrs={'type': 'NX_ANGLE',
                            'units': 'rad',
                            'description': '-rot2'}
                     )
    save_single_data(root[f"/{h5_group}/instrument/detector"], 'rotation_angle', params.rot3,
                     attrs={'type': 'NX_ANGLE',
                            'units': 'rad',
                            'description': 'rot3'}
                     )
    save_single_data(root[f"/{h5_group}/instrument/detector"], 'aequatorial_angle', params.rot1,
                     attrs={'type': 'NX_ANGLE',
                            'units': 'rad',
                            'description': 'rot1'}
                     )
    save_single_data(root[f"/{h5_group}/instrument/detector"], 'beam_center_x', params.centerX,
                     attrs={'type': 'NX_FLOAT',
                            'units': 'm'}
                     )
    save_single_data(root[f"/{h5_group}/instrument/detector"], 'beam_center_y', params.centerY,
                     attrs={'type': 'NX_FLOAT',
                            'units': 'm'}
                     )


def check_correction(corr_matrices, attr):
    if hasattr(corr_matrices, attr):
        if getattr(corr_matrices, attr) is not None:
            return True
    return False


def fill_process_group(root, h5_group, matrix):
    corr_matrices = matrix.corr_matrices
    h5_group = "/" + h5_group
    group = root[h5_group + '/process']
    save_single_data(group, 'program', "pygid", extend_list=False)
    from . import __version__ as pygid_version
    save_single_data(group, 'version', pygid_version, extend_list=False)
    save_single_data(group, 'date', datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f'), extend_list=False)
    NOTE = (
        "Intensity corrections applied:\n"
        f"  - dark_current: {check_correction(corr_matrices, 'dark_current')}\n"
        f"  - flat_field: {check_correction(corr_matrices, 'flat_field')}\n"
        f"  - solid_angle_corr: {check_correction(corr_matrices, 'solid_angle_corr_matrix')}\n"
        f"  - pol_corr: {check_correction(corr_matrices, 'pol_corr_matrix')} "
        f"(pol_type = {matrix.pol_type})\n"
        f"  - air_attenuation_corr: {check_correction(corr_matrices, 'air_attenuation_corr_matrix')} "
        f"(air_attenuation_coeff = {matrix.air_attenuation_coeff} m-1)\n"
        f"  - sensor_attenuation_corr: {check_correction(corr_matrices, 'sensor_attenuation_corr_matrix')} "
        f"(sensor_thickness = {matrix.sensor_thickness} m, sensor_attenuation_coeff = {matrix.sensor_attenuation_coeff} m-1)\n"
        f"  - absorption_corr: {check_correction(corr_matrices, 'absorption_corr_matrix')} "
        f"(sample_thickness = {matrix.sample_thickness} m, sample_attenuation_coeff = {matrix.sample_attenuation_coeff} m-1)\n"
        f"  - lorentz_corr: {check_correction(corr_matrices, 'lorentz_corr_matrix')} "
        f"(powder_dim = {matrix.powder_dim})\n"
    )

    save_single_data(group, 'NOTE', NOTE, extend_list=False)


def fill_analysis_group(root, h5_group, img_number_to_add, img_container_detect, img_container_fit, matched_data,
                        unit_cell_data):
    """
        Creates analysis-related fields in a specified group within an HDF5 file.

        This function ensures that the necessary fields for analysis are present in the HDF5 file.
        It will create new datasets or groups as needed within the provided group.

        Parameters
        ----------
        root : h5py.File
            The root or parent group of the HDF5 file.
        h5_group : str
            The name of the group within the HDF5 file where the analysis fields will be created.
        """
    analysis_path = f"/{h5_group}/data/analysis"

    if img_container_detect is not None:
        if not isinstance(img_container_detect, list):
            img_container_detect = [img_container_detect]
    if img_container_fit is not None:
        if not isinstance(img_container_fit, list):
            img_container_fit = [img_container_fit]
    if matched_data is not None:
        if not isinstance(matched_data, list):
            matched_data = [matched_data]

    if unit_cell_data is not None:
        if not isinstance(unit_cell_data, list):
            unit_cell_data = [unit_cell_data]

    if analysis_path not in root:
        root.create_group(analysis_path)
        root[analysis_path].attrs.update({'NX_class': 'NXparameters', 'EX_required': 'true'})
    group = root[analysis_path]
    subgroups = [name for name in group if isinstance(group[name], h5py.Group)]
    img_number_current = len(subgroups)
    for i in range(img_number_current, img_number_current + img_number_to_add):
        group_name = f"/{h5_group}/data/analysis/frame{str(i).zfill(5)}"
        root.create_group(group_name)
        root[group_name].attrs.update({'NX_class': 'NXparameters', 'EX_required': 'true'})
        if img_container_detect is not None and i - img_number_current < len(img_container_detect):
            _save_img_container_detect(root, group_name, img_container_detect[i - img_number_current])
        if img_container_fit is not None and i - img_number_current < len(img_container_fit):
            _save_img_container_fit(root, group_name, img_container_fit[i - img_number_current])
        if matched_data is not None and i - img_number_current < len(matched_data):
            _save_matched_data(root, group_name, matched_data[i - img_number_current])
        if unit_cell_data is not None and i - img_number_current < len(unit_cell_data):
            _save_unit_cell_data(root, group_name, unit_cell_data[i - img_number_current])
        # root.create_group(group_name + "/detected_peaks")
        # root[f"{group_name}/detected_peaks"].attrs.update({'NX_class': 'NXdata', 'EX_required': 'true'})
        # root.create_group(f"{group_name}/fitted_peaks")
        # root[f"{group_name}/fitted_peaks"].attrs.update({'NX_class': 'NXdata', 'EX_required': 'true'})


unit_cell_data_dtype = np.dtype([
    ('h', 'i4'),
    ('k', 'i4'),
    ('l', 'i4'),
    ('a', 'f4'),
    ('b', 'f4'),
    ('c', 'f4'),
    ('alpha', 'f4'),
    ('beta', 'f4'),
    ('gamma', 'f4'),
    ('a_err', 'f4'),
    ('b_err', 'f4'),
    ('c_err', 'f4'),
    ('alpha_err', 'f4'),
    ('beta_err', 'f4'),
    ('gamma_err', 'f4'),
    ('q_loss', 'f4'),
    ('is_converged', 'bool'),
    ('score', 'f4'),
])


def _save_unit_cell_data(root, group_name, unit_cell_data):
    group = root[group_name]

    orientation_array, unit_cells, unit_cell_errors, q_losses, is_converged, score = unit_cell_data
    results_array = np.zeros(orientation_array.shape[0], dtype=unit_cell_data_dtype)
    results_array['h'] = orientation_array[:, 0]
    results_array['k'] = orientation_array[:, 1]
    results_array['l'] = orientation_array[:, 2]

    results_array['a'] = unit_cells[:, 0]
    results_array['b'] = unit_cells[:, 1]
    results_array['c'] = unit_cells[:, 2]
    results_array['alpha'] = unit_cells[:, 3]
    results_array['beta'] = unit_cells[:, 4]
    results_array['gamma'] = unit_cells[:, 5]

    results_array['a_err'] = unit_cell_errors[:, 0]
    results_array['b_err'] = unit_cell_errors[:, 1]
    results_array['c_err'] = unit_cell_errors[:, 2]
    results_array['alpha_err'] = unit_cell_errors[:, 3]
    results_array['beta_err'] = unit_cell_errors[:, 4]
    results_array['gamma_err'] = unit_cell_errors[:, 5]

    results_array['q_loss'] = q_losses
    results_array['is_converged'] = is_converged
    results_array['score'] = score
    if f"unit_cell_data" in group:
        del group[f"unit_cell_data"]
    group.create_dataset(f"unit_cell_data", data=results_array, dtype=unit_cell_data_dtype)


def _save_matched_data(root, group_name, matched_data):
    group = root[group_name]
    data_matched, unique_solutions = matched_data
    key = list(unique_solutions.keys())[0]
    for i, sol in enumerate(unique_solutions[key]):
        indices, names = zip(*[(el[0], el[1] + ' ' + str(el[2])) for el in sol])
        # indices, names = zip(*list({(el[0], el[1] + ' ' + str(el[2])) for el in sol}))
        print(f"Names = {names}")
        dtype_descr = list({(n, np.float32) for n in names})
        # dtype_descr = [(n, np.float32) for n in names]
        print(dtype_descr)
        try:
            results_array = np.zeros(len(data_matched[key]['peaks']), dtype=np.dtype(dtype_descr))
        except:
            print(f"\n\n\n\nError is in line 752")
            print(f"Names error = {names}")
            print(f"Error sol = {sol}")
            print(f"unique solutions = {unique_solutions}")
            print(f"error data matched = {data_matched}\n\n\n\n")
            raise
        cur_data = data_matched[key]
        for idx, name in zip(indices, names):
            cur_data = cur_data[idx]
            results_array[name][cur_data['indices_real_matched_all']] = cur_data['probability']
        if f"solution_{i}" in group:
            del group[f"solution_{i}"]
        group.create_dataset(f"solution_{i}", data=results_array, dtype=dtype_descr)


pygid_results_dtype = np.dtype([
    ('amplitude', 'f4'),
    ('angle', 'f4'),
    ('angle_width', 'f4'),
    ('radius', 'f4'),
    ('radius_width', 'f4'),
    ('q_z', 'f4'),
    ('q_xy', 'f4'),
    ('theta', 'f4'),
    ('score', 'f4'),
    ('A', 'f4'),
    ('B', 'f4'),
    ('C', 'f4'),
    ('is_ring', 'bool'),
    ('is_cut_qz', 'bool'),
    ('is_cut_qxy', 'bool'),
    ('visibility', 'i4'),
    ('id', 'i4'),
])


def _save_img_container_detect(root, group_name, img_container_detect):
    results_array = get_results_detect_array(img_container_detect)
    group = root[group_name]
    if 'detected_peaks' in group:
        del group['detected_peaks']
    group.create_dataset('detected_peaks', data=results_array, dtype=pygid_results_dtype)


def get_results_detect_array(img_container):
    results_array = np.zeros(len(img_container.radius_width), dtype=pygid_results_dtype)
    results_array['amplitude'] = [0] * len(img_container.radius_width)
    results_array['angle'] = img_container.angle
    results_array['angle_width'] = [abs(num) for num in img_container.angle_width]
    results_array['radius'] = img_container.radius
    results_array['radius_width'] = img_container.radius_width
    results_array['q_z'] = img_container.qzqxyboxes[0]
    results_array['q_xy'] = img_container.qzqxyboxes[1]
    results_array['theta'] = [0] * len(img_container.radius_width)
    results_array['A'] = [0] * len(img_container.radius_width)
    results_array['B'] = [0] * len(img_container.radius_width)
    results_array['C'] = [0] * len(img_container.radius_width)
    results_array['is_ring'] = [0] * len(img_container.radius_width)
    results_array['is_cut_qz'] = [0] * len(img_container.radius_width)
    results_array['is_cut_qxy'] = [0] * len(img_container.radius_width)
    results_array['visibility'] = [0] * len(img_container.radius_width)
    results_array['score'] = img_container.scores
    results_array['id'] = list(range(len(img_container.radius)))
    return results_array


def _save_img_container_fit(root, group_name, img_container_fit):
    results_array = get_results_fit_array(img_container_fit)
    results_err_array = get_results_fit_err_array(img_container_fit)
    group = root[group_name]
    if 'fitted_peaks' in group:
        del group['fitted_peaks']
    group.create_dataset('fitted_peaks', data=results_array, dtype=pygid_results_dtype)
    if 'fitted_peaks_errors' in group:
        del group['fitted_peaks_errors']
    group.create_dataset('fitted_peaks_errors', data=results_err_array, dtype=pygid_results_dtype)


def get_results_fit_array(img_container):
    results_array = np.zeros(len(img_container.radius_width), dtype=pygid_results_dtype)
    results_array['amplitude'] = img_container.amplitude
    results_array['angle'] = img_container.angle
    results_array['angle_width'] = img_container.angle_width
    results_array['radius'] = img_container.radius
    results_array['radius_width'] = img_container.radius_width
    results_array['q_z'] = img_container.qzqxyboxes[0]
    results_array['q_xy'] = img_container.qzqxyboxes[1]
    results_array['theta'] = img_container.theta
    results_array['A'] = img_container.A
    results_array['B'] = img_container.B
    results_array['C'] = img_container.C
    results_array['is_ring'] = img_container.is_ring
    results_array['is_cut_qz'] = img_container.is_cut_qz
    results_array['is_cut_qxy'] = img_container.is_cut_qxy
    results_array['visibility'] = img_container.visibility
    results_array['score'] = img_container.score
    results_array['id'] = img_container.id
    return results_array


def get_results_fit_err_array(img_container):
    results_array = np.zeros(len(img_container.radius_width), dtype=pygid_results_dtype)
    results_array['amplitude'] = img_container.amplitude_err
    results_array['angle'] = img_container.angle_err
    results_array['angle_width'] = img_container.angle_width_err
    results_array['radius'] = img_container.radius_err
    results_array['radius_width'] = img_container.radius_width_err
    results_array['q_z'] = img_container.qzqxyboxes_err[0]
    results_array['q_xy'] = img_container.qzqxyboxes_err[1]
    results_array['theta'] = img_container.theta_err
    results_array['A'] = img_container.A_err
    results_array['B'] = img_container.B_err
    results_array['C'] = img_container.C_err
    results_array['is_ring'] = img_container.is_ring
    results_array['is_cut_qz'] = img_container.is_cut_qz
    results_array['is_cut_qxy'] = img_container.is_cut_qxy
    results_array['visibility'] = img_container.visibility
    results_array['score'] = img_container.score
    results_array['id'] = img_container.id
    return results_array