from __future__ import annotations
from typing import TYPE_CHECKING
import os
from pathlib import Path
import yaml
from dataclasses import dataclass
from typing import Any
import numpy as np
import h5py
import re
from datetime import datetime
import warnings
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# from silx.io.h5py_utils import open_item

if TYPE_CHECKING:
    from . import Conversion


class ExpMetadata:
    """
    Container class for storing sample and experimental metadata.

    This class is designed to hold arbitrary experimental parameters, such as
    sample name, substrate type, growth conditions, and instrument settings.
    All stored attributes are intended to be saved in the `sample` group of
    an HDF5 file for later retrieval or data analysis.

    Examples
    --------
    meta = pygid.ExpMetadata(sample_name="CuPc", substrate="SiO2", temperature=300)
    print(meta)
    ExpMetadata({'sample_name': 'CuPc', 'substrate': 'SiO2', 'temperature': 300})

    Attributes
    ----------
    **kwargs : dict
        Arbitrary keyword arguments defining the metadata fields.
        Each key–value pair becomes an attribute of the instance.
    """

    def __init__(self, **kwargs):
        # Dynamically assign all provided keyword arguments as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
        if not hasattr(self, 'extend_fields'):
            self.extend_fields = []

    def __setattr__(self, name, value):
        self.__dict__[name] = value

    def __repr__(self):
        # Compact string representation for easy inspection
        return f"{self.__class__.__name__}({self.__dict__})"


class SampleMetadata:
    """
        Class for managing sample metadata: loading from and saving to files.

        This class provides a lightweight interface for reading and writing sample
        metadata to text or YAML files. Metadata is stored internally as a dictionary
        (`self.data`). Upon initialization, it can automatically load from an existing
        file or save to a specified path.

        Parameters
        ----------
        path_to_save : str or None, optional
            Path where metadata should be saved. If provided, `save()` is called automatically.
        path_to_load : str or None, optional
            Path from which metadata should be loaded. If provided, `load()` is called automatically.
        data : dict or None, optional
            Dictionary containing metadata key–value pairs. If None, an empty dictionary is created.

        Attributes
        ----------
        data : dict
            Metadata storage dictionary. All loaded or manually added fields are stored here.
        path_to_save : str or None
            Path used for saving metadata.
        path_to_load : str or None
            Path used for loading metadata.

        Examples
        --------
        data = {
            "name": "240306_DIP",
            "structure": {
                "stack": "air | DIP 0-25| SiOx 1| Si",
                "materials": {
                    "C60": {
                        "name": "Diindenoperylene DIP",
                        "thickness": 25,  # optional
                        "cif": "DIP.cif",  # optional
                        "type": "gradient film"  # optional /layer
                    },
                    "SiOx": {
                        "name": "native SiOx",
                        "thickness": 1,
                    },
                    "Si": {
                        "name": "Si wafer",
                    }
                }
            },
            "preparation": "gradient thin film prepared by thermal evaporation",
            "experimental_conditions": "standard conditions, on air"
        }
        smpl_metadata = pygid.SampleMetadata(path_to_save="sample.yaml", data=data)
        """

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
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.info(f"Saved sample metadata to {Path(filepath).resolve()}")

    def load(self, filepath=None):
        filepath = filepath or self.path_to_load
        if filepath is None:
            raise ValueError("File path is not defined for loading.")

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
    Class for saving processed experimental or sample data to HDF5 files.

    This class provides a structured interface for storing processed data,
    metadata, and analysis results in HDF5 format. It supports saving
    sample images, fitted results, matched data, and unit cell parameters,
    along with associated experimental and sample metadata.

    Attributes
    ----------
    sample : Conversion
        An instance of the `Conversion` class representing the sample data
        to be saved.
    path_to_save : str, optional
        Full path including the file name where the data should be saved.
        The file format should be `.h5`. Default is None.
    h5_group : str, optional
        The specific group in the HDF5 file where the data will be stored.
        Default is `{file name}_0000`.
    overwrite_file : bool, optional
        Whether to overwrite an existing HDF5 file. Default is True.
    overwrite_group : bool, optional
        Whether to overwrite an existing group within an HDF5 file. Default is False.
    exp_metadata : ExpMetadata, optional
        An instance of `ExpMetadata` containing experimental metadata. Default is None.
    smpl_metadata : SampleMetadata, optional
        An instance of `SampleMetadata` containing sample-related metadata. Default is None.
    img_container_detect : Any, optional
        Container for detected boxes from mlgidDETECT. Default is None.
    img_container_fit : Any, optional
        Container for fitted peaks from pygidFIT. Default is None.
    matched_data : Any, optional
        Data resulting from matching from mlgidMATCH. Default is None.
    unit_cell_data : Any, optional
        Data containing unit cell parameters from mlgidMATCH. Default is None.
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
        """
        Post-initialization to prepare the DataSaver instance for saving.

        This method extracts relevant data and parameters from the provided `sample`
        (Conversion instance) and prepares the HDF5 saving configuration. It also
        saves the first available dataset from the sample and removes it from memory.
        """

        # Create an instance-level logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        # Extract the first matrix and its parameters
        self.matrix = self.sample.matrix[0]
        self.params = self.matrix.params

        # Store original path if it exists
        if hasattr(self.sample, 'path'):
            self.original_path = self.sample.path

        # Ensure the list of incident angles exists
        if hasattr(self.sample, 'ai_list'):
            self.ai_list = self.sample.ai_list
        else:
            raise ValueError("conversion process is not correct. ai_list was not calculated")

        # Ensure the converted frame number exists
        if hasattr(self.sample, 'converted_frame_num'):
            self.frame_num = self.sample.converted_frame_num
        else:
            raise ValueError("conversion process is not correct. converted_frame_num was not calculated")

        # Set default save path and entry group if none provided
        if self.path_to_save is None:
            self.path_to_save = "result.h5"
        if self.h5_group is None:
            self.h5_group = os.path.basename(os.path.splitext(self.path_to_save)[0])
            self.h5_group = modify_string('entry', first_modification=True)

        # List of potential data attributes to save from the sample
        keys = [
            "img_gid_q", "img_q", "img_gid_pol",
            "img_pol", "img_gid_pseudopol", "img_pseudopol",
            "rad_cut_gid", "azim_cut_gid", "horiz_cut_gid",
            "rad_cut", "azim_cut"
        ]

        # Find the first available attribute in the sample to save
        name, data = next(
            ((key, getattr(self.sample, key)) for key in keys if hasattr(self.sample, key)),
            (None, None)
        )

        # Save the data and remove it from the sample to free memory
        self._save_data_(name, data)
        delattr(self.sample, name)

    def _save_data_(self, name, data):
        """
        Save converted sample data and associated metadata to an HDF5 file.

        This method handles creation of the HDF5 file and its internal groups,
        saving experimental parameters, sample metadata, processed images,
        and analysis results. It supports automatic group creation, appending,
        and overwriting based on the instance configuration.

        Parameters
        ----------
        name : str
            Name of the dataset to save (usually derived from the sample attribute).
        data : np.ndarray or None
            Image or processed data array to be saved. If None, only metadata is saved.

        Notes
        -----
        - Automatically creates parent directories if they do not exist.
        - Sets `NX_class` attributes for NeXus compliance.
        - Checks for group existence and type/shape mismatches, creating new groups if needed.
        - Saves experimental parameters, incident angles, frame numbers, and optional metadata.
        - Calls helper functions to save images, matrices, and analysis results.
        """

        # Ensure the folder exists
        folder_to_save = os.path.dirname(self.path_to_save)
        if not os.path.exists(folder_to_save) and folder_to_save != "":
            os.makedirs(folder_to_save)
        file_name = self.path_to_save

        # Determine HDF5 file mode: overwrite or append
        mode = 'w' if os.path.exists(file_name) and self.overwrite_file else 'a'

        # Open HDF5 file
        with h5py.File(file_name, mode) as root:
            root.attrs["NX_class"] = "NXroot"
            # Handle existing group: check for overwrite or shape mismatch
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
                    del root[self.h5_group]  # Overwrite group if requested

            # Create group if it does not exist
            if self.h5_group not in root:
                _make_groups_(root, self.h5_group)
                save_single_data(root[f"/{self.h5_group}"], 'definition', 'NXsas')
                save_single_data(root[f"/{self.h5_group}"], 'title', str(self.h5_group))
                save_expparams(root, self.h5_group, self.params)

            # Save incident angles and frame numbers
            save_single_data(root[f"/{self.h5_group}/instrument"], 'angle_of_incidence', self.ai_list, extend_list=True)
            save_single_data(root[f"/{self.h5_group}/data"], 'frame_num', self.frame_num, extend_list=True)

            # Save experimental metadata
            if self.exp_metadata is None:
                self.exp_metadata = ExpMetadata(filename=self.original_path)
            if not hasattr(self.exp_metadata, "filename"):
                self.exp_metadata.filename = self.original_path
            if not 'filename' in self.exp_metadata.extend_fields:
                self.exp_metadata.extend_fields.append('filename')
            save_exp_metadata(root, self.exp_metadata, self.h5_group)

            # Save sample metadata
            if self.smpl_metadata is not None:
                save_smpl_metadata(root, self.smpl_metadata, self.h5_group)

            # Save processed data
            if data is not None:
                create_dataset(root, self.h5_group, name, data)

            # Save matrix and associated analysis results
            save_matrix(root, self.h5_group, self.matrix, name)
            fill_process_group(root, self.h5_group, self.matrix)
            fill_analysis_group(root, self.h5_group, len(data), self.img_container_detect, self.img_container_fit,
                                self.matched_data, self.unit_cell_data)
            self.logger.info(f"Saved in {Path(self.path_to_save).resolve()} in group {self.h5_group}")
        return


def save_matrix(root, h5_group, matrix, img_name):
    """
    Save coordinate map data (matrix) to an HDF5 file.

    This function extracts relevant coordinate arrays from the provided `matrix`
    object based on the image type (`img_name`) and saves them under the
    `/data` subgroup in the specified HDF5 group. Attributes for interpretation
    and units are added for NeXus/SAS compliance.

    Parameters
    ----------
    root : h5py.File
        The root HDF5 file object where the matrix will be saved.
    h5_group : str
        Name of the group within the HDF5 file under which the matrix data will be stored.
    matrix : object
        An object containing coordinate arrays as attributes. Typically the
        `matrix` attribute of a Conversion instance.
    img_name : str
        The name of the image/data type. Determines which coordinate keys are saved.

    Notes
    -----
    - Adds 'signal' and 'axes' attributes to the saved data for proper NeXus conventions.
    - Only attributes existing on `matrix` are saved; missing keys are skipped.
    """
    # Mapping of image names to expected coordinate attributes
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

    # Determine which keys to save for this image type
    keys = keys_map.get(img_name, [])

    # Extract existing attributes from the matrix
    coords_dict = {key: getattr(matrix, key) for key in keys if hasattr(matrix, key)}
    keys = list(coords_dict.keys())

    # Save each coordinate array as float32 with proper NeXus attributes
    for name in coords_dict:
        data = coords_dict[name]
        save_single_data(root[f"{h5_group}/data"], name,
                         np.array(data, dtype=np.float64), attrs={'interpretation': 'axis', 'units': '1/Angstrom'})

    # Add 'signal' and 'axes' attributes for NeXus compliance and silx visualisation
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
    Check if the dataset shapes in the given HDF5 group match the provided data.
    If a mismatch is detected, generate a new group name recursively until a compatible
    group is found.

    Parameters
    ----------
    root : h5py.File or h5py.Group
        The root HDF5 object containing the group to check.
    h5_group : str
        Name of the group within the HDF5 file to verify.
    name : str
        Name of the dataset to be saved.
    data_shape : tuple
        Shape of the new dataset intended to be saved.

    Returns
    -------
    str
        A valid HDF5 group name where the dataset can be safely saved.

    Notes
    -----
    - Recursively modifies the group name using `modify_string` until the shapes
      of existing datasets do not conflict with the new data.
    """
    # Get shapes of existing datasets in the group
    data_old = read_dataset_size(root, h5_group)

    # Check if any dataset has a different shape or a different name
    change = any(
        key != name or data_shape_old != data_shape
        for key, data_shape_old in data_old.items()
    )

    # If there is a mismatch, generate a new group name recursively
    if change:
        new_h5_group = modify_string(h5_group, first_modification=False)
        return change_h5_group(root, new_h5_group, name, data_shape)

    # Otherwise, return the current group name
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
    save_single_metadata(root[f"/{h5_group}"], exp_metadata, 'start_time', 'start_time', "NX_DATE_TIME", required=True,
                         extend_list=True if 'start_time' in exp_metadata.extend_fields else False)
    save_single_metadata(root[f"/{h5_group}"], exp_metadata, 'end_time', 'end_time', "NX_DATE_TIME", required=True,
                         extend_list=True if 'end_time' in exp_metadata.extend_fields else False)
    # metadata that shold be extended
    save_single_metadata(root[f"/{h5_group}/data"], exp_metadata, 'filename', 'filename', required=False,
                         extend_list=True)
    # save_single_metadata(root[f"/{h5_group}/instrument"], exp_metadata, 'frame_idx', 'frame_idx', required=False,
    #                      extend_list=True)
    # save_single_metadata(root[f"/{h5_group}/instrument"], exp_metadata, 'epoch', 'epoch', required=False,
    #                      extend_list=True)
    # save_single_metadata(root[f"/{h5_group}/instrument"], exp_metadata, 'mon', 'mon', required=False,
    #                      extend_list=True)
    # save_single_metadata(root[f"/{h5_group}/instrument"], exp_metadata, 'watt0', 'watt0', required=False,
    #                      extend_list=True)
    # save_single_metadata(root[f"/{h5_group}/instrument"], exp_metadata, 'transmission', 'transmission', required=False,
    #                      extend_list=True)

    saved_attr = ['extend_fields','instrument_name', 'source_type', 'source_probe', 'source_name', 'wavelength_spread',
                  'source_name', 'start_time', 'end_time', 'detector_name', 'detector', 'source',
                  'filename']
    for attr_name in exp_metadata.__dict__:
        if attr_name not in saved_attr:
            save_single_metadata(root[f"/{h5_group}/instrument"], exp_metadata, attr_name, attr_name,
                                 extend_list=True if attr_name in exp_metadata.extend_fields else False)


def save_smpl_metadata(root, smpl_metadata=None, h5_group="entry"):
    """
        Save sample metadata to an HDF5 file under the specified group.

        This function recursively writes all key-value pairs from a `SampleMetadata`
        instance into the `/sample` subgroup of the given HDF5 group. Nested dictionaries
        are stored as subgroups. Non-array values that cannot be directly saved in
        HDF5 are converted to strings.

        Parameters
        ----------
        root : h5py.File or h5py.Group
            The root HDF5 object where the metadata will be saved.
        smpl_metadata : SampleMetadata
            Instance containing sample metadata to save.
        h5_group : str, optional
            HDF5 group under which the metadata will be stored. Default is "entry".
    """
    # Ensure a valid SampleMetadata instance is provided
    if smpl_metadata is None or not isinstance(smpl_metadata, SampleMetadata):
        raise ValueError("Valid SampleMetadata instance must be provided.")

    h5_path = f"/{h5_group}/sample"
    group = root.require_group(h5_path)

    def write_dict_to_group(h5grp, data):
        """Recursively write a dictionary to an HDF5 group."""
        for key, value in data.items():
            if isinstance(value, dict):
                # Create a subgroup for nested dictionaries
                subgrp = h5grp.require_group(key)
                write_dict_to_group(subgrp, value)
            else:
                # Remove existing dataset if it exists
                if key in h5grp:
                    del h5grp[key]
                # Try to save value; fallback to string if necessary
                try:
                    h5grp.create_dataset(key, data=value)
                except TypeError:
                    h5grp.create_dataset(key, data=str(value))

    # Write the SampleMetadata dictionary to the HDF5 group
    write_dict_to_group(group, smpl_metadata.data)

    # Warn if a 'name' attribute is missing
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
    """
        Fill the 'process' subgroup of an HDF5 group with processing metadata.

        This function records information about the program, version, timestamp,
        and all intensity corrections applied to the sample data. Each correction
        is documented with relevant parameters from the `matrix` object.

        Parameters
        ----------
        root : h5py.File or h5py.Group
            The root HDF5 object containing the target group.
        h5_group : str
            Name of the HDF5 group under which the 'process' subgroup exists.
        matrix : CoordMaps object
            Stroes corrections imformation

        Notes
        -----
        - Saves the program name, version, and current timestamp.
        - Summarizes all intensity corrections applied with relevant parameters.
        - Stores the summary in the 'NOTE' dataset under the 'process' subgroup.
    """
    corr_matrices = matrix.corr_matrices
    h5_group = "/" + h5_group
    group = root[h5_group + '/process']
    # Save program metadata
    save_single_data(group, 'program', "pygid", extend_list=False)
    from . import __version__ as pygid_version
    save_single_data(group, 'version', pygid_version, extend_list=False)
    save_single_data(group, 'date', datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f'), extend_list=False)
    # Construct detailed note on intensity corrections
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

    # Save the corrections summary
    save_single_data(group, 'NOTE', NOTE, extend_list=False)


def fill_analysis_group(root, h5_group, img_number_to_add, img_container_detect, img_container_fit, matched_data,
                        unit_cell_data):
    """
    Create analysis-related fields in a specified HDF5 group.

    This function ensures that all necessary datasets and subgroups for analysis
    are present in the HDF5 file. It can create new groups for additional frames
    and save detected images, fitted images, matched data, and unit cell data
    for each frame.

    Parameters
    ----------
    root : h5py.File or h5py.Group
        The root HDF5 object or parent group where analysis fields will be created.
    h5_group : str
        Name of the group within the HDF5 file where analysis datasets are stored.
    img_number_to_add : int
        Number of new analysis frames (subgroups) to add.
    img_container_detect : array-like or list, optional
        Detected images to store in the new analysis frames. If not a list, it will
        be converted to a single-item list.
    img_container_fit : array-like or list, optional
        Fitted images to store in the new analysis frames. If not a list, it will
        be converted to a single-item list.
    matched_data : array-like or list, optional
        Matched data to store in the new analysis frames. If not a list, it will
        be converted to a single-item list.
    unit_cell_data : array-like or list, optional
        Unit cell data to store in the new analysis frames. If not a list, it will
        be converted to a single-item list.
    """
    analysis_path = f"/{h5_group}/data/analysis"

    # Convert single items to lists for consistent indexing
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

    # Create the main analysis group if it does not exist
    if analysis_path not in root:
        root.create_group(analysis_path)
        root[analysis_path].attrs.update({'NX_class': 'NXparameters', 'EX_required': 'true'})
    group = root[analysis_path]

    # Count existing frames
    subgroups = [name for name in group if isinstance(group[name], h5py.Group)]
    img_number_current = len(subgroups)

    # Create new frames and store provided data
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
    """
    Save unit cell analysis results to a specified HDF5 group.

    Parameters
    ----------
    root : h5py.File or h5py.Group
        The root HDF5 object containing the target group.
    group_name : str
        The name of the HDF5 group where the unit cell data will be stored.
    unit_cell_data : tuple
        A tuple containing:
        - orientation_array : ndarray of shape (N, 3), Miller indices (h, k, l)
        - unit_cells : ndarray of shape (N, 6), lattice parameters (a, b, c, alpha, beta, gamma)
        - unit_cell_errors : ndarray of shape (N, 6), errors for lattice parameters
        - q_losses : ndarray of shape (N,), loss values
        - is_converged : ndarray of shape (N,), convergence flags
        - score : ndarray of shape (N,), scoring metric
    """
    group = root[group_name]

    # Unpack the unit cell data
    orientation_array, unit_cells, unit_cell_errors, q_losses, is_converged, score = unit_cell_data

    # Initialize structured array for results
    results_array = np.zeros(orientation_array.shape[0], dtype=unit_cell_data_dtype)

    # Save Miller indices
    results_array['h'] = orientation_array[:, 0]
    results_array['k'] = orientation_array[:, 1]
    results_array['l'] = orientation_array[:, 2]

    # Save lattice parameters
    results_array['a'] = unit_cells[:, 0]
    results_array['b'] = unit_cells[:, 1]
    results_array['c'] = unit_cells[:, 2]
    results_array['alpha'] = unit_cells[:, 3]
    results_array['beta'] = unit_cells[:, 4]
    results_array['gamma'] = unit_cells[:, 5]

    # Save lattice parameter errors
    results_array['a_err'] = unit_cell_errors[:, 0]
    results_array['b_err'] = unit_cell_errors[:, 1]
    results_array['c_err'] = unit_cell_errors[:, 2]
    results_array['alpha_err'] = unit_cell_errors[:, 3]
    results_array['beta_err'] = unit_cell_errors[:, 4]
    results_array['gamma_err'] = unit_cell_errors[:, 5]

    # Save additional parameters
    results_array['q_loss'] = q_losses
    results_array['is_converged'] = is_converged
    results_array['score'] = score

    # Overwrite existing dataset if it exists
    if f"unit_cell_data" in group:
        del group[f"unit_cell_data"]

    # Create new dataset in HDF5 group
    group.create_dataset(f"unit_cell_data", data=results_array, dtype=unit_cell_data_dtype)


def _save_matched_data(root, group_name, matched_data):
    """
        Save mlgidMATCH data results to an HDF5 group.

        Parameters
        ----------
        root : h5py.File or h5py.Group
            The root HDF5 object containing the target group.
        group_name : str
            The HDF5 group where matched data will be stored.
        matched_data : tuple
            A tuple containing:
            - data_matched : dict
                Dictionary containing probability data for peaks.
            - unique_solutions : dict
                Dictionary mapping keys to lists of unique solutions.
    """

    group = root[group_name]
    data_matched, unique_solutions = matched_data

    # Take the first key
    key = list(unique_solutions.keys())[0]

    # Iterate over all solutions for this key
    for i, sol in enumerate(unique_solutions[key]):
        indices, names = zip(*[(el[0], el[1] + ' ' + str(el[2])) for el in sol])

        dtype_descr = list({(n, np.float32) for n in np.unique(names)})
        # Initialize structured array to store results
        results_array = np.zeros(len(data_matched[key]['peaks']), dtype=np.dtype(dtype_descr))

        cur_data = data_matched[key]

        # Fill structured array with probabilities
        for idx, name in zip(indices, names):
            cur_data = cur_data[idx]
            mask = np.where(results_array[name] == 0)
            results_array[name][np.intersect1d(mask, cur_data['indices_real_matched_all'])] = cur_data['probability']
            mask = np.where(results_array[name] == 0)
            results_array[name][np.intersect1d(mask, cur_data['indices_real_matched'])] = cur_data['probability']

        # Overwrite existing dataset if it exists
        if f"solution_{i}" in group:
            del group[f"solution_{i}"]
        # Create dataset for the solution
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
    """
    Save detected data from mlgidDETECT to the HDF5 group.

    Parameters
    ----------
    root : h5py.File or h5py.Group
        The root HDF5 object containing the target group.
    group_name : str
        The name of the HDF5 group where detected peaks will be stored.
    img_container_detect : mlgidDETECT.ImageContainer
        Container of detected image results to be saved.
    """
    # Convert detected images to structured array
    results_array = get_results_detect_array(img_container_detect)
    group = root[group_name]

    # Remove existing dataset if it exists
    if 'detected_peaks' in group:
        del group['detected_peaks']

    # Create dataset for detected peaks
    group.create_dataset('detected_peaks', data=results_array, dtype=pygid_results_dtype)


def get_results_detect_array(img_container):
    """
        Convert detected image container data into a structured NumPy array.

        Parameters
        ----------
        img_container : object
            An object containing detected image properties. Expected to have attributes:
            - radius_width, angle, angle_width, radius, qzqxyboxes, scores
    """
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
    """
    Save fitted peaks data from pygidFIT and associated errors to an HDF5 group.

    Parameters
    ----------
    root : h5py.File or h5py.Group
        The root HDF5 object containing the target group.
    group_name : str
        The HDF5 group where fitted peaks and errors will be stored.
    img_container_fit : pygidFIT.ImageContainer object
        Container of fitted image results.
    """

    # Convert fitted images to structured arrays
    results_array = get_results_fit_array(img_container_fit)
    results_err_array = get_results_fit_err_array(img_container_fit)
    group = root[group_name]

    # Remove existing datasets if they exist
    if 'fitted_peaks' in group:
        del group['fitted_peaks']
    if 'fitted_peaks_errors' in group:
        del group['fitted_peaks_errors']
    # Create datasets for fitted peaks and errors
    group.create_dataset('fitted_peaks', data=results_array, dtype=pygid_results_dtype)
    group.create_dataset('fitted_peaks_errors', data=results_err_array, dtype=pygid_results_dtype)


def get_results_fit_array(img_container):
    """
        Convert fitted image container data into a structured NumPy array.

        Parameters
        ----------
        img_container : pygidFIT.ImageContainer object
            Container of fitted image results. Expected to have attributes:
            amplitude, angle, angle_width, radius, radius_width, qzqxyboxes,
            theta, A, B, C, is_ring, is_cut_qz, is_cut_qxy, visibility, score, id
    """
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
    """
        Convert fitted image container error data into a structured NumPy array.

        Parameters
        ----------
        img_container : pygidFIT.ImageContainer object
            Container of fitted peaks errors. Expected to have attributes:
            amplitude_err, angle_err, angle_width_err, radius_err, radius_width_err,
            qzqxyboxes_err, theta_err, A_err, B_err, C_err, is_ring, is_cut_qz,
            is_cut_qxy, visibility, score, id
    """
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
