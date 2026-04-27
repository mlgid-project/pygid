import h5py
import numpy as np
from dataclasses import dataclass
from pprint import pprint

from . import ExpParams, CoordMaps, Conversion
from .datasaver import (save_smpl_metadata, ensure_group_exists, SampleMetadata,
                        save_exp_metadata, save_single_metadata, ExpMetadata)

@dataclass
class NexusFile:
    """
       Interface for reading, inspecting, and modifying pygid NeXus (HDF5) files.

       This class provides high-level access to converted pygid data stored in
       NeXus-compliant HDF5 files. It supports inspection of file structure,
       loading of Conversion instances, modification of datasets, and extraction
       or overwriting of experimental and sample metadata.

       Parameters
       ----------
       path : str
           Path to the NeXus (HDF5) file.
    """
    path: str

    def __post_init__(self):
        """
        Initialize the NexusFile instance by reading the file structure.
        """
        self.entry_dict = self.read_structure()

    def read_structure(self):
        """
           Read and parse the structure of the NeXus file.

           This method inspects all entries in the HDF5 file and extracts
           information about the stored datasets, including signal type,
           axes, description, and data shape.

           Returns
           -------
           dict
               Dictionary mapping entry names to their metadata, including
               image type, axes, description, and shape.

           Raises
           ------
           ValueError
               If no valid entries are found in the file.
        """
        entry_dict = {}
        with h5py.File(self.path, "r") as root:
            for entry in root:
                # try:
                    img_type, axes, description, shape = get_entry_type(root, entry)
                    entry_dict[entry] = {
                        "img_type": img_type.decode() if isinstance(img_type, bytes) else img_type,
                        "axes": [
                                a.decode() if isinstance(a, bytes) else a
                                for a in axes
                                if (a.decode() if isinstance(a, bytes) else a) != "frame_num"
                            ],
                        "description": description,
                        "shape": shape,
                    }

        if not entry_dict:
            raise ValueError("No valid entries found")
        return entry_dict

    def print_file_structure(self):
        """
            Print a representation of the file structure.

            Displays the parsed entry dictionary, including dataset types
            and geometries, for inspection.
        """
        entry_dict = self.entry_dict
        if entry_dict is None:
            return ValueError("File structure was not read")
        print(f"File structure: {self.path}")
        pprint(entry_dict)

    def load_entry(self, entry, frame_num = None):
        """
            Load a pygid.Conversion instance from a given NeXus entry.

            Parameters
            ----------
            entry : str
                Name of the NeXus entry to load.
            frame_num : int, list or None, optional
                Frame index to load or list of int. If None, all frames are loaded.

            Returns
            -------
            pygid.Conversion
                Conversion instance reconstructed from the NeXus file.

            Raises
            ------
            ValueError
                If the specified entry does not exist.
        """
        with h5py.File(self.path, "r") as root:
            if not entry in root:
                raise ValueError(f"Entry {entry} not found")
            entry_info = self.entry_dict.get(entry)
            matrix = load_martix(root[entry], entry_info.get('axes'), frame_num)
            analysis = load_analysis(root[entry], matrix, entry_info.get('img_type'), frame_num)
            return analysis

    def change_smpl_metadata(self, entry, smpl_metadata):
        """
        Overwrite sample metadata for a given entry.

        Parameters
        ----------
        entry : str
            Name of the NeXus entry.
        smpl_metadata : SampleMetadata
            Sample metadata instance to be written.

        Raises
        ------
        ValueError
            If the specified entry does not exist.
        """
        with h5py.File(self.path, "r+") as root:
            if not entry in root:
                raise ValueError(f"Entry {entry} not found")
            if 'sample' in root[entry]:
                del root[f'{entry}/sample']
            ensure_group_exists(root, f'/{entry}/sample', {'NX_class': 'NXsample', 'EX_required': 'true'})
            save_smpl_metadata(root, smpl_metadata, entry)

    def change_exp_metadata(self, entry, exp_metadata):
        """
        Overwrite experimental metadata for a given entry.

        Parameters
        ----------
        entry : str
            Name of the NeXus entry.
        exp_metadata : ExpMetadata
            Experimental metadata instance to be written.

        Raises
        ------
        ValueError
            If the specified entry does not exist.
        """
        exp_metadata.extend_fields = []
        with h5py.File(self.path, "r+") as root:
            if not entry in root:
                raise ValueError(f"Entry {entry} not found")
            save_exp_metadata(root, exp_metadata, entry)

    def get_smpl_metadata(self, entry, path_to_save = None):
        """
        Retrieve sample metadata from a NeXus entry.

        Parameters
        ----------
        entry : str
            Name of the NeXus entry.
        path_to_save : str or None, optional
            If provided, saves the metadata as a YAML file.

        Returns
        -------
        SampleMetadata
            Sample metadata instance reconstructed from the file.

        Raises
        ------
        ValueError
            If the specified entry does not exist.
        """
        with h5py.File(self.path, "r") as root:
            if not entry in root:
                raise ValueError(f"Entry {entry} not found")
            group = root[f'{entry}/sample']
            data = read_group(group)
            smpl_metadata = SampleMetadata(data = data, path_to_save=path_to_save)
            return smpl_metadata

    def get_exp_metadata(self, entry):
        """
        Retrieve experimental metadata from a NeXus entry.

        Parameters
        ----------
        entry : str
            Name of the NeXus entry.

        Returns
        -------
        ExpMetadata
            Experimental metadata instance reconstructed from the file.

        Raises
        ------
        ValueError
            If the specified entry does not exist.
        """
        with h5py.File(self.path, "r") as root:
            if not entry in root:
                raise ValueError(f"Entry {entry} not found")
            filename = self.check_dataset(root, f'/{entry}/data/filename', None)
            start_time = self.check_dataset(root, f'/{entry}/start_time', 'start_time')
            end_time = self.check_dataset(root, f'/{entry}/end_time', 'end_time')
            instrument_name = self.check_dataset(root, f'/{entry}/instrument/name', 'instrument_name')
            source_name = self.check_dataset(root, f'/{entry}/instrument/source/name', 'source_name')
            source_type = self.check_dataset(root, f'/{entry}/instrument/source/type', 'source_type')
            metadata_dict = read_single_group(root[f'/{entry}/instrument'], ['name', 'angle_of_incidence'])
            exp_metadata = ExpMetadata(
                start_time = start_time,
                end_time = end_time,
                source_name = source_name,
                filename = filename,
                instrument_name = instrument_name,
                source_type = source_type,
            )
            for key, value in metadata_dict.items():
                setattr(exp_metadata, key, value)
            return exp_metadata

    def check_dataset(self, root, dataset_root, default):
        if dataset_root in root:
            value = root[dataset_root][()]
            if isinstance(value, np.generic):
                value = value.item()
            # Decode bytes
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            return value
        else:
            return default

    def rename_dataset(self, old_name, new_name):
        """Rename a dataset or group in the HDF5 file."""
        with h5py.File(self.path, "r+") as root:
            if old_name not in root:
                raise ValueError(f"{old_name} not found in the file")
            if new_name in root:
                raise ValueError(f"{new_name} already exists")
            root.move(old_name, new_name)

    def get_dataset(self, data_root, frame_num = None):
        with h5py.File(self.path, "r+") as root:
            if not data_root in root:
                raise ValueError(f"data_root {data_root} not found")
            dset = root[data_root]
            if frame_num is not None:
                return dset[frame_num]
            else:
                return dset[()]


    def create_dataset(self, data_root, data):
        with h5py.File(self.path, "r+") as root:
            if data_root in root:
                del root[data_root]
            root.create_dataset(data_root, data = data)

    def change_dataset(self, data_root, frame_num = None, data = None):
        """
        Modify an existing dataset in the NeXus file.

        Parameters
        ----------
        data_root : str
            Full path to the dataset.
        frame_num : int or None, optional
            Index of the frame to modify. If None, the entire dataset is overwritten.
        data : any
            New data to write.
        """
        with h5py.File(self.path, "r+") as root:
            if not data_root in root:
                raise ValueError(f"data_root {data_root} not found")
            dset = root[data_root]
            if frame_num is None:
                del root[data_root]
                root.create_dataset(data_root, data=data)
            elif isinstance(frame_num, int):
                dset[frame_num] = data
            return

    def delete_dataset(self, data_root):
        """
        Delete a dataset from the NeXus file.

        Parameters
        ----------
        data_root : str
            Full path to the dataset to be deleted.

        Raises
        ------
        ValueError
            If the dataset does not exist.
        """
        with h5py.File(self.path, "r+") as root:
            if not data_root in root:
                raise ValueError(f"data_root {data_root} not found")
            del root[data_root]



def read_group(h5grp):
    """Recursively read an HDF5 group into a dictionary."""
    data = {}
    for key, item in h5grp.items():
        if isinstance(item, h5py.Group):
            data[key] = read_group(item)
        else:  # h5py.Dataset
            value = item[()]
            if isinstance(value, np.generic):
                value = value.item()
            # Decode bytes
            if isinstance(value, bytes):
                value = value.decode("utf-8")

            data[key] = value
    return data

def read_single_group(h5grp, prohibited_keys = None):
    """Read an HDF5 group."""
    data = {}
    for key, item in h5grp.items():
        if isinstance(item, h5py.Group):
            continue
        else:  # h5py.Dataset
            value = item[()]
            if isinstance(value, np.generic):
                value = value.item()
            # Decode bytes
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            if prohibited_keys and key in prohibited_keys:
                continue
            data[key] = value
    return data



def load_analysis(root_entry, matrix, img_name, frame_num = None):
    img = get_img(root_entry, img_name, frame_num)
    analysis = Conversion(matrix, img_raw = None)
    setattr(analysis, img_name, img)
    analysis.ai_list = analysis.matrix[0].params.ai
    # analysis.converted_frame_num = root_entry['data/frame_num'][()]
    analysis.converted_frame_num = list(range(len(img)))
    analysis.filename = get_filename(root_entry, frame_num)
    return analysis

import numpy as np

def get_filename(root_entry, frame_num=None):
    """
    Retrieve filename(s) from HDF5 'data/filename'.

    Parameters
    ----------
    root_entry : h5py.Group
        The root HDF5 entry containing 'data/filename'.
    frame_num : int or list of int or np.ndarray, optional
        Frame index or list of indices to select filenames.

    Returns
    -------
    filename : str or list of str
        Filename(s) corresponding to the specified frame(s).
    """
    try:
        filenames_dataset = root_entry['data/filename'][()]
        # Convert bytes to string if needed
        if isinstance(filenames_dataset, bytes):
            filenames_dataset = filenames_dataset.decode('utf-8')
        # If it is a single string and no frame_num is specified
        if isinstance(filenames_dataset, str):
            return filenames_dataset
        # If it's an array of strings (e.g., np.ndarray of bytes)
        if frame_num is None:
            # return all filenames as a list of strings
            return [f.decode('utf-8') if isinstance(f, bytes) else f for f in filenames_dataset]
        if isinstance(frame_num, int):
            f = filenames_dataset[frame_num]
            return f.decode('utf-8') if isinstance(f, bytes) else f
        if isinstance(frame_num, (list, np.ndarray)):
            return [
                f.decode('utf-8') if isinstance(f, bytes) else f
                for i, f in enumerate(filenames_dataset) if i in frame_num
            ]
        # raise ValueError("Invalid type for frame_num. Must be int, list, or np.ndarray.")
    except:
        return None


def get_img(root_entry, img_name, frame_num):
    img = None
    if frame_num is None:
        img = root_entry[f'data/{img_name}'][()]
    elif isinstance(frame_num, int):
        img = [root_entry[f'data/{img_name}'][frame_num]]
    elif isinstance(frame_num, list) or isinstance(frame_num, np.ndarray):
        img = [root_entry[f'data/{img_name}'][num] for num in frame_num]
    else:
        raise ValueError("frame_num must be int or list")
    return img


def load_martix(root_entry, axes, frame_num):
    params = load_params(root_entry, frame_num)
    matrix = CoordMaps(params)
    if matrix.sub_matrices is None:
        matrix.sub_matrices = [matrix]
    for mat in matrix.sub_matrices:
        for ax in axes:
            ax_value= root_entry[f'data/{ax}'][()]
            setattr(mat, ax, ax_value)
            if ax.startswith("q"):
                mat.dq = ax_value[1]-ax_value[0]
            elif ax.startswith("ang"):
                mat.dang = ax_value[1]-ax_value[0]
    return matrix




def load_params(root_entry, frame_num):
    ai = get_ai(root_entry, frame_num)
    params = ExpParams(
        SDD=root_entry['instrument/detector/distance'][()],
        wavelength=root_entry['instrument/monochromator/wavelength'][()] * 1e10,
        rot1=root_entry['instrument/detector/aequatorial_angle'][()],
        rot2=-root_entry['instrument/detector/polar_angle'][()],
        rot3=root_entry['instrument/detector/rotation_angle'][()],
        centerX=root_entry['instrument/detector/beam_center_x'][()],
        centerY=root_entry['instrument/detector/beam_center_y'][()],
        px_size=root_entry['instrument/detector/x_pixel_size'][()],
        ai=ai,
        fliplr=get_h5_value(root_entry, 'process/fliplr', False),
        flipud=get_h5_value(root_entry, 'process/flipud', False),
        transp=get_h5_value(root_entry, 'process/transp', False),
    )

    params._calc_poni_from_center()

    return params


def get_ai(root_entry, frame_num):
    ai = None
    if frame_num is None:
        ai = root_entry['instrument/angle_of_incidence'][()]
        # if len(ai) == 1:
        #     return ai[0]
    elif isinstance(frame_num, int):
        ai = [root_entry['instrument/angle_of_incidence'][()][frame_num]]
    elif isinstance(frame_num, list) or isinstance(frame_num, np.ndarray):
        ai = [root_entry['instrument/angle_of_incidence'][num] for num in frame_num]
    else:
        raise ValueError("frame_num must be int or list")
    return ai



def get_entry_type(root, entry):
    if not entry in root:
        raise ValueError(f"entry {entry} was not found in file")

    entry_data = root[f"/{entry}/data"]
    img_type = entry_data.attrs.get("signal")
    axes = entry_data.attrs.get("axes")
    description = get_description(img_type)
    shape = root[f'{entry}/data/{img_type}'].shape

    if img_type is None or axes is None or description is None:
        raise ValueError(f"{entry} is not a valid entry")

    return img_type, axes, description, shape


def get_description(img_type):
    description_dict = {
        'img_gid_q': 'cylindrical coordinate conversion for GID geometry',
        'img_gid_pol': 'polar coordinate conversion for GID geometry',
        'img_gid_pseudopol': 'pseudopolar coordinate conversion for GID geometry',
        'img_q': 'Cartesian coordinate conversion for transmission geometry',
        'img_pol': 'polar coordinate conversion for transmission geometry',
        'img_pseudopol': 'pseudopolar coordinate conversion for transmission geometry',
        'rad_cut_gid': 'radial profile for GID geometry',
        'rad_cut': 'radial profile for transmission geometry',
        'azim_cut_gid': 'azimuthal profile for GID geometry',
        'azim_cut': 'azimuthal profile for transmission geometry',
        'horiz_cut_gid': 'horizontal profile for GID geometry',
        'vert_cut_gid': 'vertical profile for GID geometry',
    }
    return description_dict.get(img_type)


def get_h5_value(root, path, default=None):
    try:
        return root[path][()]
    except:
        return default