import os
import numpy as np
import h5py, hdf5plugin
import fabio
from typing import Union, List, Any, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import warnings, logging

@dataclass
class DataLoader:
    """
    A class to load raw image data and apply transformations.

    Attributes
    ----------
    path : str, optional
        The path to the raw data file.
    dataset : str, optional
        The necessary dataset root in .h5 and .nxs files. Default is 'measurement/eiger4m'.
    frame_num : float, optional
        The specific frame number (or list of numbers) is dataset to process. Default is None (all frames).
    roi_range : list, optional
        The range of the region of interest (ROI) (left, right, down, up). Default is [None, None, None, None].
    batch_size : int, optional
        The batch size for batch analysis. Default is 32.
    activate_batch: bool, optional
        Flag to activate batch analysis. Default is False.
    number_of_frames: int, optional
        Number of loaded frames to process. Default is None (all frames). Will be changed if > batch_size
    multiprocessing : bool, optional
        Whether to use multiprocessing for convetion and coordinate maps calculation. Default is False.
    """
    path: Union[str, List[str]] = None
    dataset: str = "measurement/eiger4m"
    frame_num: float = None
    roi_range: Any = False
    batch_size: int = 32
    activate_batch: bool = False
    number_of_frames: int = None
    multiprocessing: bool = False
    build_image_P03: bool = False

    def __post_init__(self):
        if self.build_image_P03:
            self.img_raw =self._reconstruct_lmbd_()
            return
        self.img_raw = self._process_path_()




    def _process_path_(self):
        """
        Handles the input path, distinguishing whether it is a string (single file) or a list (multiple files).
        If the number of files exceeds `batch_size`, it triggers batch analysis for the provided files.
        """

        warnings.filterwarnings("ignore", category=UserWarning, module="fabio.TiffIO")
        logging.getLogger("fabio.TiffIO").setLevel(logging.ERROR)

        if isinstance(self.path, str):
            img_raw = self._image_loading_(path=self.path, frame_num=self.frame_num, dataset=self.dataset)
            return img_raw

        elif isinstance(self.path, list):
            img_raw = []

            if len(self.path) > self.batch_size:
                print(f"Number of frames is more than {self.batch_size}. The batch processing has been activated.")
                self.activate_batch = True
                self.number_of_frames = len(self.path)
                return
            if self.multiprocessing:
                with ThreadPoolExecutor() as executor:
                    img_raw = list(executor.map(
                        lambda file: self._image_loading_(path=file, frame_num=self.frame_num, dataset=self.dataset),
                        self.path))
                # img_raw = np.array(img_raw)
                # img_raw = img_raw.reshape(-1, img_raw.shape[2], img_raw.shape[3])
            else:
                img_raw = [self._image_loading_(path=file, frame_num=self.frame_num, dataset=self.dataset) for file in self.path]
            img_raw = np.array(img_raw)
            img_raw = img_raw.reshape(-1, img_raw.shape[2], img_raw.shape[3])
            return img_raw

        else:
            raise FileNotFoundError(
                "Invalid path format. Should be either a str or a list of str.")

    def _image_loading_(self, path=None, frame_num=None, dataset= None):
        """
        Loads a single file based on its format. If the file format can be opened with `fabio`, it will be loaded using `fabio`.
        If the format cannot be handled by `fabio`, the function will call the `h5py` loading function to open HDF5 files.
        Optionally extracts a region of interest (ROI) from the loaded frame.

        path: str
            The file path.
        frame_num: int, optional
            The specific frame number to load from a multi-frame dataset (if applicable). Defaults to None.
        dataset: object, optional
            The necessary dataset root in .h5 and .nxs files.
        """
        check_file_exists(path)
        fmt = os.path.splitext(path)[1][1:]
        img_raw = None
        roi = (slice(self.roi_range[0], self.roi_range[1]), slice(self.roi_range[2], self.roi_range[3]))

        if fmt in ['hdf5', 'h5', 'nxs']:
            img_raw = self._load_with_h5py(path, frame_num, dataset, roi)
        else:
            try:
                img_raw = fabio.open(path).data[roi[0], roi[1]].astype('float32')
            except:
                raise FileNotFoundError(
                    "Invalid format. Only 'h5', 'nxs','tiff', 'cbf' and 'edf' are supported.")

        if img_raw is None:
            return
        if img_raw.ndim == 2:
            img_raw = np.expand_dims(img_raw, axis=0)

        return img_raw

    def _load_with_h5py(self, path, frame_num, dataset, roi):

        """
           Loads a specific frame from an HDF5 file using the `h5py` library. Optionally extracts a region of interest (ROI) from the loaded frame.

           Parameters
           ----------
           path: str
               The file path to the HDF5 file.
           frame_num: int
               The frame number to load from the HDF5 file.
           dataset: object
               The dataset root inside the `.h5` or `.nxs` file where the image data is stored.
           roi: tuple
               A tuple defining the region of interest (ROI) to extract from the frame (ymin, ymax, xmin, xmax) in pixels.

        """

        with h5py.File(path, 'r') as root:
            if dataset not in root:
                raise FileNotFoundError(f"Dataset '{dataset}' not found in file: {path}")

            if frame_num is None:
                dataset_shape = root[dataset].shape
                number_of_frames = root[dataset].shape[0] if len(dataset_shape) == 3 else 1
                if number_of_frames > self.batch_size:
                    print(
                        f"Number of frames ({number_of_frames}) is more than {self.batch_size}. The batch processing has been activated.")
                    self.activate_batch = True
                    self.number_of_frames = number_of_frames
                    return

                return root[dataset][:, roi[0], roi[1]].astype('float32') if len(dataset_shape) == 3 else root[dataset][roi[0], roi[1]].astype('float32')

            elif isinstance(frame_num, list):
                if len(frame_num) > self.batch_size:
                    print(
                        f"Number of frames is more than {self.batch_size}. The batch processing has been activated.")
                    self.activate_batch = True
                    self.number_of_frames = len(frame_num)
                    return
                return np.array(
                    [np.array(root[dataset][frame][roi[0], roi[1]]).astype('float32') for frame in frame_num])
            else:
                return np.array(root[dataset][frame_num][roi]).astype('float32')

    # def change_dim(self):
    #     if self.img_raw.ndim == 2:
    #         self.img_raw = np.expand_dims(self.img_raw, axis=0)
    #     shape = self.img_raw.shape
    #     if len(shape) == 4:
    #         new_shape = (shape[0] * shape[1], shape[2], shape[3])
    #         self.img_raw = self.img_raw.reshape(new_shape)

    def _reconstruct_lmbd_(self):
        """
        Stitches together individual detector image pieces into a full Lambda detector image (DESY).
        """

        if isinstance(self.path, list):
            translation = []
            data = []
            flatfield = []
            mask = []

            for file in self.path:
                with h5py.File(file, 'r') as root:
                    translation.append(root['entry/instrument/detector/translation/distance'][:])
                    data.append(root['entry/instrument/detector/data'][:].astype(np.float32))
                    flatfield.append(root['entry/instrument/detector/flatfield'][:].astype(np.float32))
                    mask.append(root['entry/instrument/detector/pixel_mask'][:])

            translation = np.array(translation).astype(np.int32)
            data_shape = np.array([d.shape for d in data])
            max_translation = translation.max(axis=0)
            max_translation = np.array(max_translation)[::-1]
            max_data_shape = data_shape.max(axis=0)

            lmbd_img = np.full((
                int(max_data_shape[0] + max_translation[0]),
                int(max_data_shape[1] + max_translation[1]),
                int(max_data_shape[2] + max_translation[2])
            ), -1, dtype=np.float32)


            for i, (d, ff, m, t) in enumerate(zip(data, flatfield, mask, translation)):
                t=t[::-1]
                slices = tuple(slice(t[j], t[j] + d.shape[j]) for j in range(3))
                lmbd_img[slices] = d * ff
                lmbd_img[slices][:, (m[0, :, :]).astype(bool)] = -1
            lmbd_img[lmbd_img<0] = np.nan
            image = fabio.tifimage.TifImage(lmbd_img[0])
            image.write("reconstructed_image.tiff")
            print("Reconstructed image saved to reconstructed_image.tiff")
            del translation, data, flatfield, mask
            return lmbd_img
        else:
            raise ValueError('path should be a list of strings, when build_image_P03 is True')

        return img_raw

def check_file_exists(filepath):
    """
    Checks if a file exists at the specified file path. If the file is not found, raises a `FileNotFoundError`.

    Parameters
    ----------
    filepath: str
        The path to the file that needs to be checked.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"The file '{filepath}' does not exist.")

