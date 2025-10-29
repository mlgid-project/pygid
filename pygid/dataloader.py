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
    Class for loading raw image data and applying optional preprocessing or transformations.

    This class provides a flexible interface for reading experimental image data from
    various file formats (e.g., HDF5, NXS, TIFF) and optionally processing subsets of
    frames, regions of interest (ROI), or batch analyses. It supports multiprocessing
    for faster data loading when handling large datasets.

    Attributes
    ----------
    path : str or list of str, optional
        Path or list of paths to the raw data files.
    dataset : str, optional
        Dataset root path within HDF5 or NXS files.
        Default is "measurement/eiger4m".
    frame_num : float or list of float, optional
        Specific frame number(s) to load. If None, all frames are processed.
    roi_range : list of float or None, optional
        Region of interest (ROI) specified as [left, right, bottom, top].
        If None or [None, None, None, None], the full image is used.
    batch_size : int, optional
        Number of frames per batch for batch processing. Default is 32.
    activate_batch : bool, optional
        If True, enables batch processing of data. Default is False.
    number_of_frames : int, optional
        Number of frames to load. If None, all available frames are processed.
        Automatically adjusted if greater than `batch_size`.
    multiprocessing : bool, optional
        If True, enables multiprocessing for parallelized data loading.
        Default is False.
    build_image_P03 : bool, optional
        Specific flag used for beamline P03 data reconstruction (if applicable).
        Default is False.
    fmt : str, optional
        File format specifier (e.g., 'h5', 'nxs', 'tiff'). If None, inferred from file extension.
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
    fmt: str = None

    def __post_init__(self):
        """Post-initialization method to load raw image data.

        This method is automatically executed after dataclass initialization.
        It determines the data source and loads the raw image data accordingly:
        - If `build_image_P03` is True, it reconstructs the image using the P03-specific method.
        - Otherwise, it loads the image(s) from the specified path.

        Notes:
            The resulting raw image data is stored in `self.img_raw`.
        """
        # Reconstruct image data if P03 beamline data format is used
        if self.build_image_P03:
            self.img_raw = self._reconstruct_lmbd_()
            return

        # Otherwise, load the image(s) from the provided file path
        self.img_raw = self._process_path_()

    def _process_path_(self):
        """
        Process the input path(s) and load corresponding image data.

        This method determines whether `path` refers to a single file (string) or multiple
        files (list). It then loads the image data accordingly. When multiple files are provided,
        it can optionally use multiprocessing for faster loading or activate batch analysis
        if the number of files exceeds the defined `batch_size`.

        Returns
        -------
        np.ndarray
            The loaded image data array. The array is reshaped to have dimensions
            (N, height, width), where N is the number of frames.
            Returns None if batch processing is activated.

        Raises
        ------
        FileNotFoundError
            If the `path` is neither a string nor a list of strings.

        Notes
        -----
        - Suppresses Fabio library warnings for cleaner output.
        - Automatically infers the file format from the first file extension.
        """
        # Suppress Fabio TIFF-related warnings for cleaner console output
        warnings.filterwarnings("ignore", category=UserWarning, module="fabio.TiffIO")
        logging.getLogger("fabio.TiffIO").setLevel(logging.ERROR)

        # Case 1: Single file input
        if isinstance(self.path, str):
            img_raw = self._image_loading_(path=self.path, frame_num=self.frame_num, dataset=self.dataset)
            return img_raw

        # Case 2: Multiple files input
        elif isinstance(self.path, list):
            # Infer file format from the first file extension
            fmt = os.path.splitext(self.path[0])[1][1:]
            self.fmt = fmt

            # Activate batch mode if number of files exceeds batch_size
            if len(self.path) > self.batch_size:
                print(f"Number of frames exceeds {self.batch_size}. Batch processing activated.")
                self.activate_batch = True
                self.number_of_frames = len(self.path)
                return

            # Load images using multiprocessing if enabled
            if self.multiprocessing:
                with ThreadPoolExecutor() as executor:
                    img_raw = list(executor.map(
                        lambda file: self._image_loading_(path=file, frame_num=self.frame_num, dataset=self.dataset),
                        self.path))
            else:
                # Sequential image loading
                img_raw = [
                    self._image_loading_(path=file, frame_num=self.frame_num, dataset=self.dataset)
                    for file in self.path
                ]

            # Convert list to array and reshape to (N, height, width)
            img_raw = np.array(img_raw)
            img_raw = img_raw.reshape(-1, img_raw.shape[2], img_raw.shape[3])
            return img_raw

        # Case 3: Invalid path type
        else:
            raise FileNotFoundError("Invalid path format. Expected str or list of str.")

    def _image_loading_(self, path=None, frame_num=None, dataset=None):
        """
        Load a single image file or dataset frame.

        This method loads raw image data from a given file path, automatically detecting
        the file format. It supports Fabio-compatible image formats (e.g., TIFF, CBF, EDF)
        as well as HDF5-based formats (e.g., `.h5`, `.hdf5`, `.nxs`). For HDF5 files,
        loading is delegated to `_load_with_h5py`. If a region of interest (ROI) is defined,
        only that subset of the frame is returned.

        Parameters
        ----------
        path : str
            Path to the image or dataset file.
        frame_num : int or None, optional
            Specific frame number to load from a multi-frame dataset.
            If None, loads the entire dataset or the first frame.
        dataset : str or None, optional
            Internal dataset path used for HDF5/NXS files.
            Ignored for Fabio-compatible image formats.

        Returns
        -------
        np.ndarray
            Loaded image data as a NumPy array with shape (N, height, width),
            where N is the number of frames (1 for single-frame images).

        Raises
        ------
        FileNotFoundError
            If the file does not exist or the format is unsupported.

        Notes
        -----
        - Fabio is used for non-HDF5 formats such as TIFF, CBF, or EDF.
        - ROI extraction is applied before returning the image data.
        """
        # Verify that the file exists before attempting to load it
        check_file_exists(path)

        # Detect the file format from the extension
        fmt = os.path.splitext(path)[1][1:]
        img_raw = None

        # Define the region of interest (ROI) using slice objects
        roi = (slice(self.roi_range[0], self.roi_range[1]),
               slice(self.roi_range[2], self.roi_range[3]))

        # Case 1: HDF5/NXS file — use the h5py-based loader
        if fmt in ['hdf5', 'h5', 'nxs']:
            img_raw = self._load_with_h5py(path, frame_num, dataset, roi)

        # Case 2: Fabio-compatible image format
        else:
            try:
                img_raw = fabio.open(path).data[roi[0], roi[1]].astype('float32')
            except Exception as e:
                raise FileNotFoundError(
                    "Invalid format. Supported formats: 'h5', 'nxs', 'tiff', 'cbf', 'edf'."
                ) from e

        # Handle missing or invalid data
        if img_raw is None:
            return

        # Ensure output has a consistent 3D shape: (N, height, width)
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

                return root[dataset][:, roi[0], roi[1]].astype('float32') if len(dataset_shape) == 3 else root[dataset][
                    roi[0], roi[1]].astype('float32')

            elif isinstance(frame_num, list) or isinstance(frame_num, np.ndarray):
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

    def _reconstruct_lmbd_(self):
        """
        Stitches together individual detector image pieces into a full Lambda detector image (DESY).
        """

        if not isinstance(self.path, list):
            raise ValueError('path should be a list of strings, when build_image_P03 is True')

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
            t = t[::-1]
            slices = tuple(slice(t[j], t[j] + d.shape[j]) for j in range(3))
            lmbd_img[slices] = d * ff
            lmbd_img[slices][:, (m[0, :, :]).astype(bool)] = -1
        lmbd_img[lmbd_img < 0] = np.nan
        image = fabio.tifimage.TifImage(lmbd_img[0])
        image.write("reconstructed_image.tiff")
        print("Reconstructed image saved to reconstructed_image.tiff")
        del translation, data, flatfield, mask
        return lmbd_img




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
