from . import CoordMaps
from . import DataLoader
from . import DataSaver, SampleMetadata, ExpMetadata
from .visualization import (get_plot_context, get_plot_params, plot_img_raw, _plot_single_image,
                            _plot_profile)
from .simulation import make_simulation_old, make_simulation_new
import os
from typing import Optional, Any
import numpy as np
import cv2
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.notebook import tqdm as log_progress
import warnings
import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)


from pygidsim.experiment import ExpParameters
from pygidsim.giwaxs_sim import GIWAXSFromCif



@dataclass
class Conversion:
    """
        A class that performs convesion of raw data and applies corrections.
        Takes data from DataLoader and sends tp DataSaver.

        Attributes:
        -----------
        matrix : CoordMaps, optional
            A 'CoordMaps' class instanse with coordinate and correction matrix.
        path : str, optional
            The path to the raw data file.
        dataset : str, optional
            The necessary dataset root in .h5 and .nxs files. Default is 'measurement/eiger4m'.
        frame_num : float, optional
            The specific frame number (or list of numbers) is dataset to process. Default is None (all frames).
        img_raw : np.array, optional
            The raw image data. Default is None.
        roi_range : list, optional
            The range of the region of interest (ROI) (left, right, down, up). Default is [None, None, None, None].
        average_all : bool, optional
            Averages all loaded frames. Default is False.
        sum_all : bool, optional
            Averages all loaded frames. Default is False.
        number_to_average : int, optional
            The number of frames to average before processing. Default is None (no average).
        number_to_sum : int, optional
            The number of frames to sum before processing. Default is None (no sum).
        use_gpu : bool, optional
            Whether to use GPU for computation. Default is True.
        multiprocessing : bool, optional
            Whether to use multiprocessing for convetion and coordinate maps calculation. Default is False.
        batch_size : int, optional
            The batch size for batch analysis. Default is 32.
        batch_activated: bool, optional
            Whether batch analysis is used. Default is False.

        Example:
        --------
        analysis = Conversion(matrix = matrix, path = "expamle.h5", dataset = '/6.1/measurement/eiger4m', average_all = False,
                              frame_num = 0, multiprocessing = False)

        """
    matrix: CoordMaps
    path: str = None
    dataset: str = 'measurement/eiger4m'
    frame_num: float = None
    img_raw: Optional[Any] = None
    average_all: bool = False
    sum_all: bool = False
    use_gpu: bool = True
    roi_range: list = field(default_factory=lambda: [None, None, None, None])
    multiprocessing: bool = False
    batch_size: int = 32
    path_batches: Any = None
    sub_class: Any = None
    frame_batches: Any = None
    number_to_average: int = None
    number_to_sum: int = None
    batch_activated: bool = False
    build_image_P03: bool = False
    global_frame_offset: int = 0  # Tracks global frame index offset for batch processing
    plot_params = get_plot_params()

    def __post_init__(self):
        """
        Initializes the object after dataclass creation.

        If no image is provided, this method automatically loads the data. It then applies
        flipping to the raw image, computes the q-range and angular range, generates
        correction maps, and applies the necessary corrections. Finally, it activates
        batch analysis if applicable.
        """

        if hasattr(self.matrix, "sub_matrices") and self.matrix.sub_matrices is not None:
            self.matrix_to_save = self.matrix
            self.matrix = self.matrix.sub_matrices
        self.matrix = [self.matrix] if not isinstance(self.matrix, list) else self.matrix
        self.params = self.matrix[0].params

        self.check_keys()


        if self.img_raw is None and self.path is None:
            # logging.info("img_raw or path should be specified")
            return

        if self.img_raw is not None:
            self.img_raw = np.array(self.img_raw)
            if self.img_raw.ndim == 2:
                self.img_raw = np.expand_dims(self.img_raw, axis=0)
            self.fmt = None
        else:
            loaded_data = DataLoader(path=self.path,
                                     frame_num=self.frame_num, dataset=self.dataset,
                                     roi_range=self.roi_range,
                                     batch_size=self.batch_size,
                                     multiprocessing=self.multiprocessing,
                                     build_image_P03=self.build_image_P03)
            self.fmt = loaded_data.fmt
            if loaded_data.activate_batch:
                self.batch_activated = True
                self.number_of_frames = loaded_data.number_of_frames
                return
            else:
                self.img_raw = loaded_data.img_raw
            del loaded_data

        self.update_conversion()

    def check_keys(self):
        if self.average_all and self.sum_all:
            raise ValueError("average_all and sum_all cannot be used at the same time")
        if (not self.number_to_average is None) and (not self.number_to_sum is None):
            raise ValueError("number_to_average and number_to_sum cannot be used at the same time")
        for num in (self.number_to_average,self.number_to_sum):
            if not num is None:
                if not (isinstance(num, int) and num > 0):
                    raise ValueError("number_to_average/number_to_sum must be positive integer")
        self.number_to_combine = self.number_to_average or self.number_to_sum

        if self.number_to_combine:
            if self.average_all:
                raise ValueError("average_all and number_to_average/number_to_sum cannot be used at the same time")
            if self.sum_all:
                raise ValueError("sum_all and number_to_average/number_to_sum cannot be used at the same time")

    def _adjust_batch_size(self):
        """
        Adjusts batch size to be compatible with number_to_combine.
        
        If number_to_combine is set, ensures batch_size is a multiple of it
        to avoid processing incomplete groups.
        """
        if self.number_to_combine>self.batch_size:
            raise ValueError("number_to_combine cannot be greater than batch size")
        if self.number_to_combine is not None:
            rest = self.batch_size % self.number_to_combine
            if rest != 0:
                self.batch_size -= rest

    def _create_path_batches(self):
        """
        Creates batches from a list of file paths.
        
        Returns
        -------
        list of list
            Batches of file paths.
        """
        return [self.path[i:i + self.batch_size] for i in range(0, len(self.path), self.batch_size)]

    def _create_frame_batches(self):
        """
        Creates batches from frame numbers in a single file.
        
        Returns
        -------
        list of list
            Batches of frame indices.
        """
        if isinstance(self.frame_num, list):
            batches = []
            for i in range(0, self.number_of_frames, self.batch_size):
                batches.append(self.frame_num[i:min(i + self.batch_size, len(self.frame_num))])
            return batches
        else:
            return [list(range(i, min(i + self.batch_size, self.number_of_frames)))
                    for i in range(0, self.number_of_frames, self.batch_size)]

    def _load_batch_images(self, batch_data, is_path_batch=True):
        """
        Loads raw images from a batch of paths or frames.
        
        Parameters
        ----------
        batch_data : list
            Either list of file paths or frame indices.
        is_path_batch : bool
            Whether batch_data contains paths (True) or frame indices (False).
        
        Returns
        -------
        np.ndarray
            Loaded raw image data.
        """
        loader_kwargs = {
            "frame_num": self.frame_num if is_path_batch else batch_data,
            "dataset": self.dataset,
            "roi_range": self.roi_range,
            "batch_size": self.batch_size,
            "multiprocessing": self.multiprocessing,
            "build_image_P03": self.build_image_P03
        }
        
        if is_path_batch:
            loader_kwargs["path"] = batch_data
        else:
            loader_kwargs["path"] = self.path
        
        return DataLoader(**loader_kwargs).img_raw

    def _aggregate_batch_images(self, batch_images):
        """
        Aggregates batch images using average or sum operation.
        
        Parameters
        ----------
        batch_images : np.ndarray
            Array of images to aggregate.
        
        Returns
        -------
        np.ndarray
            Aggregated image.
        """
        if self.average_all:
            return np.nanmean(batch_images, axis=0, keepdims=False)
        elif self.sum_all:
            return np.nansum(batch_images, axis=0, keepdims=False)
        return None

    def _process_aggregated_batches(self, batches, remap_func, is_path_batch, path_to_save, 
                                   h5_group, exp_metadata, smpl_metadata, overwrite_file, 
                                   overwrite_group, plot_result, return_result, save_result):
        """
        Processes batches with aggregation (average/sum) into a single result.
        
        Parameters
        ----------
        batches : list of list
            Batches of paths or frame indices.
        remap_func : str
            Name of the remapping function to call.
        is_path_batch : bool
            Whether batches contain paths or frame indices.
        ... (other parameters as per Batch signature)
        
        Returns
        -------
        result or None
            Result from remap function if return_result is True, otherwise None.
        """
        aggregated_images = []
        
        for batch in log_progress(batches, desc='Processing batches'):
            batch_img = self._load_batch_images(batch, is_path_batch)
            aggregated = self._aggregate_batch_images(batch_img)
            if aggregated is not None:
                aggregated_images.append(aggregated)
        
        # Set aggregated images and prepare for final remapping
        if aggregated_images:
            self.img_raw = np.array(aggregated_images)
            self.update_conversion()
            
            remap = getattr(self, remap_func, None)
            if remap is None:
                raise ValueError(f"Remapping function '{remap_func}' not found")
            
            self.batch_activated = False
            
            return remap(
                plot_result=plot_result,
                return_result=return_result,
                multiprocessing=False,
                save_result=save_result,
                overwrite_file=overwrite_file,
                overwrite_group=overwrite_group,
                exp_metadata=exp_metadata,
                smpl_metadata=smpl_metadata,
                path_to_save=path_to_save,
                h5_group=h5_group
            )
        
        return None

    def _process_individual_batches(self, batches, remap_func, is_path_batch, path_to_save, 
                                   h5_group, exp_metadata, smpl_metadata, overwrite_file, 
                                   overwrite_group, plot_result, return_result):
        """
        Processes batches individually, saving each one separately.
        
        Tracks global frame indices to ensure correct matrix selection across batches.

        Parameters
        ----------
        batches : list of list
            Batches of paths or frame indices.
        remap_func : str
            Name of the remapping function to call.
        is_path_batch : bool
            Whether batches contain paths or frame indices.
        ... (other parameters as per Batch signature)
        """
        _overwrite_file = overwrite_file
        _overwrite_group = overwrite_group
        _exp_metadata = exp_metadata
        _smpl_metadata = smpl_metadata
        
        global_frame_offset = 0  # Track global position across batches

        for batch_idx, batch in enumerate(log_progress(batches, desc='Processing batches')):
            # logging.info(f"Processing batch {batch_idx + 1}/{len(batches)} "
            #             f"(global frame offset: {global_frame_offset})")

            # Update frame_num for frame-based batches
            if not is_path_batch:
                self.frame_num = batch
                # For frame-based batches, set global offset for matrix selection
                self.global_frame_offset = global_frame_offset

            # Process batch
            batch_path = batch if is_path_batch else self.path
            self.process_batch(
                path_batch=batch_path,
                frame_num=None if is_path_batch else batch,
                remap_func=remap_func,
                overwrite_file=_overwrite_file,
                overwrite_group=_overwrite_group,
                exp_metadata=_exp_metadata,
                smpl_metadata=_smpl_metadata,
                path_to_save=path_to_save,
                h5_group=h5_group,
                global_frame_offset=global_frame_offset
            )

            # Update global offset for next batch
            if not is_path_batch:
                global_frame_offset += len(batch)
            else:
                # For path-based batches, update offset by the number of frames that were just processed
                # Use the AI list size as indicator of how many frames were loaded
                if hasattr(self, 'ai_list') and self.ai_list:
                    global_frame_offset += len(self.ai_list)
                else:
                    # Fallback: assume single frame per file in the batch
                    global_frame_offset += len(batch) if isinstance(batch, list) else 1

            # Only write metadata on first batch
            _overwrite_file = False
            _overwrite_group = False
            _exp_metadata = None
            _smpl_metadata = None
        
        # Reset state for frame-based batches
        if not is_path_batch:
            self.frame_num = None
            self.global_frame_offset = 0

        # Warn if unsupported options were used
        if plot_result or return_result:
            logging.getLogger().warning("Plotting and returning of the result are not supported in batch analysis mode.")

    def Batch(self, path_to_save, remap_func="det2q_gid", h5_group=None, exp_metadata=None, smpl_metadata=None,
              overwrite_file=True, overwrite_group=False,
              save_result=True, plot_result=False, return_result=False):
        """
        Divides raw images into batches and processes them separately.
        
        Two batching modes are supported:
        1. Path-based: Multiple files, each processed as a batch
        2. Frame-based: Single file with many frames, split into batches
        
        Supports aggregation (average/sum) of batches into a single result,
        or individual processing and saving of each batch.

        Parameters
        ----------
        path_to_save : str
            Path where the processed data will be saved.
        remap_func : str, optional
            Name of the remapping function to call. Default is "det2q_gid".
        h5_group : str or None, optional
            HDF5 group name under which to store results. Default is None.
        exp_metadata : ExpMetadata or None, optional
            Experimental metadata for first batch. Default is None.
        smpl_metadata : SampleMetadata or None, optional
            Sample metadata for first batch. Default is None.
        overwrite_file : bool, optional
            Whether to overwrite existing file. Default is True.
        overwrite_group : bool, optional
            Whether to overwrite existing group. Default is False.
        save_result : bool, optional
            Whether to save results. Default is True.
        plot_result : bool, optional
            Whether to plot results. Default is False.
        return_result : bool, optional
            Whether to return results. Default is False.
        
        Returns
        -------
        result or None
            Result from aggregated batch processing if return_result is True, otherwise None.
        """
        # Adjust batch size for compatibility
        self._adjust_batch_size()
        
        # Determine if path-based or frame-based batching
        is_path_batch = isinstance(self.path, list)
        
        # Create batches
        batches = self._create_path_batches() if is_path_batch else self._create_frame_batches()
        
        logging.info(f"Processing {len(batches)} batches using {'path-based' if is_path_batch else 'frame-based'} mode")
        
        # Process batches
        if self.average_all or self.sum_all:
            # Aggregate all batches into single result
            return self._process_aggregated_batches(
                batches, remap_func, is_path_batch, path_to_save, h5_group,
                exp_metadata, smpl_metadata, overwrite_file, overwrite_group,
                plot_result, return_result, save_result
            )
        else:
            # Process each batch individually
            self._process_individual_batches(
                batches, remap_func, is_path_batch, path_to_save, h5_group,
                exp_metadata, smpl_metadata, overwrite_file, overwrite_group,
                plot_result, return_result
            )

    def _load_batch_data(self, path_batch, frame_num):
        """
        Loads raw image data from a batch.
        
        Parameters
        ----------
        path_batch : str or list
            File path(s) to load.
        frame_num : int, list, or None
            Frame number(s) to load.
        
        Returns
        -------
        np.ndarray
            Loaded raw image data.
        """
        try:
            return DataLoader(
                path=path_batch,
                frame_num=frame_num,
                dataset=self.dataset,
                roi_range=self.roi_range,
                batch_size=self.batch_size,
                multiprocessing=self.multiprocessing,
                build_image_P03=self.build_image_P03
            ).img_raw
        except Exception as e:
            logging.error(f"Failed to load batch data: {e}")
            raise

    def _prepare_batch_metadata(self, exp_metadata, path_batch, size):
        """
        Prepares or creates batch metadata.
        
        Parameters
        ----------
        exp_metadata : ExpMetadata or None
            Existing metadata to update, or None to create new.
        path_batch : str or list
            Path or paths for this batch.
        
        Returns
        -------
        ExpMetadata
            Prepared metadata object.
        """
        if isinstance(path_batch, list):
            filename = path_batch
        else:
            filename = [path_batch]*size
        if exp_metadata is None:
            # Extract filename from path_batch
            exp_metadata = ExpMetadata(filename=filename)
        else:
            # Update filename
            exp_metadata.filename = filename
        
        return exp_metadata

    def _get_remap_function(self, remap_func):
        """
        Retrieves and validates the remapping function.
        
        Parameters
        ----------
        remap_func : str
            Name of the remapping function.
        
        Returns
        -------
        callable
            The remapping function.
        
        Raises
        ------
        ValueError
            If the remapping function is not found.
        """
        remap = getattr(self, remap_func, None)
        if remap is None:
            raise ValueError(f"Remapping function '{remap_func}' not found in {self.__class__.__name__}")
        return remap

    def _clean_batch_results(self):
        """
        Cleans up temporary result attributes after batch processing.
        
        Removes all cached result attributes to free memory and prevent conflicts.
        """
        result_attrs = [
            "img_gid_q", "img_q", "img_gid_pol", "img_pol",
            "img_gid_pseudopol", "img_pseudopol",
            "rad_cut", "azim_cut", "horiz_cut"
        ]
        
        count = 0
        for attr in result_attrs:
            if hasattr(self, attr):
                delattr(self, attr)
                count += 1
        
        if count > 0:
            logging.debug(f"Cleaned up {count} result attributes")

    def process_batch(
            self, path_batch, frame_num, remap_func, overwrite_file, overwrite_group,
            exp_metadata, smpl_metadata, path_to_save, h5_group, global_frame_offset=0
    ):
        """
        Processes a single batch of data: loads, converts, and saves results.
        
        Supports global frame index tracking for correct matrix selection in batch processing.

        Parameters
        ----------
        path_batch : str or list
            File path(s) for this batch.
        frame_num : int, list, or None
            Frame number(s) for this batch (local indices within batch).
        remap_func : str
            Name of the remapping function to call.
        overwrite_file : bool
            Whether to overwrite existing file.
        overwrite_group : bool
            Whether to overwrite existing group.
        exp_metadata : ExpMetadata or None
            Experiment metadata for this batch.
        smpl_metadata : SampleMetadata or None
            Sample metadata for this batch.
        path_to_save : str
            Path to save results.
        h5_group : str or None
            HDF5 group name.
        global_frame_offset : int, optional
            Global frame index offset for matrix selection during batch processing.
            Default is 0.
        """
        try:
            # Step 1: Load batch data
            logging.debug(f"Loading batch: {path_batch}")
            self.img_raw = self._load_batch_data(path_batch, frame_num)
            
            # Step 2: Mark batch processing as complete
            self.batch_activated = False
            
            # Step 3: Update conversion parameters
            logging.debug("Updating conversion parameters")
            self.update_conversion()
            
            # Step 4: Prepare metadata
            exp_metadata = self._prepare_batch_metadata(exp_metadata, path_batch, len(self.img_raw))
            
            # Step 5: Get remapping function
            remap = self._get_remap_function(remap_func)
            
            # Step 6: Set global frame offset for matrix selection
            self.global_frame_offset = global_frame_offset

            # Step 7: Execute remapping with global frame offset
            # logging.info(f"Remapping batch with {remap_func} (global offset: {global_frame_offset})")

            # Call remap function with proper parameters
            # The remap function will handle frame processing through _remap_general_
            # which will use global_frame_offset for matrix selection
            remap(
                plot_result=False,
                return_result=False,
                multiprocessing=self.multiprocessing,
                save_result=True,
                overwrite_file=overwrite_file,
                overwrite_group=overwrite_group,
                exp_metadata=exp_metadata,
                smpl_metadata=smpl_metadata,
                path_to_save=path_to_save,
                h5_group=h5_group
            )
            
            # Step 8: Clean up temporary data
            logging.debug("Cleaning up batch results")
            self.img_raw = None
            self._clean_batch_results()
            
        except Exception as e:
            logging.error(f"Error processing batch: {e}")
            raise

    def update_conversion(self):

        """
        Raw image peprocessing that includes averaging, flipping and masking. Call experimental parametes and coordinate
        maps update and application of corrections.

        """

        if self.average_all:
            self.img_raw = np.nanmean(self.img_raw, axis=0, keepdims=True)
        elif self.sum_all:
            self.img_raw = np.nansum(self.img_raw, axis=0, keepdims=True)
        elif self.number_to_combine is not None and self.number_to_combine > 1:
            num_images = len(self.img_raw)
            blocks = num_images // self.number_to_combine
            averaged_images = []
            for i in range(0, blocks * self.number_to_combine, self.number_to_combine):
                averaged_images.append(np.nanmean(self.img_raw[i:i + self.number_to_combine], axis=0))
            remaining = num_images % self.number_to_combine
            if remaining > 0:
                logging.getLogger().warning(f"{remaining} images left, averaging them separately.")
                averaged_images.append(np.mean(self.img_raw[-remaining:], axis=0))
            self.img_raw = np.array(averaged_images)

        self.img_raw = np.array([process_image(img, self.params.mask, self.params.flipud, self.params.fliplr,
                                               self.params.transp, self.roi_range, self.params.count_range) for
                                 img in self.img_raw])

        self.update_params()
        self.update_coordmaps()
        self._apply_corrections_()
        if self.frame_num is None:
            self.frame_num = np.array(range(len(self.img_raw)))
        if self.fmt in ["tif", "edf"]:
            self.frame_num *= 0

        self.x = np.linspace(0, self.img_raw.shape[2] - 1, self.img_raw.shape[2]) - self.params.centerX
        self.y = np.linspace(0, self.img_raw.shape[1] - 1, self.img_raw.shape[1]) - self.params.centerY

    def update_params(self):
        """
        Updates experimental parameters as image size and ROI is known.

        """

        if self.matrix[0].params.img_dim is None:
            self.matrix[0].params.img_dim = list(self.img_raw[0].shape)
            if self.matrix[0].params.poni1 is None:
                if self.roi_range[0]:
                    self.matrix[0].params.centerY -= self.roi_range[0]
                if self.roi_range[2]:
                    self.matrix[0].params.centerX -= self.roi_range[2]
            else:
                if self.roi_range[0]:
                    self.matrix[0].params.poni1 -= self.roi_range[0] * self.matrix[0].params.px_size
                if self.roi_range[2]:
                    self.matrix[0].params.poni2 -= self.roi_range[2] * self.matrix[0].params.px_size
            self.matrix[0].params._exp_params_update_()
        if len(self.matrix) != 1:
            for matrix in self.matrix:
                matrix.params = self.matrix[0].params

    def update_coordmaps(self):
        """
        Updates coordinate maps. Finds q- and angular ranges. Lower values are taken from corrdinate map with the lowest
        angle of incidence, and upper ranges are taken from corrdinate map with the highest. Normlize the ranges for all
        coordinate maps.

        """

        if len(self.matrix) == 1:
            if self.matrix[0].img_dim is None:
                self.matrix[0]._coordmaps_update_()
            return

        q_xy_ranges = [matrix.q_xy_range for matrix in self.matrix]
        q_z_ranges = [matrix.q_z_range for matrix in self.matrix]
        if any(q_xy is None for q_xy in q_xy_ranges) or any(q_z is None for q_z in q_z_ranges) or \
                any(q_xy != q_xy_ranges[0] for q_xy in q_xy_ranges) or \
                any(q_z != q_z_ranges[0] for q_z in q_z_ranges):

            q_xy_range, q_z_range = [], []
            ai_min_index = np.argmin([matrix.ai for matrix in self.matrix])
            self.matrix[ai_min_index]._coordmaps_update_()
            q_xy_range.append(self.matrix[ai_min_index].q_xy_range[0])
            q_z_range.append(self.matrix[ai_min_index].q_z_range[0])
            corr_matrices = self.matrix[ai_min_index].corr_matrices
            q = self.matrix[ai_min_index].q
            q_min = self.matrix[ai_min_index].radial_range[0]
            ang_min = self.matrix[ai_min_index].angular_range[0]
            ang_max = self.matrix[ai_min_index].angular_range[1]
            q_lab_from_p = self.matrix[ai_min_index].q_lab_from_p

            ai_max_index = np.argmax([matrix.ai for matrix in self.matrix])
            self.matrix[ai_max_index].corr_matrices = []
            self.matrix[ai_max_index].angular_range = (ang_min, ang_max)
            self.matrix[ai_max_index]._coordmaps_update_()
            q_xy_range.append(self.matrix[ai_max_index].q_xy_range[1])
            q_z_range.append(self.matrix[ai_max_index].q_z_range[1])
            q_max = self.matrix[ai_max_index].radial_range[1]

            for matrix in self.matrix:
                matrix.q_xy_range = q_xy_range
                matrix.q_z_range = q_z_range
                matrix.radial_range = (q_min, q_max)
                matrix.angular_range = (ang_min, ang_max)
                matrix.q = q
                matrix.corr_matrices = []
                matrix._coordmaps_update_()
                matrix.q_lab_from_p = q_lab_from_p
            self.matrix[0].corr_matrices = corr_matrices
        else:
            self.matrix[0]._coordmaps_update_()
            corr_matrices = self.matrix[0].corr_matrices
            for i in range(1, len(self.matrix)):
                self.matrix[i].corr_matrices = corr_matrices
                self.matrix[i]._coordmaps_update_()

    def _apply_corrections_(self):
        """
        Applies all calulated corrections. Only absorption_corr_matrix and lorentz_corr_matrix depend on  the angle
        of incidence.

        """
        corr_matrices = self.matrix[0].corr_matrices.__dict__
        if corr_matrices['dark_current'] is not None:
            for i in range(len(self.img_raw)):
                self.img_raw[i] -= corr_matrices['dark_current']
            logging.info("Dark current is subtracted")
        for corr_matrix in corr_matrices:
            if corr_matrix != 'dark_current' and corr_matrices[corr_matrix] is not None:
                if corr_matrix == 'absorption_corr_matrix' or corr_matrix == 'lorentz_corr_matrix':
                    for i, matrix in enumerate(self.matrix):
                        self.img_raw[i] /= matrix.corr_matrices.__dict__[corr_matrix]
                logging.info(f"{corr_matrix} was applied")
                self.img_raw /= corr_matrices[corr_matrix]

    def save_nxs(self, **kwargs):
        """
        Calls conveted data saving.

        Parameters
        ----------
        kwargs : tuple
            Turple with saving parametes like path_to_save, h5_group, overwrite_file and metadata.
        """

        DataSaver(self, **kwargs)
        return

    @classmethod
    def set_plot_defaults(cls, font_size=14, axes_titlesize=14, axes_labelsize=18, grid=False, grid_color='gray',
                          grid_linestyle='--', grid_linewidth=0.5, xtick_labelsize=14, ytick_labelsize=14,
                          legend_fontsize=12, legend_loc='best', legend_frameon=True, legend_borderpad=1.0,
                          legend_borderaxespad=1.0, figure_titlesize=16, figsize=(6.4, 4.8), axes_linewidth=0.5,
                          savefig_dpi=600, savefig_transparent=False, savefig_bbox_inches=None,
                          savefig_pad_inches=0.1, line_linewidth=2, line_color='blue', line_linestyle='-',
                          line_marker=None, scatter_marker='o', scatter_edgecolors='black',
                          cmap='inferno'):
        """
        Sets the default settings for various parts of a Matplotlib plot, including font sizes, gridlines,
        legend, figure properties, and line styles. The function configures the default style for future
        plots created with Matplotlib.

        Parameters:
        - font_size (int): Default font size for text elements (e.g., title, labels, ticks).
        - axes_titlesize (int): Font size for axes titles.
        - axes_labelsize (int): Font size for axes labels (x and y).
        - grid (bool): Whether or not to display gridlines (True/False).
        - grid_color (str): Color of the gridlines (e.g., 'gray', 'black').
        - grid_linestyle (str): Line style of the gridlines (e.g., '--', '-', ':').
        - grid_linewidth (float): Width of the gridlines.
        - xtick_labelsize (int): Font size for x-axis tick labels.
        - ytick_labelsize (int): Font size for y-axis tick labels.
        - legend_fontsize (int): Font size for the legend text.
        - legend_loc (str): Location of the legend (e.g., 'best', 'upper right', 'lower left').
        - legend_frameon (bool): Whether to display a frame around the legend.
        - legend_borderpad (float): Padding between the legend's content and the legend's frame.
        - legend_borderaxespad (float): Padding between the legend and axes.
        - figure_titlesize (int): Font size for the figure title.
        - figsize (tuple): Size of the figure in inches (e.g., (6, 6)).
        - savefig_dpi (int): DPI for saving the figure (higher DPI means better quality).
        - savefig_transparent (bool): Whether the saved figure should have a transparent background.
        - savefig_bbox_inches (str): Defines what part of the plot to save (e.g., 'tight' to crop extra whitespace).
        - savefig_pad_inches (float): Padding added around the figure when saving.
        - line_linewidth (float): Line width for plot lines.
        - line_color (str): Color of the plot lines (e.g., 'blue', 'red').
        - line_linestyle (str): Line style (e.g., '-', '--', ':').
        - line_marker (str): Marker style for plot lines (e.g., 'o', 'x').
        - scatter_marker (str): Marker style for scatter plots (e.g., 'o', 'x').
        - scatter_edgecolors (str): Color for the edges of scatter plot markers (e.g., 'black').
        - cmap (str): Image colormap
        """
        cls.plot_params.update(get_plot_params(font_size, axes_titlesize, axes_labelsize, grid, grid_color,
                                               grid_linestyle, grid_linewidth, xtick_labelsize,
                                               ytick_labelsize,
                                               legend_fontsize, legend_loc, legend_frameon, legend_borderpad,
                                               legend_borderaxespad, figure_titlesize, figsize,
                                               axes_linewidth,
                                               savefig_dpi, savefig_transparent, savefig_bbox_inches,
                                               savefig_pad_inches, line_linewidth, line_color, line_linestyle,
                                               line_marker, scatter_marker, scatter_edgecolors,
                                               cmap))
        # type(self).plot_params.update()

    def plot_raw_image(self, **kwargs):
        """
        Old naming of self.plot_img_raw() function
        """
        return self.plot_img_raw(**kwargs)

    def plot_img_raw(self, return_result=False, frame_num=None, plot_result=True,
                     clims=None, xlim=(None, None), ylim=(None, None), save_fig=False, path_to_save_fig="img.png"):
        """
        Plots the raw image from the detector with optional display, return and saving.

        Parameters
        ----------
        return_result : bool, optional
            If True, returns the image data and axes used for plotting. Default is False.
        frame_num : int or None, optional
            Frame number to plot. If None, uses the first frame.
        plot_result : bool, optional
            Whether to display the plot. Default is True.
        clims : tuple, optional
            Tuple specifying color limits (vmin, vmax) for the image. Default is (1e1, 4e4).
        xlim : tuple or None, optional
            Limits for the x-axis. If None, uses full range.
        ylim : tuple or None, optional
            Limits for the y-axis. If None, uses full range.
        save_fig : bool, optional
            Whether to save the figure to a file. Default is False.
        path_to_save_fig : str, optional
            Path to save the figure if save_fig is True. Default is "img.png".

        Returns
        -------
        x : array
            The x-axis values of the image (in pixels).
        y : array
            The y-axis values of the image (in pixels).
        img : 2D-array or list of 2D-arrays
            The raw image data plotted.
        """
        with get_plot_context(type(self).plot_params):
            return plot_img_raw(self.img_raw, self.x, self.y, return_result,
                            frame_num,
                            plot_result, clims, xlim, ylim,
                            save_fig, path_to_save_fig)

    def get_result(self, frame_num=None):
        """
        Returns axes and image(s) after the conversion.

        Parameters
        -----------
        frame_num : int or None, optional
            Frame number to return. If None, returns all frames. Default is None.

        Returns
        -------
        x : array
            The x-axis values of the image (in pixels).
        y : array
            The y-axis values of the image (in pixels).
        img : single 2D-array or 1D-array or list of arrays
            The converted image/profile.
        """
        key_maps = {
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
            "horiz_cut_gid": ["q_xy"],
            "vert_cut_gid": ["q_z"]
        }

        img, axes_labels = None, [None, None]
        for key in key_maps.keys():
            if hasattr(self, key):
                img = getattr(self, key)
                axes_labels = key_maps.get(key)
                break

        if img is None:
            raise ValueError('conversion should be called first')

        if isinstance(frame_num, int):
            img = np.array(img[frame_num])
        elif isinstance(frame_num, (list, tuple, np.ndarray)):
            img = np.array([img[i] for i in frame_num])
        elif frame_num is None:
            pass
        else:
            raise ValueError(
                "frame_num should be an integer, a sequence of integers, or None"
            )
        if len(img)==1:
            img=img[0]

        if len(axes_labels) == 2:
            x_key, y_key = tuple(axes_labels)
            x = getattr(self.matrix[0], x_key)
            y = getattr(self.matrix[0], y_key)
            return x, y, img
        else:
            x_key = axes_labels[0]
            x = getattr(self.matrix[0], x_key)
            return x, img


    def plot_result(self, return_result=False, frame_num=None, plot_result=True, shift=1,
                     clims=None, xlim=(None, None), ylim=(None, None), save_fig=False, path_to_save_fig="img_result.png"):
        """
        Plots the converted images/profiles with optional display, return and saving.

        Parameters
        ----------
        return_result : bool, optional
            If True, returns the image data and axes used for plotting. Default is False.
        frame_num : int or None, optional
            Frame number to plot. If None, uses the first frame.
        plot_result : bool, optional
            Whether to display the plot. Default is True.
        clims : tuple, optional
            Tuple specifying color limits (vmin, vmax) for the image. Default is (1e1, 4e4).
        xlim : tuple or None, optional
            Limits for the x-axis. If None, uses full range.
        ylim : tuple or None, optional
            Limits for the y-axis. If None, uses full range.
        save_fig : bool, optional
            Whether to save the figure to a file. Default is False.
        path_to_save_fig : str, optional
            Path to save the figure if save_fig is True. Default is "img.png".

        Returns
        -------
        x : array
            The x-axis values of the image (in pixels).
        y : array
            The y-axis values of the image (in pixels).
        img : list of 2D-array or 1D-arrays
            The converted image/profile plotted.
        """

        key_maps = {
            "img_gid_q": ["q_xy", "q_z", r'$q_{xy}$ [$\mathrm{\AA}^{-1}$]', r'$q_{z}$ [$\mathrm{\AA}^{-1}$]', 'equal'],
            "img_q": ["q_x", "q_y", r'$q_{y}$ [$\mathrm{\AA}^{-1}$]', r'$q_{y}$ [$\mathrm{\AA}^{-1}$]', 'equal'],
            "img_gid_pol": ["q_gid_pol", "ang_gid_pol", r"$|q|\ \mathrm{[\AA^{-1}]}$", r"$\chi$ [$\degree$]", 'auto'],
            "img_pol": ["q_pol", "ang_pol", r"$|q|\ \mathrm{[\AA^{-1}]}$", r"$\chi$ [$\degree$]", 'auto'],
            "img_gid_pseudopol": ["q_gid_rad", "q_gid_azimuth", r"$|q|\ \mathrm{[\AA^{-1}]}$", r"$q_{\phi}\ \mathrm{[\AA^{-1}]}$]", 'auto'],
            "img_pseudopol": ["q_rad", "q_azimuth", r"$|q|\ \mathrm{[\AA^{-1}]}$", r"$q_{\phi}\ \mathrm{[\AA^{-1}]}$]", 'auto'],
            "rad_cut": ["q_pol", r"$|q|\ \mathrm{[\AA^{-1}]}$"],
            "rad_cut_gid": ["q_gid_pol", r"$|q|\ \mathrm{[\AA^{-1}]}$"],
            "azim_cut": ["ang_pol", r"$\chi$ [$\degree$]"],
            "azim_cut_gid": ["ang_gid_pol", r"$\chi$ [$\degree$]"],
            "horiz_cut_gid": ["q_xy", r'$q_{xy}$ [$\mathrm{\AA}^{-1}$]'],
            "vert_cut_gid": ["q_z", r'$q_{z}$ [$\mathrm{\AA}^{-1}$]']
        }

        img, axes_labels = None, [None, None]
        for key in key_maps.keys():
            if hasattr(self, key):
                img = getattr(self, key)
                axes_labels = key_maps.get(key)
                break

        if img is None:
            raise ValueError('conversion should be called first')

        if frame_num is None:
            frame_num = list(range(len(img)))
        elif type(frame_num) is int:
            frame_num = [frame_num]

        if len(axes_labels) == 5:
            x_key, y_key, x_label, y_label, aspect = tuple(axes_labels)
            x = getattr(self.matrix[0], x_key)
            y = getattr(self.matrix[0], y_key)
            img_list = []
            for i in frame_num:
                _plot_single_image(get_plot_context(type(self).plot_params), img[i], x, y, clims, xlim, ylim,
                                   x_label,
                                   y_label, aspect, plot_result,
                                   save_fig, add_frame_number(path_to_save_fig, i))
                img_list.append(img)
            if return_result:
                return x, y, img_list

        elif len(axes_labels) == 2:
            x_key, x_label = tuple(axes_labels)
            x = getattr(self.matrix[0], x_key)
            img_list = [img[i] for i in frame_num]

            _plot_profile(plot_context = get_plot_context(type(self).plot_params),
                          x_values = x,
                          profiles = img_list,
                          xlabel = x_label,
                          shift = shift,
                          xlim = xlim,
                          ylim = ylim,
                          plot_result = plot_result,
                          save_fig = save_fig,
                          path_to_save_fig = path_to_save_fig)
            return x, img_list


    def _clean_previous_results(self):
        """
        Removes previous result attributes to avoid conflicts.
        
        Deletes all cached result attributes from previous conversions.
        """
        result_keys = [
            "img_gid_q", "img_q", "img_gid_pol",
            "img_pol", "img_gid_pseudopol", "img_pseudopol",
            "rad_cut", "azim_cut", "horiz_cut", "vert_cut"
        ]
        for key in result_keys:
            if hasattr(self, key):
                delattr(self, key)

    def _get_matrix_for_frame(self, frame_num, global_frame_index=None):
        """
        Retrieves the appropriate coordinate matrix for a given frame.
        
        Supports both local (within batch) and global (across all data) frame indexing.
        Uses global frame index when available for correct matrix selection during batch processing.

        Parameters
        ----------
        frame_num : int
            Frame index to get the matrix for (local index within batch or overall).
        global_frame_index : int or None, optional
            Global frame index across all data. Used during batch processing to select
            the correct matrix when processing multiple batches. If None, uses frame_num.

        Returns
        -------
        matrix : CoordMaps
            The coordinate matrix for the frame. If a single matrix is used for all frames,
            returns matrix[0]; otherwise returns matrix[index], where index is either
            global_frame_index (if provided) or frame_num.

        Raises
        ------
        IndexError
            If the frame index exceeds available matrices.
        """
        # Use global index if provided, otherwise use local frame_num
        index = global_frame_index if global_frame_index is not None else frame_num

        # Handle single matrix case (same for all frames)
        if len(self.matrix) == 1:
            return self.matrix[0]

        # Validate index
        if index >= len(self.matrix):
            logging.getLogger().warning(
                f"Frame index {index} exceeds available matrices ({len(self.matrix)}). "
                f"Using last matrix. This may indicate incorrect global_frame_index tracking."
            )
            return self.matrix[-1]

        if index < 0:
            logging.getLogger().warning(f"Negative frame index {index}. Using first matrix.")
            return self.matrix[0]

        return self.matrix[index]

    def _remap_frame_internal(self, img, mat, **kwargs):
        """
        Performs remapping for a single image with the given coordinate matrix.
        
        Parameters
        ----------
        img : np.ndarray
            Raw detector image to remap.
        mat : CoordMaps
            Coordinate transformation matrix.
        **kwargs : dict
            Dictionary containing transformation parameters:
            - p_y_key : str, attribute name for Y coordinates in matrix
            - p_x_key : str, attribute name for X coordinates in matrix
            - interp_type : str, interpolation method
            - multiprocessing : bool, whether multiprocessing is enabled
        
        Returns
        -------
        result_img : np.ndarray
            Remapped image data.
        """
        return self._remap_single_image_(
            img_raw=img,
            p_y=getattr(mat, kwargs["p_y_key"]),
            p_x=getattr(mat, kwargs["p_x_key"]),
            interp_type=kwargs["interp_type"],
            multiprocessing=kwargs["multiprocessing"],
        )

    def _build_ai_list(self, frame_num, global_frame_offset=0):
        """
        Builds a list of angle of incidence values for each frame.
        
        Parameters
        ----------
        frame_num : list or int
            Frame indices to process.
        global_frame_offset : int, optional
            Global frame index offset for batch processing. Default is 0.

        Returns
        -------
        ai_list : list
            List of angle of incidence values corresponding to each frame.
        """
        ai_list = []
        for frame in frame_num:
            if isinstance(self.params.ai, list) or isinstance(self.params.ai, np.ndarray):
                ai_list.append(self.params.ai[frame + global_frame_offset])
            else:
                ai_list.append(self.params.ai)
        return ai_list

    def _build_converted_frame_num(self, frame_num):
        """
        Builds a list of original frame numbers for converted frames.
        
        Parameters
        ----------
        frame_num : list or int
            Frame indices that were converted.
        
        Returns
        -------
        converted_frame_num : list
            List of original frame numbers (if available) or indices.
        """
        converted_frame_num = []
        if self.frame_num is None:
            converted_frame_num = frame_num
        else:
            for i in frame_num:
                if isinstance(self.frame_num, (int, np.int64)):
                    converted_frame_num.append(self.frame_num)
                else:
                    converted_frame_num.append(self.frame_num[i])
        return converted_frame_num

    def _process_frame_list_multiprocessing(self, frame_num, **kwargs):
        """
        Processes multiple frames using multiprocessing.
        
        Handles global frame index tracking for batch processing.

        Parameters
        ----------
        frame_num : list
            List of frame indices to process (local within batch or overall).
        **kwargs : dict
            Configuration dictionary for remapping.
            May include 'global_frame_index' for matrix selection context.

        Returns
        -------
        result_img : list
            List of remapped images.
        """
        result_img = []
        kwargs_copy = kwargs.copy()
        kwargs_copy["return_result"] = True
        kwargs_copy["save_result"] = False
        
        with ThreadPoolExecutor() as executor:
            # Use enumerate to track position in batch for global indexing
            futures = []
            for local_idx, frame in enumerate(frame_num):
                # Calculate global index if offset is provided
                if "global_frame_index" in kwargs:
                    kwargs_copy["global_frame_index"] = kwargs["global_frame_index"] + local_idx

                future = executor.submit(self._remap_general_, frame, **kwargs_copy)
                futures.append(future)

            # Collect results in order
            for future in futures:
                result_img.append(future.result()[2])

        return result_img

    def _process_frame_list_sequential(self, frame_num, **kwargs):
        """
        Processes multiple frames sequentially (without multiprocessing).
        
        Handles global frame index tracking for batch processing.

        Parameters
        ----------
        frame_num : list
            List of frame indices to process (local within batch or overall).
        **kwargs : dict
            Configuration dictionary for remapping.
            May include 'global_frame_index' for matrix selection context.

        Returns
        -------
        result_img : list
            List of remapped images.
        """
        result_img = []
        kwargs_copy = kwargs.copy()
        kwargs_copy["return_result"] = True
        kwargs_copy["save_result"] = False
        
        # Use enumerate to track position in batch for global indexing
        for local_idx, frame in enumerate(frame_num):
            # Calculate global index if offset is provided
            if "global_frame_index" in kwargs:
                kwargs_copy["global_frame_index"] = kwargs["global_frame_index"] + local_idx

            result_img.append(self._remap_general_(frame, **kwargs_copy)[2])
        
        return result_img

    def _process_frame_list(self, frame_num, **kwargs):
        """
        Handles processing of multiple frames with optional multiprocessing and result saving.
        
        Parameters
        ----------
        frame_num : list
            List of frame indices to process.
        **kwargs : dict
            Configuration dictionary containing transformation and saving parameters.
        
        Returns
        -------
        tuple or None
            If return_result is True, returns (matrix_x, matrix_y, result_img).
            Otherwise returns None.
        """
         # Process frames
        if kwargs["multiprocessing"]:
            result_img = self._process_frame_list_multiprocessing(frame_num, **kwargs)
        else:
            result_img = self._process_frame_list_sequential(frame_num, **kwargs)
        
        # Build metadata lists
        self.ai_list = self._build_ai_list(frame_num, global_frame_offset=self.global_frame_offset)
        self.converted_frame_num = self._build_converted_frame_num(frame_num)
        
        # Store results
        setattr(self, kwargs["result_attr"], result_img)
        
        # Save if requested
        if kwargs["save_result"]:
            self.save_nxs(
                path_to_save=kwargs["path_to_save"],
                h5_group=kwargs["h5_group"],
                overwrite_file=kwargs["overwrite_file"],
                overwrite_group=kwargs["overwrite_group"],
                exp_metadata=kwargs["exp_metadata"],
                smpl_metadata=kwargs["smpl_metadata"],
            )
        
        # Return if requested
        if kwargs["return_result"]:
            matrix_x = getattr(self.matrix[0], kwargs["x_key"])
            matrix_y = getattr(self.matrix[0], kwargs["y_key"])
            return matrix_x, matrix_y, result_img
        
        return None

    def _process_single_frame(self, frame_num, **kwargs):
        """
        Handles processing of a single frame with optional result saving.
        
        Parameters
        ----------
        frame_num : int
            Frame index to process (local within batch or overall).
        **kwargs : dict
            Configuration dictionary containing transformation and saving parameters.
            May include 'global_frame_index' for batch processing context.

        Returns
        -------
        tuple or None
            If return_result is True, returns (matrix_x, matrix_y, result_img).
            Otherwise returns None.
        """
        img = self.img_raw[frame_num]

        # Get global frame index for matrix selection (used in batch processing)
        global_frame_idx = kwargs.get("global_frame_index", None)
        mat = self._get_matrix_for_frame(frame_num, global_frame_index=global_frame_idx)
        result_img = self._remap_frame_internal(img, mat, **kwargs)
        
        # Set metadata
        self.ai_list = mat.ai
        self.converted_frame_num = [self.frame_num] if hasattr(self, 'frame_num') else [frame_num]
        
        # Store results
        setattr(self, kwargs["result_attr"], [result_img])
        
        # Save if requested
        if kwargs["save_result"]:
            self.save_nxs(
                path_to_save=kwargs["path_to_save"],
                h5_group=kwargs["h5_group"],
                overwrite_file=kwargs["overwrite_file"],
                overwrite_group=kwargs["overwrite_group"],
                exp_metadata=kwargs["exp_metadata"],
                smpl_metadata=kwargs["smpl_metadata"],
            )
        
        # Return if requested
        if kwargs["return_result"]:
            return getattr(mat, kwargs["x_key"]), getattr(mat, kwargs["y_key"]), result_img
        
        return None

    def _remap_general_(self, frame_num, **kwargs):
        """
        Chooses a coordinate matrix for the given frame and calls remapping. Activates multiprocessing if True.

        Plots the raw image from the detector with optional display, return and saving.

        Parameters
        ----------
        frame_num : int or list, optional
            Frame number to plot. If None, uses the first frame.
        kwargs: dict
            A dictionary with saving parameters.
            Can include 'global_frame_index' for batch processing context.
        """
        # Clean previous results
        self._clean_previous_results()
        
        # Default to all frames if not specified
        if frame_num is None:
            frame_num = list(range(len(self.img_raw)))
        
        # Set up global frame index for matrix selection if not already provided
        if "global_frame_index" not in kwargs and self.global_frame_offset > 0:
            if isinstance(frame_num, list):
                kwargs["global_frame_index"] = self.global_frame_offset
            else:
                kwargs["global_frame_index"] = self.global_frame_offset + frame_num

        # Process based on frame_num type
        if isinstance(frame_num, list):
            return self._process_frame_list(frame_num, **kwargs)
        else:
            return self._process_single_frame(frame_num, **kwargs)

    def det2q_gid(
            self,
            frame_num=None,
            interp_type="INTER_LINEAR",
            multiprocessing=None,
            return_result=False,
            q_xy_range=None,
            q_z_range=None,
            dq=None,
            plot_result=False,
            clims=None,
            xlim=(None, None),
            ylim=(None, None),
            save_fig=False,
            path_to_save_fig="img.png",
            save_result=False,
            path_to_save='result.h5',
            h5_group=None,
            overwrite_file=True,
            overwrite_group=False,
            exp_metadata=None,
            smpl_metadata=None,
    ):
        """
        Converts a detector image to a reciprocal-space map (q_xy, q_z) for grazing-incidence diffraction (GID) geometry.

        Parameters
        ----------
        frame_num : int, list, or None, optional
            Frame index or list of indices to process. If None, the first or current frame is used.
        interp_type : str, optional
            Interpolation method used for remapping. Default is "INTER_LINEAR".
        multiprocessing : bool or None, optional
            Whether to use multiprocessing during computation. If None, the class default is used.
        return_result : bool, optional
            If True, returns the calculated reciprocal-space axes and image(s).
        q_xy_range : tuple of float or None, optional
            (min, max) limits for the q_xy range. If None, the full range is used.
        q_z_range : tuple of float or None, optional
            (min, max) limits for the q_z range. If None, the full range is used.
        dq : float or None, optional
            Step size in reciprocal space (Δq). If None, the existing resolution is used.
        plot_result : bool, optional
            If True, displays the resulting reciprocal-space map. Default is False.
        clims : tuple of float or None, optional
            Color scale limits (vmin, vmax) for plotting. Default is None.
        xlim : tuple, optional
            X-axis limits for the plot. Default is (None, None).
        ylim : tuple, optional
            Y-axis limits for the plot. Default is (None, None).
        save_fig : bool, optional
            If True, saves the plotted figure. Default is False.
        path_to_save_fig : str, optional
            Path to save the figure if save_fig is True. Default is "img.png".
        save_result : bool, optional
            If True, saves the resulting data to an HDF5 file. Default is False.
        path_to_save : str, optional
            Path to save the HDF5 file if `save_result` is True. Default is "result.h5".
        h5_group : str or None, optional
            HDF5 group name under which the data are stored. Default is None.
        overwrite_file : bool, optional
            If True, overwrites an existing HDF5 file. Default is True.
        overwrite_group : bool, optional
            If True, overwrites an existing group within the HDF5 file. Default is False.
        exp_metadata : pygid.ExpMetadata or None, optional
            Experimental metadata to be stored with the result. Default is None.
        smpl_metadata : pygid.SampleMetadata or None, optional
            Sample-related metadata to be stored with the result. Default is None.

        Returns
        -------
        q_xy : ndarray
            The q_xy-axis values of the converted data (Å⁻¹).
        q_z : ndarray
            The q_z-axis values of the converted data (Å⁻¹).
        img_gid_q : ndarray or list of ndarray
            The reciprocal-space image(s) corresponding to (q_xy, q_z).
        """

        # If batch mode is active, delegate the task to the batch processor
        if self.batch_activated:
            res = self.Batch(path_to_save, "det2q_gid", h5_group, exp_metadata, smpl_metadata, overwrite_file,
                             overwrite_group, save_result, plot_result, return_result)
            self.batch_activated = True
            return res

        # Determine whether recalculation of transformation matrices is required
        recalc = (determine_recalc_key(q_xy_range, self.matrix[0].q_xy_range, self.matrix[0].q_xy, self.matrix[0].dq) \
                      if hasattr(self.matrix[0], "q_xy") else True) or (
                     determine_recalc_key(q_z_range, self.matrix[0].q_z_range,
                                          self.matrix[0].q_z, self.matrix[0].dq) \
                         if hasattr(self.matrix[0], "q_z") else True)
        # Force recalculation if dq (step size) differs from the current one
        if dq is not None:
            recalc = True if dq != self.matrix[0].dq else recalc

        # Calculate coordinate transformation matrices
        self.calc_matrices("p_y_gid", recalc, multiprocessing=multiprocessing or self.multiprocessing,
                           q_xy_range=q_xy_range,
                           q_z_range=q_z_range, dq=dq)

        # Remap detector image from pixel to reciprocal space (q_xy, q_z)
        x, y, img = self._remap_general_(
            frame_num,
            p_y_key="p_y_gid",
            p_x_key="p_x_gid",
            x_key="q_xy",
            y_key="q_z",
            result_attr="img_gid_q",
            interp_type=interp_type,
            multiprocessing=multiprocessing,
            return_result=True,
            save_result=save_result,
            path_to_save=path_to_save,
            h5_group=h5_group,
            overwrite_file=overwrite_file,
            overwrite_group=overwrite_group,
            exp_metadata=exp_metadata,
            smpl_metadata=smpl_metadata)

        # Ensure result is always a list (for consistent handling of multiple frames)
        img = [img] if not isinstance(img, list) else img
        if plot_result or save_fig:
            for i in range(len(img)):
                _plot_single_image(get_plot_context(type(self).plot_params), img[i], x, y, clims, xlim, ylim,
                                   r'$q_{xy}$ [$\mathrm{\AA}^{-1}$]',
                                   r'$q_{z}$ [$\mathrm{\AA}^{-1}$]', 'equal', plot_result,
                                   save_fig, add_frame_number(path_to_save_fig, i))
        # Return calculated axes and image(s) if required
        if return_result:
            return x, y, img

    def det2q(
            self,
            frame_num=None,
            interp_type="INTER_LINEAR",
            multiprocessing=None,
            return_result=False,
            q_x_range=None,
            q_y_range=None,
            dq=None,
            plot_result=False,
            clims=None,
            xlim=(None, None),
            ylim=(None, None),
            save_fig=False,
            path_to_save_fig="img.png",
            save_result=False,
            path_to_save='result.h5',
            h5_group=None,
            overwrite_file=True,
            overwrite_group=False,
            exp_metadata=None,
            smpl_metadata=None,
    ):

        """
        Converts a detector image to a reciprocal-space map (q_x, q_y) for transmission geometry.

        Parameters
        ----------
        frame_num : int, list, or None, optional
            Frame index or list of indices to process. If None, the first or current frame is used.
        interp_type : str, optional
            Interpolation method used for remapping. Default is "INTER_LINEAR".
        multiprocessing : bool or None, optional
            Whether to use multiprocessing during computation. If None, the class default is used.
        return_result : bool, optional
            If True, returns the calculated reciprocal-space axes and image(s).
        q_x_range : tuple of float or None, optional
            (min, max) limits for the q_x range. If None, the full range is used.
        q_y_range : tuple of float or None, optional
            (min, max) limits for the q_y range. If None, the full range is used.
        dq : float or None, optional
            Step size in reciprocal space (Δq). If None, the existing resolution is used.
        plot_result : bool, optional
            If True, displays the resulting reciprocal-space map. Default is False.
        clims : tuple of float or None, optional
            Color scale limits (vmin, vmax) for plotting. Default is None.
        xlim : tuple, optional
            X-axis limits for the plot. Default is (None, None).
        ylim : tuple, optional
            Y-axis limits for the plot. Default is (None, None).
        save_fig : bool, optional
            If True, saves the plotted figure. Default is False.
        path_to_save_fig : str, optional
            Path to save the figure if save_fig is True. Default is "img.png".
        save_result : bool, optional
            If True, saves the resulting data to an HDF5 file. Default is False.
        path_to_save : str, optional
            Path to save the HDF5 file if `save_result` is True. Default is "result.h5".
        h5_group : str or None, optional
            HDF5 group name under which the data are stored. Default is None.
        overwrite_file : bool, optional
            If True, overwrites an existing HDF5 file. Default is True.
        overwrite_group : bool, optional
            If True, overwrites an existing group within the HDF5 file. Default is False.
        exp_metadata : pygid.ExpMetadata or None, optional
            Experimental metadata to be stored with the result. Default is None.
        smpl_metadata : pygid.SampleMetadata or None, optional
            Sample-related metadata to be stored with the result. Default is None.

        Returns
        -------
        q_x : ndarray
            The q_x-axis values of the converted data (Å⁻¹).
        q_y : ndarray
            The q_y-axis values of the converted data (Å⁻¹).
        img_q : ndarray or list of ndarray
            The reciprocal-space image(s) corresponding to (q_x, q_y).
        """
        # If batch mode is active, delegate execution to the batch processor
        if self.batch_activated:
            res = self.Batch(path_to_save, "det2q", h5_group, exp_metadata, smpl_metadata, overwrite_file,
                             overwrite_group,
                             save_result, plot_result, return_result)
            self.batch_activated = True
            return res

        # Determine if coordinate matrices need to be recalculated
        recalc = (determine_recalc_key(q_x_range, self.matrix[0].q_x_range, self.matrix[0].q_x, self.matrix[0].dq) \
                      if hasattr(self.matrix[0], "q_x") else True) or (
                     determine_recalc_key(q_y_range, self.matrix[0].q_y_range,
                                          self.matrix[0].q_y, self.matrix[0].dq) \
                         if hasattr(self.matrix[0], "q_y") else True)

        if dq is not None:
            recalc = True if dq != self.matrix[0].dq else recalc

        # Compute coordinate transformation matrices for transmission geometry
        self.calc_matrices("p_y_ewald", recalc, multiprocessing=multiprocessing or self.multiprocessing,
                           q_x_range=q_x_range, q_y_range=q_y_range, dq=dq)
        # Remap detector image from pixel space to reciprocal space (q_x, q_y)
        x, y, img = self._remap_general_(
            frame_num,
            p_y_key="p_y_ewald",
            p_x_key="p_x_ewald",
            x_key="q_x",
            y_key="q_y",
            result_attr="img_q",
            interp_type=interp_type,
            multiprocessing=multiprocessing,
            return_result=True,
            save_result=save_result,
            path_to_save=path_to_save,
            h5_group=h5_group,
            overwrite_file=overwrite_file,
            overwrite_group=overwrite_group,
            exp_metadata=exp_metadata,
            smpl_metadata=smpl_metadata)

        img = [img] if not isinstance(img, list) else img

        # Plot and/or save reciprocal-space maps if requested
        if plot_result or save_fig:
            for i in range(len(img)):
                _plot_single_image(get_plot_context(type(self).plot_params), img[i], x, y, clims, xlim, ylim,
                                   r'$q_{x}$ [$\mathrm{\AA}^{-1}$]',
                                   r'$q_{y}$ [$\mathrm{\AA}^{-1}$]', 'equal', plot_result,
                                   save_fig, add_frame_number(path_to_save_fig, i))
        # Return calculated axes and reciprocal-space image(s) if requested
        if return_result:
            return x, y, img

    def det2pol(
            self,
            frame_num=None,
            interp_type="INTER_LINEAR",
            multiprocessing=None,
            return_result=False,
            radial_range=None,
            angular_range=None,
            dang=None,
            dq=None,
            plot_result=False,
            clims=None,
            xlim=(None, None),
            ylim=(None, None),
            save_fig=False,
            path_to_save_fig="img.png",
            save_result=False,
            path_to_save='result.h5',
            h5_group=None,
            overwrite_file=True,
            overwrite_group=False,
            exp_metadata=None,
            smpl_metadata=None,
    ):
        """
        Converts a detector image to a polar reciprocal-space map (|q|, χ) for transmission geometry.

        Parameters
        ----------
        frame_num : int, list, or None, optional
            Frame index or list of indices to process. If None, the first or current frame is used.
        interp_type : str, optional
            Interpolation method used for remapping. Default is "INTER_LINEAR".
        multiprocessing : bool or None, optional
            Whether to use multiprocessing during computation. If None, the class default is used.
        return_result : bool, optional
            If True, returns the calculated reciprocal-space axes and image(s).
        radial_range : tuple of float or None, optional
            (min, max) limits for the radial q range (|q|). If None, the full range is used.
        angular_range : tuple of float or None, optional
            (min, max) limits for the azimuthal angle χ (in degrees). If None, the full range is used.
        dang : float or None, optional
            Step size for the angular coordinate (Δχ). If None, the existing resolution is used.
        dq : float or None, optional
            Step size in reciprocal space (Δq). If None, the existing resolution is used.
        plot_result : bool, optional
            If True, displays the resulting polar reciprocal-space map. Default is False.
        clims : tuple of float or None, optional
            Color scale limits (vmin, vmax) for plotting. Default is None.
        xlim : tuple, optional
            X-axis limits for the plot. Default is (None, None).
        ylim : tuple, optional
            Y-axis limits for the plot. Default is (None, None).
        save_fig : bool, optional
            If True, saves the plotted figure. Default is False.
        path_to_save_fig : str, optional
            Path to save the figure if save_fig is True. Default is "img.png".
        save_result : bool, optional
            If True, saves the resulting data to an HDF5 file. Default is False.
        path_to_save : str, optional
            Path to save the HDF5 file if `save_result` is True. Default is "result.h5".
        h5_group : str or None, optional
            HDF5 group name under which the data are stored. Default is None.
        overwrite_file : bool, optional
            If True, overwrites an existing HDF5 file. Default is True.
        overwrite_group : bool, optional
            If True, overwrites an existing group within the HDF5 file. Default is False.
        exp_metadata : pygid.ExpMetadata or None, optional
            Experimental metadata to be stored with the result. Default is None.
        smpl_metadata : pygid.SampleMetadata or None, optional
            Sample-related metadata to be stored with the result. Default is None.

        Returns
        -------
        q_pol : ndarray
            The radial q-axis values of the converted data (Å⁻¹).
        ang_pol : ndarray
            The azimuthal angle χ values of the converted data (degrees).
        img_pol : ndarray or list of ndarray
            The polar reciprocal-space image(s) corresponding to (|q|, χ).
        """

        # If batch mode is active, delegate the task to the batch processor
        if self.batch_activated:
            res = self.Batch(path_to_save, "det2pol", h5_group, exp_metadata, smpl_metadata, overwrite_file,
                             overwrite_group,
                             save_result, plot_result, return_result)
            self.batch_activated = True
            return res

        # Determine whether recalculation of coordinate transformation matrices is required
        recalc = ((determine_recalc_key(angular_range, self.matrix[0].angular_range,
                                        self.matrix[0].ang_pol, self.matrix[0].dang) \
                       if hasattr(self.matrix[0], "ang_pol") else True) or
                  (determine_recalc_key(radial_range, self.matrix[0].radial_range,
                                        self.matrix[0].q_pol, self.matrix[0].dq) \
                       if hasattr(self.matrix[0], "q_pol") else True))
        if dq is not None:
            recalc = True if dq != self.matrix[0].dq else recalc
        if dang is not None:
            recalc = True if dang != self.matrix[0].dang else recalc

        # Compute polar transformation matrices (|q|, χ mapping)
        self.calc_matrices("p_y_lab_pol", recalc, multiprocessing=multiprocessing or self.multiprocessing,
                           radial_range=radial_range,
                           angular_range=angular_range, dang=dang, dq=dq)

        # Remap detector image from pixel space to polar reciprocal space (|q|, χ)
        x, y, img = self._remap_general_(
            frame_num,
            p_y_key="p_y_lab_pol",
            p_x_key="p_x_lab_pol",
            x_key="q_pol",
            y_key="ang_pol",
            result_attr="img_pol",
            interp_type=interp_type,
            multiprocessing=multiprocessing,
            return_result=True,
            save_result=save_result,
            path_to_save=path_to_save,
            h5_group=h5_group,
            overwrite_file=overwrite_file,
            overwrite_group=overwrite_group,
            exp_metadata=exp_metadata,
            smpl_metadata=smpl_metadata)

        img = [img] if not isinstance(img, list) else img

        # Plot and/or save polar reciprocal-space maps if requested
        if plot_result or save_fig:
            for i in range(len(img)):
                _plot_single_image(get_plot_context(type(self).plot_params), img[i], x, y, clims, xlim, ylim,
                                   r"$|q|\ \mathrm{[\AA^{-1}]}$", r"$\chi$ [$\degree$]", 'auto', plot_result,
                                   save_fig, add_frame_number(path_to_save_fig, i))
        # Return calculated axes and polar image(s) if requested
        if return_result:
            return x, y, img

    def det2pol_gid(
            self,
            frame_num=None,
            interp_type="INTER_LINEAR",
            multiprocessing=None,
            return_result=False,
            radial_range=None,
            angular_range=None,
            dang=None,
            dq=None,
            plot_result=False,
            clims=None,
            xlim=(None, None),
            ylim=(None, None),
            save_fig=False,
            path_to_save_fig="img.png",
            save_result=False,
            path_to_save='result.h5',
            h5_group=None,
            overwrite_file=True,
            overwrite_group=False,
            exp_metadata=None,
            smpl_metadata=None,
    ):
        """
        Converts a detector image to a polar reciprocal-space map (|q|, χ) for grazing-incidence diffraction (GID) geometry.

        Parameters
        ----------
        frame_num : int, list, or None, optional
            Frame index or list of indices to process. If None, the first or current frame is used.
        interp_type : str, optional
            Interpolation method used for remapping. Default is "INTER_LINEAR".
        multiprocessing : bool or None, optional
            Whether to use multiprocessing during computation. If None, the class default is used.
        return_result : bool, optional
            If True, returns the calculated reciprocal-space axes and image(s).
        radial_range : tuple of float or None, optional
            (min, max) limits for the radial q range (|q|). If None, the full range is used.
        angular_range : tuple of float or None, optional
            (min, max) limits for the azimuthal angle χ (in degrees). If None, the full range is used.
        dang : float or None, optional
            Step size for the angular coordinate (Δχ). If None, the existing resolution is used.
        dq : float or None, optional
            Step size in reciprocal space (Δq). If None, the existing resolution is used.
        plot_result : bool, optional
            If True, displays the resulting polar reciprocal-space map. Default is False.
        clims : tuple of float or None, optional
            Color scale limits (vmin, vmax) for plotting. Default is None.
        xlim : tuple, optional
            X-axis limits for the plot. Default is (None, None).
        ylim : tuple, optional
            Y-axis limits for the plot. Default is (None, None).
        save_fig : bool, optional
            If True, saves the plotted figure. Default is False.
        path_to_save_fig : str, optional
            Path to save the figure if save_fig is True. Default is "img.png".
        save_result : bool, optional
            If True, saves the resulting data to an HDF5 file. Default is False.
        path_to_save : str, optional
            Path to save the HDF5 file if `save_result` is True. Default is "result.h5".
        h5_group : str or None, optional
            HDF5 group name under which the data are stored. Default is None.
        overwrite_file : bool, optional
            If True, overwrites an existing HDF5 file. Default is True.
        overwrite_group : bool, optional
            If True, overwrites an existing group within the HDF5 file. Default is False.
        exp_metadata : pygid.ExpMetadata or None, optional
            Experimental metadata to be stored with the result. Default is None.
        smpl_metadata : pygid.SampleMetadata or None, optional
            Sample-related metadata to be stored with the result. Default is None.

        Returns
        -------
        q_gid_pol : ndarray
            The radial q-axis values of the converted data (Å⁻¹).
        ang_gid_pol : ndarray
            The azimuthal angle χ values of the converted data (degrees).
        img_gid_pol : ndarray or list of ndarray
            The polar reciprocal-space image(s) corresponding to (|q|, χ).
        """

        # If batch mode is active, delegate execution to the batch processor
        if self.batch_activated:
            res = self.Batch(path_to_save, "det2pol_gid", h5_group, exp_metadata, smpl_metadata, overwrite_file,
                             overwrite_group, save_result, plot_result, return_result)
            self.batch_activated = True
            return res

        # Determine whether recalculation of GID polar coordinate matrices is required
        recalc = ((determine_recalc_key(angular_range, self.matrix[0].angular_range,
                                        self.matrix[0].ang_gid_pol, self.matrix[0].dang) \
                       if hasattr(self.matrix[0], "ang_gid_pol") else True) or
                  (determine_recalc_key(radial_range, self.matrix[0].radial_range,
                                        self.matrix[0].q_gid_pol, self.matrix[0].dq) \
                       if hasattr(self.matrix[0], "q_gid_pol") else True))
        if dq is not None:
            recalc = True if dq != self.matrix[0].dq else recalc
        if dang is not None:
            recalc = True if dang != self.matrix[0].dang else recalc

        # Compute polar transformation matrices for GID geometry (|q|, χ mapping)
        self.calc_matrices("p_y_smpl_pol", recalc, multiprocessing=multiprocessing or self.multiprocessing,
                           radial_range=radial_range,
                           angular_range=angular_range, dang=dang, dq=dq)

        # Remap detector image from pixel space to polar reciprocal space (|q|, χ)
        x, y, img = self._remap_general_(
            frame_num,
            p_y_key="p_y_smpl_pol",
            p_x_key="p_x_smpl_pol",
            x_key="q_gid_pol",
            y_key="ang_gid_pol",
            result_attr="img_gid_pol",
            interp_type=interp_type,
            multiprocessing=multiprocessing,
            return_result=True,
            save_result=save_result,
            path_to_save=path_to_save,
            h5_group=h5_group,
            overwrite_file=overwrite_file,
            overwrite_group=overwrite_group,
            exp_metadata=exp_metadata,
            smpl_metadata=smpl_metadata)

        img = [img] if not isinstance(img, list) else img

        # Plot and/or save each polar GID map if requested
        if plot_result or save_fig:
            for i in range(len(img)):
                _plot_single_image(get_plot_context(type(self).plot_params), img[i], x, y, clims, xlim, ylim,
                                   r"$|q|\ \mathrm{[\AA^{-1}]}$", r"$\chi$ [$\degree$]", 'auto', plot_result,
                                   save_fig, add_frame_number(path_to_save_fig, i))
        # Return calculated axes and polar GID image(s) if requested
        if return_result:
            return x, y, img

    def det2pseudopol(
            self,
            frame_num=None,
            interp_type="INTER_LINEAR",
            multiprocessing=None,
            return_result=False,
            q_azimuth_range=None,
            q_rad_range=None,
            dang=None,
            dq=None,
            plot_result=False,
            clims=None,
            xlim=(None, None), ylim=(None, None),
            save_fig=False,
            path_to_save_fig="img.png",
            save_result=False,
            path_to_save='result.h5',
            h5_group=None,
            overwrite_file=True,
            overwrite_group=False,
            exp_metadata=None,
            smpl_metadata=None,
    ):
        """
        Converts a detector image to pseudopolar coordinates (q_rad, q_azimuth) for transmission geometry.

        Parameters
        ----------
        frame_num : int, list, or None, optional
            Frame index or list of indices to process. If None, the first or current frame is used.
        interp_type : str, optional
            Interpolation method used for remapping. Default is "INTER_LINEAR".
        multiprocessing : bool or None, optional
            Whether to use multiprocessing during computation. If None, the class default is used.
        return_result : bool, optional
            If True, returns the calculated axes and image(s).
        q_rad_range : tuple of float or None, optional
            (min, max) limits for the radial q-axis. If None, the full range is used.
        q_azimuth_range : tuple of float or None, optional
            (min, max) limits for the azimuthal q-axis. If None, the full range is used.
        dq : float or None, optional
            Step size in reciprocal space (Δq). If None, the existing resolution is used.
        dang : float or None, optional
            Step size for the azimuthal coordinate (Δφ). If None, the existing resolution is used.
        plot_result : bool, optional
            If True, displays the resulting pseudopolar map. Default is False.
        clims : tuple of float or None, optional
            Color scale limits (vmin, vmax) for plotting. Default is None.
        xlim : tuple, optional
            X-axis limits for the plot. Default is (None, None).
        ylim : tuple, optional
            Y-axis limits for the plot. Default is (None, None).
        save_fig : bool, optional
            If True, saves the plotted figure. Default is False.
        path_to_save_fig : str, optional
            Path to save the figure if save_fig is True. Default is "img.png".
        save_result : bool, optional
            If True, saves the resulting data to an HDF5 file. Default is False.
        path_to_save : str, optional
            Path to save the HDF5 file if `save_result` is True. Default is "result.h5".
        h5_group : str or None, optional
            HDF5 group name under which the data are stored. Default is None.
        overwrite_file : bool, optional
            If True, overwrites an existing HDF5 file. Default is True.
        overwrite_group : bool, optional
            If True, overwrites an existing group within the HDF5 file. Default is False.
        exp_metadata : pygid.ExpMetadata or None, optional
            Experimental metadata to be stored with the result. Default is None.
        smpl_metadata : pygid.SampleMetadata or None, optional
            Sample-related metadata to be stored with the result. Default is None.

        Returns
        -------
        q_rad : ndarray
            The radial q-axis values of the converted data (Å⁻¹).
        q_azimuth : ndarray
            The azimuthal q-axis values of the converted data (Å⁻¹).
        img_pseudopol : ndarray or list of ndarray
            The pseudopolar image(s) corresponding to (q_rad, q_azimuth).
        """

        # If batch mode is active, delegate execution to the batch processor
        if self.batch_activated:
            res = self.Batch(path_to_save, "det2pseudopol", h5_group, exp_metadata, smpl_metadata, overwrite_file,
                             overwrite_group,
                             save_result, plot_result, return_result)
            self.batch_activated = True
            return res

        # Determine whether recalculation of pseudopolar matrices is required
        recalc = False
        if hasattr(self.matrix[0], "q_rad"):
            if q_rad_range is None:
                recalc = False
            else:
                recalc = False if (np.isclose(q_rad_range[0], np.nanmin(self.matrix[0].q_rad), rtol=0.01) and
                                   np.isclose(q_rad_range[1], np.nanmax(self.matrix[0].q_rad), atol=0.01)) else True

        if hasattr(self.matrix[0], "q_azimuth"):
            if q_azimuth_range is not None:
                recalc = recalc or (
                    False if (np.isclose(q_azimuth_range[0], np.nanmin(self.matrix[0].q_azimuth), rtol=0.01) and
                              np.isclose(q_azimuth_range[1], np.nanmax(self.matrix[0].q_azimuth), atol=0.01)) else True)

        if dq is not None:
            recalc = True if dq != self.matrix[0].dq else recalc
        if dang is not None:
            recalc = True if dang != self.matrix[0].dang else recalc

        # Compute pseudopolar transformation matrices
        self.calc_matrices("p_y_lab_pseudopol", recalc, multiprocessing=multiprocessing or self.multiprocessing,
                           q_rad_range=q_rad_range,
                           q_azimuth_range=q_azimuth_range, dang=dang, dq=dq)

        # Remap detector image from pixel space to pseudopolar coordinates
        x, y, img = self._remap_general_(
            frame_num,
            p_y_key="p_y_lab_pseudopol",
            p_x_key="p_x_lab_pseudopol",
            x_key="q_rad",
            y_key="q_azimuth",
            result_attr="img_pseudopol",
            interp_type=interp_type,
            multiprocessing=multiprocessing,
            return_result=True,
            save_result=save_result,
            path_to_save=path_to_save,
            h5_group=h5_group,
            overwrite_file=overwrite_file,
            overwrite_group=overwrite_group,
            exp_metadata=exp_metadata,
            smpl_metadata=smpl_metadata)

        img = [img] if not isinstance(img, list) else img

        # Plot and/or save each pseudopolar map if requested
        if plot_result or save_fig:
            for i in range(len(img)):
                _plot_single_image(get_plot_context(type(self).plot_params), img[i], x, y, clims, xlim, ylim,
                                   r"$|q|\ \mathrm{[\AA^{-1}]}$", r"$q_{\phi}\ \mathrm{[\AA^{-1}]}$", 'auto', plot_result,
                                   save_fig, add_frame_number(path_to_save_fig, i))
        # Return calculated axes and pseudopolar image(s) if requested
        if return_result:
            return x, y, img

    def det2pseudopol_gid(
            self,
            frame_num=None,
            interp_type="INTER_LINEAR",
            multiprocessing=None,
            return_result=False,
            q_rad_range=None,
            q_azimuth_range=None,
            dang=None,
            dq=None,
            plot_result=False,
            clims=None,
            xlim=(None, None),
            ylim=(None, None),
            save_fig=False,
            path_to_save_fig="img.png",
            save_result=False,
            path_to_save='result.h5',
            h5_group=None,
            overwrite_file=True,
            overwrite_group=False,
            exp_metadata=None,
            smpl_metadata=None,
    ):
        """
        Converts a detector image to pseudopolar coordinates (q_rad, q_azimuth) for grazing-incidence diffraction (GID) geometry.

        Parameters
        ----------
        frame_num : int, list, or None, optional
            Frame index or list of indices to process. If None, the first or current frame is used.
        interp_type : str, optional
            Interpolation method used for remapping. Default is "INTER_LINEAR".
        multiprocessing : bool or None, optional
            Whether to use multiprocessing during computation. If None, the class default is used.
        return_result : bool, optional
            If True, returns the calculated axes and image(s).
        q_rad_range : tuple of float or None, optional
            (min, max) limits for the radial q-axis. If None, the full range is used.
        q_azimuth_range : tuple of float or None, optional
            (min, max) limits for the azimuthal q-axis. If None, the full range is used.
        dq : float or None, optional
            Step size in reciprocal space (Δq). If None, the existing resolution is used.
        dang : float or None, optional
            Step size for the azimuthal coordinate (Δφ). If None, the existing resolution is used.
        plot_result : bool, optional
            If True, displays the resulting pseudopolar GID map. Default is False.
        clims : tuple of float or None, optional
            Color scale limits (vmin, vmax) for plotting. Default is None.
        xlim : tuple, optional
            X-axis limits for the plot. Default is (None, None).
        ylim : tuple, optional
            Y-axis limits for the plot. Default is (None, None).
        save_fig : bool, optional
            If True, saves the plotted figure. Default is False.
        path_to_save_fig : str, optional
            Path to save the figure if save_fig is True. Default is "img.png".
        save_result : bool, optional
            If True, saves the resulting data to an HDF5 file. Default is False.
        path_to_save : str, optional
            Path to save the HDF5 file if `save_result` is True. Default is "result.h5".
        h5_group : str or None, optional
            HDF5 group name under which the data are stored. Default is None.
        overwrite_file : bool, optional
            If True, overwrites an existing HDF5 file. Default is True.
        overwrite_group : bool, optional
            If True, overwrites an existing group within the HDF5 file. Default is False.
        exp_metadata : pygid.ExpMetadata or None, optional
            Experimental metadata to be stored with the result. Default is None.
        smpl_metadata : pygid.SampleMetadata or None, optional
            Sample-related metadata to be stored with the result. Default is None.

        Returns
        -------
        q_gid_rad : ndarray
            The radial q-axis values of the converted data (Å⁻¹).
        q_gid_azimuth : ndarray
            The azimuthal q-axis values of the converted data (Å⁻¹).
        img_gid_pseudopol : ndarray or list of ndarray
            The pseudopolar GID image(s) corresponding to (q_rad, q_azimuth).
        """

        # If batch mode is active, delegate execution to the batch processor
        if self.batch_activated:
            res = self.Batch(path_to_save, "det2pseudopol_gid", h5_group, exp_metadata, smpl_metadata, overwrite_file,
                             overwrite_group, save_result, plot_result, return_result)
            self.batch_activated = True
            return res

        # Determine whether recalculation of pseudopolar GID matrices is required
        recalc = False
        if hasattr(self.matrix[0], "q_gid_rad"):
            if q_rad_range is None:
                recalc = False
            else:
                recalc = False if (np.isclose(q_rad_range[0], np.nanmin(self.matrix[0].q_gid_rad), rtol=0.01) and
                                   np.isclose(q_rad_range[1], np.nanmax(self.matrix[0].q_gid_rad), atol=0.01)) else True

        if hasattr(self.matrix[0], "q_gid_azimuth"):
            if q_azimuth_range is not None:
                recalc = recalc or (
                    False if (np.isclose(q_azimuth_range[0], np.nanmin(self.matrix[0].q_gid_azimuth), rtol=0.01) and
                              np.isclose(q_azimuth_range[1], np.nanmax(self.matrix[0].q_gid_azimuth),
                                         atol=0.01)) else True)

        # Force recalculation if dq or dang differ from current configuration
        if dq is not None:
            recalc = True if dq != self.matrix[0].dq else recalc
        if dang is not None:
            recalc = True if dang != self.matrix[0].dang else recalc

        # Compute pseudopolar transformation matrices for GID
        self.calc_matrices("p_y_smpl_pseudopol", recalc, multiprocessing=multiprocessing or self.multiprocessing,
                           q_gid_rad_range=q_rad_range,
                           q_gid_azimuth_range=q_azimuth_range, dang=dang, dq=dq)

        # Remap detector image to pseudopolar GID coordinates
        x, y, img = self._remap_general_(
            frame_num,
            p_y_key="p_y_smpl_pseudopol",
            p_x_key="p_x_smpl_pseudopol",
            x_key="q_gid_rad",
            y_key="q_gid_azimuth",
            result_attr="img_gid_pseudopol",
            interp_type=interp_type,
            multiprocessing=multiprocessing,
            return_result=True,
            save_result=save_result,
            path_to_save=path_to_save,
            h5_group=h5_group,
            overwrite_file=overwrite_file,
            overwrite_group=overwrite_group,
            exp_metadata=exp_metadata,
            smpl_metadata=smpl_metadata)
        img = [img] if not isinstance(img, list) else img

        # Plot and/or save each pseudopolar GID map if requested
        if plot_result or save_fig:
            for i in range(len(img)):
                _plot_single_image(get_plot_context(type(self).plot_params), img[i], x, y, clims, xlim, ylim,
                                   r"$|q|\ \mathrm{[\AA^{-1}]}$", r"$q_{\phi}\ \mathrm{[\AA^{-1}]}$", 'auto', plot_result,
                                   save_fig, add_frame_number(path_to_save_fig, i))
        # Return calculated axes and pseudopolar GID image(s) if requested
        if return_result:
            return x, y, img

    def _get_polar_data(self, key, frame_num, radial_range, angular_range, dang, dq):
        """
        Calls polar remapping of detector data based on the specified geometry.

        Parameters
        ----------
        key : str
            "gid" or "transmission"
        frame_num : int
            Frame number to process.
        radial_range : tuple
            Tuple specifying the minimum and maximum q values for the radial axis.
        angular_range : tuple
            Tuple specifying the minimum and maximum values of azimuthal angle (in degrees).
        dang : float
            Angular resolution step size (in degrees).
        dq : float
            Radial resolution step size.

        Returns
        -------
        tuple
            Contains arrays for q values, azimuthal angles, and the remapped image.
        """
        method = self.det2pol_gid if key == "gid" else self.det2pol
        return method(return_result=True, plot_result=False, frame_num=frame_num,
                      radial_range=radial_range, angular_range=angular_range, dang=dang, dq=dq)



    def radial_profile_gid(
            self,
            frame_num=None,
            radial_range=None,
            angular_range=[0, 90],
            multiprocessing=None,
            return_result=False,
            save_result=False,
            save_fig=False,
            path_to_save_fig='rad_cut.tiff',
            plot_result=False,
            shift=1,
            xlim=(None, None),
            ylim=(None, None),
            dang=0.5,
            dq=None,
            path_to_save='result.h5',
            h5_group=None,
            overwrite_file=True,
            overwrite_group=False,
            exp_metadata=None,
            smpl_metadata=None,
    ):
        """
        Computes and optionally plots the radial profile from 2D scattering data for GID geometry.

        Parameters
        ----------
        frame_num : int, list or None, optional
            Frame number to analyze. If None, all data will be used.
        radial_range : list or tuple, optional
            Radial (q) range as [min, max] in Å⁻¹. If None, full range is used.
        angular_range : list, optional
            Angular range in degrees as [min, max] over which to integrate (default: [0, 90]).
        multiprocessing : bool or None, optional
            If True, use multiprocessing for faster processing. If None, use default setting.
        return_result : bool, optional
            If True, returns the computed profile.
        save_result : bool, optional
            If True, saves the computed profile to an HDF5 file.
        save_fig : bool, optional
            If True, saves the plot of the profile to a file.
        path_to_save_fig : str, optional
            Path where the figure will be saved (if `save_fig` is True).
        plot_result : bool, optional
            If True, displays the radial profile plot.
        shift : float, optional
            Vertical shift applied to the profile for display purposes.
        xlim : tuple or None, optional
            X-axis limits as (min, max). If None, limits are auto-scaled.
        ylim : tuple or None, optional
            Y-axis limits as (min, max). If None, limits are auto-scaled.
        dang : float, optional
            Angular resolution in degrees for binning (default: 0.5).
        dq : float or None, optional
            Radial bin width in Å⁻¹. If None, uses default binning.
        path_to_save : str, optional
            Path where results should be saved (if `save_result` is True).
        h5_group : str or None, optional
            HDF5 group name for saving results. If None, uses default group.
        overwrite_file : bool, optional
            If True, overwrites existing file when saving results. Otherwise, appends to the existing h5-file.
        exp_metadata : pygid.ExpMetadata or None
                Experimental metadata to include in the output file.
        smpl_metadata : pygid.SampleMetadata or None
                Sample metadata to include in the output file.

        Returns
        -------
        q_abs_values : array
            The q_abs_values-axis values of the converted data (in 1/A).
        rad_cut_gid : 1D-array or list of 1D-arrays
            Integrated image profile rad_cut.
        """

        key = 'gid'
        remap_func = "radial_profile_gid"
        name = "rad_cut_gid"

        return self.calculate_radial_profile(
            key = key,
            frame_num = frame_num,
            radial_range = radial_range,
            angular_range = angular_range,
            multiprocessing = multiprocessing,
            return_result = return_result,
            save_result = save_result,
            save_fig = save_fig,
            path_to_save_fig = path_to_save_fig,
            plot_result = plot_result,
            shift = shift,
            xlim = xlim,
            ylim = ylim,
            dang = dang,
            dq = dq,
            path_to_save = path_to_save,
            h5_group = h5_group,
            overwrite_file = overwrite_file,
            overwrite_group = overwrite_group,
            exp_metadata = exp_metadata,
            smpl_metadata = smpl_metadata,
            remap_func = remap_func,
            name = name)

    def radial_profile(
            self,
            frame_num=None,
            radial_range=None,
            angular_range=[0, 90],
            multiprocessing=None,
            return_result=False,
            save_result=False,
            save_fig=False,
            path_to_save_fig='rad_cut.tiff',
            plot_result=False,
            shift=1,
            xlim=(None, None),
            ylim=(None, None),
            dang=0.5,
            dq=None,
            path_to_save='result.h5',
            h5_group=None,
            overwrite_file=True,
            overwrite_group=False,
            exp_metadata=None,
            smpl_metadata=None,
    ):
        """
        Computes and optionally plots the radial profile from 2D scattering data for transmission geometry.

        Parameters
        ----------
        frame_num : int, list, or None, optional
            Frame number to analyze. If None, all data will be used.
        radial_range : list or tuple, optional
            Radial (q) range as [min, max] in Å⁻¹. If None, full range is used.
        angular_range : list, optional
            Angular range in degrees as [min, max] over which to integrate (default: [0, 90]).
        multiprocessing : bool or None, optional
            If True, use multiprocessing for faster processing. If None, use default setting.
        return_result : bool, optional
            If True, returns the computed profile.
        save_result : bool, optional
            If True, saves the computed profile to an HDF5 file.
        save_fig : bool, optional
            If True, saves the plot of the profile to a file.
        path_to_save_fig : str, optional
            Path where the figure will be saved (if `save_fig` is True).
        plot_result : bool, optional
            If True, displays the radial profile plot.
        shift : float, optional
            Vertical shift applied to the profile for display purposes.
        xlim : tuple or None, optional
            X-axis limits as (min, max). If None, limits are auto-scaled.
        ylim : tuple or None, optional
            Y-axis limits as (min, max). If None, limits are auto-scaled.
        dang : float, optional
            Angular resolution in degrees for binning (default: 0.5).
        dq : float or None, optional
            Radial bin width in Å⁻¹. If None, uses default binning.
        path_to_save : str, optional
            Path where results should be saved (if `save_result` is True).
        h5_group : str or None, optional
            HDF5 group name for saving results. If None, uses default group.
        overwrite_file : bool, optional
            If True, overwrites existing file when saving results. Otherwise, appends to the existing h5-file.
        exp_metadata : pygid.ExpMetadata or None
                Experimental metadata to include in the output file.
        smpl_metadata : pygid.SampleMetadata or None
                Sample metadata to include in the output file.

        Returns
        -------
        q_abs_values : array
            The q_abs_values-axis values of the converted data (in 1/A).
        rad_cut : 1D-array or list of 1D-arrays
            Integrated image profile rad_cut.
        """

        key = 'transmission'
        remap_func = "radial_profile"
        name = "rad_cut"

        return self.calculate_radial_profile(
            key=key,
            frame_num=frame_num,
            radial_range=radial_range,
            angular_range=angular_range,
            multiprocessing=multiprocessing,
            return_result=return_result,
            save_result=save_result,
            save_fig=save_fig,
            path_to_save_fig=path_to_save_fig,
            plot_result=plot_result,
            shift=shift,
            xlim=xlim,
            ylim=ylim,
            dang=dang,
            dq=dq,
            path_to_save=path_to_save,
            h5_group=h5_group,
            overwrite_file=overwrite_file,
            overwrite_group=overwrite_group,
            exp_metadata=exp_metadata,
            smpl_metadata=smpl_metadata,
            remap_func=remap_func,
            name=name)

    def calculate_radial_profile(
            self,
            key,
            frame_num,
            radial_range,
            angular_range,
            multiprocessing,
            return_result,
            save_result,
            save_fig,
            path_to_save_fig,
            plot_result,
            shift,
            xlim,
            ylim,
            dang,
            dq,
            path_to_save,
            h5_group,
            overwrite_file,
            overwrite_group,
            exp_metadata,
            smpl_metadata,
            remap_func,
            name
    ):
        """
            Computes and optionally plots the radial intensity profile from 2D scattering data.

            The method integrates the intensity over the azimuthal direction within a given angular range,
            producing a 1D profile as a function of the scattering vector magnitude (q_abs).

            Parameters
            ----------
            key : str
                Geometry key ("gid" or "transmission") indicating which dataset to process.
            frame_num : int, list, or None
                Frame index or list of indices to analyze. If None, all frames are used.
            radial_range : list or tuple
                Radial (q) range in Å⁻¹ as [min, max]. If None, the full range is used.
            angular_range : list or tuple
                Azimuthal range in degrees as [min, max] over which to integrate.
            multiprocessing : bool or None
                If True, enables multiprocessing for faster processing. If None, uses default setting.
            return_result : bool
                If True, returns the computed radial profile.
            save_result : bool
                If True, saves the computed profile to an HDF5 file.
            save_fig : bool
                If True, saves a plot of the radial profile.
            path_to_save_fig : str
                Path for saving the figure if `save_fig` is True.
            plot_result : bool
                If True, displays the radial profile plot.
            shift : float
                Vertical shift applied to the plotted profile.
            xlim : tuple or None
                Limits for the X-axis (q-range). Default is None (auto).
            ylim : tuple or None
                Limits for the Y-axis (intensity). Default is None (auto).
            dang : float
                Angular resolution in degrees for binning. Default is 0.5.
            dq : float or None
                Radial bin width in Å⁻¹. If None, uses default binning.
            path_to_save : str
                Path where results should be saved if `save_result` is True.
            h5_group : str or None
                HDF5 group name for storing the results. Default is None.
            overwrite_file : bool
                If True, overwrites the existing HDF5 file. Default is True.
            overwrite_group : bool
                If True, overwrites the existing HDF5 group. Default is False.
            exp_metadata : pygid.ExpMetadata or None
                Experimental metadata to include in the output file.
            smpl_metadata : pygid.SampleMetadata or None
                Sample metadata to include in the output file.
            remap_func : str
                Name of the remapping function used for batch processing.
            name : str
                Attribute name under which the computed profile is stored in the class instance.

            Returns
            -------
            q_abs_values : ndarray
                Scattering vector magnitude values in Å⁻¹.
            radial_profile : ndarray or list of ndarray
                Computed radial profile(s).
        """
        # Check if batch mode is active
        if self.batch_activated:
            # Choose the appropriate batch function based on geometry key
            res = self.Batch(path_to_save, remap_func, h5_group, exp_metadata, smpl_metadata, overwrite_file,
                             overwrite_group,
                             save_result, plot_result, return_result)
            self.batch_activated = True
            return res

        # Retrieve polar-transformed image data
        q_abs_values, _, img_pol = self._get_polar_data(key, frame_num, radial_range, angular_range, dang, dq)
        img_pol = np.array(img_pol)

        # Expand dimensions if a single frame (2D array) to unify processing
        img_pol = np.expand_dims(img_pol, axis=0) if img_pol.ndim == 2 else img_pol

        # Compute radial profile by averaging over angular direction
        radial_profile = np.nanmean(img_pol, axis=1)

        # Plot the radial profile if requested
        if plot_result or save_fig:
            _plot_profile(plot_context = get_plot_context(type(self).plot_params),
                          x_values = q_abs_values,
                          profiles = radial_profile,
                          xlabel = r"$q_{abs}\ [\AA^{-1}]$",
                          shift = shift,
                          xlim = xlim,
                          ylim = ylim,
                          plot_result = plot_result,
                          save_fig = save_fig,
                          path_to_save_fig = path_to_save_fig)

        setattr(self, name, radial_profile)
        delattr(self, "img_gid_pol") if key == "gid" else delattr(self, "img_pol")

        # Save the profile to file if requested
        if save_result:
            self.save_nxs(path_to_save=path_to_save,
                          h5_group=h5_group,
                          overwrite_file=overwrite_file,
                          overwrite_group=overwrite_group,
                          exp_metadata=exp_metadata,
                          smpl_metadata=smpl_metadata)
        # Return computed profile if requested
        if return_result:
            return (q_abs_values, radial_profile[0]) if radial_profile.shape[0] == 1 else (
                q_abs_values, radial_profile)

    def azim_profile_gid(
            self,
            frame_num=None,
            radial_range=None,
            angular_range=[0, 90],
            multiprocessing=None,
            return_result=False,
            save_result=False,
            save_fig=False,
            path_to_save_fig='azim_cut.tiff',
            plot_result=False,
            shift=1,
            xlim=(None, None),
            ylim=(None, None),
            path_to_save='result.h5',
            dang=0.5,
            dq=None,
            h5_group=None,
            overwrite_file=True,
            overwrite_group=False,
            exp_metadata=None,
            smpl_metadata=None,
    ):
        """
            Computes and optionally plots the azimuthal profile from 2D scattering data for GID geometry.

            Parameters
            ----------
            frame_num : int, list, or None, optional
                Frame index or list of indices to analyze. If None, all frames are used.
            radial_range : list or tuple, optional
                Radial (q) range as [min, max] in Å⁻¹. If None, full range is used.
            angular_range : list, optional
                Azimuthal integration range in degrees as [min, max] (default [0, 90]).
            multiprocessing : bool or None, optional
                If True, use multiprocessing for faster processing. If None, use default setting.
            return_result : bool, optional
                If True, returns the computed azimuthal profile.
            save_result : bool, optional
                If True, saves the computed profile to an HDF5 file.
            save_fig : bool, optional
                If True, saves the plot of the profile to a file.
            path_to_save_fig : str, optional
                File path for saving the figure if `save_fig` is True. Default is 'azim_cut.tiff'.
            plot_result : bool, optional
                If True, displays the azimuthal profile plot.
            shift : float, optional
                Vertical shift applied to the profile for display purposes.
            xlim : tuple or None, optional
                X-axis limits as (min, max). If None, limits are auto-scaled.
            ylim : tuple or None, optional
                Y-axis limits as (min, max). If None, limits are auto-scaled.
            dang : float, optional
                Angular resolution in degrees for binning (default: 0.5).
            dq : float or None, optional
                Radial bin width in Å⁻¹. If None, uses default binning.
            path_to_save : str, optional
                HDF5 file path for saving results if `save_result` is True. Default is 'result.h5'.
            h5_group : str or None, optional
                HDF5 group name for saving results. Default is None.
            overwrite_file : bool, optional
                If True, overwrites existing HDF5 file. Default is True.
            overwrite_group : bool, optional
                If True, overwrites existing group within the HDF5 file. Default is False.
            exp_metadata : pygid.ExpMetadata or None, optional
                Experimental metadata to store with results.
            smpl_metadata : pygid.SampleMetadata or None, optional
                Sample metadata to store with results.

            Returns
            -------
            phi_abs_values : ndarray
                Azimuthal angle values in degrees.
            azim_cut_gid : ndarray or list of ndarray
                Integrated azimuthal profile(s).
        """
        remap_func = "azim_profile_gid"
        name = "azim_cut_gid"
        key = 'gid'

        return self.calculate_azim_profile(
            key,
            frame_num,
            radial_range,
            angular_range,
            multiprocessing,
            return_result,
            save_result,
            save_fig,
            path_to_save_fig,
            plot_result,
            shift,
            xlim,
            ylim,
            dang,
            dq,
            path_to_save,
            h5_group,
            overwrite_file,
            overwrite_group,
            exp_metadata,
            smpl_metadata,
            remap_func,
            name
        )

    def azim_profile(
            self,
            frame_num=None,
            radial_range=None,
            angular_range=[0, 90],
            multiprocessing=None,
            return_result=False,
            save_result=False,
            save_fig=False,
            path_to_save_fig='azim_cut.tiff',
            plot_result=False,
            shift=1,
            xlim=(None, None),
            ylim=(None, None),
            path_to_save='result.h5',
            dang=0.5,
            dq=None,
            h5_group=None,
            overwrite_file=True,
            overwrite_group=False,
            exp_metadata=None,
            smpl_metadata=None):
        """
        Computes and optionally plots the azimuthal profile from 2D scattering data for transmission geometry.

        Parameters
        ----------
        frame_num : int, list, or None, optional
            Frame index or list of indices to analyze. If None, all frames are used.
        radial_range : list or tuple, optional
            Radial (q) range as [min, max] in Å⁻¹. If None, full range is used.
        angular_range : list, optional
            Azimuthal integration range in degrees as [min, max] (default [0, 90]).
        multiprocessing : bool or None, optional
            If True, use multiprocessing for faster processing. If None, use default setting.
        return_result : bool, optional
            If True, returns the computed azimuthal profile.
        save_result : bool, optional
            If True, saves the computed profile to an HDF5 file.
        save_fig : bool, optional
            If True, saves the plot of the profile to a file.
        path_to_save_fig : str, optional
            File path for saving the figure if `save_fig` is True. Default is 'azim_cut.tiff'.
        plot_result : bool, optional
            If True, displays the azimuthal profile plot.
        shift : float, optional
            Vertical shift applied to the profile for display purposes. Default is 1.
        xlim : tuple or None, optional
            Limits for the X-axis (phi range). Default is None (auto).
        ylim : tuple or None, optional
            Limits for the Y-axis. Default is None (auto).
        dang : float, optional
            Angular resolution in degrees for binning. Default is 0.5.
        dq : float or None, optional
            Radial bin width in Å⁻¹. If None, uses default binning.
        path_to_save : str, optional
            HDF5 file path for saving results if `save_result` is True. Default is 'result.h5'.
        h5_group : str or None, optional
            HDF5 group name for saving results. Default is None.
        overwrite_file : bool, optional
            If True, overwrites existing HDF5 file. Default is True.
        overwrite_group : bool, optional
            If True, overwrites existing group within the HDF5 file. Default is False.
        exp_metadata : pygid.ExpMetadata or None, optional
            Experimental metadata to store with results.
        smpl_metadata : pygid.SampleMetadata or None, optional
            Sample metadata to store with results.

        Returns
        -------
        phi_abs_values : ndarray
            Azimuthal angle values in degrees.
        azim_cut : ndarray or list of ndarray
            Integrated azimuthal profile(s).
        """

        remap_func = "azim_profile"
        name = "azim_cut"
        key = 'transmission'

        return self.calculate_azim_profile(
            key,
            frame_num,
            radial_range,
            angular_range,
            multiprocessing,
            return_result,
            save_result,
            save_fig,
            path_to_save_fig,
            plot_result,
            shift,
            xlim,
            ylim,
            dang,
            dq,
            path_to_save,
            h5_group,
            overwrite_file,
            overwrite_group,
            exp_metadata,
            smpl_metadata,
            remap_func,
            name
        )

    def calculate_azim_profile(
            self,
            key,
            frame_num,
            radial_range,
            angular_range,
            multiprocessing,
            return_result,
            save_result,
            save_fig,
            path_to_save_fig,
            plot_result,
            shift,
            xlim,
            ylim,
            dang,
            dq,
            path_to_save,
            h5_group,
            overwrite_file,
            overwrite_group,
            exp_metadata,
            smpl_metadata,
            remap_func,
            name
    ):
        """
        Computes and optionally plots the azimuthal intensity profile from 2D scattering data.

        The method integrates the scattering intensity over the radial (q) direction within a given
        q-range, resulting in a 1D azimuthal profile as a function of the scattering angle (phi).

        Parameters
        ----------
        key : str
            Geometry key ("gid" or "transmission") indicating which dataset to process.
        frame_num : int, list, or None
            Frame index or list of indices to analyze. If None, all frames are used.
        radial_range : list or tuple
            Radial (q) range in Å⁻¹ as [min, max]. If None, the full range is used.
        angular_range : list or tuple
            Azimuthal range in degrees as [min, max] over which to integrate.
        multiprocessing : bool or None
            If True, enables multiprocessing for faster processing. If None, uses default setting.
        return_result : bool
            If True, returns the computed azimuthal profile.
        save_result : bool
            If True, saves the computed profile to an HDF5 file.
        save_fig : bool
            If True, saves a plot of the azimuthal profile.
        path_to_save_fig : str
            Path for saving the figure if `save_fig` is True.
        plot_result : bool
            If True, displays the azimuthal profile plot.
        shift : float
            Vertical shift applied to the plotted profile.
        xlim : tuple or None
            Limits for the X-axis (phi range). Default is None (auto).
        ylim : tuple or None
            Limits for the Y-axis (intensity). Default is None (auto).
        dang : float
            Angular resolution in degrees for binning. Default is 0.5.
        dq : float or None
            Radial bin width in Å⁻¹. If None, uses default binning.
        path_to_save : str
            Path where results should be saved if `save_result` is True.
        h5_group : str or None
            HDF5 group name for storing the results. Default is None.
        overwrite_file : bool
            If True, overwrites the existing HDF5 file. Default is True.
        overwrite_group : bool
            If True, overwrites the existing HDF5 group. Default is False.
        exp_metadata : pygid.ExpMetadata or None
            Experimental metadata to include in the output file.
        smpl_metadata : pygid.SampleMetadata or None
            Sample metadata to include in the output file.
        remap_func : str
            Name of the remapping function used for batch processing.
        name : str
            Attribute name under which the computed profile is stored in the class instance.

        Returns
        -------
        phi_abs_values : ndarray
            Azimuthal angle values in degrees.
        azim_profile : ndarray or list of ndarray
            Computed azimuthal profile(s).
        """

        if self.batch_activated:
            res = self.Batch(path_to_save, remap_func, h5_group, exp_metadata, smpl_metadata, overwrite_file,
                             overwrite_group,
                             save_result, plot_result, return_result)
            self.batch_activated = True
            return res

        # Extract polar data: returns (radial, azimuthal, image array)
        _, phi_abs_values, img_pol = self._get_polar_data(key, frame_num, radial_range, angular_range, dang, dq)
        img_pol = np.array(img_pol)
        img_pol = np.expand_dims(img_pol, axis=0) if img_pol.ndim == 2 else img_pol

        # Integrate over radial dimension to obtain azimuthal profile
        azim_profile = np.nanmean(img_pol, axis=2)

        # Plot profile if requested
        if plot_result or save_fig:
            _plot_profile(plot_context = get_plot_context(type(self).plot_params),
                          x_values = phi_abs_values,
                          profiles = azim_profile,
                          xlabel = r"$\chi\ [\degree]$",
                          shift = shift,
                          xlim = xlim,
                          ylim = ylim,
                          plot_result = plot_result,
                          save_fig = save_fig,
                          path_to_save_fig = path_to_save_fig)


        setattr(self, name, azim_profile)
        delattr(self, "img_gid_pol") if key == "gid" else delattr(self, "img_pol")

        # Save results to HDF5 if requested
        if save_result:
            self.save_nxs(path_to_save=path_to_save,
                          h5_group=h5_group,
                          overwrite_file=overwrite_file,
                          overwrite_group=overwrite_group,
                          exp_metadata=exp_metadata,
                          smpl_metadata=smpl_metadata)
        # Return results if requested
        if return_result:
            return (phi_abs_values, azim_profile[0]) if azim_profile.shape[0] == 1 else (
                phi_abs_values, azim_profile)

    def _get_q_data(self, frame_num, q_xy_range=None, q_z_range=None, dq=None):

        """
        Calls GID remapping of detector data.

        Parameters
        ----------
        frame_num : int
            Frame number to process.
        q_xy_range : tuple
            Tuple specifying the minimum and maximum q_xy values for the radial axis.
        q_z_range : tuple
            Tuple specifying the minimum and maximum q_z values for the radial axis.
        """

        method = self.det2q_gid
        return method(return_result=True, plot_result=False, frame_num=frame_num,
                      q_xy_range=q_xy_range, q_z_range=q_z_range, dq=dq)

    def horiz_profile(self, **kwargs):
        return self.horiz_profile_gid(**kwargs)

    def horiz_profile_gid(
            self,
            frame_num=None,
            q_xy_range=[0, 4],
            q_z_range=[0, 0.2],
            dq=None,
            multiprocessing=None,
            return_result=False,
            save_result=False,
            save_fig=False,
            path_to_save_fig='hor_cut.tiff',
            plot_result=False,
            shift=1,
            xlim=(None, None),
            ylim=(None, None),
            path_to_save='result.h5',
            h5_group=None,
            overwrite_file=True,
            overwrite_group=False,
            exp_metadata=None,
            smpl_metadata=None
    ):
        """
        Computes and optionally plots the horizontal (q_xy) line profile from a GID reciprocal-space map.

        The method integrates the 2D reciprocal-space image along the q_z axis within a given range,
        resulting in a 1D horizontal intensity profile as a function of q_xy.

        Parameters
        ----------
        frame_num : int, list, or None, optional
            Frame index or list of indices to analyze. If None, the first or current frame is used.
        q_xy_range : list or tuple, optional
            In-plane momentum transfer range (Å⁻¹) as [min, max]. Default is [0, 4].
        q_z_range : list or tuple, optional
            Out-of-plane momentum transfer range (Å⁻¹) as [min, max]. Default is [0, 0.2].
        dq : float or None, optional
            Reciprocal-space step size (Δq). If None, existing resolution is used.
        multiprocessing : bool or None, optional
            If True, enables multiprocessing for faster computation. If None, uses default setting.
        return_result : bool, optional
            If True, returns the computed horizontal profile.
        save_result : bool, optional
            If True, saves the computed profile to an HDF5 file.
        save_fig : bool, optional
            If True, saves the horizontal profile plot to file.
        path_to_save_fig : str, optional
            Path for saving the figure if `save_fig` is True. Default is 'hor_cut.tiff'.
        plot_result : bool, optional
            If True, displays the computed horizontal profile. Default is False.
        shift : float, optional
            Vertical offset applied to the plotted profile. Default is 1.
        xlim : tuple or None, optional
            Limits for the X-axis (q_xy). Default is None (auto).
        ylim : tuple or None, optional
            Limits for the Y-axis (intensity). Default is None (auto).
        path_to_save : str, optional
            Path where the results will be saved if `save_result` is True. Default is 'result.h5'.
        h5_group : str or None, optional
            HDF5 group name under which to store the data. Default is None.
        overwrite_file : bool, optional
            If True, overwrites existing HDF5 file when saving. Default is True.
        overwrite_group : bool, optional
            If True, overwrites existing group within the HDF5 file. Default is False.
        exp_metadata : pygid.ExpMetadata or None, optional
            Experimental metadata to be stored with the result. Default is None.
        smpl_metadata : pygid.SampleMetadata or None, optional
            Sample-related metadata to be stored with the result. Default is None.

        Returns
        -------
        q_hor_values : ndarray
            q_xy-axis values of the horizontal profile (Å⁻¹).
        horiz_cut : ndarray or list of ndarray
            Computed horizontal intensity profile(s).
        """

        if self.batch_activated:
            res = self.Batch(path_to_save, "horiz_profile", h5_group, exp_metadata, smpl_metadata, overwrite_file,
                             overwrite_group,
                             save_result, plot_result, return_result)
            self.batch_activated = True
            return res

        q_hor_values, _, img_q = self._get_q_data(frame_num, q_xy_range, q_z_range, dq)
        img_q = np.array(img_q)
        img_q = np.expand_dims(img_q, axis=0) if img_q.ndim == 2 else img_q
        horiz_profile = np.nanmean(img_q, axis=1)
        if plot_result or save_fig:
            _plot_profile(plot_context = get_plot_context(type(self).plot_params),
                          x_values = q_hor_values,
                          profiles = horiz_profile,
                          xlabel = r'$q_{xy}$ [$\mathrm{\AA}^{-1}$]',
                          shift = shift,
                          xlim = xlim,
                          ylim = ylim,
                          plot_result = plot_result,
                          save_fig = save_fig,
                          path_to_save_fig = path_to_save_fig)

        setattr(self, "horiz_cut_gid", horiz_profile)
        delattr(self, "img_gid_q")
        if save_result:
            self.save_nxs(path_to_save=path_to_save,
                          h5_group=h5_group,
                          overwrite_file=overwrite_file,
                          overwrite_group=overwrite_group,
                          exp_metadata=exp_metadata,
                          smpl_metadata=smpl_metadata)

        if return_result:
            return (q_hor_values, horiz_profile[0]) if horiz_profile.shape[0] == 1 else (
                q_hor_values, horiz_profile)

    def vert_profile(self, **kwargs):
        return self.vert_profile_gid(**kwargs)

    def vert_profile_gid(
            self,
            frame_num=None,
            q_xy_range=[0, 0.2],
            q_z_range=[0, 4],
            dq=None,
            multiprocessing=None,
            return_result=False,
            save_result=False,
            save_fig=False,
            path_to_save_fig='vert_cut.tiff',
            plot_result=False,
            shift=1,
            xlim=(None, None),
            ylim=(None, None),
            path_to_save='result.h5',
            h5_group=None,
            overwrite_file=True,
            overwrite_group=False,
            exp_metadata=None,
            smpl_metadata=None
    ):
        """
        Computes and optionally plots the vertical (q_z) line profile from a GID reciprocal-space map.

        The method integrates the 2D reciprocal-space image along the q_xy axis within a given range,
        resulting in a 1D vertical intensity profile as a function of q_z.

        Parameters
        ----------
        frame_num : int, list, or None, optional
            Frame index or list of indices to analyze. If None, the first or current frame is used.
        q_xy_range : list or tuple, optional
            In-plane momentum transfer range (Å⁻¹) as [min, max]. Default is [0, 4].
        q_z_range : list or tuple, optional
            Out-of-plane momentum transfer range (Å⁻¹) as [min, max]. Default is [0, 0.2].
        dq : float or None, optional
            Reciprocal-space step size (Δq). If None, existing resolution is used.
        multiprocessing : bool or None, optional
            If True, enables multiprocessing for faster computation. If None, uses default setting.
        return_result : bool, optional
            If True, returns the computed vertical profile.
        save_result : bool, optional
            If True, saves the computed profile to an HDF5 file.
        save_fig : bool, optional
            If True, saves the vertical profile plot to file.
        path_to_save_fig : str, optional
            Path for saving the figure if `save_fig` is True. Default is 'vert_cut.tiff'.
        plot_result : bool, optional
            If True, displays the computed vertical profile. Default is False.
        shift : float, optional
            Vertical offset applied to the plotted profile. Default is 1.
        xlim : tuple or None, optional
            Limits for the X-axis (q_z). Default is None (auto).
        ylim : tuple or None, optional
            Limits for the Y-axis (intensity). Default is None (auto).
        path_to_save : str, optional
            Path where the results will be saved if `save_result` is True. Default is 'result.h5'.
        h5_group : str or None, optional
            HDF5 group name under which to store the data. Default is None.
        overwrite_file : bool, optional
            If True, overwrites existing HDF5 file when saving. Default is True.
        overwrite_group : bool, optional
            If True, overwrites existing group within the HDF5 file. Default is False.
        exp_metadata : pygid.ExpMetadata or None, optional
            Experimental metadata to be stored with the result. Default is None.
        smpl_metadata : pygid.SampleMetadata or None, optional
            Sample-related metadata to be stored with the result. Default is None.

        Returns
        -------
        q_vert_values : ndarray
            q_z-axis values of the vertical profile (Å⁻¹).
        vert_cut : ndarray or list of ndarray
            Computed vertical intensity profile(s).
        """

        if self.batch_activated:
            res = self.Batch(path_to_save, "vert_profile", h5_group, exp_metadata, smpl_metadata, overwrite_file,
                             overwrite_group,
                             save_result, plot_result, return_result)
            self.batch_activated = True
            return res

        q_hor_values, q_vert_values, img_q = self._get_q_data(frame_num, q_xy_range, q_z_range, dq)
        img_q = np.array(img_q)
        img_q = np.expand_dims(img_q, axis=0) if img_q.ndim == 2 else img_q
        vert_profile = np.nanmean(img_q, axis=2)
        if plot_result or save_fig:
            _plot_profile(plot_context = get_plot_context(type(self).plot_params),
                          x_values = q_vert_values,
                          profiles = vert_profile,
                          xlabel = r'$q_{z}$ [$\mathrm{\AA}^{-1}$]',
                          shift = shift,
                          xlim = xlim,
                          ylim = ylim,
                          plot_result = plot_result,
                          save_fig = save_fig,
                          path_to_save_fig = path_to_save_fig)

        setattr(self, "vert_cut_gid", vert_profile)
        delattr(self, "img_gid_q")
        if save_result:
            self.save_nxs(path_to_save=path_to_save,
                          h5_group=h5_group,
                          overwrite_file=overwrite_file,
                          overwrite_group=overwrite_group,
                          exp_metadata=exp_metadata,
                          smpl_metadata=smpl_metadata)

        if return_result:
            return (q_vert_values, vert_profile[0]) if vert_profile.shape[0] == 1 else (
                q_vert_values, vert_profile)

    def _remap_single_image_(self, img_raw=None, interp_type="INTER_LINEAR", multiprocessing=False, p_y=None, p_x=None):
        """
        Applies a geometric transformation to a single 2D image using remapping coordinates.

        Parameters
        ----------
        img_raw : np.ndarray, optional
            Input image to be remapped
        interp_type : str, optional
            Interpolation method used for remapping. Must be a valid OpenCV interpolation flag
            (e.g., 'INTER_NEAREST', 'INTER_LINEAR'). Default is 'INTER_LINEAR'.
        multiprocessing : bool, optional
            If True, enables multiprocessing for parallel remapping. Default is False.
        p_y : np.ndarray or None, optional
            Array specifying the y-coordinates (rows) for remapping.
        p_x : np.ndarray or None, optional
            Array specifying the x-coordinates (columns) for remapping.

        Returns
        -------
        np.ndarray
            The remapped image as a 2D array.
        """

        remap_image = fast_pixel_remap(img_raw, p_y, p_x, use_gpu=self.use_gpu, interp_type=interp_type,
                                       multiprocessing=multiprocessing)
        return remap_image

    def calc_matrices(self, key, recalc=False, multiprocessing=True, **kwargs):
        """Processes all matrices in the given list, optionally using threads."""
        if multiprocessing:
            with ThreadPoolExecutor() as executor:
                executor.map(lambda matrix: calc_matrix(matrix, key, recalc, **kwargs), self.matrix)
        else:
            for matrix in self.matrix:
                calc_matrix(matrix, key, recalc, **kwargs)
        if hasattr(self, "matrix_to_save"):
            self.matrix_to_save.save_instance()
        else:
            self.matrix[0].save_instance()

    def make_simulation(self, frame_num=0, crystal=None,
                        path_to_cif=None, orientation=None,
                        plot_result=True, plot_mi=False, return_result=False, move_fromMW=False,
                        min_int=None, clims=None, vmin=0, vmax=1, linewidth=1, radius=0.1, cmap=None,
                        text_color='black', save_fig=False, path_to_save_fig='simul_result.png',
                        xlim=(None, None), ylim=(None, None)
                        ):
        """
            Perform GIWAXS simulation based on crystal definitions.

            This method generates simulated scattering data for one or multiple
            crystals and optionally visualizes the result together with experimental
            data. The simulation uses `make_simulation_new` as the computational backend.

            Parameters
            ----------
            frame_num : int, optional
                Frame index of the loaded data used to extract experimental geometry.

            crystal : dict or list of dict
                Crystal definition(s). Each dictionary must be compatible with the
                simulation pipeline (e.g., contain 'path_to_cif' or 'lat_par').

            plot_result : bool, optional
                If True, plot simulated data overlaid with experimental intensity.

            return_result : bool, optional
                If True, return processed simulation results.

            move_fromMW : bool, optional
                If True, peak positions are shifted from the missing wedge.

            save_fig : bool, optional
                If True, save the generated plot to file.

            path_to_save_fig : str, optional
                File path for saving the figure.

            clims : tuple, optional
                Color limits for experimental image visualization.

            xlim, ylim : tuple, optional
                Axis limits for the plot in reciprocal space coordinates.


            Examples
            --------
            Example of a crystal description dictionary:

                cryst = {
                    'path_to_cif': './cifs/1_BA2PbI4_n1.cif',
                    'orientation': "random",
                    'min_int': 0.2,
                }

            Optional visualization-related parameters (if supported downstream):

                cryst = {
                    'path_to_cif': './cifs/1_BA2PbI4_n1.cif',
                    'orientation': "random",
                    'min_int': 0.2,
                    'cmap': 'winter',
                    'marker': 'o',
                    'marker_size': 50,
                    'line_width': 1,
                    'line_style': "dashed",
                    'text_color': "black"
                }

            Returns
            -------
            list of tuples or tuple or None
                If `return_result` is True:

                - Returns a list of tuples, one per crystal:
                    (q, intensity, mi)

                  where:
                    q : ndarray
                        Scattering vector(s), either:
                        - shape (2, N) for 2D data (q_xy, q_z), or
                        - shape (N,) for 1D data (|q|)

                    intensity : ndarray
                        Normalized intensities sorted in ascending order of |q|.

                    mi : ndarray
                        Miller indices corresponding to each q-point, sorted consistently.

                - If only a single crystal is provided, returns a single tuple
                  (q, intensity, mi) instead of a list.

                If `return_result` is False, returns None.

            Notes
            -----
            - Data are sorted by increasing scattering vector magnitude |q|.
            - For 2D q (shape (2, N)), sorting is based on sqrt(q_xy^2 + q_z^2).
            """

        if crystal:
            return make_simulation_new(self,
                                       frame_num=frame_num, crystal=crystal, plot_result=plot_result,
                                       return_result=return_result, move_fromMW=move_fromMW,
                                       save_fig=save_fig, path_to_save_fig=path_to_save_fig,
                                       clims=clims, xlim=xlim, ylim=ylim
                                       )
        else:
            return make_simulation_old(self,
                                       frame_num=frame_num, path_to_cif=path_to_cif, orientation=orientation,
                                       plot_result=plot_result, plot_mi=plot_mi, return_result=return_result,
                                       move_fromMW=move_fromMW, min_int=min_int, clims=clims, vmin=vmin, vmax=vmax,
                                       linewidth=linewidth, radius=radius, cmap=cmap,
                                       text_color=text_color, save_fig=save_fig, path_to_save_fig=path_to_save_fig,
                                       xlim=xlim, ylim=ylim)

def determine_recalc_key(current_range, global_range, array, step):
    """
        Determines whether recalculation is needed based on the position of minimum and maximum values
        within a given array, relative to specified ranges.

        Parameters
        ----------
        current_range : tuple or list
            The current processing range as (min, max).
        global_range : tuple or list
            The global valid range as (min, max).
        array : array-like
            Data array used to determine extrema (e.g., q-values, intensity values).
        step : float
            Step size used to check whether recalculation is required near boundaries.

        Returns
        -------
        recalc : bool
            True if recalculation is needed (i.e., extrema are close to or outside `global_range`),
            False otherwise.
    """
    recalc = (determine_recalc_key_index(current_range, global_range, array, step, np.nanargmin(array), 0) or
              determine_recalc_key_index(current_range, global_range, array, step, np.nanargmax(array), -1))
    return recalc


def determine_recalc_key_index(current_range, global_range, array, step, arr_index, index):
    """
    Checks whether a recalculation is needed for a given array boundary value.

    This function compares an element of the array (typically at its minimum or maximum)
    with the corresponding boundary of either the current or global range. If the element
    is sufficiently close (within `step`) to the boundary, no recalculation is needed.

    Parameters
    ----------
    current_range : tuple, list, or None
        The current data range (min, max). If None, the global range is used for comparison.
    global_range : tuple or list
        The global valid range (min, max).
    array : array-like
        The array containing data values (e.g., q, intensity, etc.).
    step : float
        Absolute tolerance value used to determine proximity.
    arr_index : int
        Index of the array element to compare (e.g., output of `np.nanargmin` or `np.nanargmax`).
    index : int
        Index of the boundary to compare against (0 for lower, -1 for upper).

    Returns
    -------
    recalc : bool
        True if recalculation is required (i.e., the array value differs from the boundary
        by more than `step`), False otherwise.
    """
    if current_range is None:
        recalc = False if np.isclose(global_range[index],
                                     array[arr_index], atol=step) else True
    else:
        recalc = False if np.isclose(current_range[index],
                                     array[arr_index], atol=step) else True
    return recalc


def calc_matrix(matrix, key, recalc, **kwargs):
    """Function to process each matrix with given parameters."""
    if recalc or not hasattr(matrix, key):
        func_map = {
            "p_y_smpl_pseudopol": matrix._calc_pseudopol_giwaxs_,
            "p_y_lab_pseudopol": matrix._calc_pseudopol_ewald_,
            "p_y_smpl_pol": matrix._calc_pol_giwaxs_,
            "p_y_lab_pol": matrix._calc_pol_ewald_,
            "p_y_ewald": matrix._calc_recip_ewald_,
            "p_y_gid": matrix._calc_recip_giwaxs_
        }
        func_map.get(key, lambda: None)(**kwargs)


def fast_pixel_remap(original_image, new_coords_x, new_coords_y, use_gpu=True, interp_type="INTER_LINEAR",
                     multiprocessing=False):
    """
    Wrapper function to choose between CPU and GPU implementation.
    """
    interp_methods = {
        "INTER_NEAREST": 0,  # Nearest-neighbor interpolation
        "INTER_LINEAR": 1,  # Bilinear interpolation
        "INTER_CUBIC": 2,  # Bicubic interpolation
        "INTER_AREA": 3,  # Area-based interpolation
        "INTER_LANCZOS4": 4,  # Lanczos interpolation
    }

    try:
        interp_method = interp_methods[interp_type]
    except:
        raise ValueError(f"Unknown interpolation method: {interp_type}")

    if use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0:
        return fast_pixel_remap_gpu(original_image, new_coords_x, new_coords_y, interp_method=interp_method)
    else:
        return fast_pixel_remap_cpu(original_image, new_coords_x, new_coords_y, interp_method=interp_method,
                                    multiprocessing=multiprocessing)


def fast_pixel_remap_cpu(original_image, new_coords_x, new_coords_y, interp_method, multiprocessing=False):
    """
    Perform fast pixel remapping using OpenCV's remap function on CPU.
    """

    if original_image.ndim == 2:
        return cv2.remap(original_image, new_coords_y, new_coords_x, interp_method,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    else:
        raise ValueError("Input image must be 2D")


def remap_worker(i, original_image, new_coords_x, new_coords_y, interp_method):
    return cv2.remap(original_image[i], new_coords_x, new_coords_y, interp_method,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)


def fast_pixel_remap_gpu(original_image, new_coords_x, new_coords_y, interp_method):
    """
    Perform pixel remapping using OpenCV's CUDA remap function on GPU.
    """

    gpu_map_x = cv2.cuda_GpuMat()
    gpu_map_y = cv2.cuda_GpuMat()
    gpu_map_x.upload(new_coords_x)
    gpu_map_y.upload(new_coords_y)

    if original_image.ndim == 2:
        gpu_image = cv2.cuda_GpuMat()
        gpu_image.upload(original_image)
        gpu_result = cv2.cuda.remap(gpu_image, gpu_map_x, gpu_map_y, interp_method,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
        return gpu_result.download()

    elif original_image.ndim == 3:
        remapped_image = np.empty((original_image.shape[0], *new_coords_x.shape))
        stream = cv2.cuda.Stream()
        for i in range(original_image.shape[0]):
            gpu_image = cv2.cuda_GpuMat()
            gpu_image.upload(original_image[i])
            gpu_result = cv2.cuda.remap(gpu_image, gpu_map_x, gpu_map_y, interp_method,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan,
                                        stream=stream)
            gpu_result.download(dst=remapped_image[i])
        stream.waitForCompletion()
        return remapped_image
    else:
        raise ValueError("Input image must be 2D")


def process_image(img, mask=None, flipud=False, fliplr=False, transp=False, roi_range=[None, None, None, None],
                  count_range=None):
    """
        Process an image by applying a mask, count range limits, transposition, flips, and ROI selection.

        Parameters
        ----------
        img : np.ndarray
            Input image array.
        mask : np.ndarray, optional
            Boolean mask where True values will be replaced with NaN.
        flipud : bool, optional
            Flip image upside down.
        fliplr : bool, optional
            Flip image left to right.
        transp : bool, optional
            Transpose the image.
        roi_range : tuple of 4 ints, optional
            Region of interest as (y_start, y_end, x_start, x_end). None means full range.
        count_range : tuple of 2 numbers, optional
            Pixel value limits; values outside are set to NaN.

        Returns
        -------
        np.ndarray
            Processed image.
        """

    if img.dtype != np.float32:
        img = img.astype(np.float32)
    if mask is not None:
        mask = mask[roi_range[0]:roi_range[1], roi_range[2]:roi_range[3]]
        img[mask] = np.nan
    if count_range is not None:
        dynamic_mask = np.logical_or(img < count_range[0], img > count_range[1])
        img[dynamic_mask] = np.nan
    if transp:
        img = img.T
    if flipud:
        img = np.flipud(img)
    if fliplr:
        img = np.fliplr(img)
    return img


def add_frame_number(filename, frame_num):
    """
        Appends a zero-padded frame number to a filename before its extension.

        Parameters
        ----------
        filename : str
            Original filename.
        frame_num : int
            Frame number to append.

        Returns
        -------
        str
            Filename with frame number appended, e.g. 'file_0001.ext'.
        None
            If filename is None.
        """
    if filename is None:
        return
    file_root, file_ext = os.path.splitext(filename)
    frame_str = str(frame_num).zfill(4)
    return f"{file_root}_{frame_str}{file_ext}"

