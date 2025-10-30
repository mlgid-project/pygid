from . import CoordMaps
from . import DataLoader
from . import DataSaver, SampleMetadata, ExpMetadata
from .visualization import (get_plot_context, get_plot_params, plot_img_raw, _plot_single_image,
                            plot_simul_data, _plot_profile)
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

try:
    from pygidsim.experiment import ExpParameters
    from pygidsim.giwaxs_sim import GIWAXSFromCif
except:
    pass
    # warnings.warn("pygidsim is not installed. make_simulation function cannot be used.")


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
        number_to_average : int, optional
            The number of frames to average before processing. Default is None (no average).
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
    img_raw: Optional[np.array] = None
    average_all: bool = False
    use_gpu: bool = True
    roi_range: list = field(default_factory=lambda: [None, None, None, None])
    multiprocessing: bool = False
    batch_size: int = 32
    path_batches: Any = None
    sub_class: Any = None
    frame_batches: Any = None
    number_to_average: int = None
    batch_activated: bool = False
    build_image_P03: bool = False
    plot_params = get_plot_params()

    def __post_init__(self):
        """
        Initializes the object after dataclass creation.

        If no image is provided, this method automatically loads the data. It then applies
        flipping to the raw image, computes the q-range and angular range, generates
        correction maps, and applies the necessary corrections. Finally, it activates
        batch analysis if applicable.
        """

        # set default settings fot figures
        # plot_params = type(self).plot_params.update(get_plot_params())

        if hasattr(self.matrix, "sub_matrices") and self.matrix.sub_matrices is not None:
            self.matrix_to_save = self.matrix
            self.matrix = self.matrix.sub_matrices
        self.matrix = [self.matrix] if not isinstance(self.matrix, list) else self.matrix
        self.params = self.matrix[0].params

        if self.img_raw is not None:
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

        self.img_raw = np.array([process_image(img, self.params.mask, self.params.flipud, self.params.fliplr,
                                               self.params.transp, self.roi_range, self.params.count_range) for
                                 img in self.img_raw])

        self.update_conversion()

    def Batch(self, path_to_save, remap_func="det2q_gid", h5_group=None, exp_metadata=None, smpl_metadata=None,
              overwrite_file=True, overwrite_group=False,
              save_result=True, plot_result=False, return_result=False):
        """
        Devidea raw images in batches and process them separately. There are two cases: either path amount
        or frames number in a single h5-file can be bigger than batch size.

        Parameters
        ----------
        path_to_save : str
            Path where the processed data will be saved.
        remap_func : str or callable, optional
            Name or function used to remap the data. Default is "det2q_gid".
        h5_group : h5py.Group, optional
            The name of the group within the HDF5 file under which the matrix data will be stored.
        metadata : Metadata, optional
            Metadata class instance containing metadata values.
        overwrite_file : bool, optional
            Whether to overwrite the file if it already exists. Default is True.
        """

        if self.number_to_average is not None:
            rest = self.batch_size % self.number_to_average
            if rest != 0:
                self.batch_size -= rest
        if isinstance(self.path, list):
            self.path_batches = [self.path[i:i + self.batch_size] for i in range(0, len(self.path), self.batch_size)]
            if self.average_all:
                averaged_image = []
                for path_batch in log_progress(self.path_batches, desc='Progress'):
                    self.img_raw = DataLoader(
                        path=path_batch,
                        frame_num=self.frame_num,
                        dataset=self.dataset,
                        roi_range=self.roi_range,
                        batch_size=self.batch_size,
                        multiprocessing=self.multiprocessing,
                        build_image_P03=self.build_image_P03
                    ).img_raw
                    averaged_image.append(np.mean(self.img_raw, axis=0, keepdims=False))

                self.img_raw = np.array([process_image(img, self.params.mask, self.params.flipud, self.params.fliplr,
                                                       self.params.transp, self.roi_range, self.params.count_range) for
                                         img in averaged_image])
                self.update_conversion()
                remap = getattr(self, remap_func, None)
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


            else:
                for path_batch in log_progress(self.path_batches, desc='Progress'):
                    self.process_batch(
                        path_batch=path_batch,
                        frame_num=self.frame_num,
                        remap_func=remap_func,
                        overwrite_file=overwrite_file,
                        overwrite_group=overwrite_group,
                        exp_metadata=exp_metadata,
                        smpl_metadata=smpl_metadata,
                        path_to_save=path_to_save,
                        h5_group=h5_group
                    )
                    overwrite_file = False
                    overwrite_group = False
                    exp_metadata = None
                    smpl_metadata = None
                if plot_result or return_result:
                    warnings.warn("Plotting and returning of the result are not supported in batch analysis mode.",
                                  category=UserWarning)

        else:
            if isinstance(self.frame_num, list):
                self.frame_batches = []
                for i in range(0, self.number_of_frames, self.batch_size):
                    self.frame_batches.append(self.frame_num[i:min(i + self.batch_size, len(self.frame_num))])
            else:
                self.frame_batches = [list(range(i, min(i + self.batch_size, self.number_of_frames)))
                                      for i in range(0, self.number_of_frames, self.batch_size)]
            if self.average_all:
                averaged_image = []
                for frame_num in log_progress(self.frame_batches, desc='Progress'):
                    self.img_raw = DataLoader(
                        path=self.path,
                        frame_num=frame_num,
                        dataset=self.dataset,
                        roi_range=self.roi_range,
                        batch_size=self.batch_size,
                        multiprocessing=self.multiprocessing,
                        build_image_P03=self.build_image_P03
                    ).img_raw
                    averaged_image.append(np.mean(self.img_raw, axis=0, keepdims=False))

                self.img_raw = np.array([process_image(img, self.params.mask, self.params.flipud, self.params.fliplr,
                                                       self.params.transp, self.roi_range, self.params.count_range) for
                                         img in averaged_image])
                self.update_conversion()
                remap = getattr(self, remap_func, None)
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
            else:
                for frame_num in log_progress(self.frame_batches, desc='Progress'):
                    self.frame_num = frame_num
                    self.process_batch(
                        path_batch=self.path,
                        frame_num=frame_num,
                        remap_func=remap_func,
                        overwrite_file=overwrite_file,
                        overwrite_group=overwrite_group,
                        exp_metadata=exp_metadata,
                        smpl_metadata=smpl_metadata,
                        path_to_save=path_to_save,
                        h5_group=h5_group
                    )
                    overwrite_file = False
                    overwrite_group = False
                    exp_metadata = None
                    smpl_metadata = None
                self.frame_num = None
                if plot_result or return_result:
                    warnings.warn("Plotting and returning of the result are not supported in batch analysis mode.",
                                  category=UserWarning)

    def process_batch(
            self, path_batch, frame_num, remap_func, overwrite_file, overwrite_group,
            exp_metadata, smpl_metadata, path_to_save, h5_group
    ):
        self.img_raw = DataLoader(
            path=path_batch,
            frame_num=frame_num,
            dataset=self.dataset,
            roi_range=self.roi_range,
            batch_size=self.batch_size,
            multiprocessing=self.multiprocessing,
            build_image_P03=self.build_image_P03
        ).img_raw

        self.batch_activated = False

        self.img_raw = np.array([
            process_image(
                img, self.params.mask,
                self.params.flipud, self.params.fliplr,
                self.params.transp, self.roi_range,
                self.params.count_range
            )
            for img in self.img_raw
        ])

        self.update_conversion()

        remap = getattr(self, remap_func, None)
        if exp_metadata is None:
            exp_metadata = ExpMetadata(filename=path_batch)
        else:
            exp_metadata.filename = path_batch

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

        self.img_raw = None
        for attr in [
            "img_gid_q", "img_q", "img_gid_pol", "img_pol",
            "img_gid_pseudopol", "img_pseudopol",
            "rad_cut", "azim_cut", "horiz_cut"
        ]:
            if hasattr(self, attr):
                delattr(self, attr)

    def update_conversion(self):

        """
        Raw image peprocessing that includes averaging, flipping and masking. Call experimental parametes and coordinate
        maps update and application of corrections.

        """
        if self.average_all:
            self.img_raw = np.mean(self.img_raw, axis=0, keepdims=True)
        elif self.number_to_average is not None and self.number_to_average > 1:
            num_images = len(self.img_raw)
            blocks = num_images // self.number_to_average
            averaged_images = []
            for i in range(0, blocks * self.number_to_average, self.number_to_average):
                averaged_images.append(np.mean(self.img_raw[i:i + self.number_to_average], axis=0))
            remaining = num_images % self.number_to_average
            if remaining > 0:
                warnings.warn(f"{remaining} images left, averaging them separately.", UserWarning)
                averaged_images.append(np.mean(self.img_raw[-remaining:], axis=0))
            self.img_raw = np.array(averaged_images)

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
        return plot_img_raw(self.img_raw, self.x, self.y, get_plot_context(type(self).plot_params), return_result,
                            frame_num,
                            plot_result, clims, xlim, ylim,
                            save_fig, path_to_save_fig)

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
        """

        def process_frame(img, mat, path_to_save_fig):
            """
            Calls remapping for a single image.
            """
            return self._remap_single_image_(
                img_raw=img,
                p_y=getattr(mat, kwargs["p_y_key"]),
                p_x=getattr(mat, kwargs["p_x_key"]),
                interp_type=kwargs["interp_type"],
                multiprocessing=kwargs["multiprocessing"],
            )

        keys = [
            "img_gid_q", "img_q", "img_gid_pol",
            "img_pol", "img_gid_pseudopol", "img_pseudopol",
            "rad_cut", "azim_cut", "horiz_cut"
        ]
        for key in keys:
            if hasattr(self, key):
                delattr(self, key)

        result_img = []
        matrix = self.matrix[0] if len(self.matrix) == 1 else None
        if frame_num is None:
            frame_num = list(range(len(self.img_raw)))

        if isinstance(frame_num, list):
            result_img = []
            kwargs_copy = kwargs.copy()
            kwargs_copy["return_result"] = True
            kwargs_copy["save_result"] = False
            if kwargs["multiprocessing"]:
                with ThreadPoolExecutor() as executor:
                    result_img = list(
                        executor.map(lambda frame: self._remap_general_(frame, **kwargs_copy)[2], frame_num))

            else:
                for frame in frame_num:
                    result_img.append(self._remap_general_(frame, **kwargs_copy)[2])

            self.ai_list = []
            for frame in frame_num:
                if isinstance(self.params.ai, list):
                    self.ai_list.append(self.params.ai[frame])
                else:
                    self.ai_list.append(self.params.ai)

            self.converted_frame_num = []
            if self.frame_num is None:
                self.converted_frame_num = frame_num
            else:
                for i in frame_num:
                    if isinstance(self.frame_num, int) or isinstance(self.frame_num, np.int64):
                        self.converted_frame_num.append(self.frame_num)
                    else:
                        self.converted_frame_num.append(self.frame_num[i])

            setattr(self, kwargs["result_attr"], result_img)
            if kwargs["save_result"]:
                self.save_nxs(path_to_save=kwargs["path_to_save"],
                              h5_group=kwargs["h5_group"],
                              overwrite_file=kwargs["overwrite_file"],
                              overwrite_group=kwargs["overwrite_group"],
                              exp_metadata=kwargs["exp_metadata"],
                              smpl_metadata=kwargs["smpl_metadata"],
                              )
            if kwargs["return_result"]:
                matrix_x = getattr(self.matrix[0], kwargs["x_key"])
                matrix_y = getattr(self.matrix[0], kwargs["y_key"])
                return matrix_x, matrix_y, result_img
        else:
            img = self.img_raw[frame_num]
            mat = matrix or self.matrix[frame_num]
            result_img = process_frame(img, mat, frame_num)
            self.ai_list = mat.ai
            self.converted_frame_num = [self.frame_num] if hasattr(self, 'frame_num') else [frame_num]
            setattr(self, kwargs["result_attr"], [result_img])
            if kwargs["save_result"]:
                self.save_nxs(path_to_save=kwargs["path_to_save"],
                              h5_group=kwargs["h5_group"],
                              overwrite_file=kwargs["overwrite_file"],
                              overwrite_group=kwargs["overwrite_group"],
                              exp_metadata=kwargs["exp_metadata"],
                              smpl_metadata=kwargs["smpl_metadata"],
                              )
            if kwargs["return_result"]:
                return getattr(mat, kwargs["x_key"]), getattr(mat, kwargs["y_key"]), result_img

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
            Path to save the figure if `save_fig` is True. Default is "img.png".
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
            Path to save the figure if `save_fig` is True. Default is "img.png".
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
            Path to save the figure if `save_fig` is True. Default is "img.png".
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

        # If batch mode is active, delegate execution to the batch processor
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

        # Remap detector image from pixel space to polar reciprocal space
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
            smpl_metadata=smpl_metadata
        )
        img = [img] if not isinstance(img, list) else img

        # Plot and/or save each polar reciprocal-space map if requested
        if plot_result or save_fig:
            for i in range(len(img)):
                _plot_single_image(get_plot_context(type(self).plot_params), img[i], x, y, clims, xlim, ylim,
                                   r"$|q|\ \mathrm{[\AA^{-1}]}$",
                                   r"$\chi$ [$\degree$]", 'auto', plot_result,
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
            Path to save the figure if `save_fig` is True. Default is "img.png".
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
                             overwrite_group,
                             save_result, plot_result, return_result)
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
                           angular_range=angular_range,
                           dang=dang,
                           dq=dq)

        # Remap detector image from pixel space to polar reciprocal space
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
                                   r"$|q|\ \mathrm{[\AA^{-1}]}$",
                                   r"$\chi$ [$\degree$]", 'auto', plot_result,
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
            If True, returns the calculated axes and pseudopolar image(s).
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
            Path to save the figure if `save_fig` is True. Default is "img.png".
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

        # Delegate to batch processor if batch mode is active
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

        # Remap detector image to pseudopolar coordinates
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
                                   r"$|q|\ \mathrm{[\AA^{-1}]}$",
                                   r"$q_{\phi}\ \mathrm{[\AA^{-1}]}$", 'auto', plot_result,
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
            If True, returns the calculated axes and pseudopolar image(s).
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
            Path to save the figure if `save_fig` is True. Default is "img.png".
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

        # Delegate to batch processor if batch mode is active
        if self.batch_activated:
            res = self.Batch(path_to_save, "det2pseudopol_gid", h5_group, exp_metadata, smpl_metadata, overwrite_file,
                             overwrite_group,
                             save_result, plot_result, return_result)
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
                                   r"$|q|\ \mathrm{[\AA^{-1}]}$",
                                   r"$q_{\phi}\ \mathrm{[\AA^{-1}]}$", 'auto', plot_result,
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
            xlim=None,
            ylim=None,
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
            name)

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
            xlim=None,
            ylim=None,
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
        rad_cut : 1D-array or list of 1D-arrays
            Integrated image profile rad_cut.
        """

        key = 'transmission'
        remap_func = "radial_profile"
        name = "rad_cut"

        return self.calculate_radial_profile(
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
            name)

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
                Computed radial intensity profile(s).
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
            _plot_profile(get_plot_context(type(self).plot_params), q_abs_values, radial_profile, r"$q_{abs}\ [\AA^{-1}]$", shift,
                               xlim, ylim, plot_result, save_fig, path_to_save_fig)
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
            xlim=None,
            ylim=None,
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
                Radial q-range as [min, max] in Å⁻¹. If None, full range is used.
            angular_range : list, optional
                Azimuthal integration range in degrees as [min, max] (default [0, 90]).
            multiprocessing : bool or None, optional
                If True, use multiprocessing for faster processing. If None, uses default class setting.
            return_result : bool, optional
                If True, returns the computed azimuthal profile.
            save_result : bool, optional
                If True, saves the profile to an HDF5 file.
            save_fig : bool, optional
                If True, saves a plot of the profile.
            path_to_save_fig : str, optional
                File path for saving the figure if `save_fig` is True. Default is 'azim_cut.tiff'.
            plot_result : bool, optional
                If True, displays the azimuthal profile plot.
            shift : float, optional
                Vertical shift applied to the profile for display purposes. Default is 1.
            xlim : tuple or None, optional
                Limits for the X-axis (degrees). Default is None (auto).
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
                If True, overwrites existing HDF5 group. Default is False.
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
            xlim=None,
            ylim=None,
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
            Radial q-range as [min, max] in Å⁻¹. If None, full range is used.
        angular_range : list, optional
            Azimuthal integration range in degrees as [min, max] (default [0, 90]).
        multiprocessing : bool or None, optional
            If True, use multiprocessing for faster processing. If None, uses default class setting.
        return_result : bool, optional
            If True, returns the computed azimuthal profile.
        save_result : bool, optional
            If True, saves the profile to an HDF5 file.
        save_fig : bool, optional
            If True, saves a plot of the profile.
        path_to_save_fig : str, optional
            File path for saving the figure if `save_fig` is True. Default is 'azim_cut.tiff'.
        plot_result : bool, optional
            If True, displays the azimuthal profile plot.
        shift : float, optional
            Vertical shift applied to the profile for display purposes. Default is 1.
        xlim : tuple or None, optional
            Limits for the X-axis (degrees). Default is None (auto).
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
            If True, overwrites existing HDF5 group. Default is False.
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
            Computed azimuthal intensity profile(s).
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
            _plot_profile(get_plot_context(type(self).plot_params), phi_abs_values, azim_profile, r"$\chi\ [\degree]$", shift, xlim,
                               ylim, plot_result, save_fig, path_to_save_fig)
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
            xlim=None,
            ylim=None,
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
        xlabel = r'$q_{xy}$ [$\mathrm{\AA}^{-1}$]'
        if plot_result or save_fig:
            _plot_profile(get_plot_context(type(self).plot_params), q_hor_values, horiz_profile, xlabel, shift, xlim,
                               ylim, plot_result, save_fig, path_to_save_fig)
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

    def make_simulation(self, frame_num=0, path_to_cif=None, orientation=None,
                        plot_result=True, plot_mi=False, return_result=False,
                        min_int=None, clims=None, vmin=0, vmax=1, linewidth=1, radius=0.1, cmap=None,
                        text_color='black', save_result=False, path_to_save='simul_result.png'):
        """
        Simulates and visualizes diffraction pattern for the given crystallographic data.

        Parameters:
            frame_num (int): Image frame number to visualize.
            path_to_cif (str): Path to a CIF file containing the crystal structure.
            orientation (list): Crystal orientation. None the for poweder pattern.
            plot_result (bool): Whether to plot the result of simulation and experimental data.
            plot_mi (bool): Whether to plot the Miller indices.
            return_result (bool): Whether to return the result of simulation.
            min_int (float or None): Minimum intensity threshold for display
            clims (list): Intensity range for the color scale of experimental data
            vmin (float): Normalization limits for the color scale of simulated data
            vmax (float): Normalization limits for the color scale of simulated data
            linewidth (float): Simulated peaks line thickness for visualization
            radius (float): Simulated peaks radius for visualization
            cmap (matplotlib colormap): Colormap used in the visualization.
            text_color (str): Color of any text annotations.
            save_result (bool): If True, saves the figure image.
            path_to_save (str): File path to save the simulation figure.

        Returns
        -------
        (q_xy, q_z) : (array, array)
           q_xy, q_z positions of the simulated data (in 1/A).
                            or
        q_abs: array
            q_abs positions of the simulated rings

        intensity : array
           The intensity values of the simulated data.
        mi : array
           Miller indices of the simulated data.

        """
        q_xy_max = self.matrix[0].q_xy_range[1]
        q_z_max = self.matrix[0].q_z_range[1]
        radius /= np.sqrt(q_xy_max ** 2 + q_z_max ** 2) / 4.37
        ai = self.matrix[0].ai if len(self.matrix) == 1 else self.matrix[frame_num].ai
        try:
            simul_params = ExpParameters(q_xy_max=q_xy_max, q_z_max=q_z_max, en=12398 / self.params.wavelength, ai=ai)
        except:
            raise ValueError("pygidsim package is not installed. Please visit https://github.com/mlgid-project/pygidSIM")

        path_to_cif = [path_to_cif] if not isinstance(path_to_cif, list) else path_to_cif

        for path in path_to_cif:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"File does not exist: {path}")

        min_int = [min_int] if not isinstance(min_int, list) else min_int

        if orientation is not None:
            orientation = [orientation] if not isinstance(orientation[0], list) else orientation
        else:
            orientation = [orientation]
        if len(orientation) == 1:
            orientation *= len(path_to_cif)
        if len(path_to_cif) == 1:
            path_to_cif *= len(orientation)
        if len(min_int) == 1:
            min_int *= len(path_to_cif)

        if len(path_to_cif) != len(orientation) or len(path_to_cif) != len(orientation):
            raise ValueError("orientation and path_to_cif have different length. They should be equal or "
                             "at least one should be equal to 1")

        simulated_data = [simul_single_data(path_to_cif[i], orientation[i], simul_params, min_int[i]) for i in
                          range(len(path_to_cif))]

        q_xy, q_z, img = self._get_q_data(frame_num)


        if plot_result:
            plot_simul_data(get_plot_context(type(self).plot_params), img[0], q_xy, q_z, clims, simulated_data,
                            cmap, save_result, path_to_save,
                    vmin, vmax, linewidth, radius, text_color, plot_mi)
            logging.info(f"frame_num = {frame_num} was plotted")
        if return_result:
            simulated_data = sort_simul_data(simulated_data)
            if len(simulated_data):
                return simulated_data[0]
            else:
                return simulated_data


def sort_simul_data(simulated_data):
    """
    Sorts simulated scattering data by increasing q-values.

    Parameters
    ----------
    simulated_data : list of tuples
        A list where each element is a tuple of the form `(q, value, mi)`:
        - `q` : array-like
            Wavevector values. Can be 1D (`(N,)`) or 2D (`(2, N)`), where the latter
            represents (q_x, q_z) or similar components.
        - `value` : array-like
            Corresponding simulated intensities.
        - `mi` : array-like
            Miller indices .

    Returns
    -------
    simulated_data : list of tuples
        The same list where each tuple has been sorted by increasing |q|.

    Raises
    ------
    AssertionError
        If input arrays within a tuple do not have consistent lengths.
    ValueError
        If the shape of `q` is not supported.

    Notes
    -----
    - If `q` is 2D with shape (2, N), sorting is performed by the magnitude |q|.
    - The sorting is applied in-place to the input list elements.
    """
    for i in range(len(simulated_data)):
        q, value, mi = simulated_data[i]

        q = np.array(q)
        value = np.array(value)
        mi = np.array(mi)

        assert q.shape[-1] == len(value) == len(mi), "Mismatch in array lengths"

        if q.ndim == 2 and q.shape[0] == 2:
            q_abs = np.linalg.norm(q, axis=0)
            indices = np.argsort(q_abs)
        elif q.ndim == 1:
            indices = np.argsort(q)
        else:
            raise ValueError(f"Unsupported q shape: {q.shape}")

        # Apply sorting
        q_sorted = q[:, indices] if q.ndim == 2 else q[indices]
        value_sorted = np.array(value)[indices]
        mi_sorted = np.array(mi)[indices]

        simulated_data[i] = (q_sorted, value_sorted, mi_sorted)
    return simulated_data


def simul_single_data(path_to_cif, orientation, simul_params, min_int):
    """
    Simulates GIWAXS data from a CIF file and filters the results based on intensity.

    This function generates simulated scattering data using the specified CIF structure
    and simulation parameters. The resulting intensities are normalized and optionally
    filtered by a minimum intensity threshold. The Miller indices (m_i) are adjusted
    to select the most relevant entries.

    Parameters
    ----------
    path_to_cif : str
        Path to the CIF file containing the crystal structure.
    orientation : array-like or None
        Orientation matrix (3x3) or None. If provided, determines the sample orientation
        for simulation.
    simul_params : dict
        Dictionary of simulation parameters to be passed to `GIWAXSFromCif`.
    min_int : float or None
        Minimum normalized intensity threshold. Peaks below this value are filtered out.
        If None, all simulated points are retained.

    Returns
    -------
    q : ndarray
        Scattering vector(s). Shape (2, N) for 2D data or (N,) for 1D data, depending
        on the orientation mode.
    intensity : ndarray
        Normalized scattering intensity values.
    mi : ndarray
        Corresponding Miller indices after filtering and sorting.

    Notes
    -----
    - Intensities are normalized by their maximum value.
    - If `orientation` is provided, q-vectors are 2D (q_xy, q_z); otherwise, 1D magnitudes.
    """
    logging.info(
        f"Simulating GIWAXS data: path_to_cif='{path_to_cif}', "
        f"orientation={orientation}, min_int={min_int}"
    )

    if orientation is not None:
        orientation = np.array(orientation)
    el = GIWAXSFromCif(path_to_cif, simul_params)
    q, intensity, mi = el.giwaxs.giwaxs_sim(orientation, return_mi=True)
    mi = np.array([x[0] if len(x) == 1 else select_best_array(x) for x in mi])
    intensity /= np.max(intensity)

    if min_int is not None:
        index = ~(intensity < min_int)
        mi = mi[index]
        intensity = intensity[index]
        sort_index = np.argsort(intensity)

        mi = mi[sort_index]
        intensity = intensity[sort_index]
        if orientation is not None:
            q = np.stack((q[0][index], q[1][index]), axis=0)
            q = q[:, sort_index]
        else:
            q = q[index]
            q = q[sort_index]
    return q, intensity, mi


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


def select_best_array(arrays):
    """
        Selects the "best" array from a list based on sum of squares and element magnitudes.

        Parameters
        ----------
        arrays : list of ndarray
            List of arrays to choose from.

        Returns
        -------
        ndarray
            The array with the minimal sum of squares, breaking ties by element magnitude.
        """

    def sort_key(arr):
        return (
            np.sum(arr ** 2),
            *[(abs(x), -x) for x in arr]
        )

    return min(arrays, key=sort_key)
