from . import CoordMaps
from . import DataLoader
from . import DataSaver, SampleMetadata, ExpMetadata
import os
from typing import Optional, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib import colors
from matplotlib.cm import ScalarMappable
import cv2
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.notebook import tqdm as log_progress
from matplotlib.ticker import LogLocator
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as ticker
from adjustText import adjust_text
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)

try:
    from pygidsim.experiment import ExpParameters
    from pygidsim.giwaxs_sim import GIWAXSFromCif
except:
    warnings.warn("pygidsim is not installed. make_simulation function cannot be used.")

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

    def __post_init__(self):
        """
        Calls data loader if imgage if not provided. Applies flipping of raw image, calls q- and angular
        ranges calculation. Calculates correction maps and applies corrections. Activates batch analysis.
        """

        # set default settings fot figures
        self.set_plot_defaults()

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
                print(f"Warning: {remaining} images left, averaging them separately.")
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
        else:
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
                q_min = self.matrix[ai_min_index].q_min
                ang_min = self.matrix[ai_min_index].ang_min
                ang_max = self.matrix[ai_min_index].ang_max
                q_lab_from_p = self.matrix[ai_min_index].q_lab_from_p

                ai_max_index = np.argmax([matrix.ai for matrix in self.matrix])
                self.matrix[ai_max_index].corr_matrices = []
                self.matrix[ai_max_index].ang_min = ang_min
                self.matrix[ai_max_index].ang_max = ang_max
                self.matrix[ai_max_index]._coordmaps_update_()
                q_xy_range.append(self.matrix[ai_max_index].q_xy_range[1])
                q_z_range.append(self.matrix[ai_max_index].q_z_range[1])
                q_max = self.matrix[ai_max_index].q_max

                for matrix in self.matrix:
                    matrix.q_xy_range = q_xy_range
                    matrix.q_z_range = q_z_range
                    matrix.q_max = q_max
                    matrix.q_min = q_min
                    matrix.ang_max = ang_max
                    matrix.ang_min = ang_min
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
            print("dark_current is subtracted")
        for corr_matrix in corr_matrices:
            if corr_matrix != 'dark_current' and corr_matrices[corr_matrix] is not None:
                if corr_matrix == 'absorption_corr_matrix' or corr_matrix == 'lorentz_corr_matrix':
                    for i, matrix in enumerate(self.matrix):
                        self.img_raw[i] /= matrix.corr_matrices.__dict__[corr_matrix]
                print(corr_matrix, "was applied")
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

    def set_plot_defaults(self, font_size=14, axes_titlesize=14, axes_labelsize=18, grid=False, grid_color='gray',
                          grid_linestyle='--', grid_linewidth=0.5, xtick_labelsize=14, ytick_labelsize=14,
                          legend_fontsize=12, legend_loc='best', legend_frameon=True, legend_borderpad=1.0,
                          legend_borderaxespad=1.0, figure_titlesize=16, figsize=(6.4, 4.8), axes_linewidth = 0.5,
                          savefig_dpi=600, savefig_transparent=False, savefig_bbox_inches=None,
                          savefig_pad_inches=0.1, line_linewidth=2, line_color='blue', line_linestyle='-',
                          line_marker=None, scatter_marker='o', scatter_edgecolors='black', grid_major_linestyle='-',
                          grid_minor_linestyle=':', grid_major_linewidth=0.7, grid_minor_linewidth=0.3,
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
        - grid_major_linestyle (str): Style of major gridlines (e.g., '-', '--').
        - grid_minor_linestyle (str): Style of minor gridlines (e.g., ':').
        - grid_major_linewidth (float): Width of major gridlines.
        - grid_minor_linewidth (float): Width of minor gridlines.
        - cmap (str): Image colormap
        """
        # Font settings
        plt.rc('font', size=font_size)  # Controls default text sizes
        plt.rc('axes', titlesize=axes_titlesize)  # Font size for axes title
        plt.rc('axes', labelsize=axes_labelsize)  # Font size for axes labels (x and y)
        plt.rc('xtick', labelsize=xtick_labelsize)  # Font size for x-axis tick labels
        plt.rc('ytick', labelsize=ytick_labelsize)  # Font size for y-axis tick labels
        plt.rc('legend', fontsize=legend_fontsize)  # Font size for legend text
        plt.rc('figure', titlesize=figure_titlesize)  # Font size for figure title

        # Axes settings
        plt.rcParams['axes.grid'] = grid  # Enable or disable grid
        plt.rcParams['axes.grid'] = grid  # Enable or disable grid
        plt.rcParams['axes.linewidth'] = axes_linewidth
        plt.rcParams['grid.color'] = grid_color  # Grid line color
        plt.rcParams['grid.linestyle'] = grid_linestyle  # Grid line style
        plt.rcParams['grid.linewidth'] = grid_linewidth  # Grid line width

        plt.rcParams['legend.loc'] = legend_loc  # Legend location
        plt.rcParams['legend.frameon'] = legend_frameon  # Enable or disable legend frame
        plt.rcParams['legend.borderpad'] = legend_borderpad  # Padding inside the legend box
        plt.rcParams['legend.borderaxespad'] = legend_borderaxespad  # Padding between legend and axes

        # Figure settings
        plt.rcParams['figure.figsize'] = figsize  # Default figure size
        plt.rcParams['savefig.dpi'] = savefig_dpi  # DPI when saving figure
        plt.rcParams['savefig.transparent'] = savefig_transparent  # Transparent background for saved figure
        plt.rcParams['savefig.bbox'] = savefig_bbox_inches  # Bounding box for saving figure
        plt.rcParams['savefig.pad_inches'] = savefig_pad_inches  # Padding when saving figure

        # Line settings
        plt.rc('lines', linewidth=line_linewidth)  # Line width for plot lines
        plt.rc('lines', color=line_color)  # Line color for plot lines
        plt.rc('lines', linestyle=line_linestyle)  # Line style for plot lines
        if line_marker is not None:
            plt.rc('lines', marker=line_marker)  # Marker style for plot lines

        # Scatter settings
        plt.rc('scatter', marker=scatter_marker)  # Marker style for scatter plot
        plt.rcParams['scatter.edgecolors'] = scatter_edgecolors  # Edge color for scatter markers

        # Set grid settings
        plt.rcParams['axes.grid'] = grid  # Enable or disable grid
        plt.rcParams['grid.color'] = grid_color  # Grid line color
        plt.rcParams['grid.linestyle'] = grid_linestyle  # Grid line style
        plt.rcParams['grid.linewidth'] = grid_linewidth  # Grid line width

        # Image colormap
        plt.rcParams['image.cmap'] = cmap

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

        if self.img_raw is None:
            raise AttributeError("img_raw is not loaded")
        if not isinstance(self.img_raw, np.ndarray):
            self.img_raw = np.array(self.img_raw)

        if frame_num is None and self.img_raw.shape[0] != 1:
            frame_num = np.arange(1, self.img_raw.shape[0],1)
        if isinstance(frame_num, list) or isinstance(frame_num, np.ndarray):
            img_list = []
            for num in frame_num:
                x, y, img = self.plot_img_raw(return_result=True, frame_num=num, plot_result=plot_result,
                             clims=clims, xlim=xlim, ylim=ylim, save_fig=save_fig,
                             path_to_save_fig=make_numbered_filename(path_to_save_fig, num))
                img_list.append(img)
            if return_result:
                return self.x, self.y, img_list
            return

        if frame_num is None:
            frame_num = 0
        img = np.array(self.img_raw[frame_num])

        if clims is None:
            clims = [np.nanmin(img[img>0]), np.nanmax(img)]

        if self.img_raw is not None:

            def fill_limits(lim, data):
                return [np.nanmin(data) if lim[0] is None else lim[0],
                        np.nanmax(data) if lim[1] is None else lim[1]]

            xlim = fill_limits(xlim, self.x)
            ylim = fill_limits(ylim, self.y)

            fig, ax = plt.subplots()
            plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
            img[img < 0] = clims[0]
            log_img = np.log(img / clims[1])
            log_img = np.nan_to_num(log_img, nan=np.nan, posinf=np.log(clims[0] / clims[1]),
                                    neginf=np.log(clims[0] / clims[1]))

            p = ax.imshow(log_img,
                          vmin=np.log(clims[0] / clims[1]), vmax=np.log(clims[1] / clims[1]),
                          extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                          aspect='equal',
                          origin='lower')

            ax.set_xlabel(r'$y$ [px]')
            ax.set_ylabel(r'$z$ [px]')
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=False, prune=None, nbins=4))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, prune=None, nbins=4))
            ax.tick_params(axis='both')

            cb = fig.colorbar(mappable=p, ax=ax)
            cb.set_label(label='Intensity [arb. units]')
            cb.ax.yaxis.labelpad = 1
            cb.set_ticks([np.log(clims[0] / clims[1]), np.log(clims[1] / clims[1])])
            cb.set_ticklabels([change_clim_format(str(clims[0])), change_clim_format(str(clims[1]))])

            if save_fig:
                if path_to_save_fig is not None:
                    if (path_to_save_fig.endswith('.svg') or path_to_save_fig.endswith('.pdf')
                            or path_to_save_fig.endswith('.eps') or path_to_save_fig.endswith('.pgf')):
                        plt.axis('square')
                        plt.savefig(path_to_save_fig)
                    else:
                        plt.savefig(path_to_save_fig, bbox_inches='tight')
                    print(f"Saved figure in {path_to_save_fig}")
                else:
                    raise ValueError("path_to_save_fig is not defined.")
                if not plot_result:
                    plt.close()
                    del fig, ax
            else:
                if plot_result:
                    plt.show()
                else:
                    plt.close()
                    del fig, ax
        else:
            print("img_raw is not loaded")

        if return_result:
            return self.x, self.y, img

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

    def det2q_gid(self, frame_num=None, interp_type="INTER_LINEAR", multiprocessing=None, return_result=False,
                  q_xy_range=None, q_z_range=None, dq=None,
                  plot_result=False, clims=None,
                  xlim=(None, None), ylim=(None, None),
                  save_fig=False, path_to_save_fig="img.png",
                  save_result=False,
                  path_to_save='result.h5',
                  h5_group=None,
                  overwrite_file=True,
                  overwrite_group=False,
                  exp_metadata=None,
                  smpl_metadata=None,
                  ):
        """
            Converts detector image to reciprocal space map (q_xy, q_z) for GID geometry.

            Parameters
            ----------
            frame_num : int, list or None, optional
                Frame number to process. If None, defaults to the first or current frame.
            interp_type : str, optional
                Interpolation method used for remapping. Default is "INTER_LINEAR".
            multiprocessing : bool, optional
                Whether to use multiprocessing for processing. If None, uses default setting.
            return_result : bool, optional
                If True, returns the result: two axes and images.
            q_xy_range : tuple or None, optional
                Tuple specifying the min and max of q_xy range. If None, uses full range.
            q_z_range : tuple or None, optional
                Tuple specifying the min and max of q_z range. If None, uses full range.
            plot_result : bool, optional
                Whether to plot the resulting reciprocal space map. Default is False.
            clims : tuple, optional
                Color scale limits (vmin, vmax) for plotting. Default is (1e1, 4e4).
            xlim : tuple, optional
                X-axis limits for plot. Default is (None, None).
            ylim : tuple, optional
                Y-axis limits for plot. Default is (None, None).
            save_fig : bool, optional
                Whether to save the figure. Default is False.
            path_to_save_fig : str, optional
                Path to save the figure if save_fig is True. Default is "img.png".
            save_result : bool, optional
                Whether to save the resulting data to h5-file. Default is False.
            path_to_save : str, optional
                Path to save the result if save_result is True. Default is 'result.h5'.
            h5_group : str or None, optional
                HDF5 group under which to save data.
            overwrite_file : bool, optional
                Whether to overwrite existing file. Default is True.
            metadata : dict or None, optional
                Additional metadata to store with result. Default is None.

            Returns
            -------
            q_xy : array
                The q_xy-axis values of the converted data (in 1/A).
            q_z : array
                The q_z-axis values of the converted data (in 1/A).
            img_gid_q : 2D-array or list of 2D-arrays
                The converted image img_gid_q.
            """

        if self.batch_activated:
            res = self.Batch(path_to_save, "det2q_gid", h5_group, exp_metadata, smpl_metadata, overwrite_file,
                             overwrite_group, save_result, plot_result, return_result)
            self.batch_activated = True
            return res

        recalc = (determine_recalc_key(q_xy_range, self.matrix[0].q_xy_range, self.matrix[0].q_xy, self.matrix[0].dq) \
                      if hasattr(self.matrix[0], "q_xy") else True) or (
                     determine_recalc_key(q_z_range, self.matrix[0].q_z_range,
                                          self.matrix[0].q_z, self.matrix[0].dq) \
                         if hasattr(self.matrix[0], "q_z") else True)
        if dq is not None:
            recalc = True if dq != self.matrix[0].dq else recalc

        self.calc_matrices("p_y_gid", recalc, multiprocessing=multiprocessing or self.multiprocessing,
                           q_xy_range=q_xy_range,
                           q_z_range=q_z_range, dq=dq)
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
        img = [img] if not isinstance(img, list) else img
        if plot_result or save_fig:
            for i in range(len(img)):
                self._plot_single_image(img[i], x, y, clims, xlim, ylim, r'$q_{xy}$ [$\mathrm{\AA}^{-1}$]',
                                        r'$q_{z}$ [$\mathrm{\AA}^{-1}$]', 'equal', plot_result,
                                        save_fig, add_frame_number(path_to_save_fig, i))
        if return_result:
            return x, y, img

    def det2q(self, frame_num=None, interp_type="INTER_LINEAR", multiprocessing=None, return_result=False,
              q_x_range=None, q_y_range=None, dq=None,
              plot_result=False, clims=None,
              xlim=(None, None), ylim=(None, None),
              save_fig=False, path_to_save_fig="img.png",
              save_result=False,
              path_to_save='result.h5',
              h5_group=None,
              overwrite_file=True,
              overwrite_group=False,
              exp_metadata=None,
              smpl_metadata=None,
              ):
        """
            Converts detector image to reciprocal space map (q_x, q_y) for transmission geometry.

            Parameters
            ----------
            frame_num : int, list or None, optional
                Frame number to process. If None, defaults to the first or current frame.
            interp_type : str, optional
                Interpolation method used for remapping. Default is "INTER_LINEAR".
            multiprocessing : bool, optional
                Whether to use multiprocessing for processing. If None, uses default setting.
            return_result : bool, optional
                If True, returns the result: two axes and images.
            q_x_range : tuple or None, optional
                Tuple specifying the min and max of q_x range. If None, uses full range.
            q_y_range : tuple or None, optional
                Tuple specifying the min and max of q_y range. If None, uses full range.
            plot_result : bool, optional
                Whether to plot the resulting reciprocal space map. Default is False.
            clims : tuple, optional
                Color scale limits (vmin, vmax) for plotting. Default is (1e1, 4e4).
            xlim : tuple, optional
                X-axis limits for plot. Default is (None, None).
            ylim : tuple, optional
                Y-axis limits for plot. Default is (None, None).
            save_fig : bool, optional
                Whether to save the figure. Default is False.
            path_to_save_fig : str, optional
                Path to save the figure if save_fig is True. Default is "img.png".
            save_result : bool, optional
                Whether to save the resulting data to h5-file. Default is False.
            path_to_save : str, optional
                Path to save the result if save_result is True. Default is 'result.h5'.
            h5_group : str or None, optional
                HDF5 group under which to save data.
            recalc : bool, optional
                Whether to force recalculation even if coodinate map exists. Default is False.
            overwrite_file : bool, optional
                Whether to overwrite existing file. Default is True.
            metadata : dict or None, optional
                Additional metadata to store with result. Default is None.

            Returns
            -------
            q_x : array
                The q_x-axis values of the converted data (in 1/A).
            q_y : array
                The q_y-axis values of the converted data (in 1/A).
            img_q : 2D-array or list of 2D-arrays
                The converted image img_q.

            """
        if self.batch_activated:
            res = self.Batch(path_to_save, "det2q", h5_group, exp_metadata, smpl_metadata, overwrite_file,
                             overwrite_group,
                             save_result, plot_result, return_result)
            self.batch_activated = True
            return res

        recalc = (determine_recalc_key(q_x_range, self.matrix[0].q_x_range, self.matrix[0].q_x, self.matrix[0].dq) \
                      if hasattr(self.matrix[0], "q_x") else True) or (
                     determine_recalc_key(q_y_range, self.matrix[0].q_y_range,
                                          self.matrix[0].q_y, self.matrix[0].dq) \
                         if hasattr(self.matrix[0], "q_y") else True)

        if dq is not None:
            recalc = True if dq != self.matrix[0].dq else recalc

        self.calc_matrices("p_y_ewald", recalc, multiprocessing=multiprocessing or self.multiprocessing,
                           q_x_range=q_x_range, q_y_range=q_y_range, dq=dq)
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
        if plot_result or save_fig:
            for i in range(len(img)):
                self._plot_single_image(img[i], x, y, clims, xlim, ylim, r'$q_{x}$ [$\mathrm{\AA}^{-1}$]',
                                        r'$q_{y}$ [$\mathrm{\AA}^{-1}$]', 'equal', plot_result,
                                        save_fig, add_frame_number(path_to_save_fig, i))
        if return_result:
            return x, y, img

    def det2pol(self, frame_num=None, interp_type="INTER_LINEAR", multiprocessing=None, return_result=False,
                radial_range=None, angular_range=None, dang=None, dq=None,
                plot_result=False, clims=None,
                xlim=(None, None), ylim=(None, None),
                save_fig=False, path_to_save_fig="img.png",
                save_result=False,
                path_to_save='result.h5',
                h5_group=None,
                overwrite_file=True,
                overwrite_group=False,
                exp_metadata=None,
                smpl_metadata=None,
                ):
        """
         Converts detector image to polar coordinates for transmission geometry.

         Parameters
         ----------
         frame_num : int, list or None, optional
             Frame number to process. If None, defaults to the first or current frame.
         interp_type : str, optional
             Interpolation method used for remapping. Default is "INTER_LINEAR".
         multiprocessing : bool, optional
             Whether to use multiprocessing for processing. If None, uses default setting.
         return_result : bool, optional
             If True, returns the result: two axes and images.
         radial_range : tuple or None, optional
             Tuple specifying the min and max of q. If None, uses full range.
         angular_range : tuple or None, optional
             Tuple specifying the min and max of azimuthal angle. If None, uses full range.
         plot_result : bool, optional
             Whether to plot the resulting reciprocal space map. Default is False.
         clims : tuple, optional
             Color scale limits (vmin, vmax) for plotting. Default is (1e1, 4e4).
         xlim : tuple, optional
             X-axis limits for plot. Default is (None, None).
         ylim : tuple, optional
             Y-axis limits for plot. Default is (None, None).
         save_fig : bool, optional
             Whether to save the figure. Default is False.
         path_to_save_fig : str, optional
             Path to save the figure if save_fig is True. Default is "img.png".
         save_result : bool, optional
             Whether to save the resulting data to h5-file. Default is False.
         path_to_save : str, optional
             Path to save the result if save_result is True. Default is 'result.h5'.
         h5_group : str or None, optional
             HDF5 group under which to save data.
         recalc : bool, optional
             Whether to force recalculation even if coodinate map exists. Default is False.
         overwrite_file : bool, optional
             Whether to overwrite existing file. Default is True.
         metadata : dict or None, optional
             Additional metadata to store with result. Default is None.

         Returns
         -------
         q_pol : array
             The q_pol-axis values of the converted data (in 1/A).
         ang_pol : array
             The ang_pol-axis values of the converted data (in degrees).
         img_pol : 2D-array or list of 2D-arrays
             The converted image img_pol.

         """

        if self.batch_activated:
            res = self.Batch(path_to_save, "det2pol", h5_group, exp_metadata, smpl_metadata, overwrite_file,
                             overwrite_group,
                             save_result, plot_result, return_result)
            self.batch_activated = True
            return res

        recalc = ((determine_recalc_key(angular_range, [self.matrix[0].ang_min, self.matrix[0].ang_max],
                                        self.matrix[0].ang_pol, self.matrix[0].dang) \
                       if hasattr(self.matrix[0], "ang_pol") else True) or
                  (determine_recalc_key(radial_range, [self.matrix[0].q_min, self.matrix[0].q_max],
                                        self.matrix[0].q_pol, self.matrix[0].dq) \
                       if hasattr(self.matrix[0], "q_pol") else True))
        if dq is not None:
            recalc = True if dq != self.matrix[0].dq else recalc
        if dang is not None:
            recalc = True if dang != self.matrix[0].dang else recalc

        self.calc_matrices("p_y_lab_pol", recalc, multiprocessing=multiprocessing or self.multiprocessing,
                           radial_range=radial_range,
                           angular_range=angular_range, dang=dang, dq=dq)

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
        if plot_result or save_fig:
            for i in range(len(img)):
                self._plot_single_image(img[i], x, y, clims, xlim, ylim, r"$|q|\ \mathrm{[\AA^{-1}]}$",
                                        r"$\chi$ [$\degree$]", 'auto', plot_result,
                                        save_fig, add_frame_number(path_to_save_fig, i))
        if return_result:
            return x, y, img

    def det2pol_gid(self, frame_num=None, interp_type="INTER_LINEAR", multiprocessing=None, return_result=False,
                    radial_range=None, angular_range=None, dang=None, dq=None,
                    plot_result=False, clims=None,
                    xlim=(None, None), ylim=(None, None),
                    save_fig=False, path_to_save_fig="img.png",
                    save_result=False,
                    path_to_save='result.h5',
                    h5_group=None,
                    overwrite_file=True,
                    overwrite_group=False,
                    exp_metadata=None,
                    smpl_metadata=None,
                    ):
        """
         Converts detector image to polar coordinates for GID geometry.

         Parameters
         ----------
         frame_num : int, list or None, optional
             Frame number to process. If None, defaults to the first or current frame.
         interp_type : str, optional
             Interpolation method used for remapping. Default is "INTER_LINEAR".
         multiprocessing : bool, optional
             Whether to use multiprocessing for processing. If None, uses default setting.
         return_result : bool, optional
             If True, returns the result: two axes and images.
         radial_range : tuple or None, optional
             Tuple specifying the min and max of q. If None, uses full range.
         angular_range : tuple or None, optional
             Tuple specifying the min and max of azimuthal angle. If None, uses full range.
         plot_result : bool, optional
             Whether to plot the resulting reciprocal space map. Default is False.
         clims : tuple, optional
             Color scale limits (vmin, vmax) for plotting. Default is (1e1, 4e4).
         xlim : tuple, optional
             X-axis limits for plot. Default is (None, None).
         ylim : tuple, optional
             Y-axis limits for plot. Default is (None, None).
         save_fig : bool, optional
             Whether to save the figure. Default is False.
         path_to_save_fig : str, optional
             Path to save the figure if save_fig is True. Default is "img.png".
         save_result : bool, optional
             Whether to save the resulting data to h5-file. Default is False.
         path_to_save : str, optional
             Path to save the result if save_result is True. Default is 'result.h5'.
         h5_group : str or None, optional
             HDF5 group under which to save data.
         recalc : bool, optional
             Whether to force recalculation even if coodinate map exists. Default is False.
         overwrite_file : bool, optional
             Whether to overwrite existing file. Default is True.
         metadata : dict or None, optional
             Additional metadata to store with result. Default is None.

         Returns
         -------
         q_gid_pol : array
             The q_gid_pol-axis values of the converted data (in 1/A).
         ang_gid_pol : array
             The ang_gid_pol-axis values of the converted data (in degrees).
         img_gid_pol : 2D-array or list of 2D-arrays
             The converted image img_gid_pol.

         """
        if self.batch_activated:
            res = self.Batch(path_to_save, "det2pol_gid", h5_group, exp_metadata, smpl_metadata, overwrite_file,
                             overwrite_group,
                             save_result, plot_result, return_result)
            self.batch_activated = True
            return res

        recalc = ((determine_recalc_key(angular_range, [self.matrix[0].ang_min, self.matrix[0].ang_max],
                                        self.matrix[0].ang_gid_pol, self.matrix[0].dang) \
                       if hasattr(self.matrix[0], "ang_gid_pol") else True) or
                  (determine_recalc_key(radial_range, [self.matrix[0].q_min, self.matrix[0].q_max],
                                        self.matrix[0].q_gid_pol, self.matrix[0].dq) \
                       if hasattr(self.matrix[0], "q_gid_pol") else True))
        if dq is not None:
            recalc = True if dq != self.matrix[0].dq else recalc
        if dang is not None:
            recalc = True if dang != self.matrix[0].dang else recalc

        self.calc_matrices("p_y_smpl_pol", recalc, multiprocessing=multiprocessing or self.multiprocessing,
                           radial_range=radial_range,
                           angular_range=angular_range,
                           dang=dang,
                           dq=dq)

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
        if plot_result or save_fig:
            for i in range(len(img)):
                self._plot_single_image(img[i], x, y, clims, xlim, ylim, r"$|q|\ \mathrm{[\AA^{-1}]}$",
                                        r"$\chi$ [$\degree$]", 'auto', plot_result,
                                        save_fig, add_frame_number(path_to_save_fig, i))
        if return_result:
            return x, y, img

    def det2pseudopol(self, frame_num=None, interp_type="INTER_LINEAR", multiprocessing=None, return_result=False,
                      q_azimuth_range=None, q_rad_range=None, dang=None, dq=None,
                      plot_result=False, clims=None,
                      xlim=(None, None), ylim=(None, None),
                      save_fig=False, path_to_save_fig="img.png",
                      save_result=False,
                      path_to_save='result.h5',
                      h5_group=None,
                      overwrite_file=True,
                      overwrite_group=False,
                      exp_metadata=None,
                      smpl_metadata=None,
                      ):
        """
        Converts detector image to pseudopolar coordinates for transmssion geometry.

        Parameters
        ----------
        frame_num : int, list or None, optional
            Frame number to process. If None, defaults to the first or current frame.
        interp_type : str, optional
            Interpolation method used for remapping. Default is "INTER_LINEAR".
        multiprocessing : bool, optional
            Whether to use multiprocessing for processing. If None, uses default setting.
        return_result : bool, optional
            If True, returns the result: two axes and images.
        plot_result : bool, optional
            Whether to plot the resulting reciprocal space map. Default is False.
        clims : tuple, optional
            Color scale limits (vmin, vmax) for plotting. Default is (1e1, 4e4).
        xlim : tuple, optional
            X-axis limits for plot. Default is (None, None).
        ylim : tuple, optional
            Y-axis limits for plot. Default is (None, None).
        save_fig : bool, optional
            Whether to save the figure. Default is False.
        path_to_save_fig : str, optional
            Path to save the figure if save_fig is True. Default is "img.png".
        save_result : bool, optional
            Whether to save the resulting data to h5-file. Default is False.
        path_to_save : str, optional
            Path to save the result if save_result is True. Default is 'result.h5'.
        h5_group : str or None, optional
            HDF5 group under which to save data.
        overwrite_file : bool, optional
            Whether to overwrite existing file. Default is True.
        metadata : dict or None, optional
            Additional metadata to store with result. Default is None.

        Returns
        -------
        q_rad : array
            The q_rad-axis values of the converted data (in 1/A).
        q_azimuth : array
            The q_azimuth-axis values of the converted data (in 1/A).
        img_pseudopol : 2D-array or list of 2D-arrays
            The converted image img_pseudopol.

        """

        if self.batch_activated:
            res = self.Batch(path_to_save, "det2pseudopol", h5_group, exp_metadata, smpl_metadata, overwrite_file,
                             overwrite_group,
                             save_result, plot_result, return_result)
            self.batch_activated = True
            return res

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

        self.calc_matrices("p_y_lab_pseudopol", recalc, multiprocessing=multiprocessing or self.multiprocessing,
                           q_rad_range=q_rad_range,
                           q_azimuth_range=q_azimuth_range, dang=dang, dq=dq)

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
        if plot_result or save_fig:
            for i in range(len(img)):
                self._plot_single_image(img[i], x, y, clims, xlim, ylim, r"$|q|\ \mathrm{[\AA^{-1}]}$",
                                        r"$q_{\phi}\ \mathrm{[\AA^{-1}]}$", 'auto', plot_result,
                                        save_fig, add_frame_number(path_to_save_fig, i))
        if return_result:
            return x, y, img

    def det2pseudopol_gid(self, frame_num=None, interp_type="INTER_LINEAR", multiprocessing=None, return_result=False,
                          q_rad_range=None, q_azimuth_range=None, dang=None, dq=None,
                          plot_result=False, clims=None,
                          xlim=(None, None), ylim=(None, None),
                          save_fig=False, path_to_save_fig="img.png",
                          save_result=False,
                          path_to_save='result.h5',
                          h5_group=None,
                          overwrite_file=True,
                          overwrite_group=False,
                          exp_metadata=None,
                          smpl_metadata=None,
                          ):
        """
        Converts detector image to pseudopolar coordinates for GID geometry.

        Parameters
        ----------
        frame_num : int, list or None, optional
            Frame number to process. If None, defaults to the first or current frame.
        interp_type : str, optional
            Interpolation method used for remapping. Default is "INTER_LINEAR".
        multiprocessing : bool, optional
            Whether to use multiprocessing for processing. If None, uses default setting.
        return_result : bool, optional
            If True, returns the result: two axes and images.
        plot_result : bool, optional
            Whether to plot the resulting reciprocal space map. Default is False.
        clims : tuple, optional
            Color scale limits (vmin, vmax) for plotting. Default is (1e1, 4e4).
        xlim : tuple, optional
            X-axis limits for plot. Default is (None, None).
        ylim : tuple, optional
            Y-axis limits for plot. Default is (None, None).
        save_fig : bool, optional
            Whether to save the figure. Default is False.
        path_to_save_fig : str, optional
            Path to save the figure if save_fig is True. Default is "img.png".
        save_result : bool, optional
            Whether to save the resulting data to h5-file. Default is False.
        path_to_save : str, optional
            Path to save the result if save_result is True. Default is 'result.h5'.
        h5_group : str or None, optional
            HDF5 group under which to save data.
        overwrite_file : bool, optional
            Whether to overwrite existing file. Default is True.
        metadata : dict or None, optional
            Additional metadata to store with result. Default is None.

        Returns
        -------
        q_gid_rad : array
            The q_gid_rad-axis values of the converted data (in 1/A).
        q_gid_azimuth : array
            The q_gid_azimuth-axis values of the converted data (in 1/A).
        img_gid_pseudopol : 2D-array or list of 2D-arrays
            The converted image img_gid_pseudopol.
        """

        if self.batch_activated:
            res = self.Batch(path_to_save, "det2pseudopol_gid", h5_group, exp_metadata, smpl_metadata, overwrite_file,
                             overwrite_group,
                             save_result, plot_result, return_result)
            self.batch_activated = True
            return res

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

        if dq is not None:
            recalc = True if dq != self.matrix[0].dq else recalc
        if dang is not None:
            recalc = True if dang != self.matrix[0].dang else recalc

        self.calc_matrices("p_y_smpl_pseudopol", recalc, multiprocessing=multiprocessing or self.multiprocessing,
                           q_gid_rad_range=q_rad_range,
                           q_gid_azimuth_range=q_azimuth_range, dang=dang, dq=dq)

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
        if plot_result or save_fig:
            for i in range(len(img)):
                self._plot_single_image(img[i], x, y, clims, xlim, ylim, r"$|q|\ \mathrm{[\AA^{-1}]}$",
                                        r"$q_{\phi}\ \mathrm{[\AA^{-1}]}$", 'auto', plot_result,
                                        save_fig, add_frame_number(path_to_save_fig, i))
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

    def _plot_profile(self, x_values, profiles, xlabel, shift, xlim, ylim, plot_result,
                      save_fig, path_to_save_fig):
        """
        Plots one or multiple radial/azmuthal or horizontal profiles with optional vertical shifting and formatting.

        Parameters
        ----------
        x_values : array-like
            The x-axis values (e.g., q, angle, etc.).
        profiles : array-like or list of arrays
            One or more profiles to be plotted.
        xlabel : str
            Label for the x-axis.
        shift : float
            Amount by which to vertically shift each profile (for clarity in stacked plots).
        xlim : tuple or None
            Limits for the x-axis as (min, max). Use None to auto-scale.
        ylim : tuple or None
            Limits for the y-axis as (min, max). Use None to auto-scale.
        plot_result : bool
            If True, displays the plot.
        save_fig : bool
            If True, saves the plot to a file.
        path_to_save_fig : str
            Path where the figure will be saved if `save_fig` is True.
        """

        fig, ax = plt.subplots()
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Intensity [arb. units]")
        ax.set_yscale('log')
        ax.tick_params(axis='both')
        if xlim: ax.set_xlim(xlim)
        if ylim: ax.set_ylim(ylim)
        fig.tight_layout(pad=3)
        cmap = colors.LinearSegmentedColormap.from_list("mycmap", ["royalblue", "mediumorchid", "orange"])
        norm = Normalize(vmin=0, vmax=len(profiles))
        if not plot_result:
            plt.close()
        for i, line in enumerate(profiles):
            ax.plot(x_values, line * 2 ** (i * shift), color=cmap(norm(i)))
        if save_fig:
            if path_to_save_fig is not None:
                fig.canvas.draw()
                plt.savefig(path_to_save_fig)
                print(f"Saved figure in {path_to_save_fig}")
            else:
                raise ValueError("path_to_save_fig is not defined.")
            if plot_result:
                plt.show()
            else:
                plt.close()
                del fig, ax
        if plot_result:
            plt.show()


    def radial_profile_gid(self, **kwargs):
        kwargs['key'] = "gid"
        return self.radial_profile(**kwargs)

    def radial_profile(self, key="transmission", frame_num=None, radial_range=None, angular_range=[0, 90], multiprocessing=None,
                       return_result=False, save_result=False, save_fig=False, path_to_save_fig='rad_cut.tiff',
                       plot_result=False, shift=1, xlim=None, ylim=None, dang=0.5, dq=None,
                       path_to_save='result.h5',
                       h5_group=None,
                       overwrite_file=True,
                       overwrite_group=False,
                       exp_metadata=None,
                       smpl_metadata=None, ):

        """
        Computes and optionally plots the radial profile from 2D scattering data for a given angular range.

        Parameters
        ----------
        key : str, optional
            Key indicating which geometry was used (default is "gid").
        frame_num : int, list or None, optional
            Frame number to analyze. If None, all data will be used.
        radial_range : list or tuple, optional
            Radial (q) range as [min, max] in Å⁻¹. If None, full range is used.
        angular_range : list, optional
            Angular range in degrees as [min, max] over which to integrate (default: [0, 90]) (in degrees).
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
        metadata : dict or None, optional
            Optional metadata to include when saving results.

        Returns
        -------
        q_abs_values : array
            The q_abs_values-axis values of the converted data (in 1/A).
        rad_cut : 1D-array or list of 1D-arrays
            Integrated image profile rad_cut.
        """

        if self.batch_activated:
            remap_func = "radial_profile_gid" if key == "gid" else "radial_profile"
            res = self.Batch(path_to_save, remap_func, h5_group, exp_metadata, smpl_metadata, overwrite_file,
                             overwrite_group,
                             save_result, plot_result, return_result)
            self.batch_activated = True
            return res

        q_abs_values, _, img_pol = self._get_polar_data(key, frame_num, radial_range, angular_range, dang, dq)
        img_pol = np.array(img_pol)

        img_pol = np.expand_dims(img_pol, axis=0) if img_pol.ndim == 2 else img_pol
        radial_profile = np.nanmean(img_pol, axis=1)
        if plot_result or save_fig:
            self._plot_profile(q_abs_values, radial_profile, r"$q_{abs}\ [\AA^{-1}]$", shift,
                               xlim, ylim, plot_result, save_fig, path_to_save_fig)
        name = "rad_cut_gid" if key == "gid" else "rad_cut"
        setattr(self, name, radial_profile)
        delattr(self, "img_gid_pol") if key == "gid" else delattr(self, "img_pol")

        if save_result:
            self.save_nxs(path_to_save=path_to_save,
                          h5_group=h5_group,
                          overwrite_file=overwrite_file,
                          overwrite_group=overwrite_group,
                          exp_metadata=exp_metadata,
                          smpl_metadata=smpl_metadata)

        if return_result:
            return (q_abs_values, radial_profile[0]) if radial_profile.shape[0] == 1 else (
                q_abs_values, radial_profile)


    def azim_profile_gid(self, **kwargs):
        kwargs['key'] = "gid"
        return self.azim_profile(**kwargs)


    def azim_profile(self, key="transmission", frame_num=None, radial_range=None, angular_range=[0, 90], multiprocessing=None,
                     return_result=False, save_result=False, save_fig=False, path_to_save_fig='azim_cut.tiff',
                     plot_result=False, shift=1, xlim=None, ylim=None,
                     path_to_save='result.h5', dang=0.5, dq=None,
                     h5_group=None,
                     overwrite_file=True,
                     overwrite_group=False,
                     exp_metadata=None,
                     smpl_metadata=None, ):
        """
        Computes and optionally plots the azmuthal profile from 2D scattering data for a given angular range.

        Parameters
        ----------
        key : str, optional
            Key indicating which geometry was used (default is "gid").
        frame_num : int, list or None, optional
            Frame number to analyze. If None, all data will be used.
        radial_range : list or tuple, optional
            Radial (q) range as [min, max] in Å⁻¹. If None, full range is used.
        angular_range : list, optional
            Angular range in degrees as [min, max] over which to integrate (default: [0, 90]) (in degrees).
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
        metadata : dict or None, optional
            Optional metadata to include when saving results.

        Returns
        -------
        phi_abs_values : array
            The phi_abs_values-axis values of the converted data (in degrees).
        azim_cut : 1D-array or list of 1D-arrays
            Integrated image profile azim_cut.
        """

        if self.batch_activated:
            remap_func = "azim_profile_gid" if key == "gid" else "azim_profile"
            res = self.Batch(path_to_save, remap_func, h5_group, exp_metadata, smpl_metadata, overwrite_file,
                             overwrite_group,
                             save_result, plot_result, return_result)
            self.batch_activated = True
            return res

        _, phi_abs_values, img_pol = self._get_polar_data(key, frame_num, radial_range, angular_range, dang, dq)
        img_pol = np.array(img_pol)
        img_pol = np.expand_dims(img_pol, axis=0) if img_pol.ndim == 2 else img_pol
        azim_profile = np.nanmean(img_pol, axis=2)
        if plot_result or save_fig:
            self._plot_profile(phi_abs_values, azim_profile, r"$\phi\ (\degree)$", shift, xlim,
                               ylim, plot_result, save_fig, path_to_save_fig)
        name = "azim_cut_gid" if key == "gid" else "azim_cut"
        setattr(self, name, azim_profile)
        delattr(self, "img_gid_pol") if key == "gid" else delattr(self, "img_pol")

        if save_result:
            self.save_nxs(path_to_save=path_to_save,
                          h5_group=h5_group,
                          overwrite_file=overwrite_file,
                          overwrite_group=overwrite_group,
                          exp_metadata=exp_metadata,
                          smpl_metadata=smpl_metadata)

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

    def horiz_profile_gid(self, frame_num=None, q_xy_range=[0, 4], q_z_range=[0, 0.2], dq=None, multiprocessing=None,
                      return_result=False, save_result=False, save_fig=False, path_to_save_fig='hor_cut.tiff',
                      plot_result=False, shift=1, xlim=None, ylim=None,
                      path_to_save='result.h5',
                      h5_group=None,
                      overwrite_file=True,
                      overwrite_group=False,
                      exp_metadata=None,
                      smpl_metadata=None):
        """
        Computes and optionally plots the azmuthal profile from 2D scattering data for a given angular range.

        Parameters
        ----------
        key : str, optional
            Key indicating which geometry was used (default is "gid").
        frame_num : int, list or None, optional
            Frame number to analyze. If None, all data will be used.
        q_xy_range : list or tuple, optional
            q_xy range as [min, max] in Å⁻¹. If None, full range is used.
        q_z_range : list or tuple, optional
            q_z range as [min, max] in Å⁻¹. If None, full range is used.
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
        metadata : dict or None, optional
            Optional metadata to include when saving results.

        Returns
        -------
        q_hor_values : array
            The q_hor_values-axis values of the converted data (in 1/A).
        horiz_cut : 1D-array or list of 1D-arrays
            Integrated image profile horiz_cut.

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
            self._plot_profile(q_hor_values, horiz_profile, xlabel, shift, xlim,
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

    def _plot_single_image(self, img, x, y, clims, xlim, ylim, x_label, y_label, aspect, plot_result, save_fig,
                           path_to_save_fig):

        fig_lenth = 6.4
        if aspect == 'auto':
            fig_lenth -= 0.4

        fig, ax = plt.subplots(figsize=(fig_lenth, 4.8))
        plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
        if clims is None:
            clims = [np.nanmin(img[img>0]), np.nanmax(img)]
        img[img < 0] = clims[0]
        log_img = np.log(img / clims[1])
        log_img = np.nan_to_num(log_img, nan=np.nan, posinf=np.log(clims[0] / clims[1]),
                                neginf=np.log(clims[0] / clims[1]))

        p = ax.imshow(log_img,
                      vmin=np.log(clims[0] / clims[1]), vmax=np.log(clims[1] / clims[1]),
                      extent=[x.min(), x.max(), y.min(), y.max()],
                      aspect=aspect,
                      origin='lower')

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=False, prune=None, nbins=4))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=False, prune=None, nbins=4))
        ax.tick_params(axis='both')
        cb = fig.colorbar(mappable=p, ax=ax)
        cb.set_label(label='Intensity [arb. units]')
        cb.ax.yaxis.labelpad = 1

        cb.set_ticks([np.log(clims[0] / clims[1]), np.log(clims[1] / clims[1])])
        cb.set_ticklabels([change_clim_format(str(clims[0])), change_clim_format(str(clims[1]))])

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        if save_fig:
            if path_to_save_fig is not None:
                if (path_to_save_fig.endswith('.svg') or path_to_save_fig.endswith('.pdf')
                        or path_to_save_fig.endswith('.eps') or path_to_save_fig.endswith('.pgf')):
                    if aspect == 'equal':
                        plt.axis('square')
                    else:
                        ax.set_aspect('auto', 'box')

                    plt.savefig(path_to_save_fig)
                else:
                    plt.savefig(path_to_save_fig, bbox_inches='tight')
                print(f"Saved figure in {path_to_save_fig}")
            else:
                raise ValueError("path_to_save_fig is not defined.")
            if not plot_result:
                plt.close()
                del fig, ax

        if plot_result:
            plt.show()

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
                        min_int=None, clims=None, vmin=0, vmax=1, linewidth=1, radius=0.1, cmap=cm.Blues,
                        text_color='black', max_shift=1, save_result=False, path_to_save='simul_result.png'):
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
            max_shift (float): Maximum positional shift allowed in the simulation.
            save_result (bool): If True, saves the figure image.
            path_to_save (str): File path to save the simulation figure.

        Returns
        -------
        (q_xy, q_z) : (array, array)
           q_xy, q_z positions of the simulated data (in 1/A).
        intensity : array
           The intensity values of the simulated data.
        mi : array
           Miller indices of the simulated data.

        """


        q_xy_max = self.matrix[0].q_xy_range[1]
        q_z_max = self.matrix[0].q_z_range[1]
        radius/=np.sqrt(q_xy_max**2+q_z_max**2)/4.37
        ai = self.matrix[0].ai if len(self.matrix) == 1 else self.matrix[frame_num].ai
        try:
            simul_params = ExpParameters(q_xy_max=q_xy_max, q_z_max=q_z_max, en=12398 / self.params.wavelength, ai=ai)
        except:
            raise ValueError("pygidsim package is not installed.")

        path_to_cif = [path_to_cif] if not isinstance(path_to_cif, list) else path_to_cif
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
        img = img[0]
        if clims is None:
            clims = [np.nanmin(img[img>0]), np.nanmax(img)]
        img[img < 0] = clims[0]
        log_img = np.log(img / clims[1])
        log_img = np.nan_to_num(log_img, nan=np.nan, posinf=np.log(clims[0] / clims[1]),
                                neginf=np.log(clims[0] / clims[1]))

        if plot_result:

            fig, ax = plt.subplots()
            p = ax.imshow(log_img,
                          vmin=np.log(clims[0] / clims[1]), vmax=np.log(clims[1] / clims[1]),
                          extent=[np.nanmin(q_xy), np.nanmax(q_xy), np.nanmin(q_z), np.nanmax(q_z)],
                          aspect='equal',
                          origin='lower')
            ax.set_xlabel(r'$q_{xy}$ [$\mathrm{\AA}^{-1}$]')
            ax.set_ylabel(r'$q_{z}$ [$\mathrm{\AA}^{-1}$]')
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=False, prune=None, nbins=4))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=False, prune=None, nbins=4))
            ax.tick_params(axis='both')

            for i, dataset in enumerate(simulated_data):
                cmap_i = cmap if not isinstance(cmap, list) else cmap[i]
                norm = plot_single_simul_data(dataset, ax, cmap_i, vmin, vmax, linewidth, radius, text_color, plot_mi,
                                              max_shift)

            cb = fig.colorbar(mappable=p, ax=ax)
            cb.set_label(label='Intensity [arb. units]')
            cb.ax.yaxis.labelpad = 1
            cb.set_ticks([np.log(clims[0] / clims[1]), np.log(clims[1] / clims[1])])
            cb.set_ticklabels([change_clim_format(str(clims[0])), change_clim_format(str(clims[1]))])
            print(f"frame_num = {frame_num} was plotted")

            if save_result:

                if (path_to_save.endswith('.svg') or path_to_save.endswith('.pdf')
                        or path_to_save.endswith('.eps') or path_to_save.endswith('.pgf')):
                    plt.axis('square')
                    plt.savefig(path_to_save)
                else:
                    plt.savefig(path_to_save, bbox_inches='tight')
                print(f"Saved figure in {path_to_save}")
            plt.show()
        if return_result:
            simulated_data = sort_simul_data(simulated_data)
            if len(simulated_data):
                return simulated_data[0]
            else:
                return simulated_data


def sort_simul_data(simulated_data):
    for i in range(len(simulated_data)):
        q, value, mi = simulated_data[i]

        q = np.array(q)
        value = np.array(value)
        mi = np.array(mi)

        assert q.shape[-1] == len(value) == len(mi), "Mismatch in array lengths"

        if q.ndim == 2 and q.shape[0] == 2:
            # q is shape (2, N) → compute |q| for each column
            q_abs = np.linalg.norm(q, axis=0)  # axis=0: по столбцам
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



def plot_single_simul_data(dataset, ax, cmap, vmin, vmax, linewidth, radius, text_color, plot_mi, max_shift):
    q, intensity, mi = dataset
    vmin = max(vmin, 1e-10)
    if vmax is not None:
        vmax = np.nanmax(intensity)
    norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    q_xy_max = xlim[1]
    q_z_max = ylim[1]

    if q.ndim == 2:
        existing_texts = None
        texts = []
        for x, y, inten, text in zip(q[0], q[1], intensity, mi):
            color = cmap(norm(inten))
            circle = plt.Circle((x, y), radius, edgecolor=color, facecolor='none', linewidth=linewidth)
            ax.add_patch(circle)
            if plot_mi:
                txt = ax.text(x, y, str(text), fontsize=8, color=text_color,
                              weight='bold', ha='center', va='bottom')
                texts.append(txt)
        if plot_mi and texts:
            adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    else:
        size = len(intensity)
        num = 1
        texts = []
        for rad, i, text in zip(q, intensity, mi):
            color = cmap(norm(i))
            circle = plt.Circle((0, 0), rad, color=color, fill=False, linestyle="dashed", linewidth=linewidth)
            ax.add_patch(circle)
            if plot_mi:
                angle_to_plot = np.pi / 2 / size * num - 0.1
                pos_xy = rad * np.sin(angle_to_plot)
                pos_z = rad * np.cos(angle_to_plot)

                if pos_xy > q_xy_max or pos_z > q_z_max:
                    angle_to_plot = np.arctan(q_z_max / q_xy_max) - 0.1
                    pos_xy = rad * np.sin(angle_to_plot) - q_xy_max * 0.2
                    pos_z = rad * np.cos(angle_to_plot)

                txt = ax.text(pos_xy, pos_z, str(text), fontsize=8, color=text_color, weight='bold')
                texts.append(txt)
                num += 1

            if plot_mi and texts:
                adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=1))
    return norm


def determine_recalc_key(current_range, global_range, array, step):
    recalc = (determine_recalc_key_index(current_range, global_range, array, step, np.nanargmin(array), 0) or
              determine_recalc_key_index(current_range, global_range, array, step, np.nanargmax(array), -1))
    return recalc


def determine_recalc_key_index(current_range, global_range, array, step, arr_index, index):
    if current_range is None:
        recalc = False if np.isclose(global_range[index],
                                     array[arr_index], atol=step) else True
    else:
        recalc = False if np.isclose(current_range[index],
                                     array[arr_index], atol=step) else True
    return recalc


def simul_single_data(path_to_cif, orientation, simul_params, min_int):
    print(f"path_to_cif = {path_to_cif}, orientation = {orientation}, min_int = {min_int}")
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
        return remapped_image
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
    if mask is not None:
        img = img.astype(float)
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
    if filename is None:
        return
    file_root, file_ext = os.path.splitext(filename)
    frame_str = str(frame_num).zfill(4)
    return f"{file_root}_{frame_str}{file_ext}"


def change_clim_format(s):
    f = f"{float(s):.0e}"
    base, exp = f.split('e')
    exp = exp.lstrip('+0') or '0'
    return f"{base}e{exp}"


def select_best_array(arrays):
    def sort_key(arr):
        return (
            np.sum(arr ** 2),
            *[(abs(x), -x) for x in arr]
        )

    return min(arrays, key=sort_key)


def make_numbered_filename(base_filename, frame_num):
    name, ext = os.path.splitext(base_filename)
    return f"{name}_{frame_num:04d}{ext}"