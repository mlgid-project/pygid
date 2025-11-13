from adjustText import adjust_text
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import logging
import os
from pathlib import Path
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
from matplotlib import cm

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s"
)

def get_plot_context(rc_params):
    return plt.rc_context(rc=rc_params)


def get_plot_params(font_size=14, axes_titlesize=14, axes_labelsize=18, grid=False, grid_color='gray',
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

    Returns a Matplotlib rc_context that can be used as a context manager
    to apply plot settings locally.
    """

    rc_params = {
        # Font
        'font.size': font_size,
        'axes.titlesize': axes_titlesize,
        'axes.labelsize': axes_labelsize,
        'xtick.labelsize': xtick_labelsize,
        'ytick.labelsize': ytick_labelsize,
        'legend.fontsize': legend_fontsize,
        'figure.titlesize': figure_titlesize,
        # Axes and grid
        'axes.grid': grid,
        'axes.linewidth': axes_linewidth,
        'grid.color': grid_color,
        'grid.linestyle': grid_linestyle,
        'grid.linewidth': grid_linewidth,
        'legend.loc': legend_loc,
        'legend.frameon': legend_frameon,
        'legend.borderpad': legend_borderpad,
        'legend.borderaxespad': legend_borderaxespad,
        # Figure
        'figure.figsize': figsize,
        'savefig.dpi': savefig_dpi,
        'savefig.transparent': savefig_transparent,
        'savefig.bbox': savefig_bbox_inches,
        'savefig.pad_inches': savefig_pad_inches,
        # Lines
        'lines.linewidth': line_linewidth,
        'lines.color': line_color,
        'lines.linestyle': line_linestyle,
        'lines.marker': line_marker if line_marker is not None else '',
        # Scatter
        'scatter.marker': scatter_marker,
        'scatter.edgecolors': scatter_edgecolors,
        # Colormap
        'image.cmap': cmap
    }
    return rc_params


def plot_img_raw(img_raw, x, y, plot_context, return_result=False, frame_num=None, plot_result=True,
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

        if img_raw is None:
            raise AttributeError("img_raw is not loaded")
        if not isinstance(img_raw, np.ndarray):
            img_raw = np.array(img_raw)

        if frame_num is None and img_raw.shape[0] != 1:
            frame_num = np.arange(1, img_raw.shape[0],1)
        if isinstance(frame_num, list) or isinstance(frame_num, np.ndarray):
            img_list = []
            for num in frame_num:
                x, y, img = plot_img_raw(return_result=True, frame_num=num, plot_result=plot_result,
                             clims=clims, xlim=xlim, ylim=ylim, save_fig=save_fig,
                             path_to_save_fig=make_numbered_filename(path_to_save_fig, num))
                img_list.append(img)
            if return_result:
                return x, y, img_list
            return

        if frame_num is None:
            frame_num = 0
        img = np.array(img_raw[frame_num])

        if clims is None:
            clims = [np.nanmin(img[img>0]), np.nanmax(img)]

        if img_raw is None:
            raise ValueError("img_raw is not loaded")


        def fill_limits(lim, data):
            return [np.nanmin(data) if lim[0] is None else lim[0],
                    np.nanmax(data) if lim[1] is None else lim[1]]

        xlim = fill_limits(xlim, x)
        ylim = fill_limits(ylim, y)

        with plot_context:
            fig = plt.figure()
            margin = 0.1
            ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])

            img[img < 0] = clims[0]

            p = ax.imshow(np.clip(img, clims[0], clims[1]),
                          norm=LogNorm(vmin=clims[0], vmax=clims[1]),
                          extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                          aspect='equal',
                          origin='lower')

            ax.set_xlabel(r'$y$ [px]')
            ax.set_ylabel(r'$z$ [px]')
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=False, prune=None, nbins=4))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=False, prune=None, nbins=4))
            ax.tick_params(axis='both')

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = fig.colorbar(p, cax=cax)
            cb.set_label('Intensity [arb. units]')
            cb.ax.yaxis.labelpad = 5

            cb.ax.yaxis.set_minor_locator(ticker.NullLocator())
            cb.set_ticks([clims[0], clims[1]])
            cb.set_ticklabels([change_clim_format(str(clims[0])),
                               change_clim_format(str(clims[1]))])

        if save_fig:
            if path_to_save_fig is not None:
                plt.savefig(path_to_save_fig)
                logging.info(f"Saved figure in {Path(path_to_save_fig).resolve()}")

            else:
                raise ValueError("path_to_save_fig is not defined.")
            if not plot_result:
                plt.close()
                del fig, ax

        if plot_result:
            plt.show()
        else:
            plt.close()



        if return_result:
            return x, y, img


def _plot_single_image(
            plot_context,
            img,
            x,
            y,
            clims,
            xlim,
            ylim,
            x_label,
            y_label,
            aspect,
            plot_result,
            save_fig,
            path_to_save_fig
    ):
        """
        Plots a single 2D image (e.g., reciprocal-space map) in logarithmic scale.

        Parameters
        ----------
        img : ndarray
            2D array representing image intensity values.
        x : ndarray
            X-axis coordinates (e.g., q_xy or 2θ values).
        y : ndarray
            Y-axis coordinates (e.g., q_z or χ values).
        clims : list or None
            Intensity limits for color scaling as [vmin, vmax].
            If None, computed automatically from positive pixel values.
        xlim : tuple or None
            Limits for X-axis. Default is None (auto).
        ylim : tuple or None
            Limits for Y-axis. Default is None (auto).
        x_label : str
            Label for the X-axis.
        y_label : str
            Label for the Y-axis.
        aspect : {'auto', 'equal'}
            Aspect ratio for the image display.
        plot_result : bool
            If True, displays the plot interactively.
        save_fig : bool
            If True, saves the figure to file.
        path_to_save_fig : str
            Path to save the image if `save_fig` is True.

        Notes
        -----
        The image is plotted in logarithmic scale using normalized intensity values.
        Negative and NaN pixels are set to the minimum displayable intensity.
        """
        if clims is None:
            clims = [np.nanmin(img[img > 0]), np.nanmax(img)]

        with plot_context:
            # fig, ax = plt.subplots()
            fig = plt.figure()
            margin = 0.1
            ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])
            # plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9, top=0.9, #wspace=0.4, hspace=0.4
            #                     )

            p = ax.imshow(np.clip(img, clims[0], clims[1]),
                          norm=LogNorm(vmin=clims[0], vmax=clims[1]),
                          extent=[x.min(), x.max(), y.min(), y.max()],
                          aspect=aspect,
                          origin='lower')

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=False, prune=None, nbins=4))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=False, prune=None, nbins=4))
            ax.tick_params(axis='both')

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = fig.colorbar(p, cax=cax)
            cb.set_label('Intensity [arb. units]')
            cb.ax.yaxis.labelpad = 5

            cb.ax.yaxis.set_minor_locator(ticker.NullLocator())
            cb.set_ticks([clims[0], clims[1]])
            cb.set_ticklabels([change_clim_format(str(clims[0])),
                               change_clim_format(str(clims[1]))])

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        if save_fig:
            if path_to_save_fig is not None:
                # if (path_to_save_fig.endswith('.svg') or path_to_save_fig.endswith('.pdf')
                #         or path_to_save_fig.endswith('.eps') or path_to_save_fig.endswith('.pgf')):
                #     if aspect == 'equal':
                #         plt.axis('square')
                #     else:
                #         ax.set_aspect('auto', 'box')
                #
                #     plt.savefig(path_to_save_fig)
                # else:
                #     plt.savefig(path_to_save_fig, bbox_inches='tight')
                plt.savefig(path_to_save_fig)
                logging.info(f"Saved figure in {Path(path_to_save_fig).resolve()}")
            else:
                raise ValueError("path_to_save_fig is not defined.")
            if not plot_result:
                plt.close()
                del fig, ax

        if plot_result:
            plt.show()

def plot_simul_data(plot_context, img, q_xy, q_z, clims, simulated_data, cmap, save_result, path_to_save,
                    vmin, vmax, linewidth, radius, text_color, plot_mi):
    if cmap is not None:
        cmap = [cmap] if not isinstance(cmap, list) else cmap
        if len(cmap) != len(simulated_data):
            raise ValueError("cmap and path_to_cif lists must have same length.")
    else:
        cmap = ['Blues', 'Greens', 'Oranges', 'Purples', 'Greys', 'Reds', 'YlOrBr', 'YlGnBu', 'PuBuGn',
                'Spectral']
    n = len(simulated_data)
    cmap = [cm.get_cmap(cmap[i % len(cmap)]) for i in range(n)]


    if clims is None:
        clims = [np.nanmin(img[img > 0]), np.nanmax(img)]
    with plot_context:
        fig = plt.figure()
        margin = 0.1
        ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])
        p = ax.imshow(np.clip(img, clims[0], clims[1]),
                      norm=LogNorm(vmin=clims[0], vmax=clims[1]),
                      extent=[np.nanmin(q_xy), np.nanmax(q_xy), np.nanmin(q_z), np.nanmax(q_z)],
                      aspect='equal',
                      origin='lower')

        ax.set_xlabel(r'$q_{xy}$ [$\mathrm{\AA}^{-1}$]')
        ax.set_ylabel(r'$q_{z}$ [$\mathrm{\AA}^{-1}$]')
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=False, prune=None, nbins=4))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=False, prune=None, nbins=4))
        ax.tick_params(axis='both')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        for i, dataset in enumerate(simulated_data):
            cmap_i = cmap[i]
            norm = add_single_simul_data(dataset, ax, cmap_i, vmin, vmax,
                                          linewidth, radius, text_color, plot_mi,
                                          )

        cb = fig.colorbar(p, cax=cax)
        cb.set_label('Intensity [arb. units]')
        cb.ax.yaxis.labelpad = 5

        cb.ax.yaxis.set_minor_locator(ticker.NullLocator())
        cb.set_ticks([clims[0], clims[1]])
        cb.set_ticklabels([change_clim_format(str(clims[0])),
                           change_clim_format(str(clims[1]))])

    if save_result:
        # if (path_to_save.endswith('.svg') or path_to_save.endswith('.pdf')
        #         or path_to_save.endswith('.eps') or path_to_save.endswith('.pgf')):
        #     plt.axis('square')
        #     plt.savefig(path_to_save)
        # else:
        #     plt.savefig(path_to_save, bbox_inches='tight')
        plt.savefig(path_to_save)
        logging.info(f"Saved figure in {Path(path_to_save).resolve()}")
    plt.show()

def add_single_simul_data(
        dataset,
        ax,
        cmap,
        vmin,
        vmax,
        linewidth,
        radius,
        text_color,
        plot_mi,
):
    """
        Plots simulated scattering data (either q-2D or radial) on a given matplotlib axis.

        Parameters
        ----------
        dataset : tuple
            A tuple of (q, intensity, mi):
            - `q` : array-like
                2D array with shape (2, N) for (q_x, q_z) coordinates or 1D array with shape (N,)
                for radial values.
            - `intensity` : array-like
                Corresponding intensity values.
            - `mi` : array-like
                Metadata (e.g., model index, fit ID).
        ax : matplotlib.axes.Axes
            Axis object on which to plot the data.
        cmap : matplotlib.colors.Colormap
            Colormap used for mapping intensities to colors.
        vmin : float
            Minimum intensity value for logarithmic normalization.
        vmax : float or None
            Maximum intensity value for normalization. If None, it is set automatically.
        linewidth : float
            Line width for circle outlines.
        radius : float
            Circle radius (used for q-2D data visualization).
        text_color : str
            Color of the text labels (metadata).
        plot_mi : bool
            Whether to annotate the plot with metadata labels (`mi` values).

        Returns
        -------
        norm : matplotlib.colors.LogNorm
            Logarithmic normalization object used for color mapping.

        Notes
        -----
        - For 2D q data, the function draws colored circles centered at (q_x, q_z).
        - For 1D q data, it draws concentric dashed rings (e.g., for isotropic scattering).
        - If `plot_mi=True`, metadata labels are drawn and adjusted to avoid overlap.
        """
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



def _plot_profile(plot_context, x_values, profiles, xlabel, shift, xlim, ylim, plot_result,
                      save_fig, path_to_save_fig):
        """
        Plots one or multiple radial, azimuthal or horizontal profiles with optional vertical shifting and formatting.

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
        with plot_context:
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
                logging.info(f"Saved figure in {Path(path_to_save_fig).resolve()}")
            else:
                raise ValueError("path_to_save_fig is not defined.")
            if plot_result:
                plt.show()
            else:
                plt.close()
                del fig, ax
        if plot_result:
            plt.show()


def make_numbered_filename(base_filename, frame_num):
    """
        Generates a filename with a zero-padded frame number appended.

        Parameters
        ----------
        base_filename : str
            Original filename.
        frame_num : int
            Frame number to append.

        Returns
        -------
        str
            Filename with frame number appended, e.g. 'base_0001.ext'.
        """
    name, ext = os.path.splitext(base_filename)
    return f"{name}_{frame_num:04d}{ext}"

def change_clim_format(s):
    """
        Converts a numeric string to scientific notation with simplified exponent format.

        Parameters
        ----------
        s : str or float
            Input number.

        Returns
        -------
        str
            Number in format like '1e3', '2.5e-2'.
    """
    f = f"{float(s):.0e}"
    base, exp = f.split('e')
    exp = exp.lstrip('+0') or '0'
    return f"{base}e{exp}"