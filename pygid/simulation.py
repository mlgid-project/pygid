from pygidsim.experiment import ExpParameters
from pygidsim.giwaxs_sim import GIWAXSFromCif
from pygidsim.giwaxs_sim import GIWAXS, Crystal

from .visualization import plot_simul_data_old, get_plot_context, plot_simul_data
import numpy as np
import sys
import os
import logging

logger = logging.getLogger(__name__)

def sort_simul_data(crystal_list):
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

    Notes
    -----
    - If `q` is 2D with shape (2, N), sorting is performed by the magnitude |q|.
    - The sorting is applied in-place to the input list elements.
    """
    result = []
    for i in range(len(crystal_list)):
        crystal = crystal_list[i]
        q, value, mi = crystal['q'], crystal['intensity'], crystal['mi']
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
        result.append((q_sorted, value_sorted, mi_sorted))
    return result


def simul_single_data(crystal, simul_params, move_fromMW):
    """
    Simulate GIWAXS scattering data for a given crystal description and
    post-process the results.

    The function supports two crystal definitions:
    (i) via a CIF file (`path_to_cif`) or
    (ii) via lattice parameters (`lat_par`).
    It computes scattering vectors, intensities, and corresponding Miller indices,
    followed by normalization and optional intensity-based filtering.

    Parameters
    ----------
    crystal : dict
        Dictionary describing the crystal. Must contain either:
        - 'path_to_cif' : str, path to a CIF file, or
        - 'lat_par' : array-like, lattice parameters.

        Optional keys:
        - 'orientation' : array-like or str, orientation matrix (3x3) or
          a string (e.g., "random"). Defaults to "random".
        - 'min_int' : float, minimum normalized intensity threshold for filtering.

    simul_params : dict
        Dictionary of simulation parameters passed to the underlying
        simulation backend.

    move_fromMW : bool
        If True, peak positions are shifted from the missing wedge.

    Returns
    -------
    None
        Results are stored directly in the input `crystal` dictionary.

    Side Effects
    ------------
    The input `crystal` dictionary is modified in-place with the following keys:

    q : ndarray
        Scattering vectors:
        - Shape (2, N) for oriented simulations (q_xy, q_z),
        - Shape (N,) for random/powder-like simulations (|q|).

    intensity : ndarray
        Normalized scattering intensities (scaled by maximum value).

    mi : ndarray
        Processed Miller indices corresponding to each scattering point.

    Notes
    -----
    - The simulation backend is selected automatically:
      * CIF-based (`simul_single_data_cif`) if 'path_to_cif' is provided.
      * Lattice-based (`simul_single_data_lat`) if 'lat_par' is provided.
    - Intensities are always normalized to their maximum value.
    - If 'min_int' is specified, reflections below the threshold are removed
      and the remaining data are sorted by intensity.
    - If `orientation` is given as a string (e.g., "random"), the output is
      reduced to 1D scattering vector magnitudes.
    - Miller indices with multiple candidates are reduced using
      `select_best_array`.
    """
    if crystal.get("path_to_cif"):
        logging.info(
            f"Simulating GIWAXS data from CIF: {crystal.get('path_to_cif')}, "
            f"orientation: {crystal.get('orientation', 'random')}"
        )
    else:
        logging.info(
            f"Simulating GIWAXS data from Crystal, "
            f"orientation: {crystal.get('orientation', 'random')}"
        )

    crystal['orientation'] = crystal.get('orientation', "random")
    crystal['orientation'] = np.array(crystal['orientation']) if isinstance(crystal['orientation'], list) else crystal['orientation']

    if crystal.get("path_to_cif") is not None:
        el = simul_single_data_cif(crystal, simul_params)
    elif crystal.get("lat_par") is not None:
        el = simul_single_data_lat(crystal, simul_params)
    else:
        raise ValueError("path_to_cif or lat_par must be provided for crystals")
    q, intensity, mi = el.giwaxs_sim(crystal.get('orientation', "random"), return_mi=True, move_fromMW=move_fromMW)
    mi = np.array([x[0] if len(x) == 1 else select_best_array(x) for x in mi])
    intensity /= np.max(intensity)

    min_int = crystal.get("min_int")
    orientation = crystal.get("orientation")
    if min_int is not None:
        index = ~(intensity < min_int)
        mi = mi[index]
        intensity = intensity[index]
        sort_index = np.argsort(intensity)

        mi = mi[sort_index]
        intensity = intensity[sort_index]
        q = np.stack((q[0][index], q[1][index]), axis=0)
        q = q[:, sort_index]
    if isinstance(orientation, str):
        q = np.sqrt(q[0]**2 + q[1]**2)
    crystal['q'] = q
    crystal['intensity'] = intensity
    crystal['mi'] = mi


def simul_single_data_cif(crystal, simul_params):
    cif_path = crystal.get("path_to_cif")
    if not cif_path or not os.path.isfile(cif_path):
        raise FileNotFoundError(f"Invalid CIF path: {cif_path}")
    return GIWAXSFromCif(crystal.get('path_to_cif'), simul_params).giwaxs

def simul_single_data_lat(crystal, simul_params):
    spgr = crystal.get('spgr', 1)
    lat_par = np.array(crystal.get('lat_par'), dtype=np.float32)
    atoms = np.array(crystal.get('atoms'))
    atom_positions = np.array(crystal.get('atom_positions'))
    occupancy =np.array(crystal.get('occupancy'))
    cr = Crystal(lat_par, spgr, atoms, atom_positions, occupancy)
    return GIWAXS(cr, simul_params)

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


class SuppressPrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def make_simulation_new(conversion, frame_num=0, crystal=None, plot_result=True,
            return_result=False, move_fromMW=True,
            save_fig=False, path_to_save_fig='simul_result.png',
            clims = None, xlim=(None, None), ylim=(None, None),
            return_fig = False):
    """
        Perform GIWAXS simulation using conversion data and plot or return the results.

        This function generates simulated scattering data for one or multiple crystal
        definitions using experimental parameters.
        The simulation is performed via `simul_single_data`, and results can be visualized
        together with experimental data or returned for further analysis.

        Parameters
        ----------
        conversion : pygid.Conversion
            Conversion object containing reciprocal space mapping and experimental
            parameters.

        frame_num : int, optional
            Frame index used to extract experimental geometry and data.

        crystal : dict or list of dict
            Crystal definition(s). Each dictionary must be compatible with
            `simul_single_data` (e.g., contain 'path_to_cif' or 'lat_par').
            A single dictionary is automatically converted into a list.

        plot_result : bool, optional
            If True, plot simulated data overlaid with experimental intensity.

        return_result : bool, optional
            If True, return processed simulation results.

        move_fromMW : bool, optional
            If True, peak positions are corrected for the missing wedge effect.

        save_fig : bool, optional
            If True, save the generated plot to file.

        path_to_save_fig : str, optional
            File path for saving the figure.

        clims : tuple, optional
            Color limits for experimental image visualization.

        xlim, ylim : tuple, optional
            Axis limits for the plot in reciprocal space coordinates.

        return_fig: bool, optional
            If True, return the matplotlib figure and axes objects.

        Returns
        -------
        list, dict, tuple, or None
            The returned value depends on the selected options:

            - If `return_result=True`, returns the simulated dataset(s):
                - A list of datasets, one for each crystal, if multiple crystals
                  are provided.
                - A single dataset if only one crystal is provided.

            - If `return_fig=True`, returns a tuple ``(fig, ax)`` containing
              the generated matplotlib figure and axes.

            - If both `return_result=True` and `return_fig=True`, returns the
              simulation result together with the figure and axes.

            - If both `return_result=False` and `return_fig=False`, returns
              ``None``.

            Each simulated dataset is generated using `sort_simul_data`.
        """
    q_xy_min, q_xy_max, q_z_min, q_z_max = _get_simul_ranges(conversion.matrix[0], xlim, ylim)
    need_update = True

    if hasattr(conversion, "simul_params"):
        sp = conversion.simul_params
        need_update = (
                sp.q_xy_range != (q_xy_min, q_xy_max)
                or sp.q_z_range != (q_z_min, q_z_max)
        )

    if need_update:
        ai = (
            conversion.matrix[0].ai
            if len(conversion.matrix) == 1
            else conversion.matrix[frame_num].ai
        )

        simul_params = ExpParameters(
            q_xy_range=(q_xy_min, q_xy_max),
            q_z_range=(q_z_min, q_z_max),
            en=12398 / conversion.params.wavelength,
            ai=ai,
        )
        conversion.simul_params = simul_params
    else:
        simul_params = conversion.simul_params

    if crystal is None:
        raise ValueError("No structures were set in crystal parameter")
    if isinstance(crystal, dict):
        crystal = [crystal]
    if not isinstance(crystal, list):
        raise TypeError("crystal must be a list or dictionary")

    with SuppressPrint():
        for i in range(len(crystal)):
            simul_single_data(crystal[i], simul_params, move_fromMW)

    if hasattr(conversion, "converted_frame_num") and hasattr(conversion,
                                                        "img_gid_q") and frame_num in conversion.converted_frame_num:
        index = conversion.converted_frame_num.index(frame_num)
        q_xy, q_z, img = conversion.matrix[0].q_xy, conversion.matrix[0].q_z, [conversion.img_gid_q[index]]
        logging.info(f'Use already converted image with frame num {frame_num}')
    else:
        q_xy, q_z, img = conversion._get_q_data(frame_num)

    if plot_result or save_fig or return_fig:
        fig, ax = plot_simul_data(get_plot_context(type(conversion).plot_params), img[0], q_xy, q_z,
         crystal, clims, save_fig, path_to_save_fig, xlim, ylim, plot_result, save_fig)
        if return_fig:
            return fig, ax

    if return_result:
        simulated_data = sort_simul_data(crystal)
        if len(simulated_data) == 1:
            return simulated_data[0]
        else:
            return simulated_data

def _get_simul_ranges(matrix, xlim, ylim):
    "    Determine the q_xy and q_z ranges for simulation based on conversion matrix and optional limits."
    if xlim != (None, None):
        q_xy_min, q_xy_max = xlim[0], xlim[1]
    else:
        try:
            q_xy_min = matrix.q_xy_range[0]
            q_xy_max = matrix.q_xy_range[1]
        except:
            q_xy_min = matrix.q_xy[0]
            q_xy_max = matrix.q_xy[-1]

    if ylim != (None, None):
        q_z_min, q_z_max = ylim[0], ylim[1]
    else:
        try:
            q_z_min = matrix.q_z_range[0]
            q_z_max = matrix.q_z_range[1]
        except:
            q_z_min = matrix.q_z[0]
            q_z_max = matrix.q_z[-1]
    return q_xy_min, q_xy_max, q_z_min, q_z_max



def make_simulation_old(conversion, frame_num=0, path_to_cif=None, orientation=None,
                        plot_result=True, plot_mi=False, return_result=False, move_fromMW=False,
                        min_int=None, clims=None, vmin=0, vmax=1, linewidth=1, radius=0.1, cmap=None,
                        text_color='black', save_fig=False, path_to_save_fig='simul_result.png',
                        xlim=(None, None), ylim=(None, None)):
    """
    Simulates and visualizes diffraction pattern for the given crystallographic data.

    Parameters:
        frame_num (int): Image frame number to visualize.
        path_to_cif (str or List[str]): Path to a CIF file(s) containing the crystal structure.
        orientation (list): Crystal orientation. None the for poweder pattern.
        plot_result (bool): Whether to plot the result of simulation and experimental data.
        move_fromMW (bool): Whether to move peaks from the missing wedge
        plot_mi (bool): Whether to plot the Miller indices.
        return_result (bool): Whether to return the result of simulation.
        min_int (float or None or List[float]): Minimum intensity threshold(s) for display
        clims (list): Intensity range for the color scale of experimental data
        vmin (float): Normalization limits for the color scale of simulated data
        vmax (float): Normalization limits for the color scale of simulated data
        linewidth (float): Simulated peaks line thickness for visualization
        radius (float): Simulated peaks radius for visualization
        cmap (str or List[str]): Colormap(s) used in the visualization.
        text_color (str): Color of any text annotations.
        save_fig (bool): If True, saves the figure image.
        path_to_save_fig (str): File path to save the simulation figure.

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

    If `return_fig=True`, returns a tuple ``(fig, ax)`` containing
              the generated matplotlib figure and axes.


    NOTE
    -------
    The function is not supported. Use make_simulation_new.

    """
    try:
        q_xy_max = conversion.matrix[0].q_xy_range[1]
        q_z_max = conversion.matrix[0].q_z_range[1]
    except:
        q_xy_max = conversion.matrix[0].q_xy[-1]
        q_z_max = conversion.matrix[0].q_z[-1]

    ai = conversion.matrix[0].ai if len(conversion.matrix) == 1 else conversion.matrix[frame_num].ai

    simul_params = ExpParameters(q_xy_max=q_xy_max, q_z_max=q_z_max, en=12398 / conversion.params.wavelength, ai=ai)

    path_to_cif = [path_to_cif] if not isinstance(path_to_cif, list) else path_to_cif

    for path in path_to_cif:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File does not exist: {path}")

    min_int = [min_int] if not isinstance(min_int, list) else min_int

    if orientation is not None:
        orientation = [orientation] if not (
                    isinstance(orientation[0], list) or orientation[0] is None) else orientation
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

    simulated_data = [
        simul_single_data_old(path_to_cif[i], orientation[i], simul_params, min_int[i], move_fromMW) for i in
        range(len(path_to_cif))]

    if hasattr(conversion, "converted_frame_num") and hasattr(conversion, "img_gid_q") and frame_num in conversion.converted_frame_num:
        index = conversion.converted_frame_num.index(frame_num)
        q_xy, q_z, img = conversion.matrix[0].q_xy, conversion.matrix[0].q_z, [conversion.img_gid_q[index]]
        logging.info(f'Use already converted image with frame num {frame_num}')
    else:
        q_xy, q_z, img = conversion._get_q_data(frame_num)

    if plot_result:
        plot_simul_data_old(get_plot_context(type(conversion).plot_params), img[0], q_xy, q_z, clims, simulated_data,
                            cmap, save_fig, path_to_save_fig,
                            vmin, vmax, linewidth, radius, text_color, plot_mi, xlim, ylim)
        logging.info(f"frame_num = {frame_num} was plotted")
    if return_result:
        simulated_data = sort_simul_data_old(simulated_data)
        if len(simulated_data) == 1:
            return simulated_data[0]
        else:
            return simulated_data



def simul_single_data_old(path_to_cif, orientation, simul_params, min_int, move_fromMW):
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

    with SuppressPrint():
        if orientation is not None:
            orientation = np.array(orientation)
        el = GIWAXSFromCif(path_to_cif, simul_params)
        q, intensity, mi = el.giwaxs.giwaxs_sim(orientation, return_mi=True, move_fromMW=move_fromMW)
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




def sort_simul_data_old(simulated_data):
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