# _pygid_

## Fast Preprocessing of Grazing Incidence Diffraction Data (GID)

<p align="center">
  <img src="image/pygid-logo.png" width="400" alt="pygid">
</p>

The package converts raw detector images into cylindrical, Cartesian, polar, and pseudopolar coordinates and saves the
result as a NXsas file.

## Installation

### Install from source

First, clone the repository:

```bash
git clone https://github.com/mlgid-project/pygid.git
```

Then, to install all required modules, navigate to the cloned directory and execute:

```bash
cd pygid
pip install -e .
```

## How to use

Below is a short example of how to use the package.

```python
import pygid

# loading of poni-file and mask 
params = pygid.ExpParams(poni_path='LaB6.poni',
                         mask_path='mask.npy',
                         ai=0.1,
                         fliplr=True, flipud=True, transp=False)

# creation of coordinate matrix based on given parameters
matrix = pygid.CoordMaps(params)

# Description of the experiment and samples to save with converted images. OPTIONAL
exp_metadata = pygid.ExpMetadata(
   start_time=r"2021-03-29T15:51:41.343788",
   source_type="synchrotron",
   source_name="ESRF ID10",
   detector_name="eiger4m",
   instrument_name="ID10")

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

# data loading

data_path = "LaB6_0001.h5"
analysis = pygid.Conversion(matrix=matrix, path=data_path, dataset='/1.1/measurement/eiger4m', frame_num=0)

# Conversion

analysis.det2q_gid(frame_num=0, return_result=False,
                   plot_result=True, clims=(50, 8000),
                   save_result=True, path_to_save="result.h5",
                   exp_metadata=exp_metadata, smpl_metadata=smpl_metadata)
analysis.det2pol_gid(plot_result=True, return_result=False, save_result=False)
analysis.det2pseudopol_gid(plot_result=True, return_result=False, save_result=False)

```

![The resulting images](image/image1.PNG)

### Detailed overview of package usage:

1. Import pygid

```python
import pygid
```

2. Create an instance of ExpParams, the class which contains the experimental parameters. Load poni-file and mask.edf(
   .npy, .tiff) (optional).
   Use the fliplr and flipud options to flip the raw image vertically and horizontally, respectively. Use the rot option
   to transpose the image relative to the left bottom corner.
   Define angle of incidence (ai) as a float value or a list of angles. Use scan option for angular scans, e.g. scan = "
   ascan om 0.0400 0.1000 12 3" or scan = "0.0400 0.1000 12" (start, stop, number-1).
   The ai list will be calculated automatically.

```python
# loading of poni-file and mask (optional)
params = pygid.ExpParams(poni_path='LaB6.poni',  # poni file location
                         mask_path='mask.npy',  # mask file location (edf/tiff/npy)
                         fliplr=True, flipud=True, transp=False,  # flags for horizontal and vertical flipping and transpose
                         count_range=(10, 10000),  # the intensity range is used to mask hot and dead pixels 
                         ai=[0, 0.05, 0.1])  # angle of incidence in GID experiments (in degrees) or list of angles
```

It is also possible to manually add experimental parameters. You should provide values for (poni1 and poni2) in meters
or for beam positions (centerX and centerY) in pixels (relative to the bottom left corner) in the raw image.

```python
params = pygid.ExpParams(
    fliplr=True,  # Flag for horizontal flipping (left-right)
    flipud=True,  # Flag for vertical flipping (up-down)
    transp=False,  # Flag for applying rotation
    SDD=0.3271661836504515,  # Sample-to-detector distance (in meters)
    wavelength=0.6199,  # Wavelength (in angstroms)
    rot1=-0.00263,  # Detector rotation angle along the horizontal direction (X axis) (in radians)
    rot2=-0.00465,  # Detector rotation angle along the vertical direction (Y axis) (in radians)
    centerX=2000,  # Beam position in the horizontal direction (in pixels)
    centerY=2145,  # Beam position in the vertical direction (in pixels)
    px_size=75e-6,  # Detector pixel size (in meters)
    count_range=(10, 10000),  # the intensity range is used to mask hot and dead pixels 
    ai=0  # angle of incidence in GID experiments (in degrees) or list of angles     
)
```

Creation of poni-file based on a calibrant image using pyFAi-calib2 GUI is described in
Ref.: https://www.silx.org/doc/pyFAI/latest/usage/cookbook/calib-gui/index.html
please use the Detector_config with the orientation set to 3 for the script to work correctly.

3. Create instances of ExpMetadata and SampleMetadata classes with a description of the experiment and the samples that
   you want to save with converted images in an NeXus format.
   All fields as well as the class element are optional. However, we highly recommend to add the following metadata:

```python
exp_metadata = pygid.ExpMetadata(
    start_time=r"2021-03-29T15:51:41.343788",
    source_type="synchrotron",
    source_name="ESRF ID10",
    detector_name="eiger4m",
    instrument_name="ID10")

data = {
    "name": "240306_DIP",
    "structure": {
        "stack": "air | DIP 0-25| SiOx 1| Si",
        "materials": {
            "DIP": {
                "name": "Diindenoperylene DIP",
                "thickness": 25e-9,  # optional
                "cif": "DIP.cif",  # optional
                "type": "gradient film"  # optional
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
```

Sample metadata can be saved as an YAML file using path_to_save and loaded from an YAML file using path_to_load.

4. Create CoordMaps instance based on ExpParams. If ExpParams instance (params) consists of a list of incident angles,
   multiple matrices will be created for each angle.

```python
matrix = pygid.CoordMaps(params,  # experimental parameters
                         q_xy_range=None, q_z_range=None, dq=0.003,  # q-range and resolution (in A-1)
                         ang_min=0, ang_max=90, dang=0.1,  # angle range and resolution (in degrees)
                         hor_positive=False, vert_positive=False,  # flags for only positive values of q in h
                         make_pol_corr=True,  # Flag to calculate polarization correction matrix
                         pol_type=0.98,
                         # Polarization parameter from 0 to 1. 0.98 for synchrotrons, 0.5 for unpolorized tubes.
                         make_solid_angle_corr=True,  # Flag to calculate solid angle correction matrix
                         make_air_attenuation_corr=False,  # Flag to calculate air attenuation correction matrix
                         air_attenuation_coeff=1,  # Linear coefficient for air attenuation correction (in 1/m)
                         make_sensor_attenuation_corr=False,  # Flag to calculate sensor attenuation correction matrix
                         sensor_attenuation_coeff=1,  # Linear coefficient for sensor attenuation correction (in 1/m)
                         sensor_thickness=0.1,  # Thickness of the detector sensor (in m)
                         make_absorption_corr=False,  # Flag to calculate absorption correction matrix
                         sample_attenuation_coeff=1,  # Linear coefficient for sample attenuation correction (in 1/m)
                         sample_thickness=200e-9,  # Thickness of the sample (in m)
                         make_lorentz_corr=False,  # Flag to calculate Lorentz correction matrix
                         powder_dim=3,  # Dimension of powder for Lorentz correction: 2 or 3
                         dark_current=None,  # Array for dark current values
                         flat_field=None,  # Array for flat field correction values
                         path_to_save='matrix.pkl',
                         # Path where coordinate map will be saved. Path format should be '.pkl'
                         path_to_load=None
                         # Path from which coordinate map will be loaded. Path format should be '.pkl'
                         )       
```

One can save the coordinate matrices as pkl-file using path_to_save and load them by using path_to_load. However, the
saving- and loading time is comparable with the calculation time

```python
matrix = pygid.CoordMaps(
    path_to_load='matrix.pkl')  # Path from which coordinate map will be loaded. Path format should be '.pkl'
```

4. Create Conversion class instance based on raw data file (edf/tiff/cbf/h5) or list of files.
   In the case of h5 files, add dataset key like 'measurement/eiger4m' which is a root to the raw data in h5 file.
   The key frame_num is used for choosing several raw images in the dataset. It can be None (all images will be loaded),
   integer (single image) or list of numbers.
   In case of angular scans, number of ai in experimental parameters should be equal to number of loaded images.

```python
data_path = "LaB6_0001.h5"

analysis = pygid.Conversion(matrix=matrix,  # coordinate map
                            path=data_path,  # data file location (h5, tiff or edf) or list of them 
                            dataset='1.1/measurement/eiger4m',  # raw image location in h5 file
                            frame_num=0,  # list or number of necessary frame in series 
                            average_all=False,  # key for averaging of all frames 
                            number_to_average=5,  # key for partial averaging      
                            roi_range=[0, 500, 0, 500],  # raw image range of interest 
                            multiprocessing=False,  # key for multiprocessing mode activation
                            batch_size=32,  # Size of the batches in the Batch processing mode
                            )
```

5. Raw image plotting

```python
x, y, img = analysis.plot_img_raw(clims=(0.1, 100),  # tuple specifying color limits (vmin, vmax) for the image.
                                  frame_num=0,  # number of frame to plot
                                  xlim=(None, None), ylim=(None, None),  # X and Y image limits
                                  return_result=True,
                                  # if True, returns the image data and axes used for plotting. Default is False.
                                  plot_result=True,  # whether to display the plot. Default is True. 
                                  save_fig=False,  # whether to save the figure to a file. Default is False.
                                  path_to_save_fig="img.png",
                                  # path to save the figure if save_fig is True. Default is "img.png".  
                                  fontsize=14,  # font size for tick labels. Default is 14.
                                  labelsize=18,  # font size for axis labels. Default is 18.       
                                  )

```

6. Remapping to reciprocal/polar/pseudopolar coordinates in reflection (GIWAXS) geometry. For all remapping functions
   the parameters are same. Dataset is automatically deleted after saving.

```python
analysis.det2q_gid(clims=(50, 8000),  # Tuple specifying color limits (vmin, vmax) for the image.
                   frame_num=[0, 1, 2],
                   # frame number or list of numbers of loaded raw images to convert. If None, will convet all loaded images   
                   plot_result=True,  # flag to plot the result
                   return_result=False,  # if True, returns the image data and corresponding axes. Default is False.
                   radial_range=None,  # radial range (in angstroms)
                   angular_range=(0, 90),  # angular range (in degrees)
                   save_result=True,  # flag to save the result as a NeXus (HDF5) file
                   path_to_save="result.h5",  # path to save the result with experimental params.
                   h5_group="entry_0000",  # dataset name in the h5-file
                   overwrite_file=True,  # whether to overwrite existing HDF5 file. Default is True.
                   overwrite_group=False,  # whether to overwrite NXentry group in the HDF5 file. Default is True
                   exp_metadata=exp_metadata,  # experiment metadata that will be saved with result
                   smpl_metadata=smpl_metadata,  # sample metadata that will be saved with result
                   save_fig=False,  # flag to save the results as a picture
                   path_to_save_fig="graph.tiff",  # path to save the image
                   xlim=(None, None), ylim=(None, None),  # X and Y image limits
                   fontsize=14,  # font size for tick labels. Default is 14.
                   labelsize=18,  # font size for axis labels. Default is 18. 
                   cmap='inferno',  # colormap used for plotting. Default is "inferno".
                   interp_type="INTER_LINEAR",  # interpolation method used for remapping. Default is "INTER_LINEAR".
                   multiprocessing=None,
                   # key for multiprocessing mode activation. If None, uses default setting defined in Conversion class instance creation.
                   q_xy_range=(0, 4),  # tuple specifying the min and max of q_xy range. If None, uses full range.
                   q_z_range=(0, 4),  # tuple specifying the min and max of q_z range. If None, uses full range.

                   )
analysis.det2pol_gid(plot_result=True, return_result=False, frame_num=0, save_result=False)
analysis.det2pseudopol_gid(plot_result=True, return_result=False, frame_num=0, save_result=False)

```

Remapping to reciprocal/polar/pseudopolar coordinates in transmission geometry

```python
analysis.det2q(clims=clims, frame_num=0, plot_result=False)
analysis.det2pol(clims=clims, frame_num=0, plot_result=False)
analysis.det2pseudopol(clims=clims, frame_num=0, plot_result=False)
```

Table 1. Conversion functions with description

| Function              | Description                                          | Name of Output Image | Corresponding Matrix Coordinates |
|-----------------------|------------------------------------------------------|----------------------|----------------------------------|
| `det2q_gid()`         | GIWAXS coordinates                                   | `img_gid_q`          | `q_xy`, `q_z`                    |
| `det2pol_gid()`       | polar coordinates for GID experiments                | `img_gid_pol`        | `q_gid_pol`, `ang_gid_pol`       |
| `det2pseudopol_gid()` | pseudopolar coordinates for GID experiments          | `img_gid_pseudopol`  | `q_gid_azimuth`, `q_gid_rad`     |
| `det2q()`             | Cartesian coordinates for transmission experiments   | `img_q`              | `q_x`, `q_y`                     |
| `det2pol()`           | polar coordinates for transmission experiments       | `img_pol`            | `q_pol`, `ang_pol`               |
| `det2pseudopol()`     | pseudopolar coordinates for transmission experiments | `img_pseudopol`      | `q_azimuth`, `q_rad`             |

7. Manual saving of NeXus (HDF5) file with converted data. Can be useful if one wants to change the result of conversion
   before saving.

```python
analysis.save_nxs(path_to_save="result.h5",
                  # location with the name (.h5) to save, or the key_to_change of the raw image path otherwise it will be saved to the same directory 
                  overwrite_file=False,  # the existing file will be overwritten if True
                  exp_metadata=exp_metadata, smpl_metadata=smpl_metadata)  # Metadata class element
```

### Batch analysis

If you want to process more than batch_size (32 default), the Batch() function activates.
At the initialization step analysis = pygid.Conversion(...), the images will not be loaded.
When conversion functions are used, the raw data paths will be divided into batches and processed one-by-one.
In this case, the functionality of the code is limited,
converted images will not be plotted, result cannot be returned, only saving is possible, except the case of
average_all = True.

```python
analysis = pygid.Conversion(matrix=matrix, path=data_path, img_loc_hdf5='1.1/measurement/eiger4m',
                            batch_size=32,  # maximum size of the batch (32 default)
                            )
analysis.det2pol_gid(plot_result=False, return_result=False, multiprocessing=True,
                     save_result=True, path_to_save=r"result_converted.h5", overwrite_file=True)

```

### Line profiles and 1D integration

radial_profile_gid() and azim_profile_gid() functions takes radial and angular ranges, performs polar integration and
averages along angle or radial axes, respectively.
Horizontal profile horiz_profile_gid() makes transformation to the GID coordinates and averages in the given q_z range.

For the transmission geometry use radial_profile() and azim_profile() functions.

```

q, I = analysis.radial_profile_gid(
                    frame_num=None,                   # Frame number(s) to analyze; if None, all are processed
                    radial_range=(0,4),               # Radial q-range (min, max) in Å⁻¹; full range if None
                    angular_range=(0, 90),            # Angular range in degrees (min, max) for integration
                    multiprocessing=None,             # Use multiprocessing for faster computation if True
                    return_result=True,               # Return computed q-values, intensity profile
                    save_result=True,                 # Save computed profile to an NeXus (HDF5) file if True
                    save_fig=False,                   # Save the figure if True
                    path_to_save_fig="graph.tiff",    # Path to save the figure
                    plot_result=True,                 # Display the radial profile plot
                    shift=0.5,                        # Vertical shift between plotted lines for clarity
                    tick_size=None,                   # Font size for tick labels; default if None
                    fontsize=None,                    # Font size for axis labels; default if None
                    xlim=None,                        # X-axis limits as (min, max); auto-scaled if None
                    ylim=None,                        # Y-axis limits as (min, max); auto-scaled if None
                    dang=0.5,                         # Angular resolution in degrees for binning
                    dq=None,                          # Radial bin width in Å⁻¹; default binning if None
                    path_to_save="result.h5",         # Path to save the result file
                    h5_group=None,                    # Group name in the HDF5 file; default if None
                    overwrite_file=False,             # Overwrite existing HDF5 file if True
                    exp_metadata = exp_metadata,      # experiment metadata that will be saved with result
                    smpl_metadata = smpl_metadata,    # sample metadata that will be saved with result
                )

chi, I = analysis.azim_profile( plot_result = True, shift = 0.5, radial_range = (1.34,1.4), angular_range = (0,180), return_result = True)
q_xy, I = analysis.horiz_profile( plot_result = True, shift = 1, q_xy_range = None, q_z_range = (0, 3), return_result = True)


q, I = analysis.radial_profile(plot_result = True)
chi, I = analysis.azim_profile(plot_result = True)

```

Table 2. Conversion line profile functions

| Function Name          | Output Data Name | Axes Name     | Description                                                                           |
|------------------------|------------------|---------------|---------------------------------------------------------------------------------------|
| `radial_profile_gid()` | `rad_cut_gid`    | `q_gid_pol`   | Makes polar remapping and averages in the given angular range (GID geometry)          |
| `radial_profile()`     | `rad_cut`        | `q_pol`       | Makes polar remapping and averages in the given angular range (transmission geometry) |
| `azim_profile_gid()`   | `azim_cut_gid`   | `ang_gid_pol` | Makes polar remapping and averages in the given radial range (GID geometry)           |
| `azim_profile()`       | `azim_cut`       | `ang_pol`     | Makes polar remapping and averages in the given radial range (transmission geometry)  |
| `horiz_profile_gid()`  | `horiz_cut_gid`  | `q_xy`        | Makes cylindrical remapping and averages in the given $q_z$ range (GID geometry)      |

### GID pattern simulation

This part is based on pygidSIM package that simulates GIWAXS patterns based on a CIF file with crystal structure.   
The function make_simulation() plots the simulated data and converted experimental image.

```python
result = analysis.make_simulation(
    frame_num=0,  # Frame number of the experimental data to plot
    clims=(30, 8000),  # Intensity limits for the color scale of the experimental image
    path_to_cif="struct.cif",  # Path to the .cif file defining the crystal structure
    orientation=[1, 0, 0],  # Crystal orientation in the lab frame; set to None for random orientation
    min_int=5e-1,  # Minimum intensity threshold for displaying simulated reflections
    plot_result=True,  # Display the simulation overlay on experimental data if True
    cmap=cm.Blues,  # Colormap (matplotlib) used for the simulated diffraction peaks 
    vmin=0.5, vmax=1,  # Normalization range for simulated peak intensity color scaling
    linewidth=1.5,  # Line width of the simulated diffraction peaks
    radius=0.1,  # Radius of simulated diffraction peaks in display units
    plot_mi=False,  # If True, annotate simulated peaks with Miller indices (hkl)
    return_result=True  # If True, return simulation result object
)
```

In order to plot multiple simulated patterns based on different orientations or CIF files, make_simulation() function
supports
lists of path_to_cif, orientation, min_int and cmap. If numbers of elements in path_to_cif and orientation are equal,
they will be used respectively. 

