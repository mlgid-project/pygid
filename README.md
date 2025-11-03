# pygid

## Fast Preprocessing of Grazing Incidence Diffraction (GID) Data 

The `pygid` package converts raw detector images into **cylindrical, Cartesian, polar, and pseudopolar coordinates** and saves the results in **NXsas (NeXus) format**.  

Key features:
- Supports **GID (reflection) and transmission geometries**.  
- Performs **radial and azimuthal integration** to extract 1D line profiles.  
- Handles **single frames, multiple frames, and batch processing**.  
- Provides **polarization, solid angle, absorption, Lorentz, and detector corrections**.  
- Integrates **simulation of GIWAXS patterns** using CIF crystal structure files (via `pygidSIM`).  
- Allows you to **manage and save experimental and sample metadata** along with the converted data in NXsas (HDF5) files.

[//]: # (<p align="center">)

[//]: # (  <img src="docs/images/mlgid_logo_pygid.png" width="400" alt="pygid">)

[//]: # (</p>)

<p align="center">
  <img src="https://raw.githubusercontent.com/mlgid-project/pygid/main/docs/images/mlgid_logo_pygid.png" width="400" alt="pygid">
</p>


Usage examples can be found in [`example/pygid_example.ipynb`](example/pygid_example.ipynb) and on [https://doi.org/10.5281/zenodo.17466183](https://doi.org/10.5281/zenodo.17466183)

### Citation

Abukaev, A., Völter, C., Romodin, M., Schwartzkopff, S., Bertram, F., Konovalov, O., Hinderhofer, A., Lapkin, D., & Schreiber, F. (2025). *pygid: Fast Preprocessing of Grazing Incidence Diffraction Data (GID). Usage examples.*  
[Data set]. Zenodo. [https://doi.org/10.5281/zenodo.17466183](https://doi.org/10.5281/zenodo.17466183)

## Installation

### Install using pip

[//]: # (```bash)

[//]: # ()
[//]: # (pip install --index-url https://test.pypi.org/simple --extra-index-url https://pypi.org/simple pygid)

[//]: # ()
[//]: # (```)

```bash
pip install pygid
```

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

## Short version
Below is a minimal working example demonstrating how to use `pygid` to convert 2D detector images 
to reciprocal space coordinates in a grazing-incidence diffraction (GID) geometry.


```python
import pygid

# Define file paths and experimental data
data_path = r"eiger4m_0000_240124_PEN_DIP.edf"
poni_path = r"LaB6_2024_07_ESRF_ID10.poni"
mask_path = r"mask_2024_07_ESRF_ID10.npy"

# Load experimental parameters from the PONI file
params = pygid.ExpParams(
    poni_path=poni_path,
    mask_path=mask_path,
    fliplr=True,
    flipud=True,
    transp=False,
    ai=0.075,  # angle of incidence in degrees
)

# Create coordinate maps based on geometry and experimental setup
matrix = pygid.CoordMaps(
    params,
    vert_positive=True,
    hor_positive=True,
)

# Initialize conversion instance and load the detector image
analysis = pygid.Conversion(
    matrix=matrix,
    path=data_path,
)

# Perform GID geometry conversion and plot the result
q_xy, q_z, img = analysis.det2q_gid(
    plot_result=True,                            # plot the result of conversion
    clims=(600, 1e5),                            # image color limits
    save_fig=True, path_to_save_fig="240124_PEN_DIP_polar.png",  # save figure
    return_result=True,                          # return arrays
    save_result=True, path_to_save="240124_PEN_DIP_result.h5",   # save data as HDF5 (NXsas format)
    overwrite_file=False,                        # append to existing file if present
)
```

## Detailed overview:

### 1. Import pygid

```python
import pygid
```

### 2. Experimental parameters 

Create an instance of `ExpParams`, the class which contains the experimental parameters. 
The parameters can be loaded from a PONI file (**poni_path**):

```python
params = pygid.ExpParams(
   poni_path='LaB6.poni',  # poni file location
   mask_path='example/mask.npy',  # mask file location (edf/tiff/npy)
   fliplr=True, flipud=True, transp=False, # horizontal and vertical flipping and transpose
   count_range=(10, 10000),  # the intensity range is used to mask hot and dead pixels 
   ai=[0, 0.05, 0.1])  # angle of incidence in GID experiments (in degrees) or list of angles
```
or manually 
```python
params = pygid.ExpParams(
    SDD=0.32716,  # Sample-to-detector distance (in meters)
    wavelength=0.6199,  # Wavelength (in angstroms)
    rot1=-0.00263,  # Detector rotation angle along the horizontal direction (X axis) (in radians)
    rot2=-0.00465,  # Detector rotation angle along the vertical direction (Y axis) (in radians)
    poni1=0.148,  # Beam position in the vertical direction (in m) OR use centerY (pixels)
    poni2=0.144,  # Beam position in the horizontal direction (in m) OR use centerX (pixels)
    px_size=75e-6,  # Detector pixel size (in meters)
    count_range=(10, 10000),  # the intensity range is used to mask hot and dead pixels 
    ai=0,  # angle of incidence in GID experiments (in degrees) or list of angles 
    fliplr=True,  # Flag for horizontal flipping (left-right)
    flipud=True,  # Flag for vertical flipping (up-down)
    transp=False,  # Flag for applying rotation
)
```
Instead of using **poni1** and **poni2**, a user can define beam center using **centerX** and **centerY** in pixels relative to the bottom left corner. 

Other parameters

- **fliplr** and **flipud** – flip the raw image horizontally and vertically, respectively.  
- **transp** – transpose the image relative to the bottom-left corner.  
- **mask_path** – path to the mask file (`.edf`, `.tiff`, `.npy`). Values: `1` = mask, `0` = ignore.  
- **ai** – angle of incidence, can be a single float or a list of angles.  

**OR**  

- **scan** – angular scan command for multiple incidence angles. Example:  
  `"ascan om 0.0400 0.1000 12"` or `"0.0400 0.1000 12"`  
  (start, stop, number-1). The corresponding `ai` list will be calculated automatically.

The definition of the coordinates (poni1, poni2) and an example of creating a .poni calibration file 
based on a calibrant image using the pyFAI-calib2 GUI are described in the
Ref.: https://www.silx.org/doc/pyFAI/latest/usage/cookbook/calib-gui/index.html
Please ensure that the Detector_config has the orientation set to 3 for the script to work correctly.

### 3. Coordinate maps


Create a `CoordMaps` instance based on an `ExpParams` object.  

- If the `ExpParams` instance (`params`) contains a **list of incident angles**, multiple coordinate matrices will be created, one for each angle.  
- Users can define **q- and angle-ranges**:  
  - `q_xy_range`, `q_z_range` – for GID geometry and cylindrical conversion (Å⁻¹)  
  - `q_x_range`, `q_y_range` – for transmission geometry (Å⁻¹)  
  - `radial_range`, `angular_range` – for polar conversion (Å⁻¹ and degrees)  
- The **resolution** of the converted image can be adjusted using:  
  - `dq` – q-resolution (Å⁻¹ per pixel)  
  - `dang` – angular resolution (degrees per pixel)  

#### Correction and geometry flags
- `hor_positive`, `vert_positive` – keep only positive values of q in horizontal or vertical directions  
- `make_pol_corr` – calculate polarization correction matrix  
- `pol_type` – polarization parameter (0–1). `1` = vertical polarization, `0.5` = unpolarized tube  
- `make_solid_angle_corr` – calculate solid angle correction matrix  
- `make_air_attenuation_corr` – calculate air attenuation correction matrix  
- `air_attenuation_coeff` – linear coefficient for air attenuation (1/m)  
- `make_sensor_attenuation_corr` – calculate sensor attenuation correction matrix  
- `sensor_attenuation_coeff` – linear coefficient for sensor attenuation (1/m)  
- `sensor_thickness` – detector sensor thickness (m)  
- `make_absorption_corr` – calculate sample absorption correction matrix  
- `sample_attenuation_coeff` – linear coefficient for sample absorption (1/m)  
- `sample_thickness` – sample thickness (m)  
- `make_lorentz_corr` – calculate Lorentz correction matrix  
- `powder_dim` – powder dimension for Lorentz correction (2 or 3)  

#### Dark and flat field corrections
- `dark_current` – array of dark current values  
- `flat_field` – array of flat field correction values  

#### Save / Load
- `path_to_save` – path to save the coordinate map (`.pkl`)  
- `path_to_load` – path to load an existing coordinate map (`.pkl`)  

```python
matrix = pygid.CoordMaps(params,  # experimental parameters
                         q_xy_range=(0, 4), q_z_range=(0, 4),   # q-range for GID geometry (in A-1) (optional)
                         q_x_range=(0, 4), q_y_range=(0, 4), # q-range for transmission geometry and resolution (in A-1) (optional)
                         dq=0.003, # q-resolution (in A-1) (optional)
                         radial_range=(0, 4), # q_abs range for polar conversion
                         angular_range=(0, 90), # angular range for polar conversion (in degrees) (optional)
                         dang=0.3,  # resolution (in degrees) (optional)
                         hor_positive=False, vert_positive=False,  # flags for only positive values of q in h
                         make_pol_corr=True,  # Flag to calculate polarization correction matrix
                         pol_type=0.98,  # Polarization parameter from 0 to 1. 1 for the vertical polarization, 0.5 for unpolorized tubes.
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
                         path_to_save='matrix.pkl',  # Path where coordinate map will be saved. Path format should be '.pkl'
                         path_to_load=None  # Path from which coordinate map will be loaded. Path format should be '.pkl'
                         )       
```
Load the saved matrix:

```python
matrix = pygid.CoordMaps(
    path_to_load='matrix.pkl')  # Path from which coordinate map will be loaded. Path format should be '.pkl'
```

### 4. Experimental and Sample metadata
Create instances of the `ExpMetadata` and `SampleMetadata` classes 
to describe the experiment and the samples that you want to save along with 
the converted images in NXsas format. 
All fields, as well as the class itself, are optional. 
However, we highly recommend including the following metadata:
- Experimental metadata
```python
exp_metadata = pygid.ExpMetadata(
    start_time=r"2021-03-29T15:51:41.343788",
    end_time=r"2024-07-12T08:26:22Z",
    source_type="synchrotron",
    source_name="ESRF ID10",
    detector="eiger4m",
    monitor = 1.1e5,
    extend_fields = ['monitor'] # list of fields to de appended to the existing data in the NXsas file
)
```
All fields except those listed in **extend_fields** will be overwritten in the NXsas file. 
Parameters in **extend_fields** will be appended to the existing fields.

- Sample metadata
```python
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

```python
smpl_metadata = pygid.SampleMetadata(path_to_save="sample.yaml", data=data)
```

OR 

```python
smpl_metadata = pygid.SampleMetadata(path_to_load="sample.yaml")
```



### 5. Raw image loading and plotting

Create a `Conversion` class instance based on a raw data file (`.edf`, `.tiff`, `.cbf`, `.h5`) or a list of files.  

- In the case of **HDF5 files**, specify the `dataset` key (e.g., `'measurement/eiger4m'`) that points to the raw data in the file.  
- `frame_num` can be:  
  - `None` – load all images  
  - `int` – load a single frame  
  - `list` – load a list of frames  
- For **angular scans**, the number of angles (`ai`) in `ExpParams` should match the number of loaded images.  

#### Parameters

- `matrix` – `CoordMaps` instance 
- `path` – path to the data file or list of files  
- `dataset` – HDF5 dataset key pointing to the raw data  
- `frame_num` – frame(s) to load (None, int, or list)  
- `average_all` – if `True`, average all frames  
- `number_to_average` – number of frames to average for partial averaging  
- `roi_range` – region of interest in the raw image `[y_min, y_max, x_min, x_max]`  
- `multiprocessing` – enable multiprocessing mode (`True` or `False`)  
- `batch_size` – batch size for batch processing mode  

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
Use the `plot_img_raw` method to visualize detector images in pixel coordinates.
#### Parameters

- `frame_num` – number of the frame to plot (int).  
- `clims` – tuple `(vmin, vmax)` specifying color scale limits for the image intensity.  
- `return_result` – if `True`, returns the image data and axes (`x`, `y`, `img`). Default is `False`.  
- `plot_result` – whether to display the plot (`True`/`False`). Default is `True`.  
- `xlim`, `ylim` – tuples specifying X and Y axis limits, e.g. `(0, 500)`. Default is `(None, None)`.  
- `save_fig` – whether to save the figure to a file. Default is `False`.  
- `path_to_save_fig` – path to save the figure if `save_fig=True`. Default is `"img.png"`.  

```python
x, y, img = analysis.plot_img_raw(
    frame_num=0,                 # number of frame to plot
    clims=(0.1, 100),            # color limits (vmin, vmax)
    return_result=True,          # return image data and axes
    plot_result=True,            # display the plot
    xlim=(None, None), ylim=(None, None),  # X and Y axis limits
    save_fig=False,              # save figure flag
    path_to_save_fig="img.png",  # path to save the figure
)

```
### 6. Conversion

The `Conversion` class provides several remapping functions for transforming detector images into reciprocal or polar coordinates.  
These methods convert raw detector data into physically meaningful quantities such as q_xy,q_z, or q_abs.  
All remapping functions use similar parameters. The dataset in memory is automatically deleted after saving.


#### Parameters (for all remapping functions)


- `frame_num` – frame index or list of frame numbers to convert. If `None`, all loaded images are converted.
- `return_result` – if `True`, returns converted image and axes arrays. Default is `False`.  
- `save_result` – whether to save the result as an HDF5 file in NXsas format.  
- `path_to_save` – output path for the HDF5 result file.  
- `h5_group` – name of the NXentry dataset within the HDF5 file.  
- `overwrite_file` – if `True`, overwrites existing HDF5 file. Default is `True`.  
- `overwrite_group` – if `True`, overwrites existing NXentry group. Default is `True`.  
- `exp_metadata` – `ExpMetadata` instance containing experimental information.  
- `smpl_metadata` – `SampleMetadata` instance containing sample information.  
- `plot_result` – whether to plot the remapped image.  
- `clims` – tuple `(vmin, vmax)` defining color scale limits.  
- `xlim`, `ylim` – tuples defining X and Y axis limits for the plot.  
- `save_fig` – whether to save the plotted image to a file.  
- `path_to_save_fig` – path to save the plot (e.g. `.png`, `.tiff`).  
- `interp_type` – interpolation method for remapping (`"INTER_LINEAR"`, `"INTER_NEAREST"`, etc.). Default: `"INTER_LINEAR"`.  
- `multiprocessing` – flag to enable multiprocessing. If `None`, uses the default defined in the `Conversion` class.  

Additionally, q_xy_range, q_z_range, q_x_range, q_y_range, radial_range, angular_range can be changed at this step. 

#### Reciprocal Space Conversion (GID Geometry)

```python
q_xy, q_z, img = analysis.det2q_gid(
    frame_num=[0, 1, 2],           # frames to convert; None = all images
    q_xy_range=(0, 4),             # q_xy range (Å⁻¹)
    q_z_range=(0, 4),              # q_z range (Å⁻¹)
    return_result=True,            # return data arrays
    save_result=True,              # save result as NXsas file
    path_to_save="result.h5",      # path for output file
    h5_group="entry_0000",         # HDF5 dataset name
    overwrite_file=True,           # overwrite existing file
    overwrite_group=False,         # keep existing NXentry group
    exp_metadata=exp_metadata,     # experiment metadata
    smpl_metadata=smpl_metadata,   # sample metadata
    plot_result=True,              # plot result
    clims=(50, 8000),              # color limits
    xlim=(None, None), ylim=(None, None),  # plot limits
    save_fig=False,                # save plot flag
    path_to_save_fig="graph.tiff", # save plot path
    interp_type="INTER_LINEAR",    # interpolation method
    multiprocessing=None,          # multiprocessing flag
)

```
#### Polar and Pseudopolar Coordinate Conversion (GID Geometry)

```python
q_abs, ang, img = analysis.det2pol_gid(plot_result=True, return_result=False, frame_num=0, save_result=False)
q_rad, q_azim, img = analysis.det2pseudopol_gid(plot_result=True, return_result=False, frame_num=0, save_result=False)

```

#### Remapping to reciprocal/polar/pseudopolar coordinates in transmission geometry

```python
q_x, q_y, img =  analysis.det2q(plot_result=True, return_result=True)
q_abs, ang, img = analysis.det2pol( plot_result=True, return_result=True)
q_rad, q_azim, img = analysis.det2pseudopol(plot_result=True, return_result=True)
```

Table 1. Conversion functions with description

| Function              | Description                                          | Name of Output Image | Corresponding Matrix Coordinates |
|-----------------------|------------------------------------------------------|----------------------|----------------------------------|
| `det2q_gid()`         | GID coordinates                                      | `img_gid_q`          | `q_xy`, `q_z`                    |
| `det2pol_gid()`       | polar coordinates for GID experiments                | `img_gid_pol`        | `q_gid_pol`, `ang_gid_pol`       |
| `det2pseudopol_gid()` | pseudopolar coordinates for GID experiments          | `img_gid_pseudopol`  | `q_gid_azimuth`, `q_gid_rad`     |
| `det2q()`             | Cartesian coordinates for transmission experiments   | `img_q`              | `q_x`, `q_y`                     |
| `det2pol()`           | polar coordinates for transmission experiments       | `img_pol`            | `q_pol`, `ang_pol`               |
| `det2pseudopol()`     | pseudopolar coordinates for transmission experiments | `img_pseudopol`      | `q_azimuth`, `q_rad`             |


## Batch Analysis

If you need to process more data files than the defined `batch_size` (default: 32),  
the `Batch()` function is automatically activated.  

At the initialization step (`analysis = pygid.Conversion(...)`), **images are not loaded into memory**.  
When the conversion functions (`det2q_gid`, `det2pol_gid`, etc.) are called,  
the raw data paths are divided into batches and processed sequentially.

In batch mode:
- Converted images **will not be plotted**.  
- The **results cannot be returned** directly to the workspace.  
- Only **saving to disk** is supported, except when `average_all=True`,  
  in which case the averaged result can be accessed.

#### Parameters

- `batch_size` – maximum number of frames or files processed per batch. Default is 32.  
- `multiprocessing` – enables multiprocessing for faster batch execution.  
- `plot_result` – should be set to `False` (plotting is disabled in batch mode).  
- `save_result` – must be `True` to store converted data in HDF5/NXsas format.  
- `path_to_save` – path where converted data will be saved.  
- `overwrite_file` – whether to overwrite the existing result file.

#### Example

```python
analysis = pygid.Conversion(
    matrix=matrix,
    path=data_path,
    img_loc_hdf5='1.1/measurement/eiger4m',
    batch_size=32,  # maximum batch size (default = 32)
)

analysis.det2pol_gid(
    plot_result=False,              # disable plotting
    return_result=False,            # no data returned in batch mode
    multiprocessing=True,           # enable multiprocessing
    save_result=True,               # save converted data
    path_to_save=r"result_converted.h5",  # output file path
    overwrite_file=True,            # overwrite existing file
)

```

## Line Profiles and 1D Integration

The `radial_profile_gid()` and `azim_profile_gid()` functions perform **polar integration** of GIWAXS data:  
- `radial_profile_gid()` averages intensity **over the angular range**, producing a 1D **radial profile** \( I(q_{abs}) \).  
- `azim_profile_gid()` averages intensity **over the radial range**, producing an **azimuthal profile** \( I(\chi) \).  

The `horiz_profile_gid()` function transforms data into **GID coordinates** and averages over the specified \( q_z \) range, producing a **horizontal cut** \( I(q_{xy}) \).  

For **transmission geometry**, use:
- `radial_profile()` — for radial integration I(q_abs) 
- `azim_profile()` — for azimuthal integration I(chi)

All functions return the **x-axis values** (`q_abs`, `ang`, or `q_xy`) and the corresponding list of intensity profiles.

#### Parameters

- `frame_num` – frame index or list of frames to process. If `None`, all are processed.  
- `radial_range` – tuple `(min, max)` for the radial \( q \)-range in Å⁻¹. If `None`, uses the full range.  
- `angular_range` – tuple `(min, max)` for the angular range in degrees.  
- `dang` – angular bin size (degrees).  
- `dq` – radial bin width (Å⁻¹).  
- `return_result` – if `True`, returns computed x-axis and intensity arrays.  
- `save_result` – if `True`, saves the result to an HDF5 file in NXsas format.  
- `path_to_save` – output path for the HDF5 result.  
- `h5_group` – group name inside the HDF5 file.  
- `overwrite_file` – whether to overwrite an existing HDF5 file.  
- `overwrite_group` – whether to overwrite an existing NXentry group.  
- `exp_metadata` – experimental metadata (`ExpMetadata` instance).  
- `smpl_metadata` – sample metadata (`SampleMetadata` instance).  
- `plot_result` – whether to plot the integrated profile.  
- `shift` – vertical offset applied to separate multiple plotted curves.  
- `xlim`, `ylim` – x- and y-axis limits as `(min, max)`; auto-scaled if `None`.  
- `save_fig` – if `True`, saves the plotted figure.  
- `path_to_save_fig` – output path for the saved figure.  
- `multiprocessing` – enables multiprocessing for faster computation.
#### Example — Radial profile in GIWAXS Geometry

```python
q, intensity = analysis.radial_profile_gid(
    frame_num=None,                   # Frame(s) to analyse; all if None
    radial_range=(0, 4),              # Radial q-range in Å⁻¹
    angular_range=(0, 90),            # Angular range in degrees
    dang=0.5,                         # Angular resolution (degrees)
    dq=None,                          # Radial bin width (Å⁻¹)
    return_result=True,               # Return computed q and intensity
    save_result=True,                 # Save result to HDF5
    path_to_save="result.h5",         # Output path
    h5_group=None,                    # Group name in HDF5
    overwrite_file=False,             # Overwrite HDF5 file
    overwrite_group=False,            # Overwrite NXentry group
    exp_metadata=exp_metadata,        # Experimental metadata
    smpl_metadata=smpl_metadata,      # Sample metadata
    plot_result=True,                 # Plot the radial profile
    shift=0.5,                        # Vertical offset between curves
    xlim=None, ylim=None,             # Axis limits
    save_fig=False,                   # Save figure flag
    path_to_save_fig="graph.tiff",    # Figure save path
    multiprocessing=None,             # Use multiprocessing
)
```

#### Example — Azimuthal profile in GIWAXS Geometry
```python
chi, intensity = analysis.azim_profile_gid(
    radial_range=(1.34, 1.4),
    angular_range=(0, 180),
    return_result=True,
    plot_result=True,
    shift=0.5,
)
```

#### Example — Horizontal Integration in GIWAXS Geometry
```python
q_xy, intensity = analysis.horiz_profile_gid(
    q_xy_range=None,
    q_z_range=(0, 3),
    return_result=True,
    plot_result=True,
    shift=1,
)
```
#### Example — Transmission Geometry Integration
```python
q, intensity = analysis.radial_profile(
    plot_result=True,
    return_result=True,
)

chi, intensity = analysis.azim_profile(
    plot_result=True,
    return_result=True,
)
```

Table 2. Conversion line profile functions

| Function Name          | Output Data Name | Axes Name     | Description                                                                           |
|------------------------|------------------|---------------|---------------------------------------------------------------------------------------|
| `radial_profile_gid()` | `rad_cut_gid`    | `q_gid_pol`   | Makes polar remapping and averages in the given angular range (GID geometry)          |
| `radial_profile()`     | `rad_cut`        | `q_pol`       | Makes polar remapping and averages in the given angular range (transmission geometry) |
| `azim_profile_gid()`   | `azim_cut_gid`   | `ang_gid_pol` | Makes polar remapping and averages in the given radial range (GID geometry)           |
| `azim_profile()`       | `azim_cut`       | `ang_pol`     | Makes polar remapping and averages in the given radial range (transmission geometry)  |
| `horiz_profile_gid()`  | `horiz_cut_gid`  | `q_xy`        | Makes cylindrical remapping and averages in the given $q_z$ range (GID geometry)      |

## GID pattern simulation

This part uses the **pygidSIM** package to simulate GIWAXS patterns based on a **CIF file** containing the crystal structure.  
The function `make_simulation()` overlays the simulated diffraction peaks onto the experimental GIWAXS image.

#### Parameters

- `frame_num` – frame number of the experimental data to plot.  
- `clims` – tuple `(vmin, vmax)` specifying intensity limits for the experimental image.  
- `path_to_cif` – path to the `.cif` file defining the crystal structure. Can also be a list of paths for multiple simulations.  
- `orientation` – crystal orientation in the lab frame as `[u, v, w]`; set to `None` for random orientation. Can also be a list for multiple simulations.  
- `min_int` – minimum intensity threshold for displaying simulated reflections. Can also be a list for multiple simulations.  
- `plot_result` – if `True`, displays the overlay of simulated peaks on experimental data.  
- `cmap` – colormap (matplotlib) for the simulated diffraction peaks. Can also be a list for multiple simulations.  
- `vmin`, `vmax` – normalization range for color scaling of simulated peaks.  
- `linewidth` – line width of the simulated diffraction peaks.  
- `radius` – radius of the simulated peaks in display units.  
- `plot_mi` – if `True`, annotate simulated peaks with Miller indices (hkl).  
- `return_result` – if `True`, returns a simulation result object.  

```python
result = analysis.make_simulation(
    frame_num=0,                  # Frame of experimental data
    clims=(30, 8000),             # Color scale limits for experimental image
    path_to_cif="struct.cif",     # Path to crystal structure CIF
    orientation=[1, 0, 0],        # Crystal orientation in lab frame
    min_int=5e-1,                 # Minimum intensity for reflections
    plot_result=True,             # Display simulation overlay
    cmap=cm.Blues,                # Colormap for simulated peaks
    vmin=0.5, vmax=1,             # Color normalization range
    linewidth=1.5,                # Line width of peaks
    radius=0.1,                   # Peak radius
    plot_mi=False,                # Annotate peaks with Miller indices
    return_result=True            # Return simulation result
)
```
#### Multiple Simulations

To plot multiple simulated patterns from different CIF files or orientations, `make_simulation()` accepts **lists** for the following parameters:

- `path_to_cif` – list of CIF file paths  
- `orientation` – list of crystal orientations  
- `min_int` – list of minimum intensity thresholds for reflections  
- `cmap` – list of colormaps for simulated peaks  

**Behavior:**  
- If the numbers of elements in `path_to_cif` and `orientation` are equal, they will be **paired** and used sequentially for the simulations.  
- Lists of `min_int` and `cmap` will be applied correspondingly to each simulation.

In order to plot multiple simulated patterns based on different orientations or CIF files, make_simulation() function
supports
lists of path_to_cif, orientation, min_int and cmap. If numbers of elements in path_to_cif and orientation are equal,
they will be used respectively. 

## Set Plot Parameters

To change global plotting parameters for all plots, use the `Conversion.set_plot_defaults()` function:

```python
analysis.set_plot_defaults(cmap='jet')  # Example: set global colormap
```
The whole list of keys:
```python
analysis.set_plot_defaults(
    font_size=14,                 # Default font size for text elements (titles, labels, ticks)
    axes_titlesize=14,            # Font size for axes titles
    axes_labelsize=18,            # Font size for axes labels (x and y)
    grid=False,                   # Display gridlines (True/False)
    grid_color='gray',            # Gridline color
    grid_linestyle='--',          # Gridline style ('--', '-', ':')
    grid_linewidth=0.5,           # Gridline width
    xtick_labelsize=14,           # Font size for x-axis tick labels
    ytick_labelsize=14,           # Font size for y-axis tick labels
    legend_fontsize=12,           # Font size for legend text
    legend_loc='best',            # Legend location ('best', 'upper right', etc.)
    legend_frameon=True,          # Display frame around legend (True/False)
    legend_borderpad=1.0,         # Padding between legend content and frame
    legend_borderaxespad=1.0,     # Padding between legend and axes
    figure_titlesize=16,          # Font size for figure title
    figsize=(6.4, 4.8),           # Figure size in inches
    axes_linewidth=0.5,           # Axes line width
    savefig_dpi=600,              # DPI for saving figures
    savefig_transparent=False,    # Save figure with transparent background
    savefig_bbox_inches=None,     # Portion of figure to save ('tight' to crop)
    savefig_pad_inches=0.1,       # Padding around figure when saving
    line_linewidth=2,             # Line width for plot lines
    line_color='blue',            # Default line color
    line_linestyle='-',           # Line style for plot lines
    line_marker=None,             # Marker style for line plots
    scatter_marker='o',           # Marker style for scatter plots
    scatter_edgecolors='black',   # Edge color for scatter plot markers
    cmap='inferno'                # Colormap for images ('inferno', 'viridis', etc.)
)
```