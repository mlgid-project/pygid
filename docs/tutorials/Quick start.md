## How to use


All tutorials use publicly available datasets hosted on the Zenodo repository: https://zenodo.org/records/17466183 The required data are downloaded automatically when running the scripts.

```python
from pygid.datasets import get_dataset, clear_dataset_cache

files = get_dataset("tutorial_00")
# clean up downloaded dataset
clear_dataset_cache()
```
Below is a minimal working example demonstrating how to use `pygid` to convert 2D detector images 
to reciprocal space coordinates in a grazing-incidence diffraction (GID) geometry.


1. Download example dataset from Zenodo or set your own files: 
```python
files = pygid.datasets.get_dataset("tutorial_00")
data_path = files["data_peaks"]
poni_path = files["poni_peaks"]
mask_path = files["mask_peaks"]
```

2. Load experimental parameters from the PONI file
```python
import pygid

params = pygid.ExpParams(
    poni_path=poni_path,        # path to the PONI file
    mask_path=mask_path,        # path to the mask file (EDF/ NPY/ TIFF)
    fliplr=True,                # horizontal flipping of the image
    flipud=True,                # vertical flipping of the image
    transp=False,               # 90 deg rotation of the image
    ai=0.075,                   # angle of incidence in degrees
)
```
3. Create coordinate maps based on geometry and experimental setup
```python
matrix = pygid.CoordMaps(
    params,
    vert_positive=False,        # Cut the positive values for the vertical axis
    hor_positive=False,         # Cut the positive values for the horizontal axis
)
```
4. Initialize pygid.Conversion instance and load the detector image

```python
analysis = pygid.Conversion(
    matrix=matrix,
    path=data_path,             # path to the data file
)
```
5. Perform GID geometry conversion and plot the result, returns the axes and the converted image (list of images)
```python
q_xy, q_z, img = analysis.det2q_gid(
    plot_result=True,                             # plot the result of conversion
    clims=(600, 1e5),                            # image color limits
    save_fig=True, path_to_save_fig="240124_PEN_DIP_polar.png",  # save figure
    return_result=True,                          # return arrays
    save_result=True, path_to_save="240124_PEN_DIP_result.h5",   # save data as a NXsas (NeXus) file
    overwrite_file=False,                        # overwrite the existing file
)
```