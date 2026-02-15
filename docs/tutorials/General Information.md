# pygid

## Fast Preprocessing of Grazing Incidence Diffraction (GID) Data 

`pygid` is a Python-based package for fast conversion of 2D detector images into reciprocal (Cartesian and polar) coordinates. Although the package focuses on the grazing-incidence geometry, it can also be used for transmission (SAXS/WAXS) experiments.  

Key features:  
- Supports **grazing-incidence and transmission geometries**.  
- Converts area detector images to **Cartesian, polar, and pseudopolar** coordinates.  
- Based on widely used detector geometry description (**PONI files**).  
- Performs **radial and azimuthal** profiling.  
- Handles **single frames, multiple frames, and batch processing**.  
- Provides a wide range of intensity corrections, including **polarization, solid angle, absorption, Lorentz, and detector** corrections.  
- Utilizes **simulation of GIWAXS peak positions** using CIF crystal structure files (via `pygidSIM`).  
- Includes **experimental and sample metadata** management.  
- Allows **plotting** of the conversion results with adjustable parameters.  
- **Reuses the coordinate maps** for several images with the same geometry.  
- Supports **several interpolation techniques**.  
- Saves the results of conversion and metadata as a **NXsas** (NeXus) file.  
- Can be used as a first step in the **`mlgid` analysis pipeline**.
 provided examples demonstrate the functionality of `pygid`, covering the main workflow described in Tutorials 1–4. A minimal working example is available in the *Quick Start*(`quick_start.ipynb`).

All tutorials use publicly available datasets hosted on the Zenodo repository:  
https://zenodo.org/records/17466183
The required data are downloaded automatically when running the scripts.
Example of loading:
```python
from pygid.datasets import get_dataset
files = get_dataset("tutorial_00")
poni_path = files["poni"]
mask_path = files["mask"]
data_path = files["data"]
```

 After completing the analysis, the downloaded files can be removed using:

```python
from pygid.datasets import clear_dataset_cache
clear_dataset_cache()
```
