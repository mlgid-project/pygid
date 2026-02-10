from pathlib import Path
from typing import Dict, Optional
from urllib.parse import quote
import urllib.request
import shutil
import os

CACHE_ROOT = Path.home() / ".cache" / "pygid"
ZENODO_BASE = "https://zenodo.org/records/17466183/files"
ZENODO_FILES = {
    "tutorial_00": {
        "data": "eiger4m_0000_240124_PEN_DIP.h5",
        "poni": "LaB6_2024_07_ESRF_ID10.poni",
        "mask": "mask_2024_07_ESRF_ID10.npy",
    },
    "tutorial_01": {
        # "data": "eiger4m_0000_240124_PEN_DIP.h5",
        "poni": "LaB6_2024_07_ESRF_ID10.poni",
        "mask": "mask_2024_07_ESRF_ID10.npy",
    },
    "tutorial_02": {
        "poni": "LaB6_2024_07_ESRF_ID10.poni",
        "mask": "mask_2024_07_ESRF_ID10.npy",
    },
    "tutorial_03": {
        "data_h5": "eiger4m_0000_240124_PEN_DIP.h5",
        "poni_for_h5": "LaB6_2024_07_ESRF_ID10.poni",
        "mask_for_h5": "mask_2024_07_ESRF_ID10.npy",
        "data1_tiff": "S121_MAI_A2_00841.tif",
        "data2_tiff": "S121_MAI_A2_00841.tif",
        "data3_tiff": "S121_MAI_A2_00841.tif",
        "poni_for_tiff": "LaB6_2021_12_DESY_P08.poni",
    },
    "tutorial_04": {
        "data": "eiger4m_0000_240124_PEN_DIP.h5",
        "poni": "LaB6_2024_07_ESRF_ID10.poni",
        "mask": "mask_2024_07_ESRF_ID10.npy",
    },
    "tutorial_05": {
        "data": "eiger4m_0000_240124_PEN_DIP.h5",
        "poni": "LaB6_2024_07_ESRF_ID10.poni",
        "mask": "mask_2024_07_ESRF_ID10.npy",
    },
    "tutorial_06": {
        "data": "eiger4m_0000_240124_PEN_DIP.h5",
        "poni": "LaB6_2024_07_ESRF_ID10.poni",
        "mask": "mask_2024_07_ESRF_ID10.npy",
    },
    "tutorial_07": {
        "data": "eiger4m_0000_240124_PEN_DIP.h5",
        "poni": "LaB6_2024_07_ESRF_ID10.poni",
        "mask": "mask_2024_07_ESRF_ID10.npy",
    },
    "tutorial_08": {
        "data": "eiger4m_0000_240124_PEN_DIP.h5",
        "poni": "LaB6_2024_07_ESRF_ID10.poni",
        "mask": "mask_2024_07_ESRF_ID10.npy",
    },
    "tutorial_09": {
        "data_peaks": "eiger4m_0000_240124_PEN_DIP.h5",
        "poni_peaks": "LaB6_2024_07_ESRF_ID10.poni",
        "mask_peaks": "mask_2024_07_ESRF_ID10.npy",
        "cif_peaks":  "DIP_thin_film_642482.cif",
        "data_rings": "S121_MAI_A2_00841.tif",
        "poni_rings": "LaB6_2021_12_DESY_P08.poni",
        "cif_rings": "MAPBTI02 - stoumpos_cubicMAPbI3.cif",
    },
    "tutorial_10": {
        "data": "S121_MAI_A2_00841.tif",
        "poni": "LaB6_2021_12_DESY_P08.poni",
    },
    "tutorial_11": {
        "data": "eiger4m_0000_240124_PEN_DIP.h5",
        "poni": "LaB6_2024_07_ESRF_ID10.poni",
        "mask": "mask_2024_07_ESRF_ID10.npy",
        "smpl_metadata": "240124_PEN_DIP_metadata.yaml",
        "cif": "DIP_thin_film_642482.cif",
    },
}


def get_dataset(name: str) -> Dict[str, Path]:
    """
    Download all files for a given dataset from Zenodo and return local paths.

    Files are cached locally. Incomplete or corrupted downloads are
    automatically re-downloaded.
    """
    files = _get_dataset_files(name)
    cache_dir = _prepare_cache_dir(name)

    result: Dict[str, Path] = {}
    for key, filename in files.items():
        result[key] = _get_file(filename, cache_dir)

    return result


def _get_dataset_files(name: str) -> Dict[str, str]:
    if name not in ZENODO_FILES:
        raise KeyError(f"Unknown dataset: {name}")
    return ZENODO_FILES[name]


def _prepare_cache_dir(name: str) -> Path:
    cache_dir = CACHE_ROOT / name
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _get_file(filename: str, cache_dir: Path) -> Path:
    url = _build_url(filename)
    final_path = cache_dir / filename
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")

    if _needs_download(final_path, url):
        _download_file(url, final_path, tmp_path)

    return str(final_path)


def _build_url(filename: str) -> str:
    encoded = quote(filename)
    return f"{ZENODO_BASE}/{encoded}?download=1"


def _needs_download(path: Path, url: str) -> bool:
    if not path.exists():
        return True

    try:
        with urllib.request.urlopen(url) as response:
            remote_size = response.headers.get("Content-Length")
        if remote_size is None:
            return False
        return path.stat().st_size != int(remote_size)
    except Exception:
        return True


def _download_file(url: str, final_path: Path, tmp_path: Path) -> None:
    final_path.unlink(missing_ok=True)
    tmp_path.unlink(missing_ok=True)

    with urllib.request.urlopen(url) as response, open(tmp_path, "wb") as f:
        shutil.copyfileobj(response, f)

    os.replace(tmp_path, final_path)

def clear_dataset_cache(dataset_name: Optional[str] = None) -> None:
    """
    Remove cached dataset files.

    Parameters
    ----------
    dataset_name : str, optional
        Name of the dataset to remove. If None, all cached datasets are removed.
    """
    if dataset_name is None:
        _clear_all_datasets()
    else:
        _clear_single_dataset(dataset_name)


def _clear_all_datasets() -> None:
    if CACHE_ROOT.exists():
        shutil.rmtree(CACHE_ROOT)


def _clear_single_dataset(dataset_name: str) -> None:
    dataset_path = CACHE_ROOT / dataset_name
    if dataset_path.exists():
        shutil.rmtree(dataset_path)
