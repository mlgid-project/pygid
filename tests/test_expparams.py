import pytest
import numpy as np
from pathlib import Path
from pygid import ExpParams

@pytest.fixture
def test_data_paths():
    """Fixture to provide paths to test data."""
    return {
        'poni_path': Path(__file__).parent / 'test_data' / 'LaB6_calib.poni',
        'mask_path': Path(__file__).parent / 'test_data' / 'mask.npy'
    }

def test_exp_params_initialization_poni(test_data_paths):
    poni_path = test_data_paths['poni_path']
    mask_path = test_data_paths['mask_path']

    assert poni_path.exists(), f"Missing PONI file at {poni_path}"
    assert mask_path.exists(), f"Missing mask file at {mask_path}"

    exp_params = ExpParams(
        poni_path=poni_path,
        mask_path=mask_path,
        fliplr=False,
        flipud=False,
        transp=False,
        ai=0.01
    )

    assert exp_params.poni_path == poni_path, "poni_path was not set correctly"
    assert exp_params.px_size is not None, "px_size was not set correctly"
    assert isinstance(exp_params.mask, np.ndarray), "Mask is not a NumPy array"
    assert exp_params.mask.shape == (2162, 2068), f"Unexpected mask shape: {exp_params.mask.shape}"
    assert exp_params.ai == 0.01, f"AI angle mismatch: expected 0.01, got {exp_params.ai}"

def test_exp_params_initialization_manual():
    exp_params = ExpParams(
        fliplr=True,
        flipud=True,
        transp=False,
        SDD=0.3271661836504515,
        wavelength=0.6199,
        rot1=-0.00263,
        rot2=-0.00465,
        centerX=2000,
        centerY=2145,
        px_size=75e-6,
        count_range=[10, 10000],
        ai=0.01
    )

    assert exp_params.poni_path is None, "Unexpected poni_path"
    assert exp_params.mask is None, f"Unexpected mask"
    assert exp_params.ai == 0.01, f"AI angle mismatch: expected 0.01, got {exp_params.ai}"
    assert exp_params.centerX == 2000, f"centerX angle mismatch: expected 2000, got {exp_params.centerX}"


if __name__ == "__main__":
    pytest.main()
