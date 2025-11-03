import pytest
import numpy as np
from pathlib import Path
from pygid import ExpParams, CoordMaps, Conversion


@pytest.fixture
def test_data_paths():
    """Fixture to provide paths to test data."""
    return {
        'poni_path': Path(__file__).parent / 'test_data' / 'LaB6_calib.poni',
        'mask_path': Path(__file__).parent / 'test_data' / 'mask.npy',
        'data_path': Path(__file__).parent / 'test_data' / 'LaB6_raw.edf'
    }

def test_coordmaps(test_data_paths):
    poni_path = test_data_paths['poni_path']
    mask_path = test_data_paths['mask_path']
    data_path = test_data_paths['data_path'].as_posix()

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

    matrix = CoordMaps(exp_params, hor_positive=False, vert_positive=False, dang=0.1, dark_current=None, dq=0.002,
                        make_pol_corr=False, pol_type=1,
                        make_solid_angle_corr=False,
                        make_air_attenuation_corr=False, air_attenuation_coeff=1,
                        make_sensor_attenuation_corr=False, sensor_attenuation_coeff=1, sensor_thickness=0.1,
                        make_absorption_corr=False, sample_attenuation_coeff=1, sample_thickness=200e-9,
                        make_lorentz_corr=False, powder_dim=3)
    analysis = Conversion(matrix=matrix, path=data_path)
    analysis.det2q_gid(plot_result=False)

    assert matrix.ai == 0.01, "angle of incidence changed"
    assert matrix.dang == 0.1, "dang changed"
    assert matrix.dq == 0.002, "dq changed"
    assert hasattr(matrix, "p_x_gid"), "p_x_gid was not calculated"
    assert hasattr(matrix, "p_y_gid"), "p_y_gid was not calculated"
    assert matrix.radial_range is not None, "radial_range was not calculated"
    assert matrix.q_xy_range[0] < 0 and matrix.q_xy_range[1] > 0, "q_xy_range was not calculated correctly"

def test_coordmaps_pos(test_data_paths):
    poni_path = test_data_paths['poni_path']
    mask_path = test_data_paths['mask_path']
    data_path = test_data_paths['data_path'].as_posix()

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

    matrix = CoordMaps(exp_params, hor_positive=True, vert_positive=True, dang=0.1, dark_current=None, dq=0.002,
                       make_pol_corr=False, pol_type=1,
                       make_solid_angle_corr=False,
                       make_air_attenuation_corr=False, air_attenuation_coeff=1,
                       make_sensor_attenuation_corr=False, sensor_attenuation_coeff=1, sensor_thickness=0.1,
                       make_absorption_corr=False, sample_attenuation_coeff=1, sample_thickness=200e-9,
                       make_lorentz_corr=False, powder_dim=3)
    analysis = Conversion(matrix=matrix, path=data_path)
    analysis.det2q_gid(plot_result=False)

    assert matrix.ai == 0.01, "angle of incidence changed"
    assert matrix.dang == 0.1, "dang changed"
    assert matrix.dq == 0.002, "dq changed"
    assert hasattr(matrix, "p_x_gid"), "p_x_gid was not calculated"
    assert hasattr(matrix, "p_y_gid"), "p_y_gid was not calculated"
    assert matrix.radial_range is not None, "radial_range was not calculated"
    assert matrix.q_xy_range[0] == 0 and matrix.q_xy_range[1] > 0, "q_xy_range was not calculated correctly"
    assert matrix.q_z_range[0] == 0 and matrix.q_z_range[1] > 0, "q_z_range was not calculated correctly"

def test_corrmaps(test_data_paths):
    poni_path = test_data_paths['poni_path']
    mask_path = test_data_paths['mask_path']
    data_path = test_data_paths['data_path'].as_posix()

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

    matrix = CoordMaps(exp_params, hor_positive=False, vert_positive=False, dang=0.1, dark_current=None, dq=0.002,
                       make_pol_corr=True, pol_type=1,
                       make_solid_angle_corr=True,
                       make_air_attenuation_corr=True, air_attenuation_coeff=1,
                       make_sensor_attenuation_corr=True, sensor_attenuation_coeff=1, sensor_thickness=0.1,
                       make_absorption_corr=True, sample_attenuation_coeff=1, sample_thickness=200e-9,
                       make_lorentz_corr=True, powder_dim=3)
    analysis = Conversion(matrix=matrix, path=data_path)
    analysis.det2q_gid(plot_result=False)

    corr_maps = matrix.corr_matrices
    assert corr_maps.pol_corr_matrix is not None, "pol_corr_matrix was not calculated"
    assert corr_maps.solid_angle_corr_matrix is not None, "solid_angle_corr_matrix was not calculated"
    assert corr_maps.air_attenuation_corr_matrix is not None, "air_attenuation_corr_matrix was not calculated"
    assert corr_maps.sensor_attenuation_corr_matrix is not None, "sensor_attenuation_corr_matrix was not calculated"
    assert corr_maps.absorption_corr_matrix is not None, "absorption_corr_matrix was not calculated"
    assert corr_maps.lorentz_corr_matrix is not None, "lorentz_corr_matrix was not calculated"

if __name__ == "__main__":
    pytest.main()
