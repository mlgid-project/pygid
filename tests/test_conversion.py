import pytest
import numpy as np
import h5py, os
from pathlib import Path
from pygid import ExpParams, CoordMaps, Conversion


@pytest.fixture
def test_data_paths():
    """Fixture to provide paths to test data."""
    return {
        'poni_path': Path(__file__).parent / 'test_data' / 'LaB6_calib.poni',
        'mask_path': Path(__file__).parent / 'test_data' / 'mask.npy',
        'data_path': Path(__file__).parent / 'test_data' / 'LaB6_raw.edf',
        'cif_path': Path(__file__).parent / 'test_data' / 'LaB6_raw.cif',
    }


def test_conversion(test_data_paths):
    poni_path = test_data_paths['poni_path']
    mask_path = test_data_paths['mask_path']
    data_path = test_data_paths['data_path']
    data_path = data_path.as_posix()

    assert poni_path.exists(), "Missing PONI file"
    assert mask_path.exists(), "Missing mask file"

    exp_params = ExpParams(
        poni_path=poni_path,
        mask_path=mask_path,
        fliplr=True,
        flipud=True,
        transp=False,
        ai=0.01
    )

    matrix = CoordMaps(exp_params, hor_positive=False, vert_positive=False, dang=0.1, dark_current=None, dq=0.002)
    analysis = Conversion(matrix=matrix, path=data_path)

    x, y, img = analysis.det2q_gid(plot_result=False, return_result=True)
    assert x is not None, "Missing x"
    assert y is not None, "Missing y"
    assert img is not None, "Missing image"
    assert analysis.img_gid_q is not None, "Image was not converted"

    x, y, img = analysis.det2q_gid(plot_result=False, return_result=True, q_xy_range=[0, 1], q_z_range=[0, 1])
    assert [np.nanmin(x), np.round(np.nanmax(x), 1)] == [0, 1], f"q_xy_range is not correct {np.nanmin(x), np.nanmax(x)}"
    assert [np.nanmin(y), np.round(np.nanmax(y), 1)] == [0, 1], f"q_z_range is not correct {np.nanmin(y), np.nanmax(y)}"


def test_conversion_all_2D_functions(test_data_paths):
    poni_path = test_data_paths['poni_path']
    mask_path = test_data_paths['mask_path']
    data_path = test_data_paths['data_path']
    data_path = data_path.as_posix()

    assert poni_path.exists(), "Missing PONI file"
    assert mask_path.exists(), "Missing mask file"

    exp_params = ExpParams(
        poni_path=poni_path,
        mask_path=mask_path,
        fliplr=True,
        flipud=True,
        transp=False,
        ai=0.01
    )

    matrix = CoordMaps(exp_params)
    analysis = Conversion(matrix=matrix, path=data_path)

    analysis.det2pol_gid(plot_result=False, return_result=False)
    assert analysis.img_gid_pol is not None, "image img_gid_pol was not converted"

    analysis.det2pseudopol_gid(plot_result=False, return_result=False)
    assert analysis.img_gid_pseudopol is not None, "image img_gid_pseudopol was not converted"

    analysis.det2q(plot_result=False, return_result=False)
    assert analysis.img_q is not None, "img_q was not converted"

    analysis.det2pol(plot_result=False, return_result=False)
    assert analysis.img_pol is not None, "image img_pol was not converted"

    analysis.det2pseudopol(plot_result=False, return_result=False)
    assert analysis.img_pseudopol is not None, "image img_pseudopol was not converted"


def test_conversion_all_1D_functions(test_data_paths):
    poni_path = test_data_paths['poni_path']
    mask_path = test_data_paths['mask_path']
    data_path = test_data_paths['data_path']
    data_path = data_path.as_posix()

    assert poni_path.exists(), "Missing PONI file"
    assert mask_path.exists(), "Missing mask file"

    exp_params = ExpParams(
        poni_path=poni_path,
        mask_path=mask_path,
        fliplr=True,
        flipud=True,
        transp=False,
        ai=0.01
    )

    matrix = CoordMaps(exp_params)
    analysis = Conversion(matrix=matrix, path=data_path)

    analysis.radial_profile_gid(plot_result=False, return_result=False)
    assert analysis.rad_cut_gid is not None, "rad_cut_gid was not calculated"

    analysis.radial_profile(plot_result=False, return_result=False)
    assert analysis.rad_cut is not None, "rad_cut was not calculated"

    analysis.azim_profile_gid(plot_result=False, return_result=False)
    assert analysis.azim_cut_gid is not None, "azim_cut_gid was not calculated"

    analysis.azim_profile(plot_result=False, return_result=False)
    assert analysis.azim_cut is not None, "azim_cut was not calculated"

    analysis.horiz_profile_gid(plot_result=False, return_result=False)
    assert analysis.horiz_cut_gid is not None, "horiz_cut_gid was not calculated"


def test_saving(test_data_paths):
    poni_path = test_data_paths['poni_path']
    mask_path = test_data_paths['mask_path']
    data_path = test_data_paths['data_path']
    data_path = data_path.as_posix()

    assert poni_path.exists(), "Missing PONI file"
    assert mask_path.exists(), "Missing mask file"

    exp_params = ExpParams(
        poni_path=poni_path,
        mask_path=mask_path,
        fliplr=True,
        flipud=True,
        transp=False,
        ai=0.01
    )

    matrix = CoordMaps(exp_params)
    analysis = Conversion(matrix=matrix, path=data_path)

    filename = Path(__file__).parent / 'test_data' / 'LaB6_q.h5'
    filename = filename.as_posix()
    h5_group = "entry_0000"
    analysis.det2q_gid(plot_result=False, return_result=True, save_result=True, path_to_save=filename, h5_group=h5_group)
    assert Path(filename).exists(), "HDF5 file was not created"
    with h5py.File(filename, "r") as f:
        assert h5_group in f, f"Group '{h5_group}' not found in HDF5 file"
    os.remove(filename)
    assert not Path(filename).exists(), "HDF5 file was not deleted"


if __name__ == "__main__":
    pytest.main()


# def test_simulation():
#     poni_path = test_data_paths['poni_path']
#     mask_path = test_data_paths['mask_path']
#     data_path = test_data_paths['data_path'].as_posix()
#     cif_path = test_data_paths['cif_path'].as_posix()
#
#
#     assert os.path.exists(poni_path), "Missing PONI file"
#     assert os.path.exists(mask_path), "Missing mask file"
#
#     exp_params = ExpParams(
#         poni_path=poni_path,
#         mask_path=mask_path,
#         fliplr=True,
#         flipud=True,
#         transp=False,
#         ai=0.01
#     )
#
#     matrix = CoordMaps(exp_params)
#     analysis = Conversion(matrix=matrix, path=data_path)
#     q, i, mi = analysis.make_simulation(frame_num=0, path_to_cif=cif_path, orientation=[1, 0, 0], min_int=5e-4,
#                                          plot_result=False, vmin=0.5, vmax=1, linewidth=1.5, radius=0.1,
#                                          plot_mi=False, return_result=True, max_shift=4)
#     assert len(q) == 2, "wrong number of q"
#     assert len(i) > 0, "wrong number of i"
#     assert len(mi) == len(i), "wrong number of mi"
#
#     matrix = CoordMaps(exp_params)
#     analysis = Conversion(matrix=matrix, path=data_path)
#     q, i, mi = analysis.make_simulation(frame_num=0, path_to_cif=cif_path, orientation=None, min_int=5e-4,
#                                          plot_result=False, vmin=0.5, vmax=1, linewidth=1.5, radius=0.1,
#                                          plot_mi=False, return_result=True, max_shift=4)
#     assert len(q) == len(i), "wrong number of q"
#     assert len(mi) == len(i), "wrong number of mi"
