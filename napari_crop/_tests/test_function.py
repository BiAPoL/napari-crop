from napari_crop._function import crop_region
import pytest
import numpy as np


arr_2d = np.arange(0, 25).reshape((5, 5))  # 2d case
# array([[ 0,  1,  2,  3,  4],
#        [ 5,  6,  7,  8,  9],
#        [10, 11, 12, 13, 14],
#        [15, 16, 17, 18, 19],
#        [20, 21, 22, 23, 24]])
shapes = [np.array([[1, 1], [1, 3], [4, 3], [4, 1]])]
shape_types = [
    "rectangle",
]
crop_expected = [np.array([[6, 7, 8], [11, 12, 13], [16, 17, 18], [21, 22, 23]])]
# rectangle crop
# array([[ 6,  7,  8],
#        [11, 12, 13],
#        [16, 17, 18],
#        [21, 22, 23]])


@pytest.mark.parametrize(
    "shape,shape_type,crop_expected",
    zip(shapes, shape_types, crop_expected),
    ids=shape_types,
)
def test_crop_function_values_2d(make_napari_viewer, shape, shape_type, crop_expected):
    """Test that the cropped output is the expected array."""

    viewer = make_napari_viewer()
    img_layer = viewer.add_image(arr_2d)
    shapes_layer = viewer.add_shapes(shape, shape_type=shape_type)
    cropped_actual = crop_region(img_layer, shapes_layer, viewer)
    assert np.array_equal(crop_expected, cropped_actual.data)


image_data = [
    [np.random.random((8, 8, 3)), True],  # 2d rgb
    [np.random.random((8, 8, 4)), True],  # 2d rgba
    [np.random.random((8, 8)), False],  # 2d
    [np.random.random((2, 8, 8)), False],  # 3d
    [np.random.random((2, 2, 2, 2, 8, 8)), False],  # 6d
]
shape_data = [
    np.array([[2, 2], [2, 5], [4, 5], [4, 2]]),  # 2x3 crop
    np.array([[-2, -2], [-2, 5], [4, 5], [4, -2]]),  # neg crop
    np.array([[-100, -100], [-100, 100], [100, 100], [100, -100]]),  # oversided crop
]
shape_types = ["rectangle", "rectangle", "rectangle"]
image_data_ids = ["2D_rgb", "2D_rgba", "2D", "3D", "6D"]
shape_data_ids = ["2x3_crop", "neg_crop", "big_crop"]


@pytest.mark.parametrize("image_data,rgb", image_data, ids=image_data_ids)
@pytest.mark.parametrize(
    "shape_data,shape_type", zip(shape_data, shape_types), ids=shape_data_ids
)
def test_crop_function_nd(image_data, rgb, shape_data, shape_type, make_napari_viewer):
    viewer = make_napari_viewer()

    img_layer = viewer.add_image(image_data, rgb=rgb)

    # shape data is (N,D) array where N is num verts and D is num dims
    diff_dims = image_data.ndim - shape_data.shape[1]
    shape_data = np.insert(shape_data, [-1], np.zeros(diff_dims), axis=1)
    shp_layer = viewer.add_shapes(shape_data, shape_type=shape_type)

    nlayers = len(viewer.layers)
    crop_region(img_layer, shp_layer, viewer)
    assert len(viewer.layers) == nlayers + 1
