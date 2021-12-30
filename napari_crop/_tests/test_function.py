from napari_crop._function import crop_region
import pytest
import numpy as np


arr_2d = np.arange(0, 25).reshape((5, 5))  # 2d case

# array([[ 0,  1,  2,  3,  4],
#        [ 5,  6,  7,  8,  9],
#        [10, 11, 12, 13, 14],
#        [15, 16, 17, 18, 19],
#        [20, 21, 22, 23, 24]])
shapes = [
    np.array([[1, 1], [1, 3], [4, 3], [4, 1]]),
    np.array([[0.5, 0.5], [0.5, 3.5], [4.51, 3.5], [4.51, 0.5]]),
    np.array([[0, 2], [4, 4], [4, 2], [2, 0]]),
]
shape_types = ["rectangle", "ellipse", "polygon"]
crop_expected = [
    np.array([[6, 7, 8], [11, 12, 13], [16, 17, 18], [21, 22, 23]]),
    np.array([[0, 7, 0], [11, 12, 13], [16, 17, 18], [0, 22, 0]]),
    np.array([[0, 0, 2, 0, 0],
              [0, 6, 7, 0, 0],
              [0, 11, 12, 13, 0],
              [0, 0, 17, 18, 0],
              [0, 0, 0, 0, 0]]),  # fmt: skip
]

# rectangle crop
# array([[ 6,  7,  8],
#        [11, 12, 13],
#        [16, 17, 18],
#        [21, 22, 23]])

# ellipse crop
# array([[ 0,  7,  0],
#        [11, 12, 13],
#        [16, 17, 18],
#        [0, 22, 0]])

# diamond crop (bool mask based on napari.layers.Shapes.to_mask())
# array([[ 0,  0,  2,  0,  0],
#        [ 0,  6,  7,  0,  0],
#        [ 0, 11, 12, 13,  0],
#        [ 0,  0, 17, 18,  0],
#        [ 0,  0,  0,  0,  0]])


@pytest.mark.parametrize(
    "shape,shape_type,crop_expected",
    zip(shapes, shape_types, crop_expected),
    ids=shape_types,
)
def test_crop_function_values_2d(make_napari_viewer, shape, shape_type,
                                 crop_expected):
    """Test that the cropped output is the expected array."""

    viewer = make_napari_viewer()
    img_layer = viewer.add_image(arr_2d)
    shapes_layer = viewer.add_shapes(shape, shape_type=shape_type)
    cropped_actual = crop_region(img_layer, shapes_layer)
    cropped_actual_arrays = [cropped[0] for cropped in cropped_actual][0]
    assert np.array_equal(crop_expected, cropped_actual_arrays)


def test_crop_multiple_shapes(make_napari_viewer):
    """Test that 'n' drawn shapes return 'n' new cropped layers"""

    viewer = make_napari_viewer()
    img_layer = viewer.add_image(arr_2d)
    shapes_layer = viewer.add_shapes(shapes, shape_type=shape_types)
    cropped_actual = crop_region(img_layer, shapes_layer)

    assert len(shapes) == len(cropped_actual)


layer_data = [
    [np.random.random((8, 8, 4)), True, "image"],  # 2d rgba
    [np.random.random((8, 8)), False, "image"],  # 2d
    [np.random.random((8, 8, 8)), False, "image"],  # 3d
    [np.random.random((8, 8, 8, 8)), False, "image"],  # 4d
    [np.arange(64).reshape(8, 8), False, "labels"],  # labels data
]
shape_data = [
    np.array([[2, 2], [2, 5], [4, 5], [4, 2]]),  # 2x3 crop
    np.array([[-2, -2], [-2, 5], [4, 5], [4, -2]]),  # neg crop
    np.array([[-100, -100], [-100, 100], [100, 100], [100, -100]]),  # oversized crop
    np.array([[0, 2], [4, 4], [4, 2], [2, 0]]),  # diamond crop
]
shape_types = ["rectangle", "rectangle", "rectangle", "polygon"]
layer_data_ids = ["2D_rgba", "2D", "3D", "4D", "labels"]
shape_data_ids = ["2x3_crop", "neg_crop", "big_crop", "poly_crop"]


@pytest.mark.parametrize("layer_data,rgb,layer_type", layer_data, ids=layer_data_ids)
@pytest.mark.parametrize(
    "shape_data,shape_type", zip(shape_data, shape_types), ids=shape_data_ids
)
def test_crop_function_nd(layer_data, rgb, layer_type, shape_data, shape_type,
                          make_napari_viewer):

    viewer = make_napari_viewer()

    if layer_type == "image":
        layer = viewer.add_image(layer_data, rgb=rgb)
    elif layer_type == "labels":
        layer = viewer.add_labels(layer_data)

    # shape data is (N,D) array where N is num verts and D is num dims
    diff_dims = layer_data.ndim - shape_data.shape[1]
    if rgb:
        diff_dims -= 1
    shape_data = np.insert(shape_data, [-1], np.zeros(diff_dims), axis=1)
    shapes_layer = viewer.add_shapes(shape_data, shape_type=shape_type)

    nlayers = len(viewer.layers)

    #  Get first tuple element (data) of first list element (LayerDataTuple)
    cropped_data = crop_region(layer, shapes_layer)[0][0]
    viewer.add_image(cropped_data)

    assert len(viewer.layers) == nlayers + 1
