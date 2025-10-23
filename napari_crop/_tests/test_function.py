from napari_crop._function import crop_region, cut_with_plane, draw_fixed_shapes
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
    np.array([[0, 2], [2, 0], [3, 2], [2, 4]]),
]
shape_types = ["rectangle", "ellipse", "polygon"]
crop_expected = [
    np.array([[6, 7, 8], [11, 12, 13], [16, 17, 18], [21, 22, 23]]),
    np.array([[0, 7, 0], [11, 12, 13], [16, 17, 18], [0, 22, 0]]),
    np.array([[0, 0, 2, 0, 0],
              [0, 6, 7, 8, 0],
              [10, 11, 12, 13, 14],
              [0, 0, 17, 0, 0]]),  # fmt: skip
]
bbox_expected = [(1.0, 1.0, 5.0, 4.0),
                 (1.0, 1.0, 5.0, 4.0),
                 (0.0, 0.0, 4.0, 5.0)]

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
# array([[ 0,  2,  0],
#        [ 6,  7,  0],
#        [ 11, 12, 13],
#        [ 0, 17, 18]])


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


@pytest.mark.parametrize(
    "shape,shape_type,bbox_expected",
    zip(shapes, shape_types, bbox_expected)
)
def test_bbox_values(make_napari_viewer, shape, shape_type,
                     bbox_expected):
    """Test that the bbox returned in metadata is correct."""

    viewer = make_napari_viewer()
    img_layer = viewer.add_image(arr_2d)
    shapes_layer = viewer.add_shapes(shape, shape_type=shape_type)
    cropped_actual = crop_region(img_layer, shapes_layer)[0][1]  # get layer properties
    bbox = cropped_actual['metadata']['bbox']
    assert np.array_equal(bbox_expected, bbox)


def test_crop_multiple_shapes(make_napari_viewer):
    """Test that 'n' drawn shapes return 'n' new cropped layers"""

    viewer = make_napari_viewer()
    img_layer = viewer.add_image(arr_2d)
    shapes_layer = viewer.add_shapes(shapes, shape_type=shape_types)
    cropped_actual = crop_region(img_layer, shapes_layer, viewer=viewer)

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
    shape_data = np.insert(shape_data, [0], np.zeros(diff_dims), axis=1)
    shapes_layer = viewer.add_shapes(shape_data, shape_type=shape_type)

    nlayers = len(viewer.layers)

    #  Get first tuple element (data) of first list element (LayerDataTuple)
    cropped_data = crop_region(layer, shapes_layer)[0][0]
    viewer.add_image(cropped_data)

    assert len(viewer.layers) == nlayers + 1

# Tests for cut_with_plane function
# Test different plane normal vectors


volume_data = np.arange(27).reshape(3, 3, 3)
plane_position_2 = (2, 2, 2)  # plane position coordinates

plane_normal_z = (1, 0, 0)
plane_normal_y = (0, 1, 0)
plane_normal_x = (0, 0, 1)
plane_normal_oblique = (0.5, 0.5, 0)
plane_normal_list = [plane_normal_z, plane_normal_y, plane_normal_x, plane_normal_oblique]
plane_normal_ids = ["z", "y", "x", "oblique"]

output_expected_normal_z = np.array([
    [[0, 0, 0,],
     [0, 0, 0],
     [0, 0, 0]],
    [[0, 0, 0],
     [0, 0, 0],
     [0, 0, 0]],
    [[18, 19, 20],
     [21, 22, 23],
     [24, 25, 26]]
], dtype=volume_data.dtype)
output_expected_normal_y = np.array([
    [[0, 0, 0,],
     [0, 0, 0],
     [6, 7, 8]],
    [[0, 0, 0],
     [0, 0, 0],
     [15, 16, 17]],
    [[0, 0, 0],
     [0, 0, 0],
     [24, 25, 26]]
], dtype=volume_data.dtype)
output_expected_normal_x = np.array([
    [[0, 0, 2,],
     [0, 0, 5],
     [0, 0, 8]],
    [[0, 0, 11],
     [0, 0, 14],
     [0, 0, 17]],
    [[0, 0, 20],
     [0, 0, 23],
     [0, 0, 26]]
], dtype=volume_data.dtype)
output_expected_normal_oblique = np.array([
    [[0, 0, 0,],
     [0, 0, 0],
     [0, 0, 0]],
    [[0, 0, 0],
     [0, 0, 0],
     [0, 0, 0]],
    [[0, 0, 0],
     [0, 0, 0],
     [24, 25, 26]]
], dtype=volume_data.dtype)
output_expected_normal_list = [
    output_expected_normal_z,
    output_expected_normal_y,
    output_expected_normal_x,
    output_expected_normal_oblique]


@pytest.mark.parametrize(
    "plane_normal,output_expected_normal", zip(plane_normal_list, output_expected_normal_list), ids=plane_normal_ids
)
def test_cut_with_plane_normals(plane_normal, output_expected_normal):
    image_cut = cut_with_plane(volume_data, plane_normal, plane_position_2)
    assert np.array_equal(output_expected_normal, image_cut)


# Test different plane positions
plane_position_1 = (1, 1, 1)  # plane position coordinates
plane_position_1_4 = (1.4, 1.4, 1.4)  # plane position coordinates
plane_position_list = [plane_position_1, plane_position_1_4]
plane_position_ids = ["position_1", "position_1_4"]

output_expected_normal_z_position_1 = np.array([
    [[0, 0, 0,],
     [0, 0, 0],
     [0, 0, 0]],
    [[9, 10, 11],
     [12, 13, 14],
     [15, 16, 17]],
    [[18, 19, 20],
     [21, 22, 23],
     [24, 25, 26]]
], dtype=volume_data.dtype)
output_expected_normal_z_position_1_4 = output_expected_normal_z

output_expected_positions_list = [output_expected_normal_z_position_1, output_expected_normal_z_position_1_4]


@pytest.mark.parametrize(
    "plane_position,output_expected_position", zip(plane_position_list, output_expected_positions_list), ids=plane_position_ids
)
def test_cut_with_plane_position(plane_position, output_expected_position):
    image_cut = cut_with_plane(volume_data, plane_normal_z, plane_position)
    assert np.array_equal(output_expected_position, image_cut)


# Test negative cut
output_expected_normal_z_position_1_negative = np.array([
    [[0, 1, 2,],
     [3, 4, 5],
     [6, 7, 8]],
    [[0, 0, 0,],
     [0, 0, 0],
     [0, 0, 0]],
    [[0, 0, 0,],
     [0, 0, 0],
     [0, 0, 0]],
], dtype=volume_data.dtype)


def test_cut_with_plane_negative():
    image_cut = cut_with_plane(volume_data, plane_normal_z, plane_position_1, positive_cut=False)
    assert np.array_equal(output_expected_normal_z_position_1_negative, image_cut)


# Tests for draw_fixed_shapes function
points_2d = np.array([[2, 2], [1, 4]])  # 2D points
points_3d = np.array([[1, 2, 2], [1, 1, 4]])  # 3D points with z=1
shape_types_fixed = ["rectangle", "ellipse"]
shape_sizes = [(100, 50), (256, 256)]  # (x, y) sizes


@pytest.mark.parametrize("shape_type", shape_types_fixed)
@pytest.mark.parametrize("shape_size_x,shape_size_y", shape_sizes)
def test_draw_fixed_shapes_2d(make_napari_viewer, shape_type, shape_size_x, shape_size_y):
    """Test drawing fixed shapes in 2D with different types and sizes."""
    viewer = make_napari_viewer()
    points_layer = viewer.add_points(points_2d)
    
    widget = draw_fixed_shapes()
    widget.shape_type.value = shape_type
    widget.shape_size_x.value = shape_size_x
    widget.shape_size_y.value = shape_size_y

    shapes_layer = widget()
    
    # Check that we get the correct number of shapes
    assert len(shapes_layer.data) == len(points_2d)
    
    # Check that all shapes have the correct type
    assert all(st == shape_type for st in shapes_layer.shape_type)
    
    # Check that shapes are rectangles with 4 vertices each
    for shape_data in shapes_layer.data:
        assert shape_data.shape == (4, 2)  # 4 vertices, 2D coordinates


def test_draw_fixed_shapes_3d(make_napari_viewer):
    """Test drawing fixed shapes in 3D."""
    viewer = make_napari_viewer()
    points_layer = viewer.add_points(points_3d)

    widget = draw_fixed_shapes()
    widget.shape_type.value = "rectangle"
    widget.shape_size_x.value = 100
    widget.shape_size_y.value = 100

    shapes_layer = widget()

    # Check that we get the correct number of shapes
    assert len(shapes_layer.data) == len(points_3d)

    # Check that shapes have 3D coordinates (z, y, x)
    for shape_data in shapes_layer.data:
        assert shape_data.shape == (4, 3)  # 4 vertices, 3D coordinates
        shape_size_y=100,
        viewer=viewer
    
    
    # Check that we get the correct number of shapes
    assert len(shapes_layer.data) == len(points_3d)
    
    # Check that shapes have 3D coordinates (z, y, x)
    for shape_data in shapes_layer.data:
        assert shape_data.shape == (4, 3)  # 4 vertices, 3D coordinates

