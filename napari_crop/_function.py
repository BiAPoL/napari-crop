import warnings

import numpy as np
from napari_plugin_engine import napari_hook_implementation
from napari_tools_menu import register_function
import napari


# This is the actual plugin function, where we export our function
# (The functions themselves are defined below)
@napari_hook_implementation
def napari_experimental_provide_function():
    return [crop_region]


@register_function(menu="Utilities > Crop region")
def crop_region(
    layer: napari.layers.Layer,
    shapes_layer: napari.layers.Shapes,
) -> napari.layers.Layer:

    if shapes_layer is None:
        shapes_layer.mode = "add_rectangle"
        warnings.warn("Please annotate a region to crop.")
        return

    if not (
        isinstance(layer, napari.layers.Image)
        or isinstance(layer, napari.layers.Labels)
    ):
        warnings.warn("Please select an image or labels layer to crop.")
        return

    layer_data, layer_props, layer_type = layer.as_layer_data_tuple()

    shape_types = shapes_layer.shape_type
    shapes = shapes_layer.data
    for shape, shape_type in zip(shapes, shape_types):
        # round shape vertices to integer, not sure if necessary here
        shape = np.round(shape)

        # move shape vertices to within image coordinate limits
        layer_shape = np.array(layer_data.shape)
        if layer_props["rgb"]:
            layer_shape = layer_shape[:-1]
        shape = np.max([shape, np.zeros(shape.shape)], axis=0)
        shape = np.min([shape, np.resize(layer_shape, shape.shape)], axis=0)

        start = np.min(shape, axis=0)
        stop = np.max(shape, axis=0)

        # create slicing indices
        slices = tuple(
            slice(first, last + 1) if first != last else slice(0, None)
            for first, last in np.stack([start, stop]).astype(int).T
        )
        cropped_data = layer_data[slices]


    layer_props["name"] = layer_props["name"] + " (cropped)"
    if layer_type == "image":
        return napari.layers.Image(cropped_data, **layer_props)
    if layer_type == "labels":
        return napari.layers.Labels(cropped_data, **layer_props)
