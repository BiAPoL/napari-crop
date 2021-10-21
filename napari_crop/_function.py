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
def crop_region(layer: napari.layers.Layer, shapes_layer: napari.layers.Shapes, viewer : napari.Viewer):
    if shapes_layer is None:
        shapes_layer = viewer.add_shapes([])
        shapes_layer.mode = 'add_rectangle'
        warnings.warn("Please annotate a region to crop.")
        return

    if not(isinstance(layer, napari.layers.Image) or isinstance(layer, napari.layers.Labels)):
        warnings.warn("Please select an image or labels layer to crop.")
        return

    data = layer.data

    rectangle = viewer.layers[1].data[-1]
    start_position = rectangle.min(axis=0).astype(int)
    end_position = rectangle.max(axis=0).astype(int)

    cropped_data = data[start_position[0]:end_position[0], start_position[1]:end_position[1]]


    if isinstance(layer, napari.layers.Image):
        new_layer = viewer.add_image(cropped_data,
            name = layer.name + "(cropped)",
            opacity = layer.opacity,
            gamma = layer.gamma,
            contrast_limits = layer.contrast_limits,
            colormap = layer.colormap,
            blending = layer.blending,
            interpolation = layer.interpolation,
        )
    else : # labels
        new_layer = viewer.add_labels(
            np.asarray(cropped_data),
            opacity=layer.opacity,
            blending=layer.blending,
        )
        new_layer.contour = layer.contour