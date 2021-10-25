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
    start_position = rectangle.min(axis=0)
    end_position = rectangle.max(axis=0)
    size = (end_position - start_position).astype(int)

    start_position = start_position.astype(int)
    end_position = start_position + size

    for i in range(len(start_position)):
        if start_position[i] == end_position[i]:
            start_position[i] = 0
            end_position[i] = data.shape[i]

    if len(data.shape) == 2:
        cropped_data = data[start_position[0]:end_position[0], start_position[1]:end_position[1]]
    elif len(data.shape) == 3:
        cropped_data = data[start_position[0]:end_position[0], start_position[1]:end_position[1],
                       start_position[2]:end_position[2]]
    elif len(data.shape) == 4:
        cropped_data = data[start_position[0]:end_position[0], start_position[1]:end_position[1],
                       start_position[2]:end_position[2], start_position[3]:end_position[3]]
    elif len(data.shape) == 5:
        cropped_data = data[start_position[0]:end_position[0], start_position[1]:end_position[1],
                       start_position[2]:end_position[2], start_position[3]:end_position[3],
                       start_position[4]:end_position[4]]
    else:
        warnings.warn("Data with " + str(len(data.shape)) + " dimensions not supported for cropping.")
        return

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