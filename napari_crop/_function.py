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
        cropped_data = layer_data[slices].copy()

        if shape_type != "rectangle":
            mask_nD_shape = np.array(
                [1 if slc.stop == None else (slc.stop - slc.start) for slc in slices]
            )
            # remove extra dimensions from shape vertices (draw in a single plane)
            verts_flat = np.array(shape - start)[:, mask_nD_shape > 1]
            # get a 2D mask
            mask_2D = (
                napari.layers.Shapes(verts_flat, shape_type=shape_type)
                .to_masks()
                .squeeze()
            )
            # add back the rgb(a) dimension
            if layer_props["rgb"]:
                mask_nD_shape = np.append(mask_nD_shape, 1)
            # add back dimensions of the original vertices
            mask_nD = mask_2D.reshape(mask_nD_shape)
            # broadcast the mask to the shape of the cropped image
            mask = np.broadcast_to(mask_nD, cropped_data.shape)
            cropped_data[~mask] = 0

    layer_props["name"] = layer_props["name"] + " (cropped)"
    if layer_type == "image":
        return napari.layers.Image(cropped_data, **layer_props)
    if layer_type == "labels":
        return napari.layers.Labels(cropped_data, **layer_props)
