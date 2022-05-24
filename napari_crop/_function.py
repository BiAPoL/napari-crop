import warnings

import numpy as np
from napari_plugin_engine import napari_hook_implementation
from napari_tools_menu import register_function
import napari
from napari.types import LayerDataTuple
from typing import List


# This is the actual plugin function, where we export our function
# (The functions themselves are defined below)
@napari_hook_implementation
def napari_experimental_provide_function():
    return [crop_region]


@register_function(menu="Utilities > Crop region(s)")
def crop_region(
    viewer: 'napari.viewer.Viewer',
    layer: napari.layers.Layer,
    shapes_layer: napari.layers.Shapes,
) -> List[LayerDataTuple]:
    """Crop regions in napari defined by shapes."""
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

    try:
        rgb = layer_props["rgb"]
    except KeyError:
        rgb = False

    shape_types = shapes_layer.shape_type
    shapes = shapes_layer.data
    cropped_list = []
    new_layer_index = 0
    new_name = layer_props["name"] + " cropped [0]"
    # Get existing layer names in viewer
    names_list = [layer.name for layer in viewer.layers]
    for shape_count, [shape, shape_type] in enumerate(zip(shapes,
                                                          shape_types)):
        # move shape vertices to within image coordinate limits
        layer_shape = np.array(layer_data.shape)
        if rgb:
            layer_shape = layer_shape[:-1]
        # find min and max for each dimension
        start = np.rint(np.min(shape, axis=0))
        stop = np.rint(np.max(shape, axis=0))
        # create slicing indices
        slices = tuple(
            slice(first, last + 1) if first != last else slice(0, None)
            for first, last in np.stack([start.clip(0),
                                         stop.clip(0)]).astype(int).T
        )
        cropped_data = layer_data[slices].copy()
        # handle polygons
        if shape_type != "rectangle":
            mask_nD_shape = np.array(
                [1 if slc.stop is None
                 else (min(slc.stop, layer_data.shape[i]) - slc.start)
                 for i, slc in enumerate(slices)]
            )
            # remove extra dimensions from shape vertices
            # (draw in a single plane)
            verts_flat = np.array(shape - start.clip(0))[:, mask_nD_shape > 1]
            # get a 2D mask
            mask_2D = (
                napari.layers.Shapes(verts_flat, shape_type=shape_type)
                .to_masks()
                .squeeze()
            )
            # match cropped_data (x,y) shape with mask_2D shape
            if rgb:
                shape_dif_2D = np.array(cropped_data.shape[-3:-1]) \
                    - np.array(mask_2D.shape)
            else:
                shape_dif_2D = np.array(cropped_data.shape[-2:]) \
                    - np.array(mask_2D.shape)
            shape_dif_2D = [None if i == 0 else i
                            for i in shape_dif_2D.tolist()]
            mask_2D = mask_2D[:shape_dif_2D[-2], :shape_dif_2D[-1]]
            # add back the rgb(a) dimension
            if rgb:
                mask_nD_shape = np.append(mask_nD_shape, 1)
            # add back dimensions of the original vertices
            mask_nD = mask_2D.reshape(mask_nD_shape)
            # broadcast the mask to the shape of the cropped image
            mask = np.broadcast_to(mask_nD, cropped_data.shape)
            # erase pixels outside drawn shape
            cropped_data[~mask] = 0

            # trim zeros
            non_zero = np.where(cropped_data != 0)
            indices = [slice(min(i_nz), max(i_nz) + 1) for i_nz in non_zero]
            if rgb:
                indices[-1] = slice(None)
            cropped_data = cropped_data[tuple(indices)]

        new_layer_props = layer_props.copy()
        # If layer name is in viewer or is about to be added,
        # give it a different name
        while True:
            new_name = layer_props["name"] \
                + f" cropped [{shape_count+new_layer_index}]"
            if new_name not in names_list:
                break
            else:
                new_layer_index += 1
        new_layer_props["name"] = new_name
        names_list.append(new_name)
        cropped_list.append((cropped_data, new_layer_props, layer_type))
    return cropped_list
