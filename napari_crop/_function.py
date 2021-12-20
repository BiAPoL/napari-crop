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

def crop_irregular_shape(cropped_data,data_shape,shapes_masks,start_position,end_position,ctz=None):
    '''Çrops data with irregular shapes by means of applying a nD mask'''
    mask2D = np.zeros((data_shape[-2],data_shape[-1]),dtype=bool)
    if len(shapes_masks.shape) > 2:  # get mask where it was drawn (channel, time, z)
        if len(shapes_masks.shape) == 3:
            shapes_masks2D = shapes_masks[ctz[0],:,:]
        elif len(shapes_masks.shape) == 4:
            shapes_masks2D = shapes_masks[ctz[0],ctz[1],:,:]
        elif len(shapes_masks.shape) == 5:
            shapes_masks2D = shapes_masks[ctz[0],ctz[1],ctz[2],:,:]
        else:
            return(None)
    else:
        shapes_masks2D = shapes_masks
    mask2D[:shapes_masks.shape[-2],:shapes_masks.shape[-1]] = shapes_masks2D  # mask2D matches data shape
    mask = np.broadcast_to(mask2D, data_shape)  # broadcast 2D to nD
    if len(data_shape) == 2:
        cropped_mask = mask[start_position[0]:end_position[0], start_position[1]:end_position[1]]
    elif len(data_shape) == 3:
        cropped_mask = mask[start_position[0]:end_position[0], start_position[1]:end_position[1],
                           start_position[2]:end_position[2]]
    elif len(data_shape) == 4:
        cropped_mask = mask[start_position[0]:end_position[0], start_position[1]:end_position[1],
                           start_position[2]:end_position[2], start_position[3]:end_position[3]]
    elif len(data_shape) == 5:
        cropped_mask = mask[start_position[0]:end_position[0], start_position[1]:end_position[1],
                           start_position[2]:end_position[2], start_position[3]:end_position[3],
                           start_position[4]:end_position[4]]
    cropped_data[~cropped_mask] = 0  # clear pixels outside mask
    return(cropped_data)


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
    rectangles = viewer.layers[1].data
    shapes_masks = viewer.layers[1].to_masks()  # get array of masks from drawn shapes
    shape_types = viewer.layers[1].shape_type   # get shape type (rectangle, ellipse, etc)
    for index,rectangle in enumerate(rectangles):  # iterate over all drawn shapes
        start_position = rectangle.min(axis=0)
        end_position = rectangle.max(axis=0)
        size = np.ceil((end_position - start_position)).astype(int) # using np.ceil and np.around yields better coherence with drawn shapes
        start_position = np.around(start_position).astype(int)
        end_position = start_position + size
        if rectangle.shape[1]>2:  # if > 2D
            ctz = rectangle[0,:-2].astype(int)  # get current channel, time and z indices
        for i in range(len(start_position)):
            if start_position[i] == end_position[i]:
                start_position[i] = 0
                end_position[i] = data.shape[i]

        if len(data.shape) == 2:
            # hard copy to avoid messing with original data when clearing pixels outside mask
            cropped_data = np.copy(data[start_position[0]:end_position[0], start_position[1]:end_position[1]])
            if shape_types[index] != 'rectangle':
                cropped_data = crop_irregular_shape(cropped_data,data.shape,shapes_masks[index],start_position,end_position)
        elif len(data.shape) == 3:
            cropped_data = np.copy(data[start_position[0]:end_position[0], start_position[1]:end_position[1],
                           start_position[2]:end_position[2]])
            if shape_types[index] != 'rectangle':
                cropped_data = crop_irregular_shape(cropped_data,data.shape,shapes_masks[index],start_position,end_position,ctz)
        elif len(data.shape) == 4:
            cropped_data = np.copy(data[start_position[0]:end_position[0], start_position[1]:end_position[1],
                           start_position[2]:end_position[2], start_position[3]:end_position[3]])
            if shape_types[index] != 'rectangle':
                cropped_data = crop_irregular_shape(cropped_data,data.shape,shapes_masks[index],start_position,end_position,ctz)
        elif len(data.shape) == 5:
            cropped_data = np.copy(data[start_position[0]:end_position[0], start_position[1]:end_position[1],
                           start_position[2]:end_position[2], start_position[3]:end_position[3],
                           start_position[4]:end_position[4]])
            if shape_types[index] != 'rectangle':
                cropped_data = crop_irregular_shape(cropped_data,data.shape,shapes_masks[index],start_position,end_position,ctz)
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
