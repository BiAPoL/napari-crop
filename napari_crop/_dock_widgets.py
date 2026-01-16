import numpy as np

from magicgui.widgets import Container, PushButton, ComboBox, CheckBox
from typing import TYPE_CHECKING
from skimage.segmentation import relabel_sequential
from napari_tools_menu import register_dock_widget

from ._utils import array_allclose_in_list, find_array_allclose_position_in_list
from ._function import cut_with_plane, trim_zeros, get_nonzero_slices
import napari.layers
if TYPE_CHECKING:
    import napari.viewer


@register_dock_widget(menu="Utilities > Cut volume with plane (napari-crop)")
class CutWithPlane(Container):
    input_layer_types = (
        napari.layers.Image,
        napari.layers.Labels,)

    def __init__(self, viewer: "napari.viewer.Viewer",
                 plane_data_source: str = 'reference_layer_data', positive_cut: bool = True, **kwargs):
        if kwargs:
            import warnings
            warnings.warn(
                "The 'crop' argument is deprecated and will be removed in a future version. "
                "Cropping will always be applied.",
                DeprecationWarning,
                stacklevel=2
            )
        self._viewer = viewer
        # Create plane layer variable (needed for proper reference image combobox initialization)
        self._plane_layer = None
        # Create reference image combobox
        self._reference_image_combobox = ComboBox(
            choices=self._get_reference_layers,
            label='Reference Layer:',
            tooltip='Data source to be displayed in plane layer.\nData dimensions must be 3 and not rgb.')
        self._viewer.layers.events.inserted.connect(self._reference_image_combobox.reset_choices)
        self._viewer.layers.events.removed.connect(self._reference_image_combobox.reset_choices)
        # Create layer to be cut combobox
        self._layer_to_be_cut_combobox = ComboBox(
            choices=self._get_layers_to_be_cut,
            label='Layer to Be Cut:',
            tooltip='Layer containing data be cut.\nData dimensions must be 3 and not rgb.')
        self._viewer.layers.events.inserted.connect(self._layer_to_be_cut_combobox.reset_choices)
        self._viewer.layers.events.removed.connect(self._layer_to_be_cut_combobox.reset_choices)
        # Create plane data source combobox
        self._plane_data_source = plane_data_source
        self._plane_data_combobox = ComboBox(
            choices=[
                'blank',
                'reference_layer_data'],
            value=self._plane_data_source,
            label='Plane Data:')
        # Create positive cut checkbox
        self._positive_cut = positive_cut
        self._positive_cut_checkbox = CheckBox(value=self._positive_cut, label='Positive Cut')
        # Create ortogonal plane swap button
        self._ortogonal_plane_swap_btn = PushButton(label="Ortogonal Planes",
                                                    tooltip='Swap plane to ortogonal direction.\nShortcut: \'P\' key')
        # Create cut button
        self._run_btn = PushButton(label="Cut")
        # Connect callbacks
        self._run_btn.clicked.connect(self._on_cut_clicked)
        self._reference_image_combobox.changed.connect(self._on_image_layer_changed)
        self._plane_data_combobox.changed.connect(self._on_plane_data_source_changed)
        self._positive_cut_checkbox.changed.connect(self._on_positive_cut_changed)
        self._viewer.bind_key('p', self._swap_normal_direction, overwrite=True)
        self._ortogonal_plane_swap_btn.changed.connect(self._swap_normal_direction)
        # Replace plane layer variable with plane containing initial values
        plane_layer = self._reference_image_combobox.value
        if plane_layer is not None:
            self._add_plane_layer()

        super().__init__(
            widgets=[
                self._reference_image_combobox,
                self._layer_to_be_cut_combobox,
                self._plane_data_combobox,
                self._positive_cut_checkbox,
                self._ortogonal_plane_swap_btn,
                self._run_btn])

    def _add_plane_layer(self):
        '''Add plane layer to viewer'''
        plane_layer = self._reference_image_combobox.value
        self.plane_ortogonal_unity_vector_list = [np.array([0, 1, 0]), np.array([1, 0, 0]), np.array([0, 0, 1])]
        # Get plane layer translation if it exists
        plane_translate = np.asarray(plane_layer.translate)
        plane_position = np.array([plane_layer.data.shape[0] // 2 - 0.5, plane_layer.data.shape[1] // 2 - 0.5, plane_layer.data.shape[2] // 2 - 0.5])
        # Apply translation scaled by layer scale
        plane_position = plane_position + (plane_translate / np.asarray(plane_layer.scale))
        plane_parameters = {
            # z, y, x (intital position in the middle of the image, at the edge of the voxel)
            'position': tuple(plane_position),
            'normal': tuple(self.plane_ortogonal_unity_vector_list[0]),
            'thickness': 1,
        }
        self._plane_layer = self._viewer.add_image(plane_layer.data,
                                                   rendering='average',
                                                   name='plane',
                                                   depiction='plane',
                                                   blending='additive',
                                                   opacity=0.5,
                                                   scale=plane_layer.scale,
                                                   translate=plane_layer.translate,
                                                   gamma=0.4,
                                                   contrast_limits=[0, plane_layer.data.max()],
                                                   plane=plane_parameters
                                                   )

    def _swap_normal_direction(self, viewer=None):
        '''Swap plane normal direction to ortogonal direction'''
        current_normal_unit_verctor = np.asarray(self._plane_layer.plane.normal)
        # If plane normal vector not in list (plane was changed by user), generate new ortogonal vectors
        if not array_allclose_in_list(current_normal_unit_verctor, self.plane_ortogonal_unity_vector_list):
            ortogonal_unit_vector_1 = np.random.randn(3)
            ortogonal_unit_vector_1 -= ortogonal_unit_vector_1.dot(current_normal_unit_verctor) * \
                current_normal_unit_verctor / np.linalg.norm(current_normal_unit_verctor)
            ortogonal_unit_vector_2 = np.cross(current_normal_unit_verctor, ortogonal_unit_vector_1)
            ortogonal_unit_vector_1 /= np.linalg.norm(ortogonal_unit_vector_1)
            ortogonal_unit_vector_2 /= np.linalg.norm(ortogonal_unit_vector_2)
            self.plane_ortogonal_unity_vector_list = [
                current_normal_unit_verctor, ortogonal_unit_vector_1, ortogonal_unit_vector_2]

        # switch to next vector in list
        pos = find_array_allclose_position_in_list(current_normal_unit_verctor, self.plane_ortogonal_unity_vector_list)
        self._plane_layer.plane.normal = tuple(self.plane_ortogonal_unity_vector_list[(pos + 1) % 3])

    def _get_reference_layers(self, combo_box):
        '''Get layers of type image or labels and excludes the plane layer'''
        # Currently accepts only 3D data
        return [layer for layer in self._viewer.layers if isinstance(
            layer, napari.layers.Image) and layer != self._plane_layer and layer.rgb is False and layer.ndim == 3]

    def _get_layers_to_be_cut(self, combo_box):
        '''Get layers of type image or labels and excludes the plane layer'''
        # Currently accepts only 3D data
        return [layer for layer in self._viewer.layers if isinstance(
            layer, self.input_layer_types) and layer != self._plane_layer and 
            (not isinstance(layer, napari.layers.Image) or layer.rgb is False) and layer.ndim == 3]

    def _on_plane_data_source_changed(self, new_value: str):
        '''Update plane data source and plane layer data'''
        self._plane_data_source = new_value
        self._on_image_layer_changed(self._reference_image_combobox.value)

    def _on_image_layer_changed(self, new_layer: napari.layers.Image):
        '''Update plane layer data'''
        if self._plane_layer is None:
            self._add_plane_layer()
        self._update_plane_layer(new_layer)

    def _update_plane_layer(self, layer: napari.layers.Image):
        '''Update plane layer data and a few layout parameters'''
        if self._plane_data_source == 'reference_layer_data':
            self._plane_layer.data = layer.data
            self._plane_layer.scale = layer.scale
            self._plane_layer.contrast_limits = layer.contrast_limits
        elif self._plane_data_source == 'blank':
            self._plane_layer.data = np.ones(layer.data.shape)
            self._plane_layer.scale = layer.scale
            self._plane_layer.contrast_limits = [0, 2]

    def _on_positive_cut_changed(self, new_value: bool):
        '''Update positive cut parameter'''
        self._positive_cut = new_value

    def _on_cut_clicked(self):
        '''Cut image with plane and add new layer to viewer'''
        # Get plane parameters from plane layer
        plane_normal = self._plane_layer.plane.normal
        plane_position = self._plane_layer.plane.position
        # get layer to be cut from chosen value in combobox
        layer_to_be_cut = self._layer_to_be_cut_combobox.value
        output_layer_type = layer_to_be_cut._type_string
        # Call cut function
        image_cut = cut_with_plane(layer_to_be_cut.data, plane_normal, plane_position, self._positive_cut)
        shift = (0, 0, 0)
        # Calculate translation vector
        slices = get_nonzero_slices(image_cut)
        start = [slc.start for slc in slices if slc is not None]
        stop = [slc.stop for slc in slices if slc is not None]
        shift = tuple(start)
        # Crop image
        image_cut = trim_zeros(image_cut)

        # Apply layer translation scaled by layer scaling factor
        output_translate = np.asarray(shift) * np.asarray(layer_to_be_cut.scale)
        # Check if layer to be cut already has translation
        layer_with_translation = not np.array_equal(np.asarray(layer_to_be_cut.translate), np.zeros(layer_to_be_cut.ndim))
        if layer_with_translation:
            # add original layer translation
            output_translate = output_translate + np.asarray(layer_to_be_cut.translate)

        if output_layer_type == 'labels':
            image_cut = relabel_sequential(image_cut)[0]
        self._viewer._add_layer_from_data(
            image_cut,
            meta={
                'name': self._layer_to_be_cut_combobox.value.name + ' cut',
                'scale': layer_to_be_cut.scale,
                'translate': tuple(output_translate),
                'metadata': {'bbox': tuple(start + stop)},
            },
            layer_type=output_layer_type)
