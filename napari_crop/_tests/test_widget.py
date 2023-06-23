from napari_crop._dock_widgets import CutWithPlane
import numpy as np

volume = np.random.random((10, 10, 10))
labels = np.zeros((10, 10, 10), dtype=int)
labels[1:4, 1:4, 1:4] = 1
labels[6:9, 6:9, 6:9] = 2
labels[1:9, 4:6, 4:6] = 3


def test_cut_with_plane_widget(make_napari_viewer):
    """Test that widget creates new layers and the cut output has correct number of labels."""

    viewer = make_napari_viewer(ndisplay=3)

    img_layer = viewer.add_image(volume)
    labels_layer = viewer.add_labels(labels)

    widget = CutWithPlane(viewer)

    # Assert plane layer creation upon widget creation
    assert len(viewer.layers) == 3

    # Set labels layer to be cut
    widget._layer_to_be_cut_combobox.value = labels_layer
    # Cut with plane
    widget._on_cut_clicked()

    assert len(viewer.layers) == 4
    assert viewer.layers[-1].data.max() == 2  # one label was cut off
