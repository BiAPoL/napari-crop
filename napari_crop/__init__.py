try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._dock_widgets import CutWithPlane
from ._function import crop_region, cut_with_plane, draw_fixed_shapes
