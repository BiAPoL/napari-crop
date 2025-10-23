# napari-crop

[![License](https://img.shields.io/pypi/l/napari-crop.svg?color=green)](https://github.com/BiAPoL/napari-crop/raw/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-crop.svg?color=green)](https://pypi.org/project/napari-crop)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-crop.svg?color=green)](https://python.org)
[![tests](https://github.com/BiAPoL/napari-crop/workflows/tests/badge.svg)](https://github.com/BiAPoL/napari-crop/actions)
[![codecov](https://codecov.io/gh/BiAPoL/napari-crop/branch/main/graph/badge.svg)](https://codecov.io/gh/BiAPoL/napari-crop)
[![DOI](https://zenodo.org/badge/419822240.svg)](https://zenodo.org/badge/latestdoi/419822240)

🚨🚨🚨 **Call for maintainers** 🚨🚨🚨

If you are interested in further developing it, please get in touch or create a fork.

Crop regions in napari manually

![](https://github.com/BiAPoL/napari-crop/raw/main/images/screencast.gif)

Crop in any dimension

![](https://github.com/BiAPoL/napari-crop/raw/main/images/side_crop.gif)

Cut a volume using a plane

*Note:*

*- this functionality currently only works with 3D data*

*- to drag the plane, hold the 'SHIFT' key from your keyboard*

![](https://github.com/BiAPoL/napari-crop/raw/main/images/napari_crop_cut_with_plane_demo.gif)

Draw shapes of fixed size at given coordinates for later cropping

![](https://github.com/BiAPoL/napari-crop/raw/main/images/napari_crop_draw_shapes.gif)

## Usage
Create a new shapes layer to annotate the region you would like to crop:

![](https://github.com/BiAPoL/napari-crop/raw/main/images/shapes.png)

Use the rectangle tool to annotate a region. Start the `crop` tool from the `Plugins > napari-crop > Crop region` or, if available, `Tools > Utilities > Crop region` menu. 
Click the `Run` button to crop the region.

![](https://github.com/BiAPoL/napari-crop/raw/main/images/draw_rectangle.png)

You can also use the `Select shapes` tool to move the rectangle to a new place and crop another region by clicking on `Run`.

![](https://github.com/BiAPoL/napari-crop/raw/main/images/move_rectangle.png)

Hint: You can also use the [napari-tabu](https://www.napari-hub.org/plugins/napari-tabu) plugin to send all your cropped images to a new napari window.

![](https://github.com/BiAPoL/napari-crop/raw/main/images/new_window.gif)

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using with [@napari]'s [cookiecutter-napari-plugin] template.

## Installation

You can install `napari-crop` via [pip]:

    pip install napari-crop

## Contributing

Contributions are very welcome. 

## License

Distributed under the terms of the [BSD-3] license,
"napari-crop" is free and open source software

## Issues

If you encounter any problems, please create a thread on [image.sc] along with a detailed description and tag [@haesleinhuepf].

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin
[file an issue]: https://github.com/BiAPoL/napari-crop/issues
[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
[image.sc]: https://image.sc
[@haesleinhuepf]: https://twitter.com/haesleinhuepf

