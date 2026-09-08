# Third-party notices

F2 HistoGNN depends on independently licensed software. Consult each installed distribution for its complete license text.

| Package | Project | License family |
|---|---|---|
| NumPy | https://numpy.org/ | BSD-3-Clause plus separately licensed bundled components |
| PyTorch | https://pytorch.org/ | BSD-3-Clause |
| PyTorch Geometric | https://pyg.org/ | MIT |
| pytest (test only) | https://pytest.org/ | MIT |
| OpenSlide Python | https://openslide.org/ | LGPL-2.1 |
| gdown | https://github.com/wkentaro/gdown | MIT |
| docopt | https://github.com/docopt/docopt | MIT |

The optional real-data smoke downloads, but does not redistribute, HoVer-Net
code pinned to revision `67e2ce5e3f1a64a2ece77ad1c24233653a9e0901` and a
PanNuke-trained checkpoint pinned by SHA-256. The HoVer-Net code is MIT; the
PanNuke-derived checkpoint is governed by CC BY-NC-SA 4.0 and is restricted to
the approved non-commercial research workflow. Raw TCGA slide data and model
weights are never part of this repository.

F2 HistoGNN itself is distributed under the MIT License in `LICENSE`. Third-party components remain governed by their own licenses.
