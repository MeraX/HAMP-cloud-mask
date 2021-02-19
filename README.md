# HAMP-cloud-mask
HALO Microwave Package (HAMP) Cloud Mask Products

This Python generates cloud mask using the HAMP cloud radar and microwave
radiometer.

![Example plot](https://atmos.meteo.uni-koeln.de/~mjacob/HAMP-cloud-mask_example.png)

## Requirements
Python >= 3.6 and following python packages:

  * `numpy` (tested with version 1.18.1 and 1.19.0)
  * `scipy` (tested with version 1.2.0 and 1.5.1)
  * `xarray` (tested with version 0.14.1 and 0.15.1)
  * `matplotlib` (tested with version 3.1.2 and 3.2.2)
  * `mpl_toolkits` (probably part of matplotlib)
  * `cv2` (tested with version 4.4.0)

For `plot_hamp_wales_specmacs.py`:

  * `eurec4a` (tested with version 0.0.2, need the additional packages `pydap` and `intake-xarray`	)
  * `pydap` (tested with version 3.2.2)
  * `intake-xarray` (tested with version 0.6.0)


## License
HAMP-cloud-mask is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation version 3

HAMP-cloud-mask is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with gr-air-modes; see the file COPYING.  If not, write to
the Free Software Foundation, Inc., 51 Franklin Street,
Boston, MA 02110-1301, USA.
