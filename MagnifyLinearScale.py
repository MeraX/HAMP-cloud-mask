import numpy as np
import matplotlib as mpl
from matplotlib.scale import ScaleBase, register_scale
from matplotlib.transforms import Transform
from matplotlib.ticker import (
    NullFormatter, ScalarFormatter,
    NullLocator, AutoLocator, AutoMinorLocator,
)


class MagnifyLinearTransform(Transform):
    input_dims = output_dims = 1
    is_separable = True

    def __init__(self, magnify_segments):
        super().__init__(self)
        self.magnify_segments = magnify_segments
        self.inverse_segments = [
            (
                self.transform_non_affine(np.array([p0]))[0],
                self.transform_non_affine(np.array([p1]))[0],
                1.0/scale
            )
            for (p0, p1, scale) in self.magnify_segments
        ]

    def transform_non_affine(self, a):
        result = a.copy().astype(float)
        with np.errstate(invalid="ignore"):
            for (p0, p1, scale) in self.magnify_segments:
                mask = np.logical_and(p0<a, a<=p1)
                result[mask] += (a[mask] - p0) * (scale - 1)
                result[p1<a] += (p1 - p0) * (scale - 1)
        return result

    def inverted(self):
        return MagnifyLinearTransform(self.inverse_segments)

class MagnifyLinearScale(ScaleBase):
    name = "magnifylinear"

    def __init__(self, axis, magnify_segments):
        """
        A MPL ax scale that magnifies given segments

        Parameters
        ----------
        axis : {matplotlib axis}
            The axis handle
        magnify_segments : {[tuples of 3-tuples]}
            Format: [(p0, p1, scale), ...]
            Magnify each segment from p0 to p1 by given scale factor.
            p0 <= p1

        Example
        -------
        ax.set_yscale('magnifylinear', magnify_segments=[[-20., 20., 10.]])

        """
        for (p0, p1, scale) in magnify_segments:
            if p0 > p1:
                raise ValueError("p0 must be less or equal p1")
            if scale <= 0:
                raise ValueError("scale must be positive")

        super().__init__(axis)
        self.magnify_segments = magnify_segments

    def get_transform(self):
        return MagnifyLinearTransform(self.magnify_segments)

    def set_default_locators_and_formatters(self, axis):
        # docstring inherited
        axis.set_major_locator(AutoLocator())
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_formatter(NullFormatter())
        # update the minor locator for x and y axis based on rcParams
        if (axis.axis_name == 'x' and mpl.rcParams['xtick.minor.visible'] or
                axis.axis_name == 'y' and mpl.rcParams['ytick.minor.visible']):
            axis.set_minor_locator(AutoMinorLocator())
        else:
            axis.set_minor_locator(NullLocator())

register_scale(MagnifyLinearScale)
