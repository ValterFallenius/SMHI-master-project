#!/usr/bin/env python3
"""Implement a grid with projection."""
import functools
import attrs
import numpy as np
import pyproj

from .grid_with_projection_helpers import (
    get_corners_xy,
    set_ellipsoid_explicitly_in_proj_params,
    validate_shape,
    validated_corners_xy,
)


@attrs.frozen(eq=False, kw_only=True)
class Grid2D:
    """A 2D, (x, y) grid."""

    shape = attrs.field(converter=np.array, validator=validate_shape)
    cell_sizes = attrs.field(converter=np.array)
    cartesian_origin = attrs.field(default=(0.0, 0.0), converter=np.array)

    @property
    def nx(self):
        return self.shape[0]

    @property
    def ny(self):
        return self.shape[1]

    @property
    def dx(self):
        return self.cell_sizes[0]

    @property
    def dy(self):
        return self.cell_sizes[1]

    @property
    def n_cells(self):
        return np.prod(self.shape)

    @property
    def xmin(self):
        return self.cartesian_origin[0]

    @property
    def ymin(self):
        return self.cartesian_origin[1]

    @property
    @functools.lru_cache()
    def xmax(self):
        return self.xmin + self.nx * self.dx

    @property
    @functools.lru_cache()
    def ymax(self):
        return self.ymin + self.ny * self.dy

    @property
    @functools.lru_cache()
    def center_xy(self):
        return 0.5 * np.array((self.xmin + self.xmax, self.ymin + self.ymax))

    def ij2xy(self, i, j):
        x = self.xmin + i * self.dx
        y = self.ymin + j * self.dy
        return np.asarray((x, y))

    def xy2ij(self, x, y):
        i = (x - self.xmin) / self.dx
        j = (y - self.ymin) / self.dy
        return np.round(np.asarray((i, j)), 6).astype(int)


@attrs.frozen(eq=False, kw_only=True)
class GridWithProjection(Grid2D):
    """A 2D grid with (x, y) <--> (lon, lat) projection capability."""

    projparams = attrs.field(converter=set_ellipsoid_explicitly_in_proj_params)
    anchor_points_lonlat = attrs.field()
    cartesian_origin = attrs.field(init=False, converter=np.array)

    @cartesian_origin.default
    def _cartesian_origin(self):
        return self._corners_xy["lower_left"]

    @property
    @functools.lru_cache()
    def center_lonlat(self):
        return self.xy2lonlat(*self.center_xy)

    @property
    @functools.lru_cache()
    def transformer(self):
        return pyproj.Proj(self.projparams, preserve_units=False)

    def lonlat2xy(self, lon, lat):
        """Convert (lon, lat), in degrees, into projected (x, y) in meters."""
        return np.asarray(self.transformer(longitude=lon, latitude=lat))

    def xy2lonlat(self, x, y):
        """Convert projected (x, y), in meters, into (lon, lat) in degrees."""
        return np.asarray(self.transformer(x, y, inverse=True))

    def lonlat2ij(self, lon, lat):
        x, y = self.lonlat2xy(lon=lon, lat=lat)
        return self.xy2ij(x=x, y=y)

    def ij2lonlat(self, i, j):
        x, y = self.ij2xy(i=i, j=j)
        return self.xy2lonlat(x=x, y=y)

    @property
    @functools.lru_cache()
    def _corners_xy(self):
        return validated_corners_xy(get_corners_xy(self), self.cell_sizes, self.shape)

    @property
    @functools.lru_cache()
    def _corners_lonlat(self):
        return {name: self.xy2lonlat(*xy) for name, xy in self._corners_xy.items()}

if __name__=="__main__":
    where_dict = {
        'LL_lat': 53.987947379235436,
        'LL_lon': 9.25569438102197,
        'LR_lat': 53.706519377463586,
        'LR_lon': 22.761914608556413,
        'UL_lat': 69.80759428237813,
        'UL_lon': 5.323837778285936,
        'UR_lat': 69.2640415703203,
        'UR_lon': 29.82199602636603,
        'projdef': b'+proj=stere +ellps=bessel +lat_0=90 +lon_0=14 +lat_ts=60 +datum=WGS84',
        'xscale': 2000.0,
        'xsize': 458,
        'yscale': 2000.0,
        'ysize': 881
        }
    grid = GridWithProjection(
        projparams="+proj=stere +ellps=bessel +lat_0=90 +lon_0=14 +lat_ts=60 +datum=WGS84",
        shape=(458, 881),
        cell_sizes=(2000.0, 2000.0),
        anchor_points_lonlat={
            "lower_left": (9.25569438102197, 53.987947379235436),
            "lower_right": (22.761914608556413, 53.706519377463586),
            "upper_right": (29.82199602636603, 69.2640415703203),
            "upper_left": (5.323837778285936, 69.80759428237813),
        },
    )
    for x in range(881):
        for y in range(458):
            latlon = grid.ij2latlon(x,y)
            print(latlon)
