#!/usr/bin/env python3
"""Implement a grid with projection."""
import logging

import numpy as np
import pyproj

logger = logging.getLogger(__name__)


class GridWithProjection:
    """Configs for a grid axis."""

    def __init__(
        self,
        projparams,
        lower_left,
        upper_left,
        lower_right,
        upper_right,
        xsize,
        ysize,
        xscale,
        yscale,
    ):
        self.transformer = pyproj.Proj(projparams, preserve_units=False)

        # Check that grid is orthogonal
        def angle_between_vectors(vec1, vec2):
            vec1 /= np.linalg.norm(vec1)
            vec2 /= np.linalg.norm(vec2)
            return np.rad2deg(np.arccos(np.dot(vec1, vec2)))

        def angle_between_lonlat_coordinates(pt1, pt2, pt3):
            segment_1 = np.array(self.lonlat2xy(*pt3)) - np.array(self.lonlat2xy(*pt2))
            segment_2 = np.array(self.lonlat2xy(*pt1)) - np.array(self.lonlat2xy(*pt2))
            return angle_between_vectors(segment_1, segment_2)

        angle_1 = angle_between_lonlat_coordinates(upper_left, lower_left, lower_right)
        angle_2 = angle_between_lonlat_coordinates(lower_left, lower_right, upper_right)
        angle_3 = angle_between_lonlat_coordinates(lower_right, upper_right, upper_left)
        angle_4 = angle_between_lonlat_coordinates(upper_right, upper_left, lower_left)
        for angle in [angle_1, angle_2, angle_3, angle_4]:
            if not np.isclose(angle, 90):
                raise ValueError("Corners don't seem to form a rectangle")

        # Check that input delta_x agree with calculated from corners
        lower_left_xy = np.array(self.lonlat2xy(*lower_left))
        upper_right_xy = np.array(self.lonlat2xy(*upper_right))
        pixel_sizes_from_corners = (upper_right_xy - lower_left_xy) / np.array(
            [xsize, ysize]
        )
        passed_pixel_sizes = np.array((xscale, yscale))
        if not np.allclose(passed_pixel_sizes, pixel_sizes_from_corners):
            msg = f"Pixel sizes calculated from corners {pixel_sizes_from_corners} "
            msg += f"differ from passed {passed_pixel_sizes}. "
            msg += "Using the calculated ones."
            logger.warning(msg)

        self.corners_lonlat = {
            "lower_left": lower_left,
            "upper_left": upper_left,
            "lower_right": lower_right,
            "upper_right": upper_right,
        }
        self.xmin, self.ymin = lower_left_xy
        self.dx, self.dy = pixel_sizes_from_corners
        self.nx = xsize
        self.ny = ysize

    @property
    def corners_xy(self):
        return {
            corner_name: self.lonlat2xy(*corner_xy)
            for corner_name, corner_xy in self.corners_lonlat.items()
        }

    @property
    def xmax(self):
        return self.xmin + self.nx * self.dx

    @property
    def ymax(self):
        return self.ymin + self.ny * self.dy

    def lonlat2xy(self, lon, lat):
        """Convert (lon, lat), in degrees, into projected (x, y) in meters."""
        return self.transformer(longitude=lon, latitude=lat)

    def xy2lonlat(self, x, y):
        """Convert projected (x, y), in meters, into (lon, lat) in degrees."""
        lon, lat = self.transformer(x, y, inverse=True)
        return lon, lat

    def ij2xy(self, i, j):
        x = self.xmin + i * self.dx
        y = self.ymin + j * self.dy
        return x, y

    def xy2ij(self, x, y):
        i = int(np.floor((x - self.xmin) / self.dx))
        j = int(np.floor((y - self.ymin) / self.dy))

    def ij2lonlat(self, i, j):
        x, y = self.ij2xy(i=i, j=j)
        return self.xy2lonlat(x=x, y=y)

    def lonlat2ij(self, lon, lat):
        x, y = self.lonlat2xy(lon=lon, lat=lat)
        return self.xy2ij(x=x, y=y)

if __name__=="__main__":
    grid =
