#!/usr/bin/env python3
"""Auxiliary routines.."""
import re
from contextlib import suppress

import numpy as np

from .constants import _PROJ_ELLPS2PARAMS


def set_ellipsoid_explicitly_in_proj_params(projparams):
    """Set ellps params explicitly to prevent them from being overriden.

    See <https://proj.org/usage/ellipsoids.html>.

    """
    if not isinstance(projparams, (str, dict)):
        return projparams

    ellipsoid = _get_value_from_projparams("ellps", projparams=projparams)
    if ellipsoid:
        ellipsoid_params = _PROJ_ELLPS2PARAMS[ellipsoid]
        for param, value in ellipsoid_params.items():
            if _get_value_from_projparams(param, projparams=projparams):
                continue
            try:
                projparams += f" +{param}={value}"
            except TypeError:
                projparams[param] = value

    return projparams


def get_corners_xy(grid_with_proj):
    if grid_with_proj.anchor_points_lonlat.keys() == {"center"}:
        center_xy = grid_with_proj.lonlat2xy(
            *grid_with_proj.anchor_points_lonlat["center"]
        )
        xmin, ymin = center_xy - 0.5 * grid_with_proj.shape * grid_with_proj.cell_sizes
        xmax, ymax = center_xy + 0.5 * grid_with_proj.shape * grid_with_proj.cell_sizes
        return dict(
            lower_left=np.array((xmin, ymin)),
            lower_right=np.array((xmax, ymin)),
            upper_right=np.array((xmax, ymax)),
            upper_left=np.array((xmin, ymax)),
        )

    expected_pts = {"lower_left", "lower_right", "upper_right", "upper_left"}
    passed_pts = grid_with_proj.anchor_points_lonlat.keys()
    if passed_pts != expected_pts:
        expected_pts = {f"'{name}'" for name in expected_pts}
        msg = f"Invalid anchor points: {', '.join(passed_pts-expected_pts)}. Expected "
        msg += f"either 'center' or all of {', '.join(expected_pts)}."
        raise ValueError(msg)

    return {
        name: grid_with_proj.lonlat2xy(*lonlat)
        for name, lonlat in grid_with_proj.anchor_points_lonlat.items()
    }


def validated_corners_xy(corners_xy, cell_sizes, shape):
    _check_that_corners_form_rectangle_in_proj(corners_xy)
    _check_that_cell_sizes_and_grid_corners_are_consistent(corners_xy, cell_sizes, shape)
    return corners_xy


def validate_shape(_, __, value):
    if len(value) != 2:
        raise ValueError(f"Invalid shape {value}: it must be two-dimensional.")
    if np.any(value < 1):
        raise ValueError(f"Invalid shape {value}: values must be greater than zero.")


def _get_value_from_projparams(param_name, projparams):
    if isinstance(projparams, dict):
        return projparams.get(param_name)

    patt = rf"(^|\s)\+\s*{re.escape(param_name)}\s*=\s*(?P<value>\S*)"
    with suppress(AttributeError):
        return re.search(patt, projparams).group("value")
    return None


def _angle_between_vectors(vec1, vec2):
    vec1 /= np.linalg.norm(vec1)
    vec2 /= np.linalg.norm(vec2)
    return np.arccos(np.dot(vec1, vec2))


def _angle_between_points(pt1, pt2, pt3):
    return _angle_between_vectors(pt3 - pt2, pt1 - pt2)


def _validate_grid_cell_sizes(passed_value, expected_value):
    # Check that input delta_x agree with calculated from corners
    if not np.allclose(passed_value, expected_value):
        msg = f"Cell sizes calculated from grid corners ({expected_value} m) "
        msg += f"differ from the ones passed ({passed_value} m). "
        msg += "Please check your grid, projection and corner parameters."
        raise ValueError(msg)


def _check_that_cell_sizes_and_grid_corners_are_consistent(corners_xy, cell_sizes, shape):
    diagonal = corners_xy["upper_right"] - corners_xy["lower_left"]
    expected_value = diagonal / shape
    _validate_grid_cell_sizes(passed_value=cell_sizes, expected_value=expected_value)


def _check_that_corners_form_rectangle_in_proj(corners_xy):
    corners_order = ["lower_left", "lower_right", "upper_right", "upper_left"]
    all_corners = [corners_xy[name] for name in corners_order]
    for i_angle in range(4):
        i_corners = (i_corner % 4 for i_corner in range(i_angle, i_angle + 3))
        selected_corners = [all_corners[i] for i in i_corners]
        angle = _angle_between_points(*selected_corners)
        if not np.isclose(angle, np.pi / 2.0):
            raise ValueError("Corners don't seem to form a projected rectangle")
