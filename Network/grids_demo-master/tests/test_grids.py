#!/usr/bin/env python3
"""Unit tests for the properties of the GridWithProjection object."""
import numpy as np
import pandas as pd
import pytest

from grids.grid_with_projection import GridWithProjection


@pytest.fixture()
def grid():
    return GridWithProjection(
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


@pytest.fixture()
def random_lonlat_coords(n_pts=100):
    return list(zip(_random_longitude(n_pts), _random_latitude(n_pts)))


@pytest.fixture()
def random_xy_coords(grid, n_pts=100):
    random_x = np.random.uniform(grid.xmin, grid.xmax, n_pts)
    random_y = np.random.uniform(grid.ymin, grid.ymax, n_pts)
    return list(zip(random_x, random_y))


@pytest.fixture()
def random_ij_coords(grid, n_pts=100):
    random_i = np.random.randint(0, grid.nx, n_pts)
    random_j = np.random.randint(0, grid.ny, n_pts)
    return list(zip(random_i, random_j))


class TestGridWithProjection:
    def test_xmax(self, grid):
        xmax1 = grid._corners_xy["lower_right"][0]
        xmax2 = grid._corners_xy["upper_right"][0]
        assert np.isclose(xmax1, grid.xmax)
        assert np.isclose(xmax2, grid.xmax)
        assert np.allclose(grid.xmax, grid.ij2xy(grid.nx, 0)[0])
        assert np.allclose(grid.xmax, grid.ij2xy(grid.nx, grid.ny)[0])

    def test_ymax(self, grid):
        ymax1 = grid._corners_xy["upper_left"][1]
        ymax2 = grid._corners_xy["upper_right"][1]
        assert np.isclose(ymax1, grid.ymax)
        assert np.isclose(ymax2, grid.ymax)
        assert np.allclose(grid.ymax, grid.ij2xy(0, grid.ny)[1])
        assert np.allclose(grid.ymax, grid.ij2xy(grid.nx, grid.ny)[1])

    def test_proj_string(self, grid):
        """Assert that ellps params are added to original proj string definition."""
        assert (
            grid.transformer.to_proj4()
            == "+proj=stere +lat_0=90 +lat_ts=60 +lon_0=14 +x_0=0 +y_0=0 +ellps=bessel +units=m +no_defs"
        )

    def test_lonlat2xy_and_xy2lonlat_are_consistent(self, grid, random_lonlat_coords):
        for lonlat in random_lonlat_coords:
            xy = grid.lonlat2xy(*lonlat)
            new_lonlat = grid.xy2lonlat(*xy)
            assert np.allclose(new_lonlat, lonlat)

    def test_vectorised_lonlat2xy(self, grid, random_lonlat_coords):
        lonlat_df = pd.DataFrame(columns=["lon", "lat"], data=random_lonlat_coords)
        xy_vec = grid.lonlat2xy(lonlat_df["lon"], lonlat_df["lat"])
        xy_df = pd.DataFrame({"x": xy_vec[0], "y": xy_vec[1]})
        expected_xy_df = pd.DataFrame(
            columns=["x", "y"],
            data=(grid.lonlat2xy(*lonlat) for lonlat in random_lonlat_coords),
        )
        assert np.allclose(xy_df, expected_xy_df)

    def test_vectorised_xy2lonlat(self, grid, random_xy_coords):
        xy_df = pd.DataFrame(columns=["x", "y"], data=random_xy_coords)
        lonlat_vec = grid.xy2lonlat(xy_df["x"], xy_df["y"])
        lonlat_df = pd.DataFrame({"lon": lonlat_vec[0], "lat": lonlat_vec[1]})
        expected_lonlat_df = pd.DataFrame(
            columns=["lon", "lat"],
            data=(grid.xy2lonlat(*xy) for xy in random_xy_coords),
        )
        assert np.allclose(lonlat_df, expected_lonlat_df)

    def test_xy2ij(self, grid):
        xy = grid._corners_xy["lower_left"]
        ij = grid.xy2ij(*xy)
        assert np.alltrue(ij == (0, 0))

        xy = grid._corners_xy["lower_right"]
        ij = grid.xy2ij(*xy)
        assert np.alltrue(ij == (grid.nx, 0))

        xy = grid._corners_xy["upper_right"]
        ij = grid.xy2ij(*xy)
        assert np.alltrue(ij == (grid.nx, grid.ny))

        xy = grid._corners_xy["upper_left"]
        ij = grid.xy2ij(*xy)
        assert np.alltrue(ij == (0, grid.ny))

    def test_ij2xy(self, grid):
        xy = grid.ij2xy(0, 0)
        assert np.allclose(xy, grid._corners_xy["lower_left"])

        xy = grid.ij2xy(grid.nx, 0)
        assert np.allclose(xy, grid._corners_xy["lower_right"])

        xy = grid.ij2xy(grid.nx, grid.ny)
        assert np.allclose(xy, grid._corners_xy["upper_right"])

        xy = grid.ij2xy(0, grid.ny)
        assert np.allclose(xy, grid._corners_xy["upper_left"])

    def test_xy2ij_and_ij2xy_are_consistent(self, grid, random_xy_coords):
        for xy in random_xy_coords:
            ij = grid.xy2ij(*xy)
            nearest_grid_point_xy = grid.ij2xy(*ij)
            new_ij = grid.xy2ij(*nearest_grid_point_xy)
            assert np.alltrue(new_ij == ij)

    def test_vectorised_xy2ij(self, grid, random_xy_coords):
        xy_df = pd.DataFrame(columns=["x", "y"], data=random_xy_coords)
        ij_vec = grid.xy2ij(xy_df["x"], xy_df["y"])
        ij_df = pd.DataFrame({"i": ij_vec[0], "j": ij_vec[1]})
        expected_ij_df = pd.DataFrame(
            columns=["i", "j"], data=(grid.xy2ij(*xy) for xy in random_xy_coords)
        )
        assert np.all(np.equal(ij_df, expected_ij_df))

    def test_vectorised_ij2xy(self, grid, random_ij_coords):
        ij_df = pd.DataFrame(columns=["i", "j"], data=random_ij_coords)
        xy_vec = grid.ij2xy(ij_df["i"], ij_df["j"])
        xy_df = pd.DataFrame({"x": xy_vec[0], "y": xy_vec[1]})
        expected_xy_df = pd.DataFrame(
            columns=["x", "y"], data=(grid.ij2xy(*ij) for ij in random_ij_coords)
        )
        assert np.allclose(xy_df, expected_xy_df)

    def test_lonlat2ij(self, grid):
        lonlat = grid._corners_lonlat["lower_left"]
        ij = grid.lonlat2ij(*lonlat)
        assert np.alltrue(ij == (0, 0))

        lonlat = grid._corners_lonlat["lower_right"]
        ij = grid.lonlat2ij(*lonlat)
        assert np.alltrue(ij == (grid.nx, 0))

        lonlat = grid._corners_lonlat["upper_right"]
        ij = grid.lonlat2ij(*lonlat)
        assert np.alltrue(ij == (grid.nx, grid.ny))

        lonlat = grid._corners_lonlat["upper_left"]
        ij = grid.lonlat2ij(*lonlat)
        assert np.alltrue(ij == (0, grid.ny))

    def test_ij2lonlat(self, grid):
        lonlat = grid.ij2lonlat(0, 0)
        assert np.allclose(lonlat, grid._corners_lonlat["lower_left"])

        lonlat = grid.ij2lonlat(grid.nx, 0)
        assert np.allclose(lonlat, grid._corners_lonlat["lower_right"])

        lonlat = grid.ij2lonlat(grid.nx, grid.ny)
        assert np.allclose(lonlat, grid._corners_lonlat["upper_right"])

        lonlat = grid.ij2lonlat(0, grid.ny)
        assert np.allclose(lonlat, grid._corners_lonlat["upper_left"])

    def test_lonlat2ij_and_ij2lonlat_are_consistent(self, grid, random_lonlat_coords):
        for lonlat in random_lonlat_coords:
            ij = grid.lonlat2ij(*lonlat)
            nearest_grid_point_lonlat = grid.ij2lonlat(*ij)
            new_ij = grid.lonlat2ij(*nearest_grid_point_lonlat)
            assert np.alltrue(new_ij == ij)

    def test_vectorised_lonlat2ij(self, grid, random_lonlat_coords):
        lonlat_df = pd.DataFrame(columns=["lon", "lat"], data=random_lonlat_coords)
        ij_vec = grid.lonlat2ij(lonlat_df["lon"], lonlat_df["lat"])
        ij_df = pd.DataFrame({"i": ij_vec[0], "j": ij_vec[1]})
        expected_ij_df = pd.DataFrame(
            columns=["i", "j"],
            data=(grid.lonlat2ij(*lonlat) for lonlat in random_lonlat_coords),
        )
        assert np.all(np.equal(ij_df, expected_ij_df))

    def test_vectorised_ij2lonlat(self, grid, random_ij_coords):
        ij_df = pd.DataFrame(columns=["i", "j"], data=random_ij_coords)
        lonlat_vec = grid.ij2lonlat(ij_df["i"], ij_df["j"])
        lonlat_df = pd.DataFrame({"lon": lonlat_vec[0], "lat": lonlat_vec[1]})
        expected_lonlat_df = pd.DataFrame(
            columns=["lon", "lat"],
            data=(grid.ij2lonlat(*ij) for ij in random_ij_coords),
        )
        assert np.allclose(lonlat_df, expected_lonlat_df)


def _random_latitude(n=1):
    """Return n random latitudes."""
    lats = np.random.uniform(-90, 90, size=n)
    if len(lats) == 1:
        return lats[0]
    return lats


def _random_longitude(n=1):
    """Return n random longitudes."""
    lons = np.random.uniform(-180, 180, size=n)
    if len(lons) == 1:
        return lons[0]
    return lons
