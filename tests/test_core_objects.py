"""Grids, access points, radio maps and the path loss model."""

from __future__ import annotations

import numpy as np
import pytest

from symloc.ap import APError, APSpec, RadioMap, Scenario, uniform_radio_map
from symloc.grid import Grid, GridError
from symloc.pathloss import PathLossError, mean_rssi_dbm, pathloss_radio_map, sample_shadowing


class TestGrid:
    def test_the_sweep_has_inclusive_endpoints(self):
        grid = Grid.from_dbm_sweep(-85.0, -60.0, 0.5)
        assert grid.size == 51
        assert grid.values[0] == -85.0
        assert grid.values[-1] == -60.0
        assert grid.span == (-85.0, -60.0)

    def test_the_step_is_recorded_because_peaks_depend_on_it(self):
        assert Grid.from_dbm_sweep(-85.0, -60.0, 0.25).step == 0.25

    def test_endpoints_out_of_order_are_rejected(self):
        with pytest.raises(GridError, match="must be greater"):
            Grid.from_dbm_sweep(-60.0, -85.0, 0.5)

    def test_a_step_that_does_not_divide_the_span_is_rejected(self):
        """Silently rounding would move an endpoint without saying so."""
        with pytest.raises(GridError, match="does not divide"):
            Grid.from_dbm_sweep(-85.0, -60.0, 0.7)

    def test_a_non_positive_step_is_rejected(self):
        with pytest.raises(GridError, match="positive"):
            Grid.from_dbm_sweep(-85.0, -60.0, 0.0)

    def test_a_single_point_is_not_a_grid(self):
        with pytest.raises(GridError, match="at least two"):
            Grid(values=np.array([1.0]), step=1.0)

    def test_non_finite_values_are_rejected(self):
        with pytest.raises(GridError, match="finite"):
            Grid(values=np.array([0.0, np.nan, 1.0]), step=1.0)

    def test_a_rectangular_floor_tiles_the_area(self):
        grid = Grid.rectangular(4.0, 2.0, 2.0)
        assert grid.size == 3 * 2
        assert grid.coordinates is not None
        assert grid.unit == "index"
        assert grid.axis_label == "reference point index"

    def test_coordinates_must_match_the_point_count(self):
        with pytest.raises(GridError, match="shape"):
            Grid(values=np.arange(3.0), step=1.0, coordinates=np.zeros((2, 2)))

    def test_nearest_index_and_containment(self):
        grid = Grid.from_dbm_sweep(-85.0, -60.0, 0.5)
        assert grid.values[grid.nearest_index(-70.2)] == pytest.approx(-70.0)
        assert grid.contains(-85.0) and grid.contains(-60.0)
        assert not grid.contains(-99.0)
        assert grid.contains(-86.0, tolerance=2.0)


class TestAPSpec:
    def test_an_honest_access_point_reports_its_true_value(self):
        assert APSpec("A", -70.0, 8.0).observed_rssi_dbm == -70.0

    def test_an_attack_is_an_offset_not_a_replacement(self):
        """Attack strength is a parameter, so it can be swept."""
        ap = APSpec("A", -70.0, 8.0, attacked=True, attack_offset_db=-20.0)
        assert ap.observed_rssi_dbm == -90.0
        assert ap.with_attack(-30.0).observed_rssi_dbm == -100.0
        assert ap.without_attack().observed_rssi_dbm == -70.0

    def test_an_offset_is_ignored_unless_the_access_point_is_attacked(self):
        ap = APSpec("A", -70.0, 8.0, attacked=False, attack_offset_db=-20.0)
        assert ap.observed_rssi_dbm == -70.0

    def test_non_positive_sigma_is_rejected(self):
        with pytest.raises(APError, match="strictly positive"):
            APSpec("A", -70.0, 0.0)

    def test_an_empty_name_is_rejected(self):
        with pytest.raises(APError, match="non-empty"):
            APSpec("", -70.0, 8.0)


class TestRadioMap:
    def test_a_uniform_map_is_degenerate_by_construction(self, grid):
        radio_map = uniform_radio_map(grid, ["A", "B", "C"])
        assert radio_map.means_dbm.shape == (3, grid.size)
        assert radio_map.is_degenerate

    def test_a_uniform_map_needs_a_dbm_axis(self):
        floor = Grid.rectangular(4.0, 4.0, 2.0)
        with pytest.raises(APError, match="dBm sweep"):
            uniform_radio_map(floor, ["A"])

    def test_row_count_must_match_the_names(self):
        with pytest.raises(APError, match="names"):
            RadioMap(np.zeros((3, 5)), ("A", "B"))

    def test_non_finite_means_are_rejected(self):
        with pytest.raises(APError, match="finite"):
            RadioMap(np.array([[0.0, np.inf]]), ("A",))


class TestScenario:
    def test_assembly_validates_the_pieces_against_each_other(self, grid):
        aps = [APSpec("A", -70.0, 8.0), APSpec("B", -70.0, 10.0)]
        scenario = Scenario.build("demo", grid, aps)
        assert scenario.n_ap == 2
        assert scenario.observed_rssi_dbm.tolist() == [-70.0, -70.0]
        assert scenario.sigma_db.tolist() == [8.0, 10.0]

    def test_a_radio_map_of_the_wrong_size_is_rejected(self, grid):
        aps = [APSpec("A", -70.0, 8.0)]
        wrong = RadioMap(np.zeros((1, 7)), ("A",))
        with pytest.raises(APError, match="covers"):
            Scenario("demo", grid, tuple(aps), wrong)

    def test_access_point_order_must_match_the_radio_map(self, grid):
        aps = (APSpec("A", -70.0, 8.0), APSpec("B", -70.0, 8.0))
        swapped = RadioMap(np.tile(grid.values, (2, 1)), ("B", "A"))
        with pytest.raises(APError, match="order disagrees"):
            Scenario("demo", grid, aps, swapped)

    def test_a_true_index_off_the_grid_is_rejected(self, grid):
        aps = (APSpec("A", -70.0, 8.0),)
        with pytest.raises(APError, match="outside the grid"):
            Scenario.build("demo", grid, aps, true_index=9999)

    def test_duplicate_access_point_names_are_rejected(self, grid):
        """Every per-access-point report is keyed by name, so a clash loses a row."""
        aps = (APSpec("A", -70.0, 5.0), APSpec("A", -99.0, 5.0), APSpec("B", -70.0, 5.0))
        with pytest.raises(APError, match="must be unique"):
            Scenario.build("demo", grid, aps)

    def test_honest_removes_every_attack(self, three_ap_scenario):
        assert three_ap_scenario.n_attacked == 1
        assert three_ap_scenario.honest().n_attacked == 0

    def test_covered_mean_span_comes_from_the_radio_map(self, grid):
        scenario = Scenario.build("demo", grid, [APSpec("A", -70.0, 8.0)])
        assert scenario.covered_mean_span().tolist() == [[-85.0, -60.0]]


class TestPathLoss:
    def test_the_reference_distance_gives_the_reference_level(self):
        assert mean_rssi_dbm(1.0, reference_rssi_dbm=-40.0, path_loss_exponent=3.0) == -40.0

    def test_a_decade_of_distance_costs_ten_n_decibels(self):
        """The defining property of the log-distance model."""
        for exponent in (2.0, 3.0, 4.0):
            near = mean_rssi_dbm(1.0, -40.0, exponent)
            far = mean_rssi_dbm(10.0, -40.0, exponent)
            assert float(near - far) == pytest.approx(10.0 * exponent)

    def test_distance_is_clipped_so_a_point_under_the_antenna_stays_finite(self):
        value = mean_rssi_dbm(0.0, -40.0, 3.0, min_distance_m=0.5)
        assert np.isfinite(value)
        assert value == pytest.approx(mean_rssi_dbm(0.5, -40.0, 3.0))

    def test_invalid_parameters_are_rejected(self):
        with pytest.raises(PathLossError, match="path_loss_exponent"):
            mean_rssi_dbm(1.0, path_loss_exponent=0.0)
        with pytest.raises(PathLossError, match="non-negative"):
            mean_rssi_dbm(-1.0)

    def test_a_path_loss_map_is_not_degenerate(self):
        """The whole point: each access point has its own mean at each point."""
        floor = Grid.rectangular(10.0, 10.0, 2.0)
        radio_map = pathloss_radio_map(floor, np.array([[0.0, 0.0], [10.0, 10.0]]), ["A", "B"])
        assert radio_map.means_dbm.shape == (2, floor.size)
        assert not radio_map.is_degenerate

    def test_the_mean_falls_off_with_distance_from_its_own_access_point(self):
        floor = Grid.rectangular(10.0, 0.0001, 1.0)
        radio_map = pathloss_radio_map(floor, np.array([[0.0, 0.0]]), ["A"])
        means = radio_map.means_dbm[0]
        assert np.all(np.diff(means[: floor.size // 2]) < 0)

    def test_per_access_point_parameters_broadcast(self):
        floor = Grid.rectangular(4.0, 4.0, 2.0)
        radio_map = pathloss_radio_map(
            floor,
            np.array([[0.0, 0.0], [4.0, 4.0]]),
            ["A", "B"],
            reference_rssi_dbm=np.array([-40.0, -50.0]),
            path_loss_exponent=np.array([2.0, 4.0]),
        )
        # Each access point sits on a grid point, so its strongest mean is the
        # one at the clipped minimum distance rather than at the 1 m reference.
        assert radio_map.means_dbm[0].max() == pytest.approx(float(mean_rssi_dbm(0.0, -40.0, 2.0)))
        assert radio_map.means_dbm[1].max() == pytest.approx(float(mean_rssi_dbm(0.0, -50.0, 4.0)))
        # The steeper exponent loses more over the same 4*sqrt(2) m diagonal.
        # np.ptp(...) rather than ndarray.ptp(), which NumPy 2.0 removed.
        assert np.ptp(radio_map.means_dbm[1]) > np.ptp(radio_map.means_dbm[0])

    def test_a_map_needs_a_grid_with_coordinates(self, grid):
        with pytest.raises(PathLossError, match="coordinates"):
            pathloss_radio_map(grid, np.array([[0.0, 0.0]]), ["A"])

    def test_shadowing_samples_have_the_requested_statistics(self, rng):
        draws = sample_shadowing(np.array([-70.0, -80.0]), np.array([4.0, 9.0]), rng, size=40000)
        assert draws.shape == (40000, 2)
        assert np.allclose(draws.mean(axis=0), [-70.0, -80.0], atol=0.2)
        assert np.allclose(draws.std(axis=0), [4.0, 9.0], rtol=0.05)

    def test_shadowing_rejects_a_non_positive_sigma(self, rng):
        with pytest.raises(PathLossError, match="strictly positive"):
            sample_shadowing(np.array([-70.0]), np.array([0.0]), rng)
