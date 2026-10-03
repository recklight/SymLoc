"""Configuration validation: a bad setting must fail loudly, not silently.

Every constant the original kept at the top of a script is a configuration
entry here, so the validation is what stands between a typo and a plausible
looking but meaningless result.
"""

from __future__ import annotations

import pytest
import yaml

from symloc.config import (
    APConfig,
    Config,
    ConfigError,
    DetectionConfig,
    GridConfig,
    MonteCarloConfig,
    NormalizeConfig,
    OutputConfig,
    RadioMapConfig,
    ScenarioConfig,
    SweepConfig,
)


def test_defaults_are_valid_and_build_a_scenario():
    config = Config()
    scenario = config.build_scenario()
    assert scenario.n_ap == 3
    assert scenario.n_attacked == 1


def test_zero_or_negative_sigma_is_rejected():
    """A zero standard deviation would collapse the likelihood to a delta."""
    with pytest.raises(ConfigError, match="strictly positive"):
        APConfig("AP1", -70.0, 0.0)
    with pytest.raises(ConfigError, match="strictly positive"):
        APConfig("AP1", -70.0, -3.0)


def test_grid_endpoints_must_be_ordered_and_the_step_positive():
    with pytest.raises(ConfigError, match="must exceed"):
        GridConfig(low_dbm=-60.0, high_dbm=-85.0)
    with pytest.raises(ConfigError, match="strictly positive"):
        GridConfig(step_db=0.0)


def test_m_must_leave_at_least_one_access_point():
    """M ranges over 0..N-1: M=N would discard every access point."""
    aps = (APConfig("A"), APConfig("B"), APConfig("C"))
    with pytest.raises(ConfigError, match="out of range"):
        ScenarioConfig(aps=aps, m_values=(3,), attacked_indices=(2,))
    ok = ScenarioConfig(aps=aps, m_values=(0, 1, 2), attacked_indices=(2,))
    assert ok.m_values == (0, 1, 2)


def test_empty_m_values_expands_to_every_valid_order():
    scenario = ScenarioConfig(aps=(APConfig("A"), APConfig("B"), APConfig("C")))
    assert scenario.m_values == (0, 1, 2)


def test_attacked_index_outside_the_access_point_list_is_rejected():
    aps = (APConfig("A"), APConfig("B"))
    with pytest.raises(ConfigError, match="outside"):
        ScenarioConfig(aps=aps, attacked_indices=(5,))
    with pytest.raises(ConfigError, match="duplicates"):
        ScenarioConfig(aps=aps, attacked_indices=(1, 1))


def test_duplicate_access_point_names_are_rejected():
    with pytest.raises(ConfigError, match="duplicate"):
        ScenarioConfig(aps=(APConfig("A"), APConfig("A")), attacked_indices=())


def test_unknown_combiner_and_normalisation_mode_are_rejected():
    with pytest.raises(ConfigError, match="not supported"):
        ScenarioConfig(combiner="nonsense", attacked_indices=())
    with pytest.raises(ConfigError, match="not supported"):
        NormalizeConfig(mode="nonsense")


def test_pathloss_radio_map_requires_a_position_for_every_access_point():
    """A path loss map cannot place an access point that has no coordinates."""
    scenario = ScenarioConfig(
        aps=(APConfig("A", position_m=(0.0, 0.0)), APConfig("B")),
        attacked_indices=(1,),
    )
    with pytest.raises(ConfigError, match="position_m"):
        Config(radio_map=RadioMapConfig(kind="pathloss"), scenario=scenario)


def test_detection_settings_must_allow_every_fold_a_sample_of_each_class():
    with pytest.raises(ConfigError, match="at least"):
        DetectionConfig(n_splits=1)
    with pytest.raises(ConfigError, match="n_trials_per_class"):
        DetectionConfig(n_splits=10, n_trials_per_class=4)


def test_a_zero_attack_offset_would_make_the_labels_unlearnable():
    """An attacked trial with a zero offset is identical to an honest one."""
    with pytest.raises(ConfigError, match=r"0\.0"):
        DetectionConfig(attack_offsets_db=(-20.0, 0.0))


def test_montecarlo_settings_are_range_checked():
    with pytest.raises(ConfigError, match="at least 1"):
        MonteCarloConfig(n_trials=0)
    with pytest.raises(ConfigError, match=r"\(0, 1\)"):
        MonteCarloConfig(confidence=1.0)


def test_sweep_axes_are_validated():
    with pytest.raises(ConfigError, match="empty"):
        SweepConfig(attack_offsets_db=())
    with pytest.raises(ConfigError, match="at least two access points"):
        SweepConfig(n_ap_values=(1,))
    with pytest.raises(ConfigError, match="not supported"):
        SweepConfig(combiners=("nope",))


def test_output_settings_are_validated():
    with pytest.raises(ConfigError, match="not supported"):
        OutputConfig(figure_formats=("jpeg",))
    with pytest.raises(ConfigError, match="empty"):
        OutputConfig(figure_formats=())
    with pytest.raises(ConfigError, match="at least 50"):
        OutputConfig(dpi=10)


def test_unknown_log_level_is_rejected():
    with pytest.raises(ConfigError, match="logging level"):
        Config(log_level="CHATTY")


def test_dotted_overrides_reach_nested_sections():
    config = Config().override(
        {"grid.step_db": 0.25, "normalize.mode": "density", "log_level": "DEBUG"}
    )
    assert config.grid.step_db == 0.25
    assert config.normalize.mode == "density"
    assert config.log_level == "DEBUG"


def test_overrides_are_revalidated():
    """An override cannot slip past the checks the constructor applies."""
    with pytest.raises(ConfigError, match="not supported"):
        Config().override({"normalize.mode": "nonsense"})


def test_unknown_override_keys_are_reported_rather_than_ignored():
    """A typo in a key must not leave a default quietly in place."""
    with pytest.raises(ConfigError, match="unknown configuration section"):
        Config().override({"gird.step_db": 0.5})
    with pytest.raises(ConfigError, match="unknown key"):
        Config().override({"grid.step_dB": 0.5})
    with pytest.raises(ConfigError, match="unknown configuration key"):
        Config().override({"nonsense": 1})


def test_unknown_yaml_keys_are_reported(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"grid": {"step_dB": 0.5}}), encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown key"):
        Config.from_yaml(path)

    path.write_text(yaml.safe_dump({"gird": {}}), encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown configuration section"):
        Config.from_yaml(path)


def test_missing_yaml_file_is_reported_clearly(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        Config.from_yaml(tmp_path / "absent.yaml")


def test_round_trips_through_yaml(tmp_path):
    """A saved configuration reloads to an equal one, which is what makes a run reproducible."""
    original = Config().override({"grid.step_db": 0.25, "scenario.combiner": "trimmed"})
    path = original.save(tmp_path / "config.yaml")
    reloaded = Config.from_yaml(path)
    assert reloaded.to_dict() == original.to_dict()
    assert reloaded.scenario.aps == original.scenario.aps


@pytest.mark.parametrize(
    "name",
    [
        "default",
        "one_attack",
        "two_attack",
        "legacy_exact",
        "sweep_attack_strength",
        "pathloss_floor",
    ],
)
def test_every_shipped_config_loads_and_builds(project_root, name):
    """Each file in configs/ is valid and produces a usable scenario."""
    config = Config.from_yaml(project_root / "configs" / f"{name}.yaml")
    scenario = config.build_scenario()
    assert scenario.n_ap >= 2
    assert scenario.radio_map.n_points == scenario.grid.size
