"""Tests for the command-line argument parser in autoforge.auto_forge."""

import pytest

from autoforge.auto_forge import parse_args


def _parse(monkeypatch, argv):
    monkeypatch.setattr("sys.argv", ["autoforge", *argv])
    return parse_args()


def test_input_image_is_required(monkeypatch):
    monkeypatch.setattr("sys.argv", ["autoforge"])
    with pytest.raises(SystemExit):
        parse_args()


def test_defaults_match_documented_values(monkeypatch):
    args = _parse(monkeypatch, ["--input_image", "img.png"])
    assert args.input_image == "img.png"
    assert args.csv_file == "" and args.json_file == ""
    assert args.output_folder == "output"
    assert args.iterations == 6000
    assert args.init_tau == 1.0 and args.final_tau == 0.01
    assert args.layer_height == 0.04
    assert args.max_layers == 75
    assert args.learning_rate == 0.015
    assert args.spike_threshold_layers == 1
    assert args.spike_removal_passes == 4


def test_overrides_are_typed(monkeypatch):
    args = _parse(
        monkeypatch,
        [
            "--input_image", "x.png",
            "--iterations", "1234",
            "--layer_height", "0.08",
            "--max_layers", "40",
        ],
    )
    assert args.iterations == 1234 and isinstance(args.iterations, int)
    assert args.layer_height == 0.08 and isinstance(args.layer_height, float)
    assert args.max_layers == 40


def test_config_file_populates_arguments(tmp_path, monkeypatch):
    cfg = tmp_path / "run.cfg"
    cfg.write_text("input_image = from_config.png\niterations = 42\n")
    args = _parse(monkeypatch, ["--config", str(cfg)])
    assert args.input_image == "from_config.png"
    assert args.iterations == 42


def test_cli_overrides_config_file(tmp_path, monkeypatch):
    cfg = tmp_path / "run.cfg"
    cfg.write_text("input_image = from_config.png\niterations = 42\n")
    args = _parse(monkeypatch, ["--config", str(cfg), "--iterations", "99"])
    assert args.iterations == 99
