"""The G-code checks the OrcaSlicer sweep and integration test rely on: they
must catch a wrong slice, or the sweep passes printers that print wrong."""

from types import SimpleNamespace

from orca_slicing import SWAP_MARKER, check_slice, read_gcode

OK = {"return_code": 0, "error_string": "Success."}
PLAN = SimpleNamespace(swaps=[(0.48, 1), (0.64, 2)])


def _gcode(tmp_path, lines):
    path = tmp_path / "plate_1.gcode"
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def _layers(*zs, bambu=False, between=None):
    out = []
    for z in zs:
        out.append(f"; Z_HEIGHT: {z}" if bambu else f";Z:{z}")
        out.extend((between or {}).get(z, []))
        out.append("G1 X1 Y1 E1")
    return out


def test_hand_swaps_all_found(tmp_path):
    g = _gcode(tmp_path, _layers(0.24, 0.32, 0.4, 0.48, 0.56, 0.64,
                                 between={0.48: [SWAP_MARKER, "M601"], 0.64: [SWAP_MARKER, "M601"]}))
    assert check_slice(OK, g, PLAN, True, 0.08, 0.24) is None


def test_missing_first_pause_is_caught(tmp_path):
    """What OrcaSlicer did on Bambu printers: the first hand swap without a pause."""
    g = _gcode(tmp_path, _layers(0.24, 0.32, 0.4, 0.48, 0.56, 0.64, bambu=True,
                                 between={0.48: ["; MANUAL_TOOL_CHANGE T1"], 0.64: [SWAP_MARKER, "M400 U1"]}))
    assert "pauses at [0.64], expected [0.48, 0.64]" in check_slice(OK, g, PLAN, True, 0.08, 0.24)


def test_the_config_dump_is_not_a_pause(tmp_path):
    g = _gcode(tmp_path, _layers(0.24, 0.32, 0.4, 0.48, 0.56, 0.64) + [f"; machine_pause_gcode = {SWAP_MARKER}\\nM601"])
    assert check_slice(OK, g, PLAN, True, 0.08, 0.24).startswith("pauses at []")


def test_tool_changes_indented_and_bambu_layers(tmp_path):
    g = _gcode(tmp_path, ["T0"] + _layers(0.24, 0.32, 0.4, 0.48, 0.56, 0.64, bambu=True,
                                          between={0.48: ["  T1 S1 L0 D0"], 0.64: ["T2"]}))
    assert read_gcode(g, False, 0.24)[1] == [(0.48, 1), (0.64, 2)]
    assert check_slice(OK, g, PLAN, False, 0.08, 0.24) is None


def test_missing_tool_change_is_caught(tmp_path):
    g = _gcode(tmp_path, _layers(0.24, 0.32, 0.4, 0.48, 0.56, 0.64, between={0.64: ["T2"]}))
    assert "no tool change at [0.48]" in check_slice(OK, g, PLAN, False, 0.08, 0.24)


def test_wrong_layer_heights_are_caught(tmp_path):
    # printed with the printer's 0.2 mm profile instead of the project's 0.08
    g = _gcode(tmp_path, _layers(0.2, 0.4, 0.6, 0.8))
    assert "base top 0.24 is no layer top" in check_slice(OK, g, PLAN, True, 0.08, 0.24)
    g = _gcode(tmp_path, _layers(0.24, 0.44, 0.64))
    assert "layer after the base at [0.44]" in check_slice(OK, g, PLAN, True, 0.08, 0.24)


def test_split_base_on_a_small_nozzle_is_fine(tmp_path):
    g = _gcode(tmp_path, _layers(0.16, 0.24, 0.32, 0.4, 0.48, 0.56, 0.64,
                                 between={0.48: [SWAP_MARKER], 0.64: [SWAP_MARKER]}))
    assert check_slice(OK, g, PLAN, True, 0.08, 0.24) is None


def test_refused_slice_is_reported(tmp_path):
    result = {"return_code": 205, "error_string": "There are some incorrect slicing parameters"}
    assert "refused it (205)" in check_slice(result, None, PLAN, True, 0.08, 0.24)
    assert check_slice(OK, None, PLAN, True, 0.08, 0.24) == "OrcaSlicer wrote no G-code"
