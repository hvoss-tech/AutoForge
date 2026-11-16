"""A tiny OrcaSlicer ``resources/profiles`` tree for the slicer-project tests.

Vendor "Acme" has a normal printer, a two-variant printer, a belt printer,
a printer whose default process doesn't list it, and a 2.85 mm printer;
``OrcaFilamentLibrary`` has the "@System" fallbacks Orca offers every printer.
"""

import json
import os


def _write(root, path, data, bom=False):
    full = os.path.join(root, *path.split("/"))
    os.makedirs(os.path.dirname(full), exist_ok=True)
    text = json.dumps(data, indent=1)
    with open(full, "w", encoding="utf-8-sig" if bom else "utf-8") as f:
        f.write(text)


def make_profiles(root):
    machines = {
        "fdm_acme_common": {
            "type": "machine", "name": "fdm_acme_common", "instantiation": "false",
            "printable_area": ["0x0", "200x0", "200x200", "0x200"], "printable_height": "200",
            "nozzle_diameter": ["0.4"], "min_layer_height": ["0.07"], "max_layer_height": ["0.28"],
            "machine_pause_gcode": "M601", "change_filament_gcode": "", "machine_start_gcode": "G28 ; acme",
            "nozzle_volume": ["100"], "retraction_length": ["0.8"],
        },
        "Acme One 0.4 nozzle": {
            "type": "machine", "name": "Acme One 0.4 nozzle", "inherits": "fdm_acme_common", "instantiation": "true",
            "printer_model": "Acme One", "printer_variant": "0.4",
            "default_print_profile": "0.20mm Standard @Acme One", "default_filament_profile": ["Acme PLA @Acme One"],
        },
        "Acme Duo 0.4 nozzle": {
            "type": "machine", "name": "Acme Duo 0.4 nozzle", "inherits": "fdm_acme_common", "instantiation": "true",
            "printer_model": "Acme Duo", "printer_variant": "0.4",
            "printable_area": "0x0,300x0,300x150,0x150",  # comma-joined, as some vendors write it
            "printer_extruder_variant": ["Direct Drive Standard", "Direct Drive High Flow"],
            "default_print_profile": "0.20mm Standard @Acme One;0.20mm Other",
            "default_filament_profile": ["Acme PLA @Acme One"],
        },
        "Acme Belt 0.4 nozzle": {
            "type": "machine", "name": "Acme Belt 0.4 nozzle", "inherits": "fdm_acme_common", "instantiation": "true",
            "printer_model": "Acme Belt", "printer_variant": "0.4", "belt_printer": "1",
            "default_print_profile": "0.20mm Standard @Acme One", "default_filament_profile": ["Acme PLA @Acme One"],
        },
        "Acme Lost 0.4 nozzle": {
            # Its default process doesn't list it; Orca would refuse the project.
            "type": "machine", "name": "Acme Lost 0.4 nozzle", "inherits": "fdm_acme_common", "instantiation": "true",
            "printer_model": "Acme Lost", "printer_variant": "0.4",
            "default_print_profile": "0.20mm Standard @Acme One", "default_filament_profile": ["Acme PLA @Acme One"],
        },
        "Acme Big 0.5 nozzle": {
            "type": "machine", "name": "Acme Big 0.5 nozzle", "inherits": "fdm_acme_common", "instantiation": "true",
            "printer_model": "Acme Big", "printer_variant": "0.5", "nozzle_diameter": ["0.5"],
            "default_print_profile": "0.20mm Standard @Acme One", "default_filament_profile": ["Acme PLA 2.85 @Acme Big"],
        },
    }
    processes = {
        "fdm_process_acme": {
            "type": "process", "name": "fdm_process_acme", "instantiation": "false",
            "layer_height": "0.2", "initial_layer_print_height": "0.2", "sparse_infill_density": "15%",
            "outer_wall_speed": ["200"], "enable_prime_tower": "1",
        },
        "0.20mm Standard @Acme One": {
            "type": "process", "name": "0.20mm Standard @Acme One", "inherits": "fdm_process_acme",
            "instantiation": "true",
            "compatible_printers": ["Acme One 0.4 nozzle", "Acme Duo 0.4 nozzle", "Acme Belt 0.4 nozzle", "Acme Big 0.5 nozzle"],
        },
        "0.16mm Optimal @Acme Lost": {
            "type": "process", "name": "0.16mm Optimal @Acme Lost", "inherits": "fdm_process_acme",
            "instantiation": "true", "compatible_printers": ["Acme Lost 0.4 nozzle"],
        },
        "0.20mm Standard @Acme Lost": {
            "type": "process", "name": "0.20mm Standard @Acme Lost", "inherits": "fdm_process_acme",
            "instantiation": "true", "compatible_printers": ["Acme Lost 0.4 nozzle"],
        },
    }
    filaments = {
        "fdm_filament_acme": {
            "type": "filament", "name": "fdm_filament_acme", "instantiation": "false", "inherits": "fdm_filament_common",
            "nozzle_temperature": ["210"], "filament_flow_ratio": ["0.98"],
        },
        "acme_template_variants": {
            "type": "filament", "name": "acme_template_variants", "instantiation": "false",
            "filament_extruder_variant": ["Direct Drive Standard", "Direct Drive High Flow"],
            "filament_max_volumetric_speed": ["15", "25"],
        },
        "Acme PLA @Acme One": {
            "type": "filament", "name": "Acme PLA @Acme One", "inherits": "fdm_filament_acme", "instantiation": "true",
            "include": ["acme_template_variants"], "filament_id": "ACME1",
            "filament_type": ["PLA"], "filament_diameter": ["1.75"],
            "compatible_printers": ["Acme One 0.4 nozzle", "Acme Duo 0.4 nozzle", "Acme Belt 0.4 nozzle", "Acme Lost 0.4 nozzle"],
        },
        "Acme PETG @Acme One": {
            "type": "filament", "name": "Acme PETG @Acme One", "inherits": "fdm_filament_acme", "instantiation": "true",
            "filament_type": ["PETG"], "filament_diameter": ["1.75"], "nozzle_temperature": ["240"],
            "compatible_printers": ["Acme One 0.4 nozzle"],
        },
        "Acme PLA 2.85 @Acme Big": {
            "type": "filament", "name": "Acme PLA 2.85 @Acme Big", "inherits": "fdm_filament_acme", "instantiation": "true",
            "filament_type": ["PLA"], "filament_diameter": ["2.85"], "compatible_printers": ["Acme Big 0.5 nozzle"],
        },
    }
    vendor_index = {
        "name": "Acme", "version": "01.00.00.00",
        "machine_model_list": [
            {"name": m, "sub_path": f"machine/{m}.json"} for m in ("Acme One", "Acme Duo", "Acme Belt", "Acme Lost", "Acme Big")
        ],
        "machine_list": [{"name": n, "sub_path": f"machine/{n}.json"} for n in machines],
        "process_list": [{"name": n, "sub_path": f"process/{n}.json"} for n in processes],
        "filament_list": [{"name": n, "sub_path": f"filament/{n}.json"} for n in filaments],
    }
    _write(root, "Acme.json", vendor_index)
    for kind, presets in (("machine", machines), ("process", processes), ("filament", filaments)):
        for name, data in presets.items():
            _write(root, f"Acme/{kind}/{name}.json", data, bom=(name == "Acme One 0.4 nozzle"))
    for model in vendor_index["machine_model_list"]:
        _write(root, f"Acme/{model['sub_path']}", {"type": "machine_model", "name": model["name"]})

    library = {
        "fdm_filament_common": {
            "type": "filament", "name": "fdm_filament_common", "instantiation": "false",
            "filament_diameter": ["1.75"], "nozzle_temperature": ["200"], "filament_type": ["PLA"],
        },
        "Generic PETG @System": {
            "type": "filament", "name": "Generic PETG @System", "inherits": "fdm_filament_common", "instantiation": "true",
            "filament_type": ["PETG"], "nozzle_temperature": ["245"],
        },
        "Generic ABS @System": {
            "type": "filament", "name": "Generic ABS @System", "inherits": "fdm_filament_common", "instantiation": "true",
            "filament_type": ["ABS"],
        },
    }
    _write(root, "OrcaFilamentLibrary.json", {
        "name": "OrcaFilamentLibrary",
        "filament_list": [{"name": n, "sub_path": f"filament/{n}.json"} for n in library],
    })
    for name, data in library.items():
        _write(root, f"OrcaFilamentLibrary/filament/{name}.json", data)
    for vendor in ("Acme", "OrcaFilamentLibrary"):
        os.makedirs(os.path.join(root, vendor), exist_ok=True)
    return root
