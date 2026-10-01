import sys
import traceback
import uuid

import numpy as np
import pandas as pd
import torch
import json


def load_materials(args):
    """
    Load material data from a CSV file.

    Args:
        csv_filename (str): Path to the hueforge CSV file containing material data.

    Returns:
        tuple: A tuple containing:
            - material_colors (jnp.ndarray): Array of material colors in float64.
            - material_TDs (jnp.ndarray): Array of material transmission/opacity parameters in float64.
            - material_names (list): List of material names.
            - colors_list (list): List of color hex strings.
    """
    df = load_materials_pandas(args)
    # An empty CSV cell is NaN in pandas; str() turned it into "nan".
    material_names = [
        " - ".join(part for part in (_text(brand), _text(name)) if part)
        for brand, name in zip(df["Brand"].tolist(), df["Name"].tolist())
    ]
    material_TDs = (df["Transmissivity"].astype(float)).to_numpy()
    colors_list = [normalize_hex(color) for color in df["Color"].tolist()]
    # Use float64 for material colors.
    material_colors = np.array(
        [hex_to_rgb(color) for color in colors_list], dtype=np.float64
    )
    material_TDs = np.array(material_TDs, dtype=np.float64)
    return material_colors, material_TDs, material_names, colors_list

def load_materials_pandas(args):
    csv_filename = args.csv_file
    json_filename = args.json_file

    if csv_filename != "":
        try:
            df = pd.read_csv(csv_filename)
        except Exception as e:
            traceback.print_exc()
            print("Error reading filament CSV file:", e)
            sys.exit(1)
    else:
        # read json
        with open(json_filename, "r") as f:
            data = json.load(f)
        if "Filaments" in data.keys():
            data = data["Filaments"]
        else:
            print(
                "Warning: No Filaments key found in JSON data. We can't use this json data."
            )
            sys.exit(1)
        # list to dataframe
        df = pd.DataFrame(data)
    # Same column spellings for both sources: a JSON library using "TD" (or
    # padded keys) used to fail with KeyError 'Transmissivity'.
    df.columns = [str(col).strip() for col in df.columns]
    if "TD" in df.columns and "Transmissivity" not in df.columns:
        df.rename(columns={"TD": "Transmissivity"}, inplace=True)
    return df

def load_materials_data(args):
    """
    Load the full material data from the CSV file.

    Args:
        csv_filename (str): Path to the CSV file containing material data.

    Returns:
        list: A list of dictionaries (one per material) with keys such as
              "Brand", "Type", "Color", "Name", "TD", "Owned", and "Uuid".
    """
    df = load_materials_pandas(args)
    # Use a consistent key naming. For example, convert 'TD' to 'Transmissivity' and 'Uuid' to 'uuid'
    records = df.to_dict(orient="records")
    return records


def _text(value) -> str:
    """A table cell as text; missing / NaN cells are empty."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def normalize_hex(hex_str) -> str:
    """``#RRGGBB`` for the hex spellings libraries use (``RRGGBB``, ``#RGB``,
    ``#RRGGBBAA``); ValueError for anything else."""
    text = str(hex_str).strip().lstrip("#")
    if len(text) == 3:
        text = "".join(c * 2 for c in text)
    elif len(text) == 8:
        text = text[:6]
    if len(text) != 6 or any(c not in "0123456789abcdefABCDEF" for c in text):
        raise ValueError(f"{hex_str!r} is not a hex color like #RRGGBB")
    return "#" + text.upper()


def hex_to_rgb(hex_str):
    """
    Convert a hex color string to a normalized RGB list.

    Args:
        hex_str (str): The hex color string ('#RRGGBB'; '#RGB', '#RRGGBBAA'
            and a missing '#' are accepted too).

    Returns:
        list: A list of three floats representing the RGB values normalized to [0, 1].
    """
    hex_str = normalize_hex(hex_str).lstrip("#")
    return [int(hex_str[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]


def extract_colors_from_swatches(swatch_data):
    # we keep only data with transmission distance
    swatch_data = [swatch for swatch in swatch_data if swatch["td"]]

    # For now we load it and convert it in the same way as the hueforge csv files
    out = {}
    for swatch in swatch_data:
        brand = swatch["manufacturer"]["name"]
        name = swatch["color_name"]
        color = swatch["hex_color"]
        td = swatch["td"]
        out[(brand, name)] = (color, td)

    # convert to the same format as the hueforge csv files
    material_names = [str(brand) + " - " + str(name) for (brand, name) in out.keys()]
    material_colors = np.array(
        [hex_to_rgb("#" + color) for color, _ in out.values()], dtype=np.float64
    )
    material_TDs = np.array([td for _, td in out.values()], dtype=np.float64)
    colors_list = [color for color, _ in out.values()]

    return material_colors, material_TDs, material_names, colors_list


def swatch_data_to_table(swatch_data):
    """
    Converts swatch JSON data into a table (list of dicts) with columns:
    "Brand", "Name", "Transmission Distance", "Hex Color".
    """
    table = []
    for swatch in swatch_data:
        if not swatch["td"]:
            continue
        brand = swatch["manufacturer"]["name"]
        name = swatch["color_name"]
        hex_color = swatch["hex_color"]
        td = swatch["td"]
        table.append(
            {
                "Brand": brand,
                "Name": name,
                "TD": td,
                "HexColor": f"#{hex_color}",
                "Uuid": str(uuid.uuid4()),
            }
        )
    return table


def count_distinct_colors(dg: torch.Tensor) -> int:
    """
    Count how many distinct color/material IDs appear in dg.
    """
    unique_mats = torch.unique(dg)
    return len(unique_mats)


def count_swaps(dg: torch.Tensor) -> int:
    """
    Count how many color changes (swaps) occur between adjacent layers.
    """
    # A 'swap' is whenever dg[i] != dg[i+1].
    return int((dg[:-1] != dg[1:]).sum().item())
