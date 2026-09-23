# AutoForge

AutoForge is a Python tool for generating 3D printed layered models from an input image. Using a learned optimization strategy with a Gumbel softmax formulation, AutoForge assigns materials per layer and produces both a discretized composite image and a 3D-printable STL file. It also generates swap instructions to guide the printer through material changes during a multi-material print. 

**TLDR:** It uses a picture to generate a 3D layer image that you can print with a 3d printer. Similar to [Hueforge](https://shop.thehueforge.com/), but without the manual work (and without the artistic control).


## You can now run Autoforge for free in your browser thanks to [Huggingface space support](https://huggingface.co/spaces/hvoss-techfak/Autoforge).
This includes the option to run it locally if you have a powerful pc and don't want to limit yourself to the Huggingface computing limits. \
For this simply go to the [Huggingface](https://huggingface.co/spaces/hvoss-techfak/Autoforge) space and pull the docker container for this project (upper right corner -> three dots -> "run locally")

## Example
All examples use only the 27 BambuLab Basic PLA filaments, currently available in Hueforge 0.9.0, the background color is set to black.
The pruning is set to a maximum of 8 color and 20 swaps, so each image uses at most 8 different colors and swaps the filament at most 20 times. 
<div style="display: flex; justify-content: center; gap: 20px;">
  <div style="text-align: center;">
    <h3>Input Image</h3>
    <img src="https://github.com/hvoss-techfak/AutoForge/blob/main/images/lofi.jpg" width="200" />
    <img src="https://github.com/hvoss-techfak/AutoForge/blob/main/images/nature.jpg" width="200" />
    <img src="https://github.com/hvoss-techfak/AutoForge/blob/main/images/cat.jpg" width="200" />
    <img src="https://github.com/hvoss-techfak/AutoForge/blob/main/images/chameleon.jpg" width="200" />
  </div>
  <div style="text-align: center;">
    <h3>Autoforge Output</h3>
    <img src="https://github.com/hvoss-techfak/AutoForge/blob/main/images/lofi_discretized.png" width="200" />
    <img src="https://github.com/hvoss-techfak/AutoForge/blob/main/images/nature_discretized.png" width="200" />
    <img src="https://github.com/hvoss-techfak/AutoForge/blob/main/images/cat_discretized.png" width="200" />
    <img src="https://github.com/hvoss-techfak/AutoForge/blob/main/images/chameleon_discretized.png" width="200" />
  </div>
</div>

## Features

- **Image-to-Model Conversion**: Converts an input image into a layered model suitable for 3D printing.
- **Learned Optimization**: Optimizes per-pixel height and per-layer material assignments using PyTorch.
- **Learned Heightmap**: Optimizes the height of the layered model to create more detailed prints.
- **Gumbel Softmax Sampling**: Leverages the Gumbel softmax method to decide material assignments for each layer.
- **FlatForge Mode**: Generate separate STL files for each color, enabling face-down printing for smooth, resin-like finishes.
- **STL File Generation**: Exports an ASCII STL file based on the optimized height map.
- **Swap Instructions**: Generates clear swap instructions for changing materials during printing.
- **Live Visualization**: (Optional) Displays live composite images during the optimization process.
- **Hueforge export**: Outputs a project file that can be opened with hueforge.

## Web UI: One-Click Install & Run

The easiest way to use AutoForge is the web UI — a local app (like ComfyUI) with drag-and-drop image upload, a filament library, live sliders for adjusting colors after optimization, and pruning. No command-line arguments needed.

1. **Clone this repository** (or download and extract the ZIP from the green "Code" button on GitHub).
2. **Install**, from the project folder:
   - Linux/macOS: `./install.sh`
   - Windows: double-click `install.bat` (or run it from a terminal)

   This installs [`uv`](https://docs.astral.sh/uv/) (a fast Python package manager) if you don't already have it, installs all Python dependencies, and builds the web UI. You'll need [Node.js](https://nodejs.org/) installed for that last step — the installer will tell you if it's missing.
3. **Run**:
   - Linux/macOS: `./run_webui.sh`
   - Windows: double-click `run_webui.bat`

   This starts the server and opens the web UI in your browser automatically (usually at `http://localhost:8000`).

   The web UI sends anonymous usage telemetry to the project via [PostHog](https://posthog.com/) by default, to notify me of problems and any bugs. This includes crash reports — unhandled errors from both the browser frontend and the backend server, with the error type, message, and stack trace — so bugs can get fixed faster. No image data, filament data, or personal information is sent. To disable it, pass `--no-telemetry` (e.g. `./run_webui.sh --no-telemetry` / `run_webui.bat --no-telemetry`), or set `AUTOFORGE_WEBUI_TELEMETRY_ENABLED=false` permanently in your environment.
4. **Update to the latest release** whenever you want, from the project folder:
   - Linux/macOS: `./update.sh`
   - Windows: double-click `update.bat`

   This checks GitHub for a newer release and, if one exists, pulls it and reinstalls dependencies for you. Add `--check` to only check without applying it (e.g. `./update.sh --check`).

If you have problems running the code on your GPU, please refer to the [Pytorch Homepage](https://pytorch.org/) for help. \
CUDA, ROCm, and MPS (Apple Metal) are supported, but you need to install the correct version of pytorch for your system — `install.sh`/`install.bat` install whatever `uv sync` resolves by default, so swap in a GPU-specific PyTorch build afterwards if you need one. (`install.sh` handles older NVIDIA GPUs automatically, see below.)

## Manual Installation (CLI only)

If you just want the command-line tool (no web UI), install the current version from PyPI:
```bash
   pip install -U autoforge
```

If you have problems running the code on your gpu, please refer to the [Pytorch Homepage](https://pytorch.org/) for help. \
CUDA, ROCm, and MPS (Apple Metal) are supported, but you need to install the correct version of pytorch for your system.

## Older NVIDIA GPUs (GTX 900/10-series, Titan V)

The default CUDA build of PyTorch only ships kernels for compute capability 7.5 (Turing) and newer. On GTX 900-series (Maxwell), GTX 10-series (Pascal) and Titan V (Volta) cards, `torch.cuda.is_available()` still returns `True`, but the first kernel launch fails with `CUDA error: no kernel image is available for execution on the device`.

`install.sh` (Linux/macOS) detects these GPUs and reinstalls PyTorch from the CUDA 12.6 index, the last build line that includes their kernels (see [pytorch/pytorch#190385](https://github.com/pytorch/pytorch/issues/190385)). `run_webui.sh` then skips `uv`'s automatic sync so the launch doesn't swap the default build back in. To do it by hand, run this in the project folder:

```bash
uv pip install --reinstall-package torch --reinstall-package torchvision torch torchvision \
    --index-url https://download.pytorch.org/whl/cu126
```

and set `UV_NO_SYNC=1` when using `uv run` so it isn't reverted. With plain `pip`, use `pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126` instead.

To verify, check that the architecture list printed by the following contains an entry your GPU can run (`sm_61` is covered by `sm_60`):

```bash
uv run --no-sync python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list())"
```

A working CUDA 12.6 build lists `sm_50`, `sm_60` and `sm_70` in addition to the newer architectures.

## Usage

The script is run from the command line and accepts several arguments. Below is an example command:

> **Note:** You will need [Hueforge](https://shop.thehueforge.com/) installed to export your filament CSV.  
> To get your CSV file, simply go to the "Filaments" menu in Hueforge, click the export button, select your filaments, and export them as a CSV file.

```bash
autoforge --input_image path/to/input_image.jpg --csv_file path/to/materials.csv 
```

We also support json files. If you want to use your personal Hueforge library (found in %APPDATA%\HueForge\Filaments\personal_library.json) you can run the command with:
```bash
autoforge --input_image path/to/input_image.jpg --json_file %APPDATA%\HueForge\Filaments\personal_library.json
```


> If you want to limit the amount of colors the program can use, you can set these as command line arguments. \
> For Example: 8 colors and a maximum of 20 swaps:

```bash
autoforge --input_image path/to/input_image.jpg --csv_file path/to/materials.csv --pruning_max_colors 8 --pruning_max_swaps 20
```

### FlatForge Mode

To use FlatForge mode for smooth, face-down printing:

```bash
autoforge --input_image path/to/input_image.jpg --csv_file path/to/materials.csv --flatforge --pruning_max_colors 4 --cap_layers 2
```

This will generate separate STL files for each color, allowing you to print face-down on the build plate for a smooth finish. With `--pruning_max_colors 4`, you'll get 2 colored materials + 1 clear filament + 1 background = 4 total filaments (perfect for a 4-slot AMS).

### Command Line Arguments

- `--config` *(Optional)* Path to a configuration file with the settings.

- `--input_image` **(Required)** Path to the input image.
- `--csv_file` Path to the CSV file containing material data. The CSV should include columns for the brand, name, color (hex code), and TD values.
- `--json_file` Path to the json file containing material data.  
  **Note:** Either a csv or json file has to be given.

- `--output_folder` Folder where output files will be saved (default: `output`).
- `--iterations` Number of optimization iterations (default: 2000).
- `--warmup_fraction` Fraction of iterations for keeping the tau at the initial value (default: 0.25).
- `--learning_rate_warmup_fraction` Fraction of iterations that the learning rate is increasing (warmup) (default: 0.25).
- `--init_tau` Initial tau value for Gumbel-Softmax (default: 1.0).
- `--final_tau` Final tau value for the Gumbel-Softmax formulation (default: 0.01).
- `--learning_rate` Learning rate for optimization (default: 0.015).
- `--layer_height` Layer thickness in millimeters (default: 0.04).
- `--max_layers` Maximum number of layers (default: 75).  
  **Note:** This is about 3mm + the background height
- `--min_layers`  Minimum number of layers (default: 0). Used to limit height of pruning.
- `--background_height` Height of the background in millimeters (default: 0.24).  
  **Note:** The background height must be divisible by the layer height.
- `--background_color` Background color in hexadecimal format (default: `#000000` aka Black).  
  **Note:** The solver currently assumes that you have a solid color in the background, which means a color with a TD value of 4 or less (if you have a background height of 0.4).
- `--visualize` enable live visualization of the composite image during optimization (default: True).
- `--stl_output_size` Size of the longest dimension of the output STL file in millimeters (default: 200).
- `--processing_reduction_factor` Reduction factor for the processing size compared to the output size (default: 2 - half resolution).
- `--nozzle_diameter` Diameter of the printer nozzle in millimeters (default: 0.4).  
  **Note:** Details smaller than half this value will be ignored.
- `--early_stopping` Number of steps without improvement before stopping (default: 10000).

- `--flatforge` Enable FlatForge mode to generate separate STL files for each color (default: False).  
  **Note:** FlatForge creates flat prints where each color is its own STL file, allowing face-down printing for smooth, resin-like finishes. Requires a multi-material printer (AMS, MMU, or tool changer).
- `--cap_layers` Number of complete transparent/clear layers to add on top in FlatForge mode (default: 0).  
  **Note:** Creates a glossy cap layer over the colored layers for a resin-filled appearance. Only used when `--flatforge` is enabled.

- `--perform_pruning`  Perform pruning after optimization (default: True).  
  **Note:** This is highly recommended even if you don't have a color/color swap limit, as it actually increases the quality of the output.
- `--fast_pruning`  Perform pruning in chunks. 10-15x speedup compared to accurate method (default: False).
- `--fast_pruning_percent` Size of fast pruning chunks in percent (default: 0.5) (50%).
- `--pruning_max_colors` Max number of colors allowed after pruning (default: 100).  
  **Note:** This includes background in both modes. In FlatForge mode, also includes clear filament.
  - Traditional: `--pruning_max_colors 4` means 3 colored + 1 background = 4 total filaments.
  - FlatForge: `--pruning_max_colors 4` means 2 colored + 1 clear + 1 background = 4 total filaments.
- `--pruning_max_swaps` Max number of swaps allowed after pruning (default: 100).
- `--pruning_max_layer` Max number of layers allowed after pruning (default: 75).
- `--random_seed` Random seed for reproducibility (default: 0 (disabled)).
- `--device` Torch device to run on, e.g. `cuda`, `cuda:1`, `mps`, `cpu`. Defaults to auto-detection: CUDA/ROCm first, then Apple Metal (MPS), then CPU. Can also be set with the `AUTOFORGE_DEVICE` environment variable.
- `--mps` *Deprecated* — Apple Metal is now detected automatically, so this flag is no longer needed. It still works, and forces MPS on a machine that also exposes a CUDA GPU; prefer `--device mps`.
- `--no-spike-removal` Disable spike removal for the final STL (not recommended).

- `--tensorboard` Flag to enable TensorBoard logging.
- `--run_name` *(Optional)* Name of the run used for TensorBoard logging.
- `--num_init_rounds` Number of rounds to choose the starting height map from (default: 1 - extra rounds are currently deterministic and don't add variety, so they only cost startup time).
- `--num_init_cluster_layers` Number of layers to cluster the image into (default: -1).
- `--disable_visualization_for_gradio` Simple switch to disable the matplotlib render window for gradio rendering (default: 0).
- `--best_of` Run the entire program multiple times and output the best result (default: 1)

## Outputs

After running, the following files will be created in your specified output folder:

**Traditional Mode:**
- **Discrete Composite Image**: `final_model.png`
- **STL File**: `final_model.stl`
- **Hueforge Project File**: `project_file.hfp` 
- **Swap Instructions**: `swap_instructions.txt`

**FlatForge Mode** (when `--flatforge` is enabled):
- **Discrete Composite Image**: `final_model.png`
- **Separate STL files for each color**: One STL per material (e.g., `BrandName_ColorName_HEXCODE.stl`)
- **Clear/Transparent STL**: Uses the most transparent material from your library
- **Background STL**: `Background_HEXCODE.stl`
- **Optional Cap Layer STL**: `Cap_MaterialName_HEXCODE.stl` (if `--cap_layers > 0`)
  
  *Note:* FlatForge generates multiple STL files that align perfectly when loaded together in your slicer, creating a solid rectangular print with each color as a separate object.

## Development

To have a "nightly" version of the repository or have live updating changes during development please do the following: 

```bash
git clone https://github.com/hvoss-techfak/AutoForge.git
cd AutoForge
conda create -n forge python=3.11
conda activate forge
pip install -e .
```

If the installed pytorch version has no cuda support execute the following:

```bash
conda activate forge
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```


## Known Bugs

- The optimizer can sometimes get stuck in a local minimum. If this happens, try running the optimization again with different settings.

## License

AutoForge © 2025 by Hendric Voss is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
The software is provided as-is and comes with no warranty or guarantee of support.

The above license applies to the software itself. How you use the generated files and prints is entirely up to you. If you want to print them for your friends or family, that's great. If you want to sell them, that's fine by me too. 

From a licensing standpoint, this means that the prints you create are entirely subject to the [MIT License](https://opensource.org/licenses/MIT), and you can do whatever you like with them. The only thing the MIT License does not give you is a warranty, and it also frees me from any liability with regard to your 3D prints. Otherwise, do whatever you like.

I would love to see what you have done with the software, so it would be great if you could send me a link to your work (even if it's just for selling). However, this is not necessary if you do not want to.

## Acknowledgements

First and foremost:
- [Hueforge](https://shop.thehueforge.com/) for providing the inspiration for this project.
Without it, this project would not have been possible.

AutoForge makes use of several open source libraries:

- [PyTorch](https://pytorch.org/)
- [Optax](https://github.com/deepmind/optax)
- [OpenCV](https://opencv.org/)
- [Matplotlib](https://matplotlib.org/)
- [Pandas](https://pandas.pydata.org/)
- [TQDM](https://github.com/tqdm/tqdm)
- [ConfigArgParse](https://github.com/bw2/ConfigArgParse)

Example Images: \
<a href="https://www.vecteezy.com/free-photos/nature">Nature Stock photos by Vecteezy</a> \
<a href="https://www.vecteezy.com/free-photos/ai-generated">Ai Generated Stock photos by Vecteezy</a> \
<a href="https://www.vecteezy.com/free-photos/animal">Animal Stock photos by Vecteezy</a> \
<a href="https://www.vecteezy.com/free-photos/people">People Stock photos by Vecteezy</a> \
<a href="https://www.vecteezy.com/free-photos/psychedelic">Psychedelic Stock photos by Vecteezy</a> \
<a href="https://www.vecteezy.com/free-photos/ocean">Ocean Stock photos by Vecteezy</a> \
<a href="https://www.vecteezy.com/free-photos/psychic">Psychic Stock photos by Vecteezy</a> \
<a href="https://www.vecteezy.com/free-photos/psychedelic">Psychedelic Stock photos by Vecteezy</a> \
<a href="https://www.vecteezy.com/free-photos/pattern">Pattern Stock photos by Vecteezy</a> \
<a href="https://www.vecteezy.com/free-photos/forest">Forest Stock photos by Vecteezy</a> \
<a href="https://www.vecteezy.com/free-photos/stick-figure-kids">Stick Figure Kids Stock photos by Vecteezy</a> \
<a href="https://www.vecteezy.com/free-photos/forest">Forest Stock photos by Vecteezy</a> \
<a href="https://www.vecteezy.com/free-photos/nature">Nature Stock photos by Vecteezy</a> \
<a href="https://www.vecteezy.com/free-photos/ai-generated">Ai Generated Stock photos by Vecteezy</a>\
<a href="https://www.vecteezy.com/free-photos/animal">Animal Stock photos by Vecteezy</a> 


Happy printing!
