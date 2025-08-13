# Time‑Series Image Registration CLI Tool
## Overview
This command‑line tool performs **hierarchical image registration** on time‑series datasets from confocal or two‑photon microscopy. It can load **multi‑page TIFF** stacks or **HDF5** files, compute an average reference, register all frames to that reference (optionally over multiple iterations and scales), and save the registered stack in **HDF5** or **TIFF** format. It also supports optional **smoothing** of the frame‑wise motion offsets.

Core functions included:

- `load_hdf5_data(file_path)`  
- `save_hdf5_data(images, metadata, output_path)`  
- `load_multitiff_image(file_path)`  
- `compute_average_reference(images, num_frames=100)`  
- `smooth_registration_offsets(reg_offsets, window_length=11, polyorder=3)`  
- `register_images(img_stack, template, downsample_rates, max_movement)`  
- `apply_registration(img_stack, reg_offsets)`  
- `save_stack(images, save_path)`  
- CLI entrypoint: `main()`

---

## Installation

### Conda (recommended)
```bash
conda create -n image_reg python=3.12 numpy pandas scipy tifffile h5py opencv tqdm
conda activate image_reg
```

### pip
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install numpy pandas scipy tifffile h5py opencv-python tqdm
```

---

## Input Formats

- **HDF5 (`.h5`, `.hd5`)**  
  Expects a dataset `image_data` with shape `(frames, height, width)`. Optional `metadata` group with attributes is supported; JSON-serializable values will be preserved.

- **TIFF (`.tif`, `.tiff`)**  
  Multi‑page TIFF stacks are read with `tifffile`. 2D single images are rejected; 4D volumetric stacks are not supported (single focal plane time‑series only).

---

## Usage

```bash
python RegistrationScriptFINAL_hd5.py   -i /path/to/input.tif   -o /path/to/output.h5   --iterate 2   -s 1   --num_frames 50   --downsample_rates [1/8,1/4,1/2,1]   --max_movement 0.5   --save_as_tif 0
```

### Arguments
| Flag | Name | Type / Choices | Default | Description |
|---|---|---|---|---|
| `-i`, `--input` | Input path | str | — | Path to input HDF5 (`.h5`, `.hd5`) or TIFF (`.tif`, `.tiff`) file. |
| `-o`, `--output` | Output path | str | — | Path to save registered output (HDF5 or TIFF). |
| `--save_as_tif` | Save as TIFF | `{0,1}` | `0` | If `1`, writes a TIFF stack instead of HDF5. |
| `--iterate` | Iterations | int | `1` | Number of registration passes (re‑register the already registered stack). |
| `-s`, `--smooth` | Smoothing | `{0,1}` | `0` | If `1`, smooths frame offsets using Savitzky–Golay filter. |
| `--num_frames` | Reference frames | int | `100` | Frames used to compute the average reference image. |
| `--downsample_rates` | Pyramid scales | list | `[1/8,1/4,1/2,1]` | List of downsampling factors for hierarchical registration. |
| `--max_movement` | Max movement | float | `0.5` | Maximum expected motion (relative to frame size) used to bound the search. |

> **Tip:** When providing `--downsample_rates` in the shell, wrap in brackets with no spaces, e.g. `--downsample_rates [1/8,1/4,1/2,1]`.

---

## Typical Workflow

1. **Load input**  
   The script loads either HDF5 (`image_data`) or a multi‑page TIFF stack.
2. **Compute reference**  
   An average of the first `--num_frames` frames is used as the registration template.
3. **Hierarchical registration**  
   For each scale in `--downsample_rates`, correlation‑based alignment estimates per‑frame `(y, x)` offsets.
4. **Optional smoothing**  
   If `--smooth 1`, offsets are smoothed with a Savitzky–Golay filter to reduce jitter.
5. **Apply registration**  
   The offsets are applied to each frame using `cv2.warpAffine`.
6. **Save output**  
   Writes HDF5 (with `image_data` and `metadata`) or TIFF, depending on `--save_as_tif`.

---

## Examples

### Register a TIFF stack and save as HDF5
```bash
python RegistrationScriptFINAL_hd5.py   -i data/input_stack.tif   -o results/registered.h5   --iterate 1 -s 1 --num_frames 100   --downsample_rates [1/8,1/4,1/2,1]   --max_movement 0.5   --save_as_tif 0
```

### Register an HDF5 stack and save as TIFF
```bash
python RegistrationScriptFINAL_hd5.py   -i data/input_stack.h5   -o results/registered.tif   --iterate 2 -s 0   --downsample_rates [1/8,1/4,1/2,1]   --save_as_tif 1
```

---

## Function Reference (Quick)

- **`load_hdf5_data(path)`** → `(images, metadata)`  
  Loads HDF5 dataset `image_data` and optional `metadata` attributes.

- **`save_hdf5_data(images, metadata, output_path)`**  
  Saves a stack to HDF5 (`image_data`) with JSON‑serialized metadata.

- **`load_multitiff_image(path)`** → `np.ndarray | None`  
  Reads multi‑page TIFF; returns stack `(frames, height, width)`.

- **`compute_average_reference(images, num_frames=100)`** → `np.ndarray`  
  Averages the first `num_frames` frames into a template image.

- **`smooth_registration_offsets(reg_offsets, window_length=11, polyorder=3)`** → `np.ndarray`  
  Smooths per‑frame `(y, x)` offsets using Savitzky–Golay.

- **`register_images(img_stack, template, downsample_rates, max_movement)`** → `np.ndarray`  
  Hierarchical correlation search to estimate per‑frame offsets `(y, x)`.

- **`apply_registration(img_stack, reg_offsets)`** → `np.ndarray`  
  Applies offsets via `cv2.warpAffine`; returns registered stack.

- **`save_stack(images, save_path)`**  
  Writes a multi‑page TIFF to `save_path`.

---

## HDF5 Structure

When saving HDF5:
```
/image_data   (dataset)  float/uint stack (frames, height, width)
/metadata     (group)    attributes with key–value pairs (strings/JSON)
```

---

## Notes & Best Practices

- **Data types:** Registration runs in `float32` internally; outputs are clipped to valid ranges for TIFF (`uint16` expected).  
- **Performance:** More downsample levels can improve robustness but increase runtime.  
- **Smoothing:** Use `--smooth 1` when motion offsets are noisy across frames.  
- **Volumetric data:** 4D (3D + time) inputs are not supported; provide a 2D time‑series stack.  
- **Metadata:** HDF5 metadata is preserved when present; TIFF has limited metadata support.

---

## Troubleshooting

- **“No image data found in the HDF5 file.”**  
  Ensure your dataset is stored at `/image_data`.

- **“Cannot accept 3D volumetric images.”**  
  The tool expects a time‑series (frames × height × width), not 3D volumes.

- **Weird offsets or drift remains.**  
  Try increasing `--iterate`, adjusting `--downsample_rates`, or enabling `--smooth`.

- **TIFF save failing.**  
  Make sure the output directory exists and that you have write permissions.

---

## License
MIT (or your preferred license)

## Citation
If you use this tool in academic work, please cite:  
*Duboué, E. (2025). Time‑Series Image Registration CLI Tool.*
