# PixelStackerCLI ![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey)

**PixelStackerCLI** is a command-line tool for 
**photographic stacking of RAW images** (Deep Sky and 
terrestrial photos), currently focused on **advanced 
noise reduction** with AI support.

**Current and Planned Features:**

* **Noise Reduction** (Fully implemented)
* **Exposure Blending / HDR** (Planned for future release)
* **Focus Stacking** (Planned for future release)

---

## Key Technologies

- **Python** (3.10+)
- **NumPy** (array operations)
- **OpenCV** (image alignment and core processing)
- **ProcessPoolExecutor** (parallel processing)
- **PyTorch** (semantic segmentation with mask2former)
- **Astropy** (sigma clipping algorithm)

---

## Requirements

- Python 3.10 or higher  
- Libraries listed in `requirements.txt`:

```text
opencv-python
numpy
rawpy
Pillow
astropy
torch
transformers
huggingface-hub
```

- On Windows, the latest Microsoft Visual C++ Redistributable for Visual Studio is required.

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/everest1993/PixelStackerCLI

cd PixelStackerCLI

# create a virtual environment
python3 -m venv .venv

# activate the virtual environment
source .venv/bin/activate          # on Linux / macOS
.venv\Scripts\activate.ps1         # on Windows PowerShell

# if you get a script execution error on Windows, run:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# install dependencies
pip install -r requirements.txt

# install package
pip install -e .

# verification: check available commands and confirm installation
pixelstacker-cli --help
```

## Usage Examples

Currently, only the **Noise Reduction** subcommand is fully implemented.

**Available Subcommands:**
* `noise`: Noise Reduction (Fully implemented)
* `focus`: Focus stacking (Coming soon)
* `exposure`: Exposure blending / HDR (Coming soon)

### Noise Reduction Command Example

Stack all input RAW files to reduce noise. By default, the 
reference image for alignment (--ref-idx) is the central 
image of the list.

```bash
# navigate to the root folder
cd PixelStackerCLI

# activate the virtual environment
source .venv/bin/activate          # on Linux / macOS
.venv\Scripts\activate.ps1         # on Windows PowerShell

# Example: Stacks all .RAW files and outputs to a TIF file.
pixelstacker noise *.RAW -o stacked_output.tif --ref-idx 1
```

### Arguments (common & noise)

| Argument               | Description                                                                                     | Default       |
|:-----------------------|:------------------------------------------------------------------------------------------------|:--------------|
| `-o, --output`         | Output file path and name (TIF)                                                                 | **Required**  |
| `list_of_input_images` | List of input RAW images (minimum 2). Use wildcard (`*.RAW`, `*.NEF`, etc.) for multiple files. | **Required**  |
| `--ref-idx`            | Reference image index for alignment.                                                            | Central image |
| `--sigma-low`          | Lower sigma clipping value.                                                                     | None          |
| `--sigma-hi`           | Upper sigma clipping value.                                                                     | None          |
| `--min-keep`           | Minimum valid values to keep (absolute count).                                                  | None          |
| `--min-keep-frac`      | Minimum valid values to keep (fraction, 0.0 to 1.0).                                            | None          |
| `--iterations`         | Number of sigma clipping iterations.                                                            | None          |

## Best Practices for Input Images

To achieve optimal results with PixelStackerCLI, follow these guidelines when capturing source images:

1. **Use a tripod** – All photos must be taken from a fixed position without moving the camera or lens between shots.  
2. **Same framing and focal length** – Avoid zooming or reframing between exposures.  
3. **Identical dimensions** – All images should have the same resolution and aspect ratio.  
4. **Consistent exposure settings** – Use manual mode (M) to keep ISO, aperture, and shutter speed fixed when possible.  
5. **Sufficient image count** – For noise reduction, use at least 4 images (8 or more recommended).
6. **RAW format needed** – Provides the highest dynamic range and color depth.  
7. **Avoid in-camera processing** – Disable noise reduction, HDR, or automatic corrections.

## Contributing

Contributions are welcome! You can help by:

- Reporting bugs via issues 
- Requesting new features via issues 
- Submitting pull requests

```bash
# Process for submitting a pull request
git checkout -b feature/feature-name
git commit -m "Description of changes"
git push origin feature/feature-name
```

- Open a pull request on the main repository

**Guidelines**

- Keep code style consistent 
- Include tests for new features when possible
- Update the README if adding commands or significant changes

## Support

For questions, bug reports, or requests, contact:

**pixelstackercli.project@gmail.com**

## Acknowledgements

- OpenCV for image alignment and transformations  
- Astropy for robust sigma-clipping algorithms  
- PyTorch + Hugging Face for semantic segmentation (Mask2Former)  

## License

PixelStackerCLI is released under **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.

- You may share and modify the code for non-commercial purposes only. 
- Proper attribution to the original author is required. 
- Commercial use is prohibited without explicit permission from the author.

License: https://creativecommons.org/licenses/by-nc/4.0/