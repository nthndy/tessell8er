# tessell8er
> *Tessellate the image — just not yet.*

<p align="center">
  <img src="tessell8er.gif" alt="tessell8er"/>
</p>

**tessell8er** is a lightweight Python package for lazily stitching fragmented microscopy image tiles — exported from [Harmony/Opera Phenix](https://www.revvity.com/) — into a single [Dask](https://www.dask.org/) array. Computation is deferred until you explicitly call `.compute()`, keeping memory usage minimal regardless of mosaic size.

---

## Features

- Lazy, chunked mosaic assembly via `dask.array`
- Affine-transform-based tile placement from Harmony stage coordinates
- Shapely STRtree for efficient chunk–tile intersection
- Optional Z-projections (max, sum) fully deferred until compute
- Harmony metadata parser (V5 / V6 XML, iterative or full-tree)
- OME-TIFF and OME-NGFF / Zarr export

---

## Installation

Clone the repository:

```bash
git clone https://github.com/nthndy/tessell8er.git
cd tessell8er
```

Create and activate the environment:

```bash
mamba env create -f environment.yml
mamba activate tessell8er
```

---

## Quick start

```python
from tessell8er import dataio, tile

# Parse Harmony metadata
metadata = dataio.read_harmony_metadata('path/to/Index.idx.xml')

# Build lazy mosaic (T, C, Z, Y, X) — no pixels loaded yet
images = tile.compile_mosaic(
    image_dir='path/to/images/',
    metadata=metadata,
    row=3,
    col=5,
    n_tile_rows=3,
    n_tile_cols=3,
)

# Trigger computation
loaded = images.compute()

# Stream directly to Zarr without loading into RAM
images.to_zarr('output.zarr', component='0', overwrite=True)
```

---

## Repository structure

```
tessell8er/
├── tessell8er/
│   ├── __init__.py
│   ├── tile.py       # Core stitching engine
│   └── dataio.py     # Harmony metadata I/O
├── notebooks/
│   └── tile_image.ipynb
├── environment.yml
├── pyproject.toml
└── README.md
```

---

## Flat-field correction
 
tessell8er supports automatic flat-field correction (FFC) using illumination profiles exported by Harmony/Opera Phenix. FFC corrects for vignetting — the systematic intensity gradient introduced by the microscope optics that causes the centre of each tile to appear brighter than the edges.
 
This is particularly important for **fluorescence intensity-based measurements** (e.g. spot detection, area quantification, morphometrics) across compiled mosaics. Without correction, cells near tile edges will have artificially lower signal than cells at tile centres, introducing a systematic spatial bias into downstream analysis.
 
### How it works
 
Harmony exports the illumination profile as a 2D polynomial surface per channel in an XML file (`FFC_Profile/FFC_Profile_Measurement 1.xml`). tessell8er parses this file, reconstructs the polynomial surface for each channel, and divides each tile by its corresponding surface before stitching. Correction is applied lazily — only when `.compute()` is called.
 
### Usage
 
```python
from tessell8er import dataio, tile
 
# Parse the FFC profile — returns a dict of {channel_id: surface_array}
surfaces = dataio.read_ffc_profile('path/to/FFC_Profile/FFC_Profile_Measurement 1.xml')
 
# Build a correction function for the channel of interest
ffc_fn = dataio.make_ffc_transform(surfaces[3])  # channel 3 (1-based)
 
# Pass as input_transforms to compile_mosaic
mosaic = tile.compile_mosaic(
    image_dir='path/to/Images/',
    metadata=metadata,
    row=2,
    col=1,
    set_channel=3,
    input_transforms=[ffc_fn],
)
```
 
### Notes
 
- Channel IDs are 1-based, matching Harmony's channel numbering
- The correction surface is normalised so its mean equals 1.0 — overall intensity is preserved
- If no FFC profile is available, simply omit `input_transforms` and stitching proceeds uncorrected
- Multiple transforms can be chained via `input_transforms=[fn1, fn2, ...]` — FFC should be applied first

---

## Acknowledgements

Parts of the tiling pipeline were adapted from [Volker Hilsenstein's DaskFusion project](https://github.com/VolkerH/DaskFusion), used under the MIT License.

---

## Contact

**Nathan J. Day**  
*Host–Pathogen Interactions in Tuberculosis Laboratory*  
The Francis Crick Institute  
[nathan.day@crick.ac.uk](mailto:nathan.day@crick.ac.uk)  
[@nthndy.bsky.social](https://bsky.app/profile/nthndy.bsky.social)
