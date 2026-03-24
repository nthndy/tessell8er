# tessell8er
> *Tessellate the image — just not yet.*

https://github.com/nthndy/tessell8er/tessell8er.mp4

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
├── data/             # Place example data here
├── environment.yml
├── pyproject.toml
└── README.md
```

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
