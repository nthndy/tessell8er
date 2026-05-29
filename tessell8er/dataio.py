"""
dataio.py
=========
Data I/O utilities for tessell8er: reading Harmony/Opera Phenix metadata
and generating the corresponding tile file URLs.

Removed from original macrohet version: btrack track export, Zarr track
packing, Prism file loading, and mask-existence checks — none of which
are part of the tiling pipeline.
"""

import os
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from lxml import etree as ET_iter
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Harmony metadata readers
# ---------------------------------------------------------------------------

def read_harmony_metadata(
    metadata_path: os.PathLike,
    assay_layout: bool = False,
    replicate_number: bool = True,
    iter: bool = True,
) -> pd.DataFrame:
    """Parse a Harmony/Opera Phenix XML metadata file into a DataFrame.

    Two parsing strategies are available via `iter`:

    * ``iter=True``  (default) — memory-efficient iterparse with a tqdm
      progress bar; recommended for large XML files on workstations.
    * ``iter=False`` — full-tree parse; faster on small files but loads
      the entire XML into RAM.

    Parameters
    ----------
    metadata_path : os.PathLike
        Path to the Harmony ``Index.idx.xml`` file.
    assay_layout : bool
        If True, parse the assay layout rather than image volume metadata.
        Consider :func:`read_harmony_assaylayout` for improved compatibility.
    replicate_number : bool
        When parsing assay layouts, add a 'Replicate #' column grouped by
        Strain / Compound / Concentration.
    iter : bool
        Use iterative (memory-efficient) parsing (default True).

    Returns
    -------
    pd.DataFrame
        Metadata table. For image metadata the index is a RangeIndex;
        for assay layouts the index is a (Row, Column) MultiIndex.
    """
    metadata = []

    if not assay_layout and iter:
        file_size = os.path.getsize(metadata_path)
        with open(metadata_path, 'rb') as f:
            pbar = tqdm(total=file_size, unit='B', unit_scale=True,
                        desc="Parsing Harmony Metadata")
            for event, elem in ET_iter.iterparse(f, events=("end",)):
                pbar.update(f.tell() - pbar.n)
                if event == "end" and "Images" in elem.tag:
                    for image_elem in elem:
                        metadata.append({
                            item.tag.split('}')[-1]: item.text
                            for item in image_elem
                        })
                    elem.clear()
            pbar.close()

    elif not assay_layout and not iter:
        try:
            root = ET_iter.parse(metadata_path).getroot()
            ns = '{http://www.perkinelmer.com/PEHH/HarmonyV5}'
            for images in root.iter(f'{ns}Images'):
                for image_elem in images:
                    metadata.append({
                        item.tag.split('}')[-1]: item.text
                        for item in image_elem
                    })
        except ET_iter.XMLSyntaxError as e:
            raise ET_iter.XMLSyntaxError(f"XML Syntax Error: {e}") from e

    if assay_layout:
        print('Consider read_harmony_assaylayout for improved V5/V6 compatibility.')
        with open(metadata_path, 'rb') as f:
            root = ET_iter.XML(f.read())
        metadata_dict = {}
        for branch in root:
            for subbranch in branch:
                if subbranch.text and subbranch.text.strip() not in ('', 'string'):
                    col_name = subbranch.text
                    metadata_dict[col_name] = {}
                for subsubbranch in subbranch:
                    if 'Row' in subsubbranch.tag:
                        r = int(subsubbranch.text)
                    elif 'Col' in subsubbranch.tag and 'Color' not in subsubbranch.tag:
                        c = int(subsubbranch.text)
                    if 'Value' in subsubbranch.tag and subsubbranch.text is not None:
                        metadata_dict[col_name][r, c] = subsubbranch.text
        metadata = metadata_dict

    df = pd.DataFrame(metadata)

    if assay_layout:
        df.index.set_names(['Row', 'Column'], inplace=True)
        if 'Cell Count' in df.columns and pd.isna(df['Cell Count']).any():
            df.drop(columns='Cell Count', inplace=True)
        if 'double' in df.columns:
            df.rename(columns={'double': 'Cell Count'}, inplace=True)
        if replicate_number:
            df['Replicate #'] = (
                df.groupby(['Strain', 'Compound', 'Concentration', 'ConcentrationEC'])
                .cumcount() + 1
            )

    print('Metadata extraction complete.')
    return df


def read_harmony_assaylayout(
    xml_path: str | Path,
    replicate_number: bool = False,
) -> pd.DataFrame:
    """Parse a PerkinElmer/Revvity Harmony assay layout XML (V5 or V6).

    Parameters
    ----------
    xml_path : str or Path
        Path to the assay layout XML file.
    replicate_number : bool
        If True, add a 'Replicate #' column when Strain + Compound +
        (Concentration and/or ConcentrationEC) columns are present.

    Returns
    -------
    pd.DataFrame
        Index = MultiIndex (Row, Column); columns = each layer name;
        values coerced to the declared ValueType where possible.
    """
    xml_path = Path(xml_path)
    root = ET.parse(xml_path).getroot()

    layers = []
    for layer in root.findall(".//{*}Layer"):
        name_el  = layer.find("./{*}Name")
        vtype_el = layer.find("./{*}ValueType")
        lname = (name_el.text or "").strip() if name_el is not None else f"Layer_{len(layers)+1}"
        vtype = (vtype_el.text or "").strip() if vtype_el is not None else None

        wells_parent = layer.find("./{*}Wells")
        well_nodes = (
            wells_parent.findall("./{*}Well")
            if wells_parent is not None
            else layer.findall("./{*}Well")
        )

        wells = []
        for w in well_nodes:
            r_el   = w.find("./{*}Row")
            c_el   = w.find("./{*}Col")
            val_el = w.find("./{*}Value")
            if r_el is None or c_el is None:
                continue
            wells.append((int(r_el.text), int(c_el.text),
                          _coerce(val_el.text if val_el is not None else None, vtype)))
        layers.append((lname, vtype, wells))

    coords = sorted({(r, c) for _, _, ws in layers for (r, c, _) in ws})
    idx = pd.MultiIndex.from_tuples(coords, names=["Row", "Column"])
    rows = {coord: {} for coord in idx}
    for lname, _, ws in layers:
        for r, c, v in ws:
            rows[(r, c)][lname] = v

    df = pd.DataFrame([rows[k] for k in idx], index=idx).where(pd.notnull, None)

    if replicate_number:
        for group in [
            ["Strain", "Compound", "Concentration", "ConcentrationEC"],
            ["Strain", "Compound", "Concentration"],
            ["Strain", "Compound", "ConcentrationEC"],
        ]:
            if set(group).issubset(df.columns):
                df["Replicate #"] = df.groupby(group, dropna=False).cumcount() + 1
                break

    return df.dropna()


def read_ffc_profile(
    ffc_xml_path: str | Path,
) -> dict[int, np.ndarray]:
    """Parse a Harmony FFC profile XML and reconstruct the flatfield correction
    surface for each channel as a 2D numpy array.

    Harmony stores the illumination profile as a 2D polynomial surface defined
    by a triangular coefficient matrix. Pixel coordinates are normalised to a
    centred unit space before evaluation using the Origin and Scale fields.

    Parameters
    ----------
    ffc_xml_path : str or Path
        Path to the FFC profile XML file
        (e.g. ``FFC_Profile/FFC_Profile_Measurement 1.xml``).

    Returns
    -------
    dict[int, np.ndarray]
        Mapping of channel ID (1-based int) to a 2D float32 surface array
        of shape (height, width) normalised so its mean equals 1.0.
        Dividing a raw tile by this surface corrects for uneven illumination.

    Notes
    -----
    The polynomial is evaluated as a sum over triangular index pairs (i, j)
    where i + j <= degree, with normalised coordinates:

        x_norm = (x - origin_x) * scale_x
        y_norm = (y - origin_y) * scale_y
        surface += coeff[i][j] * x_norm^i * y_norm^j
    """
    import ast
    import re
    import xml.etree.ElementTree as ET

    tree = ET.parse(ffc_xml_path)
    root = tree.getroot()
    ns = {'h': 'http://www.perkinelmer.com/PEHH/HarmonyV5'}

    surfaces = {}

    for entry in root.findall('.//h:Entry', ns):
        channel_id = int(entry.get('ChannelID'))
        profile_text = entry.find('h:FlatfieldProfile', ns).text

        # Extract polynomial coefficients
        coeffs_match = re.search(r'Coefficients:\s*(\[\[.*?\]\])', profile_text, re.DOTALL)
        dims_match   = re.search(r'Dims:\s*\[(\d+),\s*(\d+)\]',   profile_text, re.DOTALL)
        origin_match = re.search(r'Origin:\s*\[([0-9.]+),\s*([0-9.]+)\]', profile_text, re.DOTALL)
        scale_match  = re.search(r'Scale:\s*\[([0-9.E\-]+),\s*([0-9.E\-]+)\]', profile_text, re.DOTALL)

        coeffs = ast.literal_eval(coeffs_match.group(1))
        height, width = int(dims_match.group(1)), int(dims_match.group(2))
        origin_y, origin_x = float(origin_match.group(1)), float(origin_match.group(2))
        scale_y, scale_x = float(scale_match.group(1)), float(scale_match.group(2))

        # Build normalised coordinate grids
        ys = (np.arange(height) - origin_y) * scale_y
        xs = (np.arange(width)  - origin_x) * scale_x
        X, Y = np.meshgrid(xs, ys)

        # Evaluate triangular 2D polynomial
        # coeffs[i][j] corresponds to x^i * y^j
        surface = np.zeros((height, width), dtype=np.float32)
        for i, row in enumerate(coeffs):
            for j, coeff in enumerate(row):
                surface += coeff * (X ** i) * (Y ** j)

        # Normalise so mean == 1 to preserve overall intensity
        surface /= surface.mean()
        surfaces[channel_id] = surface

    return surfaces


def make_ffc_transform(
    surface: np.ndarray,
) -> "Callable[[np.ndarray], np.ndarray]":
    """Create a per-tile FFC correction function for use as an input transform
    in :func:`tessell8er.tile.compile_mosaic`.

    The returned function divides each tile by the flatfield surface,
    correcting for uneven illumination (vignetting) introduced by the optics.

    Parameters
    ----------
    surface : np.ndarray
        2D flatfield surface array as returned by :func:`read_ffc_profile`,
        shape (H, W), normalised so its mean equals 1.0.

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        A function that accepts a tile array and returns the corrected tile
        as the same dtype, clipped to the valid range for that dtype.

    Examples
    --------
    Parse the FFC profile, build per-channel transforms, then pass to
    compile_mosaic:

    >>> from tessell8er import dataio, tile
    >>> surfaces = dataio.read_ffc_profile('FFC_Profile/FFC_Profile_Measurement 1.xml')
    >>> # For a single-channel call, pick the relevant surface
    >>> ffc_fn = dataio.make_ffc_transform(surfaces[3])  # channel 3
    >>> mosaic = tile.compile_mosaic(
    ...     image_dir='path/to/Images/',
    ...     metadata=metadata,
    ...     row=2, col=1,
    ...     set_channel=3,
    ...     input_transforms=[ffc_fn],
    ... )
    """
    def _apply_ffc(tile: np.ndarray) -> np.ndarray:
        dtype = tile.dtype
        # Crop or pad surface to match tile size if needed
        h, w = tile.shape[:2]
        s = surface[:h, :w]
        # Avoid division by zero
        s = np.where(s == 0, 1.0, s)
        corrected = tile.astype(np.float32) / s
        # Clip to valid dtype range and restore original dtype
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            corrected = np.clip(corrected, info.min, info.max)
        return corrected.astype(dtype)

    return _apply_ffc


# ---------------------------------------------------------------------------
# OME-NGFF Zarr export
# ---------------------------------------------------------------------------

def write_ome_zarr(
    image: "ArrayLike",
    output_path: "str | Path",
    scale: "tuple[float, ...]" = (1.0, 1.0, 1.0, 1.0, 1.0),
    chunks: "tuple[int, ...]" = (1, 1, 1, 256, 256),
    compressor=None,
    overwrite: bool = True,
) -> None:
    """Write a TCZYX image array to an OME-NGFF v0.4 compliant Zarr store.

    Uses ome-zarr-py to write correct multiscales metadata and chunk
    structure. Compatible with napari-ome-zarr, Fiji/BigDataViewer, and
    other OME-NGFF-aware readers.

    The pyramid is built via :class:`ome_zarr.scaler.Scaler` (lazy for
    dask arrays) and passed directly to ``write_image``; the number of
    coordinate transformations is derived from the pyramid length so it
    always matches the number of resolution levels written.

    Parameters
    ----------
    image : array-like
        TCZYX image array (numpy or dask).
    output_path : str or Path
        Destination ``.zarr`` directory path.
    scale : tuple[float, ...]
        Physical pixel scales in (t, c, z, y, x) order. Spatial axes
        should be in micrometres (default all 1.0).
    chunks : tuple[int, ...]
        Chunk shape in TCZYX order (default (1, 1, 1, 256, 256)).
    compressor : numcodecs compressor or None
        Zarr compressor; ``None`` disables compression (default).
    overwrite : bool
        If True, remove any existing store at ``output_path`` before
        writing (default True).

    Raises
    ------
    ImportError
        If ``ome-zarr`` is not installed.
    """
    import shutil

    import ome_zarr.io
    import ome_zarr.writer
    import zarr
    from ome_zarr.scaler import Scaler

    output_path = Path(output_path)
    if overwrite and output_path.exists():
        shutil.rmtree(output_path)

    # Build pyramid via the same Scaler ome-zarr uses internally;
    # for dask arrays this is lazy — no compute triggered here.
    scaler = Scaler()
    pyramid = scaler.nearest(image)
    n_levels = len(pyramid)

    # One transform per level; XY scale doubles at each pyramid step
    coordinate_transformations = [
        [{"type": "scale", "scale": [
            scale[0],
            scale[1],
            scale[2],
            scale[3] * (2 ** i),
            scale[4] * (2 ** i),
        ]}]
        for i in range(n_levels)
    ]

    loc = ome_zarr.io.parse_url(str(output_path), mode='w')
    grp = zarr.group(loc.store)

    ome_zarr.writer.write_image(
        image=pyramid,          # pre-built pyramid skips internal rescaling
        group=grp,
        axes=[
            {"name": "t", "type": "time",    "unit": "second"},
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space",   "unit": "micrometer"},
            {"name": "y", "type": "space",   "unit": "micrometer"},
            {"name": "x", "type": "space",   "unit": "micrometer"},
        ],
        coordinate_transformations=coordinate_transformations,
        storage_options={"chunks": chunks, "compressor": compressor},
    )


def write_plate_zarr(
    image_dir: "str | Path",
    metadata: "pd.DataFrame",
    assay_layout: "pd.DataFrame",
    output_path: "str | Path",
    scale: "tuple[float, ...]" = (1.0, 1.0, 1.0, 1.0, 1.0),
    chunks: "tuple[int, ...]" = (1, 1, 1, 256, 256),
    n_tile_rows: int = 3,
    n_tile_cols: int = 3,
    compressor=None,
) -> None:
    """Write a multi-well plate to an OME-NGFF v0.4 HCS compliant Zarr store.

    Iterates over all unique (Row, Col) combinations in ``metadata``,
    compiles each well as a lazy TCZYX mosaic via
    :func:`tessell8er.tile.compile_mosaic`, and streams it directly to
    Zarr without loading into RAM. Plate and per-well condition metadata
    are written to ``.zattrs`` at the appropriate hierarchy levels.

    Resumable: wells whose field group already contains a ``.zattrs`` file
    are skipped; partially written wells are cleaned up and rewritten.

    Parameters
    ----------
    image_dir : str or Path
        Directory containing the raw tile TIFFs.
    metadata : pd.DataFrame
        Harmony image metadata as returned by
        :func:`~tessell8er.dataio.read_harmony_metadata`.
    assay_layout: "pd.DataFrame | None" = None,
        Assay layout as returned by
        :func:`~tessell8er.dataio.read_harmony_assaylayout`, with a
        (Row, Column) MultiIndex.
    output_path : str or Path
        Destination ``.zarr`` directory path.
    scale : tuple[float, ...]
        Physical pixel scales in (t, c, z, y, x) order; spatial axes in
        micrometres (default all 1.0).
    chunks : tuple[int, ...]
        Chunk shape in TCZYX order (default (1, 1, 1, 256, 256)).
    n_tile_rows : int
        Number of tile rows in each mosaic (default 3).
    n_tile_cols : int
        Number of tile columns in each mosaic (default 3).
    compressor : numcodecs compressor or None
        Zarr compressor; ``None`` disables compression (default).
    """
    import math
    import shutil

    import ome_zarr.io
    import ome_zarr.writer
    import zarr
    from tqdm.auto import tqdm

    from tessell8er import tile as tile_module

    output_path = Path(output_path)
    wells       = metadata[['Row', 'Col']].drop_duplicates().values
    rows_sorted = sorted({r for r, _ in wells}, key=int)
    cols_sorted = sorted({c for _, c in wells}, key=int)

    print(f"Output path  : {output_path}")
    print(f"Wells found  : {len(wells)}")
    print(f"Rows         : {rows_sorted}")
    print(f"Cols         : {cols_sorted}")

    loc  = ome_zarr.io.parse_url(str(output_path), mode='w')
    root = zarr.group(loc.store)

    if 'plate' not in root.attrs:
        print("Writing plate-level metadata...")
        root.attrs.update({
            'plate': {
                'version':      '0.4',
                'name':         output_path.stem,
                'field_count':  1,
                'acquisitions': [{'id': 0}],
                'columns': [{'name': str(c)} for c in cols_sorted],
                'rows':    [{'name': str(r)} for r in rows_sorted],
                'wells': [
                    {
                        'path':        f'{r}/{c}',
                        'rowIndex':    rows_sorted.index(r),
                        'columnIndex': cols_sorted.index(c),
                    }
                    for r, c in wells
                ],
            },
            **({'assay_layout': assay_layout.reset_index().to_dict(orient='records')}
                if assay_layout is not None else {}),        
        })
        print("Plate-level metadata written.")
    else:
        print("Plate-level metadata already present, skipping.")

    for row, col in tqdm(wells, desc="Writing wells"):
        field_path  = output_path / str(row) / str(col) / '0'
        zattrs_path = field_path / '.zattrs'

        if zattrs_path.exists():
            print(f"  [{row},{col}] Skipping — already complete")
            continue

        if field_path.exists():
            print(f"  [{row},{col}] Partial write detected, cleaning up...")
            shutil.rmtree(field_path)

        print(f"  [{row},{col}] Compiling mosaic...")
        images = tile_module.compile_mosaic(
            image_dir=image_dir,
            metadata=metadata,
            row=int(row), col=int(col),
            n_tile_rows=n_tile_rows,
            n_tile_cols=n_tile_cols,
        )
        print(f"  [{row},{col}] Mosaic shape: {images.shape}, dtype: {images.dtype}")

        layout_row = assay_layout.loc[(int(row), int(col))]

        well_grp = root.require_group(f'{row}/{col}')
        # per-well — skip condition block if no layout provided
        if assay_layout is not None:
            layout_row = assay_layout.loc[(int(row), int(col))]
            well_grp.attrs.update({
                'well':      {'version': '0.4', 'images': [{'path': '0', 'acquisition': 0}]},
                'condition': layout_row.to_dict(),
            })
            print(f"  [{row},{col}] Done — {layout_row.to_dict()}")
        else:
            well_grp.attrs.update({
                'well': {'version': '0.4', 'images': [{'path': '0', 'acquisition': 0}]},
            })
            print(f"  [{row},{col}] Done")

        img_grp  = well_grp.require_group('0')
        min_dim  = min(images.shape[-2], images.shape[-1])
        n_levels = max(1, math.floor(math.log2(min_dim / 64)) + 1)
        print(f"  [{row},{col}] Writing {n_levels} pyramid levels...")

        coordinate_transformations = [
            [{'type': 'scale', 'scale': [
                scale[0], scale[1], scale[2],
                scale[3] * (2 ** i),
                scale[4] * (2 ** i),
            ]}]
            for i in range(n_levels)
        ]

        ome_zarr.writer.write_image(
            image=images,
            group=img_grp,
            scale_factors=[2] * (n_levels - 1),
            axes=[
                {'name': 't', 'type': 'time',    'unit': 'second'},
                {'name': 'c', 'type': 'channel'},
                {'name': 'z', 'type': 'space',   'unit': 'micrometer'},
                {'name': 'y', 'type': 'space',   'unit': 'micrometer'},
                {'name': 'x', 'type': 'space',   'unit': 'micrometer'},
            ],
            coordinate_transformations=coordinate_transformations,
            storage_options={'chunks': chunks, 'compressor': compressor},
        )
        print(f"  [{row},{col}] Done — {layout_row.to_dict()}")

    print(f"\nPlate write complete: {output_path}")


# ---------------------------------------------------------------------------
# URL / filename utilities
# ---------------------------------------------------------------------------

def generate_url(row: pd.Series) -> str:
    """Generate the local tile filename for a single row of Harmony metadata.

    Replaces remote URLs in exported metadata with the standardised local
    filename format used by Opera Phenix.

    Parameters
    ----------
    row : pd.Series
        A single metadata row with columns: Row, Col, FieldID, PlaneID,
        ChannelID, TimepointID, FlimID.

    Returns
    -------
    str
        Formatted filename, e.g. ``r01c02f03p01-ch1sk1fk1fl1.tiff``.
    """
    return (
        f"r{row['Row'].zfill(2)}c{row['Col'].zfill(2)}"
        f"f{row['FieldID'].zfill(2)}p{row['PlaneID'].zfill(2)}"
        f"-ch{row['ChannelID']}sk{int(row['TimepointID']) + 1}"
        f"fk1fl{row['FlimID']}.tiff"
    )


# ---------------------------------------------------------------------------
# File-size utilities
# ---------------------------------------------------------------------------

def get_folder_size(folder: str | Path) -> "ByteSize":
    """Return the total size of all files in `folder` as a :class:`ByteSize`.

    Parameters
    ----------
    folder : str or Path
        Root directory to measure.

    Returns
    -------
    ByteSize
        Human-readable byte-size object.
    """
    return ByteSize(sum(f.stat().st_size for f in Path(folder).rglob('*')))


class ByteSize(int):
    """Integer subclass that formats itself as a human-readable byte size.

    Examples
    --------
    >>> bs = ByteSize(1_048_576)
    >>> str(bs)
    '1.00 MB'
    >>> bs.gigabytes
    0.0009765625
    """

    _KB = 1024
    _suffixes = 'B', 'KB', 'MB', 'GB', 'PB'

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        self.bytes = self.B = int(self)
        self.kilobytes = self.KB = self / self._KB**1
        self.megabytes = self.MB = self / self._KB**2
        self.gigabytes = self.GB = self / self._KB**3
        self.petabytes = self.PB = self / self._KB**4
        *suffixes, last = self._suffixes
        suffix = next(
            (s for s in suffixes if 1 < getattr(self, s) < self._KB), last
        )
        self.readable = suffix, getattr(self, suffix)
        super().__init__()

    def __str__(self):
        return self.__format__('.2f')

    def __repr__(self):
        return f'{self.__class__.__name__}({super().__repr__()})'

    def __format__(self, format_spec):
        suffix, val = self.readable
        return f'{val:{format_spec}} {suffix}'

    def __add__(self, other):  return self.__class__(super().__add__(other))
    def __sub__(self, other):  return self.__class__(super().__sub__(other))
    def __mul__(self, other):  return self.__class__(super().__mul__(other))
    def __radd__(self, other): return self.__class__(super().__add__(other))
    def __rsub__(self, other): return self.__class__(super().__sub__(other))
    def __rmul__(self, other): return self.__class__(super().__rmul__(other))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _coerce(val_text: str | None, value_type: str | None):
    """Coerce an XML value string to the declared Python type."""
    if val_text is None:
        return None
    s = val_text.strip()
    if not s:
        return None
    vt = (value_type or "").strip().lower()
    if vt in {"double", "float"}:
        try: return float(s)
        except ValueError: return s
    if vt in {"int", "integer"}:
        try: return int(s)
        except ValueError:
            try: return int(float(s))
            except ValueError: return s
    if vt in {"bool", "boolean"}:
        return s.lower() in {"true", "1", "yes"}
    return s