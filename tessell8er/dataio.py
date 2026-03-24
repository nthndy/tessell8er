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
