"""
tile.py
=======
Core image tiling and stitching engine for tessell8er.

Lazily stitches fragmented image tiles (e.g. exported from Harmony/Opera Phenix)
into a single Dask array using affine transformations and Shapely geometry for
chunk-tile intersection. Computation is deferred until `.compute()` is called.
"""

import glob
import logging
import os
import warnings
from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path
from typing import Any, Union

import dask
import dask.array as da
import numpy as np
import pandas as pd
from dask.array.core import normalize_chunks
from scipy.ndimage import affine_transform
from shapely.errors import ShapelyDeprecationWarning
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import Polygon
from shapely.strtree import STRtree
from skimage.io import imread
from skimage.transform import AffineTransform

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

FilePath = Path | str


class ShapeWithInfo:
    """Thin wrapper pairing a Shapely geometry with an arbitrary metadata dict.

    Newer Shapely versions do not allow setting custom attributes directly on
    geometry objects, so fuse_info is carried here instead.
    """

    def __init__(self, geometry: BaseGeometry, fuse_info: dict):
        self.geometry = geometry
        self.fuse_info = fuse_info
ArrayLike = Union[np.ndarray, "dask.array.Array"]
logger = logging.getLogger(__name__)


class FileNotFoundError(Exception):
    """Raised when an expected image tile file cannot be found on disk."""
    pass


def find_files_exist(fns: list[str], image_dir: str) -> None:
    """Check that every filename in `fns` exists inside `image_dir`.

    Parameters
    ----------
    fns : list[str]
        Filenames to verify.
    image_dir : str
        Directory to search within.

    Raises
    ------
    FileNotFoundError
        If any file is absent.
    """
    for fn in fns:
        file_path = os.path.join(image_dir, fn)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")


def compile_mosaic(
    image_dir: os.PathLike,
    metadata: pd.DataFrame,
    row: int,
    col: int,
    input_transforms: list[Callable[[np.ndarray], np.ndarray]] | None = None,
    set_plane: Any | None = None,
    set_channel: int | None = None,
    set_time: int | None = None,
    overlap_percentage: float = 0.1,
    subset_field_IDs: list[int] | None = None,
    n_tile_rows: int | None = None,
    n_tile_cols: int | None = None,
) -> da.Array:
    """Build a lazily-evaluated Dask mosaic from fragmented Harmony image tiles.

    Stitches tiles for the specified well position across all (or a subset of)
    timepoints, channels and Z-planes into a single 5-D array shaped (T, C, Z, Y, X).
    No pixel data is loaded until `.compute()` is called.

    For non-square mosaics or when stitching a subset of fields, supply
    `subset_field_IDs` and the matching `n_tile_rows` / `n_tile_cols`.

    Parameters
    ----------
    image_dir : os.PathLike
        Directory containing the raw tile images.
    metadata : pd.DataFrame
        Harmony metadata DataFrame (from :func:`tessell8er.dataio.read_harmony_metadata`).
    row : int
        Well row index.
    col : int
        Well column index.
    input_transforms : list[Callable] or None
        Optional per-tile pre-processing functions applied on load.
    set_plane : int, 'max_proj', 'sum_proj', or None
        Z-plane selector. Pass a string for Z-projections; None uses all planes.
    set_channel : int or None
        Channel selector; None uses all channels.
    set_time : int or None
        Timepoint selector; None uses all timepoints.
    overlap_percentage : float
        Fractional tile overlap (default 0.1 = 10 %).
    subset_field_IDs : list[int] or None
        Restrict stitching to these field IDs.
    n_tile_rows : int or None
        Tile rows in the mosaic. Required when mosaic is non-square.
    n_tile_cols : int or None
        Tile columns in the mosaic. Required when mosaic is non-square.

    Returns
    -------
    da.Array
        Lazy Dask array shaped (T, C, Z, Y, X).

    Raises
    ------
    ValueError
        If `row` or `col` is absent from `metadata`.
    TypeError
        If `set_plane` is a string other than 'max_proj' or 'sum_proj'.
    """
    if str(row) not in metadata['Row'].unique():
        raise ValueError("Row not found in metadata.")
    if str(col) not in metadata['Col'].unique():
        raise ValueError("Column not found in metadata.")

    if isinstance(set_plane, str):
        if set_plane not in ['max_proj', 'sum_proj']:
            raise TypeError(
                "set_plane must be an int, 'max_proj', or 'sum_proj'."
            )
        projection = set_plane
        set_plane = None
    else:
        projection = None

    channel_IDs = metadata['ChannelID'].unique() if set_channel is None else [set_channel]
    plane_IDs   = metadata['PlaneID'].unique()   if set_plane   is None else [set_plane]
    timepoint_IDs = metadata['TimepointID'].unique() if set_time is None else [set_time]

    sample_fn = metadata['URL'][
        (metadata['Row'] == str(row)) & (metadata['Col'] == str(col))
    ].iloc[0]
    dtype = imread(image_dir + f'/{sample_fn}').dtype

    number_tiles = len(subset_field_IDs) if subset_field_IDs else int(metadata['FieldID'].max())
    if not n_tile_rows:
        n_tile_rows = n_tile_cols = np.sqrt(number_tiles)

    tile_size = int(metadata['ImageSizeX'].max())
    final_image_size(tile_size, overlap_percentage, n_tile_rows, n_tile_cols)

    load_transform_image = partial(load_image, transforms=input_transforms)

    images = [
        stitch(
            load_transform_image, metadata, image_dir,
            time, plane, channel, str(row), str(col),
            n_tile_rows, n_tile_cols, subset_field_IDs,
        )[0]
        for time    in timepoint_IDs
        for channel in channel_IDs
        for plane   in plane_IDs
    ]

    images = [frame.rechunk(tile_size, tile_size) for frame in images]
    images = da.stack(images, axis=0)
    images = images.reshape((
        len(timepoint_IDs), len(channel_IDs), len(plane_IDs),
        images.shape[-2], images.shape[-1],
    ))

    if projection == 'max_proj':
        images = da.max(images, axis=2)
    elif projection == 'sum_proj':
        max_value = np.iinfo(dtype).max
        images = da.clip(da.sum(images, axis=2), 0, max_value).astype(dtype)

    return images


def stitch(
    load_transform_image: partial,
    df: pd.DataFrame,
    image_dir: str,
    time: int,
    plane: int,
    channel: int,
    row: int,
    col: int,
    n_tile_rows: int,
    n_tile_cols: int,
    subset_field_IDs=None,
) -> tuple[da.Array, list]:
    """Stitch a single (T, C, Z) frame from individual image tiles.

    Parameters
    ----------
    load_transform_image : partial
        Tile loading function (wraps :func:`load_image`).
    df : pd.DataFrame
        Harmony metadata.
    image_dir : str
        Directory of tile images.
    time, plane, channel : int
        Frame indices.
    row, col : int
        Well position.
    n_tile_rows, n_tile_cols : int
        Mosaic dimensions.
    subset_field_IDs : list or None
        Optional field ID filter.

    Returns
    -------
    frame : da.Array
        Stitched 2-D Dask array for this frame.
    tiles_shifted_shapely : list
        Shapely geometries of placed tiles (for debugging / visualisation).
    """
    conditions = (
        (df['TimepointID'] == str(time)) &
        (df['PlaneID']     == str(plane)) &
        (df['ChannelID']   == str(channel)) &
        (df['Row']         == str(row)) &
        (df['Col']         == str(col))
    )
    filtered_df = df[conditions]
    if subset_field_IDs:
        filtered_df = filtered_df[filtered_df['FieldID'].isin(subset_field_IDs)]

    fns = filtered_df['URL']
    find_files_exist(fns, image_dir)
    fns = [glob.glob(os.path.join(image_dir, fn))[0] for fn in fns]

    sample = imread(fns[0])
    _fuse_func = partial(fuse_func, imload_fn=load_transform_image, dtype=sample.dtype)

    coords = filtered_df[
        ["URL", "PositionX", "PositionY", "ImageResolutionX", "ImageResolutionY"]
    ].copy()
    coords['PositionXPix'] = coords['PositionX'].astype(float) / coords['ImageResolutionX'].astype(float)
    coords['PositionYPix'] = coords['PositionY'].astype(float) / coords['ImageResolutionY'].astype(float)
    norm_coords = list(zip(coords['PositionXPix'], coords['PositionYPix']))

    transforms = [AffineTransform(translation=c).params for c in norm_coords]
    tiles = [transform_tile_coord(sample.shape, t) for t in transforms]
    all_bboxes = np.vstack(tiles)
    stitched_shape = tuple(np.round(all_bboxes.max(axis=0) - all_bboxes.min(axis=0)).astype(int))
    shift_to_origin = AffineTransform(translation=-all_bboxes.min(axis=0))
    transforms_with_shift = [t @ shift_to_origin.params for t in transforms]
    shifted_tiles = [transform_tile_coord(sample.shape, t) for t in transforms_with_shift]

    chunk_size = (int(stitched_shape[0] / n_tile_rows), int(stitched_shape[1] / n_tile_cols))
    chunks = normalize_chunks(chunk_size, shape=stitched_shape)
    assert np.all(
        np.array(stitched_shape) == np.array(list(map(sum, chunks)))
    ), "Chunks do not fit into mosaic size."
    chunk_boundaries = list(get_chunk_coord(stitched_shape, chunk_size))

    tiles_shifted_shapely = [
        ShapeWithInfo(numpy_shape_to_shapely(s), {'file': file, 'transform': transform})
        for s, file, transform in zip(shifted_tiles, fns, transforms_with_shift)
    ]
    chunk_shapes   = [get_rect_from_chunk_boundary(b) for b in chunk_boundaries]
    chunks_shapely = [
        ShapeWithInfo(numpy_shape_to_shapely(c), {'chunk_boundary': b})
        for c, b in zip(chunk_shapes, chunk_boundaries)
    ]

    chunk_tiles = find_chunk_tile_intersections(tiles_shifted_shapely, chunks_shapely)
    frame = da.map_blocks(func=_fuse_func, chunks=chunks, input_tile_info=chunk_tiles, dtype=sample.dtype)
    frame = da.rot90(frame)

    return frame, tiles_shifted_shapely


def transform_tile_coord(shape: tuple[int, int], affine_matrix: np.ndarray) -> np.ndarray:
    """Return the four corner coordinates of a tile after an affine transform.

    Parameters
    ----------
    shape : tuple[int, int]
        Tile height and width in pixels.
    affine_matrix : np.ndarray
        3×3 affine matrix.

    Returns
    -------
    np.ndarray
        (4, 2) array of transformed corner coordinates.
    """
    h, w = shape
    baserect = np.array([[0, 0], [h, 0], [h, w], [0, w]])
    augmented = np.concatenate((baserect, np.ones((4, 1))), axis=1)
    return (affine_matrix @ augmented.T).T[:, :-1]


def get_chunk_coord(
    shape: tuple[int, int], chunk_size: tuple[int, int]
) -> Iterator[tuple[tuple[int, int], tuple[int, int]]]:
    """Yield bounding box coordinates for every chunk of a Dask array.

    Parameters
    ----------
    shape : tuple[int, int]
        Full array shape.
    chunk_size : tuple[int, int]
        Target chunk size.

    Yields
    ------
    tuple
        ((y_start, y_end), (x_start, x_end)) for each chunk.
    """
    chunksy, chunksx = normalize_chunks(chunk_size, shape=shape)
    y = 0
    for cy in chunksy:
        x = 0
        for cx in chunksx:
            yield ((y, y + cy), (x, x + cx))
            x += cx
        y += cy


def numpy_shape_to_shapely(coords: np.ndarray, shape_type: str = "polygon") -> BaseGeometry:
    """Convert a (N, 2) numpy coordinate array to a Shapely geometry.

    Parameters
    ----------
    coords : np.ndarray
        Corner coordinates in (row, col) / (y, x) order.
    shape_type : str
        'polygon' (default) or 'line'.

    Returns
    -------
    BaseGeometry
        Shapely Polygon or LineString.
    """
    _coords = coords[:, ::-1].copy()
    _coords[:, 1] *= -1
    if shape_type in ("rectangle", "polygon", "ellipse"):
        return Polygon(_coords)
    elif shape_type in ("line", "path"):
        return LineString(_coords)
    raise ValueError(f"Unsupported shape_type: '{shape_type}'")


def get_rect_from_chunk_boundary(
    chunk_boundary: tuple[tuple[int, int], tuple[int, int]]
) -> np.ndarray:
    """Convert a chunk boundary tuple to a rectangle corner array.

    Parameters
    ----------
    chunk_boundary : tuple
        ((y_min, y_max), (x_min, x_max))

    Returns
    -------
    np.ndarray
        (4, 2) rectangle in (row, col) order.
    """
    ylim, xlim = chunk_boundary
    miny, maxy = ylim[0], ylim[1] - 1
    minx, maxx = xlim[0], xlim[1] - 1
    return np.array([[miny, minx], [maxy, minx], [maxy, maxx], [miny, maxx]])


def find_chunk_tile_intersections(
    tiles_shapely: list[BaseGeometry],
    chunks_shapely: list[BaseGeometry],
) -> dict[tuple[int, int], list[tuple]]:
    """Map each output chunk to the image tiles that overlap it.

    Uses an STRtree spatial index for efficient intersection queries.

    Parameters
    ----------
    tiles_shapely : list[BaseGeometry]
        Shapely geometries of all placed tiles (carrying `.fuse_info`).
    chunks_shapely : list[BaseGeometry]
        Shapely geometries of all Dask chunks (carrying `.fuse_info`).

    Returns
    -------
    dict
        Mapping from chunk anchor point (y, x) to list of (file, transform) tuples.
    """
    chunk_to_tiles = {}
    tile_tree = STRtree([t.geometry for t in tiles_shapely])
    for chunk in chunks_shapely:
        boundary = chunk.fuse_info["chunk_boundary"]
        anchor = (boundary[0][0], boundary[1][0])
        chunk_to_tiles[anchor] = [
            (tiles_shapely[i].fuse_info["file"], tiles_shapely[i].fuse_info["transform"])
            for i in tile_tree.query(chunk.geometry)
        ]
    return chunk_to_tiles


def fuse_func(
    input_tile_info: dict,
    imload_fn: Callable | None = imread,
    block_info=None,
    dtype=np.uint16,
) -> np.ndarray:
    """Dask map_blocks callback: fuse intersecting tiles into a single chunk.

    Each tile is affine-transformed into chunk space and max-projected with
    the accumulator, so overlapping regions show the brightest pixel.

    Parameters
    ----------
    input_tile_info : dict
        Output of :func:`find_chunk_tile_intersections`.
    imload_fn : Callable or None
        Function used to load tiles from disk.
    block_info : dict
        Injected by Dask; contains array location and chunk shape.
    dtype : data-type
        Output dtype (default np.uint16).

    Returns
    -------
    np.ndarray
        Fused chunk array.
    """
    array_location = block_info[None]["array-location"]
    anchor_point = (array_location[0][0], array_location[1][0])
    chunk_shape = block_info[None]["chunk-shape"]
    fused = np.zeros(chunk_shape, dtype=dtype)

    for image_repr, tile_affine in input_tile_info[anchor_point]:
        im = imload_fn(image_repr) if imload_fn is not None else image_repr
        shift = AffineTransform(translation=(-anchor_point[0], -anchor_point[1]))
        tile_shifted = affine_transform(
            im,
            matrix=np.linalg.inv(shift.params @ tile_affine),
            output_shape=chunk_shape,
            cval=0,
        )
        fused = np.maximum(fused, tile_shifted.astype(dtype))

    return fused


def load_image(
    file: str | Path,
    transforms: list[Callable[[np.ndarray], np.ndarray]] | None = None,
) -> np.ndarray:
    """Load an image tile from disk with optional pre-processing transforms.

    Applies a 270° rotation to bridge Cartesian stage coordinates with
    Python image (row, col) coordinates.

    Parameters
    ----------
    file : str or Path
        Path to the tile image.
    transforms : list[Callable] or None
        Optional sequence of functions applied to the array after loading.

    Returns
    -------
    np.ndarray
        Loaded (and optionally transformed) tile array.
    """
    try:
        img = imread(file)
    except Exception as e:
        raise Exception(f'{e}\nCould not load file: {file}') from e

    img = np.rot90(img, k=3)

    if transforms is not None:
        for transform in transforms:
            img = transform(img)

    return img


def final_image_size(
    size_of_tile: int,
    overlap_percentage: float,
    n_tile_rows: int,
    n_tile_cols: int,
) -> tuple[int, int]:
    """Calculate the pixel dimensions of the stitched mosaic.

    Parameters
    ----------
    size_of_tile : int
        Side length of a single square tile in pixels.
    overlap_percentage : float
        Fractional overlap between adjacent tiles (e.g. 0.1 = 10 %).
    n_tile_rows : int
        Number of tile rows.
    n_tile_cols : int
        Number of tile columns.

    Returns
    -------
    tuple[int, int]
        (width, height) of the final stitched image in pixels.
    """
    overlap = overlap_percentage * size_of_tile
    width  = int((n_tile_cols * size_of_tile) - ((n_tile_cols - 1) * overlap))
    height = int((n_tile_rows * size_of_tile) - ((n_tile_rows - 1) * overlap))
    return width, height
