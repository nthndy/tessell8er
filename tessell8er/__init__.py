"""
tessell8er
==========
Lazy Dask-based image tiling and stitching for Harmony/Opera Phenix data.

    *Tessellate* the image — just not yet.

Modules
-------
tile    Core stitching engine (:func:`~tessell8er.tile.compile_mosaic`)
dataio  Harmony metadata I/O (:func:`~tessell8er.dataio.read_harmony_metadata`)
"""

from . import dataio, tile

__version__ = "0.1.0"
__author__  = "Nathan J. Day"
__all__     = ["tile", "dataio"]
