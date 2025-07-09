import asyncio
import os

from geomltoolkits.downloader import osm as OSMDownloader
from geomltoolkits.downloader import tms as TMSDownloader

ZOOM = 18
DOWNLOAD_DATA_DIR = os.path.join(os.getcwd(),"meta/data/banepa")
TMS = "https://tiles.openaerialmap.org/62d85d11d8499800053796c1/0/62d85d11d8499800053796c2/{z}/{x}/{y}"
BBOX = [85.514668, 27.628367, 85.528875, 27.638514]

os.makedirs(DOWNLOAD_DATA_DIR, exist_ok=True)

# image chips
asyncio.run(TMSDownloader.download_tiles(
    tms=TMS,
    zoom=ZOOM,
    out=DOWNLOAD_DATA_DIR,
    bbox=BBOX,
    georeference=True,
    dump_tile_geometries_as_geojson=True,
    prefix="OAM"
))

# osm data download
tiles_geojson = os.path.join(DOWNLOAD_DATA_DIR, "tiles.geojson")
asyncio.run(OSMDownloader.download_osm_data(
    geojson=tiles_geojson,
    out=os.path.join(DOWNLOAD_DATA_DIR, "labels"),
    dump_results=True,
    split_output_by_tiles=True,
    burn_splits_to_raster=True,
    burn_value=255,
))

import glob
import shutil

TRAINING_DEST_DIR = os.path.join(os.getcwd(), "data/train/banepa")

os.makedirs(os.path.join(TRAINING_DEST_DIR, "chips"), exist_ok=True)
os.makedirs(os.path.join(TRAINING_DEST_DIR, "labels"), exist_ok=True)


chip_files = glob.glob(os.path.join(DOWNLOAD_DATA_DIR, 'chips', '*.tif'))
for file in chip_files:
    filename = os.path.basename(file)
    shutil.copy2(file, os.path.join(TRAINING_DEST_DIR, "chips", filename))

src_dest_pairs = [
    (os.path.join(DOWNLOAD_DATA_DIR, 'labels', 'split', 'mask'), 
     os.path.join(TRAINING_DEST_DIR, "labels"))
]

# Copy files and clean up
for src, dest in src_dest_pairs:
    shutil.copytree(src, dest, dirs_exist_ok=True)
shutil.rmtree(DOWNLOAD_DATA_DIR)