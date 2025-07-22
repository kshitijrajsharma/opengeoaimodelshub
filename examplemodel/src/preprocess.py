#!/usr/bin/env python3
import argparse
import asyncio
import glob
import os
import shutil

from geomltoolkits.downloader import osm as OSMDownloader
from geomltoolkits.downloader import tms as TMSDownloader


def parse_args():
    p = argparse.ArgumentParser(
        description="Download TMS tiles + OSM labels, split and copy into train folders"
    )
    p.add_argument("--zoom", type=int, default=19, help="Tile zoom level")
    p.add_argument(
        "--bbox", type=str, default="85.51991979758662,27.628837632373674,85.52736620395387,27.633394557789373",
        help="Bounding box as a list: min_lon, min_lat, max_lon, max_lat"
    )
    p.add_argument(
        "--tms", type=str, default="https://tiles.openaerialmap.org/62d85d11d8499800053796c1/0/62d85d11d8499800053796c2/{z}/{x}/{y}",
        help="Tile service URL template, e.g. https://…/{z}/{x}/{y}"
    )
    p.add_argument(
        "--download-dir", "-d", type=str,
        default=os.path.join(os.getcwd(), "meta/data"),
        help="Where to download raw tiles + OSM"
    )
    p.add_argument(
        "--train-dir", "-o", type=str,
        default=os.path.join(os.getcwd(), "data/train/banepa"),
        help="Output train directory"
    )
    return p.parse_args()

async def download_tiles_and_osm(args):

    await TMSDownloader.download_tiles(
        tms=args.tms,
        zoom=args.zoom,
        out=args.download_dir,
        bbox=[float(x) for x in args.bbox.split(",")],
        georeference=True,
        dump_tile_geometries_as_geojson=True,
        prefix="OAM"
    )

    tiles_geojson = os.path.join(args.download_dir, "tiles.geojson")
    await OSMDownloader.download_osm_data(
        geojson=tiles_geojson,
        out=os.path.join(args.download_dir, "labels"),
        dump_results=True,
        split_output_by_tiles=True,
        burn_splits_to_raster=True,
        burn_value=255,
    )

def copy_and_cleanup(args):

    chips_out  = os.path.join(args.train_dir, "chips")
    labels_out = os.path.join(args.train_dir, "labels")
    os.makedirs(chips_out,  exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)


    for tif in glob.glob(os.path.join(args.download_dir, "chips", "*.tif")):
        shutil.copy2(tif, chips_out)


    src = os.path.join(args.download_dir, "labels", "split", "mask")
    shutil.copytree(src, labels_out, dirs_exist_ok=True)


    shutil.rmtree(args.download_dir)

def main():
    args = parse_args()
    os.makedirs(args.download_dir, exist_ok=True)
    asyncio.run(download_tiles_and_osm(args))
    copy_and_cleanup(args)
    print(f"Data ready under {args.train_dir}")

if __name__ == "__main__":
    main()