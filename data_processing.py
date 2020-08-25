###### To generate Color Data ######
import rasterio as rio
from affine import Affine

colour_data = []


def generate_colour_data(width, height, imagiry_data, pixel2coord):
    for i in range(1, height):
        for j in range(1, width):
            colour_data.append(
                [
                    pixel2coord(j, i)[0],
                    pixel2coord(j, i)[1],
                    imagiry_data.read([1])[0][i - 1][j - 1],
                    imagiry_data.read([2])[0][i - 1][j - 1],
                    imagiry_data.read([3])[0][i - 1][j - 1],
                    imagiry_data.read([4])[0][i - 1][j - 1],
                ]
            )


with rio.open("PATH_TIF_FILE") as imagery_data:
    T0 = imagery_data.transform
    T1 = T0 * Affine.translation(0.5, 0.5)
    pixel2coord = lambda c, r: (c, r) * T1
    width = imagery_data.width
    height = imagery_data.height
    generate_colour_data(width, height, imagery_data, pixel2coord)


import xyzspaces as xyz

xyz_token = "[YOUR-XYZ-TOKEN]"

xyz = xyz.XYZ(credentials=xyz_token)

title = "LIDAR COLOR DATA"
description = "LIDAR COLOR DATA"

space = xyz.spaces.new(title=title, description=description)


import concurrent.futures
import time
from functools import partial
from multiprocessing import Manager, Process
from geojson import Feature, Point
from xyzspaces.utils import grouper

manager = Manager()


def upload_features(features, space):
    fc = []
    try:
        for data in features:
            if data:
                lat, lng = data[1], data[0]
                f = Feature(
                    geometry=Point((lng, lat)),
                    properties={
                        "R": float(data[2]),
                        "G": float(data[3]),
                        "B": float(data[4]),
                    },
                )
                fc.append(f)
        feature_collection = dict(type="FeatureCollection", features=fc)
        space.add_features(features=feature_collection)
        time.sleep(3)
    except Exception as e:
        print(e)


features_size = 1000

groups = grouper(features_size, colour_data)

part_func = partial(upload_features, space=space)

with concurrent.futures.ProcessPoolExecutor() as executor:
    executor.map(part_func, groups, chunksize=3)

####################################

###### To generate LIDAR Data ######

import pandas as pd

csv_data = pd.read_csv("PATH_TEXT_FILE")
csv_data[3] = -1
csv_data[4] = -1
xa = csv_data.to_numpy()

####################################

###### To generate Final Data ######

from math import radians, cos, sin, asin, sqrt
from scipy.spatial.distance import cdist


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r


def func(p1, p2):
    return haversine(p1[0], p1[1], p2[0], p2[1])


import concurrent.futures
import time
from functools import partial
from multiprocessing import Manager, Process
import numpy as np
from geojson import Feature, Point
from xyzspaces.utils import grouper

manager = Manager()
final_data = manager.list()


def gen_feature_color(f, space_color, func):
    try:
        for d in f:
            fl = []
            for f in space_color.spatial_search(lon=d[0], lat=d[1], radius=3):
                fl.append(
                    [
                        f["geometry"]["coordinates"][0],
                        f["geometry"]["coordinates"][1],
                        f["properties"]["R"],
                        f["properties"]["G"],
                        f["properties"]["B"],
                    ]
                )
            closest_index = cdist(
                XA=np.array([d]), XB=np.array(fl), metric=func
            ).argmin()
            rgb = fl[closest_index]
            final_data.append([d[0], d[1], d[2], rgb[2], rgb[3], rgb[4]])
            print(len(final_data))
            time.sleep(1)
    except Exception as e:
        print(e)


features_size = 1000

groups = grouper(features_size, xa)

part_func = partial(gen_feature_color, space_color=space, func=func)

with concurrent.futures.ProcessPoolExecutor(max_workers=60) as executor:
    executor.map(part_func, groups, chunksize=3)


from pandas import DataFrame

df = DataFrame(list(final_data))

df.to_json("FINAL_DATA.json", orient="values")

####################################
