from pyproj import CRS, Proj
from pyproj import transformer

stockholm = (370,670)
tru_stockholm = (59.3293, 18.0686)

def xy_to_latlong(x,y):

    """input: pixel (x,y), output: (lat,long)"""


where_dict = {
    'LL_lat': 53.987947379235436,
    'LL_lon': 9.25569438102197,
    'LR_lat': 53.706519377463586,
    'LR_lon': 22.761914608556413,
    'UL_lat': 69.80759428237813,
    'UL_lon': 5.323837778285936,
    'UR_lat': 69.2640415703203,
    'UR_lon': 29.82199602636603,
    'projdef': b'+proj=stere +ellps=bessel +lat_0=90 +lon_0=14 +lat_ts=60 +datum=WGS84',
    'xscale': 2000.0,
    'xsize': 458,
    'yscale': 2000.0,
    'ysize': 881
    }

crs = CRS.from_proj4('+proj=stere +ellps=bessel +lat_0=90 +lon_0=14 +lat_ts=60 +datum=WGS84')
outProj =pyproj.Proj(init='epsg:4326')
trans = transformer.Transformer.from_crs(crs)
