import functools
import shapely.ops as shops
import shapely.geometry as geom 

# TODO NOT USED REMOVE
def geoms_to_list(func):
    @functools.wraps(func)
    def wrapper_geom_type_conv(input):
        res = func(input)
        if type(res) == type(geom.MultiPolygon()) or type(res) == type(geom.MultiLineString()):
            return list(res.geoms)
        elif type(res) == type(geom.Polygon()) or type(res) == type(geom.LineString()):
            return [res]
        else:
            return [res] # TODO: needed?
    return wrapper_geom_type_conv

class Utils():
    pass