import functools
import shapely.geometry as geom
import numpy as np

def geoms_to_tuple(func):
    @functools.wraps(func)
    def wrapper_geoms_to_tuple(input):
        res = func(input)
        # if type(res) == type(geom.MultiPolygon()) or type(res) == type(geom.MultiLineString()):
        #     return tuple(res.geoms)
        # elif type(res) == type(geom.Polygon()) or type(res) == type(geom.LineString()):
        #     return (res,)
        # else:
        #     return (res,) # TODO: REMOVE IF SURE THAT WORKING
        if 'Multi' in res.geom_type:
            return tuple(res.geoms)
        else:
            return (res,)
    return wrapper_geoms_to_tuple

class Utils():
    
    @staticmethod
    @geoms_to_tuple
    def geometric_difference(objs):
        return objs[0].difference(objs[1])

    @staticmethod
    def flatten(l) -> list:
        return [item for sublist in l for item in sublist]

    @staticmethod
    def cart2pol(x, y):
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return (rho, phi)

    @staticmethod
    def pol2cart(rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return (x, y)