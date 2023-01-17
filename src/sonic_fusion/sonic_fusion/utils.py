from itertools import compress
import time
import functools
import shapely.geometry as geom
import shapely.ops as shops
import shapely.predicates as shpred
import numpy as np
import shapely
import shapely.affinity as shaffin
from scipy.spatial.transform import Rotation

import geometry_msgs.msg as geomsgs
import sensor_msgs

def geoms_to_tuple(func):
    @functools.wraps(func)
    def wrapper_geoms_to_tuple(input):
        res = func(input)
        if any((n in res.geom_type) for n in ('Multi','Collection')):
            return tuple(res.geoms)
        else:
            return (res,)
    return wrapper_geoms_to_tuple

class Utils():
    
    @staticmethod
    @geoms_to_tuple
    def geometric_difference(objs: tuple):
        return objs[0].difference(objs[1])

    @staticmethod
    @geoms_to_tuple
    def geometric_union(objs: list):
        return shops.unary_union(objs)

    @staticmethod
    @geoms_to_tuple
    def geometric_intersection(objs: tuple):
        return shapely.intersection(objs[0],objs[1])

    @staticmethod
    def geoply_to_plymsg(polygon,ply_offset=0.06):
        if not type(polygon)==list:
            point_arr = []
            for point in polygon.exterior.coords:
                p = geomsgs.Point32()
                p.x = float(point[0])
                p.y = float(point[1])
                p.z = ply_offset
                point_arr.append(p)
        else:
            point_arr = []

        msg = geomsgs.PolygonStamped()
        msg.polygon.points = point_arr
        
        return msg

    @staticmethod
    def odommsg_to_list(msg):
        quat = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, 
            msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        r = Rotation.from_quat(quat)
        rpy = r.as_euler('xyz')
        res = [msg.pose.pose.position.x, msg.pose.pose.position.y, 
            msg.pose.pose.position.z, rpy[0], rpy[1], rpy[2]]  
        return res

    @staticmethod
    def min_dist_to_origin(obj):
        shl = shapely.shortest_line(geom.Point((0,0)),obj)
        return shapely.length(shl)

    @classmethod
    def sample_front(cls, rois, sample_radius, fov, pmap, focal_point, Psamp_prev, max_empty_reg, pthr: float) -> list:
        nrays = max(int(abs(fov[1] - fov[0])/np.pi * 100),10)
        phis = np.linspace(*fov,nrays)

        nrads = max(int(sample_radius/4.0 * 200),20)
        rads = np.linspace(0.0,sample_radius,nrads)
        Ra, Ph = np.meshgrid(rads,phis)
        X_rays, Y_rays = cls.pol2cart(Ra,Ph)
        X_rays = X_rays + focal_point

        if not rois:
            return [cls.pol2cart(sample_radius,phi) for phi in phis], Psamp_prev, 0*X_rays, []

        Psamp = pmap(X_rays,Y_rays) # limiting computation (as it should be..)
        Psamp_integr_parts = Psamp_prev+[Psamp]
        Psamp = cls.proba_grid_or(Psamp_integr_parts)
        Psamp_valid = Psamp >= pthr
        not_found_filt = np.all(Psamp_valid==False,axis=1)
        idxs_nearest = Psamp_valid.argmax(1) + (Psamp.shape[1]-1)*(not_found_filt.astype(int))
        idxs = np.vstack((np.arange(idxs_nearest.shape[0]),idxs_nearest))
        front_full = list(zip(X_rays[tuple(idxs)],Y_rays[tuple(idxs)]))
        
        # Find rays (intersecting roi but where no point was found) and throw out
        roi_limits = tuple(map(cls.find_angular_region, rois))
        cross_roi = [True if any([True if p>=rlim[0] and p<=rlim[1] else False for rlim in roi_limits]) else False for p in phis] #tuple([False for _ in range(X_rays.shape[0])])
        fil = np.logical_not(np.logical_and(cross_roi, not_found_filt))
        front = list(compress(front_full,fil))
        found_points = list(compress(front_full,np.logical_not(not_found_filt)))

        return front, Psamp_integr_parts, np.vstack([X_rays.ravel(), Y_rays.ravel(), Psamp.ravel()]), found_points

    @staticmethod
    def pa_or_pb(a,b):
        return a + b - a*b

    @classmethod
    def proba_grid_or(cls,proba_grids):
        res = proba_grids[0]
        for p in proba_grids[1:]:
            res = cls.pa_or_pb(res,p)
        return res

    @classmethod
    def find_angular_region(cls, ply: geom.Polygon):
        x,y = tuple(zip(*list(ply.exterior.coords)))
        x,y = np.array(x), np.array(y)
        _, phi = cls.cart2pol(x,y)
        return (np.min(phi), np.max(phi))

    @staticmethod
    def compute_error(err_type: str, gt_objects: dict, *, points=None, ref_area=None):
        if err_type=='nearest_object':
            assert(points != None)

            if not points:
                return []

            # map points to geom Point
            ray_points = list(map(geom.Point, points))
            # union gt_objects (beforehand)
            ugt = shapely.MultiPolygon(list(gt_objects.values()))
            outer_bounds = geom.MultiLineString([geom.LineString(obj.exterior.coords) 
                                                for obj in gt_objects.values()])
            # find nearest distance for each point
            nearest_points = [[o for o in shops.nearest_points(rp, outer_bounds)][-1] for rp in ray_points]
            nearest_dists = [rp.distance(nrstp) for rp,nrstp in zip(ray_points,nearest_points)]
            # check if inside or outside
            fil_contains = list(map(ugt.contains, ray_points))
            # change sign of distance value correspondigly
            min_errors = [-nrstd if cont else nrstd for nrstd,cont in zip(nearest_dists,fil_contains)]
            
            return min_errors
            
        elif err_type=='area_errors':
            assert(ref_area != None)

            area_errors = dict()
            for ido,obj in gt_objects.items():
                if shpred.intersects(ref_area,obj):
                    area_errors[ido] = shapely.area(shapely.intersection(ref_area,obj))
            return area_errors
        else:
            raise Exception("Error Type '"+err_type+"' does not exist.")

    @staticmethod
    def get_observable_region(max_empty_reg,focal_point):
        bounding_circ = geom.Polygon()

        max_r = 20.0
        for i in range(2000):
            # Step in 1mm and find max radius TODO: use function shapely?
            radius = (i+1)/(1e2)
            bounding_circ = geom.Point((focal_point,0)).buffer(radius)
            if shpred.contains(bounding_circ, max_empty_reg):
                max_r = radius
                break

        ray_ccw = geom.LineString([(focal_point,0),(-max_r,0.0)])
        ray_cw = geom.LineString([(focal_point,0),(-max_r,0.0)])
        min_phi, max_phi = 0.0, 0.0
        found_min, found_max = False, False
        ang_res = 1000
        for j in range(1,ang_res+1):
            if shpred.intersects(ray_ccw,max_empty_reg) and not found_min:
                min_phi = -np.pi + 2*(j-2)/ang_res*(np.pi)
                found_min = True
            if shpred.intersects(ray_cw,max_empty_reg) and not found_max:
                max_phi = np.pi - 2*(j-2)/ang_res*(np.pi)
                found_max = True
            ray_ccw = shaffin.rotate(ray_ccw, 2*(np.pi)/ang_res, origin=(focal_point,0), use_radians=True) #(j-1)/20*(np.pi)
            ray_cw = shaffin.rotate(ray_cw, -2*(np.pi)/ang_res, origin=(focal_point,0), use_radians=True)

        return (max_r, min_phi, max_phi)

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