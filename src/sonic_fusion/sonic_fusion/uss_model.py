import numpy as np
import shapely.geometry as geom

class USSensorModel():

    def __init__(self, cfg: dict, *, empty_thr=0.1) -> None:
        self.id = cfg['id']
        self.origin = cfg['origin']
        self.rpy = cfg['rpy']
        self.fov = cfg['fov']
        self.rng = (cfg['min_rng'], cfg['max_rng'])
        self.empty_thr = empty_thr

    def _construct_arc(self,radius):
        start_ang = self.fov/2
        end_ang = -self.fov/2
        resolution = int(1000*self.fov/180) # TODO: Make proportional to distance?
        
        # Transform coordinates from sensor to body frame
        theta = np.radians(np.linspace(start_ang, end_ang, resolution))
        x = self.origin[0] + radius * np.cos(theta+self.rpy[2])
        y = self.origin[1] + radius * np.sin(theta+self.rpy[2])
        arc = np.column_stack([x,y])
        return arc

    def _construct_seg(self,radius):
        cx = self.origin[0]
        cy = self.origin[1]

        # The coordinates of the arc
        arc = self._construct_arc(radius)
        seg = np.vstack(([cx,cy],arc,[cx,cy]))
        return seg

    def get_empty_seg(self, rng):
        radius = rng - self.empty_thr
        seg = self._construct_seg(radius)
        return geom.Polygon(seg)

    def get_arc(self, rng):
        arc = self._construct_arc(rng)
        return geom.LineString(arc)