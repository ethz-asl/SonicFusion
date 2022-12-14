import numpy as np
import shapely.geometry as geom
from sonic_fusion.utils import Utils

class USSensorModel():

    def __init__(self, cfg: dict) -> None:
        self.id = cfg['id']
        self.origin = cfg['origin']
        self.rpy = cfg['rpy']
        self.fov = cfg['fov']
        self.rng = (cfg['min_rng'], cfg['max_rng'])
        self.sigma_r = cfg['sigma_r']
        self.empty_thr = 3*cfg['sigma_r'] # empty at 3 times std

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

    def get_empty_seg_body(self, rng):
        radius = rng - self.empty_thr
        seg = self._construct_seg(radius)
        return geom.Polygon(seg)

    def get_arc_body(self, rng):
        arc = self._construct_arc(rng)
        return geom.LineString(arc)

    def get_gauss_body(self, narc, rng):
        cox, coy = tuple(self.origin[:-1])
        phi_origin = self.rpy[2]
        
        # Get nodes from current arc
        xy = np.vstack(narc.xy)
        xy_nds = np.vstack((xy[:,0],xy[:,-1])).T #col1 and col2 are nodes
        
        # Transform nodes from body to sensor frame
        x_nds_s = np.cos(phi_origin)*(xy_nds[0,:] - cox) + np.sin(phi_origin)*(xy_nds[1,:] - coy)
        y_nds_s = -np.sin(phi_origin)*(xy_nds[0,:] - cox) + np.cos(phi_origin)*(xy_nds[1,:] - coy)
        
        _, phi_nds = Utils.cart2pol(x_nds_s,y_nds_s)
        
        # sigmas
        dphi = abs(phi_nds[0] - phi_nds[1])
        phi_center = (phi_nds[0] + phi_nds[1])/2
        
        sig_r = self.sigma_r
        sig_phi = dphi/4
        
        def gaussian_on_arc(X,Y):
            # body to sensor frame
            X_s = np.cos(phi_origin+phi_center)*(X - cox) + np.sin(phi_origin+phi_center)*(Y - coy)
            Y_s = -np.sin(phi_origin+phi_center)*(X - cox) + np.cos(phi_origin+phi_center)*(Y - coy)
            
            R, Phi = Utils.cart2pol(X_s,Y_s)

            # compute probab
            pr = np.exp(-((R-rng)**2)/(2*sig_r**2))
            pth = np.exp(-(Phi**2)/(2*sig_phi**2))
            return pr*pth
        
        return gaussian_on_arc