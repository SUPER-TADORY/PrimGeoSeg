import math
import random
import copy
import numpy as np
import cv2


class Rhombus_Polygon:
    XY_W_LIMIT = (30,80)
    D_RANGE = (10,50)
    ROTATE2D_RANGE = (0,180)

    def __init__(self, N, class_id):
        """
        N:int Number of vertices of polygon (if ellipse = 0)
        class_id:int/list
        """
        self.N = N
        self.class_id = class_id
        self._set_random_param()

        self.base_vertexes = self._set_base_vertexes()
    
    def _set_random_param(self):
        """
        Randomize parameters for each instance to allow for variation
        """
        self.xy_size = random.randint(self.XY_W_LIMIT[0],self.XY_W_LIMIT[1])
        self.D = random.randint(self.D_RANGE[0],self.D_RANGE[1])
        self.line_w = 2
        self.rotate2d = random.randint(self.ROTATE2D_RANGE[0], self.ROTATE2D_RANGE[1])

        # Instance Extension in the Z-axis direction
        self.change_z = random.randint(3, self.D-3)
        self.ganma_z0  = random.uniform(0.3, 0.5)
        self.ganma_z1  = random.uniform(0.8, 1)
        self.ganma_z2  = random.uniform(0.3, 0.5)
        self._set_base_vertexes()
    
    def _set_base_vertexes(self):
        """
        Find the vertices of a polygon in the x,y plane at random. 
        Finding the vertices in polar coordinates, because a closed figure cannot be obtained if the vertices are found at random, depending on the order of the vertices.
        """
        self.center = (self.xy_size//2, self.xy_size//2)
        max_r = self.xy_size//2
        bin_l = np.linspace(0, 360, self.N+1)
        self.theta_l = [random.randint(int(bin_l[i]),int(bin_l[i+1])) for i in range(self.N)]
        self.r_l = []
        for _ in range(self.N):
            self.r_l.append(random.randint(15, max_r))
    
    def _get_tmp_section(self, ganma):
        tmp_r_l = copy.deepcopy(self.r_l)
        return [int(element * ganma) for element in tmp_r_l]
    
    def _get_z012_section(self,z):
        self.r_l_z0 = self._get_tmp_section(self.ganma_z0)
        self.r_l_z1 = self._get_tmp_section(self.ganma_z1)
        self.r_l_z2 = self._get_tmp_section(self.ganma_z2)

    def _get_z_vertexes(self,z):
        """
        Returns vertices of the slice at z
        """
        if z <= self.change_z:
            rate = self.ganma_z0 + (self.ganma_z1 - self.ganma_z0) * (z/self.change_z)
        else:
            rate = self.ganma_z1 + (self.ganma_z2 - self.ganma_z1) * ((z - self.change_z)/(self.D - self.change_z))
        
        r_l_z = self._get_tmp_section(rate)
        z_vertexes = np.array([(self.center[0]+int(r*math.cos(math.radians(theta))),self.center[1]+int(r*math.sin(math.radians(theta)))) for r,theta in zip(r_l_z,self.theta_l)])

        return z_vertexes

 
    def _get_section(self,z):
        """
        Returns an image (np.array) of a cross section in coordinate space
        """
        section = np.zeros([self.xy_size, self.xy_size])
        target = np.zeros([self.xy_size, self.xy_size])

        points = self._get_z_vertexes(z).reshape(1, -1, 2)
        
        cv2.polylines(section, points, isClosed=True, color=(1, 1, 1), thickness=self.line_w)
        cv2.fillPoly(target, points, color=(self.class_id, self.class_id, self.class_id))

        return section, target
    
    def construct_structure(self):
        """
        Returns an assembled object and a corresponding mask
        """
        self.box_size = [self.xy_size, self.xy_size, self.D]
        box = np.zeros(self.box_size)
        target_box = np.zeros(self.box_size)

        for z in range(self.D):
            section, target = self._get_section(z)
            box[:,:,z] = section
            target_box[:,:,z] = target
        
        return box, target_box
