import math
import random
import numpy as np
import cv2


class Prism:
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

        self.end_xy = self._get_endpoint()
        self.base_vertexes = self._get_base_vertexes()
    
    def _set_random_param(self):
        """
        Randomize parameters for each instance to allow for variation
        """
        self.xy_size = random.randint(self.XY_W_LIMIT[0],self.XY_W_LIMIT[1])
        self.D = random.randint(self.D_RANGE[0],self.D_RANGE[1])
        self.line_w = 2
        self.rotate2d = random.randint(self.ROTATE2D_RANGE[0], self.ROTATE2D_RANGE[1])

    def _get_endpoint(self):
        """
        Randomly find where the vertex of the cone falls on the xy-plane
        """
        end_x = self.xy_size // 2
        end_y = self.xy_size // 2 
        return (end_x,end_y)
    
    def _get_base_vertexes(self):
        """
        Find the vertices of a polygon in the x,y plane at random. 
        Finding the vertices in polar coordinates, because a closed figure cannot be obtained if the vertices are found at random, depending on the order of the vertices.
        """
        center = (self.xy_size//2, self.xy_size//2)
        max_r = self.xy_size//2
        bin_l = np.linspace(0, 360, self.N+1)
        theta_l = [random.randint(int(bin_l[i]),int(bin_l[i+1])) for i in range(self.N)]
        r_l = []
        for _ in range(self.N):
            r_l.append(random.randint(15, max_r))
        vertexes = np.array([(center[0]+int(r*math.cos(math.radians(theta))),center[1]+int(r*math.sin(math.radians(theta)))) for r,theta in zip(r_l,theta_l)])
        
        return vertexes
    
    def _get_section(self,z):
        """
        Returns an image (np.array) of a cross section in coordinate space
        """
        section = np.zeros([self.xy_size, self.xy_size])
        target = np.zeros([self.xy_size, self.xy_size])

        points = self.base_vertexes.reshape(1, -1, 2)
        
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
