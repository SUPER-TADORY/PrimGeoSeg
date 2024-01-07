import random
import numpy as np
import cv2

class Cylinder:
    R_SIZE_RANGE = (15,40)
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
    
    def _set_random_param(self):
        """
        Randomize parameters for each instance to allow for variation
        """
        self.R_max = random.randint(self.R_SIZE_RANGE[0], self.R_SIZE_RANGE[1])
        self.a = random.randint(15, self.R_max)
        self.b = random.randint(15, self.R_max)
        self.box_xysize = max(self.a, self.b) * 2
        self.D = random.randint(self.D_RANGE[0], self.D_RANGE[1])
        self.line_w = 2
        self.rotate2d = random.randint(self.ROTATE2D_RANGE[0], self.ROTATE2D_RANGE[1])

    def _calc_radius(self, z):
        """
        Calcurate radius of slice at z
        """
        a = self.a
        b = self.b

        return a,b
        
    def _get_section(self, z):
        """
        Returns an image (np.array) of a cross section in coordinate space
        """
        CP = [self.box_xysize//2, self.box_xysize//2]
        section = np.zeros([self.box_xysize, self.box_xysize])
        target = np.zeros([self.box_xysize, self.box_xysize])

        scaled_a, scaled_b = self._calc_radius(z)
        cv2.ellipse(section, CP, (scaled_a,scaled_b), self.rotate2d, 0, 360, color=(1, 1, 1), thickness=self.line_w)
        cv2.ellipse(target, CP, (scaled_a,scaled_b), self.rotate2d, 0, 360, color=(self.class_id, self.class_id, self.class_id), thickness=-1)

        return section, target
    
    def construct_structure(self):
        """
        Returns an assembled object and a corresponding mask
        """
        #boxの大きさを定義
        self.box_size = [self.box_xysize, self.box_xysize, self.D]
        box = np.zeros(self.box_size)
        target_box = np.zeros(self.box_size)

        for z in range(self.D):
            section, target = self._get_section(z)
            box[:,:,z] = section
            target_box[:,:,z] = target
        
        return box, target_box