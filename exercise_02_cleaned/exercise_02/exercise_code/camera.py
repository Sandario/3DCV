import numpy as np
from math import tan, atan
from abc import ABC, abstractmethod

def compute_relative_pose(pose_1,pose_2):
    '''
    Inputs:
    - pose_i transform from cam_i to world coordinates, matrix of shape (3,4)
    Outputs:
    - pose transform from cam_1 to cam_2 coordinates, matrix of shape (3,4)
    '''

    ########################################################################
    # TODO:                                                                #
    # Compute the relative pose, which transform from cam_1 to cam_2       #
    # coordinates.                                                         #
    ########################################################################

    pose_1_homo = np.vstack((pose_1, [0, 0, 0, 1]))
    pose_2_homo = np.vstack((pose_2, [0, 0, 0, 1]))

    pose_2_homo_inv = np.linalg.inv(pose_2_homo)

    pose = pose_2_homo_inv @ pose_1_homo

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return pose[:3, :]



class Camera(ABC):
    def __init__(self, w, h):
        self.w = w
        self.h = h

    @abstractmethod
    def project(self, pt):
        """Project the point pt onto a pixel on the screen"""
        
    @abstractmethod
    def unproject(self, pix, d):
        """Unproject the pixel pix into the 3D camera space for the given distance d"""


class Pinhole(Camera):

    def __init__(self, w, h, fx, fy, cx, cy):
        Camera.__init__(self, w, h)
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    def project(self, pt):
        '''
        Inputs:
        - pt, vector of size 3
        Outputs:
        - pix, projection of pt using the pinhole model, vector of size 2
        '''
        ########################################################################
        # TODO:                                                                #
        # project the point pt, considering the pinhole model.                 #
        ########################################################################

        x, y, z = pt

        if z == 0:
            raise ValueError("Z coordinate is zero, cannot project to image plane.")

        x = x / z
        y = y / z

        p_img_hom = self.K @ np.array([x, y, 1])
        u, v = p_img_hom[:2]

        pix = np.array([u, v])

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return pix

    def unproject(self, pix, d):
        '''
        Inputs:
        - pix, vector of size 2
        - d, scalar 
        Outputs:
        - final_pt, obtained by unprojecting pix with the distance d using the pinhole model, vector of size 3
        '''
        ########################################################################
        # TODO:                                                                #
        # Unproject the pixel pix using the distance d, considering the pinhole#
        # model. Be careful: d is the distance between the camera origin and   #
        # the desired point, not the depth.                                    #
        ########################################################################

        u, v = pix
        K_inv = np.linalg.inv(self.K)
        direction = np.array([u, v, 1.0])
        direction = K_inv @ direction
        direction /= np.linalg.norm(direction)

        final_pt = d * direction

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return final_pt


class Fov(Camera):

    def __init__(self, w, h, fx, fy, cx, cy, W):
        Camera.__init__(self, w, h)
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        self.W = W

    def project(self, pt):
        '''
        Inputs:
        - pt, vector of size 3
        Outputs:
        - pix, projection of pt using the Fov model, vector of size 2
        '''
        ########################################################################
        # TODO:                                                                #
        # project the point pt, considering the Fov model.                     #
        ########################################################################

        x, y, z = pt

        if z == 0:
            raise ValueError("Z coordinate is zero, cannot project to image plane.")

        x_u = x / z
        y_u = y / z

        r_u = np.sqrt(x_u ** 2 + y_u ** 2)

        if r_u > 1e-8:
            # Distorted radius
            tan_half_w = np.tan(self.W / 2)
            r_d = (1.0 / self.W) * np.arctan(2 * r_u * tan_half_w)

            # Scale to distorted normalized coordinates
            scale = r_d / r_u
            x_d = x_u * scale
            y_d = y_u * scale
        else:
            # No distortion at center
            x_d, y_d = x_u, y_u

        p_img_hom = self.K @ np.array([x_d, y_d, 1.0])
        u, v = p_img_hom[:2]
        #u, v = (1/(self.W*p_img_hom[:2]))*np.arctan(2*p_img_hom[:2]*np.tan(self.W/2))

        pix = np.array([u, v])

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return pix

    def unproject(self, pix, d):
        '''
        Inputs:
        - pix, vector of size 2
        - d, scalar 
        Outputs:
        - final_pt, obtained by unprojecting pix with the distance d using the Fov model, vector of size 3
        '''
        ########################################################################
        # TODO:                                                                #
        # Unproject the pixel pix using the distance d, considering the FOV    #
        # model. Be careful: d is the distance between the camera origin and   #
        # the desired point, not the depth.                                    #
        ########################################################################

        u, v = pix
        K_inv = np.linalg.inv(self.K)
        direction = np.array([u, v, 1.0])
        direction = K_inv @ direction
        x_d, y_d = direction[0], direction[1]

        r_d = np.sqrt(x_d ** 2 + y_d ** 2)

        if r_d > 1e-8:
            # Compute undistorted radius r_u
            tan_half_w = np.tan(self.W / 2)
            r_u = np.tan(r_d * self.W) / (2 * tan_half_w)

            # Scale distorted normalized coordinates to undistorted
            scale = r_u / r_d
            x_u = x_d * scale
            y_u = y_d * scale
        else:
            # No distortion at center
            x_u, y_u = x_d, y_d

        # Create direction vector
        direction = np.array([x_u, y_u, 1.0])

        # Normalize direction
        direction_normalized = direction / np.linalg.norm(direction)

        # Scale by distance
        final_pt = d * direction_normalized

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################        
        return final_pt
