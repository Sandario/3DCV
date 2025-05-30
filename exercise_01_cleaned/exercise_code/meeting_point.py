
import numpy as np

def meeting_point_linear(pts_list):
    '''
    Inputs:
    - pts_list: List[numpy.ndarray], list of each persons points in the space
    Outputs:
    - numpy.ndarray, meeting point or vectors spanning the possible meeting points of shape (m, dim_intersection)
    '''
    A = pts_list[0] # person A's points of shape (m,num_pts_A)
    B = pts_list[1] # person B's points of shape (m,num_pts_B)

    ########################################################################
    # TODO:                                                                #
    # Implement the meeting point algorithm.                               #
    #                                                                      #
    # As an input, you receive                                             #
    # - for each person, you receive a list of landmarks in their subspace.#
    #   It is guaranteed that the landmarks span each person’s whole       #
    #   subspace.                                                          #
    #                                                                      #
    # As an output,                                                        #
    # - If such a point exist, output it.                                  #
    # - If there is more than one such point,                              # 
    #   output vectors spanning the space.                                 #
    ########################################################################

    m, n1 = A.shape
    _, n2 = B.shape

    M = np.hstack([A, -B])

    u, s, vh = np.linalg.svd(M)
    rtol = 1e-5
    tol = rtol * s[0] if s.size > 0 else rtol
    rank = (s > tol).sum()
    null_space = vh[rank:].conj().T

    if null_space.size == 0:
        return np.zeros((A.shape[0], 0))
    else:
        alpha = null_space[:n1, :]
        intersection = A @ alpha
        if np.allclose(intersection, 0):
            return np.zeros((A.shape[0], 1))
        norms = np.linalg.norm(intersection, axis = 0)
        norms[norms == 0] = 1
        intersection /= norms
        return intersection
    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################
