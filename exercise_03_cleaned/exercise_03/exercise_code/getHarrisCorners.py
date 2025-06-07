import numpy as np

def getHarrisCorners(M, kappa, theta):
    # Compute Harris corners
    # Input:
    # M: structure tensor of shape (H, W, 2, 2)
    # kappa: float (parameter for Harris corner score) 
    # theta: float (threshold for corner detection)
    # Output:
    # score: numpy.ndarray (Harris corner score) of shape (H, W)
    # points: numpy.ndarray (detected corners) of shape (N, 2)

    ########################################################################
    # TODO:                                                                #
    # Compute the Harris corner score and find the corners.               #
    #                                                                      #
    # Hints:                                                               #
    # - The Harris corner score is computed using the determinant and      #
    #   trace of the structure tensor.                                     #
    # - Use the threshold theta to find the corners.                       #
    # - Use non-maximum suppression to find the corners.                   #
    ########################################################################


    pass

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return score, points

