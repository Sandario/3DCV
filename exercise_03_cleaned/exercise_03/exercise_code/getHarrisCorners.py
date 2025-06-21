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

    # Extract structure tensor components
    m11 = M[..., 0, 0]
    m12 = M[..., 0, 1]
    m22 = M[..., 1, 1]

    # Compute determinant and trace of M
    det_M = m11 * m22 - m12 ** 2
    trace_M = m11 + m22

    # Harris response score
    score = det_M - kappa * (trace_M ** 2)

    # Detect local maxima above threshold
    H, W = score.shape
    coords = []
    for y in range(1, H - 1):
        for x in range(1, W - 1):
            c = score[y, x]
            if c > theta and \
                    c > score[y - 1, x] and \
                    c > score[y + 1, x] and \
                    c > score[y, x - 1] and \
                    c > score[y, x + 1]:
                coords.append((y, x))

    points = np.array(coords, dtype=np.int32)
    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return score, points

