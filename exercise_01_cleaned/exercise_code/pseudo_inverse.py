import numpy as np

def solve_linear_equation_SVD(D, b):
    '''
    Inputs:
    - D: numpy.ndarray, matrix of shape (m,n)
    - b: numpy.ndarray, vector of shape (m,)
    Outputs:
    - x: numpy.ndarray, solution of the linear equation D*x = b
    - D_inv: numpy.ndarray, pseudo-inverse of D of shape (n,m)
    '''

    ########################################################################
    # TODO:                                                                #
    # Solve the linear equation D*x = b using the pseudo-inverse and SVD.  #
    # Your code should be able to tackle the case where D is singular.     # 
    ########################################################################

    # Compute the SVD of D
    U, S, Vt = np.linalg.svd(D, full_matrices=False)

    # Compute the pseudo-inverse using SVD components
    S_inv = np.diag([1 / s if s > 1e-10 else 0 for s in S])
    D_inv = Vt.T @ S_inv @ U.T

    # Compute the estimated x_hat
    x = D_inv @ b

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return x, D_inv

