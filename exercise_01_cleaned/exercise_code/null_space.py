import numpy as np
def get_null_vector(D):
    '''
    Inputs:
    - D: numpy.ndarray, matrix of shape (m,n)
    Outputs:
    - null_vector: numpy.ndarray, matrix of shape (dim_kern,n)
    '''

    ########################################################################
    # TODO:                                                                #
    # Get the kernel of the matrix D.                                      #
    # the kernel should consider the numerical errors.                     #
    ########################################################################

    tol = 1e-10

    # Compute the SVD of D
    U, S, Vt = np.linalg.svd(D)

    # Find the smallest singular value and its corresponding vector
    smallest_singular_value_index = np.argmin(S)
    v = Vt[-1, :]  # Corresponding right singular vector

    # Normalize the vector to ensure ||v|| = 1
    null_vector = v / np.linalg.norm(v)

    # Verify that it is in the numerical kernel
    residual = np.linalg.norm(D @ v)
    if residual > tol:
        print(f"Warning: Residual too high ({residual:.2e}). The vector might not be a true kernel element.")
    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################
    return null_vector 
