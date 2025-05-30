import numpy as np


def swap_rows(A, i, j):
    '''
    Inputs:
    - A: numpy.ndarray, matrix
    - i: int, index of the first row
    - j: int, index of the second row

    Outputs:
    - numpy.ndarray, matrix with swapped rows
    '''
    A[[i, j]] = A[[j, i]]
    return A

def multiply_row(A, i, scalar):
    '''
    Inputs:
    - A: numpy.ndarray, matrix
    - i: int, index of the row
    - scalar: float, scalar to multiply the row with

    Outputs:
    - numpy.ndarray, matrix with multiplied row
    '''
    A[i] = A[i] * scalar
    return A

def add_row(A, i, j, scalar=1):
    '''
    Inputs:
    - A: numpy.ndarray, matrix
    - i: int, index of the row to be added to
    - j: int, index of the row to be added

    Outputs:
    - numpy.ndarray, matrix with added rows
    '''
    A[i] = A[i] + A[j]*scalar
    return A

def perform_gaussian_elemination(A):
    '''
    Inputs:
    - A: numpy.ndarray, matrix of shape (dim, dim)

    Outputs:
    - ops: List[Tuple[str,int,int]], sequence of elementary operations
    - A_inv: numpy.ndarray, inverse of A
    '''
    dim = A.shape[0]
    A_inv = np.eye(dim)
    #A_inv = np.linalg.inv(A)
    ops = []
    ########################################################################
    # TODO:                                                                #
    # Implement the Gaussian elemination algorithm.                        #
    # Return the sequence of elementary operations and the inverse matrix. #
    #                                                                      #
    # The sequence of the operations should be in the following format:    #
    # • to swap to rows                                                    #
    #   ("S",<row index>,<row index>)                                      #
    # • to multiply the row with a number                                  #
    #   ("M",<row index>,<number>)                                         #
    # • to add multiple of one row to another row                          #
    #   ("A",<row index i>,<row index j>, <number>)                        #https://gitlab.com/warningnonpotablewater/libinput-config
    # Be aware that the rows are indexed starting with zero.               #
    # Output sufficient number of significant digits for numbers.          #
    # Output integers for indices.                                         #
    #                                                                      #
    # Append to the sequence of operations                                 #
    # • "DEGENERATE" if you have successfully turned the matrix into a     #
    #   form with a zero row.                                              #
    # • "SOLUTION" if you turned the matrix into the $[I|A −1 ]$ form.     #
    #                                                                      #
    # If you found the inverse, output it as a second element,             #
    # otherwise return None as a second element                            #
    ########################################################################

    # bring in row echolon form
    for i in range(dim):
        if A[i, i] != 1:
            ops.append(("M", i, 1 / A[i, i]))
            A_inv = multiply_row(A_inv, i, 1/A[i, i])
            A = multiply_row(A, i, 1/A[i, i])

        for j in range(i+1, dim):
            ops.append(("A", j, i, -A[j, i]))
            A_inv = add_row(A_inv, j, i, -A[j, i])
            A = add_row(A, j, i, -A[j, i])


    # check for degenerate
    zero_row_indices = np.where(~A.any(axis=1))[0]
    zero_row_indices.tolist()
    if zero_row_indices:
        amount = 0
        for i in zero_row_indices:
            ops.append(("S", i, dim - amount))
            A_inv = swap_rows(A_inv, i, dim-amount)
            A = swap_rows(A, i, dim-amount)

            amount += 1
        ops.append("DEGENERATE")
        return ops, A

    # bring to Identity
    for i in reversed(range(dim)):
        for j in reversed(range(i)):
            ops.append(("A", j, i, -A[j, i]))
            A_inv = add_row(A_inv, j, i, -A[j, i])
            A = add_row(A, j, i, -A[j, i])


    ops.append("SOLUTION")
    print(A)

    return ops, A_inv

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################
