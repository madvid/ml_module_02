import numpy as np

def predict_(x, theta):
    """Computes the prediction vector y_hat from two non-empty numpy.array.
Args:
x: has to be an numpy.array, a vector of shapes m * n.
theta: has to be an numpy.array, a vector of shapes (n + 1) * 1.
Return:
y_hat as a numpy.array, a vector of shapes m * 1.
None if x or theta are empty numpy.array.
None if x or theta shapes are not appropriate.
None if x or theta is not of expected type.
Raises:
This function should not raise any Exception.
"""
    try:
        if not isinstance(x, np.ndarray):
            return None
    
        if x.ndim == 1:
            x = x.reshape(-1,1)
        if any([n == 0 for n in x.shape]):
            return None

        if theta.shape != (x.shape[1] + 1, 1):
            return None

        xp = np.hstack((np.ones((x.shape[0], 1)), x))

        ypred = np.dot(xp, theta)
        return ypred
    except:
        return None
    
if __name__ == "__main__":
    x = np.arange(1,13).reshape((4,-3))
    # Example 0:
    theta1 = np.array([[5],[ 0],[ 0],[ 0]])
    pred = predict_(x, theta1)
    # Ouput:
    # array([[5.],[ 5.],[ 5.],[ 5.]])
    # Do you understand why y_hat contains only 5’s here?
    expected_pred = np.array([[5.],[ 5.],[ 5.],[ 5.]])
    print("my prediction:".ljust(25), pred.reshape(1, -1))
    print("expected prediction:".ljust(25), expected_pred.reshape(1, -1))

    # Example 1:
    theta2 = np.array([[0],[ 1],[ 0],[ 0]])
    pred = predict_(x, theta2)
    # Output:
    # array([[ 1.],[ 4.],[ 7.],[ 10.]])
    # Do you understand why y_hat == x[:,0] here?
    expected_pred = np.array([[1.],[ 4.],[ 7.],[ 10.]])
    print("my prediction:".ljust(25), pred.reshape(1, -1))
    print("expected prediction:".ljust(25), expected_pred.reshape(1, -1))

    # Example 2:
    theta3 = np.array([[-1.5],[ 0.6],[ 2.3],[ 1.98]])
    pred = predict_(x, theta3)
    # Output:
    # array([[ 9.64],[ 24.28],[ 38.92],[ 53.56]])
    expected_pred = np.array([[ 9.64],[ 24.28],[ 38.92],[ 53.56]])
    print("my prediction:".ljust(25), pred.reshape(1, -1))
    print("expected prediction:".ljust(25), expected_pred.reshape(1, -1))

    # Example 3:
    theta4 = np.array([[-3],[ 1],[ 2],[ 3.5]])
    pred = predict_(x, theta4)
    # Output:
    # array([[12.5],[ 32. ],[ 51.5],[ 71. ]])
    expected_pred = np.array([[12.5],[ 32. ],[ 51.5],[ 71. ]])
    print("my prediction:".ljust(25), pred.reshape(1, -1))
    print("expected prediction:".ljust(25), expected_pred.reshape(1, -1))
