import numpy
from scipy import sparse

# -------------------------------------------------------------------------

def simple_quadratic_function(x):
    """
    this function accepts a 2D vector as input.
    Its outputs are:
       value: h(x1, x2) = x1^2 + 3*x1*x2
       grad: A 2x1 vector that gives the partial derivatives
             of h with respect to x1 and x2
    Note that when we pass simple_quadratic_function(x) to
    compute_gradient_numerical_estimate, we're assuming
    that compute_gradient_numerical_estimate will use only
    the first returned value of this function.
    :param x:
    :return:
    """
    value = x[0] ** 2 + 3 * x[0] * x[1]

    grad = numpy.zeros(shape=2, dtype=numpy.float32)
    grad[0] = 2 * x[0] + 3 * x[1]
    grad[1] = 3 * x[0]

    return value, grad


# -------------------------------------------------------------------------

def compute_gradient_numerical_estimate(J, theta, epsilon=0.0001):
    """
    :param J: a loss (cost) function that computes the real-valued loss given parameters and data
    :param theta: array of parameters
    :param epsilon: amount to vary each parameter in order to estimate
                    the gradient by numerical difference
    :return: array of numerical gradient estimate
    """

    gradient = numpy.zeros(theta.shape)

    eps_vector = numpy.zeros(theta.shape)
    for i in range(0,theta.size):

        eps_vector[i] = epsilon
        cost1,grad1 = J(theta+eps_vector)
        cost2,grad2 = J(theta-eps_vector)
        gradient[i] = (cost1 - cost2)/(2*epsilon)
        eps_vector[i] = 0



    return gradient


# -------------------------------------------------------------------------

def test_compute_gradient_numerical_estimate():
    """
    Test of compute_gradient_numerical_estimate.
    This provides a test for your numerical gradient implementation
    in compute_gradient_numerical_estimate
    It analytically evaluates the gradient of a very simple function
    called simple_quadratic_function and compares the result with
    your numerical estimate. Your numerical gradient implementation
    is incorrect if your numerical solution deviates too much from
    the analytical solution.
    :return:
    """
    print("test_compute_gradient_numerical_estimate(): Start Test")
    print("    Testing that your implementation of ")
    print("        compute_gradient_numerical_estimate()")
    print("        is correct")
    x = numpy.array([4, 10], dtype=numpy.float64)
    (value, grad) = simple_quadratic_function(x)

    print("    Computing the numerical and actual gradient for 'simple_quadratic_function'")
    num_grad = compute_gradient_numerical_estimate(simple_quadratic_function, x)
    print("    The following two 2d arrays should be very similar:")
    print("        ", num_grad, grad)
    print("    (Left: numerical gradient estimate; Right: analytical gradient)")

    diff = numpy.linalg.norm(num_grad - grad)
    print("    Norm of the difference between numerical and analytical num_grad:")
    print("        ", diff)
    print("    (should be < 1.0e-09 ; I get about 1.7e-10)")
    print("test_compute_gradient_numerical_estimate(): DONE\n")


# -------------------------------------------------------------------------
