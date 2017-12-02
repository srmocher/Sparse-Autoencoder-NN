# Utilities for the NN exercises in ISTA 421, Introduction to ML

import numpy
from scipy import stats
import math
import visualize
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------

def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))

def kl_divergence(rho,rho_hat):
    return numpy.sum(rho*numpy.log(rho/rho_hat) + (1-rho)*numpy.log((1-rho)/(1-rho_hat)))

# -------------------------------------------------------------------------

def initialize(hidden_size, visible_size):
    """
    Sample weights uniformly from the interval [-r, r] as described in lecture 23.
    Return 1d array theta (in format as described in Exercise 2)
    :param hidden_size: number of hidden units
    :param visible_size: number of visible units (of input and output layers of autoencoder)
    :return: theta array
    """

    ### YOUR CODE HERE ###


    b1 = numpy.zeros(hidden_size,dtype=numpy.float64)
    b2 = numpy.zeros(visible_size,dtype=numpy.float64)

    r = numpy.sqrt(6) / numpy.sqrt(2*visible_size + 1)

    W1 = numpy.random.random((hidden_size, visible_size)) * 2 * r - r
    W2 = numpy.random.random((visible_size, hidden_size)) * 2 * r - r
    weights = (W1.reshape(hidden_size*visible_size),W2.reshape(visible_size*hidden_size),b1.reshape(hidden_size),b2.reshape(visible_size))
    theta = numpy.concatenate(weights)
    return theta




# -------------------------------------------------------------------------

def autoencoder_cost_and_grad(theta, visible_size, hidden_size, lambda_, data):
    """
    The input theta is a 1-dimensional array because scipy.optimize.minimize expects
    the parameters being optimized to be a 1d array.
    First convert theta from a 1d array to the (W1, W2, b1, b2)
    matrix/vector format, so that this follows the notation convention of the
    lecture notes and tutorial.
    You must compute the:
        cost : scalar representing the overall cost J(theta)
        grad : array representing the corresponding gradient of each element of theta
    """

    training_size = data.shape[1]
   # unroll theta to get (W1,W2,b1,b2) #
    W1 = theta[0:hidden_size*visible_size]
    W1 = W1.reshape(hidden_size,visible_size)

    W2 = theta[hidden_size*visible_size:2*hidden_size*visible_size]
    W2 = W2.reshape(visible_size,hidden_size)

    b1 = theta[2*hidden_size*visible_size:2*hidden_size*visible_size + hidden_size].reshape(hidden_size,1)
    b2 = theta[2*hidden_size*visible_size + hidden_size: 2*hidden_size*visible_size + hidden_size + visible_size].reshape(visible_size,1)

    # feedforward pass
    a_l1 = data

    z_l2 = W1.dot(a_l1) + b1
    a_l2 = sigmoid(z_l2)

    z_l3 = W2.dot(a_l2) + b2
    a_l3 = sigmoid(z_l3)

    # backprop
    diff = a_l3-data
    delta_l3 = numpy.multiply(diff,a_l3*(1-a_l3))
    delta_l2 = numpy.multiply(W2.T.dot(delta_l3),a_l2*(1 - a_l2))

    b2_derivative = numpy.sum(delta_l3,axis=1)/training_size
    b1_derivative = numpy.sum(delta_l2,axis=1)/training_size

    W2_derivative = numpy.dot(delta_l3,a_l2.T)/training_size + lambda_*W2
    #print(W2_derivative.shape)
    W1_derivative = numpy.dot(delta_l2,a_l1.T)/training_size + lambda_*W1

    W1_derivative = W1_derivative.reshape(hidden_size*visible_size)
    W2_derivative = W2_derivative.reshape(visible_size*hidden_size)
    b1_derivative = b1_derivative.reshape(hidden_size)
    b2_derivative = b2_derivative.reshape(visible_size)


    grad = numpy.concatenate((W1_derivative,W2_derivative,b1_derivative,b2_derivative))
    cost = 0.5 * numpy.sum(numpy.multiply(diff,diff)) / training_size + 0.5*lambda_*(numpy.sum(numpy.square(W1)) + numpy.sum(numpy.square(W2)))
    return cost,grad


def autoencoder_cost_and_grad_sparse(theta, visible_size, hidden_size, lambda_, rho_, beta_, data):
    """
    Version of cost and grad that incorporates the hidden layer sparsity constraint
        rho_ : the target sparsity limit for each hidden node activation
        beta_ : controls the weight of the sparsity penalty term relative
                to other loss components

    The input theta is a 1-dimensional array because scipy.optimize.minimize expects
    the parameters being optimized to be a 1d array.
    First convert theta from a 1d array to the (W1, W2, b1, b2)
    matrix/vector format, so that this follows the notation convention of the
    lecture notes and tutorial.
    You must compute the:
        cost : scalar representing the overall cost J(theta)
        grad : array representing the corresponding gradient of each element of theta
    """

    ### YOUR CODE HERE ###
    training_size = data.shape[1]
    # unroll theta to get (W1,W2,b1,b2) #
    W1 = theta[0:hidden_size * visible_size]
    W1 = W1.reshape(hidden_size, visible_size)

    W2 = theta[hidden_size * visible_size:2 * hidden_size * visible_size]
    W2 = W2.reshape(visible_size, hidden_size)

    b1 = theta[2 * hidden_size * visible_size:2 * hidden_size * visible_size + hidden_size].reshape(hidden_size, 1)
    b2 = theta[
         2 * hidden_size * visible_size + hidden_size: 2 * hidden_size * visible_size + hidden_size + visible_size].reshape(
        visible_size, 1)

    # feedforward pass
    a_l1 = data

    z_l2 = W1.dot(a_l1) + numpy.tile(b1, (1, training_size))
    a_l2 = sigmoid(z_l2)

    z_l3 = W2.dot(a_l2) + numpy.tile(b2, (1, training_size))
    a_l3 = sigmoid(z_l3)

    # backprop


    delta_l3 = -(data - a_l3) * a_l3 * (1 - a_l3)

    # enforce sparsity on hidden layer
    rho_hat = numpy.sum(a_l2, axis=1) / training_size
    a_l2_grad = a_l2 * (1 - a_l2)


    sparsity_delta = numpy.tile(- rho_ / rho_hat + (1 - rho_) / (1 - rho_hat), (training_size, 1)).T
    delta_l2 = numpy.multiply(W2.T.dot(delta_l3) + beta_*sparsity_delta, a_l2_grad)

    b2_derivative = numpy.sum(delta_l3, axis=1) / training_size
    b1_derivative = numpy.sum(delta_l2, axis=1) / training_size

    W2_derivative = numpy.dot(delta_l3, a_l2.T) / training_size
    # print(W2_derivative.shape)
    W1_derivative = numpy.dot(delta_l2, a_l1.T) / training_size

    W1_derivative = W1_derivative.reshape(hidden_size * visible_size)
    W2_derivative = W2_derivative.reshape(visible_size * hidden_size)
    b1_derivative = b1_derivative.reshape(hidden_size)
    b2_derivative = b2_derivative.reshape(visible_size)




    kl_div = kl_divergence(rho_,rho_hat)
    grad = numpy.concatenate((W1_derivative, W2_derivative, b1_derivative, b2_derivative))
    cost = 0.5 * numpy.sum((data - a_l3) ** 2) / training_size + 0.5 * lambda_ * (numpy.sum(numpy.square(W1)) + numpy.sum(numpy.square(W2))) + beta_*kl_div
    return cost, grad



# -------------------------------------------------------------------------

def autoencoder_feedforward(theta, visible_size, hidden_size, data):
    """
    Given a definition of an autoencoder (including the size of the hidden
    and visible layers and the theta parameters) and an input data matrix
    (each column is an image patch, with 1 or more columns), compute
    the feedforward activation for the output visible layer for each
    data column, and return an output activation matrix (same format
    as the data matrix: each column is an output activation "image"
    corresponding to the data input).

    Once you have implemented the autoencoder_cost_and_grad() function,
    simply copy the part of the code that computes the feedforward activations
    up to the output visible layer activations and return that activation.
    You do not need to include any of the computation of cost or gradient.
    It is likely that your implementation of feedforward in your
    autoencoder_cost_and_grad() is set up to handle multiple data inputs,
    in which case your only task is to ensure the output_activations matrix
    is in the same corresponding format as the input data matrix, where
    each output column is the activation corresponding to the input column
    of the same column index.

    :param theta: the parameters of the autoencoder, assumed to be in this format:
                  { W1, W2, b1, b2 }
                  W1 = weights of layer 1 (input to hidden)
                  W2 = weights of layer 2 (hidden to output)
                  b1 = layer 1 bias weights (to hidden layer)
                  b2 = layer 2 bias weights (to output layer)
    :param visible_size: number of nodes in the visible layer(s) (input and output)
    :param hidden_size: number of nodes in the hidden layer
    :param data: input data matrix, where each column is an image patch,
                  with one or more columns
    :return: output_activations: an matrix output, where each column is the
                  vector of activations corresponding to the input data columns
    """

    ### YOUR CODE HERE ###

    training_size = data.shape[1]
    W1 = theta[0:hidden_size*visible_size].reshape(hidden_size,visible_size)
    W2 = theta[hidden_size*visible_size:2*hidden_size*visible_size].reshape(visible_size,hidden_size)
    b1 = theta[2*hidden_size*visible_size:2*hidden_size*visible_size+hidden_size].reshape(hidden_size,1)
    b2 = theta[2*hidden_size*visible_size+hidden_size:2*hidden_size*visible_size+hidden_size+visible_size].reshape(visible_size,1)

    a_l1 = data
    z_l2 = W1.dot(a_l1) + numpy.tile(b1,(1,training_size))

    a_l2 = sigmoid(z_l2)
    z_l3 = W2.dot(a_l2) + numpy.tile(b2,(1,training_size))

    output_activations = sigmoid(z_l3)  # implement

    return output_activations


# -------------------------------------------------------------------------

def save_model(theta, visible_size, hidden_size, filepath, **params):
    """
    Save the model to file.  Used by plot_and_save_results.
    :param theta: the parameters of the autoencoder, assumed to be in this format:
                  { W1, W2, b1, b2 }
                  W1 = weights of layer 1 (input to hidden)
                  W2 = weights of layer 2 (hidden to output)
                  b1 = layer 1 bias weights (to hidden layer)
                  b2 = layer 2 bias weights (to output layer)
    :param visible_size: number of nodes in the visible layer(s) (input and output)
    :param hidden_size: number of nodes in the hidden layer
    :param filepath: path with filename
    :param params: optional parameters that will be saved with the model as a dictionary
    :return:
    """
    numpy.savetxt(filepath + '_theta.csv', theta, delimiter=',')
    with open(filepath + '_params.txt', 'a') as fout:
        params['visible_size'] = visible_size
        params['hidden_size'] = hidden_size
        fout.write('{0}'.format(params))


# -------------------------------------------------------------------------

def plot_and_save_results(theta, visible_size, hidden_size, root_filepath=None,
                          train_patches=None, test_patches=None, show_p=False,
                          **params):
    """
    This is a helper function to streamline saving the results of an autoencoder.
    The visible_size and hidden_size provide the information needed to retrieve
    the autoencoder parameters (w1, w2, b1, b2) from theta.

    This function does the following:
    (1) Saves the parameters theta, visible_size and hidden_size as a text file
        called '<root_filepath>_model.txt'
    (2) Extracts the layer 1 (input-to-hidden) weights and plots them as an image
        and saves the image to file '<root_filepath>_weights.png'
    (3) [optional] train_patches are intended to be a set of patches that were
        used during training of the autoencoder.  Typically these will be the first
        100 patches of the MNIST data set.
        If provided, the patches will be given as input to the autoencoder in
        order to generate output 'decoded' activations that are then plotted as
        patches in an image.  The image is saved to '<root_filepath>_train_decode.png'
    (4) [optional] test_patches are intended to be a different set of patches
        that were *not* used during training.  This permits inspecting how the
        autoencoder does decoding images it was not trained on.  The output activation
        image is generated the same way as in step (3).  The image is saved to
        '<root_filepath>_test_decode.png'

    The root_filepath is used as the base filepath name for all files generated
    by this function.  For example, if you wish to save all of the results
    using the root_filepath='results1', and you have specified the train_patches and
    test_patches, then the following files will be generated:
        results1_model.txt
        results1_weights.png
        results1_train_decode.png
        results1_test_decode.png
    If no root_filepath is provided, then the output will default to:
        model.txt
        weights.png
        train_decode.png
        test_decode.png
    Note that if those files already existed, they will be overwritten.

    :param theta: model parameters
    :param visible_size: number of nodes in autoencoder visible layer
    :param hidden_size: number of nodes in autoencoder hidden layer
    :param root_filepath: base filepath name for files generated by this function
    :param train_patches: matrix of patches (typically the first 100 patches of MNIST)
    :param test_patches: matrix of patches (intended to be patches NOT used in training)
    :param show_p: flag specifying whether to show the plots before exiting
    :param params: optional parameters that will be saved with the model as a dictionary
    :return:
    """

    filepath = 'model'
    if root_filepath:
        filepath = root_filepath + '_' + filepath
    save_model(theta, visible_size, hidden_size, filepath, **params)

    # extract the input to hidden layer weights and visualize all the weights
    # corresponding to each hidden node
    w1 = theta[0:hidden_size * visible_size].reshape(hidden_size, visible_size).T
    filepath = 'weights.png'
    if root_filepath:
        filepath = root_filepath + '_' + filepath
    visualize.plot_images(w1, show_p=False, filepath=filepath)

    if train_patches is not None:
        # Given: train_patches and autoencoder parameters,
        # compute the output activations for each input, and plot the resulting decoded
        # output patches in a grid.
        # You can then manually compare them (visually) to the original input train_patches
        filepath = 'train_decode.png'
        if root_filepath:
            filepath = root_filepath + '_' + filepath
        train_decode = autoencoder_feedforward(theta, visible_size, hidden_size, train_patches)
        visualize.plot_images(train_decode, show_p=False, filepath=filepath)

    if test_patches is not None:
        # Same as for train_patches, but assuming test_patches are patches that were not
        # used for training the autoencoder.
        # Again, you can then manually compare the decoded patches to the test_patches
        # given as input.
        test_decode = autoencoder_feedforward(theta, visible_size, hidden_size, test_patches)
        filepath = 'test_decode.png'
        if root_filepath:
            filepath = root_filepath + '_' + filepath
        visualize.plot_images(test_decode, show_p=False, filepath=filepath)

    if show_p:
        plt.show()


# -------------------------------------------------------------------------

def get_pretty_time_string(t, delta=False):
    """
    Really cheesy kludge for producing semi-human-readable string representation of time
    y = Year, m = Month, d = Day, h = Hour, m (2nd) = minute, s = second, mu = microsecond
    :param t: datetime object
    :param delta: flag indicating whether t is a timedelta object
    :return:
    """
    if delta:
        days = t.days
        hours = t.seconds // 3600
        minutes = (t.seconds // 60) % 60
        seconds = t.seconds - (minutes * 60)
        return 'days={days:02d}, h={hour:02d}, m={minute:02d}, s={second:02d}' \
                .format(days=days, hour=hours, minute=minutes, second=seconds)
    else:
        return 'y={year:04d},m={month:02d},d={day:02d},h={hour:02d},m={minute:02d},s={second:02d},mu={micro:06d}' \
                .format(year=t.year, month=t.month, day=t.day,
                        hour=t.hour, minute=t.minute, second=t.second,
                        micro=t.microsecond)


# -------------------------------------------------------------------------
