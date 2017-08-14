import numpy as np
import pandas as pd
import math
import datetime
from data_engineering import replace_design_latent

# EXAMPLE TESTING GRADIENT DESCENT FOR MLE OF MU=E(a) IN NORMAL LIKELIHOOD
def mean_example():
    a = np.array([1,2,3,4,5,3,4,2,4,6,8,2,3])
    print(np.mean(a))

    mu = 0.1
    sigma = 10
    n = len(a)
    gamma = 0.01
    iterations = 0
    last_mu = 10000
    change = 0.1
    start = datetime.datetime.now()
    while change > 0.0001:
        mu = mu + gamma * (1/sigma) * (np.sum(a) - n * mu)

        change = abs(mu - last_mu)
        last_mu = mu
        iterations += 1
    print("Iterations taken for convergence (gradient descent): %d" % iterations)
    print("Time taken: %.5f" % (datetime.datetime.now() - start).total_seconds())

def mean_example_newton():
    a = np.array([1,2,3,4,5,3,4,2,4,6,8,2,3])
    print(np.mean(a))

    mu = 0.1
    sigma = 10
    n = len(a)
    gamma = 0.01
    iterations = 0
    last_mu = 100000
    change = 0.1
    start = datetime.datetime.now()
    while change > 0.0001:
        first_derivative = (1/sigma) * (np.sum(a) - n * mu)
        second_derivative = -1 * n
        mu = mu - first_derivative / second_derivative

        change = abs(mu - last_mu)
        last_mu = mu
        iterations += 1
    print("Iterations taken for convergence (newton): %d" % iterations)
    print("Time taken: %.5f" % (datetime.datetime.now() - start).total_seconds())

def sigmoid(t):
    if t < 0:
        return 1 - 1/(1+math.exp(t))
    else:
        return 1/(1+math.exp(-t))


def calc_logistic_gradient(param_vector, X, y):
    param_gradient = np.zeros(len(param_vector))
    T = X.dot(param_vector)
    for i in range(X.shape[0]):
        diff = y[i] - sigmoid(T[i])
        param_gradient += diff * X[i, :]
    return param_gradient


def logistic_regression_model(X, y, gamma=0.00001, tol=1e-6):
    data_X = np.array(X)
    # Adding column of ones for the B0 intercept term
    one_col = np.ones((data_X.shape[0], 1))
    input_X = np.concatenate([one_col, data_X], axis=1)
    # param_vector = np.ones(X.shape[1] + 1)
    param_vector = np.array([1.,-1.,1.])
    param_change = tol + 1
    start = datetime.datetime.now()
    while param_change > tol:
        prev_params = np.copy(param_vector)
        param_gradient = calc_logistic_gradient(param_vector, input_X, y)
        param_vector += gamma * param_gradient
        param_change = np.linalg.norm(param_vector - prev_params)
    finish = datetime.datetime.now()
    time_taken = (finish - start).total_seconds() / 60.
    print("Time taken %.5f" % time_taken)

    print(param_vector)
    return param_vector


def create_one_hot(index, length):
    one_hot = np.zeros(length)
    one_hot[index] = 1
    return one_hot.reshape((-1,1))


# This takes in a lot of parameters, look into simplifying the function definition here
def margin_model_derivative_z(response, design_matrix, param_vector, indicators, weights, z, prior_means, prior_vars, MAP=False):
    """
    Method for calculating the full dataset derivative of the log likelihood of the margin model, with respect to the latent team variables
    :param response: The home margins of victory (N x 1 vector)
    :param design_matrix: The transformed matrix for predictions (N x d matrix)
    :param param_vector: The coefficients to be multiplied by each matrix row ((d+1) x 1 vector) with last element being model variance
    :param indicators: Index numbers of away, home pairs for each example (N x 2 matrix)
    :param weights: Training example weights to give/remove emphasis to specific games
    :param z: The latent variables as a vector (K x 1 vector)
    :param prior_means: The means of the latent variable prior distributions (K x 1 vector)
    :param prior_vars: The variances of the latent variable prior distributions (K x 1 vector)
    :param MAP: Boolean whether to return the MAP estimate (=True) or the MLE (=False), default is MLE
    :return: gradient vector of the log likelihood with respect to the model parameters (to be used in gradient descent update step
    """

    # Calcuating necessary elements for gradient calculation
    K_teams = len(z)
    gradient = np.zeros(K_teams).reshape((-1,1))
    second_gradient = np.zeros(K_teams).reshape((-1,1))
    predictions = design_matrix.dot(param_vector[:-1])
    difference = response - predictions
    away_derivatives = (param_vector[0] / param_vector[-1]) * difference
    home_derivatives = (param_vector[1] / param_vector[-1]) * difference

    # Summing gradient into respective team latent variables
    for i in range(design_matrix.shape[0]):
        away_indicator_vector = create_one_hot(indicators[i][0], K_teams)
        home_indicator_vector = create_one_hot(indicators[i][1], K_teams)
        gradient += weights[i] * (away_indicator_vector * away_derivatives.loc[i,0] + home_indicator_vector * home_derivatives.loc[i,0])
        second_gradient += weights[i] * (away_indicator_vector * -1 * param_vector[0] ** 2 / param_vector[-1] + home_indicator_vector * -1 * param_vector[1]**2 / param_vector[-1])

    # Adjusting gradient for MAP estimate (prior over latent variables)
    if MAP:
        MAP_gradient = -1 * (z - prior_means) / prior_vars
        gradient += MAP_gradient

    return gradient, second_gradient


def latent_margin_gradient_descent(response, design_matrix, param_vector, indicators, weights, z, prior_means, prior_vars, MAP=False, show=False, newton_update=True, gamma=0.0001, tol=1e-06, max_iter=100):
    """
    Function for performing gradient descent on latent variables of the margin model
    (Finding the latent variable vector that minimizes the log-likelihood of the margin model given fixed model parameters)
    :param response: The home margins of victory (N x 1 vector)
    :param design_matrix: The transformed pandas DataFrame for predictions (N x d matrix)
    :param param_vector: The coefficients to be multiplied by each matrix row ((d+1) x 1 vector) with last element being model variance
    :param indicators: Index numbers of away, home pairs for each example (N x 2 matrix)
    :param weights: Training example weights to give/remove emphasis to specific games
    :param z: The latent variables as a vector (K x 1 vector)
    :param prior_means: The means of the latent variable prior distributions (K x 1 vector)
    :param prior_vars: The variances of the latent variable prior distributions (K x 1 vector)
    :param MAP: Boolean whether to return the MAP estimate (=True) or the MLE (=False), default is MLE
    :param gamma: step size for gradient descent
    :param tol: minimum change allowed for termination of gradient descent
    :param max_iter: maximum amount of iterations allowed in gradient descent before termination
    :return: z: the latent variable vector that minimizes the log-likelihood of the margin model given fixed model parameters (param_vector)
    """
    z_change = tol + 1
    iterations = 0
    start = datetime.datetime.now()
    # Run until no change in latent variables or a maximum amount of iterations reached
    while z_change > tol and iterations < max_iter:
        iterations += 1
        prev_z = np.copy(z)
        # Calculate gradient of data under margin model with latent variables
        z_gradient, z_second_gradient = margin_model_derivative_z(response, design_matrix, param_vector, indicators, weights, z=prev_z,
                                               prior_means=prior_means, prior_vars=prior_vars, MAP=MAP)

        # Take a gradient step and calculate change in latent variable vector
        if not newton_update:
            z += gamma * np.array(z_gradient).reshape(-1,1)
        if newton_update:
            z -= z_gradient / z_second_gradient
        z_change = np.linalg.norm(z - prev_z)
        design_matrix = replace_design_latent(design_matrix=design_matrix, indicators=indicators, z=z)

        if show:
            print("Expectation Optimization Iteration: %d Latent Change: %.8f" % (iterations, z_change))

    if iterations == max_iter:
        print("Maximum iterations (%d) reached for termination of expectation optimization" % max_iter)

    finish = datetime.datetime.now()
    time_taken = (finish - start).total_seconds() / 60.
    if show:
        print("Time taken (minutes): %.5f" % time_taken)

    return z
