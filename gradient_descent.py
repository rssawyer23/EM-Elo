import numpy as np
import pandas as pd
import math
import datetime
from data_engineering import replace_design_latent
from data_engineering import replace_design_joint_latent


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
def margin_model_derivative_z(response, design_matrix, param_vector, intercept, indicators, weights, z, prior_means, prior_vars, MAP=False):
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
    # REQUIRES AWAY AND HOME COEFFICIENTS TO BE IN PARAM_VECTOR[0] AND PARAM_VECTOR[1] RESPECTIVELY
    # REQUIRES MODEL PRECISION IN PARAM_VECTOR[-1]
    # OTHER DIMS OF PARAMETER VECTOR SHOULD MATCH DESIGN MATRIX COLUMNS
    K_teams = int(len(z) / 2)
    gradient = np.zeros(K_teams).reshape((-1,1))
    second_gradient = np.zeros(K_teams).reshape((-1,1))
    predictions = design_matrix.dot(param_vector[:-1]) + intercept
    difference = response - predictions
    away_derivatives = (param_vector[0] / param_vector[-1]) * difference
    home_derivatives = (param_vector[1] / param_vector[-1]) * difference

    team_updates = [[] for _ in range(K_teams)]

    # Summing gradient into respective team latent variables
    for i in range(design_matrix.shape[0]):
        away_indicator_vector = create_one_hot(indicators[i][0], K_teams)
        home_indicator_vector = create_one_hot(indicators[i][1], K_teams)
        gradient += weights[i] * (away_indicator_vector * away_derivatives.loc[i,0] + home_indicator_vector * home_derivatives.loc[i,0])
        second_gradient += weights[i] * (away_indicator_vector * -1 * param_vector[0] ** 2 / param_vector[-1] + home_indicator_vector * -1 * param_vector[1]**2 / param_vector[-1])

        team_updates[indicators[i][0]].append(weights[i] * away_derivatives.loc[i,0])
        team_updates[indicators[i][1]].append(weights[i] * home_derivatives.loc[i,0])
    # Adjusting gradient for MAP estimate (prior over latent variables)
    if MAP:
        MAP_gradient = -1 * (z - prior_means) / prior_vars
        gradient += MAP_gradient
        second_gradient += -1 / prior_vars

    return gradient, second_gradient, np.array([np.std(team_game_updates) for team_game_updates in team_updates])


def joint_model_derivative_z(response, design_matrix, a_cols, h_cols, param_vector_a, param_vector_h, int_a, int_h, indicators, weights, z, prior_means, prior_vars, MAP=False):
    """
    Method for calculating the full dataset derivative of the log likelihood of the joint model, with respect to the latent team offense and defense variables
    :param response: A dataframe of dimension N x 2 with columns for AwayScore and HomeScore
    :param design_matrix: The transformed matrix for predictions (N x d matrix), incldues columns for AwayOffense, HomeDefense, HomeOffense, AwayDefense
    :param param_vector_a: The coefficients for predicting the AwayScore to be multiplied by each matrix row ((d+1) x 1 vector) with last element being model variance
    :param param_vector_h: The coefficients for predicting the HomeScore to be multiplied by each matrix row ((d+1) x 1 vector) with last element being model variance
    :param indicators: Index numbers of away, home pairs for each example (N x 2 matrix)
    :param weights: Training example weights to add/remove emphasis to specific games
    :param z: The latent variables as a vector (2K x 1 vector), offense ratings then defense ratings (starting at index K)
    :param prior_means: The means of the latent variable prior distributions (2K x 1 vector)
    :param prior_vars: The variances of the latent variable prior distributions (2K x 1 vector)
    :param MAP: Boolean whether to return the MAP estimate (=True) or the MLE (=False), default is MLE
    :return: gradient vector of the log likelihood with respect to the model parameters (to be used in gradient descent update step)
    """

    # Calcuating necessary elements for gradient calculation
    # REQUIRES AWAY AND HOME COEFFICIENTS TO BE IN PARAM_VECTOR[0] AND PARAM_VECTOR[1] RESPECTIVELY
    # REQUIRES MODEL PRECISION IN PARAM_VECTOR[-1]
    # OTHER DIMS OF PARAMETER VECTOR SHOULD MATCH DESIGN MATRIX COLUMNS
    K_teams = int(len(z) / 2)
    gradient = np.zeros(2 * K_teams).reshape((-1,1))
    second_gradient = np.zeros(2 * K_teams).reshape((-1,1))

    # Away Score-based derivatives
    away_predictions = design_matrix.loc[:, a_cols].dot(param_vector_a[:-1]) + int_a
    away_difference = response[" Away Points"].reshape(-1,1) - away_predictions
    away_offense_derivatives = (param_vector_a[0] / param_vector_a[-1]) * away_difference
    home_defense_derivatives = (param_vector_a[1] / param_vector_a[-1]) * away_difference

    # Home Score-based derivatives
    home_predictions = design_matrix.loc[:, h_cols].dot(param_vector_h[:-1]) + int_h
    home_difference = response[" Home Points"].reshape(-1,1) - home_predictions
    home_offense_derivatives = (param_vector_h[0] / param_vector_h[-1]) * home_difference
    away_defense_derivatives = (param_vector_h[1] / param_vector_h[-1]) * home_difference

    team_updates = [[] for _ in range(2 * K_teams)]

    # Summing gradient into respective team latent variables
    for i in range(design_matrix.shape[0]):
        away_offense_indicator_vector = create_one_hot(indicators[i][0], 2 * K_teams)
        away_defense_indicator_vector = create_one_hot(indicators[i][0] + K_teams, 2 * K_teams)
        home_offense_indicator_vector = create_one_hot(indicators[i][1], 2 * K_teams)
        home_defense_indicator_vector = create_one_hot(indicators[i][1] + K_teams, 2 * K_teams)

        # These do not need to be separated, just easier to read this way
        # Away score gradient adding
        gradient += weights[i] * (away_offense_indicator_vector * away_offense_derivatives.loc[i, 0] + home_defense_indicator_vector * home_defense_derivatives.loc[i, 0])
        second_gradient += weights[i] * (away_offense_indicator_vector * -1 * param_vector_a[0] ** 2 / param_vector_a[-1] + home_defense_indicator_vector * -1 * param_vector_a[1]**2 / param_vector_a[-1])

        team_updates[indicators[i][0]].append(weights[i] * away_offense_derivatives.loc[i,0])
        team_updates[indicators[i][1] + K_teams].append(weights[i] * home_defense_derivatives.loc[i,0])

        # Home score gradient adding
        gradient += weights[i] * (away_defense_indicator_vector * away_defense_derivatives.loc[i, 0] + home_offense_indicator_vector * home_offense_derivatives.loc[i, 0])
        second_gradient += weights[i] * (away_defense_indicator_vector * -1 * param_vector_h[1] ** 2 / param_vector_h[-1] + home_offense_indicator_vector * -1 * param_vector_h[0] ** 2 / param_vector_h[-1])

        team_updates[indicators[i][0] + K_teams].append(weights[i] * away_defense_derivatives.loc[i, 0])
        team_updates[indicators[i][1]].append(weights[i] * home_offense_derivatives.loc[i, 0])

    # Adjusting gradient for MAP estimate (prior over latent variables)
    if MAP:
        MAP_gradient = -1 * (z - prior_means) / prior_vars
        gradient += MAP_gradient
        second_gradient += -1 / prior_vars

    return gradient, second_gradient, np.array([np.std(team_game_updates) for team_game_updates in team_updates])


def latent_margin_optimization(response, design_matrix, param_vector, intercept, indicators, weights, z, prior_means, prior_vars, a_cols=None, h_cols=None, joint=False, MAP=False, show=False, newton_update=True, gamma=0.0001, tol=1e-06, max_iter=10):
    """
    Function for performing numerical optimization on latent variables of the margin model
    (Finding the latent variable vector that minimizes the log-likelihood of the margin model given fixed model parameters)
    :param response: The home margins of victory (N x 1 vector)
    :param design_matrix: The transformed pandas DataFrame for predictions (N x d matrix)
    :param param_vector: The coefficients to be multiplied by each matrix row ((d+1) x 1 vector) with last element being model variance
        NOTE THIS SHOULD BE A DICTIONARY WITH KEYS FOR "Away" AND "Home" FOR THE RESPECTIVE MODELS IN THE JOINT SETTING
    :param intercept: Same as param_vector, intercept of the linear model needed for predictions, same dictionary warning as param_vector
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
    gradient_std = np.zeros(len(z))
    # Run until no change in latent variables or a maximum amount of iterations reached
    while z_change > tol and iterations < max_iter:
        iterations += 1
        prev_z = np.copy(z)
        # Calculate gradient of data under margin model with latent variables
        if not joint:
            z_gradient, z_second_gradient, gradient_stds = margin_model_derivative_z(response, design_matrix, param_vector, intercept, indicators, weights, z=prev_z,
                                                   prior_means=prior_means, prior_vars=prior_vars, MAP=MAP)
        else:  # Is joint
            z_gradient, z_second_gradient, gradient_stds = joint_model_derivative_z(response, design_matrix, a_cols, h_cols, param_vector["Away"], param_vector["Home"], intercept["Away"], intercept["Home"], indicators, weights, z=prev_z,
                                                   prior_means=prior_means, prior_vars=prior_vars, MAP=MAP)

        # Take a gradient step and calculate change in latent variable vector
        if not newton_update:
            z += gamma * np.array(z_gradient).reshape(-1,1)
        if newton_update:
            z -= z_gradient / z_second_gradient
        z_change = np.linalg.norm(z - prev_z)

        # Make this an if statement to check if doing joint or margin
        if not joint:
            design_matrix = replace_design_latent(design_matrix=design_matrix, indicators=indicators, z=z)
        else:  # Is joint
            design_matrix = replace_design_joint_latent(joint_design_matrix=design_matrix, indicators=indicators, jz=z)

        if show:
            print("Expectation Optimization Iteration: %d Latent Change: %.8f" % (iterations, z_change))

    if iterations == max_iter:
        print("Maximum iterations (%d) reached for termination of expectation optimization" % max_iter)

    finish = datetime.datetime.now()
    time_taken = (finish - start).total_seconds() / 60.
    if show:
        print("Time taken (minutes): %.5f" % time_taken)

    return z, gradient_stds
