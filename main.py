import numpy as np
import pandas as pd
import math
import gradient_descent as gd
from sklearn.linear_model import LinearRegression
from data_engineering import replace_design_latent
from data_engineering import load_data
import datetime

def elo_calculate_home_win_prob(away_rating, home_rating, home_offset=0.5):
    # Given away and home ratings, calculate probability of home win
    return 1. / (1. + math.pow(math.e, away_rating - (home_rating + home_offset)))


# Unsure if making a function for this is necessary or if Elo should only be done for testing phase
# May be useful for later visualizations/rankings of teams from the models?
# Elo alterations - differential impact on magnitude of update, different values of k
def fit_elo_model(x, y, indicators, p_means, k_step=32, show=False, home_advantage=0):
    """
    
    :param x: data matrix (n x d) with entries for team ratings (can default to 0s here)
    :param y: vector of margin of victories for the games (n x 1)
    :param indicators: matrix with team identifiers as entries (n x 2) matrix
    :param p_means: vector of prior means of the latent team variables (z x 1) that serve as the initial ratings
    :param k_step: controls the magnitude of an update to latent variables after a game outcome observation
    :param show: boolean determining whether additional information should be printed to the console
    :return: final latent variables from the Elo model
    """

    new_z = np.copy(p_means)
    for i in range(len(y)): # For each game in dataset
        # Get team ratings of teams playing
        away_elo = new_z[indicators[i][0]]
        home_elo = new_z[indicators[i][1]]

        # Calculate home win probability
        expected_result = elo_calculate_home_win_prob(away_elo, home_elo + home_advantage)
        home_win = int(y[i] > 0)

        # Calculate error based on actual result
        error = home_win - expected_result
        update_amount = k_step * error

        # Update the team ratings of teams playing
        new_z[indicators[i][0]] += -1 * update_amount
        new_z[indicators[i][1]] += update_amount

    return new_z


def fit_margin_model(x, y, indicators, p_means, p_vars, MAP=False, show=False, tol=1e-07, max_iter=100):
    """
    Performs the Expectation-Maximization Algorithm for the Margin Model (point differential linear regression)
        Expectation = Calculate the optimal latent variables for each team given the linear model parameters
        Maximization = Calculate the optimal linear regression parameters given the team latent variables
        Result is the final team latent variables and model that can be used to predict future results
        
    :param x: data matrix (n x d) with entries for team ratings (can default to 0s here)
    :param y: vector of margin of victories for the games (n x 1)
    :param indicators: matrix with team identifiers as entries (n x 2) matrix
    :param p_means: vector of prior means of the latent team variables (z x 1)
    :param p_vars: vector of prior variances of the latent team variables (z x 1)
    :param MAP: boolean determining if MLE (False, default) should be used or the MAP estimate
    :param show: boolean determining whether additional information should be printed to the console
    :param tol: tolerance for convergence
    :param max_iter: maximum amount of iterations before terminating the algorithm
    :return: final latent variables (z x 1), final accuracy of the model (single float), and linear model object
    """

    # Initializing linear model parameters
    lm = LinearRegression()
    lm.fit(X=x, y=y)
    param_vector = np.append(arr=lm.coef_, values=np.std(y)).reshape((-1, 1))

    change = tol + 1
    iterations = 0
    new_z, new_acc = np.copy(p_means), 0
    old_x = x.copy()

    # Continue alternating between E and M steps until the algorithm converges or reaches the maximum amount of iterations
    while change > tol and iterations < max_iter:
        start_acc = lm.score(X=old_x, y=y)

        # Expectation step - solving for the optimal latent team variables given the linear model parameters (param_vector)
        new_z = gd.latent_margin_gradient_descent(response=y, design_matrix=old_x, param_vector=param_vector,
                                                     indicators=indicators,
                                                     weights=np.ones(len(y)).reshape((-1, 1)), z=new_z,
                                                     prior_means=p_means, prior_vars=p_vars, MAP=MAP, show=show)

        new_x = replace_design_latent(design_matrix=old_x, indicators=indicators, z=new_z)

        # Internal checks to make sure gradient descent step improves model
        finish_acc = lm.score(X=new_x, y=y)
        if finish_acc < start_acc:
            print("ERROR: GRADIENT DESCENT DECREASED MODEL ACCURACY")
        elif show:
            print("Expectation Accuracy Improvement: %.5f" % (finish_acc - start_acc))

        # Maximization step - solving for the linear model parameters given the latent team variables (z)
        lm.fit(X=new_x, y=y)
        param_vector = np.append(arr=lm.coef_, values=np.std(y)).reshape((-1, 1))
        new_acc = lm.score(X=new_x, y=y)
        change = new_acc - start_acc
        if show:
            print("Maximization Accuracy Improvement: %.5f" % change)
        iterations += 1
        old_x = old_x.copy()

    if iterations >= max_iter:
        print("WARNING: MAXIMUM ITERATIONS OF EM ALGORITHM REACHED")
    elif show:
        print("Finished with %d iterations" % iterations)
    return new_z, new_acc, lm

# Load data into appropriate format
input_filename = "NBAPointSpreadsAugmented.csv"
start = datetime.datetime.now()
print("%s Loading Data..." % datetime.datetime.now())
x, y, indicators, p_means, p_vars, z = load_data(input_filename)
print("...Finished Loading Data %s seconds" % (datetime.datetime.now() - start).total_seconds())

#TESTING
fit_margin_model(x, y, indicators, p_means, p_vars, MAP=False, show=True)

# Partition data using time-series cross validation
# Create 6 models (binary, margin, joint) x (MLE, MAP)
# Calculate accuracy on test set
# Calculate accuracy on test set using baseline (Elo)

# NOTES ON THINGS TO TRY
# Try with/without using spread as a feature
# Try using spread as dependent variable vs actual game score as dependent variable