import numpy as np
import pandas as pd
import math
import gradient_descent as gd
from sklearn.linear_model import LinearRegression
import data_engineering as de
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


def fit_margin_model(x, y, indicators, p_means, p_vars, MAP=False, show=False, tol=1e-07, max_iter=1):
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
    lm.intercept_ = np.array([0.5])
    # PARAM VECTOR SHOULD BE AWAY COEF, HOME COEF, AWAY REST COEF, HOME REST COEF
    lm.coef_ = np.array([[-1.0, 1.0, -0.5, 0.5]])
    #lm.fit(X=x, y=y)
    param_vector = np.append(arr=lm.coef_, values=np.std(y)).reshape((-1, 1))

    change = tol + 1
    iterations = 0
    new_z, new_acc = np.copy(p_means), 0
    old_x = x.copy()

    # Continue alternating between E and M steps until the algorithm converges or reaches the maximum amount of iterations
    while change > tol and iterations < max_iter:
        start_acc = lm.score(X=old_x, y=y)
        print(lm.intercept_)
        print(lm.coef_)
        if show:
            print("EM Iteration %d start accuracy %.5f" % (iterations, start_acc))

        # Expectation step - solving for the optimal latent team variables given the linear model parameters (param_vector)
        new_z = gd.latent_margin_gradient_descent(response=y, design_matrix=old_x, param_vector=param_vector,
                                                     indicators=indicators,
                                                     weights=np.ones(len(y)).reshape((-1, 1)), z=new_z,
                                                     prior_means=p_means, prior_vars=p_vars, MAP=MAP, show=show)

        new_x = de.replace_design_latent(design_matrix=old_x, indicators=indicators, z=new_z)

        # Internal checks to make sure gradient descent step improved model
        finish_acc = lm.score(X=new_x, y=y)
        if show:
            print("EM Iteration %d after gradient descent accuracy %.5f" % (iterations, finish_acc))
        if finish_acc < start_acc:
            print("ERROR: EXPECTATION OPTIMIZATION DECREASED MODEL ACCURACY (from %0.5f to %0.5f)" % (start_acc, finish_acc))
        elif show:
            print("Expectation Step Accuracy Improvement: %.5f" % (finish_acc - start_acc))

        # Maximization step - solving for the linear model parameters given the latent team variables (z)
        lm.fit(X=new_x, y=y)
        print(lm.intercept_)
        print(lm.coef_)
        param_vector = np.append(arr=lm.coef_, values=np.std(y)).reshape((-1, 1))
        new_acc = lm.score(X=new_x, y=y)
        change = new_acc - finish_acc
        if show:
            print("Maximization Step Accuracy Improvement: %.5f" % change)
        iterations += 1
        old_x = new_x.copy()

    if iterations >= max_iter:
        print("WARNING: MAXIMUM ITERATIONS OF EM ALGORITHM REACHED")
    elif show:
        print("Finished EM with %d iterations" % iterations)
    return new_z, new_acc, lm


def calculate_test_accuracy(latent_z, model, indicators, design_matrix, response):
    input_matrix = de.replace_design_latent(design_matrix=design_matrix, indicators=indicators, z=latent_z).copy()
    accuracy = model.score(X=input_matrix, y=response)
    return accuracy

# Load data into appropriate format
input_filename = "NBAPointSpreadsAugmented.csv"
start = datetime.datetime.now()
print("%s Loading Data..." % datetime.datetime.now())
x, y, indicators, p_means, p_vars, z, latent_team_dictionary = de.load_data(input_filename)
team_list = list(latent_team_dictionary.keys())
season_row_dictionary = de.parse_seasons(input_filename)
print("...Finished Loading Data %s seconds" % (datetime.datetime.now() - start).total_seconds())

result_file = open("AccuracyOutput.csv", 'w')
result_file.write("Model,Season")
interval_size = 0.05 # This represents what percent of games should be used as accuracy bins
start_percent = 0.5
for i in np.arange(start_percent, 1.0, interval_size):
    result_file.write(",Train%.2f%%" % i)
result_file.write("\n")

sortable_team_dict = dict()
for team in latent_team_dictionary.keys():
    sortable_team_dict[team] = z[latent_team_dictionary[team]]
seasons_rating_df = pd.DataFrame.from_dict(sortable_team_dict, orient="index")
seasons_rating_df.columns = ["Initialization"]
seasons_rating_df.index = team_list

# Season carryover = True/False
# For model type (the 6 models + Elo baseline + Spread baseline?)
## for season in season_row_dictionary.keys():
## for season in [2008, 2009, 2010, 2011, 2012]:
### Get season split indices
####   Train on first split
####   Test on next split (record accuracy)
iteration = 0
season = 2008  # Will be replaced by for loop through seasons
MAP = False
model_type = "Margin"

accuracy_results = dict()
model_string = "%s-%s" % (model_type, "MLE" if not MAP else "MAP")


season_x = x.loc[season_row_dictionary[season], :]
season_x.index = range(season_x.shape[0])
season_y = y[season_row_dictionary[season]]  # This will need to change for the joint setting
season_indicators = indicators[season_row_dictionary[season], :]
season_games = season_x.shape[0]
season_accuracy = []

train_end = int((start_percent + interval_size * iteration) * season_games) # this should work as long as iteration ranges from 0 to (1 - start_percent)/interval_size
test_end = int(train_end + season_games * interval_size)
season_interval = "%d%%-%d%%" % (train_end / season_games * 100, test_end / season_games * 100)

season_z, train_accuracy, season_lm = fit_margin_model(season_x.iloc[0:train_end, :], season_y[0:train_end], season_indicators[0:train_end, :],
                 p_means, p_vars, MAP=MAP, show=False)

print("Train Accuracy with %s in %s season up to %d%% data: %.5f" % (model_string, season, train_end / season_games*100, train_accuracy))
test_accuracy = calculate_test_accuracy(season_z, season_lm, season_indicators[train_end:test_end, :], season_x.iloc[train_end:test_end, :], season_y[train_end:test_end])
print("Test Accuracy with %s in %d season %s interval: %.5f" % (model_string, season, season_interval, test_accuracy))
#accuracy_results[(model_type, season, season_interval)] = test_accuracy
season_accuracy.append(test_accuracy)

write_string = "%s,%s," % (model_string, season)
write_string += ",".join(["%.5f" % e for e in season_accuracy])
result_file.write(write_string+"\n")

# Post Model Fit Diagnostics
print("Season %d using %d%% for training data" % (season, train_end/season_games*100))
# print("Accuracy on next %d%% of games" % interval_size * 100)
print(season_lm.intercept_)
print(season_lm.coef_)

sortable_team_dict = dict()
for team in latent_team_dictionary.keys():
    sortable_team_dict[team] = season_z[latent_team_dictionary[team]]
team_rating_df = pd.DataFrame.from_dict(sortable_team_dict, orient="index")
team_rating_df.columns = ["%s-%d%%" % (season, train_end/season_games*100)]
print(team_rating_df.sort_values(by="%s-%d%%" % (season, train_end/season_games*100), axis=0, ascending=False))
seasons_rating_df = pd.concat([seasons_rating_df, team_rating_df], axis=1)
seasons_rating_df.to_csv("RatingsOutput.csv", index=True)  # Index is the team names


##TESTING
#fit_margin_model(x, y, indicators, p_means, p_vars, MAP=False, show=True)

# NOTES ON THINGS TO TRY
# Try with/without using spread as a feature
# Try using spread as dependent variable vs actual game score as dependent variable
# Try using interaction term between team ratings?
# Try carry-over between seasons (and test earlier portions of data)