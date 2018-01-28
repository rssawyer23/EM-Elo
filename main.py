import numpy as np
import pandas as pd
import math
import gradient_descent as gd
import computation_functions as cf
from sklearn.linear_model import LinearRegression
import data_engineering as de
import datetime
import scipy.stats as ss

# Global variables (should be replaced with arguments)
AWAY_SCORE_COLUMNS = ["AwayOffense", "HomeDefense", "AwayRest", "HomeRest"]
HOME_SCORE_COLUMNS = ["HomeOffense", "AwayDefense", "AwayRest", "HomeRest"]
MAXIMUM_ITERATIONS = 5


def elo_calculate_home_win_prob(away_rating, home_rating, home_offset=63.2):  # 20.9 comes from using home team 53% winning in NBA
    # Given away and home ratings, calculate probability of home win
    return 1. / (1. + math.pow(10, (away_rating - (home_rating + home_offset))/400.))


def calculate_temperature(iteration, start_k=16, decay_factor=0.0005):
    temp = start_k * np.e ** (-1 * decay_factor * iteration)
    return temp


# Unsure if making a function for this is necessary or if Elo should only be done for testing phase
# May be useful for later visualizations/rankings of teams from the models?
# Elo alterations - differential impact on magnitude of update, different values of k
def fit_elo_model(x, y, indicators, p_means, mode="Decay", k_step_high=32, k_step_low=16, show=False, home_advantage=20.8):
    """
    
    :param x: data matrix (n x d) with entries for team ratings (can default to 0s here)
    :param y: vector of margin of victories for the games (n x 1)
    :param indicators: matrix with team identifiers as entries (n x 2) matrix
    :param p_means: vector of prior means of the latent team variables (z x 1) that serve as the initial ratings
    :param k_step: controls the magnitude of an update to latent variables after a game outcome observation
    :param show: boolean determining whether additional information should be printed to the console
    :return: final latent variables from the Elo model
    """

    new_z = np.copy(p_means) + 1500.

    decay_factor = -1 * math.log1p(k_step_low / k_step_high) / indicators.shape[0]
    for i in range(len(y)):  # For each game in dataset
        # Get team ratings of teams playing
        away_elo = new_z[indicators[i][0]]
        home_elo = new_z[indicators[i][1]]

        # Calculate home win probability
        expected_result = elo_calculate_home_win_prob(away_rating=away_elo, home_rating=home_elo, home_offset=home_advantage)
        home_win = int(y[i] > 0)

        # Calculate error based on actual result
        error = home_win - expected_result
        if mode == "Decay":
            k_param = calculate_temperature(iteration=i, start_k=k_step_high, decay_factor=decay_factor)
        elif mode == "High":
            k_param = k_step_high
        else:  # mode=="Low"
            k_param = k_step_low
        update_amount = k_param * error

        # Update the team ratings of teams playing
        new_z[indicators[i][0]] += -1 * update_amount
        new_z[indicators[i][1]] += update_amount

    return new_z


def fit_margin_model(x, y, indicators, p_means, p_vars, weights, MAP=False, show=False, tol=1e-03, max_iter=1):
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
    # SET THIS IN A SMARTER WAY (VARIABLE LENGTH FOR COMPATIBILITY WITH OTHER DATASETS
    lm.coef_ = np.array([[-1.0, 1.0, -0.5, 0.5]])
    if x.shape[1] < len(lm.coef_[0]):
        lm.coef_ = np.array([lm.coef_[0][:(x.shape[1])]])

    param_vector = np.append(arr=lm.coef_, values=np.var(y)).reshape((-1, 1))

    change = tol + 1
    iterations = 0
    new_z, new_acc = np.copy(p_means), 0
    old_x = x.copy()

    if weights is None:
        weights = np.ones(len(y))

    # Continue alternating between E and M steps until the algorithm converges or reaches the maximum amount of iterations
    while change > tol and iterations < max_iter:
        start_acc = lm.score(X=old_x, y=y)
        if show:
            print(lm.intercept_)
            print(lm.coef_)
            print("EM Iteration %d start accuracy %.5f" % (iterations, start_acc))
        old_z = np.copy(new_z)
        # Expectation step - solving for the optimal latent team variables given the linear model parameters (param_vector)
        new_z, z_std = gd.latent_margin_optimization(response=y, design_matrix=old_x, param_vector=param_vector,
                                                     indicators=indicators, intercept=lm.intercept_,
                                                     weights=weights, z=new_z,
                                                     prior_means=p_means, prior_vars=p_vars, MAP=MAP, show=show)
        change = np.linalg.norm(new_z - old_z)
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
        lm.fit(X=new_x, y=y, sample_weight=weights)

        # OUTPUT FOR R ANALYSIS, DELETE LATER
        # merged = new_x.copy()
        # merged["MarginResponse"] = y
        # merged.to_csv("2008NBADataOutput.csv", index=False)

        if show:
            print(lm.intercept_)
            print(lm.coef_)
        param_vector = np.append(arr=lm.coef_, values=np.std(y)).reshape((-1, 1))
        new_acc = lm.score(X=new_x, y=y)
        acc_change = new_acc - finish_acc
        if show:
            print("Maximization Step Accuracy Improvement: %.5f" % acc_change)
        iterations += 1
        old_x = new_x.copy()

    if iterations >= max_iter:
        print("WARNING: MAXIMUM ITERATIONS (%d) OF EM ALGORITHM REACHED CHANGE:%d" % (iterations, change))
    elif show:
        print("Finished EM with %d iterations" % iterations)
    return new_z, new_acc, lm, z_std


def fit_joint_model(x, y, indicators, p_means, p_vars, a_cols, h_cols, weights, MAP=False, show=False, tol=1e-03, max_iter=1, int_a=97., int_h=100., home_points=" Home Points", away_points=" Away Points"):
    """
    Performs the Expectation-Maximization Algorithm for the Joint Model (regressions for AwayScore and HomeScore)
        Expectation = Calculate the optimal latent variables for each team's offense and defense given the linear model parameters
        Maximization = Calculate the two optimal linear regressions given the team latent variables (for AwayScore and HomeScore)
        Result is the final offense and defense latent variables and pair of models that can be used to predict future results

    :param x: data matrix (n x d) with entries for team ratings (can default to 0s here)
    :param y: dataframe of shape (N x 2) with columns for AwayScore and HomeScore
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
    lm_a = LinearRegression()
    lm_h = LinearRegression()

    lm_a.intercept_ = np.array([int_a])
    lm_h.intercept_ = np.array([int_h])

    # PARAM VECTOR SHOULD BE AWAY COEF, HOME COEF, AWAY REST COEF, HOME REST COEF
    # SET THIS IN A SMARTER WAY (VARIABLE LENGTH FOR COMPATIBILITY WITH OTHER DATASETS)
    lm_a.coef_ = np.array([[1.0, -1.0, 0.5, -0.5]])
    if len(a_cols) < len(lm_a.coef_[0]):
        lm_a.coef_ = np.array([lm_a.coef_[0][:len(a_cols)]])
    lm_h.coef_ = np.array([[1.0, -1.0, -0.5, 0.5]])
    if len(h_cols) < len(lm_h.coef_[0]):
        lm_h.coef_ = np.array([lm_h.coef_[0][:len(h_cols)]])

    param_vector = dict()
    intercept = dict()
    param_vector["Away"] = np.append(arr=lm_a.coef_, values=np.var(y.loc[:, away_points])).reshape((-1, 1))
    param_vector["Home"] = np.append(arr=lm_h.coef_, values=np.var(y.loc[:, home_points])).reshape((-1, 1))
    intercept["Away"] = lm_a.intercept_
    intercept["Home"] = lm_h.intercept_

    if weights is None:
        weights = np.ones(indicators.shape[0])

    change = tol + 1
    iterations = 0
    new_z, new_acc = np.copy(p_means), 0
    old_x = x.copy()

    # Continue alternating between E and M steps until the algorithm converges or reaches the maximum amount of iterations
    while change > tol and iterations < max_iter:
        start_acc, away_r2, home_r2, start_mae, away_mar, home_mae = calculate_joint_accuracy(X=old_x, y=y, lm_a=lm_a, lm_h=lm_h,
                                                               a_cols=a_cols, h_cols=h_cols,
                                                               home_points=home_points, away_points=away_points)
        if show:
            print("Away Coefficients")
            print(lm_a.intercept_)
            print(lm_a.coef_)
            print("Home Coefficients")
            print(lm_h.intercept_)
            print(lm_h.coef_)
            print("EM Iteration %d start accuracy %.5f" % (iterations, start_acc))

        old_z = np.copy(new_z)
        # Expectation step - solving for the optimal latent team variables given the linear model parameters (param_vector)
        # Using modified gradient descent here since Jacobian needed for multivariate setting for Newton updates
        new_z, z_std = gd.latent_margin_optimization(response=y, design_matrix=old_x, param_vector=param_vector,
                                                     indicators=indicators, intercept=intercept,
                                                     weights=weights, z=new_z,
                                                     prior_means=p_means, prior_vars=p_vars, MAP=MAP, show=show,
                                                     newton_update=False,
                                                     joint=True, a_cols=a_cols, h_cols=h_cols, home_points=home_points, away_points=away_points)

        change = np.linalg.norm(new_z - old_z)
        new_x = de.replace_design_joint_latent(joint_design_matrix=old_x, indicators=indicators, jz=new_z)

        # Internal checks to make sure gradient descent step improved model
        finish_acc, finish_away_r2, finish_home_r2, finish_mae, finish_away_mae, finish_home_mae = calculate_joint_accuracy(X=new_x, y=y, lm_a=lm_a, lm_h=lm_h,
                                                                              a_cols=a_cols, h_cols=h_cols, home_points=home_points, away_points=away_points)
        if show:
            print("EM Iteration %d after gradient descent accuracy %.5f" % (iterations, finish_acc))
        if finish_away_r2 < away_r2 and finish_home_r2 < home_r2:
            print("ERROR: EXPECTATION OPTIMIZATION DECREASED JOINT MODEL ACCURACY AWAY: (from %0.5f to %0.5f)" % (away_r2, finish_away_r2))
            print("ERROR: EXPECTATION OPTIMIZATION DECREASED JOINT MODEL ACCURACY HOME: (from %0.5f to %0.5f)" % (home_r2, finish_home_r2))
        elif show:
            print("Expectation Step Accuracy Improvement: %.5f" % (finish_acc - start_acc))

        # Maximization step - solving for the linear model parameters given the latent team variables (z)
        lm_a.fit(X=new_x.loc[:, a_cols], y=y.loc[:, away_points], sample_weight=weights)
        lm_h.fit(X=new_x.loc[:, h_cols], y=y.loc[:, home_points], sample_weight=weights)
        if show:
            print("Away Coefficients")
            print(lm_a.intercept_)
            print(lm_a.coef_)
            print("Home Coefficients")
            print(lm_h.intercept_)
            print(lm_h.coef_)

        param_vector["Away"] = np.append(arr=lm_a.coef_, values=np.std(y.loc[:, away_points])).reshape((-1, 1))
        param_vector["Home"] = np.append(arr=lm_h.coef_, values=np.std(y.loc[:, home_points])).reshape((-1, 1))
        intercept["Away"] = lm_a.intercept_
        intercept["Home"] = lm_h.intercept_
        new_acc, new_away_r2, new_home_r2, new_mae, new_away_mae, new_home_mae = calculate_joint_accuracy(X=new_x, y=y, lm_a=lm_a, lm_h=lm_h,
                                                                              a_cols=a_cols,
                                                                              h_cols=h_cols, home_points=home_points, away_points=away_points)
        acc_change = new_acc - finish_acc
        if show:
            print("Maximization Step Accuracy Improvement: %.5f" % acc_change)
        iterations += 1
        old_x = new_x.copy()

    if iterations >= max_iter:
        print("WARNING: MAXIMUM ITERATIONS (%d) OF EM ALGORITHM REACHED CHANGE:%d" % (iterations, change))
    elif show:
        print("Finished EM with %d iterations" % iterations)
    return new_z, new_acc, lm_a, lm_h, z_std


def calculate_joint_accuracy(X, y, lm_a, lm_h, a_cols, h_cols, home_points=" Home Points", away_points=" Away Points"):
    """
    Function for calculating the R2 of the margin accuracy using the joint model
    :param X: Input data for linear models
    :param y: N x 2 dataframe with column for Away Points and Home Points representing the response
    :param lm_a: 
    :param lm_h: 
    :param a_cols: list of the names of columns to use from X for the away model predictions
    :param h_cols: list of the names of columns to use from X for the home model predictions
    :param joint_z: offensive and defensive ratings to put into matrix
    :param indicators: N x 2 nparray of away, home team indices
    :return: R2 of margin accuracy using the joint model, R2 on away predictions, R2 on home predictions
    """
    away_predictions = lm_a.predict(X.loc[:, a_cols])
    home_predictions = lm_h.predict(X.loc[:, h_cols])
    margin_predictions = home_predictions - away_predictions
    margin_r2 = calculate_r2(margin_predictions, y.loc[:, home_points] - y.loc[:, away_points])
    away_r2 = calculate_r2(away_predictions, y.loc[:, away_points])
    home_r2 = calculate_r2(home_predictions, y.loc[:, home_points])
    margin_mae = np.mean(np.abs(margin_predictions.reshape(-1,1) - (y.loc[:, home_points] - y.loc[:, away_points]).values.reshape(-1,1)))
    home_mae = np.mean(np.abs(home_predictions.reshape(-1,1) - y.loc[:, home_points].values.reshape(-1,1)))
    away_mae = np.mean(np.abs(away_predictions.reshape(-1,1) - y.loc[:, away_points].values.reshape(-1,1)))
    return margin_r2, away_r2, home_r2, margin_mae, away_mae, home_mae


def calculate_r2(predictions, response):  # Now this is the same as calculate_margin_baseline
    squared_errors = (predictions.reshape(-1,1) - np.array(response).reshape(-1,1))**2
    rss = np.sum(squared_errors)
    tss = np.sum((np.mean(response) - response)**2)
    r2 = 1 - rss/tss
    return r2


def calculate_test_accuracy(latent_z, model, indicators, design_matrix, response):
    input_matrix = de.replace_design_latent(design_matrix=design_matrix, indicators=indicators, z=latent_z).copy()
    predictions = model.predict(X=input_matrix)
    squared_errors = (predictions - response) ** 2
    mae = np.mean(np.abs(predictions - response))
    mae_std = np.std(np.abs(predictions - response))
    accuracy = model.score(X=input_matrix, y=response)
    return accuracy, mae, mae_std, squared_errors[:, 0]  #Reshaping the squared errors to be of (x,) instead of (x,1) (for concatenation with empty list)


def calculate_team_variances(z_std, indicators):
    var_array = np.apply_along_axis(lambda row: z_std[row[0]]**2 + z_std[row[1]]**2, axis=1, arr=indicators)
    return var_array


def elo_test_accuracy(latent_z, indicators, design_matrix, response):
    input_matrix = de.replace_design_latent(design_matrix=design_matrix, indicators=indicators, z=latent_z).copy()
    proba_predictions = input_matrix.apply(lambda row: elo_calculate_home_win_prob(row["AwayRating"], row["HomeRating"]),
                                     axis=1)
    predictions = np.array(proba_predictions >= 0.5, dtype=int).reshape(-1,1)
    accuracy = np.mean(np.equal(predictions, response))
    return accuracy


# SAME FUNCTION AS CALCULATE R2, CHECK IF BOTH WORK WITH "RESHAPE"
def calculate_margin_baseline(spreads, actual):
    rss = np.sum((spreads.reshape(-1, 1) - actual)**2)
    tss = np.sum((np.mean(actual) - actual)**2)
    r2 = 1 - rss / tss
    mae = np.mean(np.abs(spreads.reshape(-1,1) - actual))
    return r2, mae


def calculate_binary_margin_baseline(spreads, actual):
    accuracy = np.mean(np.equal(spreads.reshape(-1,1) > 0, actual > 0))
    return accuracy


def generate_team_rating_df(latent_team_dict, z_vector, show=True):
    sortable_team_dict = dict()
    for team in latent_team_dict.keys():
        sortable_team_dict[team] = z_vector[latent_team_dict[team]]
    team_rating_df = pd.DataFrame.from_dict(sortable_team_dict, orient="index")
    team_rating_df.columns = ["%s-%d%%" % (season, train_end / season_games * 100)]
    if show:
        print(team_rating_df.sort_values(by="%s-%d%%" % (season, train_end / season_games * 100), axis=0, ascending=False))
    return team_rating_df


def generate_joint_rating_df(latent_team_dict, joint_z, std_z=None, show=True):
    offense_team_dict = dict()
    offense_std_dict = dict()
    defense_team_dict = dict()
    defense_std_dict = dict()
    num_teams = len(latent_team_dict.keys())
    for team in latent_team_dict.keys():
        offense_team_dict[team] = joint_z[latent_team_dict[team]][0]
        defense_team_dict[team] = joint_z[latent_team_dict[team] + num_teams][0]
        if std_z is not None:
            offense_std_dict[team] = std_z[latent_team_dict[team]]
            defense_std_dict[team] = std_z[latent_team_dict[team] + num_teams]

    team_rating_df = pd.DataFrame.from_dict(offense_team_dict, orient='index')
    team_rating_df.columns = ["OffenseRating"]
    team_rating_df["DefenseRating"] = pd.Series(defense_team_dict)
    if std_z is not None:
        team_rating_df["OffenseStd"] = pd.Series(offense_std_dict)
        team_rating_df["DefenseStd"] = pd.Series(defense_std_dict)
    if show:
        print(team_rating_df.head())
    return team_rating_df


def initialize_team_rating_df(latent_team_dict, z_vector):
    sortable_team_dict = dict()
    teams = list(latent_team_dict.keys())
    for team in teams:
        sortable_team_dict[team] = z_vector[latent_team_dict[team]]
    seasons_rating_df = pd.DataFrame.from_dict(sortable_team_dict, orient="index")
    seasons_rating_df.columns = ["Initialization"]
    seasons_rating_df.index = teams
    return seasons_rating_df


def calculate_binary_test_accuracy(latent_z, model, indicators, design_matrix, response):
    input_matrix = de.replace_design_latent(design_matrix=design_matrix, indicators=indicators, z=latent_z).copy()
    predictions = model.predict(X=input_matrix)
    accuracy = np.mean(np.equal(predictions > 0, response > 0))
    return accuracy


def calculate_joint_baseline(predictions, actual):  # Assumes away and home score in same index location (0 and 1)
    arss = np.sum((predictions.iloc[:,0] - actual.iloc[:,0])**2)
    hrss = np.sum((predictions.iloc[:,1] - actual.iloc[:,1])**2)
    atss = np.sum((actual.iloc[:,0].mean() - actual.iloc[:,0])**2)
    htss = np.sum((actual.iloc[:,1].mean() - actual.iloc[:,1])**2)
    ar2 = 1 - arss / atss
    hr2 = 1 - hrss / htss
    return ar2, hr2

if __name__ == "__main__":
    # Load data into appropriate format
    input_filename = "NBAPointSpreadsAugmented.csv"
    show = True
    start = datetime.datetime.now()
    print("%s Loading Data..." % datetime.datetime.now())
    x, joint_x, y_dict, indicators, p_means, p_vars, z, latent_team_dictionary, baseline_dict = de.load_data(input_filename)

    # Should replace these
    joint_p_means = np.concatenate([p_means, p_means])
    np.random.shuffle(joint_p_means)
    joint_p_vars = np.concatenate([p_vars, p_vars])
    np.random.shuffle(joint_p_vars)

    team_list = list(latent_team_dictionary.keys())
    season_row_dictionary = de.parse_seasons(input_filename)
    print("...Finished Loading Data %s seconds" % (datetime.datetime.now() - start).total_seconds())

    result_file = open("AccuracyOutput.csv", 'w')
    result_file.write("Model,Season")
    interval_size = 0.05  # This represents what percent of games should be used as accuracy bins
    start_percent = 0.5  # Indicates how far into season the testing should start
    for i in np.arange(start_percent, 1.0, interval_size):
        result_file.write(",Train%.2f%%" % i)
    result_file.write("\n")

    corr_file = open("VarianceAccuracyCorrelations.csv", 'w')
    corr_file.write("Season,Correlation,p-value\n")

    seasons_margin_rating_df = initialize_team_rating_df(latent_team_dictionary, z)
    seasons_differential_df = initialize_team_rating_df(latent_team_dictionary, z)
    seasons_margin_std_df = initialize_team_rating_df(latent_team_dictionary, z)
    seasons_elod_rating_df = initialize_team_rating_df(latent_team_dictionary, z)
    seasons_elol_rating_df = initialize_team_rating_df(latent_team_dictionary, z)
    seasons_eloh_rating_df = initialize_team_rating_df(latent_team_dictionary, z)

    # Season carryover = True/False
    # For model type (the 6 models + Elo baseline + Spread baseline?)
    ## for season in season_row_dictionary.keys():
    ## for season in [2008, 2009, 2010, 2011, 2012]:
    ### Get season split indices
    #### for iteration in range((1-start_percent)/interval_size)
    ####   Train on first split
    ####   Test on next split (record accuracy)
    #season = 2008  # Will be replaced by for loop through seasons
    MAP = False
    model_type = "Margin"

    accuracy_results = dict()
    model_string = "%s-%s" % (model_type, "MLE" if not MAP else "MAP")
    first_words = ["Margin-MLE", "Margin-MLE-MAE", "Margin-MLE-Binary",
                   "Margin-Baseline", "Margin-Baseline-MAE",
                   "Margin-Baseline-Binary", "Margin-MLE-Train",
                   "Margin-Differential-R2",
                   "Margin-Differential-Binary", "Margin-Differential-MAE",
                   "Binary-EloL", "Binary-EloH", "Binary-EloD",
                   "Joint-Baseline-Away","Joint-Baseline-Home",
                   "Joint-MLE-R2", "Joint-Away-R2", "Joint-Home-R2",
                   "Joint-MLE-MAE", "Joint-Away-MAE", "Joint-Home-MAE",
                   "Joint-Differential-Binary", "Joint-Differential-MAE",
                   "Joint-Differential-MAEHome", "Joint-Differential-MAEAway",
                   "Joint-Differential-R2", "Joint-Differential-R2Home", "Joint-Differential-R2Away"]

    for season in [2008, 2009, 2010, 2011, 2012]:
        season_x = x.loc[season_row_dictionary[season], :]
        season_x.index = range(season_x.shape[0])
        season_joint_x = joint_x.loc[season_row_dictionary[season], :]
        season_joint_x.index = range(season_joint_x.shape[0])
        season_margin_y = np.array(y_dict["Margin"]).reshape(-1,1)[season_row_dictionary[season]]
        season_margin_baseline = np.array(baseline_dict["Margin"]).reshape(-1,1)[season_row_dictionary[season]]
        season_binary_y = np.array(y_dict["Binary"]).reshape(-1,1)[season_row_dictionary[season]]
        season_joint_y = y_dict["Joint"].loc[season_row_dictionary[season],:]
        season_joint_y.index = range(season_joint_y.shape[0])
        season_joint_baseline = baseline_dict["Joint"].loc[season_row_dictionary[season], :]
        season_joint_baseline.index = range(season_joint_baseline.shape[0])
        season_indicators = indicators[season_row_dictionary[season], :]
        season_games = season_x.shape[0]
        season_errors = np.array([])
        season_variances = np.array([])

        accuracy_dictionary = dict()
        for w in first_words:
            accuracy_dictionary[w] = []

        # Taking a small portion of next games to use as test set and games up to that point as training set
        for iteration in range(int((1-start_percent)/interval_size)):
            train_end = int((start_percent + interval_size * iteration) * season_games)
            test_end = int(train_end + season_games * interval_size)
            season_interval = "%d%%-%d%%" % (train_end / season_games * 100, test_end / season_games * 100)

            # ## RUNNING BINARY MODEL AND BASELINE
            # season_z, margin_train, season_lm = fit_binary_model(season_x.iloc[0:train_end, :], season_y[0:train_end],
            #                                                     season_indicators[0:train_end, :],
            #                                                     p_means, p_vars, MAP=MAP, show=False)
            elo_zl = fit_elo_model(x=season_x.iloc[0:train_end, :], y=season_binary_y[0:train_end], indicators=season_indicators[0:train_end, :],
                             p_means=p_means, mode="Low", k_step_low=8, k_step_high=16, show=False)
            elo_zh = fit_elo_model(x=season_x.iloc[0:train_end, :], y=season_binary_y[0:train_end], indicators=season_indicators[0:train_end, :],
                             p_means=p_means, mode="High", k_step_low=8, k_step_high=16, show=False)
            elo_zd = fit_elo_model(x=season_x.iloc[0:train_end, :], y=season_binary_y[0:train_end], indicators=season_indicators[0:train_end, :],
                             p_means=p_means, mode="Decay", k_step_low=8, k_step_high=16, show=False)

            accuracy_dictionary["Binary-EloL"].append(elo_test_accuracy(elo_zl, season_indicators[train_end:test_end, :], season_x.iloc[train_end:test_end, :], season_binary_y[train_end:test_end]))
            accuracy_dictionary["Binary-EloH"].append(elo_test_accuracy(elo_zh, season_indicators[train_end:test_end, :], season_x.iloc[train_end:test_end, :], season_binary_y[train_end:test_end]))
            accuracy_dictionary["Binary-EloD"].append(elo_test_accuracy(elo_zd, season_indicators[train_end:test_end, :], season_x.iloc[train_end:test_end, :], season_binary_y[train_end:test_end]))

        #     ## RUNNING MARGIN MODEL AND BASELINE-SPREADS
        #     season_z, margin_train, season_lm, season_z_std = fit_margin_model(season_x.iloc[0:train_end, :], season_margin_y[0:train_end], season_indicators[0:train_end, :],
        #                      p_means, p_vars, max_iter=MAXIMUM_ITERATIONS, MAP=MAP, show=False)
        #     test_accuracy, test_mae, mae_std, test_errors = calculate_test_accuracy(season_z, season_lm, season_indicators[train_end:test_end, :], season_x.iloc[train_end:test_end, :], season_margin_y[train_end:test_end])
        #     game_team_variances = calculate_team_variances(season_z_std, season_indicators[train_end:test_end, :])
        #     season_errors = np.concatenate([season_errors, test_errors])
        #     season_variances = np.concatenate([season_variances, game_team_variances])
        #
        #     ## Calculating baselines to directly compare the margin model to
        #     b_test_accuracy = calculate_binary_test_accuracy(season_z, season_lm, season_indicators[train_end:test_end, :], season_x.iloc[train_end:test_end, :], season_margin_y[train_end:test_end])
        #     baseline_accuracy, baseline_mae = calculate_margin_baseline(season_margin_baseline[train_end:test_end], season_margin_y[train_end:test_end])
        #     b_baseline_accuracy = calculate_binary_margin_baseline(season_margin_baseline[train_end:test_end], season_margin_y[train_end:test_end])
        #     diff_R2, diff_MAE, diff_Bin, diff_z, diff_games = cf.calculate_team_average_margins(y=season_margin_y, indicators=season_indicators, train_end=train_end, z_vec=season_z, test_end=test_end)
        #
        #     accuracy_dictionary["Margin-MLE-Train"].append(margin_train)
        #     accuracy_dictionary["Margin-MLE"].append(test_accuracy)
        #     accuracy_dictionary["Margin-MLE-MAE"].append(test_mae)
        #     accuracy_dictionary["Margin-MLE-Binary"].append(b_test_accuracy)
        #     accuracy_dictionary["Margin-Baseline"].append(baseline_accuracy)
        #     accuracy_dictionary["Margin-Baseline-MAE"].append(baseline_mae)
        #     accuracy_dictionary["Margin-Baseline-Binary"].append(b_baseline_accuracy)
        #     accuracy_dictionary["Margin-Differential-Binary"].append(diff_Bin)
        #     accuracy_dictionary["Margin-Differential-MAE"].append(diff_MAE)
        #     accuracy_dictionary["Margin-Differential-R2"].append(diff_R2)
        #
        #     if show:
        #         print("Test Accuracy with %s in %d season %s interval: %.5f" % (model_string, season, season_interval, test_accuracy))
        #         print("Train Accuracy with %s in %s season up to %d%% data: %.5f" % (model_string, season, train_end / season_games * 100, margin_train))
        #         print("Baseline Accuracy with %s in %d season %s interval: %.5f" % (model_string, season, season_interval, baseline_accuracy))
        #
        #         # Post Model Fit Diagnostics
        #         print("Season %d using %d%% for training data" % (season, train_end/season_games*100))
        #         # print("Accuracy on next %d%% of games" % interval_size * 100)
        #         print(season_lm.intercept_)
        #         print(season_lm.coef_)
        #
        #     differential_df = generate_team_rating_df(latent_team_dictionary, diff_z, show=False)
        #     seasons_differential_df = pd.concat([seasons_differential_df, differential_df], axis=1)
        #
        #     margin_rating_df = generate_team_rating_df(latent_team_dictionary, season_z, show=False)
        #     seasons_margin_rating_df = pd.concat([seasons_margin_rating_df, margin_rating_df], axis=1)
        #
        #     margin_std_df = generate_team_rating_df(latent_team_dictionary, season_z_std, show=False)
        #     seasons_margin_std_df = pd.concat([seasons_margin_std_df, margin_std_df], axis=1)
        #
        #     elod_rating_df = generate_team_rating_df(latent_team_dictionary, elo_zd, show=False)
        #     seasons_elod_rating_df = pd.concat([seasons_elod_rating_df, elod_rating_df], axis=1)
        #
        #     elol_rating_df = generate_team_rating_df(latent_team_dictionary, elo_zl, show=False)
        #     seasons_elol_rating_df = pd.concat([seasons_elol_rating_df, elol_rating_df], axis=1)
        #
        #     eloh_rating_df = generate_team_rating_df(latent_team_dictionary, elo_zh, show=False)
        #     seasons_eloh_rating_df = pd.concat([seasons_eloh_rating_df, eloh_rating_df], axis=1)
        #
        #     ## RUNNING JOINT MODEL AND BASELINE
        #     # fit function
        #     season_joint_z, joint_train, season_lm_a, season_lm_h, season_joint_z_std = fit_joint_model(season_joint_x.iloc[0:train_end, :], season_joint_y.iloc[0:train_end,:], season_indicators[0:train_end, :],
        #                      joint_p_means, joint_p_vars, max_iter=MAXIMUM_ITERATIONS, a_cols=AWAY_SCORE_COLUMNS, h_cols=HOME_SCORE_COLUMNS, MAP=MAP, show=False)
        #     # test accuracy
        #     input_matrix = de.replace_design_joint_latent(joint_design_matrix=season_joint_x.iloc[train_end:test_end, :], indicators=season_indicators[train_end:test_end,:],
        #                                             jz=season_joint_z).copy()
        #     joint_test_accuracy, joint_away, joint_home, joint_mae, joint_away_mae, joint_home_mae = calculate_joint_accuracy(X=input_matrix, y=season_joint_y.iloc[train_end:test_end, :],
        #                                                    lm_a=season_lm_a, lm_h=season_lm_h, a_cols=AWAY_SCORE_COLUMNS, h_cols=HOME_SCORE_COLUMNS)
        #     j_baseline_away_accuracy, j_baseline_home_accuracy = calculate_joint_baseline(season_joint_baseline.loc[train_end:test_end, :], season_joint_y.loc[train_end:test_end, :])
        #     r2_t, r2_h, r2_a, mae_t, mae_h, mae_a, binary_accuracy, z_joint, z_games = cf.calculate_team_average_joint(y=season_joint_y, indicators=season_indicators, train_end=train_end, z_vec=season_joint_z, test_end=test_end)
        #
        #     accuracy_dictionary["Joint-Baseline-Away"].append(j_baseline_away_accuracy)
        #     accuracy_dictionary["Joint-Baseline-Home"].append(j_baseline_home_accuracy)
        #     accuracy_dictionary["Joint-Away-R2"].append(joint_away)
        #     accuracy_dictionary["Joint-Home-R2"].append(joint_home)
        #     accuracy_dictionary["Joint-MLE-R2"].append(joint_test_accuracy)
        #     accuracy_dictionary["Joint-Away-MAE"].append(joint_away_mae)
        #     accuracy_dictionary["Joint-Home-MAE"].append(joint_home_mae)
        #     accuracy_dictionary["Joint-MLE-MAE"].append(joint_mae)
        #     accuracy_dictionary["Joint-Differential-Binary"].append(binary_accuracy)
        #     accuracy_dictionary["Joint-Differential-R2"].append(r2_t)
        #     accuracy_dictionary["Joint-Differential-R2Home"].append(r2_h)
        #     accuracy_dictionary["Joint-Differential-R2Away"].append(r2_a)
        #     accuracy_dictionary["Joint-Differential-MAE"].append(mae_t)
        #     accuracy_dictionary["Joint-Differential-MAEHome"].append(mae_h)
        #     accuracy_dictionary["Joint-Differential-MAEAway"].append(mae_a)
        #
        #     #Not appending, just returning the last dataframe of ratings
        #     joint_ratings = generate_joint_rating_df(latent_team_dictionary, season_joint_z, season_joint_z_std, show=False)
        #
        for first_word in first_words:
            write_string = "%s,%s," % (first_word, season)
            write_string += ",".join(["%.5f" % e for e in accuracy_dictionary[first_word]])
            result_file.write(write_string + "\n")
        #
        # corr_test = ss.pearsonr(x=season_errors,y =season_variances)
        # corr_file.write("%s,%.4f,%.4f\n" % (season, corr_test[0], corr_test[1]))
        # joint_ratings.to_csv("JointRatingsOutput%d.csv" % season, index=True)


    seasons_margin_rating_df.to_csv("MarginRatingsOutput.csv", index=True)  # Index is the team names
    seasons_differential_df.to_csv("DifferentialRatingsOutput.csv", index=True)
    seasons_margin_std_df.to_csv("MarginStdOutput.csv", index=True)
    seasons_elod_rating_df.to_csv("EloDRatingsOutput.csv", index=True)
    seasons_elol_rating_df.to_csv("EloLRatingsOutput.csv", index=True)
    seasons_eloh_rating_df.to_csv("EloHRatingsOutput.csv", index=True)


    ##TESTING
    #fit_margin_model(x, y, indicators, p_means, p_vars, MAP=False, show=True)

    # NOTES ON THINGS TO TRY
    # Try with/without using spread as a feature
    # Try using spread as dependent variable vs actual game score as dependent variable
    # Try using interaction term between team ratings?
    # Try carry-over between seasons (and test earlier portions of data)