# File for extra calculation functions used in NBA and General files
import numpy as np
import pandas as pd
import data_engineering as de


def generate_team_rating_df(latent_team_dict, z_vector, show=True):
    sortable_team_dict = dict()
    for team in latent_team_dict.keys():
        sortable_team_dict[team] = z_vector[latent_team_dict[team]]
    team_rating_df = pd.DataFrame.from_dict(sortable_team_dict, orient="index")
    team_rating_df.columns = ["1"]
    if show:
        print(team_rating_df.sort_values(by="1", axis=0, ascending=False))
    return team_rating_df


def calculate_team_average_margins_ro(y, indicators, train_end, z_vec):
    # Initializing
    z_margin = np.zeros(len(z_vec))
    z_games = np.zeros(len(z_vec))

    # Train process
    for i in range(train_end):
        z_margin[indicators[i, 0]] -= y[i]
        z_games[indicators[i, 0]] += 1

        z_margin[indicators[i, 1]] += y[i]
        z_games[indicators[i, 1]] += 1
    z_margin = z_margin / z_games
    return z_margin, z_games


def calculate_team_average_joint(y, indicators, train_end, z_vec, test_end=-1):
    z_joint = np.zeros(len(z_vec) * 2)
    z_games = np.zeros(len(z_vec))
    offset = len(z_vec)


    if test_end == -1:
        test_end = indicators.shape[0]

    # Calculate average for, average against for home and away teams in training set
    # Currently indicators are [away, home] but joint response is [home, away]
    for i in range(train_end):
        z_joint[indicators[i, 0]] += y.iloc[i, 1]  # Updating away team's offense by adding away points scored
        z_joint[indicators[i, 0] + offset] += y.iloc[i, 0]  # Updating away team's defense by adding home points allowed
        z_games[indicators[i, 0]] += 1

        z_joint[indicators[i, 1]] += y.iloc[i, 0]  # Updating home team's offense by adding home points scored
        z_joint[indicators[i, 1] + offset] += y.iloc[i, 1]  # Updating home team's defense by adding away points scored
        z_games[indicators[i, 1]] += 1

    rss_t, rss_h, rss_a = 0, 0, 0
    sae_t, sae_h, sae_a = 0, 0, 0
    binary_correct = 0
    for i in range(train_end, test_end):
        # Calculating home and away score predictions as average of team_for and opp_against
        home_prediction = 0.5 * z_joint[indicators[i, 1]] / z_games[[indicators[i, 1]]] + 0.5 * z_joint[
            indicators[i, 0] + offset] / z_games[[indicators[i, 0]]]
        away_prediction = 0.5 * z_joint[indicators[i, 0]] / z_games[[indicators[i, 0]]] + 0.5 * z_joint[
            indicators[i, 1] + offset] / z_games[[indicators[i, 1]]]
        margin_prediction = home_prediction - away_prediction  # Predicting how many points home team should win by
        h_error = home_prediction - y.iloc[i, 0]
        a_error = away_prediction - y.iloc[i, 1]
        t_error = margin_prediction - (y.iloc[i, 0] - y.iloc[i, 1])
        binary_correct += np.equal(margin_prediction > 0, y.iloc[i, 0] > y.iloc[i, 1])
        sae_t += abs(t_error)
        sae_h += abs(h_error)
        sae_a += abs(a_error)
        rss_t += t_error ** 2
        rss_h += h_error ** 2
        rss_a += a_error ** 2

    home_margins = y.iloc[:, 0] - y.iloc[:, 1]
    tsst = np.sum((home_margins[train_end:test_end] - np.mean(home_margins[train_end:test_end])) ** 2)
    tssh = np.sum((y.iloc[train_end:test_end, 0] - np.mean(y.iloc[train_end:test_end, 0])) ** 2)
    tssa = np.sum((y.iloc[train_end:test_end, 1] - np.mean(y.iloc[train_end:test_end, 1])) ** 2)
    test_games = test_end - train_end

    r2_t, r2_h, r2_a = rss_t / tsst, rss_h / tssh, rss_a / tssa
    mae_t, mae_h, mae_a = sae_t / test_games, sae_h / test_games, sae_a / test_games
    binary_accuracy = binary_correct / test_games
    return r2_t, r2_h, r2_a, mae_t, mae_h, mae_a, binary_accuracy, z_joint, z_games


# Returns R2, MAE, % Correct, margin variables, and games played
def calculate_team_average_margins(y, indicators, train_end, z_vec, test_end=-1):
    """
    :param y: the margin of victories for teams
    :param indicators: a (n, 2) matrix with integer IDs for away team and home team respectively
    :param train_end: an integer representing the last game index (row) to be used in training
    :param z_vec: latent variable vector, dummy for getting number of teams in league
    :param test_end: an integer representing the last game index (row) to be used in testing 
    (test interval is from train_end to test_end)
    :return: R2, MAE, % binary correct, cumulative point differentials, and games played
    """
    # Initializing
    z_margin = np.zeros(len(z_vec))
    z_games = np.zeros(len(z_vec))
    home_intercept = 0

    if test_end == -1:
        test_end = indicators.shape[0]
    if test_end < train_end:
        print("Error: Test End < Train End")

    # Train process, calculating cumulative point margin for teams and games played (for average point margin)
    for i in range(train_end):
        z_margin[indicators[i, 0]] -= y[i]
        z_games[indicators[i, 0]] += 1

        z_margin[indicators[i, 1]] += y[i]
        z_games[indicators[i, 1]] += 1

        home_intercept += y[i]
    z_margin = z_margin / z_games
    home_intercept = home_intercept / float(train_end)

    # Test process, using average point margins for predictions with simple model assumptions
    rss = 0
    ae = []
    binary_correct = 0
    for i in range(train_end, test_end):
        prediction = -1.0 * z_margin[indicators[i, 0]] + 1.0 * z_margin[
            indicators[i, 1]] + home_intercept  # Calculating margin relative to home team
        binary_correct += int(np.equal(y[i] > 0, prediction > 0))
        error = prediction - y[i]
        ae.append(abs(error))
        rss += error ** 2

    tss = np.sum((y[train_end:test_end] - np.mean(y[train_end:test_end])) ** 2)
    test_games = test_end - train_end

    # Final calculations to return
    r_squared = 1 - rss / tss  # R-Squared calculation using residual sum of squares from predictions and total sum of squares from mean of training data
    mae = sum(ae) / test_games  # sum of Absolute Errors / test games is Mean Absolute Error
    std_ae = np.std(ae)
    binary_accuracy = binary_correct / test_games

    return r_squared, mae, std_ae, binary_accuracy, z_margin, z_games


# Function for initalizing the priors of latent variables
# Uses argument means and variances if they exist and are appropriate length,
# otherwise uses default values generated from load data
def initialize_priors(arg_p_means, arg_p_vars, def_p_means, def_p_vars, num_latent, latent_team_dictionary, filename=None):
    if filename is not None:
        joint_p_means = de.load_prior_ratings(latent_team_dictionary, filename, joint="Joint" in filename)
    else:
        if arg_p_means is not None and len(def_p_means) == num_latent:
            joint_p_means = def_p_means
        else:
            joint_p_means = np.concatenate([def_p_means, def_p_means])
            np.random.shuffle(joint_p_means)

    if arg_p_vars is not None and len(def_p_vars) == num_latent:
        joint_p_vars = def_p_vars
    else:
        joint_p_vars = np.concatenate([def_p_vars, def_p_vars])
        np.random.shuffle(joint_p_vars)

    return joint_p_means, joint_p_vars

