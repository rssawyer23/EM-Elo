import numpy as np
import pandas as pd
import data_engineering as de
from main import initialize_team_rating_df
from main import fit_elo_model
from main import elo_test_accuracy
import main as m
import datetime
import scipy.stats as ss

# Global variables (should be replaced with arguments)
AWAY_SCORE_COLUMNS = ["AwayOffense", "HomeDefense"]
HOME_SCORE_COLUMNS = ["HomeOffense", "AwayDefense"]


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
        z_margin[indicators[i,0]] -= y[i]
        z_games[indicators[i,0]] += 1

        z_margin[indicators[i,1]] += y[i]
        z_games[indicators[i,1]] += 1
    z_margin = z_margin / z_games
    return z_margin, z_games


def calculate_team_average_joint(y, indicators, train_end, z_vec):
    z_joint = np.zeros(len(z_vec)*2)
    z_games = np.zeros(len(z_vec))
    offset = len(z_vec)

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
    for i in range(train_end, indicators.shape[0]):
        # Calculating home and away score predictions as average of team_for and opp_against
        home_prediction = 0.5 * z_joint[indicators[i, 1]] / z_games[[indicators[i, 1]]] + 0.5 * z_joint[indicators[i, 0] + offset] / z_games[[indicators[i, 0]]]
        away_prediction = 0.5 * z_joint[indicators[i, 0]] / z_games[[indicators[i, 0]]] + 0.5 * z_joint[indicators[i, 1] + offset] / z_games[[indicators[i, 1]]]
        margin_prediction = home_prediction - away_prediction  # Predicting how many points home team should win by
        h_error = home_prediction - y.iloc[i, 0]
        a_error = away_prediction - y.iloc[i, 1]
        t_error = margin_prediction - (y.iloc[i, 0] - y.iloc[i, 1])
        binary_correct += np.equal(margin_prediction > 0, y.iloc[i, 0] > y.iloc[i, 1])
        sae_t += abs(t_error)
        sae_h += abs(h_error)
        sae_a += abs(a_error)
        rss_t += t_error**2
        rss_h += h_error**2
        rss_a += a_error**2

    home_margins = y.iloc[:,0] - y.iloc[:,1]
    tsst = np.sum((home_margins[train_end:] - np.mean(home_margins[train_end:]))**2)
    tssh = np.sum((y.iloc[train_end:,0] - np.mean(y.iloc[train_end:,0]))**2)
    tssa = np.sum((y.iloc[train_end:,1] - np.mean(y.iloc[train_end:,1]))**2)
    test_games = indicators.shape[0] - train_end

    r2_t, r2_h, r2_a = rss_t/tsst, rss_h/tssh, rss_a/tssa
    mae_t, mae_h, mae_a = sae_t/test_games, sae_h/test_games, sae_a/test_games
    binary_accuracy = binary_correct/test_games
    return r2_t, r2_h, r2_a, mae_t, mae_h, mae_a, binary_accuracy, z_joint, z_games


# Returns R2, MAE, % Correct, margin variables, and games used
def calculate_team_average_margins(y, indicators, train_end, z_vec):
    # Initializing
    z_margin = np.zeros(len(z_vec))
    z_games = np.zeros(len(z_vec))

    # Train process, calculating average point margin for teams
    for i in range(train_end):
        z_margin[indicators[i,0]] -= y[i]
        z_games[indicators[i,0]] += 1

        z_margin[indicators[i,1]] += y[i]
        z_games[indicators[i,1]] += 1
    z_margin = z_margin / z_games

    # Test process, using average point margins for predictions with simple model assumptions
    rss = 0
    sae = 0
    binary_correct = 0
    for i in range(train_end, indicators.shape[0]):
        prediction = -1.0 * z_margin[indicators[i, 0]] + 1.0 * z_margin[indicators[i, 1]]  # Calculating margin relative to home team
        binary_correct += int(np.equal(y[i] > 0, prediction > 0))
        error = prediction - y[i]
        sae += abs(error)
        rss += error**2

    tss = np.sum((y[train_end:] - np.mean(y[train_end:]))**2)
    test_games = indicators.shape[0] - train_end

    # Final calculations to return
    r_squared = 1 - rss / tss  # R-Squared calculation using residual sum of squares from predictions and total sum of squares from mean of training data
    mae = sae / test_games  # Sum of Absolute Error / test games is Mean Absolute Error
    binary_accuracy = binary_correct / test_games

    return r_squared, mae, binary_accuracy, z_margin, z_games


# Function for initalizing the priors of latent variables
# Uses argument means and variances if they exist and are appropriate length,
# otherwise uses default values generated from load data
def initialize_priors(arg_p_means, arg_p_vars, def_p_means, def_p_vars, num_latent):
    if arg_p_means is not None and len(arg_p_means) == num_latent:
        joint_p_means = def_p_means
    else:
        joint_p_means = np.concatenate([def_p_means, def_p_means])
        np.random.shuffle(joint_p_means)

    if arg_p_vars is not None and len(arg_p_vars) == num_latent:
        joint_p_vars = def_p_vars
    else:
        joint_p_vars = np.concatenate([def_p_vars, def_p_vars])
        np.random.shuffle(joint_p_vars)

    return joint_p_means, joint_p_vars


def write_accuracy_results(first_words, season, accuracy_dictionary, output_file):
    for first_word in first_words:
        try:
            write_string = "%s,%s," % (first_word, season)
            write_string += ",".join(["%.5f" % e for e in accuracy_dictionary[first_word]])
            output_file.write(write_string + "\n")
        except KeyError:  # Occurs if not evaluating all types of models on data
            pass


# The main function for running models on a season of data
def season_train_test_models(input_filename, accuracy_output_filename, correlation_output_filename, ratings_output_base_filename,
                             model_type, test_size, MAP, arg_prior_means, arg_prior_vars, arg_joint_prior_means, arg_joint_prior_vars,
                             evaluate_dictionary, season, show):
    start = datetime.datetime.now()
    # Should move data loading out of this function to make more accessible at high level for specifying data structure
    print("%s Loading Data..." % datetime.datetime.now())
    season_x, season_joint_x, y_dict, season_indicators, p_means, p_vars, z, latent_team_dictionary, baseline_dict = de.load_data(input_filename=input_filename, spread_col=None, over_under=None, include_rest=False)
    team_list = list(latent_team_dictionary.keys())
    print("...Finished Loading Data %s seconds" % (datetime.datetime.now() - start).total_seconds())

    p_means, p_vars = initialize_priors(arg_prior_means, arg_prior_vars, p_means, p_vars, num_latent=len(team_list))
    joint_p_means, joint_p_vars = initialize_priors(arg_joint_prior_means, arg_joint_prior_vars, p_means, p_vars, num_latent=len(team_list)*2)

    result_file = open(accuracy_output_filename, 'w')
    result_file.write("Model,Season,Accuracy\n")

    corr_file = open(correlation_output_filename, 'w')
    corr_file.write("Season,Correlation,p-value\n")

    seasons_margin_rating_df = initialize_team_rating_df(latent_team_dictionary, z)


    model_string = "%s-%s" % (model_type, "MLE" if not MAP else "MAP")
    first_words = ["Margin-MLE", "Margin-MLE-MAE", "Margin-MLE-Binary",
                   "Margin-Baseline", "Margin-Baseline-Binary","Margin-Baseline-MAE", "Margin-MLE-Train",
                   "Binary-EloL", "Binary-EloH", "Binary-EloD",
                   "Joint-AwayR2", "Joint-HomeR2","Joint-MLER2",
                   "Joint-AwayMAE", "Joint-HomeMAE", "Joint-MLEMAE",
                   "Joint-Baseline-R2", "Joint-Baseline-R2Home", "Joint-Baseline-R2Away",
                   "Joint-Baseline-MAE", "Joint-Baseline-MAEHome", "Joint-Baseline-MAEAway",
                   "Joint-Baseline-Binary"]

    accuracy_dictionary = dict()
    for w in first_words:
        accuracy_dictionary[w] = []

    season_margin_y = np.array(y_dict["Margin"]).reshape(-1, 1)
    season_binary_y = np.array(y_dict["Binary"]).reshape(-1, 1)
    season_joint_y = y_dict["Joint"]
    season_games = season_x.shape[0]
    season_errors = np.array([])
    season_variances = np.array([])

    train_end = int((1 - test_size) * season_games)
    test_end = season_games - 1

    if evaluate_dictionary["Margin-Baseline"]:
        print("%s Training/Testing Baseline..." % datetime.datetime.now())
        margin_baseline, margin_mae, margin_baseline_binary, z_marg, z_games = calculate_team_average_margins(y=season_margin_y, indicators=season_indicators, train_end=train_end, z_vec=z)

        accuracy_dictionary["Margin-Baseline"].append(margin_baseline)
        accuracy_dictionary["Margin-Baseline-Binary"].append(margin_baseline_binary)
        accuracy_dictionary["Margin-Baseline-MAE"].append(margin_mae)
        baseline_rating_df = generate_team_rating_df(latent_team_dictionary, z_marg, show=False)
        games_played_df = generate_team_rating_df(latent_team_dictionary, z_games, show=False)

    if evaluate_dictionary["Joint-Baseline"]:
        print("%s Training/Testing Baseline..." % datetime.datetime.now())
        joint_baseline_r2t, joint_baseline_r2h, joint_baseline_r2a, joint_baseline_maet, joint_baseline_maeh, joint_baseline_maea, joint_baseline_binary, z_joint, z_games = calculate_team_average_joint(
            y=season_joint_y, indicators=season_indicators, train_end=train_end, z_vec=z)

        accuracy_dictionary["Joint-Baseline-R2"].append(joint_baseline_r2t)
        accuracy_dictionary["Joint-Baseline-R2Home"].append(joint_baseline_r2h)
        accuracy_dictionary["Joint-Baseline-R2Away"].append(joint_baseline_r2a)
        accuracy_dictionary["Joint-Baseline-MAE"].append(joint_baseline_maet)
        accuracy_dictionary["Joint-Baseline-MAEHome"].append(joint_baseline_maeh)
        accuracy_dictionary["Joint-Baseline-MAEAway"].append(joint_baseline_maea)
        baseline_rating_df = generate_team_rating_df(latent_team_dictionary, z_marg, show=False)
        games_played_df = generate_team_rating_df(latent_team_dictionary, z_games, show=False)

    if evaluate_dictionary["Elo"]:
        print("%s Training Elo..." % datetime.datetime.now())
        elo_zl = fit_elo_model(x=season_x.iloc[0:train_end, :], y=season_binary_y[0:train_end],
                               indicators=season_indicators[0:train_end, :],
                               p_means=p_means, mode="Low", k_step_low=1, k_step_high=4, show=False)
        elo_zh = fit_elo_model(x=season_x.iloc[0:train_end, :], y=season_binary_y[0:train_end],
                               indicators=season_indicators[0:train_end, :],
                               p_means=p_means, mode="High", k_step_low=1, k_step_high=4, show=False)
        elo_zd = fit_elo_model(x=season_x.iloc[0:train_end, :], y=season_binary_y[0:train_end],
                               indicators=season_indicators[0:train_end, :],
                               p_means=p_means, mode="Decay", k_step_low=1, k_step_high=4, show=False)

        print("%s Testing Elo..." % datetime.datetime.now())
        accuracy_dictionary["Binary-EloL"].append(elo_test_accuracy(elo_zl, season_indicators[train_end:test_end, :], season_x.iloc[train_end:test_end, :],season_binary_y[train_end:test_end]))
        accuracy_dictionary["Binary-EloH"].append(elo_test_accuracy(elo_zh, season_indicators[train_end:test_end, :], season_x.iloc[train_end:test_end, :],season_binary_y[train_end:test_end]))
        accuracy_dictionary["Binary-EloD"].append(elo_test_accuracy(elo_zd, season_indicators[train_end:test_end, :], season_x.iloc[train_end:test_end, :],season_binary_y[train_end:test_end]))

    if evaluate_dictionary["Margin"]:
        season_z, margin_train, season_lm, season_z_std = m.fit_margin_model(season_x.iloc[0:train_end, :], season_margin_y[0:train_end], season_indicators[0:train_end, :],
                                     p_means, p_vars, MAP=MAP, show=False)
        test_accuracy, test_mae, test_errors = m.calculate_test_accuracy(season_z, season_lm, season_indicators[train_end:test_end, :], season_x.iloc[train_end:test_end, :], season_margin_y[train_end:test_end])
        game_team_variances = m.calculate_team_variances(season_z_std, season_indicators[train_end:test_end, :])
        season_errors = np.concatenate([season_errors, test_errors])
        season_variances = np.concatenate([season_variances, game_team_variances])
        b_test_accuracy = m.calculate_binary_test_accuracy(season_z, season_lm, season_indicators[train_end:test_end, :], season_x.iloc[train_end:test_end, :], season_margin_y[train_end:test_end])
        accuracy_dictionary["Margin-MLE-Train"].append(margin_train)
        accuracy_dictionary["Margin-MLE"].append(test_accuracy)
        accuracy_dictionary["Margin-MLE-MAE"].append(test_mae)
        accuracy_dictionary["Margin-MLE-Binary"].append(b_test_accuracy)

    if evaluate_dictionary["Joint"]:
        season_joint_z, joint_train, season_lm_a, season_lm_h, season_joint_z_std = m.fit_joint_model(
            season_joint_x.iloc[0:train_end, :], season_joint_y.iloc[0:train_end, :], season_indicators[0:train_end, :],
            joint_p_means, joint_p_vars, MAP=MAP, show=False, a_cols=AWAY_SCORE_COLUMNS, h_cols=HOME_SCORE_COLUMNS, int_a=9.9, int_h=10.8)
        # test accuracy
        input_matrix = de.replace_design_joint_latent(joint_design_matrix=season_joint_x.iloc[train_end:test_end, :],
                                                      indicators=season_indicators[train_end:test_end, :],
                                                      jz=season_joint_z).copy()
        joint_test_accuracy, joint_away, joint_home, joint_test_accuracy_mae, joint_away_mae, joint_home_mae = m.calculate_joint_accuracy(X=input_matrix,
                                                                               y=season_joint_y.iloc[train_end:test_end, :],
                                                                               lm_a=season_lm_a, lm_h=season_lm_h,
                                                                               a_cols=AWAY_SCORE_COLUMNS,
                                                                               h_cols=HOME_SCORE_COLUMNS)

        accuracy_dictionary["Joint-AwayR2"].append(joint_away)
        accuracy_dictionary["Joint-HomeR2"].append(joint_home)
        accuracy_dictionary["Joint-MLER2"].append(joint_test_accuracy)
        accuracy_dictionary["Joint-AwayMAE"].append(joint_away_mae)
        accuracy_dictionary["Joint-HomeMAE"].append(joint_home_mae)
        accuracy_dictionary["Joint-MLEMAE"].append(joint_test_accuracy_mae)


    # RETRAINING ON FULL DATASET SO REPORTED RATINGS REFLECT THE FULL SEASON
    season_z, margin_train, season_lm, season_z_std = m.fit_margin_model(season_x, season_margin_y, season_indicators,
                                 p_means, p_vars, MAP=MAP, show=False)
    season_joint_z, joint_train, season_lm_a, season_lm_h, season_joint_z_std = m.fit_joint_model(
        season_joint_x, season_joint_y, season_indicators,
        joint_p_means, joint_p_vars, MAP=MAP, show=False, a_cols=AWAY_SCORE_COLUMNS, h_cols=HOME_SCORE_COLUMNS, int_a=9.9, int_h=10.8)

    z_marg, z_games = calculate_team_average_margins_ro(y=season_margin_y, indicators=season_indicators, train_end=season_indicators.shape[0], z_vec=z)

    baseline_rating_df = generate_team_rating_df(latent_team_dictionary, z_marg, show=False)
    games_played_df = generate_team_rating_df(latent_team_dictionary, z_games, show=False)
    # Not appending, just returning the last dataframe of ratings
    joint_ratings = m.generate_joint_rating_df(latent_team_dictionary, season_joint_z, season_joint_z_std, show=False)

    margin_rating_df = generate_team_rating_df(latent_team_dictionary, season_z, show=False)
    margin_std_df = generate_team_rating_df(latent_team_dictionary, season_z_std, show=False)

    print("%s Writing..." % datetime.datetime.now())
    margin_rating_df.to_csv(ratings_output_base_filename+"MarginRatings.csv", index=True)
    margin_std_df.to_csv(ratings_output_base_filename+"MarginStd.csv", index=True)
    baseline_rating_df.to_csv(ratings_output_base_filename+"BaselineRatings.csv", index=True)
    joint_ratings.to_csv(ratings_output_base_filename+"JointRatings.csv", index=True)
    games_played_df.to_csv(ratings_output_base_filename+"GamesPlayed.csv", index=True)

    write_accuracy_results(first_words, season, accuracy_dictionary, output_file=result_file)

    # season_errors are the squared errors of predictions
    # season_variances are the sum of variance measures for each teams rating
    # Thus a positive correlation indicates more prediction error when high uncertainty with teams
    #  and a negative correlation indicates less prediction error when high uncertainty with teams
    corr_test = ss.pearsonr(x=season_errors, y=season_variances)
    corr_file.write("%s,%.4f,%.4f\n" % (season, corr_test[0], corr_test[1]))

    print("%s Done!" % datetime.datetime.now())

if __name__ == "__main__":
    # Arguments to function here, defaults for lacrosse presented
    i_filename = "LaxPower_Formatted.csv"
    accuracy_o_filename = "Lacrosse-AccuracyOutput.csv"
    correlation_o_filename = "Lacrosse-VarianceAccuracyCorrelations.csv"
    ratings_o_filename = "Lax"
    season = 2016

    m_type = "Margin"
    prop_test = 0.2
    MAP_model = False
    prior_means, prior_vars = [], []
    joint_prior_means, joint_prior_vars = [], []
    eval_dictionary = {"Margin": True,
                       "Margin-Baseline": True,
                       "Elo": True,
                       "Joint": True,
                       "Joint-Baseline": True}

    show = True
    season_train_test_models(input_filename=i_filename,
                             accuracy_output_filename=accuracy_o_filename,
                             correlation_output_filename=correlation_o_filename,
                             ratings_output_base_filename=ratings_o_filename,
                             model_type=m_type, test_size=prop_test, MAP=MAP_model,
                             arg_prior_means=prior_means, arg_prior_vars=prior_vars,
                             arg_joint_prior_means=joint_prior_means, arg_joint_prior_vars=joint_prior_vars,
                             evaluate_dictionary=eval_dictionary, season=season, show=show)
