import numpy as np
import pandas as pd
import data_engineering as de
from main import initialize_team_rating_df
from main import fit_elo_model
from main import elo_test_accuracy
import computation_functions as cf
import main as m
import datetime
import scipy.stats as ss

# Global variables (should be replaced with arguments)
AWAY_SCORE_COLUMNS = ["AwayOffense", "HomeDefense"]
HOME_SCORE_COLUMNS = ["HomeOffense", "AwayDefense"]
MAXIMUM_ITERATIONS = 1


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
                             evaluate_dictionary, load_dict, season, show):
    start = datetime.datetime.now()
    # Should move data loading out of this function to make more accessible at high level for specifying data structure
    print("%s Loading Data..." % datetime.datetime.now())
    season_x, season_joint_x, y_dict, season_indicators, sp_means, sp_vars, z, latent_team_dictionary, baseline_dict = \
        de.load_data(input_filename=input_filename,
                     away_points=load_dict["AwayPoints"], home_points=load_dict["HomePoints"],
                     spread_col=load_dict["Spread"], over_under=load_dict["OverUnder"],
                     away_name=load_dict["AwayName"], home_name=load_dict["HomeName"],
                     include_rest=load_dict["RestAvailable"])
    team_list = list(latent_team_dictionary.keys())
    print("...Finished Loading Data %s seconds" % (datetime.datetime.now() - start).total_seconds())

    p_means, p_vars = cf.initialize_priors(arg_prior_means, arg_prior_vars, sp_means, sp_vars,
                                           num_latent=len(team_list), latent_team_dictionary=latent_team_dictionary,
                                           filename=load_dict["PrevMarginRatings"])
    # filename="LaxMarginRatings2016.csv"
    joint_p_means, joint_p_vars = cf.initialize_priors(arg_joint_prior_means, arg_joint_prior_vars, sp_means, sp_vars,
                                                       num_latent=len(team_list)*2,
                                                       latent_team_dictionary=latent_team_dictionary,
                                                       filename=load_dict["PrevJointRatings"])
    # filename="LaxJointRatings2016.csv"

    result_file = open(accuracy_output_filename, 'w')
    result_file.write("Model,Season,Accuracy\n")

    corr_file = open(correlation_output_filename, 'w')
    corr_file.write("Season,Correlation,p-value\n")

    seasons_margin_rating_df = initialize_team_rating_df(latent_team_dictionary, z)

    model_string = "%s-%s" % (model_type, "MLE" if not MAP else "MAP")
    first_words = ["Margin-MLE", "Margin-MLE-MAE", "Margin-MLE-Binary","Margin-Baseline-MAE-Std",
                   "Margin-Baseline", "Margin-Baseline-Binary","Margin-Baseline-MAE", "Margin-MLE-Train","Margin-MLE-MAE-Std",
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
        margin_baseline, margin_mae, std_mae, margin_baseline_binary, z_marg, z_games = cf.calculate_team_average_margins(y=season_margin_y, indicators=season_indicators, train_end=train_end, z_vec=z)

        accuracy_dictionary["Margin-Baseline"].append(margin_baseline)
        accuracy_dictionary["Margin-Baseline-Binary"].append(margin_baseline_binary)
        accuracy_dictionary["Margin-Baseline-MAE"].append(margin_mae)
        accuracy_dictionary["Margin-Baseline-MAE-Std"].append(std_mae)
        baseline_rating_df = cf.generate_team_rating_df(latent_team_dictionary, z_marg, show=False)
        games_played_df = cf.generate_team_rating_df(latent_team_dictionary, z_games, show=False)

    if evaluate_dictionary["Joint-Baseline"]:
        print("%s Training/Testing Baseline..." % datetime.datetime.now())
        joint_baseline_r2t, joint_baseline_r2h, joint_baseline_r2a, joint_baseline_maet, joint_baseline_maeh, joint_baseline_maea, joint_baseline_binary, z_joint, z_games = cf.calculate_team_average_joint(
            y=season_joint_y, indicators=season_indicators, train_end=train_end, z_vec=z)

        accuracy_dictionary["Joint-Baseline-R2"].append(joint_baseline_r2t)
        accuracy_dictionary["Joint-Baseline-R2Home"].append(joint_baseline_r2h)
        accuracy_dictionary["Joint-Baseline-R2Away"].append(joint_baseline_r2a)
        accuracy_dictionary["Joint-Baseline-MAE"].append(joint_baseline_maet)
        accuracy_dictionary["Joint-Baseline-MAEHome"].append(joint_baseline_maeh)
        accuracy_dictionary["Joint-Baseline-MAEAway"].append(joint_baseline_maea)
        baseline_rating_df = cf.generate_team_rating_df(latent_team_dictionary, z_marg, show=False)
        games_played_df = cf.generate_team_rating_df(latent_team_dictionary, z_games, show=False)

    if evaluate_dictionary["Elo"]:
        print("%s Training Elo..." % datetime.datetime.now())
        elo_zl = fit_elo_model(x=season_x.iloc[0:train_end, :], y=season_binary_y[0:train_end],
                               indicators=season_indicators[0:train_end, :],
                               p_means=p_means, mode="Low", k_step_low=4, k_step_high=32, show=False)
        elo_zh = fit_elo_model(x=season_x.iloc[0:train_end, :], y=season_binary_y[0:train_end],
                               indicators=season_indicators[0:train_end, :],
                               p_means=p_means, mode="High", k_step_low=4, k_step_high=32, show=False)
        elo_zd = fit_elo_model(x=season_x.iloc[0:train_end, :], y=season_binary_y[0:train_end],
                               indicators=season_indicators[0:train_end, :],
                               p_means=p_means, mode="Decay", k_step_low=4, k_step_high=32, show=False)

        print("%s Testing Elo..." % datetime.datetime.now())
        accuracy_dictionary["Binary-EloL"].append(elo_test_accuracy(elo_zl, season_indicators[train_end:test_end, :], season_x.iloc[train_end:test_end, :],season_binary_y[train_end:test_end]))
        accuracy_dictionary["Binary-EloH"].append(elo_test_accuracy(elo_zh, season_indicators[train_end:test_end, :], season_x.iloc[train_end:test_end, :],season_binary_y[train_end:test_end]))
        accuracy_dictionary["Binary-EloD"].append(elo_test_accuracy(elo_zd, season_indicators[train_end:test_end, :], season_x.iloc[train_end:test_end, :],season_binary_y[train_end:test_end]))

    if evaluate_dictionary["Margin"]:
        season_z, margin_train, season_lm, season_z_std = m.fit_margin_model(season_x.iloc[0:train_end, :],
                                                                             season_margin_y[0:train_end], season_indicators[0:train_end, :],
                                     p_means, p_vars, max_iter=MAXIMUM_ITERATIONS, MAP=MAP, show=False)
        test_accuracy, test_mae, test_mae_std, test_errors = m.calculate_test_accuracy(season_z, season_lm, season_indicators[train_end:test_end, :], season_x.iloc[train_end:test_end, :], season_margin_y[train_end:test_end])
        game_team_variances = m.calculate_team_variances(season_z_std, season_indicators[train_end:test_end, :])
        season_errors = np.concatenate([season_errors, test_errors])
        season_variances = np.concatenate([season_variances, game_team_variances])
        b_test_accuracy = m.calculate_binary_test_accuracy(season_z, season_lm, season_indicators[train_end:test_end, :], season_x.iloc[train_end:test_end, :], season_margin_y[train_end:test_end])
        accuracy_dictionary["Margin-MLE-Train"].append(margin_train)
        accuracy_dictionary["Margin-MLE"].append(test_accuracy)
        accuracy_dictionary["Margin-MLE-MAE"].append(test_mae)
        accuracy_dictionary["Margin-MLE-Binary"].append(b_test_accuracy)
        accuracy_dictionary["Margin-MLE-MAE-Std"].append(test_mae_std)
        print(season_lm.intercept_)
        print(season_lm.coef_)

    if evaluate_dictionary["Joint"]:
        season_joint_z, joint_train, season_lm_a, season_lm_h, season_joint_z_std = m.fit_joint_model(
            season_joint_x.iloc[0:train_end, :], season_joint_y.iloc[0:train_end, :], season_indicators[0:train_end, :],
            joint_p_means, joint_p_vars, MAP=MAP, show=False, max_iter=MAXIMUM_ITERATIONS,
            a_cols=AWAY_SCORE_COLUMNS, h_cols=HOME_SCORE_COLUMNS, int_a=9.9, int_h=10.8)

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


    # RETRAINING ON FULL DATASET SO FINAL REPORTED RATINGS REFLECT THE FULL SEASON
    season_z, margin_train, season_lm, season_z_std = m.fit_margin_model(season_x, season_margin_y, season_indicators,
                                 p_means, p_vars, max_iter=MAXIMUM_ITERATIONS, MAP=MAP, show=False)
    print(season_lm.intercept_)
    print(season_lm.coef_)
    season_joint_z, joint_train, season_lm_a, season_lm_h, season_joint_z_std = m.fit_joint_model(
        season_joint_x, season_joint_y, season_indicators,
        joint_p_means, joint_p_vars, MAP=MAP, show=False,
        a_cols=AWAY_SCORE_COLUMNS, h_cols=HOME_SCORE_COLUMNS, max_iter=MAXIMUM_ITERATIONS, int_a=9.9, int_h=10.8)

    z_marg, z_games = cf.calculate_team_average_margins_ro(y=season_margin_y, indicators=season_indicators, train_end=season_indicators.shape[0], z_vec=z)

    baseline_rating_df = cf.generate_team_rating_df(latent_team_dictionary, z_marg, show=False)
    games_played_df = cf.generate_team_rating_df(latent_team_dictionary, z_games, show=False)
    # Not appending, just returning the last dataframe of ratings
    joint_ratings = m.generate_joint_rating_df(latent_team_dictionary, season_joint_z, season_joint_z_std, show=False)

    margin_rating_df = cf.generate_team_rating_df(latent_team_dictionary, season_z, show=False)
    margin_std_df = cf.generate_team_rating_df(latent_team_dictionary, season_z_std, show=False)

    print("%s Writing..." % datetime.datetime.now())
    margin_rating_df.to_csv(ratings_output_base_filename+"MarginRatings%d.csv" % season, index=True)
    margin_std_df.to_csv(ratings_output_base_filename+"MarginStd%d.csv" % season, index=True)
    baseline_rating_df.to_csv(ratings_output_base_filename+"BaselineRatings%d.csv" % season, index=True)
    joint_ratings.to_csv(ratings_output_base_filename+"JointRatings%d.csv" % season, index=True)
    games_played_df.to_csv(ratings_output_base_filename+"GamesPlayed%d.csv" % season, index=True)

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
    season = 2017
    i_filename = "LaxPower_Formatted%s.csv" % season
    accuracy_o_filename = "Lacrosse-AccuracyOutput%sFull.csv" % season
    correlation_o_filename = "Lacrosse-VarianceAccuracyCorrelations%sFull.csv" % season
    ratings_o_filename = "Lax"

    m_type = "Margin"
    prop_test = 0.1
    MAP_model = True
    prior_means, prior_vars = [], []
    joint_prior_means, joint_prior_vars = [], []
    eval_dictionary = {"Margin": True,
                       "Margin-Baseline": True,
                       "Elo": True,
                       "Joint": True,
                       "Joint-Baseline": True}

    load_dictionary = {"AwayPoints":" Away Points",
                       "HomePoints":" Home Points",
                       "Spread":None,
                       "OverUnder":None,
                       "AwayName":"Away Team",
                       "HomeName":" Home Team",
                       "RestAvailable":False,
                       "PrevMarginRatings":None,  # Replace with e.g. "LaxMarginRatings2016.csv"
                       "PrevJointRatings":None}   # Replace with e.g. "LaxJointRatings2016.csv"
    show = True
    season_train_test_models(input_filename=i_filename,
                             accuracy_output_filename=accuracy_o_filename,
                             correlation_output_filename=correlation_o_filename,
                             ratings_output_base_filename=ratings_o_filename,
                             model_type=m_type, test_size=prop_test, MAP=MAP_model,
                             arg_prior_means=prior_means, arg_prior_vars=prior_vars,
                             arg_joint_prior_means=joint_prior_means, arg_joint_prior_vars=joint_prior_vars,
                             evaluate_dictionary=eval_dictionary,
                             load_dict=load_dictionary,
                             season=season, show=show)
