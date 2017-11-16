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
    mae = 0
    binary_correct = 0
    for i in range(train_end, indicators.shape[0]):
        prediction = -1.0 * z_margin[indicators[i, 0]] + 1.0 * z_margin[indicators[i, 1]]  # Calculating margin relative to home team
        binary_correct += int(np.equal(y[i] > 0, prediction > 0))
        error = prediction - y[i]
        mae += abs(error)
        rss += error**2

    tss = np.sum((y[train_end:] - np.mean(y[train_end:]))**2)
    test_games = indicators.shape[0] - train_end

    return 1 - rss/tss, mae/(indicators.shape[0]-train_end), binary_correct/test_games, z_margin, z_games


input_filename = "LaxPower_Formatted.csv"
show = True
start = datetime.datetime.now()
print("%s Loading Data..." % datetime.datetime.now())
season_x, season_joint_x, y_dict, season_indicators, p_means, p_vars, z, latent_team_dictionary, baseline_dict = de.load_data(input_filename=input_filename, spread_col=None, over_under=None, include_rest=False)
team_list = list(latent_team_dictionary.keys())
print("...Finished Loading Data %s seconds" % (datetime.datetime.now() - start).total_seconds())

# Should replace these
joint_p_means = np.concatenate([p_means, p_means])
np.random.shuffle(joint_p_means)
joint_p_vars = np.concatenate([p_vars, p_vars])
np.random.shuffle(joint_p_vars)

test_size = 0.2
result_file = open("Lacrosse-AccuracyOutput%.2f.csv" % test_size, 'w')
result_file.write("Model,Season,Accuracy\n")

corr_file = open("Lacrosse-VarianceAccuracyCorrelations.csv", 'w')
corr_file.write("Season,Correlation,p-value\n")

seasons_margin_rating_df = initialize_team_rating_df(latent_team_dictionary, z)
MAP = False
model_type = "Margin"

accuracy_results = dict()
model_string = "%s-%s" % (model_type, "MLE" if not MAP else "MAP")
first_words = ["Margin-MLE", "Margin-MLE-MAE", "Margin-MLE-Binary", "Margin-Baseline", "Margin-Baseline-Binary","Margin-Baseline-MAE", "Margin-MLE-Train",
               "Binary-EloL", "Binary-EloH", "Binary-EloD", "Joint-Away", "Joint-Home","Joint-MLE"]


# From this point on will be much simpler than the NBA-main file since only performing analysis on one season and one train/test split
accuracy_dictionary = dict()
for w in first_words:
    accuracy_dictionary[w] = []

season = 2016
season_margin_y = np.array(y_dict["Margin"]).reshape(-1, 1)
season_binary_y = np.array(y_dict["Binary"]).reshape(-1, 1)
season_joint_y = y_dict["Joint"]
season_games = season_x.shape[0]
season_errors = np.array([])
season_variances = np.array([])

train_end = int((1 - test_size) * season_games)
test_end = season_games - 1


print("%s Training/Testing Baseline..." % datetime.datetime.now())
margin_baseline, margin_mae, margin_baseline_binary, z_marg, z_games = calculate_team_average_margins(y=season_margin_y, indicators=season_indicators, train_end=train_end, z_vec=z)

accuracy_dictionary["Margin-Baseline"].append(margin_baseline)
accuracy_dictionary["Margin-Baseline-Binary"].append(margin_baseline_binary)
accuracy_dictionary["Margin-Baseline-MAE"].append(margin_mae)
baseline_rating_df = generate_team_rating_df(latent_team_dictionary, z_marg, show=False)
games_played_df = generate_team_rating_df(latent_team_dictionary, z_games, show=False)


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

season_joint_z, joint_train, season_lm_a, season_lm_h, season_joint_z_std = m.fit_joint_model(
    season_joint_x.iloc[0:train_end, :], season_joint_y.iloc[0:train_end, :], season_indicators[0:train_end, :],
    joint_p_means, joint_p_vars, MAP=MAP, show=False, a_cols=AWAY_SCORE_COLUMNS, h_cols=HOME_SCORE_COLUMNS, int_a=9.9, int_h=10.8)
# test accuracy
input_matrix = de.replace_design_joint_latent(joint_design_matrix=season_joint_x.iloc[train_end:test_end, :],
                                              indicators=season_indicators[train_end:test_end, :],
                                              jz=season_joint_z).copy()
joint_test_accuracy, joint_away, joint_home = m.calculate_joint_accuracy(X=input_matrix,
                                                                       y=season_joint_y.iloc[train_end:test_end, :],
                                                                       lm_a=season_lm_a, lm_h=season_lm_h,
                                                                       a_cols=AWAY_SCORE_COLUMNS,
                                                                       h_cols=HOME_SCORE_COLUMNS)

accuracy_dictionary["Joint-Away"].append(joint_away)
accuracy_dictionary["Joint-Home"].append(joint_home)
accuracy_dictionary["Joint-MLE"].append(joint_test_accuracy)


# RETRAINING ON FULL DATASET SO RATINGS REFLECT THE FULL SEASON
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
margin_rating_df.to_csv("LaxMarginRatings.csv", index=True)
margin_std_df.to_csv("LaxMarginStd.csv", index=True)
baseline_rating_df.to_csv("LaxBaselineRatings.csv", index=True)
joint_ratings.to_csv("LaxJointRatings.csv", index=True)
games_played_df.to_csv("LaxGamesPlayed.csv", index=True)

for first_word in first_words:
    write_string = "%s,%s," % (first_word, season)
    write_string += ",".join(["%.5f" % e for e in accuracy_dictionary[first_word]])
    result_file.write(write_string + "\n")

# season_errors are the squared errors of predictions
# season_variances are the sum of variance measures for each teams rating
# Thus a positive correlation indicates more prediction error when high uncertainty with teams
#  and a negative correlation indicates less prediction error when high uncertainty with teams
corr_test = ss.pearsonr(x=season_errors, y=season_variances)
corr_file.write("%s,%.4f,%.4f\n" % (season, corr_test[0], corr_test[1]))

print("%s Done!" % datetime.datetime.now())
