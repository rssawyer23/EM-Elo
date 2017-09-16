import numpy as np
import pandas as pd
import data_engineering as de
from main import initialize_team_rating_df
from main import fit_elo_model
from main import elo_test_accuracy
import datetime
import scipy.stats as ss


def generate_team_rating_df(latent_team_dict, z_vector, show=True):
    sortable_team_dict = dict()
    for team in latent_team_dict.keys():
        sortable_team_dict[team] = z_vector[latent_team_dict[team]]
    team_rating_df = pd.DataFrame.from_dict(sortable_team_dict, orient="index")
    team_rating_df.columns = ["1"]
    if show:
        print(team_rating_df.sort_values(by="1", axis=0, ascending=False))
    return team_rating_df


def calculate_team_average_margins(y, indicators, train_end, z_vec):
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

    # Test process
    rss = 0
    binary_correct = 0
    for i in range(train_end, indicators.shape[0]):
        prediction = -0.5 * z_margin[indicators[i, 0]] + 0.5 * z_margin[indicators[i, 1]]  # Calculating margin relative to home team
        binary_correct += int(np.equal(y[i] > 0, prediction > 0))
        error = prediction - y[i]
        rss += error**2

    tss = np.sum((y[train_end:] - np.mean(y[train_end:]))**2)
    test_games = indicators.shape[0] - train_end

    return 1 - rss/tss, binary_correct/test_games, z_margin


input_filename = "LaxPower_Formatted.csv"
show = True
start = datetime.datetime.now()
print("%s Loading Data..." % datetime.datetime.now())
season_x, y_dict, season_indicators, p_means, p_vars, z, latent_team_dictionary, baseline_dict = de.load_data(input_filename=input_filename, spread_col=None, over_under=None, include_rest=False)
team_list = list(latent_team_dictionary.keys())
print("...Finished Loading Data %s seconds" % (datetime.datetime.now() - start).total_seconds())

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
first_words = ["Margin-MLE", "Margin-MLE-Binary", "Margin-Baseline", "Margin-Baseline-Binary", "Margin-MLE-Train",
               "Binary-EloL", "Binary-EloH", "Binary-EloD", "Joint-Baseline"]


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
margin_baseline, margin_baseline_binary, z_marg = calculate_team_average_margins(y=season_margin_y, indicators=season_indicators, train_end=train_end, z_vec=z)

accuracy_dictionary["Margin-Baseline"].append(margin_baseline)
accuracy_dictionary["Margin-Baseline-Binary"].append(margin_baseline_binary)
eloh_rating_df = generate_team_rating_df(latent_team_dictionary, z_marg, show=True)


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



# margin_rating_df = generate_team_rating_df(latent_team_dictionary, season_z, show=False)
# seasons_margin_rating_df = pd.concat([seasons_margin_rating_df, margin_rating_df], axis=1)
#
# margin_std_df = generate_team_rating_df(latent_team_dictionary, season_z_std, show=False)
# seasons_margin_std_df = pd.concat([seasons_margin_std_df, margin_std_df], axis=1)
#
# elod_rating_df = generate_team_rating_df(latent_team_dictionary, elo_zd, show=False)
# seasons_elod_rating_df = pd.concat([seasons_elod_rating_df, elod_rating_df], axis=1)
#
# elol_rating_df = generate_team_rating_df(latent_team_dictionary, elo_zl, show=True)
# seasons_elol_rating_df = pd.concat([seasons_elol_rating_df, elol_rating_df], axis=1)
#
# eloh_rating_df = generate_team_rating_df(latent_team_dictionary, elo_zh, show=True)
# seasons_eloh_rating_df = pd.concat([seasons_eloh_rating_df, eloh_rating_df], axis=1)
print("%s Writing..." % datetime.datetime.now())
for first_word in first_words:
    write_string = "%s,%s," % (first_word, season)
    write_string += ",".join(["%.5f" % e for e in accuracy_dictionary[first_word]])
    result_file.write(write_string + "\n")

corr_test = ss.pearsonr(x=season_errors, y=season_variances)
corr_file.write("%s,%.4f,%.4f\n" % (season, corr_test[0], corr_test[1]))

print("%s Done!" % datetime.datetime.now())
