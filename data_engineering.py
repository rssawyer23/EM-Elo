import pandas as pd
import numpy as np
import dateparser

last_played = dict()
team_dict = {"Chicago":1500,
             "Charlotte":1525,
             "Milwaukee":1550,
             "Miami":1575,
             "Boston":1600,
             "Atlanta":1625,
             "Orlando":1650,
             "Cleveland":1675,
             "Toronto":1500,
             "Indiana":1475,
             "New York":1450,
             "Detroit":1425,
             "Philadelphia":1400,
             "Washington":1375,
             "New Jersey":1350,
             "Brooklyn":1300,
             "Oklahoma City":1500,
             "San Antonio":1525,
             "Portland":1550,
             "Utah":1575,
             "Denver":1600,
             "Phoenix":1625,
             "Dallas":1650,
             "L.A. Lakers":1675,
             "Houston":1500,
             "Memphis":1475,
             "New Orleans":1450,
             "L.A. Clippers":1425,
             "Golden State":1400,
             "Sacramento":1375,
             "Minnesota":1350}


def add_win(row):
    return int(row[" Home Points"] > row[" Away Points"])


def add_margin(row):
    return row[" Home Points"] - row[" Away Points"]


def add_team_rest(team, date, season_start_default=5):
    first = False  # Check if team not in dictionary
    if team not in last_played.keys():
        last_played[team] = date
        first = True

    if not first:  # Calculate difference between game date and team's last played date
        difference = (date - last_played[team]).days
    else:
        difference = season_start_default
    last_played[team] = date  # Replace last played date with current date
    if difference > season_start_default:
        difference = season_start_default
    return difference


def add_rest(row):
    parsed_date = dateparser.parse(row[" Date"])
    return pd.Series({"AwayRest": add_team_rest(row["Away Team"], parsed_date),
                      "HomeRest": add_team_rest(row[" Home Team"], parsed_date)})


def apply_adds(input_filename="NBAPointSpreadsReduced.csv", output_filename="NBAPointSpreadsAugmented.csv"):
    new_cols = ["HomeWin", "HomeMargin", "HomeRest", "AwayRest"]
    data = pd.read_csv(input_filename)
    for col in new_cols:
        data[col] = pd.Series([0]*data.shape[0])
    data["HomeMargin"] = data.apply(func=add_margin, axis=1)
    data["HomeWin"] = data.apply(func=add_win, axis=1)
    data[["AwayRest", "HomeRest"]] = data.apply(add_rest, axis=1)

    data.to_csv(output_filename, index=False)


def replace_teams_ratings(row):
    try:
        home_rating = team_dict[row[" Home Team"]]
    except KeyError:
        print(row[" Home Team"])
        home_rating = 1500
    try:
        away_rating = team_dict[row["Away Team"]]
    except KeyError:
        print(row["Away Team"])
        away_rating = 1500
    return pd.Series({"AwayRating":away_rating,
                      "HomeRating":home_rating})


def create_sample_data(input_filename="NBAPointSpreadsAugmented.csv", output_filename="SampleNBAFile.csv"):
    data = pd.read_csv(input_filename)
    data[["AwayRating", "HomeRating"]] = data.apply(replace_teams_ratings, axis=1)
    full_continuous = data.loc[:, ["AwayRating", "HomeRating", "AwayRest", "HomeRest","HomeWin"]]
    full_continuous.to_csv("FullSampleData.csv", index=False)
    log_reg_continuous = data.loc[:, ["AwayRating", "HomeRating", "HomeWin"]]
    log_reg_continuous.to_csv(output_filename, index=False)


def replace_design_latent(design_matrix, indicators, z):
    """
    Function for replacing the latent variables of a single row of the design matrix with potentially new latent variables in z
    :param design_matrix: transformed data for predictions/numerical calculations (N x d matrix)
    :param indicators: Index numbers of away, home pairs for each example (N x 2 matrix)
    :param z: latent variable vector, each element representing the hidden rating of a team with a specific index
    :return: design matrix: changing the away/home latent variables to match potential updates to latent variable vector z
    """
    for index in range(design_matrix.shape[0]):
        design_matrix.loc[index, "AwayRating"] = z[indicators[index, 0]]
        design_matrix.loc[index, "HomeRating"] = z[indicators[index, 1]]

    return design_matrix
#create_sample_data()
