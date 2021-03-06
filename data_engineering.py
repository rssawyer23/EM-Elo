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


def add_win(row, home_score_col=" Home Points", away_score_col=" Away Points"):
    return int(row[home_score_col] > row[away_score_col])


def add_margin(row, home_score_col=" Home Points", away_score_col=" Away Points"):
    return row[home_score_col] - row[away_score_col]


def add_team_rest(team, date, season_start_default=5):
    first = False  # Check if team not in dictionary
    if team not in last_played.keys():
        last_played[team] = date
        first = True

    if not first:  # Calculate difference between game date and team's last played date
        difference = (date - last_played[team]).days - 1
    else:
        difference = season_start_default
    last_played[team] = date  # Replace last played date with current date
    if difference > season_start_default or difference < 0:
        difference = season_start_default
    return difference


def add_rest(row, date_col=" Date", home_team_col=" Home Team", away_team_col="Away Team"):
    parsed_date = dateparser.parse(row[date_col])
    return pd.Series({"AwayRest": add_team_rest(row[away_team_col], parsed_date),
                      "HomeRest": add_team_rest(row[home_team_col], parsed_date)})


def apply_adds(input_filename="NBAPointSpreadsReduced.csv", output_filename="NBAPointSpreadsAugmented.csv",
               home_score_col=" Home Points", away_score_col=" Away Points",
               home_team_col=" Home Team", away_team_col="Away Team"):
    new_cols = ["Home-Win", "Home-Margin", "Home-Rest", "Away-Rest"]
    data = pd.read_csv(input_filename)
    for col in new_cols:
        data[col] = pd.Series([0]*data.shape[0])
    data["Home-Margin"] = data.apply(func=add_margin, axis=1,
                                    home_score_col=home_score_col, away_score_col=away_score_col)
    data["Home-Win"] = data.apply(func=add_win, axis=1,
                                    home_score_col=home_score_col, away_score_col=away_score_col)
    data[["Away-Rest", "Home-Rest"]] = data.apply(add_rest, axis=1, date_col="Date",
                                 home_team_col=home_team_col, away_team_col=away_team_col)

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


def replace_design_joint_latent(joint_design_matrix, indicators, jz):
    """
    Function for replacing the offense and defense latent variables of a single row of the design matrix with potentially new latent variables in z
    :param joint_design_matrix: transformed data for predictions/numerical calculations (N x d matrix)
    :param indicators: Index numbers of away, home pairs for each example (N x 2 matrix)
    :param jz: latent variable vector, with all offense ratings then all defense ratings. So team i has offense at position i and defense at position i + number_teams
    :return: joint_design matrix: changing the away/home offense/defense latent variables to match potential updates to latent variable vector jz
    """

    num_teams = int(len(jz) / 2)  # Since Offense, Defense rating per team

    for index in joint_design_matrix.index:
        offset = joint_design_matrix.index[0]
        joint_design_matrix.loc[index, "AwayOffense"] = jz[indicators[index - offset, 0]]
        joint_design_matrix.loc[index, "HomeDefense"] = jz[indicators[index - offset, 1] + num_teams]
        joint_design_matrix.loc[index, "HomeOffense"] = jz[indicators[index - offset, 1]]
        joint_design_matrix.loc[index, "AwayDefense"] = jz[indicators[index - offset, 0] + num_teams]

    return joint_design_matrix


def replace_design_latent(design_matrix, indicators, z):
    """
    Function for replacing the latent variables of a single row of the design matrix with potentially new latent variables in z
    :param design_matrix: transformed data for predictions/numerical calculations (N x d matrix)
    :param indicators: Index numbers of away, home pairs for each example (N x 2 matrix)
    :param z: latent variable vector, each element representing the hidden rating of a team with a specific index
    :return: design matrix: changing the away/home latent variables to match potential updates to latent variable vector z
    """

    for index in design_matrix.index:
        offset = design_matrix.index[0]
        design_matrix.loc[index, "AwayRating"] = z[indicators[index - offset, 0]]
        design_matrix.loc[index, "HomeRating"] = z[indicators[index - offset, 1]]

    return design_matrix


def create_indicator_matrix(original_matrix, away_colname="Away Team", home_colname=" Home Team"):
    """
    Function for creating the indicator matrix from a dataset that contains team names
    :param original_matrix: dataframe containing all of the team data
    :param away_colname: string for the column name of the dataframe containing the away team names
    :param home_colname: string for the column name of the dataframe containing the home team names
    :return: the indicator matrix (n x 2), the number of unique teams found, and mapping from team name to index of latent vector
    """
    team_dict = dict()
    indicator_list = []
    counter = 0  # Counts number of teams added to dictionary of team:index mappings
    for index in range(original_matrix.shape[0]):
        new_row = []
        try:
            new_row.append(team_dict[original_matrix.loc[index, away_colname]])
        except KeyError:
            team_dict[original_matrix.loc[index, away_colname]] = counter
            new_row.append(counter)
            counter += 1  # Incrementing so next team has different number
        try:
            new_row.append(team_dict[original_matrix.loc[index, home_colname]])
        except KeyError:
            team_dict[original_matrix.loc[index, home_colname]] = counter
            new_row.append(counter)
            counter += 1
        indicator_list.append(new_row)
    return np.array(indicator_list), counter, team_dict


def load_data(input_filename="NBAPointSpreadsAugmented.csv",
              away_points=" Away Points", home_points=" Home Points",
              spread_col=" Spread (Relative to Away)", over_under=" Over/Under",
              away_name="Away Team", home_name=" Home Team",
              include_rest=True, diff_mult=1.0, weight_col=None):
    """
    Function for returning useful arguments for other functions
    :param input_filename: string for the filename containing the csv of game data
    :param response_type: string that is one of {binary, margin, joint} for the different types of models
    :param away_points: string that is the column name of dataframe containing the away scores
    :param home_points: string that is the column name of dataframe containing the home scores
    :return: 
        design_matrix - the X that has (independent) data columns
        response - the Y chosen by the model type
        indicators - the (n x 2) matrix of team indicators for each game
        p_means - the prior means of the latent team ratings
        p_vars - the prior variances of the latent team ratings
        initial_z - the initial ratings for each team
        team_dict - the mapping from team name to index of the latent variable vector
        baseline - information necessary for calculating a baseline accuracy (vegas spread and over/under)
    """
    data = pd.read_csv(input_filename)

    # Getting indicator matrix
    indicator_matrix, team_number, team_dict = create_indicator_matrix(data, away_name, home_name)

    response_dict = dict()
    response_dict["Binary"] = pd.Series(data.loc[:, home_points] > data.loc[:, away_points], dtype=int)
    response_dict["Margin"] = pd.Series(diff_mult*(data.loc[:, home_points] - data.loc[:, away_points]), dtype=float)
    response_dict["Joint"] = diff_mult*data.loc[:, [home_points, away_points]]

    baseline_dict = dict()
    if spread_col is not None and over_under is not None:
        baseline_dict["Margin"] = data.loc[:, spread_col]
        jbaseline = pd.DataFrame()
        jbaseline[home_points] = data.loc[:, over_under] / 2.0 + data.loc[:, spread_col] / 2.0
        jbaseline[away_points] = data.loc[:, over_under] / 2.0 - data.loc[:, spread_col] / 2.0
        baseline_dict["Joint"] = jbaseline
    # Initializing team ratings
    p_means = np.zeros((team_number,1))
    p_vars = np.ones((team_number,1)) * 350.0 # 13.2 is std of margin from NBA dataset
    initial_z = np.random.normal(loc=0, scale=1, size=(team_number, 1))
    initial_joint_z = np.random.normal(loc=0, scale=1, size=(team_number*2, 1))

    # Creating design matrix
    design_matrix = pd.DataFrame()
    design_matrix["AwayRating"] = np.zeros(indicator_matrix.shape[0])
    design_matrix["HomeRating"] = np.zeros(indicator_matrix.shape[0])
    if include_rest:
        design_matrix["AwayRest"] = data.loc[:,"AwayRest"].copy()
        design_matrix["HomeRest"] = data.loc[:,"HomeRest"].copy()
    design_matrix = replace_design_latent(design_matrix, indicator_matrix, initial_z)

    joint_design_matrix = pd.DataFrame()
    joint_design_matrix["AwayOffense"] = np.zeros(indicator_matrix.shape[0])
    joint_design_matrix["HomeDefense"] = np.zeros(indicator_matrix.shape[0])
    joint_design_matrix["HomeOffense"] = np.zeros(indicator_matrix.shape[0])
    joint_design_matrix["AwayDefense"] = np.zeros(indicator_matrix.shape[0])
    if include_rest:
        joint_design_matrix["AwayRest"] = data.loc[:,"AwayRest"].copy()
        joint_design_matrix["HomeRest"] = data.loc[:,"HomeRest"].copy()
    joint_design_matrix = replace_design_joint_latent(joint_design_matrix, indicator_matrix, initial_joint_z)

    samp_weights = None if weight_col is None else np.array(data.loc[:, weight_col])

    #np.array(response).reshape(-1,1)
    return design_matrix, joint_design_matrix, response_dict, indicator_matrix, p_means, p_vars, initial_z, team_dict, baseline_dict, samp_weights


def parse_seasons(input_filename, season_colname=" Season"):
    """
    Function for returning a dictionary of (season name, boolean array) pairs with boolean array being true for games in that season
    :param input_filename: string for the full data file name
    :param season_colname: string for the column name containing unique season identifiers
    :return season_row_dict: dictionary of season to boolean array of if game is within season
    """
    # Load data
    data = pd.read_csv(input_filename)

    # Getting all unique season identifiers
    season_list = list(data[season_colname].unique())
    season_row_dict = dict()

    # Getting truth of season identifiers with full data rows into label of the dictionary
    for season_label in season_list:
        season_row_dict[season_label] = data[season_colname] == season_label
    return season_row_dict


def load_prior_ratings(latent_team_dictionary, filename, joint):
    ratings_df = pd.read_csv(filename)
    num_teams = len(latent_team_dictionary.keys())
    array_length = num_teams * 2 if joint else num_teams
    prior_array = np.zeros((array_length,1))
    for i,row in ratings_df.iterrows():
        try:
            team_index = latent_team_dictionary[row.iloc[0]]
            prior_array[team_index] = row.iloc[1]
            if joint:
                prior_array[team_index + num_teams] = row.iloc[2]
        except KeyError:
            pass  # Team not found in latent dictionary, defaults to 0
    return prior_array

if __name__ == "__main__":
    input_path_base = "C:/Users/robsc/Documents/Data and Stats/ScrapedData/NBA/NBA%s_FullScores.csv"
    output_path_base = "C:/Users/robsc/Documents/Data and Stats/ScrapedData/NBA/NBA%s_wRest.csv"
    # for season in [1213, 1314, 1415, 1516, 1617]:
    for season in [1718]:
        in_file_path = input_path_base % season
        out_file_path = output_path_base % season
        apply_adds(input_filename=in_file_path, output_filename=out_file_path,
                   home_score_col="Home-FinalScore", away_score_col="Away-FinalScore",
                   home_team_col="Home-Name", away_team_col="Away-Name")
        print("Finished %s Season" % season)


