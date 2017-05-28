import pandas as pd
import numpy as np
import dateparser

new_cols = ["HomeWin", "HomeMargin", "HomeRest", "AwayRest"]
data = pd.read_csv("NBAPointSpreadsReduced.csv")
for col in new_cols:
    data[col] = pd.Series([0]*data.shape[0])
last_played = dict()


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
    last_played[team] = date  # Replace date with
    if difference > season_start_default:
        difference = season_start_default
    return difference


def add_rest(row):
    parsed_date = dateparser.parse(row[" Date"])
    return pd.Series({"AwayRest": add_team_rest(row["Away Team"], parsed_date),
                      "HomeRest": add_team_rest(row[" Home Team"], parsed_date)})


data["HomeMargin"] = data.apply(func=add_margin, axis=1)
data["HomeWin"] = data.apply(func=add_win, axis=1)
data[["AwayRest", "HomeRest"]] = data.apply(add_rest, axis=1)

data.to_csv("NBAPointSpreadsAugmented.csv", index=False)
