AWAY_TEAM_START = 10
HOME_TEAM_START = 39
AWAY_SCORE_START = 68
HOME_SCORE_START = 71

def reformat_line(line):
    output_string = ""
    last_char = ""
    neutral_site = int(line[7] == 'N')
    away_team = line[AWAY_TEAM_START:HOME_TEAM_START].strip()
    home_team = line[HOME_TEAM_START:AWAY_SCORE_START].strip()
    away_score = line[AWAY_SCORE_START:AWAY_SCORE_START+2].strip()
    home_score = line[HOME_SCORE_START:HOME_SCORE_START+2].strip()
    # Can include neutral site as extra parameter in output string below
    # Currently not in same format as the NBAPointSpreadsAugmented.csv file that the other script reads
    output_string = "%s,%s,%s,%s\n" % (away_team, home_team, away_score, home_score)
    return output_string

input_filename = "LaxPower_Raw.txt"
output_filename = "LaxPower_Formatted.csv"

with open(input_filename, 'r') as input_file:
    with open(output_filename, 'w') as output_file:
        header = "Away Team, Home Team, Away Points, Home Points"  # Doing this weirdly to mimic the NBAPointSpreadsAugmented.csv file
        output_file.write(header+"\n")
        for line in input_file:
            output_file.write(reformat_line(line))