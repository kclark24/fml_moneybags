import pandas as pd
import numpy as np
from LogisticRegression import LogisticRegression

# Load the Excel file
def normalize_data(data):
    # Define columns not to normalize
    do_not_normalize = ['Player', 'Season', 'Team', 'S/C', 'Pos', 'GP', 'P/GP', 'S%', 'TOI/GP', 'FOW%']

    # Normalize all other numeric columns by games played
    for column in data.columns:
        if column not in do_not_normalize and data[column].dtype in [np.float64, np.int64]:
            data[column + '/GP'] = data[column] / data['GP']
    
    return data

def abbreviate_name(full_name):
    parts = full_name.split()  
    if len(parts) >= 2:  
        first_name = parts[0]
        last_name = parts[-1] 
        return f"{first_name[0]}. {last_name}"
    return full_name 

player_data = normalize_data(pd.read_excel('../Stats/2022-23 Player Stats.xlsx'))
player_data['Player'] = player_data['Player'].apply(abbreviate_name)
player_data.set_index('Player', inplace=True)

goalie_data = pd.read_excel('../Stats/2022-23 Goalie Stats.xlsx')
goalie_data['Player'] = goalie_data['Player'].apply(abbreviate_name)
goalie_data.set_index('Player', inplace=True)

game_data = pd.read_csv('../src/Game_data/2023-2024_game_data.csv')



def get_roster_data(roster):
    players = [player.strip() for player in roster.split(",")]
    filtered_data = player_data.loc[player_data.index.intersection(players)]
    return filtered_data

def get_goalie_data(goalie):
    return goalie_data.loc[goalie]


def get_game_info(game):
    home_roster_stats = get_roster_data(game['Home Roster'])
    home_goalie_stats = get_goalie_data(game['Home Goalie'])

    away_roster_stats = get_roster_data(game['Away Roster'])
    away_goalie_stats = get_goalie_data(game['Away Goalie'])

    # if the home team wins, return 1, otherwise return 0
    home_win = 1 if game['Home Score'] > game['Away Score'] else 0

    game_info = {
        'home_roster_stats': home_roster_stats,
        'home_goalie_stats': home_goalie_stats,
        'away_roster_stats': away_roster_stats,
        'away_goalie_stats': away_goalie_stats,
        'home_win': home_win
    }

    return game_info


def calculate_features(game_info):

    home_roster_stats = game_info['home_roster_stats']
    away_roster_stats = game_info['away_roster_stats']

    home_goalie_stats = game_info['home_goalie_stats']
    away_goalie_stats = game_info['away_goalie_stats']

    home_win = game_info['home_win']

    stats_list = ['S%', 'FOW%', 'G/GP', 'A/GP', '+/-/GP', 'PIM/GP', 'EVG/GP', 'EVP/GP', 'PPG/GP', 'PPP/GP', 'SHG/GP', 'SHP/GP', 'GWG/GP', 'S/GP']

    home_stats = {stat: 0 for stat in stats_list}
    away_stats = {stat: 0 for stat in stats_list}

    for stat in stats_list:

        for i in range(len(home_roster_stats)):
            player = home_roster_stats.iloc[i]
            print(player)
            home_stats[stat] += player[stat] / len(home_roster_stats)

    # for i in range(len(home_roster_stats)):
    #     player = home_roster_stats.iloc[i]
    #     toi = player['TOI/GP']
    return home_stats

        


    # Want to pass in average per game stats for players and just goalie sv% and gaa (weighed)
    # we have _/GP * TOI / 60 for what we want to sum up and average

    # Features : S%, TOI/GP, FOW%, G/GP, A/GP, +/-/GP, PIM/GP, EVG/GP, EVP/GP, PPG/GP, PPP/GP, SHG/GP, SHP/GP, GWG/GP, S/GP


    # return game_info
        


# Example usage with the first item in the game_data DataFrame
game_data = get_game_info(game_data.iloc[2])
print(calculate_features(game_data)['home_roster_stats'].iloc[0])

# game_roster_stats = get_game_data(game_data.iloc[0]['Home Roster'])
# print(game_roster_stats)

    

