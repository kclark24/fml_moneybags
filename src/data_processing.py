import pandas as pd
import numpy as np
from LogisticRegression import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
    if len(parts) == 2:  
        first_name = parts[0]
        last_name = parts[-1] 
        return f"{first_name[0]}. {last_name}"
    
    elif len(parts) > 2:
        first_name = parts[0]
        last_name = ""
        for i in range(len(parts[1:])):
            last_name += " " + parts[i+1]
        return f"{first_name[0]}.{last_name}"

    return full_name 

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


def average_stats(team_stats):

    stats_list = ['S%', 'FOW%', 'G/GP', 'A/GP', '+/-/GP', 'PIM/GP', 'EVG/GP', 'EVP/GP', 'PPG/GP', 'PPP/GP', 'SHG/GP', 'SHP/GP', 'GWG/GP', 'S/GP']

    average_stats = {stat: 0 for stat in stats_list}

    for stat in stats_list:

        total = 0
        players = 0
        for i in range(len(team_stats[stat])):
            if team_stats[stat].iloc[i] != '--':
                total += team_stats[stat].iloc[i]
                players += 1
        average_stats[stat] = total / players


    return average_stats

def calculate_features(game_info):

    home_roster_stats = game_info['home_roster_stats']
    away_roster_stats = game_info['away_roster_stats']

    home_goalie_stats = game_info['home_goalie_stats']
    away_goalie_stats = game_info['away_goalie_stats']

    home_win = game_info['home_win']

    stats_list = ['S%', 'FOW%', 'G/GP', 'A/GP', '+/-/GP', 'PIM/GP', 'EVG/GP', 'EVP/GP', 'PPG/GP', 'PPP/GP', 'SHG/GP', 'SHP/GP', 'GWG/GP', 'S/GP']

    home_stats = average_stats(home_roster_stats)
    home_stats['Sv%'] = home_goalie_stats['Sv%']
    home_stats['GAA'] = home_goalie_stats['GAA']

    away_stats = average_stats(away_roster_stats)
    away_stats['Sv%'] = away_goalie_stats['Sv%']
    away_stats['GAA'] = away_goalie_stats['GAA']

    return [home_stats, away_stats, home_win] # here, X is home stats and away stats, y is home_win

        
        
def prepare_data(game_data):
    expected_feature_length = 32
    features = []
    labels = []
    
    for index, game in game_data.iterrows():
        game_info = get_game_info(game)
        feature_set = calculate_features(game_info)
        
        home_features = list(feature_set[0].values())
        away_features = list(feature_set[1].values())
        combined_features = home_features + away_features

        # Check if all feature vectors are of the same length
        if len(combined_features) != expected_feature_length:  # Define expected_feature_length based on your model
            print(f"Error in game index {index}: Feature length mismatch.")
            continue  # Skip this game or handle the error as needed

        features.append(combined_features)
        labels.append(feature_set[2])  # home_win is at index 2
        
    return np.array(features), np.array(labels)




player_data = normalize_data(pd.read_excel('../Stats/2022-23 Player Stats.xlsx'))
player_data['Player'] = player_data['Player'].apply(abbreviate_name)
player_data.set_index('Player', inplace=True)

goalie_data = pd.read_excel('../Stats/2022-23 Goalie Stats.xlsx')
goalie_data['Player'] = goalie_data['Player'].apply(abbreviate_name)
goalie_data.set_index('Player', inplace=True)

game_data = pd.read_csv('../src/Game_data/2022-2023_game_data.csv')

X, y = prepare_data(game_data) # get our data to train on


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(learning_rate=0.01, num_iterations=1000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
    

