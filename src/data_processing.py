import pandas as pd
import numpy as np
from LogisticRegression import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from RandomForest import RandomForest


# STATS_LIST = ['S%', 'FOW%', 'G/GP', 'A/GP', '+/-/GP', 'PIM/GP', 'EVG/GP', 'EVP/GP', 'PPG/GP', 'PPP/GP', 'SHG/GP', 'SHP/GP', 'GWG/GP', 'S/GP']
#STATS_LIST = ['G/GP', 'A/GP', '+/-/GP', 'PIM/GP', 'EVG/GP', 'EVP/GP', 'PPG/GP', 'PPP/GP', 'S/GP']
STATS_LIST = ['G','A','P','P/GP','+/-','FOW%','PIM']
NUM_FEATURES = len(STATS_LIST * 2) + 4  # Adjust this based on the actual number of features you extract

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

def change_team_names(team_name):
    team_name_changes = {
        'TBL': 'TB',  # Tampa Bay Lightning
        'SJS': 'SJ',  # San Jose Sharks
        'NJD': 'NJ',  # New Jersey Devils
        'LAK': 'LA'   # Los Angeles Kings
    }
    # Split team names and change them accordingly
    teams = team_name.split(',')
    return ','.join([team_name_changes.get(team, team) for team in teams]) 

def get_roster_data(roster, team_name):
    players = [player.strip() for player in roster.split(",")]
    filtered_data = []
    for player in players:
        # Check for player records that match any of the team names
        player_records = player_data.loc[player_data.index.get_level_values('Player') == player]
        for idx, player_stats in player_records.iterrows():
            if team_name in idx[1].split(','):  # idx[1] should be the team part of the index
                filtered_data.append(player_stats)
    return pd.DataFrame(filtered_data)

def get_goalie_data(goalie, team_name):
    goalie_records = goalie_data.loc[goalie_data.index.get_level_values('Player') == goalie]
    for idx, goalie_stats in goalie_records.iterrows():
        if team_name in idx[1].split(','):
            if goalie_stats['Sv%'] == '--':
                goalie_stats['Sv%'] = np.float64(0.904)
                goalie_stats['GAA'] = np.float64(2.97)

            return goalie_stats
    return None  # Return None if no matching goalie is found

def get_game_info(game):
    home_team = game['Home Team']
    away_team = game['Away Team']
    
    # Get roster stats including team names to avoid duplicates
    home_roster_stats = get_roster_data(game['Home Roster'], home_team)
    home_goalie_stats = get_goalie_data(game['Home Goalie'], home_team)

    away_roster_stats = get_roster_data(game['Away Roster'], away_team)
    away_goalie_stats = get_goalie_data(game['Away Goalie'], away_team)

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
    average_stats = {stat: 0 for stat in STATS_LIST}

    for stat in STATS_LIST:

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

    home_stats = average_stats(home_roster_stats)
    home_stats['Sv%'] = home_goalie_stats['Sv%']
    home_stats['GAA'] = home_goalie_stats['GAA']

    away_stats = average_stats(away_roster_stats)
    away_stats['Sv%'] = away_goalie_stats['Sv%']
    away_stats['GAA'] = away_goalie_stats['GAA']

    return [home_stats, away_stats, home_win] # here, X is home stats and away stats, y is home_win

def prepare_data(game_data):
    num_games = len(game_data)
    num_features_per_game = NUM_FEATURES  # Adjust this based on the actual number of features you extract
    
    # Pre-allocate a NumPy array with shape (num_games, num_features_per_game)
    features = np.zeros((num_games, num_features_per_game))
    labels = np.zeros(num_games)

    for index in range(len(game_data)):
        game_info = get_game_info(game_data.iloc[index])
        feature_set = calculate_features(game_info)
        
        home_features = list(feature_set[0].values())
        away_features = list(feature_set[1].values())
        combined_features = home_features + away_features

        # Ensure combined_features is the correct length
        if len(combined_features) == NUM_FEATURES:
            try:
                features[index] = combined_features
                labels[index] = feature_set[2]  # home_win
            except:
                print(home_features)


        else:
            print(f"Feature length mismatch in game index {index}")

    return features, labels






player_data = normalize_data(pd.read_excel('../Stats/2022-23 Player Stats.xlsx'))
player_data['Team'] = player_data['Team'].apply(change_team_names)
player_data['Player'] = player_data['Player'].apply(abbreviate_name)
player_data.set_index(['Player', 'Team'], inplace=True)

goalie_data = pd.read_excel('../Stats/2022-23 Goalie Stats.xlsx')
goalie_data['Team'] = goalie_data['Team'].apply(change_team_names)
goalie_data['Player'] = goalie_data['Player'].apply(abbreviate_name)
goalie_data.set_index(['Player', 'Team'], inplace=True)

game_data = pd.read_csv('../src/Game_data/2022-2023_game_data.csv')

X, y = prepare_data(game_data)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(learning_rate=0.0001, num_iterations=100000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


clf = RandomForest(n_trees=10,n_feature=NUM_FEATURES)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")


    

