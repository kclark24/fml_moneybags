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

def get_roster_data(roster, team_name):
    # Split roster string into individual player names
    players = [player.strip() for player in roster.split(",")]

    # Fetch data for each player on the specified team
    filtered_data = []
    for player in players:
        if (player, team_name) in player_data.index:
            player_stats = player_data.loc[(player, team_name)]
            filtered_data.append(player_stats)
    return pd.DataFrame(filtered_data)


def get_goalie_data(goalie, team_name):
    # Fetch goalie stats using both name and team to ensure correct identification
    if (goalie, team_name) in goalie_data.index:
        goalie_stats = goalie_data.loc[(goalie, team_name)]
        return goalie_stats
    else:
        return None  # or handle as you see fit if no matching goalie is found


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
    num_games = len(game_data)
    num_features_per_game = 32  # Adjust this based on the actual number of features you extract
    
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
        if len(combined_features) == num_features_per_game:
            try:
                features[index] = combined_features
                labels[index] = feature_set[2]  # home_win
            except:
                print(combined_features)
        else:
            print(f"Feature length mismatch in game index {index}")

    return features, labels




def change_team_names(team_name):
    team_name_changes = {
    'TBL': 'TB',  # Tampa Bay Lightning
    'SJS': 'SJ',  # San Jose Sharks
    'NJD': 'NJ',  # New Jersey Devils
    'LAK': 'LA'   # Los Angeles Kings
    }

    if team_name in team_name_changes.keys():
        return team_name_changes[team_name]
    else:
        return team_name


player_data = normalize_data(pd.read_excel('../Stats/2022-23 Player Stats.xlsx'))
player_data['Team'] = player_data['Team'].apply(change_team_names)
player_data['Player'] = player_data['Player'].apply(abbreviate_name)
player_data.set_index(['Player', 'Team'], inplace=True)

goalie_data = pd.read_excel('../Stats/2022-23 Goalie Stats.xlsx')
goalie_data['Team'] = goalie_data['Team'].apply(change_team_names)
goalie_data['Player'] = goalie_data['Player'].apply(abbreviate_name)
goalie_data.set_index(['Player', 'Team'], inplace=True)

game_data = pd.read_csv('../src/Game_data/2022-2023_game_data.csv')




game = game_data.iloc[20]
print(get_game_info(game))




#in player data, need to change team TBL to TB, SJS to SJ, NJD to NJ, LAK to LA



X, y = prepare_data(game_data) # get our data to train on


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(learning_rate=0.01, num_iterations=1000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
    

