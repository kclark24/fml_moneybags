import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
from LogisticRegression import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from RandomForest import RandomForest
from itertools import combinations, chain
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from itertools import combinations, chain


#STATS_LIST = ['S%', 'FOW%', 'G/GP', 'A/GP', '+/-/GP', 'PIM/GP', 'EVG/GP', 'EVP/GP', 'PPG/GP', 'PPP/GP', 'SHG/GP', 'SHP/GP', 'GWG/GP', 'S/GP']
STATS_LIST = ['G','A', '+/-', 'FOW%']


def generate_feature_combinations(stats_list):
    feature_types = ['avg'] # 'max' can also be included
    all_features = []

    # Generate all possible combinations of average and maximum for each stat
    for num_features in range(1, len(stats_list) + 1):
        for combo in combinations(stats_list, num_features):
            for sub_combo in chain.from_iterable(combinations(feature_types, r) for r in range(1, len(feature_types) + 1)):
                feature_set = [f"{ft}_{stat}" for stat in combo for ft in sub_combo]
                all_features.append(feature_set)

    return all_features


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
        if player in player_data.index.get_level_values('Player'):
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
    # Return default stats if no matching goalie is found
    return pd.Series({'Sv%': 0.904, 'GAA': 2.97})


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
    stats = STATS_LIST

    average_stats = {f"avg_{stat}": 0 for stat in STATS_LIST}

    for stat in STATS_LIST:

        total = 0
        players = 0
        for i in range(len(team_stats[stat])):
            if team_stats[stat].iloc[i] != '--':
                total += team_stats[stat].iloc[i]
                players += 1
        average_stats[f"avg_{stat}"] = total / players


    return average_stats


def maximum_stats(team_stats):
    max_stats = {f"max_{stat}": 0 for stat in STATS_LIST}
    for stat in STATS_LIST:

        max_stat = 0

        for i in range(len(team_stats[stat])):
            if team_stats[stat].iloc[i] != '--' and team_stats[stat].iloc[i] > max_stat:
                max_stat = team_stats[stat].iloc[i]
        
        max_stats[f"max_{stat}"]

    return max_stats


def calculate_features(game_info, selected_features):
    home_roster_stats = game_info['home_roster_stats']
    away_roster_stats = game_info['away_roster_stats']

    home_goalie_stats = game_info['home_goalie_stats']
    away_goalie_stats = game_info['away_goalie_stats']

    home_win = game_info['home_win']

    home_avg_stats = average_stats(home_roster_stats)
    home_max_stats = maximum_stats(home_roster_stats)
    home_avg_stats['avg_Sv%'] = home_goalie_stats['Sv%']
    home_avg_stats['avg_GAA'] = home_goalie_stats['GAA']

    away_avg_stats = average_stats(away_roster_stats)
    away_max_stats = maximum_stats(away_roster_stats)
    away_avg_stats['avg_Sv%'] = away_goalie_stats['Sv%']
    away_avg_stats['avg_GAA'] = away_goalie_stats['GAA']

    home_combined_stats = {**home_avg_stats}#,**home_max_stats}
    away_combined_stats = {**away_avg_stats}#,**away_max_stats}

    # Always include Sv% and GAA in the selected features
    extra_features = ['GAA'] # interesting that goalie save percentage no impact

    home_features = {k: v for k, v in home_combined_stats.items() if k in selected_features or k in extra_features}
    away_features = {k: v for k, v in away_combined_stats.items() if k in selected_features or k in extra_features}

    return [home_features, away_features, home_win]


def prepare_data(game_data, selected_features):
    num_games = len(game_data)
    num_features_per_game = len(selected_features) * 2 # Include goalie statistics, so + 4
    
    # Pre-allocate a NumPy array with shape (num_games, num_features_per_game)
    features = np.zeros((num_games, num_features_per_game))
    labels = np.zeros(num_games)

    for index in range(len(game_data)):
        game_info = get_game_info(game_data.iloc[index])
        feature_set = calculate_features(game_info, selected_features)
        
        home_features = list(feature_set[0].values())
        away_features = list(feature_set[1].values())
        combined_features = home_features + away_features

        # Ensure combined_features is the correct length
        if len(combined_features) == num_features_per_game:
            try:
                features[index] = combined_features
                labels[index] = feature_set[2]  # home_win
            except:
                print(home_features)
        else:
            print(len(combined_features))
            print(num_features_per_game)

    return features, labels


def evaluate_feature_combinations(game_data, feature_combinations, output_file):
    best_accuracy = 0
    best_features = None

    with open(output_file, 'w') as file:
        for features in feature_combinations:
            X, y = prepare_data(game_data, features)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LogisticRegression(learning_rate=0.0001, num_iterations=1000)
            model.fit(X_train, y_train)
            certainty, y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            result = f"Features: {features} - Accuracy: {accuracy:.2f}"
            print(result)
            file.write(result + '\n')
            file.flush()  # Ensure the data is written to the file

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_features = features

    return best_features, best_accuracy


# USE TO FIND THE BEST FEATURES

# player_data = normalize_data(pd.read_excel('../Stats/2022-23 Player Stats.xlsx'))
# player_data['Team'] = player_data['Team'].apply(change_team_names)
# player_data['Player'] = player_data['Player'].apply(abbreviate_name)
# player_data.set_index(['Player', 'Team'], inplace=True)

# goalie_data = pd.read_excel('../Stats/2022-23 Goalie Stats.xlsx')
# goalie_data['Team'] = goalie_data['Team'].apply(change_team_names)
# goalie_data['Player'] = goalie_data['Player'].apply(abbreviate_name)
# goalie_data.set_index(['Player', 'Team'], inplace=True)

# game_data = pd.read_csv('../src/Game_data/2023-2024_game_data.csv')

# feature_combinations = generate_feature_combinations(STATS_LIST)
# output_file = 'feature_combinations_results.txt'
# best_features, best_accuracy = evaluate_feature_combinations(game_data, feature_combinations, output_file)

# print(f"Best features: {best_features} - Accuracy: {best_accuracy:.2f}")


selected_features = ['avg_G', 'avg_A', 'avg_+/-', 'avg_FOW%']



player_data = normalize_data(pd.read_excel('../Stats/2022-23 Player Stats.xlsx'))
player_data['Team'] = player_data['Team'].apply(change_team_names)
player_data['Player'] = player_data['Player'].apply(abbreviate_name)
player_data.set_index(['Player', 'Team'], inplace=True)

goalie_data = pd.read_excel('../Stats/2022-23 Goalie Stats.xlsx')
goalie_data['Team'] = goalie_data['Team'].apply(change_team_names)
goalie_data['Player'] = goalie_data['Player'].apply(abbreviate_name)
goalie_data.set_index(['Player', 'Team'], inplace=True)

game_data = pd.read_csv('../src/Game_data/2023-2024_game_data.csv')


X, y = prepare_data(game_data, selected_features)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(learning_rate=0.0001, num_iterations=100000)
model.fit(X_train, y_train)
certainties, y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.10f}")

# best_iterations = None
# best_accuracy = 0
# values = np.arange(100000,1000000,25000)
# for value in values:

#     model = LogisticRegression(learning_rate=1e-5, num_iterations=value)
#     model.fit(X_train, y_train)
#     certainties, y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     if accuracy > best_accuracy:
#         best_accuracy = accuracy
#         best_iterations = value
#     print(f"Value: {value}, Accuracy: {accuracy:.10f}")


bin_20 = []
bin_20_pred = []
bin_30 = []
bin_30_pred = []
bin_40 = []
bin_40_pred = []

for i in range(len(certainties)):
    certainty = certainties[i]
    prediction = y_pred[i]
    correct_pred = y_test[i]
    if 0.3 > certainty or certainty > 0.7:
        bin_20.append(prediction)
        bin_20_pred.append(correct_pred)
    if 0.4 > certainty or certainty > 0.6:
        bin_30.append(prediction)
        bin_30_pred.append(correct_pred)
    if 0.5 > certainty or certainty > 0.5:
        bin_40.append(prediction)
        bin_40_pred.append(correct_pred)




conf_matrix = confusion_matrix(bin_20, bin_20_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for 30% Interval')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

conf_matrix = confusion_matrix(bin_30, bin_30_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for 40% Interval')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

conf_matrix = confusion_matrix(bin_40, bin_40_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for 50% Interval')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


def plot_certainty_accuracy(certainties, y_test, y_pred):
    thresholds = np.arange(0.5, 0.8, 0.01)
    accuracies = []

    for threshold in thresholds:
        high_certainty_indices = np.where((certainties >= threshold) | (certainties <= (1 - threshold)))[0]
        
        high_certainty_predictions = []
        high_certainty_actuals = []

        for index in high_certainty_indices:
            high_certainty_predictions.append(y_pred[index])
            high_certainty_actuals.append(y_test[index])


        if len(high_certainty_predictions) > 0:
            accuracy = accuracy_score(high_certainty_actuals, high_certainty_predictions)
        else:
            accuracy = 0
        accuracies.append(accuracy)


        

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, accuracies)
    plt.title('Model Accuracy vs. Certainty Threshold')
    plt.xlabel('Certainty Threshold')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()

# Define the features and prepare the data (same as before)
# ...

# Fit the logistic regression model and get predictions (same as before)
# ...

# Assuming certainties and y_pred are defined after model prediction
# certainties, y_pred = model.predict(X_test)
plot_certainty_accuracy(certainties, y_test, y_pred)




player_data = normalize_data(pd.read_excel('../Stats/2023-24 Player Stats.xlsx'))
player_data['Team'] = player_data['Team'].apply(change_team_names)
player_data['Player'] = player_data['Player'].apply(abbreviate_name)
player_data.set_index(['Player', 'Team'], inplace=True)

goalie_data = pd.read_excel('../Stats/2023-24 Goalie Stats.xlsx')
goalie_data['Team'] = goalie_data['Team'].apply(change_team_names)
goalie_data['Player'] = goalie_data['Player'].apply(abbreviate_name)
goalie_data.set_index(['Player', 'Team'], inplace=True)


test_game = pd.read_csv('../src/Game_data/2023-2024_playoffs.csv')

cool_X, cool_y = prepare_data(test_game, selected_features)
y_pred = model.predict(cool_X)
print(y_pred)




# clf = RandomForest(n_trees=20,n_feature=len(selected_features) * 2 + 4)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy:.2f}")


# def build_neural_network(input_dim):
#     model = keras.Sequential()
#     model.add(keras.layers.Dense(16, input_dim=input_dim, activation='relu'))
#     model.add(keras.layers.Dense(8, activation='relu'))
#     model.add(keras.layers.Dense(4, activation='relu'))
#     model.add(keras.layers.Dense(1, activation='sigmoid'))
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     return model

# input_dim = X_train.shape[1]
# neural_network = build_neural_network(len(selected_features) * 2 + 4)
# neural_network.fit(X_train, y_train, epochs=70, batch_size=100, verbose=1)
# nn_predictions = neural_network.predict(X_test)
# nn_predictions = (nn_predictions > 0.5).astype(int)
# nn_accuracy = accuracy_score(y_test, nn_predictions)
# print(f"Neural Network Accuracy: {nn_accuracy:.2f}")


    

