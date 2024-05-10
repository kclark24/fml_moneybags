import pandas as pd
import numpy as np

# Load the Excel file
data = pd.read_excel('../Stats/2022-23 Player Stats.xlsx')

def normalize_data(data):
    # Define columns not to normalize
    do_not_normalize = ['Player', 'Season', 'Team', 'S/C', 'Pos', 'GP', 'P/GP', 'S%', 'TOI/GP', 'FOW%']

    # Normalize all other numeric columns by games played
    for column in data.columns:
        if column not in do_not_normalize and data[column].dtype in [np.float64, np.int64]:
            data[column + '/GP'] = data[column] / data['GP']
    
    return data




# Now df is a DataFrame object containing the data from the Excel file
print(normalize_data(data))  # This prints the first five rows of the DataFrame

# sort data into df's for each team


# def create_df():
#   return 0

# if __name__ == "__main__":
#   create_df()



# train learner on last years stats and test, and then we want to use this seasons data

def normalize_stats(df):
    pass
