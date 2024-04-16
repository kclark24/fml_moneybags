import pandas as pd

# Load the Excel file
df = pd.read_excel('../Stats/Summary.xlsx')

# Now df is a DataFrame object containing the data from the Excel file
print(df)  # This prints the first five rows of the DataFrame

# def create_df():
#   return 0

# if __name__ == "__main__":
#   create_df()