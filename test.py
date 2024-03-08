import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('recipe.csv')

# Display the first few rows of the DataFrame
print(df.head(5))
