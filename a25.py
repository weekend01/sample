import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
file_path = r"Lung Cancer.csv"  # Replace with your file path
df = pd.read_csv(file_path)

# Display dataset preview
print("Original Dataset Preview:")
print(df.head())

# Step 1: Handle missing values
# Replace missing values with appropriate substitutes
df.fillna(df.median(numeric_only=True), inplace=True)  # Replace missing numerical values with median
df.fillna("Unknown", inplace=True)  # Replace missing categorical values with 'Unknown'

# Step 2: Remove duplicates
df.drop_duplicates(subset=['Name', 'Surname'], keep='first', inplace=True)

# Step 3: Normalize numerical columns using Min-Max Scaling
# Identify numerical columns
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
scaler = MinMaxScaler()

# Apply Min-Max Scaling
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# scaler = MinMaxScaler(): Creates an instance of the Min-Max Scaler.
# scaler.fit_transform(df[numerical_columns]): Fits the scaler to the numerical columns and scales their values to a range of [0, 1].
# df[numerical_columns]: Updates the DataFrame with the scaled values.

# Step 4: Verify and display the cleaned dataset
print("\nCleaned Dataset Preview:")
print(df.head())

# Step 5: Output the shape and any remaining missing values
print("\nShape of Cleaned Dataset:")
print(df.shape)

print("\nRemaining Missing Values:")
print(df.isnull().sum())