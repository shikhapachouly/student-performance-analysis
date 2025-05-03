import pandas as pd
import numpy as np

def clean_and_process_data(csv_file):
    """
    Cleans the data by removing '%', replaces outliers (values less than 35 or equal to 100)
    with the mean of non-outliers, and writes the cleaned data to a new CSV file.
    Outlier replacement information is printed to the console.

    Args:
        csv_file (str): The path to the input CSV file.
    """

    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(csv_file)

    # Get the column name
    column_name = df.columns[0]

    # Remove the '%' character from the column
    df[column_name] = df[column_name].astype(str).str.replace('%', '', regex=False)

    # Convert the column to numeric
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')

    # Identify outliers (values less than 35 or equal to 100)
    outlier_mask = (df[column_name] < 35) | (df[column_name] == 100)

    # Calculate the mean of the non-outliers (values >= 35 and < 100)
    non_outlier_values = df.loc[~outlier_mask, column_name]
    if not non_outlier_values.empty:
        mean_value = non_outlier_values.mean()
    else:
        print("Warning: No non-outlier values found (between 35 and 99). Cannot calculate mean for replacement.")
        mean_value = np.nan  # Or handle this case as needed

    # Replace outliers with the mean and log the information
    for index, row in df.iterrows():
        original_value = row[column_name]  # Store the original value
        if row[column_name] < 35 or row[column_name] == 100:
            df.loc[index, column_name] = mean_value
            print(f"Replaced outlier: Old value = {original_value:.2f}, New value = {mean_value:.2f}")

    # Round the numeric values to 2 decimal places
    df[column_name] = df[column_name].round(2)

    # Write the cleaned DataFrame to a new CSV file
    df.to_csv('./columnwise_data_cleaning/10th_percent_clean_data.csv', index=False)

    print("Cleaned data written to clean_data.csv")

# Example usage:
clean_and_process_data('../../dataset/columnwise_data_cleaning/raw_data/10th_percent.csv') # Replace 'your_input_file.csv' with the actual path to your CSV file