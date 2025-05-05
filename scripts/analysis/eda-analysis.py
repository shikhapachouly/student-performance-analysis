import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure plot aesthetics
sns.set(style="whitegrid")
plt.rcParams.update({'figure.max_open_warning': 0})

# Define the path to your CSV file
csv_file_path = r'D:\GitHub\student-performance-analysis\dataset\student-dataset.csv'

# Define the directory to save the plots
output_dir = r'D:\GitHub\student-performance-analysis\eda_plots'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the dataset
df = pd.read_csv(csv_file_path)


# Display the first few rows of the dataset (optional)
# print(df.head())

# Function to plot categorical variables
def plot_categorical(column):
    plt.figure(figsize=(10, 6))
    order = df[column].value_counts().index
    sns.countplot(y=column, data=df, order=order, palette='viridis')
    plt.title(f'Count of {column}')
    plt.xlabel('Count')
    plt.ylabel(column)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{column}_countplot.png'))
    plt.close()


# Function to plot numerical variables
def plot_numerical(column):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column].dropna(), kde=True, bins=30, color='skyblue')
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{column}_histogram.png'))
    plt.close()

    # Boxplot
    plt.figure(figsize=(6, 8))
    sns.boxplot(y=df[column], color='lightgreen')
    plt.title(f'Boxplot of {column}')
    plt.ylabel(column)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{column}_boxplot.png'))
    plt.close()


# Function to plot time-based data
def plot_time(column):
    # Convert the Timestamp column to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Timestamp', y='current_cgpa', data=df, ci=None, marker='o', markersize=2)
    plt.title('Current CGPA Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Current CGPA')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'Current_CGPA_over_Time.png'))
    plt.close()


# Iterate through each column and generate appropriate plots
for column in df.columns:
    unique_values = df[column].nunique()
    print(f'Processing Column: {column} with {unique_values} unique values.')

    if column == 'Timestamp':
        # Special handling for Timestamp
        plot_time(column)
    elif df[column].dtype == 'object':
        if unique_values <= 50:
            plot_categorical(column)
        else:
            print(f"Skipping categorical plot for {column} due to high cardinality ({unique_values} unique values).")
            # Optionally, you can handle high cardinality categorical variables differently
    elif pd.api.types.is_numeric_dtype(df[column]):
        plot_numerical(column)
    else:
        print(f"Column {column} of type {df[column].dtype} is not handled explicitly.")

# Generate a correlation heatmap for numerical variables
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
plt.figure(figsize=(12, 10))
corr = df[numerical_columns].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numerical Features')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Correlation_Heatmap.png'))
plt.close()

# Generate descriptive statistics table and save as CSV
descriptive_stats = df.describe(include='all')
descriptive_stats.to_csv(os.path.join(output_dir, 'Descriptive_Statistics.csv'))

# Generate value counts for categorical variables and save as separate CSV files
categorical_columns = df.select_dtypes(include=['object']).columns

for column in categorical_columns:
    value_counts = df[column].value_counts(dropna=False)
    value_counts_df = value_counts.rename_axis(column).reset_index(name='Counts')
    value_counts_df.to_csv(os.path.join(output_dir, f'{column}_value_counts.csv'), index=False)

print("Exploratory Data Analysis completed. Plots and tables are saved in the 'eda_plots' directory.")