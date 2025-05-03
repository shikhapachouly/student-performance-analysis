import csv
import re

def clean_percentage(value, mean_value):
    # Remove '%' and ','
    value = re.sub(r'[%,]', '', value).strip()

    # Try converting to float; if not possible, return mean_value
    try:
        value = float(value)
    except ValueError:
        return mean_value

    # Apply rules: if value is 100 or less than 35, return mean_value
    if value == 100 or value < 35:
        return mean_value
    else:
        return value

def process_csv(input_file, output_file):
    # First pass: read all values to calculate the mean of valid numeric entries (>= 35)
    all_values = []
    with open(input_file, 'r') as infile:
        reader = csv.reader(infile)
        # Skip header row if present
        header = next(reader, None)
        for row in reader:
            # Skip empty rows
            if not row:
                continue

            original_value = row[0]
            value = original_value.lower()

            if 'diploma' in value:
                # Extract percentage from diploma entries
                diploma_percent = re.search(r'\d+(\.\d+)?', value)
                if diploma_percent:
                    value = diploma_percent.group()
                else:
                    value = '0'  # If no percentage found, use 0

            if value in ['-', '', 'nil', 'na', 'n/a', 'none', 'no', 'not', 'nope', '_', '.', '..', 'null']:
                value = '0'

            all_values.append(value)

    # Calculate mean of all valid numeric values (with value >= 35)
    numeric_values = []
    for v in all_values:
        try:
            num = float(v)
            if num >= 35:
                numeric_values.append(num)
        except ValueError:
            continue
    mean_value = sum(numeric_values) / len(numeric_values) if numeric_values else 0

    # Second pass: process and clean each row based on the calculated mean
    cleaned_data = []
    # Print the mean value calculated
    print(f"Mean value: {mean_value}")
    with open(input_file, 'r') as infile:
        reader = csv.reader(infile)
        # Skip header row if present
        header = next(reader, None)
        for row in reader:
            # Skip empty rows
            if not row:
                continue

            original_value = row[0]

            value = original_value.lower()

            if 'diploma' in value:
                diploma_percent = re.search(r'\d+(\.\d+)?', value)
                if diploma_percent:
                    value = diploma_percent.group()
                else:
                    value = '0'

            if value in ['Yes','-', '', 'nil', 'na', 'n/a', 'none', 'no', 'not', 'nope', '_', '.', '..', 'null']:
                value = '0'

            cleaned_value = clean_percentage(value, mean_value)
            # print original value ---> cleaned value
            print(f"{original_value} ---> {cleaned_value}")
            # Replace 0 with 'Missing' if appropriate, else use the cleaned value
            final_value = 'Missing' if cleaned_value == 0 else cleaned_value
            cleaned_data.append([final_value])

    # Write the cleaned data to output CSV file
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['Cleaned12thPercentage'])  # Write header
        writer.writerows(cleaned_data)

# Usage
input_file = '../../dataset/columnwise_data_cleaning/raw_data/12thPercentage.csv'
output_file = '../../dataset/columnwise_data_cleaning/cleaned_data/Cleaned12thPercentage.csv'
process_csv(input_file, output_file)