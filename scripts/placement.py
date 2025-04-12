#!/usr/bin/env python3
import pandas as pd

def map_placement_status(val):
    """
    Map the placement status string to 'Yes' or 'No'.

    Logic:
      1. If the value is missing or a known empty placeholder (like NaN, blank, ".", etc.), return "No".
      2. If the lowercased value contains clear no-indicators such as "not placed", "not yet",
         or if it exactly equals "no", "no.", "nope", etc., then return "No".
      3. Otherwise, return "Yes" (assuming the candidate provided an employer name or "yes").
    """
    # Ensure we are working with a string.
    if not isinstance(val, str):
        return "No"

    # Clean-up: remove extra spaces and convert to lowercase.
    v = val.strip().lower()

    empty_values = {"", "na", "nan", ".", "..", "-", "--"}
    if v in empty_values:
        return "No"

    no_indicators = [
        "not placed", "not yet", "i am not placed", "no placement", "no, i'm not placed"
    ]
    for indicator in no_indicators:
        if indicator in v:
            return "No"

    if v in {"no", "no.", "no,"} or v.startswith("no "):
        return "No"

    if "nope" in v or v == "n9":
        return "No"

    if "yes" in v:
        return "Yes"

    # Assume placement if none of the above negative conditions hold.
    return "Yes"

def main():
    # Define input and output file paths.
    input_file = "./column_data_csv/placement.csv"
    output_file = "./column_data_csv/placement_clean.csv"

    # Read the CSV file.
    # Using skip_blank_lines=False ensures that no lines are skipped, even if they are empty.
    try:
        df = pd.read_csv(input_file, skip_blank_lines=False)
    except Exception as e:
        print(f"Error reading {input_file}: {e}")
        return

    column_name = "PlacementStatusEmployerIfYes"
    if column_name not in df.columns:
        print(f"Column '{column_name}' not found in the CSV file.")
        return

    # Apply transformation while logging old vs. new values to the console.
    def log_mapping(val):
        original = val
        mapped = map_placement_status(val)
        # Log each transformation to the console.
        print(f"Original: {original} ---> Mapped: {mapped}")
        return mapped

    df["PlacementCategory"] = df[column_name].apply(log_mapping)

    # Write the cleaned DataFrame to a new CSV file.
    try:
        df.to_csv(output_file, index=False)
        print(f"\nCleaned placement data saved to {output_file}")
    except Exception as e:
        print(f"Error saving cleaned CSV: {e}")

if __name__ == '__main__':
    main()