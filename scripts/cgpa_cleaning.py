import pandas as pd
import re

def clean_cgpa(value, row_number):
    """
    Clean and standardize the CGPA value with row number tracking.
    """
    if pd.isna(value):
        print(f"Row {row_number}: {value} ---> Missing")
        return "Missing"

    value_str = str(value).strip().lower()

    if "back" in value_str:
        print(f"Row {row_number}: {value} ---> 3.0")
        return 3.0
    if "atkt" in value_str:
        print(f"Row {row_number}: {value} ---> 3.0")
        return 3.0
    if "fail" in value_str:
        print(f"Row {row_number}: {value} ---> 3.0")
        return 3.0

    drop_tokens = [
        "no", "nothing", "dont", "didn't go", "not", "na", "n/a", "-", "--", "null",
        "nil", "atkt", "fail", "not calculated", "no cgpa", "n.a.", "nill", "none",
        "i didn't go", "cred it earn", "direct second year", "current drop year", "pursuing",
        "in final semester", "1st exam", "first sem", "second sem", "third sem", "dsy", "dse"
    ]

    for token in drop_tokens:
        value_str = value_str.lower()
        if (token in value_str and ("dse" in value_str
                                    or "dsy" in value_str
                                    or "direct second year" in value_str)):
            print(f"Row {row_number}: {value} ---> DSE")
            return "DSE"
        elif token in value_str:
            print(f"Row {row_number}: {value} ---> Missing")
            return "Missing"

    value_str = re.sub(r"[^\d\.%]+", " ", value_str)
    match = re.search(r"[\d\.]+", value_str)

    if not match:
        print(f"Row {row_number}: {value} ---> Missing")
        return "Missing"

    try:
        num = float(match.group())
    except Exception:
        print(f"Row {row_number}: {value} ---> Missing")
        return "Missing"

    if "%" in value or num > 10:
        num = num / 10.0

    if num < 1 or num > 10:
        print(f"Row {row_number}: {value} ---> {num}")
        return "Missing"

    print(f"Row {row_number}: {value} ---> {num}")
    return num

def process_csv(input_file, output_file):
    try:
        # Read CSV with robust error handling
        df = pd.read_csv(input_file,
                         engine='python',
                         encoding='utf-8',
                         on_bad_lines='warn',
                         skip_blank_lines=False)
        print(f"Successfully read {len(df)} rows from CSV")
    except Exception as e:
        print("Error reading the CSV file:", e)
        return

    # Track original CSV line numbers
    df['_original_index'] = df.index + 2  # Header is line 1, data starts at line 2

    try:
        # Clean all entries with original line numbers
        df['CleanedCGPA'] = df.apply(
            lambda row: clean_cgpa(row['AggregateCGPATillCurrentSemester'], row['_original_index']),
            axis=1
        )
    except KeyError as e:
        print(f"Column error: {e} - please verify input CSV structure")
        return

    # Calculate mean from valid entries only
    try:
        valid_series = df.loc[~df['CleanedCGPA'].isin(["Missing", "DSE"]), 'CleanedCGPA'].astype(float)
        mean_value = valid_series[(valid_series >= 3) & (valid_series < 10)].mean()
        print(f"Computed mean {mean_value} from {len(valid_series)} valid entries")
    except Exception as e:
        print("Error computing mean value:", e)
        return

    def replace_with_mean(x):
        """Convert all non-valid values to mean"""
        try:
            # Handle both Missing and DSE cases
            if x in ["Missing", "DSE"]:
                return mean_value
            num = float(x)
            # Replace edge values with mean
            if num < 3 or num == 10:
                return mean_value
            return num
        except:
            # Fallback for any unexpected values
            return mean_value

    # Apply final conversions
    df['CleanedCGPA'] = df['CleanedCGPA'].apply(replace_with_mean)

    # Ensure numeric type and handle any remaining issues
    df['CleanedCGPA'] = pd.to_numeric(df['CleanedCGPA'], errors='coerce').fillna(mean_value)

    try:
        # Save final results
        df.drop('_original_index', axis=1).to_csv(output_file, index=False)
        print(f"Successfully wrote {len(df)} rows to {output_file}")
        print(f"Final missing values count: {df['CleanedCGPA'].isna().sum()}")
    except Exception as e:
        print("Error writing CSV:", e)

if __name__ == "__main__":
    input_csv = "./column_data_csv/AggregateCGPATillCurrentSemester.csv"
    output_csv = "./column_data_csv/AggregateCGPATillCurrentSemester_cleaned.csv"
    process_csv(input_csv, output_csv)