#!/usr/bin/env python3
import pandas as pd


def clean_internship_payment(value):
    """
    Cleans the internship payment status with verbose logging.
    """
    original = str(value)
    if isinstance(value, str) and value.strip().lower() == "paid":
        cleaned = value.strip().lower()
    else:
        cleaned = "free"
    print(f"Cleaning: {original.ljust(20)} -> {cleaned}")
    return cleaned


def main():
    input_file = "../../dataset/columnwise_data_cleaning/raw_data/InternshipPaymentStatusPaidFree.csv"
    output_file = "../../dataset/columnwise_data_cleaning/cleaned_data/InternshipPaymentStatusPaidFree_Cleaned.csv"

    # Configure pandas to show all rows/columns
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 40)

    try:
        # Read CSV without skipping any data
        df = pd.read_csv(
            input_file,
            skip_blank_lines=False,  # Keep blank lines
            keep_default_na=False  # Preserve original NA strings
        )
        print("\nOriginal DataFrame:")
        print(df)

        # Add index column to track original positions
        df['_original_index'] = df.index + 1  # Account for 0-based index
    except Exception as e:
        print(f"Error reading {input_file}: {e}")
        return

    column_name = "InternshipPaymentStatusPaidFree"
    if column_name not in df.columns:
        print(f"Column '{column_name}' not found")
        return

    # Clean data while preserving original index
    df[column_name] = df[column_name].apply(clean_internship_payment)

    print("\nCleaned DataFrame:")
    print(df)

    # Identify missing rows
    original_count = len(df)
    final_count = len(df.dropna(how='all'))  # Count non-blank rows

    # Find completely blank rows
    blank_rows = df[df.isnull().all(axis=1)]['_original_index'].tolist()

    print(f"\nAnalysis:")
    print(f"Total rows read: {original_count}")
    print(f"Valid data rows: {final_count}")
    print(f"Blank rows found: {len(blank_rows)}")
    if blank_rows:
        print(f"Blank row numbers (original): {blank_rows}")

    try:
        # Save with original index tracking
        df.to_csv(output_file, index=False)
        print(f"\nSaved cleaned data to {output_file}")
    except Exception as e:
        print(f"Error saving CSV: {e}")


if __name__ == '__main__':
    main()