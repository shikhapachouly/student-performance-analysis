import csv
import re  # Import the regular expression module

def standardize_institute_name(institute_name):
    """
    Standardizes institute names to a common format, handling variations.

    Args:
        institute_name: The institute name as it appears in the data.

    Returns:
        The standardized institute name in lowercase.
    """

    institute_name = institute_name.strip().lower()  # Remove extra spaces and lowercase

    # Remove unwanted characters using regular expressions
    institute_name = re.sub(r'[^a-z\s]', '', institute_name)  # Keep only letters and spaces

    # AISSMS variations
    if "aissms" in institute_name:
        if "ioit" in institute_name:
            return "aissms institute of information technology pune"
        else:
            return "aissms college of engineering pune"

    if "ioit" in institute_name:
        return "aissms institute of information technology pune"
    # Genba Sopanrao Moze variations
    elif "genba sopanrao moze" in institute_name or "g s moze" in institute_name or "gsmcoe" in institute_name:
        return "genba sopanrao moze college of engineering balewadi"

    # Matoshri variations
    elif "matoshri" in institute_name or "matoshree" in institute_name\
            or  "matho" in institute_name:
        return "matoshree college of engineering and research centre nashik"

    # if contains - genba sopanrao moze college of engineering balewadi
    # g  s moze college
    # g smoze college of  engineering balewadi
    # g smoze college of engineering balewadi
    # gamba moze college of engineering balewadi pune
    # ganba sopanraw moze college of engineering balewadi
    # gyanba sopanrao moze college of engineering balewadi
    # gyanba sopanrav moze college of engineering balewadi pune
    # moze college
    # moze collage of engineering balewadi pune
    # moze college of engineering
    elif "genba" in institute_name or "g s moze college" in institute_name or "moze" in institute_name or "gsm" in institute_name or "gamba" in institute_name or "balewadi" in institute_name or "gyanba sopanrao moze college of engineering balewadi" in institute_name or "gyanba sopanrav moze college of engineering balewadi pune" in institute_name or "moze college" in institute_name or "moze collage of engineering balewadi pune" in institute_name or "moze college of engineering" in institute_name:
        return "genba sopanrao moze college of engineering balewadi"

    # LNCT variations
    elif "narain" in institute_name or "narayan" in institute_name \
            or "lnct" in institute_name:
        return "lakshmi narayan college of technology and science bhopal"

     # if starts with gs or ge
    elif institute_name.startswith("gs") or institute_name.startswith("ge")\
            or "gsmcoe" in institute_name:
        return "genba sopanrao moze college of engineering balewadi"
    # if contains shiv  or shivaji
    elif "shiv" in institute_name:
        return "aissms college of engineering pune"
    # if contains coe
    elif "coe" in institute_name:
        return "aissms college of engineering pune"
    # if contains phule
    elif "phule" in institute_name\
            or "sppu" in institute_name:
        return "aissms college of engineering pune"
    # if contains pune
    elif "pune" in institute_name:
        return "aissms college of engineering pune"

    # If no match, return the original (after cleaning)
    return "aissms college of engineering pune"

def process_csv(input_csv_filename, output_csv_filename):
    """
    Reads institute names from an input CSV, standardizes them, and writes
    the original and standardized names to an output CSV.

    Args:
        input_csv_filename:  Path to the input CSV file.
        output_csv_filename: Path to the output CSV file.
    """
    try:
        with open(input_csv_filename, 'r', newline='', encoding='utf-8') as infile, \
                open(output_csv_filename, 'w', newline='', encoding='utf-8') as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            # Assuming the first row is a header
            header = next(reader)
            header.append("standardized_institute_name")  # Add a new header (lowercase)
            writer.writerow(header)

            for row in reader:
                institute_name = row[0].strip()  # Assuming institute name is in the first column, and strip again
                standardized_name = standardize_institute_name(institute_name)
                row.append(standardized_name)  # Add the standardized name to the row
                writer.writerow(row)

        print(f"Successfully processed '{input_csv_filename}' and saved to '{output_csv_filename}'")

    except FileNotFoundError:
        print(f"Error: Input file '{input_csv_filename}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


# Example Usage
input_file = "../../dataset/columnwise_data_cleaning/raw_data/instittue_name.csv"  # Replace with your input CSV file name
output_file = "../../dataset/columnwise_data_cleaning/cleaned_data/institute_name_cleaned.csv"  # Replace with your desired output CSV file name

process_csv(input_file, output_file)