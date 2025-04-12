#!/usr/bin/env python3
import re
import pandas as pd


def standardize_branch(branch):
    """
    Standardizes a branch string into a main category.

    New/Updated Rules priority:
    1. Hotel Management: if the branch mentions hotel management.
    2. Cyber Security: if it contains keywords like 'cyber security' or variants.
    3. Artificial Intelligence and Data Science: if AIML or similar phrases are found.
    4. IT Engineering: if the string contains "it engineering" or is "be(it)" (case-insensitive).
    5. Electronics and Telecommunication Engineering: if the branch is exactly one of:
       "entc", "e&tc" (all variants are normalized to lower-case for matching).
    6. Information Technology: if the string contains "information technology" or equals "it".
    7. Computer Engineering: if it contains phrases like "computer", "cs", "cse", "b tech cse",
       or if branch includes "genba sopanrao moze college of engineering" (assumed to be computer-related).
    8. Mechanical Engineering: if it contains "mechanical", "me", or "mech" (be cautious to avoid false matches).
    9. Electrical Engineering: if it contains "electrical".
    10. Electronics and Telecommunication Engineering: if it contains "electronics" or "vlsi".
    11. Chemical Engineering: if it contains "chemical".
    12. Civil Engineering: if it contains "civil" or is exactly "c.e".
    13. Otherwise, the original value is returned.
    """
    if not isinstance(branch, str):
        print(f"{branch} ---> branch not a string")
        return branch

    # Convert to lower-case and remove leading/trailing spaces.
    b = branch.lower().strip()

    # Special numerical or non-descriptive values, if needed, you can add explicit detection.
    # For example, if the value cannot be interpreted as a branch, it will fall back to original value.

    # Rule 1: Hotel Management
    if "hotel management" in b:
        print(f"{branch} ---> Hotel Management")
        return "Hotel Management"

    # Rule 2: Cyber Security
    if re.search(r'\bcyber\s*security\b', b) or "cse-cy" in b:
        print(f"{branch} ---> Cyber Security")
        return "Cyber Security"

    # Rule 3: Artificial Intelligence and Data Science (AIML)
    if re.search(r'\b(aiml|ai[ &-]*ds|artificial intelligence (and|&)? (data science|machine learning))\b', b) \
       or "ai&ml" in b or "ai and ds" in b or "cseaiml" in b or "ai" in b:
        print(f"{branch} ---> Artificial Intelligence and Data Science")
        return "Artificial Intelligence and Data Science"

    # Rule 4: IT Engineering (e.g., "it engineering", "be(it)" etc.)
    if "it engineering" in b or re.fullmatch(r'be\s*\(it\)', b):
        print(f"{branch} ---> Information Technology")
        return "Information Technology"

    # Rule 5: Electronics and Telecommunication Engineering (exact match variants)
    if re.fullmatch(r'(entc|e&tc)', b):
        print(f"{branch} ---> Electronics and Telecommunication Engineering")
        return "Electronics and Telecommunication Engineering"

    # Rule 6: Information Technology
    if "information technology" in b or b == "it":
        print(f"{branch} ---> Information Technology")
        return "Information Technology"

    # Rule 7: Computer Engineering
    if ("computer" in b or b in {"cs", "cse"} or "comp" in b
        or "b tech cse" in b):
        print(f"{branch} ---> Computer Engineering")
        return "Computer Engineering"

    # Additional computer-related college name:
    if "genba sopanrao moze college of engineering" in b:
        print(f"{branch} ---> Computer Engineering")
        return "Computer Engineering"

    # If branch is a common acronym for a computer degree (even if ambiguous), you could add:
    if b == "bca":
        print(f"{branch} ---> Computer Engineering")
        return "Computer Engineering"

    # Rule 8: Mechanical Engineering
    # Note: Using simple containment for "mech" might pick up unwanted matches.
    if "mechanical" in b or b in {"me", "mech"} or "mech sand" in b:
        print(f"{branch} ---> Mechanical Engineering")
        return "Mechanical Engineering"

    # Rule 9: Electrical Engineering
    if "electrical" in b :
        print(f"{branch} ---> Electrical Engineering")
        return "Electrical Engineering"

    # Rule 10: Electronics and Telecommunication Engineering (if not already caught by rule 5)
    if "electronics" in b or "vlsi" in b or "electronic" in b:
        print(f"{branch} ---> Electronics and Telecommunication Engineering")
        return "Electronics and Telecommunication Engineering"

    # Rule 11: Chemical Engineering
    if "chemical" in b:
        print(f"{branch} ---> Chemical Engineering")
        return "Chemical Engineering"

    # Rule 12: Civil Engineering
    if "civil" in b or b == "c.e":
        print(f"{branch} ---> Civil Engineering")
        return "Civil Engineering"

    # If no rules matched, return the original trimmed string.
    print(f"{branch} ---> No matching rule")
    return branch.strip()


def main():
    # Load the CSV file into a DataFrame.
    # Replace the file path with your CSV file location.
    input_file = "./column_data_csv/EngineeringCourseBranch.csv"
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading {input_file}: {e}")
        return

    # Check if the expected column is present.
    if "EngineeringCourseBranch" not in df.columns:
        print("Column 'EngineeringCourseBranch' not found in the CSV file.")
        return

    # Apply the cleaning function to the column.
    df["StandardizedBranch"] = df["EngineeringCourseBranch"].apply(standardize_branch)

    # Preview cleaned data.
    print("Cleaned DataFrame preview:")
    print(df.head(20))

    # Save the cleaned DataFrame to a new CSV file.
    output_file = "./column_data_csv/EngineeringCourseBranch_cleaned.csv"
    try:
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to {output_file}")
    except Exception as e:
        print(f"Error saving cleaned CSV: {e}")


if __name__ == "__main__":
    main()