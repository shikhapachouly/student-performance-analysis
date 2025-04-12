import pandas as pd
import numpy as np

# Provided data for InternshipPaymentStatusPaidFree column
internship_status = [
    "PAID", "FREE", np.nan, "Have not completed any internship till now",
    "Not Applicable ", "-", "Industrial Training", "I am currently interning",
    "No internship", "No", "Stipend ", "no internship", "Free and paid both",
    "Don't no ", "Not done ", "Did freelance ", "Not completed ", ". ",
    "Netleap IT training and solutions ", "Not done", "No done yet", "NA ",
    "Not done yet", "didnt do ", "Netleap IT Training and Solutions ",
    "I have not done any internship ", "No any internship ",
    "Registration charges only ", ".. ", "Dihadi", "Didn't had any internship ",
    "Not done yet ", "Not done any yet ", "I haven't completed any internship.",
    "No internship", "No internships done", "None ", "No internship completed ",
    "Not attended any internship ", "Not attained",
    "Not completed any of it yet", "No internship until now ",
    "Not complete any internship ", "Not done internship till date",
    "Not done internship", "Dont take it", "Not ", "Not completed any internship ",
    "No ", "I am not joining any internship", "Not completed",
    "I have not done any intership ", ".",
    "I am not complete any internship ", "Nit compled any internship ",
    "Not complete internship ", "I'm  not completed any kind of internship ",
    "I have not completed any internship ", "No internship till", "Not yet ",
    "I haven't did any internship yet"
]

df = pd.DataFrame({"InternshipPaymentStatusPaidFree": internship_status})


def map_internship_payment_status(val):
    """
    Map the raw internship payment status value to one of:
      "PAID", "FREE", or "NOT_DONE"

    Logic:
      1. Convert to lowercase and strip extra spaces.
      2. For empty or known placeholders, return "NOT_DONE".
      3. Check for negative phrases that indicate no internship was completed.
      4. If both "paid" and "free" are mentioned, return "PAID" (paid takes precedence).
      5. Check if the value indicates a paid internship (contains "paid", "stipend", or starts with "yes").
      6. Check if the value indicates a free internship.
      7. If none of the above conditions match, default to "NOT_DONE".
    """
    # If not a string, treat it as "NOT_DONE"
    if not isinstance(val, str):
        return "NOT_DONE"

    v = val.strip().lower()

    # Define empty/unknown values
    empty_values = {"", "nan", ".", "..", "-", "--", "na", "not applicable"}
    if v in empty_values:
        return "NOT_DONE"

    # Define negative phrases that indicate the internship was not completed
    negative_indicators = [
        "have not completed", "not completed", "not done", "no internship", "none",
        "dont", "don't", "didn't", "haven't", "nothing", "not attended", "not attained",
        "not complete", "not joining", "not did"
    ]
    for neg in negative_indicators:
        if neg in v:
            return "NOT_DONE"

    # Check if the value contains both "paid" and "free" (choose PAID in that case)
    if "paid" in v and "free" in v:
        return "PAID"

    # Look for paid internship indicators
    if "paid" in v or v.startswith("yes") or v == "yes" or "stipend" in v:
        return "PAID"

    # Look for free internship indicator
    if "free" in v:
        return "FREE"

    # Some answers may mention "training". In this dataset we choose to treat these as NOT_DONE.
    if "training" in v:
        return "NOT_DONE"

    # If none of the above conditions apply,
    # default assumption: if an employer name or positive info is provided, assume PAID.
    return "PAID"


# Apply mapping function to create a new column with values "PAID", "FREE", or "NOT_DONE"
df["InternshipPaymentMapping"] = df["InternshipPaymentStatusPaidFree"].apply(map_internship_payment_status)

# Display the DataFrame with mapping results
print(df)