import pandas as pd
import numpy as np

# Provided raw data for EngineeringCourseBranch column
engineering_branches = [
    "Electrical Engineering ",
    "Electronics and telecommunication engineering ",
    "Entc",
    "Computer engineering ",
    "Electrical ",
    "Computer Engineering ",
    "Electronic and telecommunications ",
    "EnTC",
    "Computer Science",
    "Computer",
    "CSE-AIML",
    "Computer ",
    "Mechanical engineering ",
    "Computer science engineering ",
    "Information Technology (IT)",
    "Electronics (vlsi and design)",
    "Computer Engineering",
    "ELECTRICAL ENGINEERING ",
    "Electrical",
    "Mechanical engineering",
    "Artificial intelligence and data science ",
    "Computer science ",
    "E&TC",
    "Electronics VLSI Design and Technology ",
    "Computer engineering",
    "Chemical ",
    "Chemical Engineering ",
    "CHEMICAL ENGINEERING ",
    "Mechanical Engineering ",
    "Electronics and Telecommunication ",
    "B tech aiml",
    "Chemical engineering ",
    "IT",
    "ELECTRICAL ",
    "Computer Engg",
    "Electrical engineering ",
    "Computer Engineer ",
    "Mechanical",
    "Information Technology ",
    "Conputer engineering ",
    "Electronics and telecommunication ",
    "Mechanical ",
    "Information technology ",
    "Artificial intelligence and Data science ",
    "AIML",
    "Civil ",
    "Electronics and telecommunications engineering ",
    "Artificial Intelligence and Data Science ",
    "Artificial Intelligence & Data Science ",
    "bachelor of technology",
    "Civil Engineer ",
    "Civil engineering ",
    "COMPUTER ENGINEERING ",
    "Mechanical Engineering",
    "Civil Engineering ",
    "It",
    "Electronics and Telecommunications",
    "Electronic and Telecommunication ",
    "mechanical ",
    "COMPUTER ENGINEERING",
    "Electronics & Telecommunications",
    "Civil Engineering",
    "Computer science",
    "Information technology",
    "Ai&Ds",
    "Genba sopanrao moze college of engineering ",
    "Second year of BE Computer Engineering",
    "ENTC ",
    "Comp",
    "Computer science engineering with specialization in AIML ",
    "Computer engg",
    "Information Technology Engineering",
    "GSMCOE",
    "TE computer engineering",
    "BE Computer Engg",
    "CSE-AIML ",
    "Btech cse Aiml ",
    "CHEMICAL",
    "AIDS ",
    "Ai&ds",
    "Cs",
    "B.E.(Computer Engineering)",
    "Electrical engineering",
    "Mech",
    "CSE",
    "Aiml",
    "electronics and telecommunication",
    "Information Technology",
    "INFORMATION TECHNOLOGY",
    "Electronics and telecommunication",
    "MECHANICAL ENGINEERING ",
    "B Tech Information Technology",
    "Computer Science ",
    "Electronics (vlsi design)",
    "computer engineering ",
    "AI&DS",
    "Electrical Engineering",
    "artificial intelligence and data science",
    "IT ",
    "CSE -AIML ",
    "Electronics( VLSI Design and Technology)",
    "ENTC",
    "Electronic ( VLSI design and tech ) engg.",
    "Aids",
    "Compute Engineering ",
    "Computer Enginnering",
    "E&tc ",
    "Chemical Engineering",
    "Electronics and Telecommunications Engineering ",
    "Electronics(vlsi) ",
    "Computer engineering (BE)",
    "Computer Science with Artificial Intelligence and Machine learning",
    "e&tc",
    "Electronics Engineering (VLSI Design and Technology)",
    "AI&ML",
    "VLSI Design And Technology ",
    "Artificial Intelligence and data science",
    "BE in computer engineering ",
    "Ai and data science ",
    "Artificial intelligence and data science (Ai&Ds)",
    "Computer Science Engineering with specialization in Artificial Intelligence and Machine Learning ",
    "Artificial intelligence and data science engineering ",
    "BE Computer ",
    "cs aiml ",
    "Computer engeneering",
    "CS",
    "Electronic and Telecommunication Engineering ",
    "Computer Enginerrin",
    "Artificial intelligence and machine learning",
    "Mechanical sandwich ",
    "Electronics Vlsi",
    "Artificial Intelligence ",
    "Electronics and Telecommunication Engineering ",
    "Mechanical engineering sandwich",
    "Chemical engineering",
    "AI and data science ",
    "Mechanical Sandwich",
    "AIDS",
    "Electrical Branch ",
    "Computer engineer ",
    "Print engineering ",
    "Mechanical (Sandwich) Engineering ",
    "Engg.Computer science ",
    "Mechanical Sandwich Engineering",
    "INFORMATION TECHNOLOGY ",
    "Mechanical Sandwich Engineering ",
    "Machenical",
    "C.E",
    "BE(Computer Engineering) ",
    "Vlsi design and technology ",
    "Electronics and Telecommunications ",
    "Computer Science Engineering",
    "Mechanical z",
    "Electronic engineering vlsi",
    "Artificial Intelligence and Data Science",
    "Artificial Intelligence and Machine Learning ",
    "Electronics (VLSI Design and Technology)",
    "VLSI design and technology ",
    "Artificial intelligence and data science",
    "VLSI",
    "TE Mechanical",
    "Artificial intelligence and Machine learning ",
    "Electronics VLSI Design And Technology ",
    "Artificial intelligence ",
    "IT Engineering ",
    "Electronics VLSI",
    "Civil",
    "AI & DS",
    "VLSI DESIGN AND TECHNOLOGY (ENTC)",
    "BE(IT)",
    "VLSI ENGINEERING ",
    "BE comp",
    "Computer science Engineering ",
    "Chemical",
    "Mechanical sandwich",
    "Artificial intelligence and data science (AIDS)",
    "Electronic vlsi design and technology ",
    "BE.Mechanical ",
    "BE Mechanical "
]

# Create a DataFrame
df = pd.DataFrame({"EngineeringCourseBranch": engineering_branches})


def map_engineering_branch(val):
    """
    Map a raw engineering course/branch string to a broader categorical value.

    The function returns one of the following categories:
      - "Computer Engineering"
      - "Electrical Engineering"
      - "Mechanical Engineering"
      - "Electronics and Telecommunication Engineering"
      - "Information Technology & AIML"
      - "Chemical Engineering"
      - "Civil Engineering"
      - "Other/Unknown"

    The mapping is done by doing substring (case-insensitive) checks.
    """
    if not isinstance(val, str) or not val.strip():
        return "Other/Unknown"

    # Clean the value:
    v = val.strip().lower()

    # Check for Chemical Engineering first (unique keyword)
    if "chemical" in v:
        return "Chemical Engineering"

    # Check for Civil Engineering
    if "civil" in v:
        return "Civil Engineering"

    # Check for IT & AIML (look for "information technology", "aiml", "artificial intelligence", or "data science")
    if ("information technology" in v or
            "aiml" in v or
            "artificial intelligence" in v or
            "data science" in v or
            v == "it" or v == "it "):
        return "Information Technology & AIML"

    # Check for Computer Engineering (look for "computer", "cse", "cs ", or "comp")
    # Note: This check comes after IT & AIML so that courses with specialization in AI or IT are captured above.
    if "computer" in v or "cse" in v or "cs " in v or "comp " in v:
        return "Computer Engineering"

    # Check for Electrical Engineering if the string contains "electrical"
    if "electrical" in v:
        return "Electrical Engineering"

    # Check for Electronics and Telecommunication Engineering
    # Look for substrings like "entc", "electronics", "e&tc", "vlsi", or "telecommunication"
    if "entc" in v or "electronics" in v or "e&tc" in v or "vlsi" in v or "telecommunication" in v:
        return "Electronics and Telecommunication Engineering"

    # Check for Mechanical Engineering (e.g., "mechanical", "mech", "sandwich")
    if "mechanical" in v or "mech" in v or "sandwich" in v:
        return "Mechanical Engineering"

    # If none of the above conditions hold, return Other/Unknown
    return "Other/Unknown"


# Apply the mapping function to the EngineeringCourseBranch column
df["BranchCategory"] = df["EngineeringCourseBranch"].apply(map_engineering_branch)

# Display the DataFrame to see the mapping results
print(df)