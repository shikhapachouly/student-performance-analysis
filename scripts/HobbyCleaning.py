#!/usr/bin/env python3
import pandas as pd
import re

CATEGORY_MAPPING = {
    "Sports": [
        "soccer", "chess", "cricket", "football", "basketball", "volley ball",
        "badminton", "tennis", "kabaddi", "athletics", "skating", "boxing",
        "martial arts", "golf", "sport", "foot ball", "taekwondo", "archery",
        "kabbadi", "wicket", "gilli danda", "rifel shooting", "shooting",
        "swim", "swimming", "athlete", "volley", "karate"
    ],
    "Arts & Crafts": [
        "drawing", "painting", "sketching", "doodling", "craft", "mehndi",
        "calligraphy", "art", "creativity", "colouring", "acting", "pictures"
    ],
    "Music & Dance": [
        "music", "singing", "dancing", "guitar", "flute", "tabla", "piano",
        "beatboxing", "harmonium", "dance", "song"
    ],
    "Reading & Writing": [
        "read", "novel", "book", "poetry", "writing", "poem", "script",
        "stories", "writting", "learn", "listener"
    ],
    "Travel & Adventure": [
        "travel", "trek", "hiking", "camping", "exploring", "adventure",
        "road trip", "driving", "toor", "trave", "tracking", "treaking"
    ],
    "Cooking & Baking": [
        "cook", "cooking", "baking", "food", "diy", "kitchen"
    ],
    "Technology & Coding": [
        "coding", "programming", "computing", "tech", "software", "cyber",
        "robotics", "computer", "web development", "conding", "mechanical", "electronics", "doing"
    ],
    "Gaming": [
        "game", "gaming", "esports", "video games", "virtual"
    ],
    "Fitness & Gym": [
        "gym", "work out", "workout", "bodybuilding", "exercise", "fitness",
        "weight training", "exercising", "running", "walking", "forces"
    ],
    "Social & Volunteering": [
        "social", "volunteer", "community", "networking", "event", "communicate"
    ],
    "Design": [
        "design", "graphic", "illustration", "animation", "designer",
        "vfx", "videography", "video editing", "photography", "photgraphy", "new"
    ],
    "Business & Finance": [
        "business", "finance", "stock market", "forex", "entrepreneurship",
        "trading", "money making", "mnc", "project"
    ],
    "Education & Study": [
        "math", "maths", "studying", "study", "teaching", "knowledge",
        "learning", "data analytics", "research", "elocution", "language", "studies"
    ],
    "Entertainment": [
        "movie", "movies", "cinema", "youtube", "tv", "web series", "comedy",
        "filmmaking", "film making", "watching", "watch", "reels", "dramatics"
    ],
    "Automotive & Racing": [
        "car", "cars", "car racing", "automobile", "motorcycle", "racing",
        "bike", "riding", "railfanning"
    ],
    "Occult & Astrology": [
        "occult", "astrology", "palmistry", "numerology", "astronomy"
    ],
    "Mind & Wellness": [
        "homeless", "meditation", "yoga", "sleep", "sleeping", "chilling", "loved", "family", "listening", "parents",
        "crying", "aquarium"
    ],
    "Gardening": [
        "gardening"
    ],
    "Puzzles": [
        "rubik", "cube", "problem solving"
    ]
}

NONE_VALUES = {
    "none", "-", "no", "na", "nothing", "dont know", "don't know", ".", ""
}


def categorize_hobby(text):
    """Categorize hobbies into predefined categories or return None"""
    # Handle empty/whitespace-only values and non-strings
    if not isinstance(text, str) or not text.strip():
        return "None"

    # Clean and normalize text
    text_clean = re.sub(r"[^\w\s]", " ", text.lower().strip()).strip()

    # Check for explicit none values
    if text_clean in NONE_VALUES:
        return "None"

    # Find matching categories
    categories = set()
    for category, keywords in CATEGORY_MAPPING.items():
        if any(keyword in text_clean for keyword in keywords):
            categories.add(category)

    # Fallback for "playing" keyword
    if not categories and "playing" in text_clean.split():
        categories.add("Sports")

    return ", ".join(sorted(categories)) if categories else "None"


def main():
    input_file = "./column_data_csv/HobbiesAndInterests.csv"
    output_file = "./column_data_csv/HobbiesAndInterests_cleaned.csv"

    try:
        df = pd.read_csv(input_file, keep_default_na=False)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if "HobbiesAndInterests" not in df.columns:
        print("Error: Missing required column 'HobbiesAndInterests'")
        return

    # Add cleaned categories
    df["BroadCategory"] = df["HobbiesAndInterests"].apply(categorize_hobby)

    try:
        df.to_csv(output_file, index=False)
        print(f"Successfully processed {len(df)} rows")
        print(f"Saved results to: {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")


if __name__ == "__main__":
    main()