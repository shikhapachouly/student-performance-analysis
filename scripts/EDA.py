import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from PIL import Image


def create_image_folder(folder_name="images"):
    """
    Create an images folder if it doesn't exist.
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created folder '{folder_name}' for saving images.")
    else:
        print(f"Folder '{folder_name}' already exists.")


def collate_images_to_pdf(image_folder="images", output_pdf="EDA_Collated.pdf"):
    """
    Collate all PNG images in the specified folder into a single PDF.

    Parameters:
    -----------
    image_folder : str
        The folder containing the PNG images.
    output_pdf : str
        The name of the output PDF file.
    """
    # Get list of images files sorted alphabetically
    image_files = sorted([file for file in os.listdir(image_folder) if file.endswith('.png')])

    if not image_files:
        print(f"No PNG images found in '{image_folder}' to collate.")
        return

    # Define A4 dimensions in points
    a4_width, a4_height = A4

    # Initialize canvas
    c = canvas.Canvas(os.path.join(image_folder, output_pdf), pagesize=A4)

    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        try:
            # Open the images to get its size
            with Image.open(img_path) as img:
                img_width, img_height = img.size
                aspect = img_height / float(img_width)

                # Define margins
                margin = 50  # points

                # Calculate available width and height
                available_width = a4_width - 2 * margin
                available_height = a4_height - 2 * margin

                # Calculate new dimensions to fit within A4 while maintaining aspect ratio
                if aspect > 1:
                    # Image is taller than wide
                    display_height = available_height
                    display_width = display_height / aspect
                else:
                    # Image is wider than tall
                    display_width = available_width
                    display_height = display_width * aspect

                # Center the images
                x = (a4_width - display_width) / 2
                y = (a4_height - display_height) / 2

                # Add images to PDF
                c.drawImage(ImageReader(img_path), x, y, width=display_width, height=display_height)
                c.showPage()
                print(f"Added '{img_file}' to '{output_pdf}'.")
        except Exception as e:
            print(f"Failed to add '{img_file}' to PDF: {e}")

    # Save the PDF
    c.save()
    print(f"Collated PDF saved as '{output_pdf}' in '{image_folder}' folder.")


def exploratory_analysis(csv_filename="your_data.csv"):
    """
    Perform exploratory data analysis (EDA) on the given student dataset and
    create colorful (fancy) diagrams for inclusion in a research paper.

    Parameters:
    -----------
    csv_filename : str
        The path (or filename) to the CSV file containing the dataset.
        By default, uses 'your_data.csv'.

    Notes:
    ------
    1. This script assumes the CSV contains at least these columns:
       [
         "Timestamp", "Username", "NameOfStudent", "GenderOfStudent", "AgeGroupOfStudent",
         "NameOfInstitute", "ResidentialArea", "10thPercentage", "12thPercentage",
         "AggregateCGPATillCurrentSemester", "CurrentBacklogSubjects", "AttendedNurserySchool",
         "ReasonForChoosingCollege", "EngineeringCourseBranch", "ReasonForChoosingCourse",
         "MothersEducationStatus", "MothersJobStatus", "FathersEducationStatus",
         "FathersJobStatus", "CurrentGuardian", "FamilySize", "ParentalRelationshipStatus",
         "FamilyRelationshipQualityScale1to5", "FamilyIncome", "InternetAccessAtHome",
         "ModeOfTransport", "TravelTimeForCommute", "DailyStudyTime",
         "InstituteProvidesExtracurricularCourses", "ParticipatedInWorkshopsSeminars",
         "CommunityServiceVolunteering", "CommunicationSkillsScale1to5",
         "LeadershipQualitiesScale1to5", "TeamworkSkillsScale1to5", "CompletedInternships",
         "InternshipPaymentStatusPaidFree", "CompletedProjects", "PeerGroupQualityScale1to5",
         "TimeManagementScale1to5", "CurrentHealthStatusScale1to5", "StressFrequencyScale1to5",
         "CopingMechanismsForStress", "HobbiesAndInterests", "SocializingFrequencyScale1to5",
         "InARomanticRelationship", "FamilyMotivatesExtraClasses", "AttendingPaidClasses",
         "PlacementStatusEmployerIfYes", "InterestInHigherEducation", "AlcoholConsumption",
         "Smoking", "OtherAddictions"
       ]
    2. Make sure to install the following libraries before running:
         - pandas
         - seaborn
         - matplotlib
         - numpy
         - scikit-learn
         - reportlab
         - pillow
    3. The script generates various plots (histograms, bar charts, heatmaps, etc.).
       Feel free to modify figure sizes, color palettes, and chart types as desired.
    """

    # Create images folder
    image_folder = "images"
    create_image_folder(image_folder)

    # --------------------------------------------------------------------------
    # 1. Load the Data
    # --------------------------------------------------------------------------
    try:
        df = pd.read_csv(csv_filename, encoding="utf-8")
        print(f"Successfully loaded '{csv_filename}'.")
    except FileNotFoundError:
        print(f"Error: The file '{csv_filename}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred while loading the CSV: {e}")
        return

    # --------------------------------------------------------------------------
    # 2. Basic Data Cleaning
    #    (Adjust these steps based on your actual needs.)
    # --------------------------------------------------------------------------
    # Remove obvious unique identifiers, if not relevant to analysis
    cols_to_drop = ["Timestamp", "Username", "NameOfStudent"]
    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    if existing_cols_to_drop:
        df.drop(columns=existing_cols_to_drop, inplace=True)
        print(f"Dropped columns: {existing_cols_to_drop}")
    else:
        print("No columns dropped. Columns to drop not found in the dataset.")

    # Remove 'Other' from gender
    if "GenderOfStudent" in df.columns:
        initial_count = df.shape[0]
        df = df[df["GenderOfStudent"].str.lower() != "other"]
        final_count = df.shape[0]
        removed = initial_count - final_count
        print(f"Removed {removed} rows with 'Other' gender.")
    else:
        print("Column 'GenderOfStudent' not found in the dataset.")

    # Handle missing values with forward fill
    df = df.ffill()
    print("Performed forward fill to handle missing values.")

    # Convert percentages to numeric if they have % signs (example).
    # This depends on how your CSV stores them; some rows might be "94%" or just "94".
    # Here’s a generic approach:
    percentage_cols = ["10thPercentage", "12thPercentage"]
    for percentage_col in percentage_cols:
        if percentage_col in df.columns:
            # Remove % sign if present
            df[percentage_col] = df[percentage_col].astype(str).str.replace("%", "", regex=False)
            # Convert to numeric, coercing errors to NaN
            df[percentage_col] = pd.to_numeric(df[percentage_col], errors='coerce')
            # Fill NaN with mean
            if df[percentage_col].isnull().any():
                mean_value = df[percentage_col].mean()
                df[percentage_col] = df[percentage_col].fillna(mean_value)
                print(f"Filled NaN values in '{percentage_col}' with mean: {mean_value:.2f}")
            print(f"Converted '{percentage_col}' to numeric.")
        else:
            print(f"Column '{percentage_col}' not found in the dataset.")

    # Convert CGPA to float if stored as string
    if "AggregateCGPATillCurrentSemester" in df.columns:
        df["AggregateCGPATillCurrentSemester"] = pd.to_numeric(df["AggregateCGPATillCurrentSemester"], errors="coerce")
        if df["AggregateCGPATillCurrentSemester"].isnull().any():
            mean_cgpa = df["AggregateCGPATillCurrentSemester"].mean()
            df["AggregateCGPATillCurrentSemester"] = df["AggregateCGPATillCurrentSemester"].fillna(mean_cgpa)
            print(f"Filled NaN values in 'AggregateCGPATillCurrentSemester' with mean: {mean_cgpa:.2f}")
        print("Converted 'AggregateCGPATillCurrentSemester' to numeric.")

        # ----------- Scaling CGPA to 1-10 Scale ----------- #
        # Check the current range of CGPA
        cgpa_min = df["AggregateCGPATillCurrentSemester"].min()
        cgpa_max = df["AggregateCGPATillCurrentSemester"].max()
        print(f"Original CGPA range: Min = {cgpa_min}, Max = {cgpa_max}")

        # Initialize MinMaxScaler to scale CGPA to 1-10
        scaler = MinMaxScaler(feature_range=(1, 10))
        df["cgpa_scaled"] = scaler.fit_transform(df[["AggregateCGPATillCurrentSemester"]])
        print("Scaled 'AggregateCGPATillCurrentSemester' to 'cgpa_scaled' with range 1-10.")

        # Optionally, round the scaled CGPA to two decimal places
        df["cgpa_scaled"] = df["cgpa_scaled"].round(2)
        print("Rounded 'cgpa_scaled' to two decimal places.")
    else:
        print("Column 'AggregateCGPATillCurrentSemester' not found in the dataset.")

    # --------------------------------------------------------------------------
    # 3. Identify and Encode Categorical Columns
    # --------------------------------------------------------------------------
    # Define all columns that are categorical
    categorical_cols = [
        "GenderOfStudent", "AgeGroupOfStudent", "NameOfInstitute", "ResidentialArea",
        "CurrentBacklogSubjects", "AttendedNurserySchool", "ReasonForChoosingCollege",
        "EngineeringCourseBranch", "ReasonForChoosingCourse", "MothersEducationStatus",
        "MothersJobStatus", "FathersEducationStatus", "FathersJobStatus", "CurrentGuardian",
        "FamilySize", "ParentalRelationshipStatus", "FamilyRelationshipQualityScale1to5",
        "FamilyIncome", "InternetAccessAtHome", "ModeOfTransport", "TravelTimeForCommute",
        "DailyStudyTime", "InstituteProvidesExtracurricularCourses",
        "ParticipatedInWorkshopsSeminars", "CommunityServiceVolunteering",
        "CommunicationSkillsScale1to5", "LeadershipQualitiesScale1to5",
        "TeamworkSkillsScale1to5", "CompletedInternships", "CompletedProjects",
        "PeerGroupQualityScale1to5", "TimeManagementScale1to5",
        "CurrentHealthStatusScale1to5", "StressFrequencyScale1to5",
        "CopingMechanismsForStress", "HobbiesAndInterests",
        "SocializingFrequencyScale1to5", "InARomanticRelationship",
        "FamilyMotivatesExtraClasses", "AttendingPaidClasses",
        "PlacementStatusEmployerIfYes", "InterestInHigherEducation",
        "AlcoholConsumption", "Smoking", "OtherAddictions", "InternshipPaymentStatusPaidFree"
    ]

    # Ensure that all categorical columns exist in the dataframe
    categorical_cols = [col for col in categorical_cols if col in df.columns]
    print(f"Categorical columns identified: {categorical_cols}")

    # Separate columns into categorical and numeric
    # Numeric columns are those not in categorical_cols and are of numeric dtype
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove percentage columns, 'cgpa', and 'cgpa_scaled' from numeric_cols to avoid duplication
    numeric_cols = [col for col in numeric_cols if col not in percentage_cols + ["AggregateCGPATillCurrentSemester", "cgpa_scaled"]]
    print(f"Numeric columns identified: {numeric_cols}")

    # Initialize a dictionary to store label encoders for each column
    label_encoders = {}

    # Convert binary categorical columns (Yes/No) to numeric (1/0) in new encoded columns
    binary_cols = [
        "CurrentBacklogSubjects", "AttendedNurserySchool", "InternetAccessAtHome",
        "AlcoholConsumption", "Smoking", "OtherAddictions"
    ]
    binary_existing_cols = [col for col in binary_cols if col in df.columns]
    for col in binary_existing_cols:
        encoded_col = f"{col}_encoded"
        df[encoded_col] = df[col].map({'Yes': 1, 'No': 0})
        # Handle any unmapped values by filling with mode (most frequent)
        if df[encoded_col].isnull().any():
            mode_val = df[encoded_col].mode()
            if not mode_val.empty:
                df[encoded_col] = df[encoded_col].fillna(mode_val[0])
                print(f"Filled NaN values in '{encoded_col}' with mode: {mode_val[0]}")
            else:
                df[encoded_col] = df[encoded_col].fillna(0)  # Default to 0 if mode is empty
                print(f"Filled NaN values in '{encoded_col}' with default value: 0")
        print(f"Encoded binary column '{col}' into '{encoded_col}': {df[encoded_col].unique()}")

    # Handle categorical columns with more than two categories using Label Encoding
    # Exclude binary columns already handled
    multi_cat_cols = [col for col in categorical_cols if col not in binary_existing_cols]

    for col in multi_cat_cols:
        unique_values = df[col].nunique()
        if unique_values > 2:
            try:
                le = LabelEncoder()
                encoded_col = f"{col}_encoded"
                df[encoded_col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le  # Store the label encoder
                print(
                    f"Label encoded column '{col}' into '{encoded_col}'. Unique values now: {df[encoded_col].nunique()}")
            except Exception as e:
                print(f"Error encoding column '{col}': {e}")
        elif unique_values == 2:
            # For binary categorical columns not already handled
            encoded_col = f"{col}_encoded"
            df[encoded_col] = df[col].map({'Yes': 1, 'No': 0})
            if df[encoded_col].isnull().any():
                mode_val = df[encoded_col].mode()
                if not mode_val.empty:
                    df[encoded_col] = df[encoded_col].fillna(mode_val[0])
                    print(f"Filled NaN values in '{encoded_col}' with mode: {mode_val[0]}")
                else:
                    df[encoded_col] = df[encoded_col].fillna(0)
                    print(f"Filled NaN values in '{encoded_col}' with default value: 0")
            print(f"Encoded binary column '{col}' into '{encoded_col}': {df[encoded_col].unique()}")
        else:
            print(f"Column '{col}' has {unique_values} unique values. Skipping encoding.")

    # --------------------------------------------------------------------------
    # 4. Set Up Global Plot Styles
    # --------------------------------------------------------------------------
    sns.set_theme(style="whitegrid")
    # You can try different palettes like "deep", "bright", "pastel", "dark", etc.
    # Example: sns.set_palette("coolwarm", n_colors=8)

    # --------------------------------------------------------------------------
    # 5. Distribution of Numeric Variables
    #    (Example: Histograms of 10th, 12th percentages, CGPA)
    # --------------------------------------------------------------------------
    numeric_feature_cols = percentage_cols + ["cgpa_scaled"]  # Use 'cgpa_scaled' instead of 'cgpa'
    for col in numeric_feature_cols:
        if col in df.columns:
            plt.figure(figsize=(8, 5))
            sns.histplot(data=df, x=col, kde=True, color="#1f77b4", alpha=0.7)
            plt.title(f"Distribution of {col.replace('_', ' ').title()}")
            plt.xlabel(col.replace('_', ' ').title())
            plt.ylabel("Count")
            plt.tight_layout()
            save_path = os.path.join(image_folder, f"histogram_{col}.png")
            plt.savefig(save_path)
            print(f"Saved histogram for '{col}' as '{save_path}'.")
            plt.close()
        else:
            print(f"Numeric feature column '{col}' not found.")

    # --------------------------------------------------------------------------
    # 6. Bar Charts for Categorical Variables
    #    (Example: Gender, Residential Area, Placement Status, Engineering Branch)
    # --------------------------------------------------------------------------
    bar_chart_cols = [
        "GenderOfStudent",
        "ResidentialArea",
        "PlacementStatusEmployerIfYes",
        "EngineeringCourseBranch",
        "FamilyIncome",
        "CurrentGuardian",
        "ModeOfTransport",
        "DailyStudyTime",
        "ReasonForChoosingCourse",
        "ReasonForChoosingCollege"
    ]
    bar_chart_existing_cols = [col for col in bar_chart_cols if col in df.columns]
    for col in bar_chart_existing_cols:
        plt.figure(figsize=(12, 6))
        order = df[col].value_counts().index
        sns.countplot(data=df, x=col, order=order, palette="viridis")
        plt.title(f"Bar Chart of {col.replace('_', ' ').title()}")
        plt.xlabel(col.replace('_', ' ').title())
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        save_path = os.path.join(image_folder, f"bar_chart_{col}.png")
        plt.savefig(save_path)
        print(f"Saved bar chart for '{col}' as '{save_path}'.")
        plt.close()

    # --------------------------------------------------------------------------
    # 7. Box Plots: Numeric vs. Categorical
    #    (Example: CGPA by Gender, CGPA by Residential Area, CGPA by Engineering Branch)
    # --------------------------------------------------------------------------
    box_plot_comparisons = [
        ("cgpa_scaled", "GenderOfStudent"),
        ("cgpa_scaled", "ResidentialArea"),
        ("10thPercentage", "GenderOfStudent"),
        ("12thPercentage", "GenderOfStudent"),
        ("cgpa_scaled", "EngineeringCourseBranch")
    ]
    for numeric_feature, cat_feature in box_plot_comparisons:
        if numeric_feature in df.columns and cat_feature in df.columns:
            plt.figure(figsize=(12, 6))
            sns.boxplot(
                data=df,
                x=cat_feature,
                y=numeric_feature,
                palette="Set2"
            )
            plt.title(f"{numeric_feature.replace('_', ' ').title()} by {cat_feature.replace('_', ' ').title()}")
            plt.xlabel(cat_feature.replace('_', ' ').title())
            plt.ylabel(numeric_feature.replace('_', ' ').title())
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            save_path = os.path.join(image_folder, f"boxplot_{numeric_feature}_by_{cat_feature}.png")
            plt.savefig(save_path)
            print(f"Saved box plot for '{numeric_feature}' by '{cat_feature}' as '{save_path}'.")
            plt.close()
        else:
            print(f"Cannot create box plot for '{numeric_feature}' by '{cat_feature}'. Columns not found.")

    # --------------------------------------------------------------------------
    # 8. Scatter Plot: Relationship between 10th, 12th percentages and CGPA
    #    (Colored by Gender or Residential Area)
    # --------------------------------------------------------------------------
    if all(col in df.columns for col in ["10thPercentage", "12thPercentage", "cgpa_scaled"]):
        hue_col = "GenderOfStudent" if "GenderOfStudent" in df.columns else None
        palette = "coolwarm" if hue_col else "viridis"

        # Scatter plot for 10th Percentage vs. CGPA
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=df,
            x="10thPercentage",
            y="cgpa_scaled",
            hue=hue_col,
            palette=palette,
            alpha=0.7,
            edgecolor="k"
        )
        plt.title("Relationship between 10th Percentage and CGPA (Scaled)")
        plt.xlabel("10th Percentage")
        plt.ylabel("CGPA (Scaled)")
        if hue_col:
            plt.legend(title=hue_col.replace('_', ' ').title())
        plt.grid(True)
        plt.tight_layout()
        save_path = os.path.join(image_folder, "scatter_10th_percentage_vs_cgpa_scaled.png")
        plt.savefig(save_path)
        print(f"Saved scatter plot for '10thPercentage' vs. 'cgpa_scaled' as '{save_path}'.")
        plt.close()

        # Scatter plot for 12th Percentage vs. CGPA
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=df,
            x="12thPercentage",
            y="cgpa_scaled",
            hue=hue_col,
            palette=palette,
            alpha=0.7,
            edgecolor="k"
        )
        plt.title("Relationship between 12th Percentage and CGPA (Scaled)")
        plt.xlabel("12th Percentage")
        plt.ylabel("CGPA (Scaled)")
        if hue_col:
            plt.legend(title=hue_col.replace('_', ' ').title())
        plt.grid(True)
        plt.tight_layout()
        save_path = os.path.join(image_folder, "scatter_12th_percentage_vs_cgpa_scaled.png")
        plt.savefig(save_path)
        print(f"Saved scatter plot for '12thPercentage' vs. 'cgpa_scaled' as '{save_path}'.")
        plt.close()
    else:
        print("Required columns for scatter plots not found.")

    # --------------------------------------------------------------------------
    # 9. Correlation Heatmap (Numeric Columns)
    # --------------------------------------------------------------------------
    # Combine numeric feature columns
    numeric_data = df[numeric_feature_cols + numeric_cols]
    if not numeric_data.empty:
        plt.figure(figsize=(16, 12))
        corr = numeric_data.corr()
        sns.heatmap(
            corr,
            annot=True,
            cmap="magma",
            fmt=".2f",
            square=True,
            linewidths=.5
        )
        plt.title("Correlation Matrix (Numeric Features)")
        plt.tight_layout()
        save_path = os.path.join(image_folder, "correlation_heatmap.png")
        plt.savefig(save_path)
        print(f"Saved correlation heatmap as '{save_path}'.")
        plt.close()
    else:
        print("No numeric data available for correlation heatmap.")

    # --------------------------------------------------------------------------
    # 10. Pair Plot to See Relationships Between Important Numeric Variables
    # --------------------------------------------------------------------------
    # Example with: 10th percentage, 12th percentage, CGPA (scaled), and daily_study_time if numeric
    interesting_cols = ["10thPercentage", "12thPercentage", "cgpa_scaled"]
    if "DailyStudyTime" in df.columns and pd.api.types.is_numeric_dtype(df["DailyStudyTime"]):
        interesting_cols.append("DailyStudyTime")
    existing_interesting_cols = [col for col in interesting_cols if
                                 col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

    if existing_interesting_cols:
        try:
            pairplot = sns.pairplot(
                df[existing_interesting_cols],
                diag_kind="kde",
                palette="tab10",
                plot_kws={"alpha": 0.5}
            )
            pairplot.fig.suptitle("Pair Plot of Key Numeric Features", y=1.02)
            save_path = os.path.join(image_folder, "pairplot_key_numeric_features.png")
            pairplot.savefig(save_path)
            print(f"Saved pair plot of key numeric features as '{save_path}'.")
            plt.close()
        except Exception as e:
            print(f"An error occurred while creating the pair plot: {e}")
    else:
        print("Not enough numeric columns available for pair plot.")

    # --------------------------------------------------------------------------
    # 11. Additional Plots (Examples)
    # --------------------------------------------------------------------------
    # Example: Count of Students with Backlogs
    backlog_col = "CurrentBacklogSubjects_encoded"  # Use encoded backlog
    if backlog_col in df.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x=backlog_col, palette="Set3")
        plt.title("Count of Students with Backlogs")
        plt.xlabel("Backlog (0 = No, 1 = Yes)")
        plt.ylabel("Count")
        plt.tight_layout()
        save_path = os.path.join(image_folder, "count_backlog.png")
        plt.savefig(save_path)
        print(f"Saved backlog count plot as '{save_path}'.")
        plt.close()
    else:
        print(f"Column '{backlog_col}' not found for backlog count plot.")

    # Example: Relationship between Family Income and CGPA
    if "FamilyIncome" in df.columns and "cgpa_scaled" in df.columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(
            data=df,
            x="FamilyIncome",
            y="cgpa_scaled",
            palette="Pastel1"
        )
        plt.title("CGPA (Scaled) by Family Income")
        plt.xlabel("Family Income")
        plt.ylabel("CGPA (Scaled)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        save_path = os.path.join(image_folder, "cgpa_scaled_by_family_income.png")
        plt.savefig(save_path)
        print(f"Saved CGPA (Scaled) by family income box plot as '{save_path}'.")
        plt.close()
    else:
        print("Required columns for CGPA (Scaled) by family income plot not found.")

    # Example: Distribution of Engineering Course Branch
    branch_col = "EngineeringCourseBranch"
    if branch_col in df.columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x=branch_col, palette="Set2")
        plt.title("Distribution of Engineering Course Branches")
        plt.xlabel("Engineering Course Branch")
        plt.ylabel("Count")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        save_path = os.path.join(image_folder, "distribution_engineering_branch.png")
        plt.savefig(save_path)
        print(f"Saved engineering branch distribution plot as '{save_path}'.")
        plt.close()
    else:
        print(f"Column '{branch_col}' not found for distribution plot.")

    # Example: CGPA by Engineering Course Branch
    if branch_col in df.columns and "cgpa_scaled" in df.columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(
            data=df,
            x=branch_col,
            y="cgpa_scaled",
            palette="Set3"
        )
        plt.title("CGPA (Scaled) by Engineering Course Branch")
        plt.xlabel("Engineering Course Branch")
        plt.ylabel("CGPA (Scaled)")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        save_path = os.path.join(image_folder, "cgpa_scaled_by_engineering_branch.png")
        plt.savefig(save_path)
        print(f"Saved CGPA (Scaled) by engineering branch box plot as '{save_path}'.")
        plt.close()
    else:
        print("Required columns for CGPA (Scaled) by engineering branch plot not found.")

    # --------------------------------------------------------------------------
    # 12. Collate All Images into a Single PDF
    # --------------------------------------------------------------------------
    collate_images_to_pdf(image_folder=image_folder, output_pdf="EDA_Collated.pdf")

    # --------------------------------------------------------------------------
    # End of EDA
    # --------------------------------------------------------------------------
    print("Exploratory Data Analysis Complete. Plots have been generated and saved in the 'images' folder.")

# Example usage:
exploratory_analysis('dataset/student-dataset.csv')