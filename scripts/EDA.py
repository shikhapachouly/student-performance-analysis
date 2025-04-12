import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from PIL import Image
import joblib


# ---------------------------- Utility Functions ---------------------------- #

def create_image_folder(folder_name="images"):
    """Create directory for storing visualization images"""
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created visualization folder: {folder_name}")
    else:
        print(f"Visualization folder {folder_name} already exists")


def collate_images_to_pdf(image_folder="images", output_pdf="EDA_Report.pdf"):
    """Combine all generated images into a PDF report"""
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])

    if not image_files:
        print("No images found for PDF creation")
        return

    pdf_path = os.path.join(image_folder, output_pdf)
    c = canvas.Canvas(pdf_path, pagesize=A4)

    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        try:
            with Image.open(img_path) as img:
                img_width, img_height = img.size
                aspect = img_height / img_width

                # Calculate dimensions to fit A4
                max_width = A4[0] - 100
                max_height = A4[1] - 100
                display_width = min(max_width, img_width)
                display_height = display_width * aspect

                if display_height > max_height:
                    display_height = max_height
                    display_width = display_height / aspect

                x = (A4[0] - display_width) / 2
                y = (A4[1] - display_height) / 2

                c.drawImage(img_path, x, y, width=display_width, height=display_height)
                c.showPage()
        except Exception as e:
            print(f"Error adding {img_file} to PDF: {str(e)}")

    c.save()
    print(f"PDF report saved to: {pdf_path}")


# -------------------------- Core Preprocessing Class ------------------------ #

class DataPreprocessor:
    """End-to-end data preprocessing pipeline"""

    def __init__(self, target_column=None):
        self.target_column = target_column
        self.preprocessor = None
        self.feature_names = None

    def _create_pipelines(self, numerical_cols, categorical_cols):
        """Build sklearn pipelines for different data types"""

        # Numerical data pipeline
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Categorical data pipeline
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        return ColumnTransformer([
            ('num', num_pipeline, numerical_cols),
            ('cat', cat_pipeline, categorical_cols)
        ])

    def preprocess(self, df):
        """Execute complete preprocessing workflow"""

        # Initial cleaning
        df = self._clean_data(df)

        # Identify feature types
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

        # Remove target from features if specified
        if self.target_column and self.target_column in df.columns:
            X = df.drop(columns=[self.target_column])
            y = df[self.target_column]
        else:
            X = df
            y = None

        # Build and execute pipeline
        self.preprocessor = self._create_pipelines(numerical_cols, categorical_cols)
        X_processed = self.preprocessor.fit_transform(X)

        # Get feature names
        num_features = numerical_cols
        cat_features = self.preprocessor.named_transformers_['cat'] \
            .named_steps['encoder'].get_feature_names_out(categorical_cols)
        self.feature_names = np.concatenate([num_features, cat_features])

        return pd.DataFrame(X_processed, columns=self.feature_names), y

    def _clean_data(self, df):
        """Initial data cleaning steps"""

        # Remove unique identifiers
        df = df.drop(columns=['Timestamp', 'Username', 'NameOfStudent'], errors='ignore')

        # Clean gender values
        if 'GenderOfStudent' in df.columns:
            df = df[df['GenderOfStudent'].isin(['Male', 'Female'])]

        # Convert percentages
        for col in ['10thPercentage', '12thPercentage']:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace('%', ''),
                    errors='coerce'
                )

        return df


# ------------------------ Visualization Functions -------------------------- #

def generate_eda_visualizations(df, image_folder):
    """Create exploratory data analysis visualizations"""

    # Numerical distributions
    plt.figure(figsize=(12, 8))
    for i, col in enumerate(df.select_dtypes(include=np.number).columns[:4]):
        plt.subplot(2, 2, i + 1)
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.savefig(os.path.join(image_folder, 'numerical_distributions.png'))
    plt.close()

    # Categorical distributions
    plt.figure(figsize=(12, 8))
    for i, col in enumerate(df.select_dtypes(exclude=np.number).columns[:4]):
        plt.subplot(2, 2, i + 1)
        sns.countplot(x=df[col])
        plt.title(f'Distribution of {col}')
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(image_folder, 'categorical_distributions.png'))
    plt.close()

    # Correlation matrix
    plt.figure(figsize=(14, 12))
    sns.heatmap(df.select_dtypes(include=np.number).corr(),
                annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(image_folder, 'correlation_matrix.png'))
    plt.close()


# -------------------------- Main Workflow Function ------------------------- #

def run_full_analysis(csv_path, target_column=None):
    """Execute complete data analysis and preprocessing workflow"""

    # Initialize paths and directories
    create_image_folder()

    try:
        # Load raw data
        raw_df = pd.read_csv(csv_path)
        print(f"Successfully loaded data from {csv_path}")
        print(f"Original shape: {raw_df.shape}")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return

    # Generate EDA visualizations
    generate_eda_visualizations(raw_df, 'images')

    # Initialize and run preprocessor
    preprocessor = DataPreprocessor(target_column=target_column)
    processed_df, target = preprocessor.preprocess(raw_df)

    # Save processed data
    processed_path = os.path.join(os.path.dirname(csv_path), 'processed_data_updated.csv')
    processed_df.to_csv(processed_path, index=False)
    print(f"Saved processed data to: {processed_path}")

    # Save preprocessing pipeline
    joblib.dump(preprocessor.preprocessor, 'preprocessing_pipeline.joblib')
    print("Saved preprocessing pipeline to: preprocessing_pipeline.joblib")

    # Generate processed data visualizations
    generate_eda_visualizations(processed_df, 'images')

    # Create final report
    collate_images_to_pdf()

    return processed_df


# ------------------------------ Execution ---------------------------------- #

if __name__ == "__main__":
    # Configuration
    DATA_PATH = "../dataset/student-dataset.csv"  # Update with your path


    # Execute complete workflow
    processed_data = run_full_analysis(DATA_PATH)
