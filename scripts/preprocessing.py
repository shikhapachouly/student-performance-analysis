import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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