import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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