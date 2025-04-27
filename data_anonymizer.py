# src/data_anonymizer.py

import pandas as pd
import numpy as np
import random
from faker import Faker
import os

class DataAnonymizer:
    """
    A class for anonymizing datasets using Differential Privacy for numerical columns
    and K-Anonymity strategies for string columns (suppression, generalization, synthetic replacement).
    """

    def __init__(self):
        self.fake = Faker()
        self.df = None
        self.original_df = None

    def load_data(self, filepath):
        """Load dataset from CSV file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File '{filepath}' not found. Please check the path.")
        
        self.df = pd.read_csv(filepath)
        self.original_df = self.df.copy()
        print(f"Dataset loaded successfully with {self.df.shape[0]} rows and {self.df.shape[1]} columns.")
        return self.df

    def apply_differential_privacy(self, numerical_columns, epsilon=1.0):
        """Apply Differential Privacy to numerical columns."""
        for col in numerical_columns:
            if col in self.df.columns:
                sensitivity = self.df[col].max() - self.df[col].min()
                scale = sensitivity / epsilon
                noise = np.random.laplace(loc=0, scale=scale, size=len(self.df))
                self.df[col] = self.df[col] + noise
                self.df[col] = self.df[col].round(2)
                print(f"Differential Privacy applied on '{col}' with epsilon={epsilon}.")
            else:
                print(f"Warning: Column '{col}' not found in dataset.")
        return self.df

    def apply_k_anonymity(self, quasi_identifiers, k=2, strategy='generalization'):
        """Apply K-Anonymity to string columns."""
        if strategy not in ['suppression', 'generalization', 'synthetic']:
            raise ValueError("Invalid strategy. Choose from: suppression, generalization, synthetic")
        
        groups = self.df.groupby(quasi_identifiers, dropna=False).groups
        small_groups = [group for group in groups if len(groups[group]) < k]

        for group in small_groups:
            indices = groups[group]
            for col in quasi_identifiers:
                if strategy == 'suppression':
                    # Suppress: replace value with matching number of *
                    self.df.loc[indices, col] = self.df[col].apply(lambda x: '*' * len(str(x)))

                elif strategy == 'generalization':
                    # Generalize: keep first few characters, hide rest with *
                    self.df.loc[indices, col] = self.df[col].apply(lambda x: str(x)[:2] + '*' * (len(str(x)) - 2))

                elif strategy == 'synthetic':
                    # Synthetic Replacement: generate fake realistic data
                    if 'zip' in col.lower():
                        self.df.loc[indices, col] = [self.fake.zipcode() for _ in range(len(indices))]
                    elif 'gender' in col.lower():
                        self.df.loc[indices, col] = [random.choice(['Male', 'Female']) for _ in range(len(indices))]
                    elif 'occupation' in col.lower() or 'job' in col.lower():
                        self.df.loc[indices, col] = [self.fake.job() for _ in range(len(indices))]
                    else:
                        self.df.loc[indices, col] = [self.fake.word() for _ in range(len(indices))]

        print(f"K-Anonymity ({strategy}) applied with k={k}.")
        return self.df

    def save_anonymized_data(self, output_path):
        """Save anonymized data to a CSV file."""
        self.df.to_csv(output_path, index=False)
        print(f"Anonymized dataset saved as '{output_path}'.")
        return output_path

    def get_anonymization_report(self):
        """Generate and display an anonymization report comparing original and anonymized data."""
        report = []
        for col in self.original_df.columns:
            if col in self.df.columns:
                original_unique = self.original_df[col].nunique()
                anonymized_unique = self.df[col].nunique()
                privacy_gain = original_unique - anonymized_unique
                report.append((col, original_unique, anonymized_unique, privacy_gain))

        print("\nAnonymization Report:")
        print(f"{'Column':<20} {'Original Unique':<20} {'Anonymized Unique':<20} {'Privacy Gain'}")
        print("-" * 75)
        for col, original, anonymized, gain in report:
            print(f"{col:<20} {original:<20} {anonymized:<20} {gain}")
        print("-" * 75)

if __name__ == "__main__":
    anonymizer = DataAnonymizer()

    try:
        # Step 1: Load dataset
        file_path = input("Enter the path to your dataset (CSV file): ")
        anonymizer.load_data(file_path)

        # Step 2: Select columns and settings
        numerical_cols = input("Enter numerical columns (comma-separated): ").strip().split(',')
        numerical_cols = [col.strip() for col in numerical_cols]

        epsilon = float(input("Enter privacy level (epsilon) for numerical columns (e.g., 0.5, 1.0): "))

        quasi_ids = input("Enter string/quasi-identifier columns (comma-separated): ").strip().split(',')
        quasi_ids = [col.strip() for col in quasi_ids]

        k = int(input("Enter value of k for K-Anonymity (e.g., 2, 3): "))

        print("\nChoose K-Anonymity strategy:")
        print("1 - Suppression")
        print("2 - Generalization")
        print("3 - Synthetic Replacement")
        strategy_choice = input("Enter choice (1/2/3): ")
        strategy_mapping = {'1': 'suppression', '2': 'generalization', '3': 'synthetic'}
        strategy = strategy_mapping.get(strategy_choice, 'generalization')

        # Step 3: Apply anonymization
        anonymizer.apply_differential_privacy(numerical_cols, epsilon)
        anonymizer.apply_k_anonymity(quasi_ids, k, strategy)

        # Step 4: Save anonymized data and show report
        anonymizer.save_anonymized_data('data/anonymized_data.csv')
        anonymizer.get_anonymization_report()

        print("\nAnonymization completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
