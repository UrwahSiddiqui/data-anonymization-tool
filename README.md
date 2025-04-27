# Data Anonymization Tool

This project anonymizes sensitive datasets using:
- Differential Privacy (for numerical columns)
- K-Anonymity Strategies (for string columns: Suppression, Generalization, Synthetic Replacement)

## Features
- User-defined privacy level for numerical columns (epsilon control).
- User choice of K-Anonymity strategy for string columns.
- Anonymization report comparing original and anonymized data.

## How to Run
1. Install dependencies:
    ```
    pip install pandas numpy faker
    ```

2. Run the script:
    ```
    python src/data_anonymizer.py
    ```

3. Follow the prompts to provide:
   - Dataset path
   - Columns to anonymize
   - Privacy settings

4. Output:
   - Anonymized dataset saved in `data/anonymized_data.csv`
   - Anonymization report printed on console.

## Folder Structure
