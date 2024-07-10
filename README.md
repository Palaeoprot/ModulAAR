# MoDuLAAR - Modeling and Analyzing Racemization in Amino Acids

MoDuLAAR is a comprehensive toolkit designed to model and analyze racemization processes in amino acids. This project integrates data processing, dehydration analysis, racemization simulation, parameter optimization, and result visualization to provide a robust framework for studying amino acid kinetics.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Overview

The primary goal of MoDuLAAR is to facilitate the analysis and modeling of amino acid racemization. The toolkit includes:
- Data processing and cleaning
- Dehydration analysis
- Racemization simulation
- Parameter optimization
- Comprehensive result visualization

## Features

- **Data Processing**: Clean and preprocess raw data for further analysis.
- **Dehydration Analysis**: Analyze the effects of dehydration on amino acid concentrations and D/L ratios.
- **Racemization Simulation**: Simulate hydrolysis and racemization processes using detailed kinetic models.
- **Parameter Optimization**: Optimize model parameters using gradient descent with early stopping.
- **Result Visualization**: Generate detailed plots to visualize analysis and simulation results.

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/MoDuLAAR.git
    cd MoDuLAAR
    ```

2. **Install dependencies:**

    Ensure you have the necessary Python libraries installed. You can use the following command to install them:

    ```bash
    pip install -r requirements.txt
    ```

3. **Set up Google Drive:**

    Since the program integrates with Google Drive for data storage, ensure you have the necessary permissions and setup in place. Refer to the Google Colab documentation for more details.

## Usage

1. **Mount Google Drive:**

    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

2. **Run the main notebook:**

    Open `main.ipynb` in Google Colab or Jupyter Notebook and run the cells sequentially. This will execute the entire workflow, including data processing, dehydration analysis, racemization simulation, parameter optimization, and result visualization.

## Project Structure

- `data_processor.ipynb`: Handles data loading, cleaning, and preprocessing.
- `dehydration_analyzer.ipynb`: Analyzes the effects of dehydration on amino acid concentrations and D/L ratios.
- `racemization_simulator.ipynb`: Simulates hydrolysis and racemization processes.
- `parameter_optimizer.ipynb`: Optimizes model parameters using gradient descent with early stopping.
- `result_visualizer.ipynb`: Visualizes the results of the analysis and simulations.
- `main.ipynb`: Orchestrates the execution of the entire workflow.
- `Dictionaries/Colours/colors.json`: Contains color mappings for amino acid visualization.

### Data Processor

```python
# data_processor.ipynb
import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self):
        self.raw_data = None
        self.processed_data = None
        self.real_DL = None
        self.amino_acids = ['Asx', 'Glx', 'Ser', 'Ala', 'Val', 'Phe', 'Ile']

    def load_data(self, source, is_gsheet=False):
        try:
            if is_gsheet:
                sheet_id = source['sheet_id']
                gid = source['gid']
                export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
                self.raw_data = pd.read_csv(export_url)
                print(f"Data loaded successfully from Google Sheet")
            else:
                self.raw_data = pd.read_csv(source)
                print(f"Data loaded successfully from {source}")
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            self.raw_data = None
        return self.raw_data

    def clean_data(self):
        if self.raw_data is None:
            print("No data available to clean.")
            return None

        self.processed_data = self.raw_data.copy()

        # Rename columns for consistency
        column_mapping = {
            'time (h)': 'time',
            'temp': 'temp (°C)',
            'Pre-heat bleach time': 'Pre-heat bleach time (h)'
        }
        self.processed_data.rename(columns=column_mapping, inplace=True)

        # Strip whitespace and replace non-numeric characters in relevant columns
        numeric_columns = [
            '[Asx]', '[Glx]', '[Ser]', '[Ala]', '[Val]', '[Phe]', '[Ile]',
            'Asx D/L', 'Glx D/L', 'Ser D/L', 'Ala D/L', 'Val D/L', 'Phe D/L', 'Ile D/L'
        ]

        for col in self.processed_data.columns:
            if self.processed_data[col].dtype == 'object':
                self.processed_data[col] = self.processed_data[col].str.strip()

        for col in numeric_columns:
            self.processed_data[col] = self.processed_data[col].astype(str).str.replace(r'[^\d.-]', '', regex=True)
            self.processed_data[col] = pd.to_numeric(self.processed_data[col], errors='coerce')

        # Drop rows with NaN values in numeric columns
        self.processed_data.dropna(subset=numeric_columns, inplace=True)

        print("Data cleaning completed.")
        return self.processed_data

    def calculate_real_DL(self):
        if self.processed_data is None:
            print("No processed data available. Please clean data first.")
            return None

        faa_df = self.processed_data[self.processed_data['sample'] == 'FAA']
        thaa_df = self.processed_data[self.processed_data['sample'] == 'THAA']

        self.real_DL = self.processed_data[['temp (°C)', 'time', 'pH']].drop_duplicates().reset_index(drop=True)

        for aa in self.amino_acids:
            aa_data = self._calculate_amino_acid_distribution(aa, faa_df, thaa_df)
            self.real_DL = pd.merge(self.real_DL, aa_data, on=['temp (°C)', 'time'], how='outer')

        self.real_DL['temp (K)'] = self.real_DL['temp (°C)'] + 273.15

        print("real_DL calculation completed.")
        return self.real_DL

    def _calculate_amino_acid_distribution(self, amino_acid, faa_df, thaa_df):
        aa_data = pd.DataFrame()

        # Calculate stats for FAA and THAA
        faa_stats = faa_df.groupby(['temp (°C)', 'time'])[[f'[{amino_acid}]', f'{amino_acid} D/L']].agg(['mean', 'std', 'count'])
        thaa_stats = thaa_df.groupby(['temp (°C)', 'time'])[[f'[{amino_acid}]', f'{amino_acid} D/L']].agg(['mean', 'std', 'count'])

        # Merge FAA and THAA data
        aa_data = pd.merge(faa_stats, thaa_stats, left_index=True, right_index=True, suffixes=('_FAA', '_THAA'))
        aa_data = aa_data.reset_index()

        # Flatten column names
        aa_data.columns = ['_'.join(col).strip() for col in aa_data.columns.values]

        # Rename columns
        column_mapping = {
            'temp (°C)_': 'temp (°C)',
            'time_': 'time',
            f'[{amino_acid}]_FAA_mean': f'{amino_acid}_Conc_FAA_Mean',
            f'[{amino_acid}]_FAA_std': f'{amino_acid}_Conc_FAA_Std',
            f'[{amino_acid}]_FAA_count': f'{amino_acid}_Conc_FAA_Count',
            f'{amino_acid} D/L_FAA_mean': f'{amino_acid}_D/L_FAA_Mean',
            f'{amino_acid} D/L_FAA_std': f'{amino_acid}_D/L_FAA_Std',
            f'{amino_acid} D/L_FAA_count': f'{amino_acid}_D/L_FAA_Count',
            f'[{amino_acid}]_THAA_mean': f'{amino_acid}_Conc_THAA_Mean',
            f'[{amino_acid}]_THAA_std': f'{amino_acid}_Conc_THAA_Std',
            f'[{amino_acid}]_THAA_count': f'{amino_acid}_Conc_THAA_Count',
            f'{amino_acid} D/L_THAA_mean': f'{amino_acid}_D/L_THAA_Mean',
            f'{amino_acid} D/L_THAA_std': f'{amino_acid}_D/L_THAA_Std',
            f'{amino_acid} D/L_THAA_count': f'{amino_acid}_D/L_THAA_Count'
        }
        aa_data = aa_data.rename(columns=column_mapping)

        # Calculate BAA concentrations
        aa_data[f'{amino_acid}_Conc_BAA_Mean'] =

