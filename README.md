# Global-emissions-data-analysis
# Global Emissions Data Analysis

This project analyzes the global emissions dataset to explore the relationship between emissions and per capita values across different countries. The dataset includes various emission factors, such as cement production, energy usage, and more.

## Key Steps:
1. **Data Cleaning**:
   - Dropped missing values and duplicates.
   - Renamed columns for uniformity.

2. **Exploratory Data Analysis (EDA)**:
   - Visualized the distribution of countries and emissions data using histograms, violin plots, and scatter plots.
   - Analyzed correlations between different features using heatmaps.

3. **Normalization**:
   - Scaled the dataset using standardization and min-max normalization.

4. **Machine Learning**:
   - Split the dataset into training and testing sets.
   - Applied linear regression to predict `Per_Capita` emissions values.

## Tools Used:
- **Python**
- **Pandas**
- **Matplotlib**
- **Seaborn**
- **Plotly**
- **NumPy**

## How to Run:
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install pandas matplotlib seaborn plotly numpy
