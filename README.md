# House Price Prediction Model using LightGBM

This project implements a machine learning model to predict house prices using various housing features. The model uses LightGBM regression and includes comprehensive data preprocessing, analysis, and evaluation steps.

## Features
* Data preprocessing and cleaning
* Exploratory Data Analysis (EDA)
* Feature engineering and selection
* Model training and evaluation
* Visualization of results
* Feature importance analysis

## Dataset
The project uses the house price dataset with the following key features:
* SalePrice (Target variable)
* LotArea
* YearBuilt
* OverallQual
* GarageArea
* TotalBsmtSF
* GrLivArea
* FullBath
* And many other housing characteristics

## Requirements
```
numpy
pandas
seaborn
matplotlib
scikit-learn
lightgbm
```

## Project Structure
The project follows these main steps:

1. **Data Loading and Initial Exploration**
   * Reading the dataset
   * Basic data overview
   * Memory usage analysis
   * Missing value detection
   * Duplicate value check

2. **Data Preprocessing**
   * Handling missing values
   * Outlier detection and treatment
   * Categorical variable encoding
   * Feature scaling using StandardScaler
   * Log transformation of target variable

3. **Exploratory Data Analysis**
   * Statistical summaries
   * Correlation analysis
   * Distribution analysis
   * Target variable analysis
   * Feature relationship studies

4. **Model Building**
   * Train-test split (80-20)
   * LightGBM implementation
   * Model training and prediction
   * Feature selection

5. **Model Evaluation**
   * R-squared Score
   * Root Mean Squared Error (RMSE)
   * Mean Absolute Error (MAE)
   * Feature importance visualization

## Key Functions
* `check_df()`: Provides comprehensive dataframe information
* `grab_col_names()`: Categorizes variables based on their types
* `num_summary()`: Generates numerical summaries with optional plotting
* `missing_values_table()`: Analyzes missing values
* `rare_analyser()`: Analyzes rare categories in categorical variables
* `high_correlated_cols()`: Identifies highly correlated features
* `plot_importance()`: Visualizes feature importance

## Installation
1. Clone the repository:
```bash
git clone [your-repository-url]
```

2. Install required packages:
```bash
pip install numpy pandas seaborn matplotlib scikit-learn lightgbm
```

## Usage
1. Ensure your data file is in the correct location
2. Update the file path in the code:
```python
df = pd.read_csv("path_to_your_data/HousePrice_train.csv")
```

3. Run the script:
```python
python house_price_prediction.py
```

## Model Performance
The LightGBM model provides:
* R-squared Score for regression accuracy
* RMSE and MAE metrics
* Visual representation of feature importance
* Detailed prediction analysis

## Data Preprocessing Steps
1. **Missing Value Treatment**
   * Filling missing values based on domain knowledge
   * Using median/mode imputation where appropriate
   * Creating new categories for missing values in categorical variables

2. **Feature Engineering**
   * Handling rare categories
   * Binary encoding for categorical variables
   * One-hot encoding for nominal variables
   * Scaling numerical features

3. **Outlier Treatment**
   * IQR method for outlier detection
   * Capping outliers at determined thresholds
   * Special handling for price outliers

## Contributing
Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License
[Add your license information here]

## Author
[Your Name]
