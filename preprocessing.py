# preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def preprocess_data(df):
    df = df.copy()

    for col in df.columns:
        if 'ID' in col or 'id' in col:
            df.drop(col, axis=1, inplace=True)

    if 'Dt_Customer' in df.columns:
        df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], errors='coerce')
        df['Customer_Since_Days'] = (pd.Timestamp('today') - df['Dt_Customer']).dt.days
        df.drop('Dt_Customer', axis=1, inplace=True)

    if 'Year_Birth' in df.columns:
        df['Age'] = 2025 - df['Year_Birth']
        df.drop('Year_Birth', axis=1, inplace=True)

    # Total spend (if columns exist)
    spend_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 
                  'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    if all(col in df.columns for col in spend_cols):
        df['TotalSpend'] = df[spend_cols].sum(axis=1)

    # Impute missing
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        imputer = SimpleImputer(strategy='median')
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    else:
        raise ValueError("No numeric columns found for imputation. Please check input file.")

    # Encode categoricals
    cat_cols = df.select_dtypes(include='object').columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Scale numeric only (after dummies)
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df[numeric_cols])

    return df, scaled_array
