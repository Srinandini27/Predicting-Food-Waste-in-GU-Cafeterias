import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

df = pd.read_csv('../data/food_waste_records.csv')

df['Waste_Rate'] = df['Waste_kg'] / df['Meals_Prepared']
X = df[['Meals_Prepared', 'Meals_Served']]
y = df['Waste_kg']

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, 'food_waste_predictor.pkl')
print("Model trained and saved.")