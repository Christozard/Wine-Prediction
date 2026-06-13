import pandas as pd 
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.inspection import PartialDependenceDisplay
import numpy as np
import shap
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer 
from sklearn.ensemble import RandomForestRegressor
import joblib
import json 

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from feature_engineering import (
    engineer_features,
    InteractionTransformer
)

df = pd.read_csv("../datasets/winequality-red.csv")

X = df.drop("quality",axis=1)
y = df["quality"]

X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2,random_state=42)

model = XGBRegressor(
    n_estimators=100,
    random_state=42
)
model.fit(X_train,y_train)

explainer = shap.Explainer(model)
shap_interaction_values = explainer.shap_interaction_values(X_train)

mean_abs_interactions = np.abs(shap_interaction_values).mean(axis=0)
feature_names = X_train.columns 
df_interactions = pd.DataFrame(mean_abs_interactions, columns = feature_names, index = feature_names)
df_interactions
upper_tri_indices = np.triu_indices_from(df_interactions, k=1)
top_3 = pd.DataFrame({
    'Feature_1': df_interactions.columns[upper_tri_indices[1]],
    'Feature_2': df_interactions.index[upper_tri_indices[0]],
    'Interaction_Strength': df_interactions.values[upper_tri_indices]
}).sort_values(by='Interaction_Strength', ascending=False).head(3)

interactions = []

for _, row in top_3.iterrows():
    interactions.append((row["Feature_1"],row["Feature_2"]))


eng_features = FunctionTransformer(engineer_features)

with open("../model/params.json","r") as file:
    params = json.load(file)

pipeline = Pipeline([
    ("features", eng_features),
    ("interactions", InteractionTransformer(interactions)),
    ("model", RandomForestRegressor(**params))
])

model = pipeline.fit(X,y)

joblib.dump(model,"../model/rf_pipeline.pkl")