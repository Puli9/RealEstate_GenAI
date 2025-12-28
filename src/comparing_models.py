import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression  # ✅ Changed to LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class PriceModel:
    def __init__(self):
        self.models = {}  # Store multiple models
        self.model_predictions = {}  # Store predictions per model
        self.feature_cols = {
            'numeric': ['size_sqft', 'age_yrs'],
            'categorical': ['locality', 'property_type']
        }
        self.y_mean = None
        self.preprocessor = None
        self.best_model = None

    def _rmse_metric(self, y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    def train(self, csv_path: str = "data/Hyderbad_House_price.csv"):
        df = pd.read_csv(csv_path).drop_duplicates().reset_index(drop=True)

        # ✅ FIXED: Create age_yrs FIRST if missing
        if "age_yrs" not in df.columns:
            rng = np.random.default_rng(42)
            df["age_yrs"] = np.round(rng.uniform(0.1, 5.0, size=len(df)), 1)

        # Clean columns
        for col in ["price(L)", "size_sqft", "age_yrs"] + self.feature_cols['categorical']:
            if col in df.columns:
                if col in ["price(L)", "size_sqft", "age_yrs"]:
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")

        # ✅ NOW all required columns exist
        all_features = self.feature_cols['numeric'] + self.feature_cols['categorical']
        required_cols = ["price(L)"] + all_features
        
        df_model = df.dropna(subset=required_cols)
        
        # Outlier removal
        mask = pd.Series(True, index=df_model.index)
        for col in ["price(L)", "size_sqft"]:
            q1, q3 = df_model[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            mask &= df_model[col].between(q1 - 1.5*iqr, q3 + 1.5*iqr)
        
        df_model_filt = df_model[mask].reset_index(drop=True)

        X = df_model_filt[all_features]
        y = df_model_filt["price(L)"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Pipeline with preprocessing
        self.preprocessor = ColumnTransformer([
            ('num', StandardScaler(), self.feature_cols['numeric']),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), 
             self.feature_cols['categorical'])
        ])

        X_train_proc = self.preprocessor.fit_transform(X_train)
        X_test_proc = self.preprocessor.transform(X_test)

        # ✅ MODEL 1: XGBoost (Tree-based)
        print("Training XGBoost...")
        xgb_model = XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42, verbosity=0
        )
        xgb_model.fit(X_train_proc, y_train)
        xgb_preds = xgb_model.predict(X_test_proc)
        xgb_rmse = self._rmse_metric(y_test, xgb_preds)
        xgb_mae = mean_absolute_error(y_test, xgb_preds)
        
        # ✅ MODEL 2: Linear Regression (Simple baseline)
        print("Training Linear Regression...")
        linear_model = LinearRegression()  # ✅ Changed to LinearRegression
        linear_model.fit(X_train_proc, y_train)
        linear_preds = linear_model.predict(X_test_proc)
        linear_rmse = self._rmse_metric(y_test, linear_preds)
        linear_mae = mean_absolute_error(y_test, linear_preds)

        # Store models and metrics
        self.models = {
            'XGBoost': {
                'model': xgb_model,
                'rmse': round(float(xgb_rmse), 2),
                'mae': round(float(xgb_mae), 2),
                'confidence': round(max(0.0, min(1.0, 1.0 - (xgb_rmse / 100))), 2)
            },
            'Linear_Regression': {  # ✅ Updated name
                'model': linear_model,
                'rmse': round(float(linear_rmse), 2),
                'mae': round(float(linear_mae), 2),
                'confidence': round(max(0.0, min(1.0, 1.0 - (linear_rmse / 100))), 2)
            }
        }
        
        # Select best model (lowest RMSE/highest confidence)
        self.best_model = min(self.models.keys(), 
                            key=lambda k: self.models[k]['rmse'])
        
        self.y_mean = float(y_train.mean())
        print(f"\nModel Comparison:")
        print(f"  XGBoost:           RMSE={self.models['XGBoost']['rmse']}L, Confidence={self.models['XGBoost']['confidence']}")
        print(f"  Linear Regression: RMSE={self.models['Linear_Regression']['rmse']}L, Confidence={self.models['Linear_Regression']['confidence']}")
        print(f"  Best Model:        {self.best_model}")

    def predict_for_property(self, features: dict):
        all_features = self.feature_cols['numeric'] + self.feature_cols['categorical']
        
        # Fill defaults
        for col in all_features:
            if col not in features:
                if col in self.feature_cols['numeric']:
                    features[col] = 0.0
                else:
                    features[col] = 'Unknown'
        
        X_df = pd.DataFrame([features])[all_features]
        X_proc = self.preprocessor.transform(X_df)
        
        # Get predictions from ALL models
        predictions = {}
        for model_name, model_info in self.models.items():
            pred = round(float(model_info['model'].predict(X_proc)[0]), 2)
            predictions[model_name] = {
                'predicted_price_L': pred,
                'rmse': model_info['rmse'],
                'mae': model_info['mae'],
                'confidence': model_info['confidence']
            }
        
        # Select BEST model prediction (highest confidence)
        best_model_name = self.best_model
        best_pred = predictions[best_model_name]
        
        return {
            'all_predictions': predictions,  # ✅ ALL models
            'best_model': best_model_name,   # ✅ Which one won
            'final_prediction': best_pred,   # ✅ Best prediction
            'y_mean': self.y_mean
        }

# Test
if __name__ == "__main__":
    pm = PriceModel()
    pm.train()
    
    features = {
        "size_sqft": 1200,
        "age_yrs": 2,
        "locality": "Chandanagar",
        "property_type": "Apartment"
    }
    
    result = pm.predict_for_property(features)
    print("\nPREDICTION RESULTS:")
    print(json.dumps(result, indent=2, default=str))
