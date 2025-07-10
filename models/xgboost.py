from xgboost import XGBRegressor
import joblib
import os

class XGBWarmStart(XGBRegressor):
    def __init__(self, model_path="xgb_model.pkl", **kwargs):
        super().__init__(**kwargs)
        self.model_path = model_path

        if os.path.exists(self.model_path):
            loaded_model = joblib.load(self.model_path)
            self._Booster = loaded_model.get_booster()
            print(f"[XGBWarmStart] Loaded model from {self.model_path}")

    def train_batch(self, X, y):
        try:
            booster = self.get_booster()
            self.fit(X, y, xgb_model=booster)
        except Exception:
            self.fit(X, y)

        joblib.dump(self, self.model_path)
        print(f"[XGBWarmStart] Trained & saved to {self.model_path}")

    def predict_batch(self, X):
        preds = self.predict(X)

        return preds
