from lightgbm import LGBMRegressor
import joblib
import os

class LGBMWarmStart(LGBMRegressor):
    def __init__(self, model_path="lgbm_model.pkl", **kwargs):
        super().__init__(**kwargs)
        self.model_path = model_path

        if os.path.exists(self.model_path):
            loaded_model = joblib.load(self.model_path)
            self._Booster = loaded_model.booster_
            print(f"[LGBMWarmStart] Loaded model from {self.model_path}")

    def train(self, X, y):
        try:
            booster = self.booster_
            self.fit(X, y, init_model=booster)
        except Exception:
            self.fit(X, y)

        joblib.dump(self, self.model_path)
        print(f"[LGBMWarmStart] Trained & saved to {self.model_path}")

    def predict_batch(self, X):
        preds = self.predict(X)
        
        return preds
