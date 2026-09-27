import pandas as pd
import numpy as np
import lightgbm as lgb
import shap

class TurismRadarRegressor:
    """
    Fits baseline municipal leisure spending regressor and isolates 
    border trade / airport transit spillovers as positive residuals.
    """
    def __init__(self, alpha: float = 1.0):
        self.features = ["G_overnight_stays", "P_population", "C_bed_capacity", "F_border", "F_airport"]
        self.model = lgb.LGBMRegressor(
            objective="huber",
            alpha=alpha,
            n_estimators=50,
            learning_rate=0.05,
            min_child_samples=1,  # Allows splits on small mock datasets (< 20 rows)
            random_state=42
        )
        self.explainer = None

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.features]
        # Scale target to Millions SEK for numerical stability & readable SHAP plots
        y_actual_m = df["S_total_spend"] / 1e6
        
        # Fit Huber robust regressor
        self.model.fit(X, y_actual_m)
        
        results = df.copy()
        # 1. Predict baseline leisure spending (in SEK)
        y_pred_m = np.maximum(0, self.model.predict(X))
        results["Y_predicted_leisure"] = y_pred_m * 1e6
        
        # 2. Compute residual spillover noise (R_i = Y_actual - Y_predicted)
        results["residual_spillover"] = results["S_total_spend"] - results["Y_predicted_leisure"]
        
        # 3. Compute per-capita comparisons
        results["raw_per_capita"] = results["S_total_spend"] / results["P_population"]
        results["adjusted_per_capita"] = results["Y_predicted_leisure"] / results["P_population"]
        
        # 4. Compute rank shifts
        results["raw_rank"] = results["raw_per_capita"].rank(ascending=False).astype(int)
        results["adjusted_rank"] = results["adjusted_per_capita"].rank(ascending=False).astype(int)
        results["rank_shift"] = results["raw_rank"] - results["adjusted_rank"]
        
        # 5. Extract SHAP values on scaled inputs
        self.explainer = shap.TreeExplainer(self.model)
        self.shap_values = self.explainer(X)
        
        return results

    def get_shap_explanation(self, df: pd.DataFrame) -> shap.Explanation:
        """Returns SHAP values object for waterfall visualizations."""
        X = df[self.features]
        return self.explainer(X)