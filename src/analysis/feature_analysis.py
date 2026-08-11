"""
SHAP-based feature analysis and model interpretability.
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)


class FeatureAnalyzer:
    """SHAP-based feature importance and interaction analysis."""

    def __init__(self, model, X: pd.DataFrame, model_name: str = "model"):
        self.model = model
        self.X = X
        self.model_name = model_name
        self.explainer = None
        self.shap_values = None

    def compute_shap_values(self, max_samples: int = 1000):
        """Compute SHAP values for the model."""
        logger.info(f"Computing SHAP values for {self.model_name}...")

        # Use TreeExplainer for tree-based models
        try:
            self.explainer = shap.TreeExplainer(self.model)
            X_sample = self.X.iloc[:max_samples] if len(self.X) > max_samples else self.X
            self.shap_values = self.explainer.shap_values(X_sample)
        except Exception as e:
            logger.warning(f"TreeExplainer failed: {e}. Using KernelExplainer.")
            X_sample = self.X.iloc[:min(100, len(self.X))]
            self.explainer = shap.KernelExplainer(self.model.predict, X_sample)
            self.shap_values = self.explainer.shap_values(X_sample)

        return self.shap_values

    def plot_feature_importance(self, top_n: int = 30, save_path: Optional[str] = None):
        """Plot SHAP feature importance."""
        if self.shap_values is None:
            self.compute_shap_values()

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values, self.X.iloc[:len(self.shap_values)],
            plot_type="bar", max_display=top_n, show=False
        )
        plt.title(f"SHAP Feature Importance: {self.model_name}")

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved SHAP plot to {save_path}")
        return fig

    def plot_shap_dependence(self, feature: str, save_path: Optional[str] = None):
        """Plot SHAP dependence for a specific feature."""
        if self.shap_values is None:
            self.compute_shap_values()

        feature_idx = list(self.X.columns).index(feature)
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.dependence_plot(
            feature_idx, self.shap_values,
            self.X.iloc[:len(self.shap_values)],
            show=False
        )
        plt.title(f"SHAP Dependence: {feature}")

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def get_top_features(self, top_n: int = 20) -> List[str]:
        """Get top N features by mean absolute SHAP value."""
        if self.shap_values is None:
            self.compute_shap_values()

        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        feature_importance = pd.Series(
            mean_abs_shap, index=self.X.columns[:len(mean_abs_shap)]
        ).sort_values(ascending=False)

        return feature_importance.head(top_n).index.tolist()
