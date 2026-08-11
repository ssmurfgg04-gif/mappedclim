"""
Ensembling strategies and submission generation.

Combines multiple model predictions using:
1. Weighted averaging (optimize weights for RMSE)
2. Stacking (Ridge meta-learner on OOF predictions)
3. Target reverse-engineering (attempt to match exact formula)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class WeightedAverager:
    """
    Optimize blend weights to minimize RMSE on OOF predictions.
    A single model won't win — three models that make DIFFERENT
    kinds of errors will outperform one perfect model when blended.
    """

    def __init__(self):
        self.weights = None

    def optimize_weights(
        self,
        oof_predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
        method: str = "scipy",
    ) -> np.ndarray:
        """
        Find optimal weights that minimize RMSE of weighted average.
        
        Args:
            oof_predictions: Dict of {model_name: oof_pred_array}
            y_true: True target values
            method: 'scipy' (constrained optimization) or 'grid' (grid search)
        
        Returns:
            Optimal weight vector (sums to 1, all >= 0)
        """
        model_names = list(oof_predictions.keys())
        n_models = len(model_names)
        oof_matrix = np.column_stack([oof_predictions[name] for name in model_names])

        if method == "scipy":
            def objective(weights):
                pred = oof_matrix @ weights
                return np.sqrt(mean_squared_error(y_true, pred))

            # Constraints: weights sum to 1
            constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
            # Bounds: all weights >= 0
            bounds = [(0.0, 1.0)] * n_models
            # Initial guess: equal weights
            x0 = np.ones(n_models) / n_models

            result = minimize(
                objective, x0, method="SLSQP",
                bounds=bounds, constraints=constraints,
                options={"maxiter": 1000, "ftol": 1e-10}
            )

            self.weights = result.x
            best_rmse = result.fun

        elif method == "grid":
            # Grid search over weight combinations
            best_rmse = np.inf
            best_weights = None

            if n_models == 2:
                for w1 in np.arange(0, 1.01, 0.01):
                    w2 = 1 - w1
                    weights = np.array([w1, w2])
                    pred = oof_matrix @ weights
                    rmse = np.sqrt(mean_squared_error(y_true, pred))
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_weights = weights
            elif n_models == 3:
                for w1 in np.arange(0, 1.01, 0.05):
                    for w2 in np.arange(0, 1.01 - w1, 0.05):
                        w3 = 1 - w1 - w2
                        weights = np.array([w1, w2, w3])
                        pred = oof_matrix @ weights
                        rmse = np.sqrt(mean_squared_error(y_true, pred))
                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_weights = weights
            else:
                # Fallback to equal weights
                best_weights = np.ones(n_models) / n_models
                pred = oof_matrix @ best_weights
                best_rmse = np.sqrt(mean_squared_error(y_true, pred))

            self.weights = best_weights

        # Log results
        logger.info("Optimal blend weights:")
        for name, weight in zip(model_names, self.weights):
            logger.info(f"  {name}: {weight:.4f}")
        logger.info(f"Blended CV RMSE: {best_rmse:.6f}")

        return self.weights

    def predict(
        self,
        test_predictions: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Generate blended predictions for test set."""
        if self.weights is None:
            raise ValueError("Weights not optimized yet. Call optimize_weights first.")

        model_names = list(test_predictions.keys())
        test_matrix = np.column_stack([test_predictions[name] for name in model_names])
        return test_matrix @ self.weights


class StackingEnsemble:
    """
    Stacking ensemble with Ridge meta-learner.
    Trains meta-learner on out-of-fold predictions from base models.
    """

    def __init__(self, alpha: float = 1.0):
        self.meta_model = Ridge(alpha=alpha)
        self.is_fitted = False

    def fit(
        self,
        oof_predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
    ) -> "StackingEnsemble":
        """Train meta-learner on OOF predictions."""
        model_names = list(oof_predictions.keys())
        oof_matrix = np.column_stack([oof_predictions[name] for name in model_names])

        self.meta_model.fit(oof_matrix, y_true)
        self.is_fitted = True

        # Log coefficients
        logger.info("Stacking meta-learner coefficients:")
        for name, coef in zip(model_names, self.meta_model.coef_):
            logger.info(f"  {name}: {coef:.6f}")
        logger.info(f"  intercept: {self.meta_model.intercept_:.6f}")

        # Evaluate
        pred = self.meta_model.predict(oof_matrix)
        rmse = np.sqrt(mean_squared_error(y_true, pred))
        logger.info(f"Stacking train RMSE: {rmse:.6f}")

        return self

    def predict(
        self,
        test_predictions: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Generate stacked predictions for test set."""
        if not self.is_fitted:
            raise ValueError("Meta-learner not fitted yet. Call fit first.")

        model_names = list(test_predictions.keys())
        test_matrix = np.column_stack([test_predictions[name] for name in model_names])
        return self.meta_model.predict(test_matrix)


class TargetReverseEngineer:
    """
    Attempt to reverse-engineer the organizer's coverage gap formula.
    
    The target is a deterministic function of the input data.
    If we can figure out the exact formula, we don't need ML.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)

    def search_formula(
        self,
        sub_metrics: pd.DataFrame,
        y_true: np.ndarray,
        max_complexity: int = 3,
    ) -> Dict:
        """
        Search for the coverage gap formula using symbolic regression.
        
        Args:
            sub_metrics: DataFrame of coverage sub-metrics per tract
            y_true: True coverage gap scores
            max_complexity: Maximum formula complexity
        
        Returns:
            Best formula and its RMSE
        """
        try:
            from pysr import PySRRegressor
        except ImportError:
            logger.warning("PySR not installed. Install with: pip install pysr")
            logger.info("Falling back to linear formula search")
            return self._linear_search(sub_metrics, y_true)

        # Symbolic regression
        model = PySRRegressor(
            niterations=100,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["log", "sqrt", "abs"],
            maxsize=max_complexity + 5,
            populations=20,
            population_size=50,
            parsimony=0.01,
            seed=self.seed,
        )

        model.fit(sub_metrics.values, y_true)

        logger.info(f"Best formula: {model.get_best()}")
        return {"model": model, "formula": model.get_best()}

    def _linear_search(
        self,
        sub_metrics: pd.DataFrame,
        y_true: np.ndarray,
    ) -> Dict:
        """
        Search over linear combinations of sub-metrics.
        Tests different weight configurations.
        """
        from itertools import product

        X = sub_metrics.values
        n_features = X.shape[1]
        feature_names = sub_metrics.columns.tolist()

        best_rmse = np.inf
        best_weights = None

        # Grid search over weights
        weight_range = np.arange(0, 1.01, 0.1)

        if n_features <= 3:
            # Exhaustive grid search
            for weights in product(weight_range, repeat=n_features):
                weights = np.array(weights)
                if weights.sum() == 0:
                    continue
                weights = weights / weights.sum()
                pred = X @ weights
                rmse = np.sqrt(mean_squared_error(y_true, pred))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_weights = weights
        else:
            # Use Ridge regression as proxy
            from sklearn.linear_model import RidgeCV
            ridge = RidgeCV(alphas=np.logspace(-3, 3, 20))
            ridge.fit(X, y_true)
            best_weights = ridge.coef_
            pred = ridge.predict(X)
            best_rmse = np.sqrt(mean_squared_error(y_true, pred))

        logger.info("Best linear formula weights:")
        for name, weight in zip(feature_names, best_weights):
            logger.info(f"  {name}: {weight:.4f}")
        logger.info(f"Linear formula RMSE: {best_rmse:.6f}")

        return {"weights": best_weights, "rmse": best_rmse, "feature_names": feature_names}

    def test_nonlinear_formulas(
        self,
        sub_metrics: pd.DataFrame,
        y_true: np.ndarray,
    ) -> pd.DataFrame:
        """
        Test specific nonlinear formula candidates.
        These are the most likely organizer formulas.
        """
        results = []

        building_ratio = sub_metrics.get("building_count_ratio", sub_metrics.iloc[:, 0])
        road_ratio = sub_metrics.get("road_length_ratio", sub_metrics.iloc[:, 1] if sub_metrics.shape[1] > 1 else building_ratio)
        poi_ratio = sub_metrics.get("poi_to_facility_ratio", sub_metrics.iloc[:, 2] if sub_metrics.shape[1] > 2 else building_ratio)

        # Candidate 1: Simple gap = 1 - ratio
        pred1 = 1 - building_ratio
        results.append(("1-building_ratio", np.sqrt(mean_squared_error(y_true, pred1))))

        # Candidate 2: Arithmetic mean of gaps
        pred2 = (1 - building_ratio + 1 - road_ratio + 1 - poi_ratio) / 3
        results.append(("arithmetic_mean_gap", np.sqrt(mean_squared_error(y_true, pred2))))

        # Candidate 3: Weighted mean (buildings more important)
        pred3 = 0.5 * (1 - building_ratio) + 0.3 * (1 - road_ratio) + 0.2 * (1 - poi_ratio)
        results.append(("weighted_mean_gap", np.sqrt(mean_squared_error(y_true, pred3))))

        # Candidate 4: Geometric mean of gaps
        gaps = np.clip([1 - building_ratio, 1 - road_ratio, 1 - poi_ratio], 0.001, None)
        pred4 = np.exp(np.mean(np.log(gaps), axis=0))
        results.append(("geometric_mean_gap", np.sqrt(mean_squared_error(y_true, pred4))))

        # Candidate 5: Harmonic mean of ratios
        ratios = np.clip([building_ratio, road_ratio, poi_ratio], 0.001, None)
        pred5 = len(ratios) / np.sum(1.0 / ratios, axis=0)
        pred5_gap = 1 - pred5
        results.append(("harmonic_mean_gap", np.sqrt(mean_squared_error(y_true, pred5_gap))))

        # Candidate 6: Max gap (worst dimension)
        pred6 = np.max([1 - building_ratio, 1 - road_ratio, 1 - poi_ratio], axis=0)
        results.append(("max_gap", np.sqrt(mean_squared_error(y_true, pred6))))

        results_df = pd.DataFrame(results, columns=["formula", "rmse"]).sort_values("rmse")
        logger.info("Formula search results:\n" + results_df.to_string())
        return results_df


def generate_submission(
    predictions: np.ndarray,
    test_geoids: np.ndarray,
    output_path: str,
) -> pd.DataFrame:
    """
    Generate submission file in the format expected by Zindi.
    
    Args:
        predictions: Predicted coverage gap scores
        test_geoids: Census tract GEOIDs for test set
        output_path: Path to save CSV
    
    Returns:
        Submission DataFrame
    """
    submission = pd.DataFrame({
        "GEOID": test_geoids,
        "coverage_gap": predictions,
    })

    # Clip predictions to reasonable range
    submission["coverage_gap"] = submission["coverage_gap"].clip(-1, 1)

    submission.to_csv(output_path, index=False)
    logger.info(f"Submission saved to {output_path}: {len(submission)} rows")
    logger.info(f"Prediction stats: mean={predictions.mean():.4f}, "
                f"std={predictions.std():.4f}, min={predictions.min():.4f}, max={predictions.max():.4f}")

    return submission
