"""
Model training pipeline for Bias Bounty Mapping Equity Challenge.

Trains XGBoost, LightGBM, CatBoost with Optuna hyperparameter tuning,
using spatial cross-validation for evaluation.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import optuna
import joblib
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from src.validation.spatial_cv import SpatialCV, compute_cv_scores

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


class ModelTrainer:
    """Train and evaluate models for coverage gap prediction."""

    def __init__(self, config_path: str = "config/model_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.seed = self.config["random_seed"]
        self.models = {}

    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        tune: bool = False,
    ) -> xgb.XGBRegressor:
        """Train XGBoost model with optional Optuna tuning."""
        logger.info("Training XGBoost...")

        if tune:
            best_params = self._tune_xgboost(X_train, y_train)
            params = {**self.config["models"]["xgboost"], **best_params}
        else:
            params = self.config["models"]["xgboost"]

        model = xgb.XGBRegressor(**params)

        if X_val is not None and y_val is not None:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=100,
            )
        else:
            model.fit(X_train, y_train)

        self.models["xgboost"] = model
        return model

    def train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        tune: bool = False,
    ) -> lgb.LGBMRegressor:
        """Train LightGBM model."""
        logger.info("Training LightGBM...")

        if tune:
            best_params = self._tune_lightgbm(X_train, y_train)
            params = {**self.config["models"]["lightgbm"], **best_params}
        else:
            params = self.config["models"]["lightgbm"]

        model = lgb.LGBMRegressor(**params)

        if X_val is not None and y_val is not None:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
            )
        else:
            model.fit(X_train, y_train)

        self.models["lightgbm"] = model
        return model

    def train_catboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        cat_features: Optional[List[str]] = None,
        tune: bool = False,
    ) -> cb.CatBoostRegressor:
        """Train CatBoost model."""
        logger.info("Training CatBoost...")

        if tune:
            best_params = self._tune_catboost(X_train, y_train, cat_features)
            params = {**self.config["models"]["catboost"], **best_params}
        else:
            params = self.config["models"]["catboost"]

        model = cb.CatBoostRegressor(**params)

        fit_kwargs = {}
        if cat_features:
            fit_kwargs["cat_features"] = cat_features
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = (X_val, y_val)

        model.fit(X_train, y_train, **fit_kwargs)
        self.models["catboost"] = model
        return model

    def train_all_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        tune: bool = False,
        cv_strategy: str = "group_kfold",
    ) -> Dict[str, Tuple[Any, float]]:
        """
        Train all base models with spatial CV evaluation.
        Returns dict of {model_name: (model, cv_rmse)}.
        """
        results = {}

        # Spatial CV setup
        cv = SpatialCV()

        for model_name, train_fn in [
            ("xgboost", self.train_xgboost),
            ("lightgbm", self.train_lightgbm),
            ("catboost", self.train_catboost),
        ]:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training {model_name}")
            logger.info(f"{'='*60}")

            # Get OOF predictions via spatial CV
            oof_preds, mean_rmse, std_rmse = compute_cv_scores(
                train_fn(X, y, tune=tune),
                X, y,
                cv_strategy=cv_strategy,
            )

            results[model_name] = {
                "model": self.models[model_name],
                "cv_rmse_mean": mean_rmse,
                "cv_rmse_std": std_rmse,
                "oof_predictions": oof_preds,
            }

            logger.info(f"{model_name}: CV RMSE = {mean_rmse:.6f} ± {std_rmse:.6f}")

        return results

    def _tune_xgboost(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Optuna hyperparameter tuning for XGBoost."""
        def objective(trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 500, 3000),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                "tree_method": "hist",
                "random_state": self.seed,
            }

            model = xgb.XGBRegressor(**params)

            # Use GroupKFold by county
            _, mean_rmse, _ = compute_cv_scores(
                model, X, y, cv_strategy="group_kfold"
            )
            return mean_rmse

        study = optuna.create_study(direction="minimize", sampler=optuna.TPESampler(seed=self.seed))
        study.optimize(objective, n_trials=self.config["optuna"]["n_trials"],
                       timeout=self.config["optuna"]["timeout"])

        logger.info(f"XGBoost best RMSE: {study.best_value:.6f}")
        logger.info(f"XGBoost best params: {study.best_params}")
        return study.best_params

    def _tune_lightgbm(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Optuna hyperparameter tuning for LightGBM."""
        def objective(trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "num_leaves": trial.suggest_int("num_leaves", 20, 200),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 500, 3000),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "verbose": -1,
                "random_state": self.seed,
            }

            model = lgb.LGBMRegressor(**params)
            _, mean_rmse, _ = compute_cv_scores(model, X, y, cv_strategy="group_kfold")
            return mean_rmse

        study = optuna.create_study(direction="minimize", sampler=optuna.TPESampler(seed=self.seed))
        study.optimize(objective, n_trials=self.config["optuna"]["n_trials"],
                       timeout=self.config["optuna"]["timeout"])

        logger.info(f"LightGBM best RMSE: {study.best_value:.6f}")
        return study.best_params

    def _tune_catboost(self, X: pd.DataFrame, y: pd.Series,
                        cat_features: Optional[List[str]] = None) -> Dict:
        """Optuna hyperparameter tuning for CatBoost."""
        def objective(trial):
            params = {
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
                "iterations": trial.suggest_int("iterations", 500, 3000),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0),
                "random_seed": self.seed,
                "verbose": 0,
            }

            model = cb.CatBoostRegressor(**params)
            _, mean_rmse, _ = compute_cv_scores(model, X, y, cv_strategy="group_kfold")
            return mean_rmse

        study = optuna.create_study(direction="minimize", sampler=optuna.TPESampler(seed=self.seed))
        study.optimize(objective, n_trials=self.config["optuna"]["n_trials"],
                       timeout=self.config["optuna"]["timeout"])

        logger.info(f"CatBoost best RMSE: {study.best_value:.6f}")
        return study.best_params

    def get_feature_importance(self, model_name: str, feature_names: List[str]) -> pd.DataFrame:
        """Get feature importance for a trained model."""
        model = self.models[model_name]

        if model_name in ["xgboost", "lightgbm"]:
            importance = model.feature_importances_
        elif model_name == "catboost":
            importance = model.get_feature_importance()
        else:
            raise ValueError(f"Unknown model: {model_name}")

        fi_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False)

        return fi_df

    def save_model(self, model_name: str, path: str):
        """Save trained model to disk."""
        joblib.dump(self.models[model_name], path)
        logger.info(f"Saved {model_name} to {path}")

    def load_model(self, model_name: str, path: str):
        """Load trained model from disk."""
        self.models[model_name] = joblib.load(path)
        logger.info(f"Loaded {model_name} from {path}")
