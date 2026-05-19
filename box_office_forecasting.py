"""Genre-level box office forecasting with rolling validation utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestRegressor
except ImportError:  # pragma: no cover - optional dependency for local environments
    RandomForestRegressor = None

from data_preprocessing import (
    DATASET_CUTOFF_DATE,
    prepare_dataset,
    resolve_cleaned_dataset_path,
)

FORECAST_VALIDATION_YEAR = 2017
FORECAST_ALPHA = 8.0
FORECAST_RANDOM_FOREST_TREES = 300
FORECAST_RANDOM_FOREST_MIN_LEAF = 3
FORECAST_MIN_TRAIN_ROWS = 12
FORECAST_DEFAULT_MODEL = "ridge"
FORECAST_SUPPORTED_MODELS = ("ridge", "ols", "genre_budget_ratio", "random_forest")
FEATURE_COLUMNS = (
    "log_budget",
    "log_budget_sq",
    "runtime",
    "runtime_sq",
    "year",
    "month_sin",
    "month_cos",
    "is_english",
    "is_us_production",
    "budget_observed",
    "runtime_observed",
)
FORECAST_GENRE_CALIBRATION_MIN_SAMPLES = 20
FORECAST_GENRE_CALIBRATION_BLEND_SAMPLES = 250

FORECAST_MODEL_LABELS = {
    "ridge": "Ridge Regression",
    "ols": "OLS Regression",
    "genre_budget_ratio": "Genre Budget Ratio",
    "random_forest": "Random Forest Regression",
}
FORECAST_CACHE_METADATA_VERSION = 1

HIGH_BOX_OFFICE_MIN_ROWS = 40
HIGH_BOX_OFFICE_MIN_TRAIN_ROWS = 24
HIGH_BOX_OFFICE_MIN_HOLDOUT_ROWS = 8
HIGH_BOX_OFFICE_HOLDOUT_FRACTION = 0.2
HIGH_BOX_OFFICE_THRESHOLD_QUANTILE = 0.75
HIGH_BOX_OFFICE_PROFILE_ALPHA = 6.0
HIGH_BOX_OFFICE_MAX_GENRES = 8
HIGH_BOX_OFFICE_MIN_GENRE_ROWS = 5


def _parse_list_like(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") or text.startswith("{"):
            try:
                parsed = json.loads(text.replace("'", '"'))
            except json.JSONDecodeError:
                try:
                    import ast

                    parsed = ast.literal_eval(text)
                except (SyntaxError, ValueError):
                    parsed = None
            if isinstance(parsed, list):
                items = parsed
            elif isinstance(parsed, dict):
                items = [parsed]
            else:
                items = [part.strip() for part in text.replace("|", ",").split(",") if part.strip()]
        else:
            items = [part.strip() for part in text.replace("|", ",").split(",") if part.strip()]
    else:
        items = [value]

    flattened: list[str] = []
    for item in items:
        if isinstance(item, dict):
            name = item.get("name") or item.get("iso_639_1") or item.get("iso_3166_1")
            if name:
                flattened.append(str(name).strip())
        else:
            text = str(item).strip()
            if text:
                flattened.append(text)
    return flattened


def _coerce_primary_genre(value: object) -> pd._libs.missing.NAType | str:
    tokens = _parse_list_like(value)
    return tokens[0] if tokens else pd.NA


def _normalise_token_field(value: object) -> str:
    return "|".join(_parse_list_like(value))


def _deduplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "id" in df.columns:
        ids = pd.to_numeric(df["id"], errors="coerce")
        deduplicated = df.copy()
        deduplicated["id"] = ids
        deduplicated = deduplicated.dropna(subset=["id"]).drop_duplicates(subset=["id"])
        return deduplicated

    subset = [column for column in ("title", "release_date") if column in df.columns]
    return df.drop_duplicates(subset=subset if subset else None)


def _coerce_observed_flag(value: pd.Series | object, fallback_mask: pd.Series) -> pd.Series:
    if not isinstance(value, pd.Series):
        return fallback_mask.astype("boolean")

    if str(value.dtype) == "boolean" or pd.api.types.is_bool_dtype(value):
        coerced = value.astype("boolean")
    else:
        text = value.astype("string").str.strip().str.lower()
        mapped = text.map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
                "yes": True,
                "no": False,
            }
        )
        if mapped.notna().sum() == 0:
            numeric = pd.to_numeric(value, errors="coerce")
            mapped = numeric.map(lambda item: bool(item) if pd.notna(item) else pd.NA)
        coerced = mapped.astype("boolean")

    return coerced.fillna(fallback_mask.astype("boolean"))


def prepare_forecasting_dataframe(
    df: pd.DataFrame,
    cutoff_date: pd.Timestamp = DATASET_CUTOFF_DATE,
) -> pd.DataFrame:
    """Prepare a modelling dataframe without imputing the revenue target."""
    working = df.copy()

    if "title" in working.columns:
        working = working.dropna(subset=["title"]).copy()
    working = _deduplicate_rows(working).copy()

    if "primary_genre" not in working.columns:
        if "genres" in working.columns:
            working["primary_genre"] = working["genres"].apply(_coerce_primary_genre)
        else:
            working["primary_genre"] = pd.NA
    else:
        working["primary_genre"] = working["primary_genre"].apply(_coerce_primary_genre)

    if "language" not in working.columns:
        if "original_language" in working.columns:
            working["language"] = working["original_language"]
        else:
            working["language"] = pd.NA
    working["language"] = working["language"].apply(_normalise_token_field)

    if "country" not in working.columns:
        if "production_countries" in working.columns:
            working["country"] = working["production_countries"]
        else:
            working["country"] = pd.NA
    working["country"] = working["country"].apply(_normalise_token_field)

    if "release_date" not in working.columns:
        raise KeyError("Forecasting requires a 'release_date' column.")

    working["release_date"] = pd.to_datetime(working["release_date"], errors="coerce")
    working = working.dropna(subset=["release_date"]).copy()
    working = working.loc[working["release_date"] <= cutoff_date].copy()
    working["year"] = working["release_date"].dt.year.astype(int)
    working["month"] = working["release_date"].dt.month.astype(int)

    for column in ("budget", "revenue", "runtime"):
        if column not in working.columns:
            working[column] = np.nan
        numeric = pd.to_numeric(working[column], errors="coerce")
        observed_column = f"{column}_observed"
        fallback_mask = numeric.gt(0)
        if observed_column in working.columns:
            working[observed_column] = _coerce_observed_flag(working[observed_column], fallback_mask)
        else:
            working[observed_column] = fallback_mask.astype("boolean")
        working[column] = numeric
        working.loc[working[column] <= 0, column] = np.nan
        working.loc[~working[observed_column].fillna(False), column] = np.nan

    keep_columns = [
        column
        for column in (
            "id",
            "source_id",
            "title",
            "primary_genre",
            "budget",
            "budget_observed",
            "revenue",
            "revenue_observed",
            "runtime",
            "runtime_observed",
            "release_date",
            "year",
            "month",
            "language",
            "country",
        )
        if column in working.columns
    ]
    return working[keep_columns].sort_values(["release_date", "title"]).reset_index(drop=True)


def load_forecasting_dataset(
    dataset_path: Optional[str | Path] = None,
    cutoff_date: pd.Timestamp = DATASET_CUTOFF_DATE,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Load the cleaned dataset used by the application."""
    cleaned_df, _ = prepare_dataset(dataset_path=dataset_path, cutoff_date=cutoff_date, prefer_cleaned=True)
    source_path = resolve_cleaned_dataset_path(dataset_path)
    prepared = prepare_forecasting_dataframe(cleaned_df, cutoff_date=cutoff_date)
    metadata = {
        "source_type": "cleaned_dataset",
        "source_file": str(source_path),
        "rows_loaded": int(len(cleaned_df)),
        "rows_after_preparation": int(len(prepared)),
        "coverage_end_date": cutoff_date.strftime("%Y-%m-%d"),
        "merge_metrics": {},
    }
    return prepared, metadata


def _build_feature_frame(df: pd.DataFrame, genre_levels: Optional[list[str]] = None) -> pd.DataFrame:
    months = pd.to_numeric(df["month"], errors="coerce")
    languages = df["language"].astype("string").fillna("")
    countries = df["country"].astype("string").fillna("")
    language_tokens = languages.map(
        lambda value: {
            token.strip().lower()
            for token in str(value).replace(",", "|").split("|")
            if token.strip()
        }
    )
    is_english = language_tokens.map(lambda tokens: "en" in tokens or "english" in tokens)
    is_us_production = countries.str.contains("United States of America", case=False, na=False, regex=False)

    feature_frame = pd.DataFrame(index=df.index)
    budget = pd.to_numeric(df["budget"], errors="coerce")
    runtime = pd.to_numeric(df["runtime"], errors="coerce")
    log_budget = np.log1p(budget)
    feature_frame["log_budget"] = log_budget
    feature_frame["log_budget_sq"] = np.square(log_budget)
    feature_frame["runtime"] = runtime
    feature_frame["runtime_sq"] = np.square(runtime)
    feature_frame["year"] = pd.to_numeric(df["year"], errors="coerce")
    feature_frame["month_sin"] = np.sin(2 * np.pi * months / 12.0)
    feature_frame["month_cos"] = np.cos(2 * np.pi * months / 12.0)
    feature_frame["is_english"] = is_english.astype(float)
    feature_frame["is_us_production"] = is_us_production.astype(float)
    feature_frame["budget_observed"] = df.get("budget_observed", budget.notna()).astype("boolean").astype(float)
    feature_frame["runtime_observed"] = df.get("runtime_observed", runtime.notna()).astype("boolean").astype(float)

    if genre_levels:
        genre_series = df["primary_genre"].astype("string").fillna("")
        for genre in genre_levels[1:]:
            genre_indicator = (genre_series == genre).astype(float)
            feature_frame[f"genre::{genre}"] = genre_indicator
            feature_frame[f"log_budget_x_genre::{genre}"] = log_budget.fillna(0.0).to_numpy() * genre_indicator.to_numpy()

    return feature_frame


def _fit_linear_calibration(predicted: np.ndarray, actual: np.ndarray) -> Dict[str, float]:
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if predicted.size == 0:
        return {"intercept": 0.0, "slope": 1.0}

    scale = float(np.sum(actual) / np.sum(predicted)) if np.sum(predicted) > 0 else 1.0
    if predicted.size < 2 or float(np.ptp(predicted)) <= 1e-9:
        return {"intercept": 0.0, "slope": max(scale, 0.0)}

    design = np.c_[np.ones(len(predicted)), predicted]
    coefficients = np.linalg.lstsq(design, actual, rcond=None)[0]
    intercept = float(coefficients[0]) if np.isfinite(coefficients[0]) else 0.0
    slope = float(coefficients[1]) if np.isfinite(coefficients[1]) else scale

    if slope <= 0:
        intercept = 0.0
        slope = max(scale, 0.0)

    slope = float(np.clip(slope, 0.05, 10.0))
    return {"intercept": intercept, "slope": slope}


def _apply_linear_calibration(predicted: np.ndarray, calibration: Dict[str, float]) -> np.ndarray:
    calibrated = calibration["intercept"] + calibration["slope"] * np.asarray(predicted, dtype=float)
    return np.clip(calibrated, a_min=0.0, a_max=None)


def _fit_genre_calibrations(
    train_df: pd.DataFrame,
    base_predictions: np.ndarray,
    global_calibration: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    calibrations: Dict[str, Dict[str, float]] = {}
    genre_series = train_df["primary_genre"].astype("string").fillna("")

    for genre in sorted(genre_series.unique().tolist()):
        mask = (genre_series == genre).to_numpy()
        actual = train_df.loc[mask, "revenue"].to_numpy(dtype=float)
        predicted = base_predictions[mask]

        if len(predicted) >= FORECAST_GENRE_CALIBRATION_MIN_SAMPLES and float(np.ptp(predicted)) > 1e-9:
            genre_calibration = _fit_linear_calibration(predicted, actual)
            blend_weight = min(1.0, len(predicted) / float(FORECAST_GENRE_CALIBRATION_BLEND_SAMPLES))
            intercept = global_calibration["intercept"] * (1.0 - blend_weight) + genre_calibration["intercept"] * blend_weight
            slope = global_calibration["slope"] * (1.0 - blend_weight) + genre_calibration["slope"] * blend_weight
        elif np.sum(predicted) > 0:
            total_scale = float(np.sum(actual) / np.sum(predicted))
            blend_weight = min(1.0, len(predicted) / float(FORECAST_GENRE_CALIBRATION_MIN_SAMPLES))
            intercept = global_calibration["intercept"] * (1.0 - blend_weight)
            slope = global_calibration["slope"] * (1.0 - blend_weight) + total_scale * blend_weight
        else:
            intercept = global_calibration["intercept"]
            slope = global_calibration["slope"]

        calibrations[str(genre)] = {
            "intercept": float(intercept),
            "slope": float(np.clip(slope, 0.05, 10.0)),
        }

    return calibrations


def _fit_linear_model(train_df: pd.DataFrame, alpha: float = 0.0) -> Dict[str, object]:
    genre_levels = sorted(train_df["primary_genre"].dropna().astype(str).unique().tolist())
    features = _build_feature_frame(train_df, genre_levels=genre_levels)
    medians = features.median(numeric_only=True).fillna(0.0)
    matrix = features.fillna(medians).to_numpy(dtype=float)
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales[scales == 0.0] = 1.0
    matrix = (matrix - means) / scales
    design = np.c_[np.ones(len(matrix)), matrix]

    target = np.log1p(train_df["revenue"].to_numpy(dtype=float))
    if float(alpha) > 0.0:
        penalty = np.eye(design.shape[1]) * float(alpha)
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    else:
        coefficients = np.linalg.lstsq(design, target, rcond=None)[0]

    base_prediction = np.expm1(design @ coefficients)
    base_prediction = np.clip(base_prediction, a_min=0.0, a_max=None)
    global_calibration = _fit_linear_calibration(base_prediction, train_df["revenue"].to_numpy(dtype=float))
    genre_calibrations = _fit_genre_calibrations(train_df, base_prediction, global_calibration)

    return {
        "genre_levels": genre_levels,
        "feature_columns": features.columns.tolist(),
        "medians": medians,
        "means": means,
        "scales": scales,
        "coefficients": coefficients,
        "alpha": float(alpha),
        "model_name": "ridge" if float(alpha) > 0.0 else "ols",
        "global_calibration": global_calibration,
        "genre_calibrations": genre_calibrations,
    }


def _fit_ridge_model(train_df: pd.DataFrame, alpha: float = FORECAST_ALPHA) -> Dict[str, object]:
    return _fit_linear_model(train_df, alpha=alpha)


def _fit_ols_model(train_df: pd.DataFrame) -> Dict[str, object]:
    return _fit_linear_model(train_df, alpha=0.0)


def _predict_linear_model(model: Dict[str, object], df: pd.DataFrame) -> np.ndarray:
    features = _build_feature_frame(df, genre_levels=model["genre_levels"])
    features = features.reindex(columns=model["feature_columns"])
    filled = features.fillna(model["medians"]).to_numpy(dtype=float)
    scaled = (filled - model["means"]) / model["scales"]
    design = np.c_[np.ones(len(scaled)), scaled]
    prediction = np.expm1(design @ model["coefficients"])
    prediction = np.clip(prediction, a_min=0.0, a_max=None)

    calibrated = _apply_linear_calibration(prediction, model["global_calibration"])
    genre_series = df["primary_genre"].astype("string").fillna("")
    for genre, calibration in model["genre_calibrations"].items():
        mask = (genre_series == genre).to_numpy()
        if mask.any():
            calibrated[mask] = _apply_linear_calibration(prediction[mask], calibration)

    return np.clip(calibrated, a_min=0.0, a_max=None)


def _fit_genre_budget_ratio_model(train_df: pd.DataFrame) -> Dict[str, object]:
    working = train_df.copy()
    working["budget"] = pd.to_numeric(working["budget"], errors="coerce")
    working["revenue"] = pd.to_numeric(working["revenue"], errors="coerce")

    ratio_rows = working.loc[
        working["budget"].gt(0)
        & working["revenue"].gt(0)
        & working["primary_genre"].notna()
    ].copy()
    if ratio_rows.empty:
        ratio_rows = working.loc[working["primary_genre"].notna()].copy()
        ratio_rows["revenue_to_budget"] = np.nan
    else:
        ratio_rows["revenue_to_budget"] = ratio_rows["revenue"] / ratio_rows["budget"]

    genre_ratios = (
        ratio_rows.groupby("primary_genre")["revenue_to_budget"]
        .median()
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_dict()
    )
    global_ratio = float(ratio_rows["revenue_to_budget"].replace([np.inf, -np.inf], np.nan).dropna().median())
    if not np.isfinite(global_ratio):
        global_ratio = 1.0

    genre_medians = (
        working.groupby("primary_genre")["revenue"]
        .median()
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_dict()
    )
    global_median = float(working["revenue"].replace([np.inf, -np.inf], np.nan).dropna().median())
    if not np.isfinite(global_median):
        global_median = 0.0

    base_prediction = _predict_genre_budget_ratio_raw(
        {
            "genre_ratios": genre_ratios,
            "global_ratio": global_ratio,
            "genre_medians": genre_medians,
            "global_median": global_median,
        },
        train_df,
    )
    global_calibration = _fit_linear_calibration(base_prediction, working["revenue"].to_numpy(dtype=float))
    genre_calibrations = _fit_genre_calibrations(train_df, base_prediction, global_calibration)

    return {
        "model_name": "genre_budget_ratio",
        "genre_ratios": genre_ratios,
        "global_ratio": global_ratio,
        "genre_medians": genre_medians,
        "global_median": global_median,
        "global_calibration": global_calibration,
        "genre_calibrations": genre_calibrations,
    }


def _predict_genre_budget_ratio_raw(model: Dict[str, object], df: pd.DataFrame) -> np.ndarray:
    budgets = pd.to_numeric(df.get("budget"), errors="coerce")
    genres = df["primary_genre"].astype("string").fillna("")

    predictions: list[float] = []
    for genre, budget in zip(genres, budgets):
        genre_key = str(genre)
        ratio = model["genre_ratios"].get(genre_key, model["global_ratio"])
        fallback_revenue = model["genre_medians"].get(genre_key, model["global_median"])
        if pd.notna(budget) and float(budget) > 0:
            predicted = float(budget) * float(ratio)
        else:
            predicted = float(fallback_revenue)
        predictions.append(max(predicted, 0.0))
    return np.asarray(predictions, dtype=float)


def _predict_genre_budget_ratio(model: Dict[str, object], df: pd.DataFrame) -> np.ndarray:
    prediction = _predict_genre_budget_ratio_raw(model, df)
    calibrated = _apply_linear_calibration(prediction, model["global_calibration"])
    genre_series = df["primary_genre"].astype("string").fillna("")
    for genre, calibration in model["genre_calibrations"].items():
        mask = (genre_series == genre).to_numpy()
        if mask.any():
            calibrated[mask] = _apply_linear_calibration(prediction[mask], calibration)
    return np.clip(calibrated, a_min=0.0, a_max=None)


# --- Random Forest Regression Forecast Model ---

def _fit_random_forest_model(
    train_df: pd.DataFrame,
    *,
    n_estimators: int = FORECAST_RANDOM_FOREST_TREES,
    min_samples_leaf: int = FORECAST_RANDOM_FOREST_MIN_LEAF,
) -> Dict[str, object]:
    if RandomForestRegressor is None:
        raise ImportError("Random Forest forecasting requires scikit-learn to be installed.")

    genre_levels = sorted(train_df["primary_genre"].dropna().astype(str).unique().tolist())
    features = _build_feature_frame(train_df, genre_levels=genre_levels)
    medians = features.median(numeric_only=True).fillna(0.0)
    matrix = features.fillna(medians).to_numpy(dtype=float)
    target = np.log1p(train_df["revenue"].to_numpy(dtype=float))

    estimator = RandomForestRegressor(
        n_estimators=int(n_estimators),
        min_samples_leaf=int(min_samples_leaf),
        random_state=42,
        n_jobs=-1,
    )
    estimator.fit(matrix, target)

    base_prediction = np.expm1(estimator.predict(matrix))
    base_prediction = np.clip(base_prediction, a_min=0.0, a_max=None)
    global_calibration = _fit_linear_calibration(base_prediction, train_df["revenue"].to_numpy(dtype=float))
    genre_calibrations = _fit_genre_calibrations(train_df, base_prediction, global_calibration)

    return {
        "model_name": "random_forest",
        "genre_levels": genre_levels,
        "feature_columns": features.columns.tolist(),
        "medians": medians,
        "estimator": estimator,
        "global_calibration": global_calibration,
        "genre_calibrations": genre_calibrations,
        "n_estimators": int(n_estimators),
        "min_samples_leaf": int(min_samples_leaf),
    }


def _predict_random_forest_model(model: Dict[str, object], df: pd.DataFrame) -> np.ndarray:
    features = _build_feature_frame(df, genre_levels=model["genre_levels"])
    features = features.reindex(columns=model["feature_columns"])
    matrix = features.fillna(model["medians"]).to_numpy(dtype=float)
    prediction = np.expm1(model["estimator"].predict(matrix))
    prediction = np.clip(prediction, a_min=0.0, a_max=None)

    calibrated = _apply_linear_calibration(prediction, model["global_calibration"])
    genre_series = df["primary_genre"].astype("string").fillna("")
    for genre, calibration in model["genre_calibrations"].items():
        mask = (genre_series == genre).to_numpy()
        if mask.any():
            calibrated[mask] = _apply_linear_calibration(prediction[mask], calibration)

    return np.clip(calibrated, a_min=0.0, a_max=None)


def _fit_forecast_model(
    train_df: pd.DataFrame,
    model_name: str = FORECAST_DEFAULT_MODEL,
    alpha: float = FORECAST_ALPHA,
) -> Dict[str, object]:
    if model_name == "ridge":
        return _fit_ridge_model(train_df, alpha=alpha)
    if model_name == "ols":
        return _fit_ols_model(train_df)
    if model_name == "genre_budget_ratio":
        return _fit_genre_budget_ratio_model(train_df)
    if model_name == "random_forest":
        return _fit_random_forest_model(train_df)
    raise ValueError(f"Unsupported forecast model: {model_name}")


def _predict_forecast_model(model: Dict[str, object], df: pd.DataFrame) -> np.ndarray:
    model_name = str(model.get("model_name", FORECAST_DEFAULT_MODEL))
    if model_name in {"ridge", "ols"}:
        return _predict_linear_model(model, df)
    if model_name == "genre_budget_ratio":
        return _predict_genre_budget_ratio(model, df)
    if model_name == "random_forest":
        return _predict_random_forest_model(model, df)
    raise ValueError(f"Unsupported forecast model: {model_name}")


def _build_model_type_label(model_name: str, train_samples: int, min_train_rows: int) -> str:
    base = str(model_name).strip().lower()
    if base == "ridge":
        return "global_ridge_calibrated" if int(train_samples) >= int(min_train_rows) else "global_ridge_shared"
    if base == "ols":
        return "global_ols_calibrated" if int(train_samples) >= int(min_train_rows) else "global_ols_shared"
    if base == "genre_budget_ratio":
        return "genre_budget_ratio_calibrated" if int(train_samples) >= int(min_train_rows) else "genre_budget_ratio_shared"
    if base == "random_forest":
        return "random_forest_calibrated" if int(train_samples) >= int(min_train_rows) else "random_forest_shared"
    return base


def _regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    absolute_error = np.abs(predicted - actual)
    squared_error = np.square(predicted - actual)
    denominator = float(np.sum(actual))
    ss_res = float(np.sum(squared_error))
    ss_tot = float(np.sum(np.square(actual - np.mean(actual))))
    percentage_error = np.where(actual > 0, absolute_error / actual * 100.0, np.nan)

    return {
        "mae": float(np.mean(absolute_error)),
        "rmse": float(np.sqrt(np.mean(squared_error))),
        "mape_percent": float(np.nanmean(percentage_error)) if np.isfinite(np.nanmean(percentage_error)) else 0.0,
        "wape_percent": float(np.sum(absolute_error) / denominator * 100.0) if denominator else 0.0,
        "r_squared": 1 - ss_res / ss_tot if ss_tot else 0.0,
    }


def _correlation_coefficient(actual: np.ndarray, predicted: np.ndarray) -> float:
    pair = pd.DataFrame(
        {
            "actual": pd.to_numeric(pd.Series(actual), errors="coerce"),
            "predicted": pd.to_numeric(pd.Series(predicted), errors="coerce"),
        }
    ).dropna()
    if len(pair) < 2:
        return 0.0
    if pair["actual"].nunique() < 2 or pair["predicted"].nunique() < 2:
        return 0.0

    correlation = pair["actual"].corr(pair["predicted"])
    if pd.isna(correlation) or not np.isfinite(correlation):
        return 0.0
    return float(correlation)


def _month_to_quarter(month: object) -> str:
    try:
        month_value = int(month)
    except (TypeError, ValueError):
        return "Unknown"

    if month_value in (1, 2, 3):
        return "Q1 (Jan-Mar)"
    if month_value in (4, 5, 6):
        return "Q2 (Apr-Jun)"
    if month_value in (7, 8, 9):
        return "Q3 (Jul-Sep)"
    if month_value in (10, 11, 12):
        return "Q4 (Oct-Dec)"
    return "Unknown"


def _select_high_box_office_genre_levels(
    df: pd.DataFrame,
    max_genres: int = HIGH_BOX_OFFICE_MAX_GENRES,
    min_genre_rows: int = HIGH_BOX_OFFICE_MIN_GENRE_ROWS,
) -> list[str]:
    genre_counts = (
        df["primary_genre"]
        .dropna()
        .astype(str)
        .loc[lambda series: series.str.strip() != ""]
        .value_counts()
    )
    if genre_counts.empty:
        return []

    kept_levels = genre_counts.loc[genre_counts >= int(min_genre_rows)].head(int(max_genres)).index.tolist()
    if not kept_levels:
        kept_levels = genre_counts.head(min(int(max_genres), len(genre_counts))).index.tolist()

    other_count = int(genre_counts.loc[~genre_counts.index.isin(kept_levels)].sum())
    if other_count > 0:
        kept_levels.append("Other")

    return kept_levels


def _build_high_box_office_feature_frame(
    df: pd.DataFrame,
    genre_levels: Optional[list[str]] = None,
    quarter_levels: Optional[list[str]] = None,
) -> pd.DataFrame:
    feature_frame = pd.DataFrame(index=df.index)
    budget = pd.to_numeric(df.get("budget"), errors="coerce")
    runtime = pd.to_numeric(df.get("runtime"), errors="coerce")
    languages = df.get("language", pd.Series("", index=df.index)).astype("string").fillna("")
    countries = df.get("country", pd.Series("", index=df.index)).astype("string").fillna("")
    quarters = pd.Series(df.get("month", pd.Series(index=df.index, dtype="float64")), index=df.index).map(_month_to_quarter)

    language_tokens = languages.map(
        lambda value: {
            token.strip().lower()
            for token in str(value).replace(",", "|").split("|")
            if token.strip()
        }
    )
    feature_frame["log_budget"] = np.log1p(budget)
    feature_frame["runtime"] = runtime
    feature_frame["is_english"] = language_tokens.map(lambda tokens: "en" in tokens or "english" in tokens).astype(float)
    feature_frame["is_us_production"] = countries.str.contains("United States of America", case=False, na=False, regex=False).astype(float)

    selected_quarters = quarter_levels or [
        label
        for label in ("Q1 (Jan-Mar)", "Q2 (Apr-Jun)", "Q3 (Jul-Sep)", "Q4 (Oct-Dec)")
        if label in set(quarters.dropna().tolist())
    ]
    for quarter in selected_quarters:
        feature_frame[f"quarter::{quarter}"] = (quarters == quarter).astype(float)

    selected_genres = genre_levels or _select_high_box_office_genre_levels(df)
    if selected_genres:
        allowed_genres = set(selected_genres)
        genre_series = df["primary_genre"].astype("string").fillna("Unknown")
        if "Other" in allowed_genres:
            specific_genres = allowed_genres - {"Other"}
            genre_series = genre_series.where(genre_series.isin(specific_genres), "Other")
        for genre in selected_genres:
            feature_frame[f"genre::{genre}"] = (genre_series == genre).astype(float)

    return feature_frame


def _fit_log_revenue_ridge_model(
    df: pd.DataFrame,
    *,
    alpha: float = HIGH_BOX_OFFICE_PROFILE_ALPHA,
    genre_levels: Optional[list[str]] = None,
) -> Dict[str, object]:
    selected_quarters = [
        label
        for label in ("Q1 (Jan-Mar)", "Q2 (Apr-Jun)", "Q3 (Jul-Sep)", "Q4 (Oct-Dec)")
        if label in set(pd.Series(df.get("month", pd.Series(index=df.index, dtype="float64")), index=df.index).map(_month_to_quarter).dropna().tolist())
    ]
    selected_genres = genre_levels or _select_high_box_office_genre_levels(df)
    features = _build_high_box_office_feature_frame(df, genre_levels=selected_genres, quarter_levels=selected_quarters)
    medians = features.median(numeric_only=True).fillna(0.0)
    matrix = features.fillna(medians).to_numpy(dtype=float)
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales[scales == 0.0] = 1.0
    standardized = (matrix - means) / scales
    design = np.c_[np.ones(len(standardized)), standardized]
    target = np.log1p(pd.to_numeric(df["revenue"], errors="coerce").to_numpy(dtype=float))

    penalty = np.eye(design.shape[1]) * float(alpha)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ target)

    return {
        "feature_columns": features.columns.tolist(),
        "genre_levels": selected_genres,
        "quarter_levels": selected_quarters,
        "medians": medians,
        "means": means,
        "scales": scales,
        "coefficients": coefficients,
        "alpha": float(alpha),
    }


def _predict_log_revenue_ridge_model(model: Dict[str, object], df: pd.DataFrame) -> np.ndarray:
    features = _build_high_box_office_feature_frame(
        df,
        genre_levels=model.get("genre_levels"),
        quarter_levels=model.get("quarter_levels"),
    )
    features = features.reindex(columns=model["feature_columns"], fill_value=np.nan)
    filled = features.fillna(model["medians"]).to_numpy(dtype=float)
    standardized = (filled - model["means"]) / model["scales"]
    design = np.c_[np.ones(len(standardized)), standardized]
    prediction = np.expm1(design @ model["coefficients"])
    return np.clip(prediction, a_min=0.0, a_max=None)


def _format_box_office_feature_name(feature_name: str) -> str:
    labels = {
        "log_budget": "Higher production budget",
        "runtime": "Longer runtime",
        "is_english": "English-language release",
        "is_us_production": "United States production",
    }
    if feature_name in labels:
        return labels[feature_name]
    if feature_name.startswith("quarter::"):
        return f"Release in {feature_name.split('::', 1)[1]}"
    if feature_name.startswith("genre::"):
        genre = feature_name.split("::", 1)[1]
        return f"{genre} genre"
    return feature_name.replace("_", " ").title()


def _top_feature_rows(coefficients: pd.Series, *, positive: bool, limit: int = 4) -> list[Dict[str, object]]:
    threshold = 0.03
    filtered = coefficients.loc[coefficients > threshold] if positive else coefficients.loc[coefficients < -threshold]
    if filtered.empty:
        filtered = coefficients.nlargest(limit) if positive else coefficients.nsmallest(limit)

    ordered = filtered.sort_values(ascending=not positive).head(limit)
    rows: list[Dict[str, object]] = []
    for feature_name, coefficient in ordered.items():
        rows.append(
            {
                "feature": _format_box_office_feature_name(str(feature_name)),
                "coefficient": round(float(coefficient), 4),
            }
        )
    return rows


def _series_top_values(series: pd.Series, limit: int = 3) -> list[str]:
    if series.empty:
        return []
    values = (
        series.dropna()
        .astype(str)
        .loc[lambda item: item.str.strip() != ""]
        .value_counts()
        .head(limit)
        .index
        .tolist()
    )
    return [str(value) for value in values]


def profile_high_box_office_characteristics(
    df: pd.DataFrame,
    *,
    min_rows: int = HIGH_BOX_OFFICE_MIN_ROWS,
    holdout_fraction: float = HIGH_BOX_OFFICE_HOLDOUT_FRACTION,
    min_train_rows: int = HIGH_BOX_OFFICE_MIN_TRAIN_ROWS,
    min_holdout_rows: int = HIGH_BOX_OFFICE_MIN_HOLDOUT_ROWS,
    threshold_quantile: float = HIGH_BOX_OFFICE_THRESHOLD_QUANTILE,
    alpha: float = HIGH_BOX_OFFICE_PROFILE_ALPHA,
) -> Dict[str, object]:
    """Use a ridge revenue model to summarise which movie traits align with high box office."""
    prepared = prepare_forecasting_dataframe(df)
    eligible = prepared.loc[
        prepared["primary_genre"].notna()
        & prepared["revenue"].notna()
    ].copy()
    if len(eligible) < int(min_rows):
        return {"error": f"Need at least {int(min_rows)} movies with genre and revenue data to build the ML revenue profile."}

    eligible = eligible.sort_values(["release_date", "title"]).reset_index(drop=True)
    holdout_rows = max(int(round(len(eligible) * float(holdout_fraction))), int(min_holdout_rows))
    use_holdout = len(eligible) - holdout_rows >= int(min_train_rows) and holdout_rows < len(eligible)

    if use_holdout:
        train_df = eligible.iloc[:-holdout_rows].reset_index(drop=True)
        evaluation_df = eligible.iloc[-holdout_rows:].reset_index(drop=True)
        evaluation_scope = "latest holdout movies"
    else:
        train_df = eligible.copy()
        evaluation_df = eligible.copy()
        evaluation_scope = "full current selection"

    training_genre_levels = _select_high_box_office_genre_levels(train_df)
    trained_model = _fit_log_revenue_ridge_model(train_df, alpha=alpha, genre_levels=training_genre_levels)
    evaluation_predictions = _predict_log_revenue_ridge_model(trained_model, evaluation_df)
    evaluation_actual = evaluation_df["revenue"].to_numpy(dtype=float)
    evaluation_metrics = _regression_metrics(evaluation_actual, evaluation_predictions)

    explanation_genre_levels = _select_high_box_office_genre_levels(eligible)
    explanation_model = _fit_log_revenue_ridge_model(eligible, alpha=alpha, genre_levels=explanation_genre_levels)
    full_predictions = _predict_log_revenue_ridge_model(explanation_model, eligible)
    threshold_value = float(eligible["revenue"].quantile(float(threshold_quantile)))

    predicted_high_mask = full_predictions >= threshold_value
    if int(predicted_high_mask.sum()) < max(5, int(len(eligible) * 0.12)):
        predicted_cutoff = float(np.quantile(full_predictions, float(threshold_quantile)))
        predicted_high_mask = full_predictions >= predicted_cutoff
    if int(predicted_high_mask.sum()) == 0:
        predicted_high_mask = eligible["revenue"] >= threshold_value

    profile_df = eligible.loc[predicted_high_mask].copy()
    profile_df["release_quarter"] = profile_df["month"].map(_month_to_quarter)

    coefficient_values = pd.Series(
        explanation_model["coefficients"][1:],
        index=explanation_model["feature_columns"],
        dtype=float,
    ).sort_values(ascending=False)

    top_positive = _top_feature_rows(coefficient_values, positive=True)
    top_negative = _top_feature_rows(coefficient_values, positive=False)

    top_genres = _series_top_values(profile_df["primary_genre"], limit=3)
    top_release_windows = _series_top_values(profile_df["release_quarter"], limit=2)
    english_share = profile_df["language"].astype("string").str.contains("en|english", case=False, na=False, regex=True).mean()
    us_share = profile_df["country"].astype("string").str.contains("United States of America", case=False, na=False, regex=False).mean()

    return {
        "overview": {
            "model_name": "ridge_log_revenue_profile",
            "model_label": "Ridge Revenue Profile",
            "sample_rows": int(len(eligible)),
            "training_rows": int(len(train_df)),
            "evaluation_rows": int(len(evaluation_df)),
            "evaluation_scope": evaluation_scope,
            "high_revenue_threshold": round(threshold_value, 2),
            "high_revenue_quantile": float(threshold_quantile),
            "r_squared": round(float(evaluation_metrics["r_squared"]), 4),
            "mae": round(float(evaluation_metrics["mae"]), 2),
            "wape_percent": round(float(evaluation_metrics["wape_percent"]), 2),
            "predicted_high_rows": int(len(profile_df)),
        },
        "top_positive_features": top_positive,
        "top_negative_features": top_negative,
        "high_revenue_profile": {
            "median_budget": round(float(profile_df["budget"].median()), 2) if profile_df["budget"].notna().any() else 0.0,
            "median_runtime": round(float(profile_df["runtime"].median()), 2) if profile_df["runtime"].notna().any() else 0.0,
            "top_genres": top_genres,
            "top_release_windows": top_release_windows,
            "english_share_percent": round(float(english_share) * 100.0, 2) if np.isfinite(english_share) else 0.0,
            "us_production_share_percent": round(float(us_share) * 100.0, 2) if np.isfinite(us_share) else 0.0,
        },
    }


def _evaluate_genre_revenue_models_prepared(
    prepared: pd.DataFrame,
    validation_year: int = FORECAST_VALIDATION_YEAR,
    min_train_rows: int = FORECAST_MIN_TRAIN_ROWS,
    alpha: float = FORECAST_ALPHA,
    model_name: str = FORECAST_DEFAULT_MODEL,
) -> Dict[str, object]:
    """Train and validate per-genre revenue models on a prepared dataframe."""
    eligible = prepared.loc[
        prepared["primary_genre"].notna()
        & prepared["revenue"].notna()
    ].copy()

    train_df = eligible.loc[eligible["year"] < int(validation_year)].copy()
    validation_df = eligible.loc[eligible["year"] == int(validation_year)].copy()

    if train_df.empty:
        return {"error": "No training rows are available before the validation year."}
    if validation_df.empty:
        return {"error": "No validation rows are available for the requested year."}

    train_genres = set(train_df["primary_genre"].dropna().astype(str).tolist())
    validation_df = validation_df.loc[validation_df["primary_genre"].astype(str).isin(train_genres)].copy()
    if validation_df.empty:
        return {"error": "No validation genres have matching training history before the validation year."}

    model = _fit_forecast_model(train_df, model_name=model_name, alpha=alpha)
    validation_df = validation_df.reset_index(drop=True)
    validation_df["predicted_revenue"] = _predict_forecast_model(model, validation_df)

    prediction_rows: list[Dict[str, object]] = []
    summary_rows: list[Dict[str, object]] = []
    predicted_chunks: list[np.ndarray] = []
    actual_chunks: list[np.ndarray] = []

    for genre, genre_validation in validation_df.groupby("primary_genre", sort=True):
        genre_train = train_df.loc[train_df["primary_genre"] == genre].copy()
        predicted = genre_validation["predicted_revenue"].to_numpy(dtype=float)
        model_type = _build_model_type_label(model_name, len(genre_train), int(min_train_rows))

        actual = genre_validation["revenue"].to_numpy(dtype=float)
        metrics = _regression_metrics(actual, predicted)
        predicted_chunks.append(predicted)
        actual_chunks.append(actual)

        validation_total = float(np.sum(actual))
        predicted_total = float(np.sum(predicted))
        revenue_bias_percent = ((predicted_total - validation_total) / validation_total * 100.0) if validation_total else 0.0

        summary_rows.append(
            {
                "genre": str(genre),
                "model_type": model_type,
                "train_samples": int(len(genre_train)),
                "validation_samples": int(len(genre_validation)),
                "mae": round(metrics["mae"], 2),
                "rmse": round(metrics["rmse"], 2),
                "mape_percent": round(metrics["mape_percent"], 2),
                "wape_percent": round(metrics["wape_percent"], 2),
                "r_squared": round(metrics["r_squared"], 4),
                "actual_total_revenue": round(validation_total, 2),
                "predicted_total_revenue": round(predicted_total, 2),
                "revenue_bias_percent": round(revenue_bias_percent, 2),
                "validation_start": genre_validation["release_date"].min().strftime("%Y-%m-%d"),
                "validation_end": genre_validation["release_date"].max().strftime("%Y-%m-%d"),
            }
        )

        for (_, row), predicted_value, actual_value in zip(genre_validation.iterrows(), predicted, actual):
            absolute_error = abs(predicted_value - actual_value)
            ape_percent = absolute_error / actual_value * 100.0 if actual_value else 0.0
            prediction_rows.append(
                {
                    "genre": str(genre),
                    "title": str(row["title"]),
                    "release_date": row["release_date"].strftime("%Y-%m-%d"),
                    "budget": round(float(row["budget"]), 2),
                    "budget_observed": bool(row["budget_observed"]) if pd.notna(row["budget_observed"]) else False,
                    "runtime": round(float(row["runtime"]), 2) if pd.notna(row["runtime"]) else np.nan,
                    "runtime_observed": bool(row["runtime_observed"]) if pd.notna(row["runtime_observed"]) else False,
                    "actual_revenue": round(float(actual_value), 2),
                    "revenue_observed": bool(row["revenue_observed"]) if pd.notna(row["revenue_observed"]) else False,
                    "predicted_revenue": round(float(predicted_value), 2),
                    "absolute_error": round(float(absolute_error), 2),
                    "ape_percent": round(float(ape_percent), 2),
                    "model_type": model_type,
                }
            )

    if not prediction_rows:
        return {"error": "No genre forecast could be evaluated for the requested validation year."}

    overall_actual = np.concatenate(actual_chunks)
    overall_predicted = np.concatenate(predicted_chunks)
    overall_metrics = _regression_metrics(overall_actual, overall_predicted)
    overall_actual_total = float(np.sum(overall_actual))
    overall_predicted_total = float(np.sum(overall_predicted))

    summary_df = pd.DataFrame(summary_rows).sort_values(["validation_samples", "genre"], ascending=[False, True]).reset_index(drop=True)
    predictions_df = pd.DataFrame(prediction_rows).sort_values(["genre", "release_date", "title"]).reset_index(drop=True)

    validation_start = validation_df["release_date"].min().strftime("%Y-%m-%d")
    validation_end = validation_df["release_date"].max().strftime("%Y-%m-%d")
    overview = {
        "model_name": model_name,
        "model_label": FORECAST_MODEL_LABELS.get(model_name, model_name),
        "validation_year": int(validation_year),
        "training_rows": int(len(train_df)),
        "validation_rows": int(len(predictions_df)),
        "genres_evaluated": int(summary_df["genre"].nunique()),
        "overall_mae": round(overall_metrics["mae"], 2),
        "overall_rmse": round(overall_metrics["rmse"], 2),
        "overall_mape_percent": round(overall_metrics["mape_percent"], 2),
        "overall_wape_percent": round(overall_metrics["wape_percent"], 2),
        "overall_r_squared": round(overall_metrics["r_squared"], 4),
        "actual_total_revenue": round(overall_actual_total, 2),
        "predicted_total_revenue": round(overall_predicted_total, 2),
        "validation_start": validation_start,
        "validation_end": validation_end,
        "cutoff_date": DATASET_CUTOFF_DATE.strftime("%Y-%m-%d"),
    }

    return {
        "overview": overview,
        "summary_by_genre": summary_df,
        "predictions": predictions_df,
    }


def evaluate_genre_revenue_models(
    df: pd.DataFrame,
    validation_year: int = FORECAST_VALIDATION_YEAR,
    min_train_rows: int = FORECAST_MIN_TRAIN_ROWS,
    alpha: float = FORECAST_ALPHA,
    model_name: str = FORECAST_DEFAULT_MODEL,
) -> Dict[str, object]:
    """Train and validate per-genre revenue models."""
    prepared = prepare_forecasting_dataframe(df)
    return _evaluate_genre_revenue_models_prepared(
        prepared,
        validation_year=validation_year,
        min_train_rows=min_train_rows,
        alpha=alpha,
        model_name=model_name,
    )


def backtest_genre_revenue_models(
    df: pd.DataFrame,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    min_validation_rows: int = 3,
    max_years: int = 15,
    min_train_rows: int = FORECAST_MIN_TRAIN_ROWS,
    alpha: float = FORECAST_ALPHA,
    model_name: str = FORECAST_DEFAULT_MODEL,
) -> Dict[str, object]:
    """Run rolling holdout validation across multiple years for charting."""
    prepared = prepare_forecasting_dataframe(df)
    eligible = prepared.loc[
        prepared["primary_genre"].notna()
        & prepared["budget"].notna()
        & prepared["revenue"].notna()
    ].copy()
    if eligible.empty:
        return {"error": "No eligible rows are available for forecast backtesting."}

    available_years = sorted(int(year) for year in eligible["year"].dropna().astype(int).unique().tolist())
    if start_year is not None:
        available_years = [year for year in available_years if year >= int(start_year)]
    if end_year is not None:
        available_years = [year for year in available_years if year <= int(end_year)]

    candidate_years = [
        year
        for year in available_years
        if int((eligible["year"] < year).sum()) >= int(min_train_rows)
        and int((eligible["year"] == year).sum()) >= int(min_validation_rows)
    ]
    if not candidate_years:
        return {"error": "No years have enough training and validation rows for forecast backtesting."}

    if max_years and len(candidate_years) > int(max_years):
        candidate_years = candidate_years[-int(max_years):]

    overview_rows: list[Dict[str, object]] = []
    genre_rows: list[pd.DataFrame] = []

    for year in candidate_years:
        yearly_result = _evaluate_genre_revenue_models_prepared(
            prepared,
            validation_year=year,
            min_train_rows=min_train_rows,
            alpha=alpha,
            model_name=model_name,
        )
        if "error" in yearly_result:
            continue

        overview = dict(yearly_result["overview"])
        overview_rows.append(overview)

        summary_by_genre = yearly_result["summary_by_genre"].copy()
        summary_by_genre.insert(0, "validation_year", int(year))
        genre_rows.append(summary_by_genre)

    if not overview_rows:
        return {"error": "Forecast backtesting did not produce any valid yearly results."}

    overview_df = pd.DataFrame(overview_rows).sort_values("validation_year").reset_index(drop=True)
    genre_df = (
        pd.concat(genre_rows, ignore_index=True)
        .sort_values(["validation_year", "validation_samples", "genre"], ascending=[True, False, True])
        .reset_index(drop=True)
    )

    return {
        "yearly_overview": overview_df,
        "yearly_genre_summary": genre_df,
    }


def backtest_multiple_forecast_models(
    df: pd.DataFrame,
    model_names: tuple[str, ...] | list[str] = FORECAST_SUPPORTED_MODELS,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    min_validation_rows: int = 3,
    max_years: int = 15,
    min_train_rows: int = FORECAST_MIN_TRAIN_ROWS,
    alpha: float = FORECAST_ALPHA,
) -> Dict[str, object]:
    """Run rolling holdout validation for multiple forecast models."""
    prepared = prepare_forecasting_dataframe(df)
    eligible = prepared.loc[
        prepared["primary_genre"].notna()
        & prepared["budget"].notna()
        & prepared["revenue"].notna()
    ].copy()
    if eligible.empty:
        return {"error": "No eligible rows are available for multi-model forecast backtesting."}

    available_years = sorted(int(year) for year in eligible["year"].dropna().astype(int).unique().tolist())
    if start_year is not None:
        available_years = [year for year in available_years if year >= int(start_year)]
    if end_year is not None:
        available_years = [year for year in available_years if year <= int(end_year)]

    candidate_years = [
        year
        for year in available_years
        if int((eligible["year"] < year).sum()) >= int(min_train_rows)
        and int((eligible["year"] == year).sum()) >= int(min_validation_rows)
    ]
    if not candidate_years:
        return {"error": "No years have enough training and validation rows for multi-model backtesting."}

    if max_years and len(candidate_years) > int(max_years):
        candidate_years = candidate_years[-int(max_years):]

    selected_models = tuple(dict.fromkeys(str(name).strip() for name in model_names if str(name).strip()))
    if not selected_models:
        return {"error": "No forecast models were selected for multi-model backtesting."}

    invalid_models = [name for name in selected_models if name not in FORECAST_SUPPORTED_MODELS]
    if invalid_models:
        invalid_joined = ", ".join(invalid_models)
        return {"error": f"Unsupported forecast model(s): {invalid_joined}"}

    yearly_rows: list[Dict[str, object]] = []
    for year in candidate_years:
        for model_name in selected_models:
            yearly_result = _evaluate_genre_revenue_models_prepared(
                prepared,
                validation_year=year,
                min_train_rows=min_train_rows,
                alpha=alpha,
                model_name=model_name,
            )
            if "error" in yearly_result:
                continue

            overview = dict(yearly_result["overview"])
            yearly_rows.append(
                {
                    "validation_year": int(overview["validation_year"]),
                    "model_name": model_name,
                    "model_label": FORECAST_MODEL_LABELS.get(model_name, model_name),
                    "training_rows": int(overview["training_rows"]),
                    "validation_rows": int(overview["validation_rows"]),
                    "genres_evaluated": int(overview["genres_evaluated"]),
                    "overall_mae": float(overview["overall_mae"]),
                    "overall_rmse": float(overview["overall_rmse"]),
                    "overall_mape_percent": float(overview["overall_mape_percent"]),
                    "overall_wape_percent": float(overview["overall_wape_percent"]),
                    "overall_r_squared": float(overview["overall_r_squared"]),
                    "actual_total_revenue": float(overview["actual_total_revenue"]),
                    "predicted_total_revenue": float(overview["predicted_total_revenue"]),
                    "validation_start": str(overview["validation_start"]),
                    "validation_end": str(overview["validation_end"]),
                    "cutoff_date": str(overview["cutoff_date"]),
                }
            )

    if not yearly_rows:
        return {"error": "Multi-model forecast backtesting did not produce any valid yearly results."}

    yearly_model_overview = (
        pd.DataFrame(yearly_rows)
        .sort_values(["validation_year", "model_name"])
        .reset_index(drop=True)
    )
    correlation_rows: list[Dict[str, object]] = []
    for (model_name, model_label), model_rows in yearly_model_overview.groupby(["model_name", "model_label"], sort=True):
        correlation_rows.append(
            {
                "model_name": str(model_name),
                "model_label": str(model_label),
                "yearly_revenue_correlation": _correlation_coefficient(
                    model_rows["actual_total_revenue"].to_numpy(dtype=float),
                    model_rows["predicted_total_revenue"].to_numpy(dtype=float),
                ),
            }
        )
    correlation_df = pd.DataFrame(correlation_rows)

    model_summary = (
        yearly_model_overview.groupby(["model_name", "model_label"], as_index=False)
        .agg(
            years_evaluated=("validation_year", "nunique"),
            average_wape_percent=("overall_wape_percent", "mean"),
            average_r_squared=("overall_r_squared", "mean"),
            average_mae=("overall_mae", "mean"),
            average_rmse=("overall_rmse", "mean"),
        )
    )
    model_summary = model_summary.merge(
        correlation_df,
        on=["model_name", "model_label"],
        how="left",
    )
    for column in ("average_wape_percent", "average_r_squared", "average_mae", "average_rmse", "yearly_revenue_correlation"):
        model_summary[column] = model_summary[column].round(4)
    model_summary = (
        model_summary.sort_values(
            ["yearly_revenue_correlation", "average_wape_percent", "average_r_squared", "model_name"],
            ascending=[False, True, False, True],
        )
        .reset_index(drop=True)
    )

    selected_model_name = None
    selected_model_label = None
    selection_metric_value = 0.0
    if not model_summary.empty:
        best_row = model_summary.iloc[0]
        selected_model_name = str(best_row["model_name"])
        selected_model_label = str(best_row["model_label"])
        selection_metric_value = float(best_row["yearly_revenue_correlation"])

    return {
        "yearly_model_overview": yearly_model_overview,
        "model_summary": model_summary,
        "models_evaluated": list(selected_models),
        "selected_model_name": selected_model_name,
        "selected_model_label": selected_model_label,
        "selection_metric": "yearly_revenue_correlation",
        "selection_metric_value": round(selection_metric_value, 4),
    }


def export_forecast_results(
    result: Dict[str, object],
    output_dir: str | Path = "outputs",
    validation_year: int = FORECAST_VALIDATION_YEAR,
    cache_metadata: Optional[Dict[str, object]] = None,
) -> Dict[str, str]:
    """Export forecast validation outputs."""
    output_root = Path(output_dir)
    if not output_root.is_absolute():
        output_root = Path(__file__).resolve().parent / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    genre_summary_path = output_root / f"forecast_{validation_year}_genre_summary.csv"
    predictions_path = output_root / f"forecast_{validation_year}_movie_predictions.csv"
    metadata_path = output_root / f"forecast_{validation_year}_overview.json"

    result["summary_by_genre"].to_csv(genre_summary_path, index=False)
    result["predictions"].to_csv(predictions_path, index=False)
    overview_payload = dict(result["overview"])
    if cache_metadata:
        overview_payload["cache_metadata"] = cache_metadata
    metadata_path.write_text(json.dumps(overview_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "genre_summary": str(genre_summary_path),
        "movie_predictions": str(predictions_path),
        "overview": str(metadata_path),
    }


def _resolve_output_root(output_dir: str | Path) -> Path:
    output_root = Path(output_dir)
    if not output_root.is_absolute():
        output_root = Path(__file__).resolve().parent / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


def _build_forecast_output_paths(output_dir: str | Path, validation_year: int) -> Dict[str, Path]:
    output_root = _resolve_output_root(output_dir)
    return {
        "output_root": output_root,
        "genre_summary": output_root / f"forecast_{validation_year}_genre_summary.csv",
        "movie_predictions": output_root / f"forecast_{validation_year}_movie_predictions.csv",
        "overview": output_root / f"forecast_{validation_year}_overview.json",
    }


def _build_forecast_cache_metadata(
    *,
    dataset_path: Path,
    cutoff_date: pd.Timestamp,
    validation_year: int,
    min_train_rows: int,
    alpha: float,
    model_name: str,
) -> Dict[str, object]:
    stat = dataset_path.stat()
    return {
        "version": FORECAST_CACHE_METADATA_VERSION,
        "dataset_path": str(dataset_path.resolve()),
        "dataset_size": int(stat.st_size),
        "dataset_mtime_ns": int(stat.st_mtime_ns),
        "cutoff_date": cutoff_date.strftime("%Y-%m-%d"),
        "validation_year": int(validation_year),
        "min_train_rows": int(min_train_rows),
        "alpha": float(alpha),
        "model_name": str(model_name),
    }


def _matches_forecast_cache(expected: Dict[str, object], observed: object) -> bool:
    if not isinstance(observed, dict):
        return False
    return observed == expected


def load_saved_forecast_results(
    output_dir: str | Path = "outputs",
    validation_year: int = FORECAST_VALIDATION_YEAR,
) -> Dict[str, object]:
    """Load saved forecast outputs from disk."""
    paths = _build_forecast_output_paths(output_dir, validation_year)

    overview_path = paths["overview"]
    genre_summary_path = paths["genre_summary"]
    predictions_path = paths["movie_predictions"]
    required_paths = (overview_path, genre_summary_path, predictions_path)
    if not all(path.exists() for path in required_paths):
        missing = [str(path) for path in required_paths if not path.exists()]
        return {"error": f"Saved forecast outputs are missing: {', '.join(missing)}"}

    try:
        overview = json.loads(overview_path.read_text(encoding="utf-8"))
        summary_by_genre = pd.read_csv(genre_summary_path)
        predictions = pd.read_csv(predictions_path)
    except (OSError, json.JSONDecodeError, pd.errors.EmptyDataError) as exc:
        return {"error": f"Saved forecast outputs could not be loaded: {exc}"}

    return {
        "overview": overview,
        "summary_by_genre": summary_by_genre,
        "predictions": predictions,
        "paths": {key: str(value) for key, value in paths.items() if key != "output_root"},
    }


def run_genre_box_office_forecast(
    dataset_path: Optional[str | Path] = None,
    output_dir: str | Path = "outputs",
    validation_year: int = FORECAST_VALIDATION_YEAR,
    cutoff_date: pd.Timestamp = DATASET_CUTOFF_DATE,
    min_train_rows: int = FORECAST_MIN_TRAIN_ROWS,
    alpha: float = FORECAST_ALPHA,
    model_name: str = FORECAST_DEFAULT_MODEL,
) -> Dict[str, object]:
    """Load data, run a forecast validation split, and export the outputs."""
    resolved_dataset_path = resolve_cleaned_dataset_path(dataset_path)
    cache_metadata = _build_forecast_cache_metadata(
        dataset_path=resolved_dataset_path,
        cutoff_date=cutoff_date,
        validation_year=validation_year,
        min_train_rows=min_train_rows,
        alpha=alpha,
        model_name=model_name,
    )
    cached_result = load_saved_forecast_results(output_dir=output_dir, validation_year=validation_year)
    if "error" not in cached_result and _matches_forecast_cache(
        cache_metadata,
        cached_result["overview"].get("cache_metadata"),
    ):
        forecasting_df, metadata = load_forecasting_dataset(
            dataset_path=resolved_dataset_path,
            cutoff_date=cutoff_date,
        )
        cached_result["metadata"] = metadata
        return cached_result

    forecasting_df, metadata = load_forecasting_dataset(
        dataset_path=resolved_dataset_path,
        cutoff_date=cutoff_date,
    )
    result = evaluate_genre_revenue_models(
        forecasting_df,
        validation_year=validation_year,
        min_train_rows=min_train_rows,
        alpha=alpha,
        model_name=model_name,
    )
    if "error" in result:
        result["metadata"] = metadata
        return result

    paths = export_forecast_results(
        result,
        output_dir=output_dir,
        validation_year=validation_year,
        cache_metadata=cache_metadata,
    )
    result["metadata"] = metadata
    result["paths"] = paths
    return result


if __name__ == "__main__":
    forecast_result = run_genre_box_office_forecast()
    if "error" in forecast_result:
        print(forecast_result["error"])
    else:
        print(forecast_result["overview"])
        print(forecast_result["paths"])
