"""Genre-level box office forecasting with a 2017 holdout validation set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from data_preprocessing import (
    DATASET_CUTOFF_DATE,
    prepare_dataset,
    resolve_cleaned_dataset_path,
)

FORECAST_VALIDATION_YEAR = 2017
FORECAST_ALPHA = 8.0
FORECAST_MIN_TRAIN_ROWS = 12
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


def _fit_ridge_model(train_df: pd.DataFrame, alpha: float = FORECAST_ALPHA) -> Dict[str, object]:
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
    penalty = np.eye(design.shape[1]) * float(alpha)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ target)

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
        "global_calibration": global_calibration,
        "genre_calibrations": genre_calibrations,
    }


def _predict_ridge(model: Dict[str, object], df: pd.DataFrame) -> np.ndarray:
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


def _constant_prediction(train_df: pd.DataFrame, rows: int) -> np.ndarray:
    constant = float(train_df["revenue"].median()) if not train_df.empty else 0.0
    return np.full(rows, constant, dtype=float)


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


def _evaluate_genre_revenue_models_prepared(
    prepared: pd.DataFrame,
    validation_year: int = FORECAST_VALIDATION_YEAR,
    min_train_rows: int = FORECAST_MIN_TRAIN_ROWS,
    alpha: float = FORECAST_ALPHA,
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

    model = _fit_ridge_model(train_df, alpha=alpha)
    validation_df = validation_df.reset_index(drop=True)
    validation_df["predicted_revenue"] = _predict_ridge(model, validation_df)

    prediction_rows: list[Dict[str, object]] = []
    summary_rows: list[Dict[str, object]] = []
    predicted_chunks: list[np.ndarray] = []
    actual_chunks: list[np.ndarray] = []

    for genre, genre_validation in validation_df.groupby("primary_genre", sort=True):
        genre_train = train_df.loc[train_df["primary_genre"] == genre].copy()
        predicted = genre_validation["predicted_revenue"].to_numpy(dtype=float)
        model_type = "global_ridge_calibrated" if len(genre_train) >= int(min_train_rows) else "global_ridge_shared"

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
) -> Dict[str, object]:
    """Train and validate per-genre revenue models."""
    prepared = prepare_forecasting_dataframe(df)
    return _evaluate_genre_revenue_models_prepared(
        prepared,
        validation_year=validation_year,
        min_train_rows=min_train_rows,
        alpha=alpha,
    )


def backtest_genre_revenue_models(
    df: pd.DataFrame,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    min_validation_rows: int = 3,
    max_years: int = 15,
    min_train_rows: int = FORECAST_MIN_TRAIN_ROWS,
    alpha: float = FORECAST_ALPHA,
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


def export_forecast_results(
    result: Dict[str, object],
    output_dir: str | Path = "outputs",
    validation_year: int = FORECAST_VALIDATION_YEAR,
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
    metadata_path.write_text(json.dumps(result["overview"], indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "genre_summary": str(genre_summary_path),
        "movie_predictions": str(predictions_path),
        "overview": str(metadata_path),
    }


def run_genre_box_office_forecast(
    dataset_path: Optional[str | Path] = None,
    output_dir: str | Path = "outputs",
    validation_year: int = FORECAST_VALIDATION_YEAR,
    cutoff_date: pd.Timestamp = DATASET_CUTOFF_DATE,
    min_train_rows: int = FORECAST_MIN_TRAIN_ROWS,
    alpha: float = FORECAST_ALPHA,
) -> Dict[str, object]:
    """Load data, run the 2017 forecast validation, and export the outputs."""
    forecasting_df, metadata = load_forecasting_dataset(
        dataset_path=dataset_path,
        cutoff_date=cutoff_date,
    )
    result = evaluate_genre_revenue_models(
        forecasting_df,
        validation_year=validation_year,
        min_train_rows=min_train_rows,
        alpha=alpha,
    )
    if "error" in result:
        result["metadata"] = metadata
        return result

    paths = export_forecast_results(result, output_dir=output_dir, validation_year=validation_year)
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
