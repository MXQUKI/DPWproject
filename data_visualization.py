"""Visualization utilities for the IMDB movies project."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_ROOT = PROJECT_ROOT / ".cache"
MPL_CACHE_DIR = CACHE_ROOT / "matplotlib"
FONTCONFIG_CACHE_DIR = CACHE_ROOT / "fontconfig"

for cache_dir in (MPL_CACHE_DIR, FONTCONFIG_CACHE_DIR):
    cache_dir.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from box_office_forecasting import (
    FORECAST_MODEL_LABELS,
    FORECAST_SUPPORTED_MODELS,
    backtest_multiple_forecast_models,
)
from data_analysis import aggregate_by_genre, yearly_rating_trend

FORECAST_YEAR_SELECTION_START = 2003
FORECAST_YEAR_SELECTION_END = 2017
FORECAST_BACKTEST_OVERVIEW_FILENAME = "forecast_backtest_yearly_overview.csv"
FORECAST_BACKTEST_METADATA_FILENAME = "forecast_backtest_metadata.json"
FORECAST_MULTI_MODEL_OVERVIEW_FILENAME = "forecast_model_backtest_yearly_overview.csv"
FORECAST_MULTI_MODEL_SUMMARY_FILENAME = "forecast_model_backtest_summary.csv"
FORECAST_MULTI_MODEL_METADATA_FILENAME = "forecast_model_backtest_metadata.json"
FORECAST_CACHE_SIGNATURE_COLUMNS = (
    "id",
    "source_id",
    "title",
    "primary_genre",
    "budget",
    "revenue",
    "runtime",
    "release_date",
    "language",
    "country",
)


def _build_forecast_cache_signature(df: pd.DataFrame) -> Dict[str, object]:
    relevant_columns = [column for column in FORECAST_CACHE_SIGNATURE_COLUMNS if column in df.columns]
    if not relevant_columns:
        return {
            "row_count": int(len(df)),
            "columns": [],
            "fingerprint": "0",
        }

    working = df[relevant_columns].copy()
    normalized = pd.DataFrame(index=working.index)
    for column in relevant_columns:
        series = working[column]
        if column == "release_date":
            normalized[column] = pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
        elif pd.api.types.is_numeric_dtype(series):
            normalized[column] = pd.to_numeric(series, errors="coerce").fillna(-1.0)
        else:
            normalized[column] = series.astype("string").fillna("").str.strip()

    sort_columns = [column for column in ("id", "source_id", "release_date", "title", "primary_genre") if column in normalized.columns]
    if sort_columns:
        normalized = normalized.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)

    fingerprint = int(pd.util.hash_pandas_object(normalized, index=False).astype("uint64").sum())
    return {
        "row_count": int(len(normalized)),
        "columns": relevant_columns,
        "fingerprint": str(fingerprint),
    }


def _load_cached_forecast_frames(
    output_root: Path,
    *,
    metadata_filename: str,
    frame_filenames: Dict[str, str],
    dataset_signature: Dict[str, object],
    params: Dict[str, object],
) -> Optional[Dict[str, object]]:
    metadata_path = output_root / metadata_filename
    if not metadata_path.exists():
        return None

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if metadata.get("dataset_signature") != dataset_signature:
        return None
    if metadata.get("params") != params:
        return None

    loaded: Dict[str, object] = {}
    for key, filename in frame_filenames.items():
        frame_path = output_root / filename
        if not frame_path.exists():
            return None
        loaded[key] = pd.read_csv(frame_path)

    if "models_evaluated" in metadata:
        loaded["models_evaluated"] = metadata["models_evaluated"]
    for key in (
        "selected_model_name",
        "selected_model_label",
        "selection_metric",
        "selection_metric_value",
    ):
        if key in metadata:
            loaded[key] = metadata[key]
    return loaded


def _write_cached_forecast_frames(
    output_root: Path,
    *,
    metadata_filename: str,
    frame_filenames: Dict[str, str],
    payload: Dict[str, object],
    dataset_signature: Dict[str, object],
    params: Dict[str, object],
) -> None:
    for key, filename in frame_filenames.items():
        frame = payload.get(key)
        if not isinstance(frame, pd.DataFrame):
            continue
        frame.to_csv(output_root / filename, index=False)

    metadata: Dict[str, object] = {
        "dataset_signature": dataset_signature,
        "params": params,
    }
    if "models_evaluated" in payload:
        metadata["models_evaluated"] = list(payload["models_evaluated"])
    for key in (
        "selected_model_name",
        "selected_model_label",
        "selection_metric",
        "selection_metric_value",
    ):
        if key in payload:
            metadata[key] = payload[key]
    (output_root / metadata_filename).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _get_or_create_forecast_backtest_result(df: pd.DataFrame, output_root: Path) -> Dict[str, object]:
    multi_model_result = _get_or_create_multi_model_backtest_result(df, output_root)
    if "error" in multi_model_result:
        return multi_model_result

    yearly_model_overview = multi_model_result.get("yearly_model_overview")
    if not isinstance(yearly_model_overview, pd.DataFrame) or yearly_model_overview.empty:
        return {"error": "Multi-model forecast backtest overview is empty or unavailable."}

    selected_model_name = str(
        multi_model_result.get("selected_model_name")
        or ""
    ).strip()
    if not selected_model_name:
        return {"error": "No best forecast model could be selected for yearly comparison."}

    selected_rows = (
        yearly_model_overview.loc[yearly_model_overview["model_name"].astype(str) == selected_model_name]
        .copy()
        .sort_values("validation_year")
        .reset_index(drop=True)
    )
    required = {
        "validation_year",
        "actual_total_revenue",
        "predicted_total_revenue",
        "overall_wape_percent",
    }
    if selected_rows.empty or not required.issubset(selected_rows.columns):
        return {"error": "Selected forecast model does not have a valid yearly comparison overview."}

    yearly_overview = selected_rows[
        [
            "validation_year",
            "actual_total_revenue",
            "predicted_total_revenue",
            "overall_wape_percent",
            "overall_r_squared",
            "validation_rows",
            "training_rows",
            "genres_evaluated",
            "model_name",
            "model_label",
            "validation_start",
            "validation_end",
            "cutoff_date",
        ]
    ].copy()
    result: Dict[str, object] = {
        "yearly_overview": yearly_overview,
        "selected_model_name": selected_model_name,
        "selected_model_label": multi_model_result.get("selected_model_label"),
        "selection_metric": multi_model_result.get("selection_metric"),
        "selection_metric_value": multi_model_result.get("selection_metric_value"),
    }

    dataset_signature = _build_forecast_cache_signature(df)
    params = {
        "start_year": FORECAST_YEAR_SELECTION_START,
        "end_year": FORECAST_YEAR_SELECTION_END,
        "max_years": 0,
        "model_names": list(FORECAST_SUPPORTED_MODELS),
        "result_type": "selected_model_yearly_comparison",
        "selection_metric": "yearly_revenue_correlation",
    }
    _write_cached_forecast_frames(
        output_root,
        metadata_filename=FORECAST_BACKTEST_METADATA_FILENAME,
        frame_filenames={
            "yearly_overview": FORECAST_BACKTEST_OVERVIEW_FILENAME,
        },
        payload=result,
        dataset_signature=dataset_signature,
        params=params,
    )
    return result


def _get_or_create_multi_model_backtest_result(df: pd.DataFrame, output_root: Path) -> Dict[str, object]:
    dataset_signature = _build_forecast_cache_signature(df)
    params = {
        "start_year": FORECAST_YEAR_SELECTION_START,
        "end_year": FORECAST_YEAR_SELECTION_END,
        "max_years": 0,
        "model_names": list(FORECAST_SUPPORTED_MODELS),
        "result_type": "multi_model_backtest",
    }
    cached = _load_cached_forecast_frames(
        output_root,
        metadata_filename=FORECAST_MULTI_MODEL_METADATA_FILENAME,
        frame_filenames={
            "yearly_model_overview": FORECAST_MULTI_MODEL_OVERVIEW_FILENAME,
            "model_summary": FORECAST_MULTI_MODEL_SUMMARY_FILENAME,
        },
        dataset_signature=dataset_signature,
        params=params,
    )
    if cached is not None:
        return cached

    result = backtest_multiple_forecast_models(
        df,
        model_names=FORECAST_SUPPORTED_MODELS,
        start_year=FORECAST_YEAR_SELECTION_START,
        end_year=FORECAST_YEAR_SELECTION_END,
        max_years=0,
    )
    if "error" not in result:
        _write_cached_forecast_frames(
            output_root,
            metadata_filename=FORECAST_MULTI_MODEL_METADATA_FILENAME,
            frame_filenames={
                "yearly_model_overview": FORECAST_MULTI_MODEL_OVERVIEW_FILENAME,
                "model_summary": FORECAST_MULTI_MODEL_SUMMARY_FILENAME,
            },
            payload=result,
            dataset_signature=dataset_signature,
            params=params,
        )
    return result


class DataVisualizer:
    """Generate presentation-ready charts for the project outputs."""

    def __init__(self) -> None:
        plt.rcParams["font.family"] = ["Times New Roman", "DejaVu Serif"]
        plt.rcParams["font.size"] = 14
        plt.rcParams["axes.labelsize"] = 16
        plt.rcParams["axes.titlesize"] = 20
        plt.rcParams["legend.fontsize"] = 13
        plt.rcParams["xtick.labelsize"] = 13
        plt.rcParams["ytick.labelsize"] = 13
        self.color_palette = [
            "#33658A",
            "#86BBD8",
            "#758E4F",
            "#F6AE2D",
            "#F26419",
            "#7D82B8",
            "#BFCDE0",
            "#4F5D75",
            "#D9BF77",
            "#C8553D",
        ]

    def _prepare_output(self, output_path: str | Path) -> Path:
        destination = Path(output_path)
        if not destination.is_absolute():
            destination = Path(__file__).resolve().parent / destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        return destination

    def _apply_axes_style(self, ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
        ax.set_title(title, pad=18, fontweight="bold")
        ax.set_xlabel(xlabel, fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", pad=6)

    @staticmethod
    def _currency_formatter(value: float, _position: int) -> str:
        magnitude = abs(float(value))
        if magnitude >= 1_000_000_000:
            return f"${value / 1_000_000_000:.1f}B"
        if magnitude >= 1_000_000:
            return f"${value / 1_000_000:.0f}M"
        if magnitude >= 1_000:
            return f"${value / 1_000:.0f}K"
        return f"${value:,.0f}"

    @staticmethod
    def _format_validation_year_label(year: int) -> str:
        if int(year) == 2017:
            return "2017 (Jan-Jul only)"
        return str(int(year))

    def draw_genre_bar_chart(
        self,
        genre_data: pd.DataFrame,
        output_path: str | Path = "outputs/genre_distribution.png",
        figsize: Tuple[float, float] = (13.5, 8.0),
        dpi: int = 240,
        title: Optional[str] = None,
        top_n: Optional[int] = 12,
    ) -> Dict[str, Any]:
        """Draw a horizontal bar chart for movie volume by genre."""
        if genre_data.empty:
            raise ValueError("Genre data is empty.")
        if not {"genre", "movie_count"}.issubset(genre_data.columns):
            raise KeyError("Genre data must contain 'genre' and 'movie_count'.")

        plot_data = genre_data.copy()
        plot_data = plot_data.loc[plot_data["genre"].notna()]
        if top_n:
            plot_data = plot_data.head(top_n)
        plot_data = plot_data.sort_values("movie_count", ascending=True)

        output = self._prepare_output(output_path)
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        colors = (self.color_palette * ((len(plot_data) // len(self.color_palette)) + 1))[: len(plot_data)]
        bars = ax.barh(plot_data["genre"], plot_data["movie_count"], color=colors, edgecolor="white")
        self._apply_axes_style(
            ax,
            title or "Movie Distribution by Genre",
            xlabel="Number of Movies",
            ylabel="Genre",
        )

        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + max(plot_data["movie_count"]) * 0.012,
                bar.get_y() + bar.get_height() / 2,
                f"{int(width):,}",
                va="center",
                fontsize=13,
                fontweight="bold",
            )

        plt.tight_layout()
        fig.savefig(output, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        return {
            "output_path": str(output),
            "chart": "genre_distribution",
            "num_genres": int(len(plot_data)),
            "top_genre": str(plot_data.iloc[-1]["genre"]),
            "top_genre_count": int(plot_data.iloc[-1]["movie_count"]),
        }

    def draw_yearly_rating_trend(
        self,
        yearly_data: pd.DataFrame,
        output_path: str | Path = "outputs/yearly_rating_trend.png",
        figsize: Tuple[float, float] = (13.5, 8.0),
        dpi: int = 240,
    ) -> Dict[str, Any]:
        """Draw a line chart for yearly rating trend."""
        if yearly_data.empty:
            raise ValueError("Yearly data is empty.")
        if not {"year", "avg_rating", "movie_count"}.issubset(yearly_data.columns):
            raise KeyError("Yearly data must contain 'year', 'avg_rating', and 'movie_count'.")

        plot_data = yearly_data.copy()
        plot_data["year"] = pd.to_numeric(plot_data["year"], errors="coerce")
        plot_data = plot_data.dropna(subset=["year"]).sort_values("year").copy()
        if plot_data.empty:
            raise ValueError("Yearly data does not contain any valid years.")
        plot_data["year"] = plot_data["year"].astype(int)
        min_year = int(plot_data["year"].min())
        max_year = int(plot_data["year"].max())

        output = self._prepare_output(output_path)
        fig, ax1 = plt.subplots(figsize=figsize, dpi=dpi)
        ax1.plot(plot_data["year"], plot_data["avg_rating"], color="#33658A", linewidth=3.1, marker="o", markersize=5.5)
        self._apply_axes_style(ax1, f"Average Rating Trend by Year ({min_year}-{max_year})", "Year", "Average Rating")
        ax1.set_ylim(0, 10)
        ax1.set_xlim(min_year - 0.5, max_year + 0.5)

        ax2 = ax1.twinx()
        ax2.bar(plot_data["year"], plot_data["movie_count"], alpha=0.2, color="#F6AE2D", width=0.8)
        ax2.set_ylabel("Movie Count", fontweight="bold")
        ax2.spines["top"].set_visible(False)

        plt.tight_layout()
        fig.savefig(output, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        return {
            "output_path": str(output),
            "chart": "yearly_rating_trend",
            "start_year": min_year,
            "end_year": max_year,
        }

    def draw_budget_revenue_scatter(
        self,
        df: pd.DataFrame,
        output_path: str | Path = "outputs/budget_revenue_scatter.png",
        figsize: Tuple[float, float] = (13.5, 8.0),
        dpi: int = 240,
    ) -> Dict[str, Any]:
        """Draw a budget vs revenue scatter plot with a regression line."""
        required_columns = {"budget", "revenue"}
        if not required_columns.issubset(df.columns):
            raise KeyError("Dataset must contain 'budget' and 'revenue'.")

        plot_data = df.copy()
        plot_data["budget"] = pd.to_numeric(plot_data["budget"], errors="coerce")
        plot_data["revenue"] = pd.to_numeric(plot_data["revenue"], errors="coerce")
        plot_data = plot_data.loc[
            (plot_data["budget"] > 0)
            & (plot_data["revenue"] > 0)
            & np.isfinite(plot_data["budget"])
            & np.isfinite(plot_data["revenue"])
        ]
        if len(plot_data) < 2:
            raise ValueError("Need at least two valid rows to plot budget vs revenue.")

        output = self._prepare_output(output_path)
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.scatter(plot_data["budget"], plot_data["revenue"], alpha=0.42, s=42, color="#33658A", edgecolors="none")

        x_values = plot_data["budget"].to_numpy()
        y_values = plot_data["revenue"].to_numpy()
        log_x = np.log10(x_values)
        log_y = np.log10(y_values)
        slope, intercept = np.polyfit(log_x, log_y, 1)
        regression_x = np.geomspace(x_values.min(), x_values.max(), 200)
        regression_y = 10 ** (slope * np.log10(regression_x) + intercept)
        ax.plot(regression_x, regression_y, color="#F26419", linewidth=3.0, label="Trend line")

        self._apply_axes_style(ax, "Budget vs Revenue", "Budget (USD)", "Revenue (USD)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(frameon=False)

        plt.tight_layout()
        fig.savefig(output, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        return {
            "output_path": str(output),
            "chart": "budget_revenue_scatter",
            "sample_size": int(len(plot_data)),
            "slope": round(float(slope), 4),
        }

    def draw_genre_comparison_chart(
        self,
        genre_data: pd.DataFrame,
        output_path: str | Path = "outputs/genre_comparison.png",
        figsize: Tuple[float, float] = (13.5, 8.6),
        dpi: int = 240,
    ) -> Dict[str, Any]:
        """Compare movie count and average rating by genre."""
        if genre_data.empty:
            raise ValueError("Genre data is empty.")
        required = {"genre", "movie_count", "avg_rating"}
        if not required.issubset(genre_data.columns):
            raise KeyError("Genre comparison needs 'genre', 'movie_count', and 'avg_rating'.")

        plot_data = genre_data.head(12).copy()
        output = self._prepare_output(output_path)

        fig, ax1 = plt.subplots(figsize=figsize, dpi=dpi)
        x_positions = np.arange(len(plot_data))
        colors = (self.color_palette * ((len(plot_data) // len(self.color_palette)) + 1))[: len(plot_data)]
        ax1.bar(x_positions, plot_data["movie_count"], color=colors, alpha=0.9, label="Movie Count")
        ax1.set_ylabel("Movie Count", fontweight="bold")
        ax1.set_xticks(x_positions)
        ax1.set_xticklabels(plot_data["genre"], rotation=45, ha="right")
        self._apply_axes_style(ax1, "Genre Volume and Average Rating", "Genre", "Movie Count")

        ax2 = ax1.twinx()
        ax2.plot(x_positions, plot_data["avg_rating"], color="#C8553D", marker="o", linewidth=3.0, label="Average Rating")
        ax2.set_ylabel("Average Rating", fontweight="bold")
        ax2.set_ylim(0, 10)
        ax2.spines["top"].set_visible(False)

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="upper left")

        plt.tight_layout()
        fig.savefig(output, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        return {
            "output_path": str(output),
            "chart": "genre_comparison",
            "num_genres": int(len(plot_data)),
        }

    def draw_forecast_validation_genre_comparison(
        self,
        forecast_result: Dict[str, object],
        output_path: str | Path = "outputs/forecast_validation_genre_comparison.png",
        figsize: Tuple[float, float] = (14.0, 8.6),
        dpi: int = 240,
        top_n: int = 10,
    ) -> Dict[str, Any]:
        """Draw actual vs predicted revenue totals by genre for the holdout year."""
        if "error" in forecast_result:
            raise ValueError(str(forecast_result["error"]))

        summary = forecast_result["summary_by_genre"].copy()
        overview = dict(forecast_result["overview"])
        required = {"genre", "actual_total_revenue", "predicted_total_revenue"}
        if summary.empty or not required.issubset(summary.columns):
            raise ValueError("Forecast validation summary is empty or incomplete.")

        plot_data = summary.loc[summary["genre"].notna()].copy()
        plot_data["actual_total_revenue"] = pd.to_numeric(plot_data["actual_total_revenue"], errors="coerce")
        plot_data["predicted_total_revenue"] = pd.to_numeric(plot_data["predicted_total_revenue"], errors="coerce")
        plot_data = plot_data.dropna(subset=["actual_total_revenue", "predicted_total_revenue"])
        if plot_data.empty:
            raise ValueError("Forecast validation summary does not contain valid revenue totals.")

        plot_data = plot_data.sort_values("actual_total_revenue", ascending=False).head(top_n).reset_index(drop=True)
        validation_year = int(overview["validation_year"])
        output = self._prepare_output(output_path)

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        x_positions = np.arange(len(plot_data))
        width = 0.38
        ax.bar(
            x_positions - width / 2,
            plot_data["actual_total_revenue"],
            width=width,
            color="#33658A",
            label=f"Actual {validation_year}",
        )
        ax.bar(
            x_positions + width / 2,
            plot_data["predicted_total_revenue"],
            width=width,
            color="#F26419",
            label="Predicted",
        )

        self._apply_axes_style(
            ax,
            f"Genre Revenue Forecast Validation ({validation_year} Holdout)",
            xlabel="Genre",
            ylabel="Total Revenue (USD)",
        )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(plot_data["genre"], rotation=35, ha="right")
        ax.yaxis.set_major_formatter(FuncFormatter(self._currency_formatter))
        ax.legend(frameon=False, ncol=2, loc="upper left")

        footer = (
            f"Validation rows: {int(overview['validation_rows']):,}  |  "
            f"Genres: {int(overview['genres_evaluated']):,}  |  "
            f"WAPE: {float(overview['overall_wape_percent']):.1f}%  |  "
            f"R²: {float(overview['overall_r_squared']):.2f}"
        )
        fig.subplots_adjust(bottom=0.27)
        fig.text(0.5, 0.04, footer, ha="center", fontsize=12, color="#4F5D75")
        fig.savefig(output, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        return {
            "output_path": str(output),
            "chart": "forecast_validation_genre_comparison",
            "validation_year": validation_year,
            "num_genres": int(len(plot_data)),
        }

    def draw_forecast_backtest_yearly_trend(
        self,
        backtest_result: Dict[str, object],
        output_path: str | Path = "outputs/forecast_backtest_yearly_trend.png",
        figsize: Tuple[float, float] = (14.0, 8.2),
        dpi: int = 240,
    ) -> Dict[str, Any]:
        """Draw a yearly backtest trend chart for forecast accuracy."""
        if "error" in backtest_result:
            raise ValueError(str(backtest_result["error"]))

        overview = backtest_result["yearly_overview"].copy()
        required = {"validation_year", "overall_wape_percent", "overall_r_squared", "validation_rows"}
        if overview.empty or not required.issubset(overview.columns):
            raise ValueError("Backtest overview is empty or incomplete.")

        plot_data = overview.copy()
        plot_data["validation_year"] = pd.to_numeric(plot_data["validation_year"], errors="coerce")
        plot_data["overall_wape_percent"] = pd.to_numeric(plot_data["overall_wape_percent"], errors="coerce")
        plot_data["overall_r_squared"] = pd.to_numeric(plot_data["overall_r_squared"], errors="coerce")
        plot_data["validation_rows"] = pd.to_numeric(plot_data["validation_rows"], errors="coerce")
        plot_data = plot_data.dropna(subset=["validation_year", "overall_wape_percent", "overall_r_squared"]).sort_values("validation_year")
        if plot_data.empty:
            raise ValueError("Backtest overview does not contain valid yearly metrics.")

        years = plot_data["validation_year"].astype(int).tolist()
        output = self._prepare_output(output_path)

        fig, ax1 = plt.subplots(figsize=figsize, dpi=dpi)
        ax1.bar(
            years,
            plot_data["overall_wape_percent"],
            color="#F6AE2D",
            alpha=0.82,
            width=0.72,
            label="WAPE (%)",
        )
        self._apply_axes_style(ax1, "Forecast Backtest Accuracy by Year", "Validation Year", "WAPE (%)")
        ax1.set_xticks(years)
        ax1.set_xticklabels([self._format_validation_year_label(year) for year in years], rotation=0)

        for year, validation_rows in zip(years, plot_data["validation_rows"].astype(int)):
            ax1.text(
                year,
                1.2,
                f"n={validation_rows}",
                ha="center",
                va="bottom",
                fontsize=10,
                color="#6A5A4A",
            )

        ax2 = ax1.twinx()
        ax2.plot(
            years,
            plot_data["overall_r_squared"],
            color="#33658A",
            linewidth=3.0,
            marker="o",
            markersize=6,
            label="R²",
        )
        ax2.set_ylabel("R²", fontweight="bold")
        lower_bound = min(-0.1, float(plot_data["overall_r_squared"].min()) - 0.08)
        ax2.set_ylim(lower_bound, 1.05)
        ax2.spines["top"].set_visible(False)

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="upper right")

        fig.savefig(output, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        return {
            "output_path": str(output),
            "chart": "forecast_backtest_yearly_trend",
            "start_year": int(min(years)),
            "end_year": int(max(years)),
        }

    def draw_forecast_backtest_yearly_comparison(
        self,
        backtest_result: Dict[str, object],
        output_path: str | Path = "outputs/forecast_backtest_yearly_comparison.png",
        figsize: Tuple[float, float] = (14.0, 8.4),
        dpi: int = 240,
    ) -> Dict[str, Any]:
        """Draw yearly actual vs predicted revenue totals across the backtest window."""
        if "error" in backtest_result:
            raise ValueError(str(backtest_result["error"]))

        overview = backtest_result["yearly_overview"].copy()
        required = {
            "validation_year",
            "actual_total_revenue",
            "predicted_total_revenue",
            "overall_wape_percent",
        }
        if overview.empty or not required.issubset(overview.columns):
            raise ValueError("Backtest overview is empty or incomplete.")

        plot_data = overview.copy()
        plot_data["validation_year"] = pd.to_numeric(plot_data["validation_year"], errors="coerce")
        plot_data["actual_total_revenue"] = pd.to_numeric(plot_data["actual_total_revenue"], errors="coerce")
        plot_data["predicted_total_revenue"] = pd.to_numeric(plot_data["predicted_total_revenue"], errors="coerce")
        plot_data["overall_wape_percent"] = pd.to_numeric(plot_data["overall_wape_percent"], errors="coerce")
        plot_data = plot_data.dropna(
            subset=["validation_year", "actual_total_revenue", "predicted_total_revenue"]
        ).sort_values("validation_year")
        if plot_data.empty:
            raise ValueError("Backtest overview does not contain valid yearly revenue totals.")

        selected_model_label = str(backtest_result.get("selected_model_label") or "").strip()
        selection_metric = str(backtest_result.get("selection_metric") or "").strip()
        selection_metric_value = pd.to_numeric(pd.Series([backtest_result.get("selection_metric_value")]), errors="coerce").iloc[0]
        years = plot_data["validation_year"].astype(int).tolist()
        x_positions = np.arange(len(plot_data))
        width = 0.38
        output = self._prepare_output(output_path)

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.bar(
            x_positions - width / 2,
            plot_data["actual_total_revenue"],
            width=width,
            color="#33658A",
            label="Actual revenue",
        )
        ax.bar(
            x_positions + width / 2,
            plot_data["predicted_total_revenue"],
            width=width,
            color="#F26419",
            label="Predicted revenue",
        )

        self._apply_axes_style(
            ax,
            "Yearly Forecast vs Actual Revenue",
            xlabel="Validation Year",
            ylabel="Total Revenue (USD)",
        )
        ax.set_xticks(x_positions)
        ax.set_xticklabels([self._format_validation_year_label(year) for year in years])
        ax.yaxis.set_major_formatter(FuncFormatter(self._currency_formatter))
        ax.legend(frameon=False, ncol=2, loc="upper left")

        maxima = np.maximum(
            plot_data["actual_total_revenue"].to_numpy(dtype=float),
            plot_data["predicted_total_revenue"].to_numpy(dtype=float),
        )
        for x_pos, value, wape in zip(
            x_positions,
            maxima,
            plot_data["overall_wape_percent"].fillna(0.0).to_numpy(dtype=float),
        ):
            ax.text(
                x_pos,
                value * 1.015 if value > 0 else 0.0,
                f"WAPE {wape:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                color="#6A5A4A",
            )

        footer_parts: list[str] = []
        if selected_model_label:
            footer_parts.append(f"Selected model: {selected_model_label}")
        if selection_metric == "yearly_revenue_correlation" and pd.notna(selection_metric_value):
            footer_parts.append(f"Correlation: {float(selection_metric_value):.3f}")
        if footer_parts:
            fig.subplots_adjust(bottom=0.18)
            fig.text(0.5, 0.04, "  |  ".join(footer_parts), ha="center", fontsize=12, color="#4F5D75")

        fig.savefig(output, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        return {
            "output_path": str(output),
            "chart": "forecast_backtest_yearly_comparison",
            "start_year": int(min(years)),
            "end_year": int(max(years)),
            "selected_model_name": str(backtest_result.get("selected_model_name") or ""),
            "selected_model_label": selected_model_label,
        }

    def draw_multi_model_forecast_backtest_lines(
        self,
        multi_model_backtest_result: Dict[str, object],
        output_path: str | Path = "outputs/forecast_model_backtest_comparison.png",
        figsize: Tuple[float, float] = (14.4, 8.6),
        dpi: int = 240,
    ) -> Dict[str, Any]:
        """Draw one line chart comparing actual revenue against multiple forecast models."""
        if "error" in multi_model_backtest_result:
            raise ValueError(str(multi_model_backtest_result["error"]))

        overview = multi_model_backtest_result.get("yearly_model_overview")
        if not isinstance(overview, pd.DataFrame) or overview.empty:
            raise ValueError("Multi-model forecast backtest overview is empty or unavailable.")

        required = {"validation_year", "model_name", "model_label", "actual_total_revenue", "predicted_total_revenue"}
        if not required.issubset(overview.columns):
            raise ValueError("Multi-model forecast backtest overview is incomplete.")

        plot_data = overview.copy()
        plot_data["validation_year"] = pd.to_numeric(plot_data["validation_year"], errors="coerce")
        plot_data["actual_total_revenue"] = pd.to_numeric(plot_data["actual_total_revenue"], errors="coerce")
        plot_data["predicted_total_revenue"] = pd.to_numeric(plot_data["predicted_total_revenue"], errors="coerce")
        plot_data = plot_data.dropna(subset=["validation_year", "actual_total_revenue", "predicted_total_revenue"]).copy()
        if plot_data.empty:
            raise ValueError("Multi-model forecast backtest overview does not contain valid yearly totals.")

        plot_data["validation_year"] = plot_data["validation_year"].astype(int)
        plot_data = plot_data.sort_values(["validation_year", "model_name"]).reset_index(drop=True)

        actual_series = (
            plot_data[["validation_year", "actual_total_revenue"]]
            .drop_duplicates(subset=["validation_year"])
            .sort_values("validation_year")
        )
        years = actual_series["validation_year"].tolist()
        output = self._prepare_output(output_path)

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.plot(
            years,
            actual_series["actual_total_revenue"],
            color="#1F2A44",
            linewidth=3.4,
            marker="o",
            markersize=6.5,
            label="Actual Revenue",
        )

        model_summary = multi_model_backtest_result.get("model_summary")
        if isinstance(model_summary, pd.DataFrame) and not model_summary.empty:
            summary_lookup = {
                str(row["model_name"]): row
                for _, row in model_summary.iterrows()
            }
        else:
            summary_lookup = {}

        colors = [
            "#33658A",
            "#F26419",
            "#758E4F",
            "#7D82B8",
            "#C8553D",
            "#86BBD8",
        ]
        for index, model_name in enumerate(sorted(plot_data["model_name"].astype(str).unique().tolist())):
            model_rows = plot_data.loc[plot_data["model_name"].astype(str) == model_name].sort_values("validation_year")
            summary_row = summary_lookup.get(model_name)
            label = FORECAST_MODEL_LABELS.get(model_name, model_name)
            if summary_row is not None:
                label = f"{label} (avg WAPE {float(summary_row['average_wape_percent']):.1f}%)"

            ax.plot(
                model_rows["validation_year"],
                model_rows["predicted_total_revenue"],
                linewidth=2.6,
                marker="o",
                markersize=5.5,
                color=colors[index % len(colors)],
                label=label,
            )

        self._apply_axes_style(
            ax,
            "Multi-Model Box Office Forecast Backtest",
            xlabel="Validation Year",
            ylabel="Total Revenue (USD)",
        )
        ax.set_xticks(years)
        ax.set_xticklabels([self._format_validation_year_label(year) for year in years])
        ax.yaxis.set_major_formatter(FuncFormatter(self._currency_formatter))
        ax.legend(frameon=False, ncol=2, loc="upper left")

        fig.savefig(output, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        return {
            "output_path": str(output),
            "chart": "forecast_model_backtest_comparison",
            "start_year": int(min(years)),
            "end_year": int(max(years)),
            "models_evaluated": int(plot_data["model_name"].nunique()),
        }


def create_genre_bar_chart(
    genre_summary: pd.DataFrame,
    output_path: str | Path = "outputs/genre_distribution.png",
    dpi: int = 240,
    title: Optional[str] = None,
    top_n: Optional[int] = 12,
) -> Dict[str, Any]:
    visualizer = DataVisualizer()
    return visualizer.draw_genre_bar_chart(genre_summary, output_path=output_path, dpi=dpi, title=title, top_n=top_n)


def create_genre_comparison_chart(
    genre_summary: pd.DataFrame,
    output_path: str | Path = "outputs/genre_comparison.png",
    dpi: int = 240,
) -> Dict[str, Any]:
    visualizer = DataVisualizer()
    return visualizer.draw_genre_comparison_chart(genre_summary, output_path=output_path, dpi=dpi)


def create_yearly_rating_chart(
    yearly_summary: pd.DataFrame,
    output_path: str | Path = "outputs/yearly_rating_trend.png",
    dpi: int = 240,
) -> Dict[str, Any]:
    visualizer = DataVisualizer()
    return visualizer.draw_yearly_rating_trend(yearly_summary, output_path=output_path, dpi=dpi)


def create_budget_revenue_chart(
    df: pd.DataFrame,
    output_path: str | Path = "outputs/budget_revenue_scatter.png",
    dpi: int = 240,
) -> Dict[str, Any]:
    visualizer = DataVisualizer()
    return visualizer.draw_budget_revenue_scatter(df, output_path=output_path, dpi=dpi)


def create_forecast_backtest_chart(
    backtest_result: Dict[str, object],
    output_path: str | Path = "outputs/forecast_backtest_yearly_trend.png",
    dpi: int = 240,
) -> Dict[str, Any]:
    visualizer = DataVisualizer()
    return visualizer.draw_forecast_backtest_yearly_trend(backtest_result, output_path=output_path, dpi=dpi)


def create_forecast_backtest_yearly_comparison_chart(
    backtest_result: Dict[str, object],
    output_path: str | Path = "outputs/forecast_backtest_yearly_comparison.png",
    dpi: int = 240,
) -> Dict[str, Any]:
    visualizer = DataVisualizer()
    return visualizer.draw_forecast_backtest_yearly_comparison(backtest_result, output_path=output_path, dpi=dpi)


def create_multi_model_forecast_backtest_chart(
    multi_model_backtest_result: Dict[str, object],
    output_path: str | Path = "outputs/forecast_model_backtest_comparison.png",
    dpi: int = 240,
) -> Dict[str, Any]:
    visualizer = DataVisualizer()
    return visualizer.draw_multi_model_forecast_backtest_lines(
        multi_model_backtest_result,
        output_path=output_path,
        dpi=dpi,
    )


def create_project_visuals(
    df: pd.DataFrame,
    output_dir: str | Path = "outputs",
    include_global_forecast: bool = True,
) -> Dict[str, str]:
    """Generate the full chart set used by the project report and UI."""
    output_root = Path(output_dir)
    if not output_root.is_absolute():
        output_root = Path(__file__).resolve().parent / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    for legacy_name in (
        "forecast_backtest_genre_animation.gif",
        "forecast_validation_genre_comparison.png",
        "forecast_backtest_yearly_trend.png",
    ):
        legacy_path = output_root / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()

    created: Dict[str, str] = {}
    genre_summary = aggregate_by_genre(df)
    yearly_summary = yearly_rating_trend(df)

    if not genre_summary.empty:
        created["genre_distribution"] = create_genre_bar_chart(
            genre_summary,
            output_path=output_root / "genre_distribution.png",
        )["output_path"]
        created["genre_comparison"] = create_genre_comparison_chart(
            genre_summary,
            output_path=output_root / "genre_comparison.png",
        )["output_path"]

    if not yearly_summary.empty:
        created["yearly_rating_trend"] = create_yearly_rating_chart(
            yearly_summary,
            output_path=output_root / "yearly_rating_trend.png",
        )["output_path"]

    if {"budget", "revenue"}.issubset(df.columns):
        valid_rows = df.loc[
            pd.to_numeric(df["budget"], errors="coerce").gt(0)
            & pd.to_numeric(df["revenue"], errors="coerce").gt(0)
        ]
        if len(valid_rows) >= 2:
            created["budget_revenue_scatter"] = create_budget_revenue_chart(
                df,
                output_path=output_root / "budget_revenue_scatter.png",
            )["output_path"]

    if include_global_forecast:
        backtest_result = _get_or_create_forecast_backtest_result(df, output_root)
        if "error" not in backtest_result:
            created["forecast_backtest_yearly_comparison"] = create_forecast_backtest_yearly_comparison_chart(
                backtest_result,
                output_path=output_root / "forecast_backtest_yearly_comparison.png",
            )["output_path"]

        multi_model_result = _get_or_create_multi_model_backtest_result(df, output_root)
        if "error" not in multi_model_result:
            created["forecast_model_backtest_comparison"] = create_multi_model_forecast_backtest_chart(
                multi_model_result,
                output_path=output_root / "forecast_model_backtest_comparison.png",
            )["output_path"]

    return created


if __name__ == "__main__":
    demo = pd.DataFrame(
        {
            "title": ["A", "B", "C", "D"],
            "primary_genre": ["Action", "Drama", "Action", "Comedy"],
            "vote_average": [7.2, 8.1, 6.8, 7.0],
            "release_date": ["2000-01-01", "2001-01-01", "2000-05-01", "2002-02-02"],
            "budget": [100, 120, 180, 90],
            "revenue": [240, 310, 290, 110],
        }
    )
    paths = create_project_visuals(demo)
    print(paths)
