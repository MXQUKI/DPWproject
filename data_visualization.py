"""Visualization utilities for the IMDB movies project."""

from __future__ import annotations

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

from box_office_forecasting import backtest_genre_revenue_models
from data_analysis import aggregate_by_genre, yearly_rating_trend

FORECAST_YEAR_SELECTION_START = 2003
FORECAST_YEAR_SELECTION_END = 2017


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
        ax.set_xticklabels(years)
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

        fig.savefig(output, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        return {
            "output_path": str(output),
            "chart": "forecast_backtest_yearly_comparison",
            "start_year": int(min(years)),
            "end_year": int(max(years)),
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
        backtest_result = backtest_genre_revenue_models(
            df,
            start_year=FORECAST_YEAR_SELECTION_START,
            end_year=FORECAST_YEAR_SELECTION_END,
            max_years=0,
        )
        if "error" not in backtest_result:
            created["forecast_backtest_yearly_comparison"] = create_forecast_backtest_yearly_comparison_chart(
                backtest_result,
                output_path=output_root / "forecast_backtest_yearly_comparison.png",
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
