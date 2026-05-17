"""Advanced and optional analytics for the IMDB movies project."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


class AdvancedAnalytics:
    """Higher-level analyses used for the optional project section."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._prepare_dataframe()

    def _prepare_dataframe(self) -> None:
        if "release_date" in self.df.columns and "year" not in self.df.columns:
            self.df["year"] = pd.to_datetime(self.df["release_date"], errors="coerce").dt.year

        for column in ("year", "budget", "revenue", "runtime", "vote_average"):
            if column in self.df.columns:
                self.df[column] = pd.to_numeric(self.df[column], errors="coerce")

    def analyze_yearly_production(
        self,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
    ) -> Dict[str, object]:
        """Time-series view of movie volume and rating trend."""
        if "year" not in self.df.columns:
            return {"error": "Year data is not available."}

        working = self.df.dropna(subset=["year"]).copy()
        if start_year is not None:
            working = working.loc[working["year"] >= start_year]
        if end_year is not None:
            working = working.loc[working["year"] <= end_year]
        if working.empty:
            return {"error": "No records available in the requested year range."}

        yearly = working.groupby("year").agg(
            movie_count=("title", "count"),
            avg_rating=("vote_average", "mean"),
        )
        years = yearly.index.to_numpy(dtype=float)
        counts = yearly["movie_count"].to_numpy(dtype=float)

        if len(years) > 1:
            slope, intercept = np.polyfit(years - years.min(), counts, 1)
            growth_rates = np.diff(counts) / counts[:-1]
            average_growth = float(np.nanmean(growth_rates) * 100)
        else:
            slope = 0.0
            intercept = float(counts[0])
            average_growth = 0.0

        return {
            "period": f"{int(yearly.index.min())}-{int(yearly.index.max())}",
            "total_movies": int(yearly["movie_count"].sum()),
            "average_movies_per_year": round(float(yearly["movie_count"].mean()), 2),
            "peak_year": int(yearly["movie_count"].idxmax()),
            "peak_count": int(yearly["movie_count"].max()),
            "trend_direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable",
            "trend_slope": round(float(slope), 4),
            "average_growth_rate": round(average_growth, 2),
            "yearly_data": yearly.reset_index().to_dict("records"),
            "regression_intercept": round(intercept, 4),
        }

    def analyze_budget_revenue_correlation(self) -> Dict[str, object]:
        """Regression-style analysis for budget and revenue."""
        required_columns = {"budget", "revenue"}
        if not required_columns.issubset(self.df.columns):
            return {"error": "Budget or revenue columns are missing."}

        working = self.df.loc[
            self.df["budget"].gt(0)
            & self.df["revenue"].gt(0)
            & self.df["budget"].notna()
            & self.df["revenue"].notna()
        ].copy()
        if len(working) < 2:
            return {"error": "Not enough valid rows for budget/revenue analysis."}

        budget = working["budget"].to_numpy(dtype=float)
        revenue = working["revenue"].to_numpy(dtype=float)
        finite_mask = np.isfinite(budget) & np.isfinite(revenue)
        budget = budget[finite_mask]
        revenue = revenue[finite_mask]
        working = working.loc[working.index[finite_mask]].copy()
        if len(working) < 2:
            return {"error": "Not enough finite rows for budget/revenue analysis."}

        budget_std = float(np.std(budget))
        revenue_std = float(np.std(revenue))

        if budget_std == 0.0:
            slope = 0.0
            intercept = float(np.mean(revenue))
        else:
            slope, intercept = np.polyfit(budget, revenue, 1)

        if budget_std == 0.0 or revenue_std == 0.0:
            correlation = 0.0
        else:
            correlation = float(np.corrcoef(budget, revenue)[0, 1])
            if not np.isfinite(correlation):
                correlation = 0.0

        predicted = slope * budget + intercept
        residual = revenue - predicted
        ss_res = float(np.sum(np.square(residual)))
        ss_tot = float(np.sum(np.square(revenue - np.mean(revenue))))
        r_squared = 1 - ss_res / ss_tot if ss_tot else 0.0

        working["roi"] = (working["revenue"] - working["budget"]) / working["budget"] * 100
        best_roi_row = working.loc[working["roi"].idxmax()]

        return {
            "sample_size": int(len(working)),
            "correlation": round(correlation, 4),
            "correlation_strength": self._interpret_correlation(correlation),
            "r_squared": round(float(r_squared), 4),
            "regression_slope": round(float(slope), 4),
            "regression_intercept": round(float(intercept), 2),
            "average_roi": round(float(working["roi"].mean()), 2),
            "median_roi": round(float(working["roi"].median()), 2),
            "best_roi_movie": str(best_roi_row["title"]),
            "best_roi_value": round(float(best_roi_row["roi"]), 2),
        }

    @staticmethod
    def _interpret_correlation(value: float) -> str:
        absolute = abs(value)
        if absolute >= 0.7:
            return "Strong correlation"
        if absolute >= 0.5:
            return "Moderate correlation"
        if absolute >= 0.3:
            return "Weak correlation"
        return "Very weak or no correlation"

    def genre_performance_analysis(self, min_movies: int = 5) -> pd.DataFrame:
        """Optional genre comparison with revenue and rating metrics."""
        if "primary_genre" not in self.df.columns:
            return pd.DataFrame()

        grouped = self.df.groupby("primary_genre").agg(
            movie_count=("title", "count"),
            avg_rating=("vote_average", "mean"),
            avg_runtime=("runtime", "mean"),
            avg_budget=("budget", "mean"),
            avg_revenue=("revenue", "mean"),
        )
        grouped = grouped.loc[grouped["movie_count"] >= min_movies]
        return grouped.round(2).sort_values(["movie_count", "avg_rating"], ascending=[False, False]).reset_index()

    def decade_comparison(self) -> Dict[str, object]:
        """Compare movie patterns across decades."""
        if "year" not in self.df.columns:
            return {"error": "Year data is not available."}

        working = self.df.dropna(subset=["year"]).copy()
        if working.empty:
            return {"error": "No year values available."}

        working["decade"] = (working["year"] // 10) * 10
        grouped = working.groupby("decade").agg(
            movie_count=("title", "count"),
            avg_rating=("vote_average", "mean"),
            avg_runtime=("runtime", "mean"),
            total_revenue=("revenue", "sum"),
        )

        return {
            "decade_statistics": grouped.round(2).reset_index().to_dict("records"),
            "most_productive_decade": int(grouped["movie_count"].idxmax()),
            "highest_rated_decade": int(grouped["avg_rating"].idxmax()),
        }

    def outlier_detection(self, column: str = "revenue") -> Dict[str, object]:
        """Detect outliers in a numeric column with z-score and IQR."""
        if column not in self.df.columns:
            return {"error": f"Column '{column}' not found."}

        series = pd.to_numeric(self.df[column], errors="coerce").dropna()
        if series.empty:
            return {"error": f"Column '{column}' does not contain numeric values."}

        mean = float(series.mean())
        std = float(series.std(ddof=0))
        if std == 0:
            z_mask = pd.Series(False, index=series.index)
        else:
            z_mask = ((series - mean).abs() / std) > 3

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        iqr_mask = (series < lower) | (series > upper)

        outlier_rows = self.df.loc[series.index[iqr_mask]]
        return {
            "column": column,
            "mean": round(mean, 2),
            "std": round(std, 2),
            "q1": round(q1, 2),
            "q3": round(q3, 2),
            "iqr": round(iqr, 2),
            "z_score_outliers": int(z_mask.sum()),
            "iqr_outliers": int(iqr_mask.sum()),
            "example_titles": outlier_rows["title"].head(10).astype(str).tolist() if "title" in outlier_rows.columns else [],
        }


def get_production_trend(
    df: pd.DataFrame,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
) -> Dict[str, object]:
    analyzer = AdvancedAnalytics(df)
    return analyzer.analyze_yearly_production(start_year=start_year, end_year=end_year)


def get_budget_revenue_analysis(df: pd.DataFrame) -> Dict[str, object]:
    analyzer = AdvancedAnalytics(df)
    return analyzer.analyze_budget_revenue_correlation()


def get_comprehensive_analysis(df: pd.DataFrame) -> Dict[str, object]:
    analyzer = AdvancedAnalytics(df)
    return {
        "production_trend": analyzer.analyze_yearly_production(),
        "budget_revenue_correlation": analyzer.analyze_budget_revenue_correlation(),
        "decade_comparison": analyzer.decade_comparison(),
        "genre_performance": analyzer.genre_performance_analysis().to_dict("records"),
    }


if __name__ == "__main__":
    sample = pd.DataFrame(
        {
            "title": ["Movie A", "Movie B", "Movie C", "Movie D"],
            "release_date": ["1995-01-01", "2000-01-01", "2005-01-01", "2010-01-01"],
            "budget": [10_000_000, 20_000_000, 30_000_000, 40_000_000],
            "revenue": [50_000_000, 60_000_000, 55_000_000, 100_000_000],
            "runtime": [100, 120, 95, 110],
            "vote_average": [7.0, 7.5, 6.8, 8.2],
            "primary_genre": ["Drama", "Action", "Drama", "Action"],
        }
    )
    print(get_comprehensive_analysis(sample))
