import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from box_office_forecasting import (
    DATASET_CUTOFF_DATE,
    backtest_genre_revenue_models,
    evaluate_genre_revenue_models,
    prepare_forecasting_dataframe,
)
from data_visualization import create_project_visuals


def _build_forecast_sample() -> pd.DataFrame:
    rows = []
    identifier = 1
    release_months = (1, 2, 3, 4, 5, 6)
    for genre, budget_base, revenue_base, runtime_base in (
        ("Action", 10_000_000, 35_000_000, 100),
        ("Drama", 5_000_000, 12_000_000, 94),
        ("Comedy", 4_000_000, 10_000_000, 92),
    ):
        for year in range(2012, 2018):
            month = release_months[(year - 2012) % len(release_months)]
            rows.append(
                {
                    "id": identifier,
                    "title": f"{genre} {year}",
                    "primary_genre": genre,
                    "budget": budget_base + (year - 2012) * 800_000,
                    "revenue": revenue_base + (year - 2012) * 3_200_000,
                    "runtime": runtime_base + ((year - 2012) % 3),
                    "release_date": f"{year}-{month:02d}-01",
                    "year": year,
                    "language": "en",
                    "country": "United States of America",
                    "vote_average": 6.8 + ((year - 2012) * 0.1),
                }
            )
            identifier += 1
    return pd.DataFrame(rows)


class PrepareForecastingDataframeTests(unittest.TestCase):
    def test_preserves_target_missingness_and_applies_cutoff(self) -> None:
        sample = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "title": ["Before", "Missing revenue", "After cutoff"],
                "release_date": ["2016-06-01", "2017-02-01", "2017-08-01"],
                "genres": ['[{"id": 18, "name": "Drama"}]'] * 3,
                "original_language": ["en", "en", "en"],
                "production_countries": [
                    '[{"iso_3166_1": "US", "name": "United States of America"}]',
                    '[{"iso_3166_1": "US", "name": "United States of America"}]',
                    '[{"iso_3166_1": "US", "name": "United States of America"}]',
                ],
                "budget": [100.0, 120.0, 140.0],
                "revenue": [200.0, None, 300.0],
                "runtime": [90.0, 95.0, 100.0],
            }
        )

        prepared = prepare_forecasting_dataframe(sample)

        self.assertEqual(len(prepared), 2)
        self.assertTrue((prepared["release_date"] <= DATASET_CUTOFF_DATE).all())
        missing_revenue = prepared.loc[prepared["title"] == "Missing revenue", "revenue"].iloc[0]
        self.assertTrue(pd.isna(missing_revenue))
        self.assertEqual(prepared.loc[prepared["title"] == "Before", "primary_genre"].iloc[0], "Drama")


class EvaluateGenreRevenueModelsTests(unittest.TestCase):
    def test_builds_per_genre_validation_summary(self) -> None:
        sample = _build_forecast_sample().loc[lambda frame: frame["primary_genre"].isin(["Action", "Drama"])].reset_index(drop=True)
        result = evaluate_genre_revenue_models(sample, validation_year=2017, min_train_rows=3, alpha=1.0)

        self.assertNotIn("error", result)
        self.assertEqual(result["overview"]["validation_year"], 2017)
        self.assertEqual(result["overview"]["validation_rows"], 2)
        self.assertEqual(result["overview"]["genres_evaluated"], 2)

        summary = result["summary_by_genre"]
        self.assertSetEqual(set(summary["genre"]), {"Action", "Drama"})
        self.assertTrue(summary["model_type"].isin({"global_ridge_calibrated", "global_ridge_shared"}).all())
        self.assertTrue((summary["validation_samples"] == 1).all())

        predictions = result["predictions"]
        self.assertEqual(len(predictions), 2)
        self.assertTrue((predictions["predicted_revenue"] >= 0).all())


class BacktestGenreRevenueModelsTests(unittest.TestCase):
    def test_builds_multi_year_backtest_summary(self) -> None:
        sample = _build_forecast_sample()
        result = backtest_genre_revenue_models(
            sample,
            start_year=2015,
            end_year=2017,
            min_validation_rows=3,
            max_years=10,
            min_train_rows=3,
            alpha=1.0,
        )

        self.assertNotIn("error", result)
        overview = result["yearly_overview"]
        genre_summary = result["yearly_genre_summary"]

        self.assertListEqual(overview["validation_year"].tolist(), [2015, 2016, 2017])
        self.assertSetEqual(set(genre_summary["validation_year"]), {2015, 2016, 2017})
        self.assertTrue({"overall_wape_percent", "overall_r_squared", "validation_rows"}.issubset(overview.columns))
        self.assertTrue({"genre", "actual_total_revenue", "predicted_total_revenue"}.issubset(genre_summary.columns))


class ForecastVisualizationTests(unittest.TestCase):
    def test_create_project_visuals_generates_forecast_assets(self) -> None:
        sample = _build_forecast_sample()
        with TemporaryDirectory() as temp_dir:
            created = create_project_visuals(sample, output_dir=temp_dir)

            self.assertIn("forecast_backtest_yearly_comparison", created)

            for key in (
                "forecast_backtest_yearly_comparison",
            ):
                path = Path(created[key])
                self.assertTrue(path.exists(), msg=f"{key} output was not created")
                self.assertGreater(path.stat().st_size, 0, msg=f"{key} output is empty")

            self.assertFalse((Path(temp_dir) / "forecast_backtest_genre_animation.gif").exists())


if __name__ == "__main__":
    unittest.main()
