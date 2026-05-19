import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from box_office_forecasting import (
    DATASET_CUTOFF_DATE,
    backtest_multiple_forecast_models,
    backtest_genre_revenue_models,
    evaluate_genre_revenue_models,
    load_saved_forecast_results,
    profile_high_box_office_characteristics,
    prepare_forecasting_dataframe,
    run_genre_box_office_forecast,
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
                    "vote_count": 100 + (year - 2012) * 5,
                    "popularity": 10.0 + (year - 2012) * 0.5,
                    "keyword": f"{genre.lower()}|sample",
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

    def test_run_forecast_reuses_saved_outputs(self) -> None:
        sample = _build_forecast_sample()
        with TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "cleaned_sample.csv"
            sample.to_csv(dataset_path, index=False)

            first = run_genre_box_office_forecast(
                dataset_path=dataset_path,
                output_dir=temp_dir,
                validation_year=2017,
                min_train_rows=3,
                alpha=1.0,
            )
            self.assertNotIn("error", first)

            loaded = load_saved_forecast_results(output_dir=temp_dir, validation_year=2017)
            self.assertNotIn("error", loaded)
            self.assertIn("cache_metadata", loaded["overview"])

            with patch(
                "box_office_forecasting.evaluate_genre_revenue_models",
                side_effect=AssertionError("forecast evaluation should use cached outputs"),
            ):
                second = run_genre_box_office_forecast(
                    dataset_path=dataset_path,
                    output_dir=temp_dir,
                    validation_year=2017,
                    min_train_rows=3,
                    alpha=1.0,
                )

            self.assertNotIn("error", second)
            self.assertEqual(second["overview"]["validation_year"], 2017)
            self.assertEqual(len(second["summary_by_genre"]), len(first["summary_by_genre"]))


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

    def test_builds_multi_model_backtest_summary(self) -> None:
        sample = _build_forecast_sample()
        result = backtest_multiple_forecast_models(
            sample,
            model_names=("ridge", "ols", "genre_budget_ratio"),
            start_year=2015,
            end_year=2017,
            min_validation_rows=3,
            max_years=10,
            min_train_rows=3,
            alpha=1.0,
        )

        self.assertNotIn("error", result)
        overview = result["yearly_model_overview"]
        summary = result["model_summary"]

        self.assertSetEqual(set(overview["validation_year"]), {2015, 2016, 2017})
        self.assertSetEqual(set(overview["model_name"]), {"ridge", "ols", "genre_budget_ratio"})
        self.assertEqual(summary["model_name"].nunique(), 3)
        self.assertTrue({"predicted_total_revenue", "actual_total_revenue", "overall_wape_percent"}.issubset(overview.columns))
        self.assertIn("yearly_revenue_correlation", summary.columns)
        self.assertEqual(result["selection_metric"], "yearly_revenue_correlation")
        self.assertEqual(result["selected_model_name"], str(summary.iloc[0]["model_name"]))
        self.assertGreaterEqual(float(summary.iloc[0]["yearly_revenue_correlation"]), float(summary.iloc[-1]["yearly_revenue_correlation"]))


class ForecastVisualizationTests(unittest.TestCase):
    def test_create_project_visuals_generates_forecast_assets(self) -> None:
        sample = _build_forecast_sample()
        with TemporaryDirectory() as temp_dir:
            created = create_project_visuals(sample, output_dir=temp_dir)

            self.assertIn("forecast_backtest_yearly_comparison", created)
            self.assertIn("forecast_model_backtest_comparison", created)

            for key in (
                "forecast_backtest_yearly_comparison",
                "forecast_model_backtest_comparison",
            ):
                path = Path(created[key])
                self.assertTrue(path.exists(), msg=f"{key} output was not created")
                self.assertGreater(path.stat().st_size, 0, msg=f"{key} output is empty")

            self.assertFalse((Path(temp_dir) / "forecast_backtest_genre_animation.gif").exists())

    def test_create_project_visuals_reuses_saved_forecast_data(self) -> None:
        sample = _build_forecast_sample()
        with TemporaryDirectory() as temp_dir:
            created = create_project_visuals(sample, output_dir=temp_dir)
            Path(created["forecast_backtest_yearly_comparison"]).unlink()
            Path(created["forecast_model_backtest_comparison"]).unlink()

            self.assertTrue((Path(temp_dir) / "forecast_backtest_yearly_overview.csv").exists())
            self.assertTrue((Path(temp_dir) / "forecast_model_backtest_yearly_overview.csv").exists())

            with patch(
                "data_visualization.backtest_multiple_forecast_models",
                side_effect=AssertionError("multi-model backtest should use cached data"),
            ):
                reused = create_project_visuals(sample, output_dir=temp_dir)

            self.assertTrue(Path(reused["forecast_backtest_yearly_comparison"]).exists())
            self.assertTrue(Path(reused["forecast_model_backtest_comparison"]).exists())

    def test_yearly_comparison_uses_best_correlation_model(self) -> None:
        sample = _build_forecast_sample()
        with TemporaryDirectory() as temp_dir:
            created = create_project_visuals(sample, output_dir=temp_dir)

            summary = pd.read_csv(Path(temp_dir) / "forecast_model_backtest_summary.csv")
            overview = pd.read_csv(Path(temp_dir) / "forecast_backtest_yearly_overview.csv")

            self.assertFalse(summary.empty)
            self.assertFalse(overview.empty)
            self.assertEqual(str(overview.iloc[0]["model_name"]), str(summary.iloc[0]["model_name"]))
            self.assertIn("model_label", overview.columns)
            self.assertTrue(Path(created["forecast_backtest_yearly_comparison"]).exists())


class HighBoxOfficeProfileTests(unittest.TestCase):
    def test_profiles_characteristics_for_high_box_office_movies(self) -> None:
        sample = _build_forecast_sample()
        result = profile_high_box_office_characteristics(sample, min_rows=12, min_train_rows=6, min_holdout_rows=3)

        self.assertNotIn("error", result)
        overview = result["overview"]
        profile = result["high_revenue_profile"]

        self.assertEqual(overview["model_label"], "Ridge Revenue Profile")
        self.assertGreaterEqual(overview["sample_rows"], 12)
        self.assertGreater(overview["high_revenue_threshold"], 0)
        self.assertIn(overview["evaluation_scope"], {"latest holdout movies", "full current selection"})
        self.assertTrue(result["top_positive_features"])
        self.assertTrue(result["top_negative_features"])
        self.assertTrue(isinstance(profile["top_genres"], list))
        self.assertGreaterEqual(profile["english_share_percent"], 0)
        self.assertGreaterEqual(profile["us_production_share_percent"], 0)


if __name__ == "__main__":
    unittest.main()
