"""Entry point for the integrated IMDB movies project."""

from __future__ import annotations

import argparse
from pathlib import Path

from advanced_analytics import get_comprehensive_analysis
from box_office_forecasting import run_genre_box_office_forecast
from data_analysis import analyze
from data_preprocessing import prepare_dataset
from data_visualization import create_project_visuals


def run_report(dataset_path: str | None = None, output_dir: str = "outputs") -> None:
    dataset, report = prepare_dataset(
        dataset_path=dataset_path,
        export_path=Path(output_dir) / "cleaned_imdb_movies_project.csv",
        prefer_cleaned=True,
    )
    analysis_result = analyze(dataset)
    advanced_result = get_comprehensive_analysis(dataset)
    chart_paths = create_project_visuals(dataset, output_dir=output_dir)
    forecast_result = run_genre_box_office_forecast(dataset_path=dataset_path, output_dir=output_dir)

    overview = analysis_result["overview"]
    print("IMDB Movies Project Report")
    print("=" * 40)
    print(f"Source file: {report['source_file']}")
    print(f"Rows after cleaning: {report['clean_rows']:,}")
    print(
        f"Cleaned date range: {report['cleaned_date_range']['start']} -> {report['cleaned_date_range']['end']} "
        f"(cutoff {report['dataset_cutoff_date']})"
    )
    print(f"Rows removed after cutoff: {report['rows_removed_after_dataset_cutoff']:,}")
    print(f"Genres covered: {overview['genre_count']:,}")
    print(f"Average rating: {overview['average_rating']:.2f}")
    print(f"Median runtime: {overview['median_runtime']:.2f} minutes")
    print(f"Total revenue: ${overview['total_revenue']:,.0f}")
    print()
    print("Key insights")
    for insight in analysis_result["insights"]:
        print(f"- {insight}")

    production_trend = advanced_result.get("production_trend", {})
    if production_trend and "error" not in production_trend:
        print()
        print("Advanced analysis")
        print(
            f"- Production trend {production_trend['period']}: {production_trend['trend_direction']} "
            f"(peak year {production_trend['peak_year']}, {production_trend['peak_count']} movies)."
        )

    budget_revenue = advanced_result.get("budget_revenue_correlation", {})
    if budget_revenue and "error" not in budget_revenue:
        print(
            f"- Budget/revenue correlation: {budget_revenue['correlation']} "
            f"with R^2 = {budget_revenue['r_squared']}."
        )

    print()
    print("Box office forecast validation")
    if "error" in forecast_result:
        print(f"- Forecasting was skipped: {forecast_result['error']}")
    else:
        forecast_overview = forecast_result["overview"]
        print(
            f"- Holdout year {forecast_overview['validation_year']} "
            f"({forecast_overview['validation_start']} -> {forecast_overview['validation_end']}): "
            f"{forecast_overview['validation_rows']} movies across {forecast_overview['genres_evaluated']} genres."
        )
        print(
            f"- Overall forecast error: MAE ${forecast_overview['overall_mae']:,.0f}, "
            f"WAPE {forecast_overview['overall_wape_percent']:.2f}%, "
            f"R^2 = {forecast_overview['overall_r_squared']}."
        )

        top_rows = forecast_result["summary_by_genre"].head(5)
        for _, row in top_rows.iterrows():
            print(
                f"- {row['genre']}: {int(row['validation_samples'])} validation movies, "
                f"WAPE {float(row['wape_percent']):.2f}%, "
                f"actual ${float(row['actual_total_revenue']):,.0f}, "
                f"predicted ${float(row['predicted_total_revenue']):,.0f}."
            )

    if chart_paths:
        print()
        print("Generated charts")
        for name, path in chart_paths.items():
            print(f"- {name}: {path}")

    if "paths" in forecast_result:
        print()
        print("Forecast outputs")
        for name, path in forecast_result["paths"].items():
            print(f"- {name}: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Integrated IMDB movies analysis project")
    parser.add_argument("--mode", choices=["report", "ui"], default="report", help="Run the command-line report or the Tkinter UI.")
    parser.add_argument("--dataset", default=None, help="Optional path to the dataset CSV file.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for exported charts and cleaned data.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "ui":
        from imdb_ui import launch_app

        launch_app(dataset_path=args.dataset)
        return

    run_report(dataset_path=args.dataset, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
