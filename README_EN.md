# IMDB Movies Dataset Project

This file is the English version of the documentation. The Chinese version is available at [README.md](/Users/m/Desktop/DPW%20project/DPWproject/README.md).

## Overview

This project is a course-style movie analytics application built on top of the IMDB / Kaggle movie dataset. It combines data cleaning, exploratory data analysis, chart generation, and optional advanced analysis, with two ways to use it:

- Command-line report mode: run the full pipeline and export outputs in one step
- Tkinter desktop UI mode: interactively filter records, inspect results, refresh charts, and review the data quality report

Typical use cases:

- coursework or portfolio-style data analysis projects
- practicing Pandas data cleaning and aggregation
- practicing Matplotlib chart generation
- packaging analysis logic into reusable scripts and a small desktop tool

## Features

### 1. Data Cleaning

- prioritizes the raw Kaggle movie dataset when available
- merges `movies_metadata.csv`, `keywords.csv`, and `credits.csv`
- standardizes `release_date`
- removes records released after `2017-07-31`, matching the dataset coverage window
- automatically derives `year`
- removes missing-title rows, duplicates, and invalid dates
- cleans numeric fields such as `budget`, `revenue`, `runtime`, `vote_average`, `vote_count`, and `popularity`
- fills selected numeric gaps using genre medians and global medians
- produces a data quality report with both raw and cleaned date ranges

### 2. Exploratory Data Analysis

- computes movie count, genre count, average rating, median runtime, and total revenue
- summarizes volume, rating, and average revenue by genre
- summarizes volume, rating, and average revenue by year
- generates a Top Movies view
- produces short auto-generated insights

### 3. Visualization

- `genre_distribution.png`: genre distribution chart
- `genre_comparison.png`: genre volume vs average rating chart
- `yearly_rating_trend.png`: yearly rating trend chart
- `budget_revenue_scatter.png`: budget vs revenue scatter chart

### 4. Advanced Analytics

- production trend over the cleaned year range (ending at `2017-07-31`)
- budget vs revenue correlation analysis
- decade comparison
- outlier detection interface

### 5. Desktop UI

- filter by genre
- filter by year range
- filter by minimum rating
- fuzzy filter by title or keyword
- inspect results, insights, charts, and the cleaning report
- regenerate charts for the current filtered selection
- record UI interaction logs for debugging and behavior review

## Project Structure

Key files:

- [main.py](/Users/m/Desktop/DPW%20project/DPWproject/main.py): unified entry point for `report` and `ui`
- [data_preprocessing.py](/Users/m/Desktop/DPW%20project/DPWproject/data_preprocessing.py): loading, merging, cleaning, and quality reporting
- [data_analysis.py](/Users/m/Desktop/DPW%20project/DPWproject/data_analysis.py): filtering, EDA summaries, and insight generation
- [data_visualization.py](/Users/m/Desktop/DPW%20project/DPWproject/data_visualization.py): chart generation and export
- [advanced_analytics.py](/Users/m/Desktop/DPW%20project/DPWproject/advanced_analytics.py): advanced analytics
- [imdb_ui.py](/Users/m/Desktop/DPW%20project/DPWproject/imdb_ui.py): Tkinter desktop application
- [ui_log_report.py](/Users/m/Desktop/DPW%20project/DPWproject/ui_log_report.py): UI log summarizer
- [requirements.txt](/Users/m/Desktop/DPW%20project/DPWproject/requirements.txt): dependency list

Common directories:

- `raw_data/`: raw Kaggle dataset
- `outputs/`: cleaned exports and generated charts
- `logs/`: UI operation logs
- `.cache/`: Matplotlib / fontconfig cache

## Data Source Priority

The project looks for data sources in this order:

1. `raw_data/kagglehub/datasets/rounakbanik/the-movies-dataset/versions/7/movies_metadata.csv`
2. `raw_data/the-movies-dataset/movies_metadata.csv`
3. `movies_metadata.csv` in the project root
4. `cleaned_imdb_movies.csv` in the project root

When the raw Kaggle dataset is available, the pipeline prioritizes:

- `movies_metadata.csv`
- `keywords.csv`
- `credits.csv`

Additional tables such as `ratings_small.csv` and `links_small.csv` may also be loaded, but the main cleaned dataset currently depends primarily on movie metadata, keywords, and credits.

## Requirements

- Python 3.9 or newer
- macOS / Linux / Windows for CLI usage
- a desktop display environment for Tkinter UI mode

## Installation

If you already have the project virtual environment, use:

```bash
./.venv/bin/python -m pip install -r requirements.txt
```

Or use your own Python environment:

```bash
python -m pip install -r requirements.txt
```

Main dependencies:

- `numpy`
- `pandas`
- `matplotlib`
- `Pillow`
- `kagglehub`

## Running the Project

### 1. Command-Line Report Mode

Run:

```bash
./.venv/bin/python main.py --mode report
```

This will:

- load and clean the dataset
- run EDA
- run advanced analytics
- export charts to `outputs/`
- print key metrics and insights in the terminal

Optional example:

```bash
./.venv/bin/python main.py --mode report --dataset cleaned_imdb_movies.csv --output-dir outputs
```

Arguments:

- `--mode`: `report` or `ui`
- `--dataset`: optional custom dataset path
- `--output-dir`: export directory for charts and cleaned data

### 2. Desktop UI Mode

Run:

```bash
./.venv/bin/python main.py --mode ui
```

The UI contains four main tabs:

- `Results`: paginated result table
- `Insights`: generated insights and advanced analysis summary
- `Charts`: chart preview and refresh actions
- `Data Quality`: cleaning report, missing values, and numeric summaries

Filters supported in the sidebar:

- genre
- start year
- end year
- minimum rating
- title / keyword

From the UI you can:

- click `Apply` to run filters
- click `Reset` to clear filters
- click `Refresh Charts` to rebuild charts for the current selection

Notes:

- regular charts can be rebuilt from the current filtered selection
- `Yearly Forecast vs Actual` and `Multi-Model Forecast Lines` are fixed forecast charts
- both fixed forecast charts always use the full cleaned dataset
- they do not change with filters and do not need refreshing

### 3. Regenerate Only the Cleaned Dataset

If the raw Kaggle dataset is already available and you only want to rebuild the cleaned file:

```bash
./.venv/bin/python data_preprocessing.py
```

This generates:

- `cleaned_imdb_movies.csv`

## Output Files

By default, outputs are written to `outputs/`. Typical files include:

- `cleaned_imdb_movies_project.csv`: cleaned dataset exported by the main project pipeline
- `genre_distribution.png`: genre distribution chart
- `genre_comparison.png`: genre comparison chart
- `yearly_rating_trend.png`: yearly rating trend chart
- `budget_revenue_scatter.png`: budget vs revenue scatter chart
- `chart_manifest.json`: metadata for the current chart batch

## Date Range Notes

The project now distinguishes between two date ranges:

- raw date range: recalculated directly from the source `release_date` values
- cleaned date range: recalculated after invalid dates and post-`2017-07-31` records are removed

Because this Kaggle dataset is only reliably covered through `2017-07-31`, the cleaning pipeline drops later release dates. Charts and advanced analytics run on that cleaned range.

## What the Data Quality Report Includes

The `Data Quality` tab and the generated cleaning report include:

- source file name
- loaded row count
- cleaned row count
- duplicates removed
- rows removed for missing titles
- rows removed for invalid release dates
- rows removed after the dataset cutoff date
- raw date range
- cleaned date range
- missing values by column after cleaning
- min / median / max for key numeric fields

## UI Logging and Debugging

The desktop UI records interaction logs at:

- `logs/ui_operation_log.jsonl`

To summarize the most recent UI session:

```bash
./.venv/bin/python ui_log_report.py
```

To summarize the full log history:

```bash
./.venv/bin/python ui_log_report.py --all
```

The log report helps inspect:

- analysis request counts
- debounced or ignored requests
- chart refresh conflicts
- average processing times
- recent event history

This is useful when diagnosing issues such as:

- Apply was clicked but the screen did not appear to refresh
- charts did not update immediately
- an older filter result was discarded

## Example Commands

### Generate a full report and export charts

```bash
./.venv/bin/python main.py --mode report
```

### Run against a specific dataset file

```bash
./.venv/bin/python main.py --mode report --dataset cleaned_imdb_movies.csv
```

### Open the interactive desktop UI

```bash
./.venv/bin/python main.py --mode ui
```

### Review the UI log summary

```bash
./.venv/bin/python ui_log_report.py
```

## Verified Current Behavior

Based on the current repository state, the following has been verified:

- both raw and cleaned date ranges are shown
- titles released after `2017-07-31` are removed during cleaning
- `yearly_rating_trend.png` is exported using the cleaned valid year range
- the advanced production trend respects the dataset cutoff
- the UI can regenerate charts for the current filtered selection

## Known Limitations

- the Tkinter UI is intended for local desktop use, not headless server use
- some years have very few records, which can produce unstable yearly averages
- budget and revenue fields contain many missing or imperfect source values, so those analytics should be treated as directional rather than strict financial conclusions
- the analysis layer supports `language` and `country`, but those filters are not yet exposed as UI controls

## Possible Next Steps

- add language and country filters to the UI
- export filtered results as CSV
- add sparse-year warnings to charts
- expand advanced analytics with more statistical outputs
- add unit tests and regression tests
