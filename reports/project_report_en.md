# IMDB Movie Analytics and Box Office Forecasting Report

## 1. Introduction

This project is a course-style movie analytics application built on top of the Kaggle movie dataset associated with IMDB metadata. It integrates data cleaning, exploratory data analysis (EDA), visualization, a desktop UI, and a box office forecasting module with rolling backtesting across 2003-2017. In this backtest window, the 2017 slice only covers January to July, so it should be interpreted as `2017 (Jan-Jul only)`.

The project goes beyond a simple notebook-style summary. It organizes the full workflow into reusable Python modules so that cleaning, reporting, chart generation, forecasting, and UI interaction all operate on the same cleaned dataset. This makes the project suitable both as a coursework submission and as a portfolio-style data project.

## 2. Dataset Scope

The underlying data comes from the Kaggle movie dataset and combines movie metadata with keyword and production-related information. To keep the analysis window consistent, the cleaning pipeline removes records released after `2017-07-31`. As a result, the final dataset represents movie records up to late July 2017.

The cleaned project dataset contains:

- 45,293 movies
- release dates from `1874-12-09` to `2017-07-29`
- 21 primary genres
- an average rating of 5.63
- a median runtime of 95 minutes
- total recorded revenue of about $1.19 trillion

This is a large historical dataset, which makes it appropriate for long-range descriptive analysis and a basic forecasting exercise.

## 3. Data Cleaning and Preparation

The data preprocessing module provides the foundation for the whole project. Its main responsibilities include:

- merging movie metadata, keywords, and production-related tables
- standardizing `release_date`
- removing invalid dates
- removing all records after `2017-07-31`
- deriving the `year` field automatically
- removing rows with missing titles or duplicate movies
- cleaning numeric fields such as `budget`, `revenue`, `runtime`, `vote_average`, `vote_count`, and `popularity`
- preserving `source_id` to trace records back to the original source
- exporting a cleaned and normalized dataset for reuse

Because all downstream modules share the same cleaned dataset, the project reduces inconsistencies between analysis, forecasting, and UI views.

## 4. Methods

The project uses two broad analytical approaches: descriptive analytics and predictive analytics.

### 4.1 Descriptive Analytics

The descriptive section focuses on questions such as:

- Which genres dominate the dataset?
- How do movie volume and average ratings change over time?
- How do genres differ in rating and revenue performance?
- Which movies stand out by revenue or user rating?

Pandas is used for aggregation, while Matplotlib is used to export charts, including:

- `genre_distribution.png`
- `genre_comparison.png`
- `yearly_rating_trend.png`
- `budget_revenue_scatter.png`
- `forecast_backtest_yearly_comparison.png`

### 4.2 Predictive Analytics

For forecasting, the project does not focus on only one holdout year. Instead, it uses a rolling backtest across 2003-2017. For each validation year, movies released before that year are used for training, and movies released in that year are used for validation. The model is a ridge-style regression with genre-aware calibration. Core features include:

- budget and log-transformed budget terms
- runtime and squared runtime
- release year
- cyclical month features
- whether the movie is in English
- whether it is a US production
- whether budget and runtime are directly observed values

The goal is not to reproduce an industrial forecasting system, but to test whether structured historical metadata can explain a meaningful share of box office variation.

## 5. Time Series Analysis

Beyond static genre comparisons and cross-sectional summaries, this project also supports a standalone time series perspective on the movie market. The analysis aggregates the cleaned dataset by year to examine how movie volume, average rating, and box office behavior evolve over time, and then connects those patterns to the rolling forecast backtest.

The key idea is that year is not just another field in the dataset. It reflects changes in industry scale, audience taste, and market conditions. Without preserving time order, it is difficult to properly explain long-term production growth, rating changes, and revenue trends.

In practice, the project implements time series analysis in two main ways:

- it aggregates movie data by year to measure production volume, average rating, and revenue trends
- it uses rolling backtesting in the forecasting module, where each validation year is predicted using only earlier years as training data

This is more realistic than a random train-test split because it preserves chronological order and avoids leaking future information into the past.

The results show a clear long-term increase in movie production, especially after 2000. They also show that the `2003-2017` rolling backtest can track broad year-level box office movement even when it cannot precisely explain every individual movie outcome. Because the dataset is truncated at `2017-07-31`, the 2017 slice in charts and backtests should be interpreted as `2017 (Jan-Jul only)` rather than a full calendar year.

## 6. Main Findings

### 5.1 Genre Distribution

The most common genres in the cleaned dataset are:

- `Drama`: 11,931 movies
- `Comedy`: 8,812 movies
- `Action`: 4,474 movies
- `Documentary`: 3,397 movies
- `Horror`: 2,616 movies

This indicates that the dataset is heavily concentrated in drama and comedy, which also affects many aggregate results.

### 5.2 Genre Quality and Commercial Performance

Among genres with at least 50 movies, the highest average ratings are found in:

- `Animation`: 6.31
- `War`: 5.93
- `Drama`: 5.88
- `Crime`: 5.86

The strongest genres by average revenue are:

- `Adventure`: about $100.0 million
- `Animation`: about $91.4 million
- `Science Fiction`: about $74.7 million
- `Family`: about $66.1 million

This suggests that audience-rated quality and commercial return do not perfectly overlap. Broad-appeal genres such as adventure, animation, science fiction, and family titles tend to perform especially well financially.

### 5.3 Time Trend

The automated analysis reports that:

- movie production shows a long-term increasing trend
- the busiest year is 2014 with 1,972 movies
- the `2010s` are the most productive decade in the cleaned dataset
- among years with at least five releases, 1924 has the highest average rating at 6.61

Overall, the dataset reflects strong expansion in movie output, especially after 2000.

### 5.4 Budget and Revenue Relationship

The budget-versus-revenue analysis shows:

- correlation coefficient: 0.7545
- interpreted strength: strong correlation
- `R^2`: 0.5693

This means higher-budget movies usually generate higher revenue, and budget explains a substantial share of box office variation. Still, nearly half of the variation remains unexplained by a simple linear relationship, implying that factors such as franchise strength, marketing, release timing, and audience reception also matter.

### 5.5 Representative Movies

The top revenue movies in the dataset include:

- `Avatar` (2009): $2.79 billion
- `Star Wars: The Force Awakens` (2015): $2.07 billion
- `Titanic` (1997): $1.85 billion
- `The Avengers` (2012): $1.52 billion
- `Jurassic World` (2015): $1.51 billion

Highly rated and high-vote examples include:

- `The Shawshank Redemption`
- `The Godfather`
- `Your Name.`
- `The Dark Knight`
- `Fight Club`

These examples show that the project captures both commercial blockbusters and critically well-regarded titles.

## 7. Multi-Year Forecast Backtesting (2003-2017)

To avoid overemphasizing a single year, the forecasting module performs yearly backtesting across 2003-2017 and visualizes annual predicted-versus-actual revenue totals in `forecast_backtest_yearly_comparison.png`. The main summary is:

- backtest years: 2003-2017
- number of validation years: 15
- average WAPE: 59.98%
- median WAPE: 58.61%
- average `R^2`: 0.6057
- best WAPE year: 2010 at 53.60%
- worst WAPE year: 2006 at 67.14%

These results show that the model has meaningful explanatory power across multiple years rather than only in one isolated validation split. In most years, WAPE stays in roughly the 53%-67% range, and the average `R^2` is above 0.60, which suggests that the model captures broad box office patterns reasonably well.

The yearly comparison chart also shows that predicted and actual total revenue generally move in the same direction over time, although some years are still underpredicted or overpredicted. This indicates that structured variables such as budget, runtime, release timing, language, and country explain a substantial share of revenue variation, but not the full market outcome.

Taken together, the forecasting module is best understood as a useful and interpretable multi-year academic backtest rather than a production-grade box office prediction engine.

## 8. Project Deliverables and Presentation Value

One of the strengths of this project is that it produces multiple reusable outputs:

- a cleaned export dataset
- EDA charts
- forecast result tables and a yearly backtest comparison chart
- a yearly forecast comparison chart
- a Tkinter desktop interface for interactive exploration

The `outputs/` directory already contains the core artifacts needed for both written reporting and in-class presentation. This makes the project more complete than a terminal-only or notebook-only analysis.

## 9. Limitations

The project is solid as a course project, but several limitations remain:

- the dataset ends at `2017-07-31`, so it does not reflect more recent industry patterns
- the forecasting model uses mostly structured metadata and does not include cast popularity, director effects, marketing spend, franchise strength, or release-window competition
- some genres have limited validation samples, which makes genre-level metrics unstable
- blockbuster outliers can heavily influence MAE, RMSE, and percentage-based error measures
- very early years are sparse, so some long-range conclusions should be interpreted carefully

These limitations mean the results should be treated as evidence within the project’s chosen data window and modeling design, not as a complete explanation of movie market behavior.

## 10. Conclusion

Overall, this project successfully delivers a full movie analytics workflow. It transforms raw movie records into a cleaned dataset, descriptive insights, time-based analysis, visual outputs, and a forecasting backtest component that adds a predictive dimension to the analysis.

The main conclusions are:

- drama and comedy dominate the dataset by volume
- animation stands out in average rating, while adventure and animation are especially strong in average revenue
- movie production increases over the long run and peaks in 2014 within this dataset
- budget and revenue are strongly correlated, but budget alone cannot explain all market outcomes
- historical metadata can support a moderately effective 2003-2017 box office backtesting exercise, although important market factors remain outside the current model

With further development, the project could be extended through richer feature engineering, model comparison, and more advanced interactive visualization. Even in its current form, it already demonstrates good completeness, reproducibility, and presentation value for a data analysis project.
