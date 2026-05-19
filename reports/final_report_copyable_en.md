# IMDB Movie Analytics and Box Office Forecasting

**Date:** May 19, 2026

## Project Overview

This project analyzes the Kaggle movie dataset associated with IMDB metadata and builds an integrated workflow for data cleaning, exploratory data analysis, visualization, and box office forecasting. The main objective is to transform raw movie records into interpretable findings about genre distribution, rating patterns, production trends, and revenue behavior. In the backtest window, the 2017 slice only covers January to July, so it should be interpreted as `2017 (Jan-Jul only)`.

The system is organized into reusable Python modules for preprocessing, analysis, visualization, forecasting, and a Tkinter desktop UI. All modules share the same cleaned dataset, which improves consistency between terminal reports, exported charts, and interactive exploration. In this way, the project demonstrates both descriptive analytics and a simple predictive workflow in one complete application.

Several insights stand out from the analysis. Drama and Comedy dominate the dataset by volume, Animation has the strongest average rating among large genres, Adventure and Animation show strong average revenue, movie production increases substantially over time, and budget is strongly correlated with revenue. A rolling backtest across 2003-2017 further shows that historical metadata can explain a meaningful share of box office variation.

## Data Set

The project uses The Movies Dataset from Kaggle, combining movie metadata with keyword and production-related information. The cleaned project dataset covers releases from 1874-12-09 to 2017-07-29, because records after 2017-07-31 are removed to keep the analysis window consistent.

After cleaning, the dataset contains 45,293 movies, 21 primary genres, an average rating of 5.63, a median runtime of 95 minutes, and about $1.19 trillion in total recorded revenue. Main fields used in the analysis include `title`, `primary_genre`, `budget`, `revenue`, `release_date`, `year`, `runtime`, `vote_average`, `vote_count`, `language`, `country`, and `keyword`.

The preprocessing workflow removes invalid dates, duplicate movies, and missing-title rows, standardizes numeric fields, derives the `year` column, and keeps `source_id` for traceability. This produces a unified dataset that supports both descriptive analysis and forecasting.

## Features and Approaches

### Data Cleaning, EDA, and Visualization

This feature provides the analytical core of the project. It loads the cleaned dataset, computes summary statistics, aggregates movies by genre and by year, and generates automatic insights. It also exports figures such as genre distribution, genre comparison, yearly rating trend, and a budget-versus-revenue scatter plot.

The approach relies mainly on Pandas for filtering, grouping, and metric calculation, and Matplotlib for visualization. Genre-level summaries are used to compare movie count, average rating, and average revenue, while yearly summaries highlight changes in movie output and rating patterns over time. The cleaned dataset is reused across all modules so that every chart and metric follows the same data definition.

This feature makes the project useful for both reporting and presentation. The exported charts in the `outputs/` directory can be directly included in the final report or presentation slides.

### Time Series Analysis of Movie Production and Revenue

This project also contains a standalone time series analysis perspective. Instead of treating the dataset only as a static collection of films, it aggregates records by year to study how movie production volume, rating behavior, and market outcomes change over time.

The project implements this in two ways. First, it builds yearly summaries to identify long-term changes in movie count, average rating, and revenue-related trends. Second, it uses rolling backtesting in the forecasting module, where each validation year is predicted using only earlier years as training data. This preserves chronological order and makes the evaluation more realistic than a random train-test split.

The time-based analysis shows that movie production generally increases over time, especially after 2000. It also shows that the forecasting system can follow broad year-level revenue movement across the backtest window, even if it cannot perfectly model every individual movie. Because the dataset is truncated at `2017-07-31`, the 2017 slice should be interpreted as `2017 (Jan-Jul only)`.

### Genre-Level Box Office Forecasting and Backtesting

This feature extends the project from descriptive analytics to predictive analytics. Instead of focusing on only one year, the project uses a rolling backtest from 2003 to 2017. For each validation year, movies released before that year are used as training data, and movies in the validation year are used to evaluate forecast quality.

The model is a ridge-style regression with genre-aware calibration. It uses budget, log-budget terms, runtime, squared runtime, release year, cyclical month variables, language and country indicators, and observed-data flags as features. After fitting a shared model, genre-level calibration adjusts predictions to better match historical genre behavior.

The forecasting component produces yearly backtest summaries and a global comparison chart of actual versus predicted total revenue across the 2003-2017 window. This makes it possible to evaluate not only single-movie behavior in one year, but also whether the model captures longer-term box office patterns.

## Key Findings

First, genre distribution is highly unbalanced. Drama (11,931) and Comedy (8,812) are the largest categories, followed by Action (4,474). This suggests that many overall trends are heavily influenced by these high-volume genres.

Second, genre quality and commercial performance differ. Animation has the highest average rating among sufficiently large genres (6.31), while Adventure, Animation, Science Fiction, and Family have the strongest average revenue. This means the highest-rated genres are not exactly the same as the most profitable ones.

Third, movie production shows a long-term increasing trend, with 2014 as the busiest year (1,972 movies). Budget and revenue are strongly correlated (0.7545, `R^2 = 0.5693`), showing that larger productions tend to earn more but budget alone does not fully determine performance.

Finally, the forecasting module is better described as a multi-year backtest than a single-year prediction. Across the 2003-2017 window, yearly WAPE ranges from 53.60% to 67.14%, with an average WAPE of 59.98% and an average `R^2` of 0.6057. This shows that the model captures broad box office trends, although forecast quality still varies by year.

## Conclusion and Discussion

This project demonstrates a complete data-analysis pipeline built from a large movie dataset. It combines preprocessing, descriptive analysis, visualization, forecasting, and an optional desktop UI into a consistent structure. The project successfully turns raw movie records into interpretable conclusions about genre patterns, ratings, production growth, and box office behavior.

The forecasting backtest shows that structured historical metadata can explain a meaningful share of revenue variation, but the errors also reveal clear limitations. Important factors such as cast popularity, franchise strength, marketing effort, and release competition are not included in the current feature set, and some genres have too few validation examples for stable evaluation.

Future improvements could include richer feature engineering, comparison with additional models, more robust outlier handling, and stronger interactive visualization. Even so, the current project already provides a solid coursework example with reproducible outputs and clear analytical value.
