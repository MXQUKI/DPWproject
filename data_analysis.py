"""Exploratory analysis helpers for the IMDB movies project."""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, Optional

import pandas as pd

COL_RATING = "vote_average"
COL_GENRE = "primary_genre"
COL_TITLE = "title"
COL_RELEASE = "release_date"
COL_YEAR = "year"
COL_LANGUAGE = "language"
COL_COUNTRY = "country"
COL_KEYWORD = "keyword"


def _years(df: pd.DataFrame) -> pd.Series:
    if COL_YEAR in df.columns:
        return pd.to_numeric(df[COL_YEAR], errors="coerce")
    if COL_RELEASE not in df.columns:
        raise KeyError("Dataset needs a 'year' or 'release_date' column.")

    release_dates = df[COL_RELEASE]
    if pd.api.types.is_datetime64_any_dtype(release_dates):
        return release_dates.dt.year
    return pd.to_datetime(release_dates, errors="coerce").dt.year


def _genre_tokens(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]

    text = str(value).strip()
    if not text:
        return []
    separators = ("|", ",")
    for separator in separators:
        if separator in text:
            return [part.strip() for part in text.split(separator) if part.strip()]
    return [text]


def _contains_text(value: object, needle: Optional[str]) -> bool:
    if not needle or needle.strip().lower() in {"", "all"}:
        return True
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return needle.strip().lower() in str(value).lower()


def _search_text_matches(value: object, needle: str) -> bool:
    """Match a query against a text field using whole-word or whole-phrase matching."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False

    normalized_value = _normalize_search_text(value)
    normalized_needle = _normalize_search_text(needle)
    if not normalized_value or not normalized_needle:
        return False

    pattern = re.compile(rf"(?<!\w){re.escape(normalized_needle)}(?!\w)", flags=re.IGNORECASE)
    return pattern.search(normalized_value) is not None


def _search_text_equals(value: object, needle: str) -> bool:
    normalized_value = _normalize_search_text(value)
    normalized_needle = _normalize_search_text(needle)
    return bool(normalized_value) and bool(normalized_needle) and normalized_value == normalized_needle


def _search_text_startswith(value: object, needle: str) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False

    normalized_value = _normalize_search_text(value)
    normalized_needle = _normalize_search_text(needle)
    if not normalized_value or not normalized_needle:
        return False

    pattern = re.compile(rf"^{re.escape(normalized_needle)}(?!\w)", flags=re.IGNORECASE)
    return pattern.search(normalized_value) is not None


def _keyword_token_exact_match(value: object, needle: str) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False

    tokens = _genre_tokens(value)
    return any(_search_text_equals(token, needle) for token in tokens)


def _keyword_token_match(value: object, needle: str) -> bool:
    """Match a search term against a pipe/comma-delimited keyword field."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False

    tokens = _genre_tokens(value)
    return any(_search_text_matches(token, needle) for token in tokens)


def _normalize_search_text(value: object) -> str:
    """Normalize search text so punctuation variants still match."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""

    text = unicodedata.normalize("NFKC", str(value)).lower()
    characters: list[str] = []
    for char in text:
        category = unicodedata.category(char)
        if category.startswith(("P", "S")):
            characters.append(" ")
        else:
            characters.append(char)
    normalized = "".join(characters)
    normalized = " ".join(normalized.split())
    return normalized


def apply_filters(df: pd.DataFrame, filters: Optional[Dict[str, object]] = None) -> pd.DataFrame:
    """Apply UI or CLI filters to the dataset."""
    filters = filters or {}
    filtered = df

    year_from = filters.get("year_from")
    year_to = filters.get("year_to")
    if year_from is not None or year_to is not None:
        years = _years(filtered)
        if year_from is not None:
            filtered = filtered.loc[years >= int(year_from)]
        if year_to is not None:
            filtered = filtered.loc[years <= int(year_to)]

    min_rating = filters.get("min_rating")
    if min_rating is not None and COL_RATING in filtered.columns:
        ratings = filtered[COL_RATING]
        if not pd.api.types.is_numeric_dtype(ratings):
            ratings = pd.to_numeric(ratings, errors="coerce")
        filtered = filtered.loc[ratings >= float(min_rating)]

    genre = filters.get("genre")
    if genre and str(genre).strip().lower() != "all" and COL_GENRE in filtered.columns:
        genre_mask = filtered[COL_GENRE].astype("string").str.contains(str(genre).strip(), case=False, na=False, regex=False)
        filtered = filtered.loc[genre_mask]

    language = filters.get("language")
    if language and COL_LANGUAGE in filtered.columns:
        language_mask = filtered[COL_LANGUAGE].astype("string").str.contains(str(language).strip(), case=False, na=False, regex=False)
        filtered = filtered.loc[language_mask]

    country = filters.get("country")
    if country and COL_COUNTRY in filtered.columns:
        country_mask = filtered[COL_COUNTRY].astype("string").str.contains(str(country).strip(), case=False, na=False, regex=False)
        filtered = filtered.loc[country_mask]

    keyword = filters.get("keyword") or filters.get("keywords") or filters.get("title_keyword")
    if keyword:
        keyword = str(keyword).strip()
        title_phrase_mask = pd.Series(False, index=filtered.index)
        keyword_phrase_mask = pd.Series(False, index=filtered.index)

        if COL_TITLE in filtered.columns:
            title_phrase_mask = filtered[COL_TITLE].map(lambda value: _search_text_matches(value, keyword))

        if COL_KEYWORD in filtered.columns:
            keyword_phrase_mask = filtered[COL_KEYWORD].map(lambda value: _keyword_token_match(value, keyword))

        combined_mask = title_phrase_mask | keyword_phrase_mask
        filtered = filtered.loc[combined_mask]

    return filtered.reset_index(drop=True)


def aggregate_by_genre(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise movie count and rating by genre."""
    if df.empty:
        return pd.DataFrame(columns=["genre", "movie_count", "avg_rating", "avg_revenue"])
    if COL_GENRE not in df.columns or COL_RATING not in df.columns:
        raise KeyError("Dataset needs 'primary_genre' and 'vote_average'.")

    working = df[[COL_GENRE, COL_RATING] + [col for col in ("revenue",) if col in df.columns]].copy()
    working["_genre"] = working[COL_GENRE].map(_genre_tokens)
    working = working.explode("_genre")
    working = working.loc[working["_genre"].notna() & (working["_genre"].astype("string").str.strip() != "")]
    working["_genre"] = working["_genre"].astype("string").str.strip()

    aggregations = {
        "movie_count": (COL_RATING, "count"),
        "avg_rating": (COL_RATING, "mean"),
    }
    if "revenue" in working.columns:
        aggregations["avg_revenue"] = ("revenue", "mean")

    grouped = working.groupby("_genre").agg(**aggregations).reset_index().rename(columns={"_genre": "genre"})
    grouped["avg_rating"] = grouped["avg_rating"].round(2)
    if "avg_revenue" in grouped.columns:
        grouped["avg_revenue"] = grouped["avg_revenue"].round(2)
    return grouped.sort_values(["movie_count", "avg_rating"], ascending=[False, False]).reset_index(drop=True)


def yearly_rating_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise movie volume and average rating by year."""
    if df.empty:
        return pd.DataFrame(columns=["year", "movie_count", "avg_rating", "avg_revenue"])

    working = df.copy()
    working["year"] = _years(working)
    working = working.dropna(subset=["year"])
    working["year"] = working["year"].astype(int)

    aggregations = {
        "movie_count": (COL_TITLE, "count"),
        "avg_rating": (COL_RATING, "mean"),
    }
    if "revenue" in working.columns:
        aggregations["avg_revenue"] = ("revenue", "mean")

    grouped = working.groupby("year").agg(**aggregations).reset_index().sort_values("year")
    grouped["avg_rating"] = grouped["avg_rating"].round(2)
    if "avg_revenue" in grouped.columns:
        grouped["avg_revenue"] = grouped["avg_revenue"].round(2)
    return grouped.reset_index(drop=True)


def top_movies(df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    """Return the strongest movies for display in the UI/report."""
    if df.empty:
        return pd.DataFrame(columns=[COL_TITLE, COL_GENRE, COL_YEAR, COL_RATING, "revenue"])

    working = df.copy()
    working["year"] = _years(working)
    sort_columns = [column for column in (COL_RATING, "revenue", COL_TITLE) if column in working.columns]
    ordered = working.sort_values(sort_columns, ascending=[False, False, True][: len(sort_columns)])
    columns = [column for column in (COL_TITLE, COL_GENRE, "year", COL_RATING, "runtime", "revenue") if column in ordered.columns]
    return ordered[columns].head(limit).reset_index(drop=True)


def summarize_dataset(df: pd.DataFrame) -> Dict[str, float]:
    """Compute top-level EDA metrics."""
    if df.empty:
        return {
            "movie_count": 0,
            "genre_count": 0,
            "average_rating": 0.0,
            "median_runtime": 0.0,
            "total_revenue": 0.0,
        }

    return {
        "movie_count": int(len(df)),
        "genre_count": int(df[COL_GENRE].nunique()) if COL_GENRE in df.columns else 0,
        "average_rating": round(float(pd.to_numeric(df[COL_RATING], errors="coerce").mean()), 2),
        "median_runtime": round(float(pd.to_numeric(df["runtime"], errors="coerce").median()), 2) if "runtime" in df.columns else 0.0,
        "total_revenue": round(float(pd.to_numeric(df["revenue"], errors="coerce").sum()), 2) if "revenue" in df.columns else 0.0,
    }


def generate_insights(
    df: pd.DataFrame,
    genre_summary: Optional[pd.DataFrame] = None,
    yearly_summary: Optional[pd.DataFrame] = None,
) -> list[str]:
    """Generate short, presentation-ready insights from the dataset."""
    insights: list[str] = []
    if df.empty:
        return ["No rows match the current filters."]

    genre_summary = genre_summary if genre_summary is not None else aggregate_by_genre(df)
    yearly_summary = yearly_summary if yearly_summary is not None else yearly_rating_trend(df)

    if not genre_summary.empty:
        most_common = genre_summary.iloc[0]
        insights.append(
            f"{most_common['genre']} is the most common genre with {int(most_common['movie_count'])} movies in the current selection."
        )

        stable_cutoff = max(5, min(20, len(df) // 50 if len(df) >= 50 else 5))
        comparable_genres = genre_summary.loc[genre_summary["movie_count"] >= stable_cutoff]
        comparison_pool = comparable_genres if not comparable_genres.empty else genre_summary
        best_genre = comparison_pool.sort_values("avg_rating", ascending=False).iloc[0]
        insights.append(
            f"{best_genre['genre']} has the strongest average rating at {best_genre['avg_rating']:.2f} among comparable genres."
        )

    if not yearly_summary.empty:
        peak_year = yearly_summary.sort_values("movie_count", ascending=False).iloc[0]
        insights.append(
            f"{int(peak_year['year'])} is the busiest release year with {int(peak_year['movie_count'])} movies."
        )

        rated_years = yearly_summary.loc[yearly_summary["movie_count"] >= 5]
        if not rated_years.empty:
            best_year = rated_years.sort_values("avg_rating", ascending=False).iloc[0]
            insights.append(
                f"{int(best_year['year'])} delivers the highest average score at {best_year['avg_rating']:.2f} among years with at least five releases."
            )

    if {"budget", "revenue"}.issubset(df.columns):
        budget_revenue = df.loc[
            pd.to_numeric(df["budget"], errors="coerce").gt(0)
            & pd.to_numeric(df["revenue"], errors="coerce").gt(0),
            ["budget", "revenue"],
        ].apply(pd.to_numeric, errors="coerce")
        if len(budget_revenue) >= 2:
            correlation = budget_revenue["budget"].corr(budget_revenue["revenue"])
            if pd.notna(correlation):
                insights.append(
                    f"Budget and revenue show a correlation of {correlation:.2f}, so larger productions generally earn more."
                )

    return insights[:5]


def analyze(df_clean: pd.DataFrame, filters: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """Run the full EDA pipeline used by the CLI and UI."""
    filtered = apply_filters(df_clean, filters)
    genre_summary = aggregate_by_genre(filtered)
    yearly_summary = yearly_rating_trend(filtered)
    overview = summarize_dataset(filtered)
    insights = generate_insights(filtered, genre_summary, yearly_summary)

    return {
        "filtered_df": filtered,
        "genre_summary": genre_summary,
        "yearly_summary": yearly_summary,
        "top_movies": top_movies(filtered),
        "overview": overview,
        "insights": insights,
        "meta": {
            "input_rows": int(len(df_clean)),
            "filtered_rows": int(len(filtered)),
            "genre_categories": int(len(genre_summary)),
            "years_covered": int(yearly_summary["year"].nunique()) if not yearly_summary.empty else 0,
        },
    }


if __name__ == "__main__":
    sample = pd.DataFrame(
        [
            {"title": "Toy Story", "primary_genre": "Animation", "vote_average": 8.3, "release_date": "1995-10-30", "revenue": 373554033},
            {"title": "Heat", "primary_genre": "Crime", "vote_average": 8.1, "release_date": "1995-12-15", "revenue": 187436818},
            {"title": "Up", "primary_genre": "Animation", "vote_average": 8.2, "release_date": "2009-05-29", "revenue": 735099082},
        ]
    )
    result = analyze(sample, {"genre": "Animation"})
    print(result["overview"])
    print(result["insights"])
