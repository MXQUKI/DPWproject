"""Utilities for loading, cleaning, caching, and profiling the IMDB movie dataset."""

from __future__ import annotations

import ast
from pathlib import Path
import threading
from typing import Dict, Iterable, Optional, Tuple
import unicodedata

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CLEANED_DATASET_PATH = PROJECT_ROOT / "cleaned_imdb_movies.csv"
RAW_DATASET_DIR_CANDIDATES = (
    PROJECT_ROOT / "raw_data" / "kagglehub" / "datasets" / "rounakbanik" / "the-movies-dataset" / "versions" / "7",
    PROJECT_ROOT / "raw_data" / "the-movies-dataset",
)
DEFAULT_DATASET_CANDIDATES = ("cleaned_imdb_movies.csv", "movies_metadata.csv")
NUMERIC_COLUMNS = ("budget", "revenue", "runtime", "vote_average", "vote_count", "popularity")
TEXT_COLUMNS = ("title", "original_title", "primary_genre", "language", "original_language", "country", "keyword")
TEXT_FILL_DEFAULTS = {
    "primary_genre": "Unknown",
    "language": "unknown",
    "original_language": "unknown",
    "country": "Unknown",
    "keyword": "Unknown",
}
NUMERIC_FALLBACK_DEFAULTS = {
    "budget": 0.0,
    "revenue": 0.0,
    "runtime": 0.0,
    "vote_average": 0.0,
    "vote_count": 0.0,
    "popularity": 0.0,
}
DATASET_CUTOFF_DATE = pd.Timestamp("2017-07-31")
COALESCED_COLUMN_SOURCES = {
    "title": ("title", "original_title"),
    "language": ("language", "original_language"),
    "country": ("country", "production_countries"),
    "keyword": ("keyword", "keywords"),
}
PIPE_JOINED_TEXT_COLUMNS = ("country", "keyword", "language")
OUTPUT_COLUMNS = (
    "id",
    "source_id",
    "title",
    "original_title",
    "primary_genre",
    "budget",
    "budget_observed",
    "revenue",
    "revenue_observed",
    "release_date",
    "year",
    "runtime",
    "runtime_observed",
    "vote_average",
    "vote_count",
    "popularity",
    "language",
    "original_language",
    "country",
    "keyword",
)
REQUIRED_CLEANED_COLUMNS = (
    "id",
    "title",
    "primary_genre",
    "budget",
    "budget_observed",
    "revenue",
    "revenue_observed",
    "release_date",
    "year",
    "runtime",
    "runtime_observed",
    "vote_average",
    "vote_count",
    "popularity",
    "language",
    "country",
    "keyword",
)

_DATASET_CACHE_LOCK = threading.Lock()
_DATASET_CACHE: Dict[tuple[str, str, str | None], tuple[pd.DataFrame, Dict[str, object]]] = {}


def clear_dataset_cache() -> None:
    """Clear the in-process cleaned dataset cache."""
    with _DATASET_CACHE_LOCK:
        _DATASET_CACHE.clear()


def resolve_raw_dataset_dir(dataset_dir: Optional[str | Path] = None) -> Path:
    """Resolve the raw Kaggle dataset directory when available."""
    if dataset_dir:
        candidate = Path(dataset_dir)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Raw dataset directory not found: {candidate}")

    for candidate in RAW_DATASET_DIR_CANDIDATES:
        if candidate.exists():
            return candidate

    checked = ", ".join(str(path) for path in RAW_DATASET_DIR_CANDIDATES)
    raise FileNotFoundError(f"Raw dataset directory not found. Checked: {checked}")


def resolve_dataset_path(dataset_path: Optional[str | Path] = None) -> Path:
    """Resolve a dataset file path relative to the project directory."""
    if dataset_path:
        candidate = Path(dataset_path)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Dataset not found: {candidate}")

    try:
        raw_dir = resolve_raw_dataset_dir()
        raw_movies = raw_dir / "movies_metadata.csv"
        if raw_movies.exists():
            return raw_movies
    except FileNotFoundError:
        pass

    for filename in DEFAULT_DATASET_CANDIDATES:
        candidate = PROJECT_ROOT / filename
        if candidate.exists():
            return candidate

    checked = ", ".join(DEFAULT_DATASET_CANDIDATES)
    raise FileNotFoundError(f"No dataset found. Checked: {checked}")


def resolve_cleaned_dataset_path(dataset_path: Optional[str | Path] = None) -> Path:
    """Resolve a cleaned dataset path and avoid falling back to raw source files."""
    if dataset_path:
        candidate = Path(dataset_path)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Cleaned dataset not found: {candidate}")

    if DEFAULT_CLEANED_DATASET_PATH.exists():
        return DEFAULT_CLEANED_DATASET_PATH

    raise FileNotFoundError(
        "Cleaned dataset not found. Expected cleaned_imdb_movies.csv in the project root "
        "or an explicit cleaned dataset path."
    )


def _parse_json_like(value: object) -> list[str]:
    """Parse a JSON-like cell and return a flattened list of names."""
    if value is None:
        return []

    try:
        if pd.isna(value):
            return []
    except TypeError:
        pass

    if isinstance(value, float) and pd.isna(value):
        return []

    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") or text.startswith("{"):
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                parsed = None
            if isinstance(parsed, list):
                items = parsed
            elif isinstance(parsed, dict):
                items = [parsed]
            else:
                items = [part.strip() for part in text.replace("|", ",").split(",") if part.strip()]
        else:
            items = [part.strip() for part in text.replace("|", ",").split(",") if part.strip()]
    else:
        items = [value]

    flattened: list[str] = []
    for item in items:
        try:
            if pd.isna(item):
                continue
        except TypeError:
            pass
        if isinstance(item, dict):
            name = item.get("name") or item.get("iso_639_1") or item.get("iso_3166_1")
            if name:
                flattened.append(str(name).strip())
        else:
            text = str(item).strip()
            if text:
                flattened.append(text)
    return flattened


def _extract_primary_genre(value: object) -> pd._libs.missing.NAType | str:
    genres = _parse_json_like(value)
    return genres[0] if genres else pd.NA


def _coalesce_column(df: pd.DataFrame, target: str, sources: Iterable[str]) -> None:
    """Fill a target column from the first available source columns."""
    if target not in df.columns:
        df[target] = pd.NA

    for source in sources:
        if source not in df.columns:
            continue
        df[target] = df[target].fillna(df[source])


def _normalise_text_columns(df: pd.DataFrame) -> None:
    for column in TEXT_COLUMNS:
        if column not in df.columns:
            continue
        series = df[column].astype("string").str.strip()
        df[column] = series.replace("", pd.NA)


def _normalize_movie_key(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""

    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    return " ".join(text.split())


def _row_quality_score(df: pd.DataFrame) -> pd.Series:
    score = pd.Series(0, index=df.index, dtype="int64")
    score_columns = (
        "primary_genre",
        "country",
        "keyword",
        "language",
        "budget",
        "revenue",
        "runtime",
        "vote_average",
        "vote_count",
        "popularity",
    )
    for column in score_columns:
        if column in df.columns:
            score = score.add(df[column].notna().astype("int64"), fill_value=0)

    for observed_column in ("budget_observed", "revenue_observed", "runtime_observed"):
        if observed_column in df.columns:
            observed = df[observed_column].fillna(False).astype("boolean").astype("int64")
            score = score.add(observed * 2, fill_value=0)

    return score.astype("int64")


def _build_title_resolution_candidates(row: pd.Series) -> list[str]:
    title_value = row.get("title")
    base_title = "" if pd.isna(title_value) else str(title_value).strip()
    if not base_title:
        return []

    candidates: list[str] = []
    original_title = row.get("original_title")
    if pd.notna(original_title):
        original_text = str(original_title).strip()
        if original_text and _normalize_movie_key(original_text) != _normalize_movie_key(base_title):
            candidates.append(f"{base_title} ({original_text})")

    release_date = row.get("release_date")
    if pd.notna(release_date):
        timestamp = pd.Timestamp(release_date)
        candidates.append(f"{base_title} ({timestamp.strftime('%Y-%m-%d')})")

    language = row.get("original_language")
    if pd.isna(language):
        language = row.get("language")
    if pd.notna(language):
        language_text = str(language).strip()
        if language_text:
            candidates.append(f"{base_title} [{language_text.upper()}]")

    identifier = row.get("source_id")
    if pd.isna(identifier):
        identifier = row.get("id")
    if pd.notna(identifier):
        try:
            candidates.append(f"{base_title} #{int(identifier)}")
        except (TypeError, ValueError):
            candidates.append(f"{base_title} #{identifier}")

    candidates.append(base_title)
    return candidates


def _coerce_cleaned_dataframe_types(df: pd.DataFrame) -> pd.DataFrame:
    """Restore the expected dtypes when reading an exported cleaned dataset."""
    working = df.copy()

    for column in ("id", "source_id", "year"):
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce").astype("Int64")

    for column in NUMERIC_COLUMNS:
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")

    for column in ("budget_observed", "revenue_observed", "runtime_observed"):
        if column in working.columns:
            working[column] = working[column].astype("boolean")

    if "release_date" in working.columns:
        working["release_date"] = pd.to_datetime(working["release_date"], errors="coerce")

    _normalise_text_columns(working)
    return working


def _validate_cleaned_dataset_columns(df: pd.DataFrame, source_path: Path) -> None:
    """Ensure the runtime dataset matches the cleaned-schema contract."""
    missing = [column for column in REQUIRED_CLEANED_COLUMNS if column not in df.columns]
    if missing:
        joined = ", ".join(missing)
        raise KeyError(
            f"Dataset '{source_path.name}' is not a cleaned project dataset. "
            f"Missing columns: {joined}"
        )


def _ensure_observed_numeric_flags(
    df: pd.DataFrame,
    raw_dataset_dir: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Attach observed-value flags for numeric forecasting fields when absent."""
    working = df.copy()
    observed_map = {
        "budget": "budget_observed",
        "revenue": "revenue_observed",
        "runtime": "runtime_observed",
    }
    missing_flags = [observed for observed in observed_map.values() if observed not in working.columns]
    if not missing_flags:
        return working

    if "source_id" in working.columns:
        try:
            raw_dir = resolve_raw_dataset_dir(raw_dataset_dir)
        except FileNotFoundError:
            raw_dir = None

        raw_movies_path = raw_dir / "movies_metadata.csv" if raw_dir else None
        if raw_movies_path and raw_movies_path.exists():
            raw_numeric = pd.read_csv(
                raw_movies_path,
                low_memory=False,
                usecols=["id", "budget", "revenue", "runtime"],
            )
            raw_numeric["source_id"] = pd.to_numeric(raw_numeric["id"], errors="coerce").astype("Int64")
            raw_numeric = raw_numeric.dropna(subset=["source_id"]).drop_duplicates(subset=["source_id"]).copy()

            for value_column, observed_column in observed_map.items():
                numeric = pd.to_numeric(raw_numeric[value_column], errors="coerce")
                raw_numeric[observed_column] = (numeric > 0).astype("boolean")

            observed_lookup = raw_numeric[["source_id", *observed_map.values()]]
            for observed_column in observed_map.values():
                if observed_column in working.columns:
                    observed_lookup = observed_lookup.rename(columns={observed_column: f"{observed_column}_derived"})
            working = working.merge(observed_lookup, on="source_id", how="left")

            for observed_column in observed_map.values():
                derived_column = f"{observed_column}_derived"
                if derived_column in working.columns:
                    if observed_column not in working.columns:
                        working[observed_column] = working[derived_column]
                    else:
                        working[observed_column] = working[observed_column].fillna(working[derived_column])
                    working = working.drop(columns=[derived_column])

    for value_column, observed_column in observed_map.items():
        if observed_column not in working.columns:
            numeric = pd.to_numeric(working.get(value_column), errors="coerce")
            working[observed_column] = (numeric > 0).astype("boolean")
        else:
            working[observed_column] = working[observed_column].astype("boolean")
            if working[observed_column].isna().any():
                numeric = pd.to_numeric(working.get(value_column), errors="coerce")
                fallback = (numeric > 0).astype("boolean")
                working[observed_column] = working[observed_column].fillna(fallback)

    return working


def _normalise_pipe_joined_column(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        return

    df[column] = df[column].apply(
        lambda value: "|".join(_parse_json_like(value)) if pd.notna(value) else pd.NA
    )


def _prepare_textual_columns(df: pd.DataFrame) -> None:
    for target, sources in COALESCED_COLUMN_SOURCES.items():
        _coalesce_column(df, target, sources)

    if "primary_genre" not in df.columns and "genres" in df.columns:
        df["primary_genre"] = df["genres"].apply(_extract_primary_genre)
    elif "primary_genre" in df.columns:
        df["primary_genre"] = df["primary_genre"].apply(_extract_primary_genre)

    for column in PIPE_JOINED_TEXT_COLUMNS:
        _normalise_pipe_joined_column(df, column)

    _normalise_text_columns(df)


def _ensure_title_identity_columns(df: pd.DataFrame) -> None:
    if "title" in df.columns:
        title_series = df["title"].astype("string").str.strip()
    else:
        title_series = pd.Series(pd.NA, index=df.index, dtype="string")

    if "original_title" not in df.columns:
        df["original_title"] = title_series
    else:
        original_title = df["original_title"].astype("string").str.strip()
        df["original_title"] = original_title.fillna(title_series)

    if "language" in df.columns:
        language_series = df["language"].astype("string").str.strip()
    else:
        language_series = pd.Series(pd.NA, index=df.index, dtype="string")

    if "original_language" not in df.columns:
        df["original_language"] = language_series
    else:
        original_language = df["original_language"].astype("string").str.strip()
        df["original_language"] = original_language.fillna(language_series)


def _impute_numeric_column(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        return

    df[column] = pd.to_numeric(df[column], errors="coerce")
    if column in {"budget", "revenue", "runtime"}:
        df.loc[df[column] <= 0, column] = np.nan
    elif column == "vote_count":
        df.loc[df[column] < 0, column] = np.nan
    elif column == "popularity":
        df.loc[df[column] < 0, column] = np.nan

    if column in {"budget", "revenue", "runtime", "popularity"} and "primary_genre" in df.columns and df[column].notna().any():
        genre_medians = df.groupby("primary_genre")[column].transform("median")
        df[column] = df[column].fillna(genre_medians)

    global_median = df[column].median() if df[column].notna().any() else np.nan
    if not pd.isna(global_median):
        df[column] = df[column].fillna(global_median)

    if df[column].isna().any():
        df[column] = df[column].fillna(NUMERIC_FALLBACK_DEFAULTS.get(column, 0.0))


def _date_range_from_series(series: pd.Series) -> Dict[str, Optional[str]]:
    """Build a formatted date range from a raw release-date series."""
    dates = pd.to_datetime(series, errors="coerce")
    if dates.empty:
        return {"start": None, "end": None}

    date_min = dates.min()
    date_max = dates.max()
    return {
        "start": date_min.strftime("%Y-%m-%d") if pd.notna(date_min) else None,
        "end": date_max.strftime("%Y-%m-%d") if pd.notna(date_max) else None,
    }


def _remove_missing_titles(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    missing_title_mask = df["title"].isna() if "title" in df.columns else pd.Series(True, index=df.index)
    removed_rows = int(missing_title_mask.sum())
    return df.loc[~missing_title_mask].copy(), removed_rows


def _remove_duplicate_movies(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    working = df.copy()
    working["_quality_score"] = _row_quality_score(working)

    numeric_sort_columns = ("vote_count", "popularity", "revenue", "runtime", "budget")
    sort_columns = ["_quality_score"]
    ascending = [False]
    for column in numeric_sort_columns:
        sort_column = f"__sort_{column}"
        if column in working.columns:
            working[sort_column] = pd.to_numeric(working[column], errors="coerce").fillna(-1.0)
            sort_columns.append(sort_column)
            ascending.append(False)

    if "release_date" in working.columns:
        sort_columns.append("release_date")
        ascending.append(True)

    working = working.sort_values(sort_columns, ascending=ascending, kind="mergesort", na_position="last").copy()

    source_duplicates_removed = 0
    if "id" in working.columns:
        source_duplicates_removed = int(working.duplicated(subset=["id"]).sum())
        working = working.drop_duplicates(subset=["id"], keep="first").copy()

    title_duplicates_removed = 0
    title_collision_groups = 0
    if "title" in working.columns and "year" in working.columns:
        title_key = working["title"].map(_normalize_movie_key)
        original_title_source = working["original_title"] if "original_title" in working.columns else working["title"]
        original_title_key = original_title_source.map(_normalize_movie_key)

        if "original_language" in working.columns:
            language_source = working["original_language"]
        elif "language" in working.columns:
            language_source = working["language"]
        else:
            language_source = pd.Series(pd.NA, index=working.index, dtype="object")
        language_key = language_source.astype("string").fillna("").str.strip().str.lower()

        working["_title_key"] = title_key
        working["_original_title_key"] = original_title_key
        working["_language_key"] = language_key

        duplicate_signature = ["_title_key", "year", "_original_title_key", "_language_key"]
        title_duplicates_removed = int(working.duplicated(subset=duplicate_signature).sum())
        working = working.drop_duplicates(subset=duplicate_signature, keep="first").copy()

        collision_mask = working.duplicated(subset=["_title_key", "year"], keep=False) & working["_title_key"].ne("")
        if collision_mask.any():
            collision_groups = working.loc[collision_mask, ["_title_key", "year"]].drop_duplicates()
            title_collision_groups = int(len(collision_groups))
            for _, collision in collision_groups.iterrows():
                group_mask = (
                    working["_title_key"].eq(collision["_title_key"])
                    & working["year"].eq(collision["year"])
                )
                group_indices = working.index[group_mask].tolist()
                used_keys: set[str] = set()
                for index in group_indices:
                    row = working.loc[index]
                    chosen_title = str(row["title"]).strip()
                    for candidate in _build_title_resolution_candidates(row):
                        candidate_key = _normalize_movie_key(candidate)
                        if candidate_key and candidate_key not in used_keys:
                            chosen_title = candidate
                            break

                    candidate_counter = 2
                    chosen_key = _normalize_movie_key(chosen_title)
                    while not chosen_key or chosen_key in used_keys:
                        fallback_title = f"{row['title']} #{candidate_counter}"
                        chosen_title = fallback_title
                        chosen_key = _normalize_movie_key(chosen_title)
                        candidate_counter += 1

                    working.at[index, "title"] = chosen_title
                    used_keys.add(chosen_key)

        working = working.drop(columns=["_title_key", "_original_title_key", "_language_key"])

    drop_columns = ["_quality_score", *[column for column in working.columns if column.startswith("__sort_")]]
    working = working.drop(columns=drop_columns, errors="ignore")
    return working, {
        "source_duplicates_removed": source_duplicates_removed,
        "title_duplicates_removed": title_duplicates_removed,
        "title_collisions_disambiguated": title_collision_groups,
        "duplicates_removed": source_duplicates_removed + title_duplicates_removed,
    }


def _clean_release_dates(
    df: pd.DataFrame,
    cutoff_date: pd.Timestamp = DATASET_CUTOFF_DATE,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    if "release_date" not in df.columns:
        raise KeyError("Dataset must include a 'release_date' column.")

    release_dates = pd.to_datetime(df["release_date"], errors="coerce")
    invalid_release_dates = int(release_dates.isna().sum())

    valid_dates = df.loc[release_dates.notna()].copy()
    valid_dates["release_date"] = release_dates.loc[release_dates.notna()]

    after_cutoff_mask = valid_dates["release_date"] > cutoff_date
    after_cutoff_removed = int(after_cutoff_mask.sum())
    valid_dates = valid_dates.loc[~after_cutoff_mask].copy()
    valid_dates["year"] = valid_dates["release_date"].dt.year.astype("Int64")

    return valid_dates, {
        "rows_removed_invalid_release_date": invalid_release_dates,
        "rows_removed_after_dataset_cutoff": after_cutoff_removed,
    }


def _coerce_identifier_column(df: pd.DataFrame) -> None:
    if "id" not in df.columns:
        return
    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")


def _preserve_source_identifier(df: pd.DataFrame) -> pd.DataFrame:
    """Preserve the original movie identifier before assigning cleaned ids."""
    working = df.copy()
    if "id" in working.columns:
        working["source_id"] = working["id"].astype("Int64")
    else:
        working["source_id"] = pd.array([pd.NA] * len(working), dtype="Int64")
    return working


def _reindex_cleaned_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Assign sequential cleaned ids using the current row order."""
    working = df.copy()
    if "id" not in working.columns:
        return working

    existing_ids = pd.to_numeric(working["id"], errors="coerce").astype("Int64")
    expected_ids = pd.Series(range(1, len(working) + 1), index=working.index, dtype="Int64")

    if "source_id" not in working.columns and not existing_ids.equals(expected_ids):
        working["source_id"] = existing_ids

    working["id"] = expected_ids
    return working


def _ensure_source_identifier(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    if "source_id" not in working.columns:
        return _preserve_source_identifier(working)

    working["source_id"] = pd.to_numeric(working["source_id"], errors="coerce").astype("Int64")
    if "id" in working.columns:
        fallback_ids = pd.to_numeric(working["id"], errors="coerce").astype("Int64")
        working["source_id"] = working["source_id"].fillna(fallback_ids)
    return working


def _fill_missing_text_values(df: pd.DataFrame) -> int:
    filled_cells = 0
    for column, default_value in TEXT_FILL_DEFAULTS.items():
        if column not in df.columns:
            df[column] = pd.Series([default_value] * len(df), index=df.index, dtype="string")
            filled_cells += int(len(df))
            continue

        series = df[column].astype("object")
        normalized = series.map(lambda value: value if pd.isna(value) else str(value).strip())
        missing_mask = normalized.isna() | normalized.eq("")
        filled_cells += int(missing_mask.sum())
        df[column] = normalized.mask(missing_mask, default_value).astype("string")

    return filled_cells


def _clean_numeric_columns(df: pd.DataFrame, preserve_observed_flags: bool = False) -> None:
    observed_columns = {
        "budget": "budget_observed",
        "revenue": "revenue_observed",
        "runtime": "runtime_observed",
    }
    for column, observed_column in observed_columns.items():
        if column not in df.columns:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        if preserve_observed_flags and observed_column in df.columns:
            df[observed_column] = df[observed_column].astype("boolean")
        else:
            df[observed_column] = (numeric > 0).astype("boolean")

    for column in NUMERIC_COLUMNS:
        _impute_numeric_column(df, column)

    if "vote_average" in df.columns:
        df["vote_average"] = df["vote_average"].clip(lower=0, upper=10)


def _select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    output_columns = [column for column in OUTPUT_COLUMNS if column in df.columns]
    return df[output_columns].sort_values(["release_date", "title"]).reset_index(drop=True)


def load_raw_kaggle_tables(dataset_dir: Optional[str | Path] = None) -> Dict[str, pd.DataFrame]:
    """Load the raw Kaggle movie dataset tables used by the project."""
    root = resolve_raw_dataset_dir(dataset_dir)
    tables: Dict[str, pd.DataFrame] = {}

    table_files = {
        "movies_metadata": "movies_metadata.csv",
        "keywords": "keywords.csv",
        "credits": "credits.csv",
        "ratings_small": "ratings_small.csv",
        "links_small": "links_small.csv",
    }
    for name, filename in table_files.items():
        path = root / filename
        if path.exists():
            tables[name] = pd.read_csv(path, low_memory=False)

    return tables


def build_project_dataframe_from_raw(dataset_dir: Optional[str | Path] = None) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Build the merged raw dataframe before cleaning."""
    tables = load_raw_kaggle_tables(dataset_dir)
    if "movies_metadata" not in tables:
        raise FileNotFoundError("movies_metadata.csv is required in the raw Kaggle dataset.")

    movies = tables["movies_metadata"].copy()
    merge_metrics = {
        "raw_movies_rows": int(len(movies)),
        "raw_keywords_rows": int(len(tables.get("keywords", pd.DataFrame()))),
        "raw_credits_rows": int(len(tables.get("credits", pd.DataFrame()))),
        "ratings_small_rows": int(len(tables.get("ratings_small", pd.DataFrame()))),
    }

    movies["id"] = pd.to_numeric(movies["id"], errors="coerce")
    invalid_movie_ids = int(movies["id"].isna().sum())
    movies = movies.dropna(subset=["id"]).copy()
    movies["id"] = movies["id"].astype("Int64")
    merge_metrics["invalid_movie_id_rows_removed"] = invalid_movie_ids

    if "keywords" in tables:
        keywords = tables["keywords"].copy()
        keywords["id"] = pd.to_numeric(keywords["id"], errors="coerce").astype("Int64")
        keywords = keywords.dropna(subset=["id"]).drop_duplicates(subset=["id"])
        movies = movies.merge(keywords[["id", "keywords"]], on="id", how="left")

    if "credits" in tables:
        credits = tables["credits"].copy()
        credits["id"] = pd.to_numeric(credits["id"], errors="coerce").astype("Int64")
        credits = credits.dropna(subset=["id"]).drop_duplicates(subset=["id"])
        movies = movies.merge(credits[["id", "cast", "crew"]], on="id", how="left")

    return movies, merge_metrics


def clean_movie_dataset(
    df: pd.DataFrame,
    cutoff_date: pd.Timestamp = DATASET_CUTOFF_DATE,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Clean a raw or partially cleaned IMDB dataset into a project-ready table."""
    working = df.copy()
    _prepare_textual_columns(working)
    _ensure_title_identity_columns(working)
    _coerce_identifier_column(working)

    initial_rows = len(working)
    working, title_removed = _remove_missing_titles(working)
    working, release_date_metrics = _clean_release_dates(working, cutoff_date=cutoff_date)
    _clean_numeric_columns(working)
    working, duplicate_metrics = _remove_duplicate_movies(working)
    text_fields_filled = _fill_missing_text_values(working)
    working = _preserve_source_identifier(working)
    cleaned = _select_output_columns(working)
    cleaned = _reindex_cleaned_ids(cleaned)

    metrics = {
        "rows_loaded": int(initial_rows),
        "rows_after_cleaning": int(len(cleaned)),
        "rows_removed_missing_title": title_removed,
        "duplicates_removed": duplicate_metrics["duplicates_removed"],
        "source_duplicates_removed": duplicate_metrics["source_duplicates_removed"],
        "title_duplicates_removed": duplicate_metrics["title_duplicates_removed"],
        "title_collisions_disambiguated": duplicate_metrics["title_collisions_disambiguated"],
        "text_fields_filled": text_fields_filled,
        **release_date_metrics,
    }
    return cleaned, metrics


def sanitize_cleaned_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Re-validate a cleaned dataset so exports and cached loads stay strict."""
    working = df.copy()
    _coerce_identifier_column(working)
    if "source_id" in working.columns:
        working["source_id"] = pd.to_numeric(working["source_id"], errors="coerce").astype("Int64")
    if "release_date" in working.columns:
        working["release_date"] = pd.to_datetime(working["release_date"], errors="coerce")
    if "year" not in working.columns and "release_date" in working.columns:
        working["year"] = working["release_date"].dt.year.astype("Int64")

    _normalise_text_columns(working)
    _ensure_title_identity_columns(working)
    working, title_removed = _remove_missing_titles(working)
    working, release_date_metrics = _clean_release_dates(working)
    _clean_numeric_columns(working, preserve_observed_flags=True)
    working, duplicate_metrics = _remove_duplicate_movies(working)
    text_fields_filled = _fill_missing_text_values(working)
    working = _ensure_source_identifier(working)
    cleaned = _select_output_columns(working)
    cleaned = _reindex_cleaned_ids(cleaned)

    metrics = {
        "rows_loaded": int(len(df)),
        "rows_after_cleaning": int(len(cleaned)),
        "rows_removed_missing_title": title_removed,
        "duplicates_removed": duplicate_metrics["duplicates_removed"],
        "source_duplicates_removed": duplicate_metrics["source_duplicates_removed"],
        "title_duplicates_removed": duplicate_metrics["title_duplicates_removed"],
        "title_collisions_disambiguated": duplicate_metrics["title_collisions_disambiguated"],
        "text_fields_filled": text_fields_filled,
        **release_date_metrics,
    }
    return cleaned, metrics


def build_data_quality_report(
    source_path: Path,
    original_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    metrics: Dict[str, int],
    cutoff_date: pd.Timestamp = DATASET_CUTOFF_DATE,
) -> Dict[str, object]:
    """Generate a concise data quality report for the cleaned dataset."""
    source_release_dates = original_df["release_date"] if "release_date" in original_df.columns else pd.Series(dtype="object")
    source_date_range = _date_range_from_series(source_release_dates)
    cleaned_date_range = _date_range_from_series(cleaned_df["release_date"]) if "release_date" in cleaned_df.columns else {"start": None, "end": None}

    numeric_summary: Dict[str, Dict[str, float]] = {}
    for column in NUMERIC_COLUMNS:
        if column not in cleaned_df.columns:
            continue
        numeric_summary[column] = {
            "min": round(float(cleaned_df[column].min()), 2),
            "median": round(float(cleaned_df[column].median()), 2),
            "max": round(float(cleaned_df[column].max()), 2),
        }

    return {
        "source_file": source_path.name,
        "source_rows": int(len(original_df)),
        "clean_rows": metrics["rows_after_cleaning"],
        "duplicates_removed": metrics["duplicates_removed"],
        "source_duplicates_removed": metrics.get("source_duplicates_removed", 0),
        "title_duplicates_removed": metrics.get("title_duplicates_removed", 0),
        "title_collisions_disambiguated": metrics.get("title_collisions_disambiguated", 0),
        "text_fields_filled": metrics.get("text_fields_filled", 0),
        "rows_removed_missing_title": metrics["rows_removed_missing_title"],
        "rows_removed_invalid_release_date": metrics["rows_removed_invalid_release_date"],
        "rows_removed_after_dataset_cutoff": metrics["rows_removed_after_dataset_cutoff"],
        "columns_available": list(cleaned_df.columns),
        "missing_values_after_cleaning": cleaned_df.isna().sum().astype(int).to_dict(),
        "genre_count": int(cleaned_df["primary_genre"].nunique()) if "primary_genre" in cleaned_df.columns else 0,
        "source_valid_release_dates": int(pd.to_datetime(source_release_dates, errors="coerce").notna().sum()),
        "dataset_cutoff_date": cutoff_date.strftime("%Y-%m-%d"),
        "date_range": source_date_range,
        "source_date_range": source_date_range,
        "cleaned_date_range": cleaned_date_range,
        "numeric_summary": numeric_summary,
    }


def build_cached_cleaned_report(
    source_path: Path,
    cleaned_df: pd.DataFrame,
    cutoff_date: pd.Timestamp = DATASET_CUTOFF_DATE,
    source_rows: Optional[int] = None,
    metrics: Optional[Dict[str, int]] = None,
) -> Dict[str, object]:
    """Generate a lightweight quality report for an already-cleaned dataset."""
    cleaned_date_range = _date_range_from_series(cleaned_df["release_date"]) if "release_date" in cleaned_df.columns else {"start": None, "end": None}
    metrics = metrics or {}

    numeric_summary: Dict[str, Dict[str, float]] = {}
    for column in NUMERIC_COLUMNS:
        if column not in cleaned_df.columns:
            continue
        numeric_summary[column] = {
            "min": round(float(cleaned_df[column].min()), 2),
            "median": round(float(cleaned_df[column].median()), 2),
            "max": round(float(cleaned_df[column].max()), 2),
        }

    return {
        "source_file": source_path.name,
        "source_rows": int(source_rows if source_rows is not None else len(cleaned_df)),
        "clean_rows": int(len(cleaned_df)),
        "duplicates_removed": int(metrics.get("duplicates_removed", 0)),
        "source_duplicates_removed": int(metrics.get("source_duplicates_removed", 0)),
        "title_duplicates_removed": int(metrics.get("title_duplicates_removed", 0)),
        "title_collisions_disambiguated": int(metrics.get("title_collisions_disambiguated", 0)),
        "text_fields_filled": int(metrics.get("text_fields_filled", 0)),
        "rows_removed_missing_title": int(metrics.get("rows_removed_missing_title", 0)),
        "rows_removed_invalid_release_date": int(metrics.get("rows_removed_invalid_release_date", 0)),
        "rows_removed_after_dataset_cutoff": int(metrics.get("rows_removed_after_dataset_cutoff", 0)),
        "columns_available": list(cleaned_df.columns),
        "missing_values_after_cleaning": cleaned_df.isna().sum().astype(int).to_dict(),
        "genre_count": int(cleaned_df["primary_genre"].nunique()) if "primary_genre" in cleaned_df.columns else 0,
        "source_valid_release_dates": int(pd.to_datetime(cleaned_df.get("release_date", pd.Series(dtype="object")), errors="coerce").notna().sum()),
        "dataset_cutoff_date": cutoff_date.strftime("%Y-%m-%d"),
        "date_range": cleaned_date_range,
        "source_date_range": cleaned_date_range,
        "cleaned_date_range": cleaned_date_range,
        "numeric_summary": numeric_summary,
        "raw_dataset_dir": None,
        "merge_metrics": {},
        "data_origin": "cleaned_cache",
    }


def export_clean_dataset(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Export the cleaned dataset to CSV."""
    destination = Path(output_path)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False, date_format="%Y-%m-%d")
    return destination


def prepare_dataset(
    dataset_path: Optional[str | Path] = None,
    export_path: Optional[str | Path] = None,
    raw_dataset_dir: Optional[str | Path] = None,
    cutoff_date: pd.Timestamp = DATASET_CUTOFF_DATE,
    prefer_cleaned: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Load, clean, and optionally export a project-ready dataset."""
    if prefer_cleaned:
        source_path = resolve_cleaned_dataset_path(dataset_path)
        export_key = None
        if export_path is not None:
            export_candidate = Path(export_path)
            if not export_candidate.is_absolute():
                export_candidate = PROJECT_ROOT / export_candidate
            export_key = str(export_candidate.resolve())
        cache_key = (str(source_path.resolve()), cutoff_date.strftime("%Y-%m-%d"), export_key)

        with _DATASET_CACHE_LOCK:
            cached = _DATASET_CACHE.get(cache_key)
        if cached is not None:
            cached_df, cached_report = cached
            return cached_df, dict(cached_report)

        cleaned_df = _coerce_cleaned_dataframe_types(pd.read_csv(source_path, low_memory=False))
        cleaned_df = _ensure_observed_numeric_flags(cleaned_df, raw_dataset_dir=raw_dataset_dir)
        _validate_cleaned_dataset_columns(cleaned_df, source_path)
        source_rows = int(len(cleaned_df))
        cleaned_df, sanitize_metrics = sanitize_cleaned_dataset(cleaned_df)
        report = build_cached_cleaned_report(
            source_path,
            cleaned_df,
            cutoff_date=cutoff_date,
            source_rows=source_rows,
            metrics=sanitize_metrics,
        )

        if export_path:
            export_clean_dataset(cleaned_df, export_path)

        with _DATASET_CACHE_LOCK:
            _DATASET_CACHE[cache_key] = (cleaned_df, dict(report))
        return cleaned_df, report

    source_path = resolve_dataset_path(dataset_path)

    raw_dir = None
    try:
        raw_dir = resolve_raw_dataset_dir(raw_dataset_dir)
    except FileNotFoundError:
        raw_dir = None

    if raw_dir and source_path.resolve() == (raw_dir / "movies_metadata.csv").resolve():
        original_df, merge_metrics = build_project_dataframe_from_raw(raw_dir)
        report_source = raw_dir / "movies_metadata.csv"
    else:
        original_df = pd.read_csv(source_path, low_memory=False)
        merge_metrics = {}
        report_source = source_path

    cleaned_df, metrics = clean_movie_dataset(original_df, cutoff_date=cutoff_date)
    report = build_data_quality_report(report_source, original_df, cleaned_df, metrics, cutoff_date=cutoff_date)
    report["raw_dataset_dir"] = str(raw_dir) if raw_dir else None
    report["merge_metrics"] = merge_metrics

    if export_path:
        export_clean_dataset(cleaned_df, export_path)

    return cleaned_df, report


def get_prepared_dataset(
    dataset_path: Optional[str | Path] = None,
    cutoff_date: pd.Timestamp = DATASET_CUTOFF_DATE,
) -> pd.DataFrame:
    """Convenience wrapper for modules that only need the cleaned dataframe."""
    cleaned_df, _ = prepare_dataset(dataset_path, cutoff_date=cutoff_date)
    return cleaned_df


if __name__ == "__main__":
    cleaned_df, report = prepare_dataset(export_path="cleaned_imdb_movies.csv", prefer_cleaned=False)
    print("Prepared dataset summary")
    print(f"Source file: {report['source_file']}")
    print(f"Rows after cleaning: {report['clean_rows']}")
    print(f"Genres available: {report['genre_count']}")
    print(f"Rows removed after dataset cutoff ({report['dataset_cutoff_date']}): {report['rows_removed_after_dataset_cutoff']}")
    print(f"Raw date range: {report['source_date_range']['start']} -> {report['source_date_range']['end']}")
    print(f"Cleaned date range: {report['cleaned_date_range']['start']} -> {report['cleaned_date_range']['end']}")
