import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data_preprocessing import DATASET_CUTOFF_DATE, clean_movie_dataset, clear_dataset_cache, prepare_dataset


class CleanMovieDatasetDateCutoffTests(unittest.TestCase):
    def test_removes_rows_after_dataset_cutoff(self) -> None:
        sample = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "title": ["Before cutoff", "At cutoff", "After cutoff"],
                "release_date": ["2017-07-30", "2017-07-31", "2017-08-01"],
                "genres": ['[{"id": 18, "name": "Drama"}]'] * 3,
                "vote_average": [7.0, 7.5, 8.0],
                "vote_count": [10, 20, 30],
            }
        )

        cleaned, metrics = clean_movie_dataset(sample)

        self.assertEqual(metrics["rows_removed_after_dataset_cutoff"], 1)
        self.assertEqual(len(cleaned), 2)
        self.assertTrue((cleaned["release_date"] <= DATASET_CUTOFF_DATE).all())
        self.assertListEqual(cleaned["title"].tolist(), ["Before cutoff", "At cutoff"])

    def test_deduplicates_on_movie_id(self) -> None:
        sample = pd.DataFrame(
            {
                "id": ["42", "42", "43"],
                "title": ["Movie A", "Movie A alternate row", "Movie B"],
                "release_date": ["2017-07-20", "2017-07-20", "2017-07-21"],
                "genres": ['[{"id": 18, "name": "Drama"}]'] * 3,
                "vote_average": [7.0, 8.5, 6.5],
                "vote_count": [10, 15, 5],
            }
        )

        cleaned, metrics = clean_movie_dataset(sample)

        self.assertEqual(metrics["duplicates_removed"], 1)
        self.assertEqual(len(cleaned), 2)
        self.assertListEqual(cleaned["id"].astype(int).tolist(), [1, 2])
        self.assertListEqual(cleaned["source_id"].astype(int).tolist(), [42, 43])

    def test_prepare_dataset_reindexes_cleaned_csv_and_preserves_old_ids(self) -> None:
        sample = pd.DataFrame(
            {
                "id": [315946, 194079, 426903],
                "title": ["Movie A", "Movie B", "Movie C"],
                "primary_genre": ["Drama", "Comedy", "Action"],
                "budget": [100.0, 200.0, 300.0],
                "revenue": [1000.0, 2000.0, 3000.0],
                "release_date": ["2017-01-01", "2017-01-02", "2017-01-03"],
                "year": [2017, 2017, 2017],
                "runtime": [90.0, 95.0, 100.0],
                "vote_average": [7.0, 6.5, 8.0],
                "vote_count": [10.0, 20.0, 30.0],
                "popularity": [1.0, 2.0, 3.0],
                "language": ["en", "en", "en"],
                "country": ["United States of America"] * 3,
                "keyword": ["hero", "funny", "fight"],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "cleaned_sample.csv"
            export_path = Path(temp_dir) / "reindexed_sample.csv"
            sample.to_csv(dataset_path, index=False)
            clear_dataset_cache()

            cleaned, report = prepare_dataset(
                dataset_path=dataset_path,
                export_path=export_path,
                prefer_cleaned=True,
            )

            self.assertEqual(report["clean_rows"], 3)
            self.assertListEqual(cleaned["id"].astype(int).tolist(), [1, 2, 3])
            self.assertListEqual(cleaned["source_id"].astype(int).tolist(), [315946, 194079, 426903])

            exported = pd.read_csv(export_path)
            self.assertListEqual(exported["id"].astype(int).tolist(), [1, 2, 3])
            self.assertListEqual(exported["source_id"].astype(int).tolist(), [315946, 194079, 426903])

    def test_disambiguates_same_title_year_movies_with_different_languages(self) -> None:
        sample = pd.DataFrame(
            {
                "id": [101, 102],
                "title": ["Adrift", "Adrift"],
                "original_title": ["À Deriva", "Choi Voi"],
                "original_language": ["pt", "vi"],
                "release_date": ["2009-07-31", "2009-11-13"],
                "genres": ['[{"id": 18, "name": "Drama"}]'] * 2,
                "vote_average": [5.9, 6.2],
                "vote_count": [34, 3],
                "budget": [0, 0],
                "revenue": [0, 0],
                "runtime": [97, 110],
                "popularity": [2.3, 0.15],
                "production_countries": ['[{"name": "Brazil"}]', '[{"name": "Vietnam"}]'],
                "keywords": ['[{"name": "family"}]', '[{"name": "relationship"}]'],
            }
        )

        cleaned, metrics = clean_movie_dataset(sample)

        self.assertEqual(len(cleaned), 2)
        self.assertEqual(metrics["duplicates_removed"], 0)
        self.assertEqual(metrics["title_collisions_disambiguated"], 1)
        self.assertEqual(cleaned["title"].nunique(), 2)
        self.assertFalse(cleaned["title"].duplicated().any())

    def test_fills_missing_text_fields_and_deduplicates_same_movie_signature(self) -> None:
        sample = pd.DataFrame(
            {
                "id": [42, 43],
                "title": ["Movie A", "Movie A"],
                "original_title": ["Movie A", "Movie A"],
                "original_language": ["en", "en"],
                "release_date": ["2017-07-20", "2017-07-21"],
                "genres": [pd.NA, pd.NA],
                "vote_average": [7.0, 6.5],
                "vote_count": [10, 5],
                "budget": [0, 0],
                "revenue": [0, 0],
                "runtime": [90, 88],
                "popularity": [1.2, 0.8],
                "production_countries": [pd.NA, pd.NA],
                "keywords": [pd.NA, pd.NA],
                "language": [pd.NA, pd.NA],
            }
        )

        cleaned, metrics = clean_movie_dataset(sample)

        self.assertEqual(len(cleaned), 1)
        self.assertEqual(metrics["duplicates_removed"], 1)
        self.assertEqual(metrics["text_fields_filled"], 3)
        self.assertFalse(cleaned.isna().any().any())
        self.assertEqual(cleaned.loc[0, "primary_genre"], "Unknown")
        self.assertEqual(cleaned.loc[0, "country"], "Unknown")
        self.assertEqual(cleaned.loc[0, "keyword"], "Unknown")
        self.assertEqual(cleaned.loc[0, "language"], "en")
        self.assertNotIn("overview", cleaned.columns)


if __name__ == "__main__":
    unittest.main()
