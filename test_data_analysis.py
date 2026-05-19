import unittest

import pandas as pd

from advanced_analytics import get_comprehensive_analysis
from data_analysis import apply_filters


class SearchFilterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sample = pd.DataFrame(
            {
                "title": [
                    "Blacksmith Scene",
                    "It",
                    "Toy Story",
                    "Holiday Short",
                    "The Ring Thing",
                ],
                "primary_genre": ["Drama", "Horror", "Animation", "Animation", "Comedy"],
                "release_date": ["1893-01-01", "2017-09-05", "1995-11-22", "2011-06-16", "2004-01-01"],
                "year": [1893, 2017, 1995, 2011, 2004],
                "vote_average": [6.0, 7.2, 7.8, 6.5, 5.0],
                "keyword": [
                    "blacksmith|beer|workmen",
                    "clown|horror",
                    "toy story|pixar animation",
                    "toy story|short",
                    "fictional place|marriage proposal|spoof",
                ],
            }
        )

    def test_short_query_requires_whole_word_match(self) -> None:
        filtered = apply_filters(self.sample, {"title_keyword": "it"})

        self.assertListEqual(filtered["title"].tolist(), ["It"])

    def test_phrase_query_returns_all_title_and_keyword_matches(self) -> None:
        filtered = apply_filters(self.sample, {"title_keyword": "toy story"})

        self.assertListEqual(filtered["title"].tolist(), ["Toy Story", "Holiday Short"])

    def test_single_word_query_matches_title_word_not_substring(self) -> None:
        filtered = apply_filters(self.sample, {"title_keyword": "ring"})

        self.assertListEqual(filtered["title"].tolist(), ["The Ring Thing"])

    def test_keyword_search_returns_keyword_matches_when_title_does_not_match(self) -> None:
        filtered = apply_filters(self.sample, {"title_keyword": "clown"})

        self.assertListEqual(filtered["title"].tolist(), ["It"])


class ComprehensiveAnalysisTests(unittest.TestCase):
    def test_includes_high_box_office_profile_when_revenue_data_is_available(self) -> None:
        rows = []
        for index in range(1, 61):
            rows.append(
                {
                    "title": f"Movie {index}",
                    "primary_genre": "Action" if index % 3 == 0 else "Drama" if index % 3 == 1 else "Comedy",
                    "release_date": f"{2010 + (index % 8)}-{((index - 1) % 12) + 1:02d}-01",
                    "year": 2010 + (index % 8),
                    "vote_average": 6.0 + (index % 5) * 0.4,
                    "budget": 2_000_000 + index * 700_000,
                    "revenue": 8_000_000 + index * 2_500_000,
                    "runtime": 88 + (index % 25),
                    "language": "en",
                    "country": "United States of America",
                }
            )

        sample = pd.DataFrame(rows)
        result = get_comprehensive_analysis(sample)

        self.assertIn("high_box_office_profile", result)
        self.assertNotIn("error", result["high_box_office_profile"])
        self.assertIn("top_positive_features", result["high_box_office_profile"])
        self.assertIn("time_series_summary", result)
        self.assertNotIn("error", result["time_series_summary"])
        self.assertIn("peak_movie_count_year", result["time_series_summary"])


if __name__ == "__main__":
    unittest.main()
