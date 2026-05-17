import unittest

import pandas as pd

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

    def test_phrase_query_prefers_exact_title_only(self) -> None:
        filtered = apply_filters(self.sample, {"title_keyword": "toy story"})

        self.assertListEqual(filtered["title"].tolist(), ["Toy Story"])

    def test_single_word_query_matches_title_word_not_substring(self) -> None:
        filtered = apply_filters(self.sample, {"title_keyword": "ring"})

        self.assertListEqual(filtered["title"].tolist(), ["The Ring Thing"])

    def test_keyword_search_falls_back_when_no_title_matches(self) -> None:
        filtered = apply_filters(self.sample, {"title_keyword": "clown"})

        self.assertListEqual(filtered["title"].tolist(), ["It"])


if __name__ == "__main__":
    unittest.main()
