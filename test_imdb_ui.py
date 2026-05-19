import unittest

from imdb_ui import IMDbExplorerApp


class GlobalChartBehaviorTests(unittest.TestCase):
    def test_multi_model_forecast_chart_is_treated_as_global(self) -> None:
        self.assertIn("forecast_model_backtest_comparison", IMDbExplorerApp.GLOBAL_CHART_KEYS)
        self.assertIn("forecast_backtest_yearly_comparison", IMDbExplorerApp.GLOBAL_CHART_KEYS)


if __name__ == "__main__":
    unittest.main()
