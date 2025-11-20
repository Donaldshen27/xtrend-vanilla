"""Tests for bloomberg_viz.regimes_tab module."""
import pytest
import pandas as pd
from scripts.bloomberg_viz.regimes_tab import parse_cache_filename


class TestParseCacheFilename:
    """Test cache filename parsing with various symbol formats."""

    def test_parse_simple_symbol(self):
        """Test parsing filename with simple symbol (no underscores)."""
        filename = "BR_20230301_20231229_211_lb21_th0.85_min5_max63.pkl"
        result = parse_cache_filename(filename)

        assert result["symbol"] == "BR"
        assert result["start"] == pd.Timestamp("2023-03-01")
        assert result["end"] == pd.Timestamp("2023-12-29")
        assert result["series_len"] == 211
        assert result["lookback"] == 21
        assert result["threshold"] == 0.85
        assert result["min_length"] == 5
        assert result["max_length"] == 63

    def test_parse_symbol_with_underscore(self):
        """Test parsing filename with symbol containing underscore."""
        filename = "LF98OAS_Index_19940228_20231229_5926_lb21_th0.85_min5_max63.pkl"
        result = parse_cache_filename(filename)

        assert result["symbol"] == "LF98OAS_Index"
        assert result["start"] == pd.Timestamp("1994-02-28")
        assert result["end"] == pd.Timestamp("2023-12-29")
        assert result["series_len"] == 5926
        assert result["lookback"] == 21
        assert result["threshold"] == 0.85
        assert result["min_length"] == 5
        assert result["max_length"] == 63

    def test_parse_multiple_underscores_in_symbol(self):
        """Test parsing filename with multiple underscores in symbol."""
        filename = "SOME_LONG_SYMBOL_NAME_20200101_20201231_365_lb21_th0.9_min3_max42.pkl"
        result = parse_cache_filename(filename)

        assert result["symbol"] == "SOME_LONG_SYMBOL_NAME"
        assert result["start"] == pd.Timestamp("2020-01-01")
        assert result["end"] == pd.Timestamp("2020-12-31")
        assert result["series_len"] == 365

    def test_parse_all_working_cache_files(self):
        """Test parsing all known working cache files."""
        filenames = [
            "BR_20230301_20231229_211_lb21_th0.85_min5_max63.pkl",
            "DA_19960112_20231229_6836_lb21_th0.85_min5_max63.pkl",
            "ER_20170711_20231229_1633_lb21_th0.85_min5_max63.pkl",
            "GI_20020703_20231229_5474_lb21_th0.85_min5_max63.pkl",
            "JO_19941116_19980302_732_lb21_th0.85_min5_max63.pkl",
        ]

        for filename in filenames:
            result = parse_cache_filename(filename)
            assert "symbol" in result
            assert "start" in result
            assert "end" in result
            assert isinstance(result["start"], pd.Timestamp)
            assert isinstance(result["end"], pd.Timestamp)
