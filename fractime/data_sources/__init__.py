"""
Data Sources Module

Unified interface for accessing time series data from multiple free sources
across equities, crypto, forex, bonds, commodities, and economic indicators.
"""

from .base import DataSource, DataSourceConfig, TimeSeriesData
from .registry import DataSourceRegistry, get_data_source, list_sources, get_data_with_fallback

__all__ = [
    'DataSource',
    'DataSourceConfig',
    'TimeSeriesData',
    'DataSourceRegistry',
    'get_data_source',
    'list_sources',
    'get_data_with_fallback',
]
