"""
Economic indicator data sources (FRED, World Bank, OECD, ECB).
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
import polars as pl
import pandas as pd

from .base import DataSource, DataSourceConfig, TimeSeriesData


class FREDSource(DataSource):
    """Federal Reserve Economic Data (FRED) source."""

    @property
    def source_name(self) -> str:
        return "fred"

    @property
    def supported_asset_types(self) -> List[str]:
        return ['economic', 'bond']

    def get_data(
        self,
        symbol: str,
        start_date: str,
        end_date: Optional[str] = None,
        interval: str = '1d',
        **kwargs
    ) -> TimeSeriesData:
        """Fetch economic data from FRED."""
        if not self.config.api_key:
            raise ValueError("FRED requires an API key")

        from fredapi import Fred

        self._enforce_rate_limit()

        def _fetch():
            fred = Fred(api_key=self.config.api_key)

            # Get series data
            series = fred.get_series(
                series_id=symbol,
                observation_start=start_date,
                observation_end=end_date
            )

            if series.empty:
                raise ValueError(f"No data returned for {symbol}")

            # Convert to DataFrame
            df = series.to_frame(name='Close')
            df = df.reset_index()
            df = df.rename(columns={'index': 'Date'})

            # FRED only provides values, not OHLCV
            df['Open'] = df['Close']
            df['High'] = df['Close']
            df['Low'] = df['Close']
            df['Volume'] = 0.0

            # Convert to Polars
            pl_df = pl.from_pandas(df)

            return TimeSeriesData(
                symbol=symbol,
                source=self.source_name,
                data=pl_df,
                metadata={
                    'series_id': symbol,
                    'note': 'FRED data provides single values, not OHLCV',
                    'fetched_at': datetime.now().isoformat()
                }
            )

        return self._retry_with_backoff(_fetch)

    def search_symbols(
        self,
        query: str,
        asset_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for FRED series."""
        if not self.config.api_key:
            raise ValueError("FRED requires an API key")

        from fredapi import Fred

        self._enforce_rate_limit()

        try:
            fred = Fred(api_key=self.config.api_key)
            search_results = fred.search(query, limit=limit)

            results = []
            for idx, row in search_results.iterrows():
                results.append({
                    'symbol': row.get('id', idx),
                    'name': row.get('title', ''),
                    'type': 'economic',
                    'frequency': row.get('frequency_short', ''),
                    'units': row.get('units_short', ''),
                    'seasonal_adjustment': row.get('seasonal_adjustment_short', ''),
                    'popularity': row.get('popularity', 0),
                    'source': self.source_name
                })

            return results
        except Exception as e:
            print(f"FRED search error: {e}")
            return []

    def get_metadata(self, symbol: str) -> Dict[str, Any]:
        """Get metadata for a FRED series."""
        if not self.config.api_key:
            raise ValueError("FRED requires an API key")

        from fredapi import Fred

        self._enforce_rate_limit()

        try:
            fred = Fred(api_key=self.config.api_key)
            info = fred.get_series_info(symbol)

            return {
                'symbol': symbol,
                'title': info.get('title', ''),
                'observation_start': str(info.get('observation_start', '')),
                'observation_end': str(info.get('observation_end', '')),
                'frequency': info.get('frequency', ''),
                'units': info.get('units', ''),
                'seasonal_adjustment': info.get('seasonal_adjustment', ''),
                'notes': info.get('notes', '')[:500],
                'source': self.source_name
            }
        except Exception as e:
            return {
                'symbol': symbol,
                'error': str(e),
                'source': self.source_name
            }


class WorldBankSource(DataSource):
    """World Bank data source."""

    @property
    def source_name(self) -> str:
        return "world_bank"

    @property
    def supported_asset_types(self) -> List[str]:
        return ['economic']

    def get_data(
        self,
        symbol: str,
        start_date: str,
        end_date: Optional[str] = None,
        interval: str = 'annual',
        **kwargs
    ) -> TimeSeriesData:
        """
        Fetch economic data from World Bank.

        Args:
            symbol: Indicator code (e.g., 'NY.GDP.MKTP.CD' for GDP)
            start_date: Start year (e.g., '2010')
            end_date: End year (e.g., '2023')
            interval: 'annual' (World Bank data is annual)
            **kwargs: Should include 'country' parameter (e.g., 'USA', 'CHN')
        """
        import wbgapi as wb

        self._enforce_rate_limit()

        def _fetch():
            country = kwargs.get('country', 'USA')

            # Parse years from dates
            start_year = int(start_date.split('-')[0])
            if end_date:
                end_year = int(end_date.split('-')[0])
            else:
                end_year = datetime.now().year

            # Fetch data
            df = wb.data.DataFrame(
                symbol,
                country,
                range(start_year, end_year + 1)
            )

            if df.empty:
                raise ValueError(f"No data returned for {symbol} in {country}")

            # Reset index and rename columns
            df = df.reset_index()
            df = df.rename(columns={'time': 'Date', symbol: 'Close'})

            # Convert year to date
            df['Date'] = pd.to_datetime(df['Date'], format='%Y')

            # Add OHLCV columns (World Bank only has single values)
            df['Open'] = df['Close']
            df['High'] = df['Close']
            df['Low'] = df['Close']
            df['Volume'] = 0.0

            # Convert to Polars
            pl_df = pl.from_pandas(df)

            return TimeSeriesData(
                symbol=symbol,
                source=self.source_name,
                data=pl_df,
                metadata={
                    'indicator': symbol,
                    'country': country,
                    'frequency': 'annual',
                    'note': 'World Bank data is annual',
                    'fetched_at': datetime.now().isoformat()
                }
            )

        return self._retry_with_backoff(_fetch)

    def search_symbols(
        self,
        query: str,
        asset_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for World Bank indicators."""
        import wbgapi as wb

        self._enforce_rate_limit()

        try:
            # Search indicators
            indicators = wb.series.info(q=query)

            results = []
            count = 0
            for idx, row in indicators.iterrows():
                if count >= limit:
                    break

                results.append({
                    'symbol': idx,
                    'name': row.get('value', ''),
                    'type': 'economic',
                    'note': 'Requires country parameter',
                    'source': self.source_name
                })
                count += 1

            return results
        except Exception as e:
            print(f"World Bank search error: {e}")
            return []

    def get_metadata(self, symbol: str) -> Dict[str, Any]:
        """Get metadata for a World Bank indicator."""
        import wbgapi as wb

        self._enforce_rate_limit()

        try:
            info = wb.series.info(symbol)

            if not info.empty:
                row = info.iloc[0]
                return {
                    'symbol': symbol,
                    'name': row.get('value', ''),
                    'note': 'Annual data, requires country parameter',
                    'source': self.source_name
                }
            else:
                return {
                    'symbol': symbol,
                    'source': self.source_name
                }
        except Exception as e:
            return {
                'symbol': symbol,
                'error': str(e),
                'source': self.source_name
            }


class OECDSource(DataSource):
    """OECD data source."""

    @property
    def source_name(self) -> str:
        return "oecd"

    @property
    def supported_asset_types(self) -> List[str]:
        return ['economic']

    def get_data(
        self,
        symbol: str,
        start_date: str,
        end_date: Optional[str] = None,
        interval: str = 'quarterly',
        **kwargs
    ) -> TimeSeriesData:
        """
        Fetch economic data from OECD.

        Note: This is a simplified implementation. OECD API uses SDMX format
        which is complex. For production, consider using pandasdmx library.
        """
        import requests

        self._enforce_rate_limit()

        def _fetch():
            # This is a basic CSV download approach
            # For more advanced usage, use pandasdmx library

            # Example: Download quarterly GDP data
            url = f"https://data-api.oecd.org/sdmx-json/data/{symbol}"

            try:
                response = requests.get(url, timeout=self.config.timeout)
                response.raise_for_status()

                # This would need proper SDMX parsing
                # Placeholder for now
                raise NotImplementedError(
                    "OECD data source requires pandasdmx library. "
                    "Use 'import pandasdmx as sdmx' for full implementation."
                )

            except requests.RequestException as e:
                raise ConnectionError(f"OECD API error: {e}")

        return self._retry_with_backoff(_fetch)

    def search_symbols(
        self,
        query: str,
        asset_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for OECD indicators."""
        # OECD search requires SDMX knowledge
        return [{
            'symbol': query.upper(),
            'name': f'OECD: {query}',
            'type': 'economic',
            'note': 'Use OECD Data Explorer to find indicator codes',
            'source': self.source_name
        }]

    def get_metadata(self, symbol: str) -> Dict[str, Any]:
        """Get metadata for an OECD indicator."""
        return {
            'symbol': symbol,
            'note': 'OECD data requires SDMX parsing. Visit data.oecd.org',
            'source': self.source_name
        }


class ECBSource(DataSource):
    """European Central Bank data source."""

    @property
    def source_name(self) -> str:
        return "ecb"

    @property
    def supported_asset_types(self) -> List[str]:
        return ['economic', 'forex', 'bond']

    def get_data(
        self,
        symbol: str,
        start_date: str,
        end_date: Optional[str] = None,
        interval: str = '1d',
        **kwargs
    ) -> TimeSeriesData:
        """
        Fetch data from European Central Bank.

        Uses ecbdata library for easier access to ECB data.
        """
        try:
            from ecbdata import ecbdata
        except ImportError:
            raise ImportError("ecbdata library required. Install: pip install ecbdata")

        self._enforce_rate_limit()

        def _fetch():
            # Get ECB data
            df = ecbdata.get_series(symbol, start=start_date, end=end_date)

            if df.empty:
                raise ValueError(f"No data returned for {symbol}")

            # Reset index
            df = df.reset_index()
            df = df.rename(columns={df.columns[0]: 'Date', df.columns[1]: 'Close'})

            # Add OHLCV columns
            df['Open'] = df['Close']
            df['High'] = df['Close']
            df['Low'] = df['Close']
            df['Volume'] = 0.0

            # Convert to Polars
            pl_df = pl.from_pandas(df)

            return TimeSeriesData(
                symbol=symbol,
                source=self.source_name,
                data=pl_df,
                metadata={
                    'series_key': symbol,
                    'note': 'ECB data provides single values',
                    'fetched_at': datetime.now().isoformat()
                }
            )

        return self._retry_with_backoff(_fetch)

    def search_symbols(
        self,
        query: str,
        asset_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for ECB series."""
        # ECB search is complex, requires knowledge of series keys
        return [{
            'symbol': query.upper(),
            'name': f'ECB: {query}',
            'type': 'economic',
            'note': 'Use ECB Data Portal to find series keys',
            'source': self.source_name
        }]

    def get_metadata(self, symbol: str) -> Dict[str, Any]:
        """Get metadata for an ECB series."""
        return {
            'symbol': symbol,
            'note': 'Visit sdw.ecb.europa.eu for series information',
            'source': self.source_name
        }
