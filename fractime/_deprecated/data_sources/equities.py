"""
Equity data sources (stocks, ETFs, indices).
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
import polars as pl
import pandas as pd

from .base import DataSource, DataSourceConfig, TimeSeriesData


class YFinanceSource(DataSource):
    """Yahoo Finance data source via yfinance library."""

    @property
    def source_name(self) -> str:
        return "yahoo_finance"

    @property
    def supported_asset_types(self) -> List[str]:
        return ['equity', 'etf', 'index', 'mutual_fund']

    def get_data(
        self,
        symbol: str,
        start_date: str,
        end_date: Optional[str] = None,
        interval: str = '1d',
        **kwargs
    ) -> TimeSeriesData:
        """Fetch data from Yahoo Finance."""
        import yfinance as yf

        self._enforce_rate_limit()

        def _fetch():
            if end_date is None:
                end = datetime.now().strftime('%Y-%m-%d')
            else:
                end = end_date

            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end, interval=interval)

            if data.empty:
                raise ValueError(f"No data returned for {symbol}")

            # Reset index to make Date a column
            data = data.reset_index()

            # Convert to Polars
            df = pl.from_pandas(data)

            # Standardize column names
            df = df.rename({col: col for col in df.columns})

            return TimeSeriesData(
                symbol=symbol,
                source=self.source_name,
                data=df,
                metadata={
                    'interval': interval,
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
        """
        Search for symbols (limited - yfinance doesn't have great search).

        Note: This is a basic implementation. For production, consider
        integrating with a symbol search API.
        """
        # yfinance doesn't have a native search API
        # This is a placeholder - in production, use a dedicated search service
        return [{
            'symbol': query.upper(),
            'name': f'{query.upper()} (Search via yfinance)',
            'type': 'equity',
            'source': self.source_name
        }]

    def get_metadata(self, symbol: str) -> Dict[str, Any]:
        """Get symbol metadata."""
        import yfinance as yf

        self._enforce_rate_limit()

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return {
                'symbol': symbol,
                'name': info.get('longName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'marketCap': info.get('marketCap', 0),
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange', ''),
                'source': self.source_name
            }
        except Exception as e:
            return {
                'symbol': symbol,
                'error': str(e),
                'source': self.source_name
            }


class AlphaVantageSource(DataSource):
    """Alpha Vantage data source."""

    @property
    def source_name(self) -> str:
        return "alpha_vantage"

    @property
    def supported_asset_types(self) -> List[str]:
        return ['equity', 'etf', 'forex', 'crypto', 'commodity']

    def get_data(
        self,
        symbol: str,
        start_date: str,
        end_date: Optional[str] = None,
        interval: str = '1d',
        **kwargs
    ) -> TimeSeriesData:
        """Fetch data from Alpha Vantage."""
        if not self.config.api_key:
            raise ValueError("Alpha Vantage requires an API key")

        from alpha_vantage.timeseries import TimeSeries

        self._enforce_rate_limit()

        def _fetch():
            ts = TimeSeries(key=self.config.api_key, output_format='pandas')

            # Alpha Vantage has different endpoints for different intervals
            if interval == '1d':
                data, meta = ts.get_daily(symbol=symbol, outputsize='full')
            elif interval == '1wk':
                data, meta = ts.get_weekly(symbol=symbol)
            elif interval == '1mo':
                data, meta = ts.get_monthly(symbol=symbol)
            else:
                raise ValueError(f"Unsupported interval for Alpha Vantage: {interval}")

            # Reset index to make date a column
            data = data.reset_index()
            data = data.rename(columns={'date': 'Date'})

            # Rename OHLCV columns to standard format
            column_mapping = {
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close',
                '5. volume': 'Volume'
            }
            data = data.rename(columns=column_mapping)

            # Filter by date range
            data['Date'] = pd.to_datetime(data['Date'])
            data = data[data['Date'] >= start_date]
            if end_date:
                data = data[data['Date'] <= end_date]

            # Convert to Polars
            df = pl.from_pandas(data)

            return TimeSeriesData(
                symbol=symbol,
                source=self.source_name,
                data=df,
                metadata={
                    'interval': interval,
                    'meta': meta,
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
        """Search symbols via Alpha Vantage."""
        if not self.config.api_key:
            raise ValueError("Alpha Vantage requires an API key")

        from alpha_vantage.timeseries import TimeSeries

        self._enforce_rate_limit()

        try:
            ts = TimeSeries(key=self.config.api_key)
            data, _ = ts.get_symbol_search(query)

            results = []
            for idx, row in data.head(limit).iterrows():
                results.append({
                    'symbol': row.get('1. symbol', ''),
                    'name': row.get('2. name', ''),
                    'type': row.get('3. type', ''),
                    'region': row.get('4. region', ''),
                    'currency': row.get('8. currency', ''),
                    'source': self.source_name
                })

            return results
        except Exception as e:
            print(f"Alpha Vantage search error: {e}")
            return []

    def get_metadata(self, symbol: str) -> Dict[str, Any]:
        """Get symbol metadata (limited in Alpha Vantage)."""
        return {
            'symbol': symbol,
            'source': self.source_name,
            'note': 'Alpha Vantage has limited metadata API'
        }


class TwelveDataSource(DataSource):
    """Twelve Data API source."""

    @property
    def source_name(self) -> str:
        return "twelve_data"

    @property
    def supported_asset_types(self) -> List[str]:
        return ['equity', 'etf', 'index', 'forex', 'crypto']

    def get_data(
        self,
        symbol: str,
        start_date: str,
        end_date: Optional[str] = None,
        interval: str = '1day',
        **kwargs
    ) -> TimeSeriesData:
        """Fetch data from Twelve Data."""
        if not self.config.api_key:
            raise ValueError("Twelve Data requires an API key")

        from twelvedata import TDClient

        self._enforce_rate_limit()

        def _fetch():
            td = TDClient(apikey=self.config.api_key)

            # Build parameters
            params = {
                'symbol': symbol,
                'interval': interval,
                'start_date': start_date,
                'outputsize': 5000
            }
            if end_date:
                params['end_date'] = end_date

            ts = td.time_series(**params)
            df = ts.as_pandas()

            if df.empty:
                raise ValueError(f"No data returned for {symbol}")

            # Reset index and standardize columns
            df = df.reset_index()
            df = df.rename(columns={
                'datetime': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })

            # Convert to Polars
            pl_df = pl.from_pandas(df)

            return TimeSeriesData(
                symbol=symbol,
                source=self.source_name,
                data=pl_df,
                metadata={
                    'interval': interval,
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
        """Search symbols via Twelve Data."""
        if not self.config.api_key:
            raise ValueError("Twelve Data requires an API key")

        from twelvedata import TDClient

        self._enforce_rate_limit()

        try:
            td = TDClient(apikey=self.config.api_key)
            result = td.symbol_search(symbol=query).as_json()

            results = []
            for item in result.get('data', [])[:limit]:
                results.append({
                    'symbol': item.get('symbol', ''),
                    'name': item.get('instrument_name', ''),
                    'type': item.get('instrument_type', ''),
                    'exchange': item.get('exchange', ''),
                    'currency': item.get('currency', ''),
                    'country': item.get('country', ''),
                    'source': self.source_name
                })

            return results
        except Exception as e:
            print(f"Twelve Data search error: {e}")
            return []

    def get_metadata(self, symbol: str) -> Dict[str, Any]:
        """Get symbol metadata."""
        # Twelve Data has limited metadata in free tier
        return {
            'symbol': symbol,
            'source': self.source_name
        }
