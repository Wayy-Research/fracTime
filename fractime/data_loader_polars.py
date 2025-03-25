"""
Time Series Data Loader Module for FracTime using Polars

This module provides utilities for loading time series data from various sources
including financial databases, UCI repository, and more, using Polars for efficient data processing.
"""

import os
import polars as pl
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Union, List, Dict, Tuple, Any
import urllib.request
import zipfile
import io
import requests
from pathlib import Path

# Optional imports
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import fredapi
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False

try:
    import quandl
    QUANDL_AVAILABLE = True
except ImportError:
    QUANDL_AVAILABLE = False


class TimeSeriesDataLoader:
    """
    A class for loading time series data from various sources using Polars.
    
    Attributes:
        data_dir (Path): Directory to store downloaded data
        cache_dir (Path): Directory to cache processed data
    """
    
    def __init__(self, data_dir: Optional[str] = None, cache_dir: Optional[str] = None):
        """
        Initialize the TimeSeriesDataLoader.
        
        Args:
            data_dir: Directory to store downloaded data
            cache_dir: Directory to cache processed data
        """
        self.data_dir = Path(data_dir or os.path.join(os.path.expanduser("~"), ".fractime", "data"))
        self.cache_dir = Path(cache_dir or os.path.join(os.path.expanduser("~"), ".fractime", "cache"))
        
        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_yahoo_data(self, 
                       symbol: str, 
                       start_date: Union[str, datetime], 
                       end_date: Optional[Union[str, datetime]] = None,
                       interval: str = "1d",
                       force_download: bool = False) -> pl.DataFrame:
        """
        Load financial data from Yahoo Finance.
        
        Args:
            symbol: Stock/index symbol (e.g., 'AAPL', '^GSPC')
            start_date: Start date for data retrieval
            end_date: End date for data retrieval (defaults to today)
            interval: Data frequency ('1d', '1wk', '1mo')
            force_download: If True, download even if cached data exists
            
        Returns:
            Polars DataFrame with OHLCV data
        """
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance package is required but not installed. "
                              "Install it with 'pip install yfinance'")
        
        # Handle date inputs
        if isinstance(start_date, str):
            start_date = pl.from_pandas(pd.to_datetime(start_date))
        
        if end_date is None:
            end_date = datetime.now()
        elif isinstance(end_date, str):
            end_date = pl.from_pandas(pd.to_datetime(end_date))
            
        # Cache file path
        start_date_str = start_date.strftime('%Y%m%d') if isinstance(start_date, datetime) else start_date
        end_date_str = end_date.strftime('%Y%m%d') if isinstance(end_date, datetime) else end_date
        cache_file = self.cache_dir / f"yahoo_{symbol}_{start_date_str}_{end_date_str}_{interval}.parquet"
        
        # If cache exists and we're not forcing a download, load from cache
        if cache_file.exists() and not force_download:
            try:
                return pl.read_parquet(cache_file)
            except Exception as e:
                print(f"Error loading from cache: {e}. Downloading fresh data.")
        
        # Download data
        df = yf.download(symbol, start=start_date, end=end_date, interval=interval)
        
        if df.empty:
            raise ValueError(f"No data found for symbol {symbol}")
            
        # Convert pandas to polars DataFrame
        df = df.reset_index()
        df_pl = pl.from_pandas(df)
        
        # Save to cache
        df_pl.write_parquet(cache_file)
        
        return df_pl

    def get_fred_data(self, 
                     series_id: Union[str, List[str]], 
                     start_date: Union[str, datetime], 
                     end_date: Optional[Union[str, datetime]] = None,
                     api_key: Optional[str] = None,
                     force_download: bool = False) -> pl.DataFrame:
        """
        Load economic data from Federal Reserve Economic Database (FRED).
        
        Args:
            series_id: FRED series ID or list of IDs
            start_date: Start date for data retrieval
            end_date: End date for data retrieval (defaults to today)
            api_key: FRED API key (if not provided, will look for FRED_API_KEY env var)
            force_download: If True, download even if cached data exists
            
        Returns:
            Polars DataFrame with economic data
        """
        if not FRED_AVAILABLE:
            raise ImportError("fredapi package is required but not installed. "
                              "Install it with 'pip install fredapi'")
        
        # Get API key from env var if not provided
        api_key = api_key or os.environ.get('FRED_API_KEY')
        if not api_key:
            raise ValueError("FRED API key must be provided or set as FRED_API_KEY environment variable")
            
        # Handle date inputs
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        
        if end_date is None:
            end_date = datetime.now()
        elif isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
            
        # Convert to list if single series_id
        if isinstance(series_id, str):
            series_id = [series_id]
            
        # Generate cache file name
        series_str = "_".join(series_id)
        cache_file = self.cache_dir / f"fred_{series_str}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.parquet"
        
        # If cache exists and we're not forcing a download, load from cache
        if cache_file.exists() and not force_download:
            try:
                return pl.read_parquet(cache_file)
            except Exception as e:
                print(f"Error loading from cache: {e}. Downloading fresh data.")
        
        # Initialize FRED API
        fred = fredapi.Fred(api_key=api_key)
        
        # Download each series
        data_dict = {}
        for sid in series_id:
            data = fred.get_series(sid, observation_start=start_date, observation_end=end_date)
            data_dict[sid] = data
            
        # Combine into DataFrame
        df = pd.DataFrame(data_dict)
        df.index.name = 'Date'
        df = df.reset_index()
        
        # Convert to polars
        df_pl = pl.from_pandas(df)
        
        # Save to cache
        df_pl.write_parquet(cache_file)
        
        return df_pl

    def get_quandl_data(self, 
                       dataset_code: str,
                       start_date: Union[str, datetime], 
                       end_date: Optional[Union[str, datetime]] = None,
                       api_key: Optional[str] = None,
                       force_download: bool = False) -> pl.DataFrame:
        """
        Load data from Quandl.
        
        Args:
            dataset_code: Quandl dataset code (e.g., 'WIKI/AAPL')
            start_date: Start date for data retrieval
            end_date: End date for data retrieval (defaults to today)
            api_key: Quandl API key (if not provided, will look for QUANDL_API_KEY env var)
            force_download: If True, download even if cached data exists
            
        Returns:
            Polars DataFrame with requested data
        """
        if not QUANDL_AVAILABLE:
            raise ImportError("quandl package is required but not installed. "
                              "Install it with 'pip install quandl'")
        
        # Get API key from env var if not provided
        api_key = api_key or os.environ.get('QUANDL_API_KEY')
        if api_key:
            quandl.ApiConfig.api_key = api_key
            
        # Handle date inputs
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        
        if end_date is None:
            end_date = datetime.now()
        elif isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
            
        # Cache file path
        cache_file = self.cache_dir / f"quandl_{dataset_code.replace('/', '_')}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.parquet"
        
        # If cache exists and we're not forcing a download, load from cache
        if cache_file.exists() and not force_download:
            try:
                return pl.read_parquet(cache_file)
            except Exception as e:
                print(f"Error loading from cache: {e}. Downloading fresh data.")
        
        # Download data
        df = quandl.get(dataset_code, start_date=start_date, end_date=end_date)
        
        if df.empty:
            raise ValueError(f"No data found for dataset {dataset_code}")
            
        # Convert to polars
        df = df.reset_index()
        df_pl = pl.from_pandas(df)
        
        # Save to cache
        df_pl.write_parquet(cache_file)
        
        return df_pl

    def load_uci_air_quality(self, force_download: bool = False) -> pl.DataFrame:
        """
        Load the UCI Air Quality dataset.
        
        Args:
            force_download: If True, download even if the file exists locally
            
        Returns:
            Polars DataFrame with air quality time series data
        """
        # File paths
        data_file = self.data_dir / "AirQualityUCI.csv"
        cache_file = self.cache_dir / "uci_air_quality.parquet"
        
        # If cache exists and we're not forcing a download, load from cache
        if cache_file.exists() and not force_download:
            try:
                return pl.read_parquet(cache_file)
            except Exception as e:
                print(f"Error loading from cache: {e}. Processing raw data.")
        
        # Download if needed
        if not data_file.exists() or force_download:
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip"
            print(f"Downloading Air Quality dataset from {url}")
            
            response = requests.get(url)
            if response.status_code != 200:
                raise ConnectionError(f"Failed to download data: {response.status_code}")
                
            # Extract from zip
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                with open(data_file, 'wb') as f:
                    f.write(z.read('AirQualityUCI.csv'))
        
        # Process the data with polars
        df = pl.read_csv(data_file, separator=';', decimal_separator=',')
        
        # Clean and prepare the data
        df = df.with_columns([
            pl.col('Date').str.strptime(pl.Date, '%d/%m/%Y', strict=False),
            pl.col('Time').str.strptime(pl.Time, '%H.%M.%S', strict=False)
        ])
        
        # Combine date and time
        df = df.with_columns([
            pl.concat_str([pl.col('Date').cast(pl.Utf8), pl.lit(' '), pl.col('Time').cast(pl.Utf8)])
            .str.strptime(pl.Datetime, '%Y-%m-%d %H:%M:%S')
            .alias('DateTime')
        ])
        
        # Drop original columns
        df = df.drop(['Date', 'Time'])
        
        # Replace -200 values (missing data) with null
        df = df.with_columns([
            pl.when(pl.col(col) == -200).then(None).otherwise(pl.col(col))
            for col in df.columns if col != 'DateTime'
        ])
        
        # Drop unnamed column if it exists
        if 'Unnamed: 15' in df.columns:
            df = df.drop('Unnamed: 15')
        
        # Ensure DateTime is first column and rename it to Date for consistency
        df = df.select(['DateTime'] + [col for col in df.columns if col != 'DateTime'])
        df = df.rename({'DateTime': 'Date'})
        
        # Save to cache
        df.write_parquet(cache_file)
        
        return df

    def load_ucr_dataset(self, dataset_name: str, force_download: bool = False) -> Dict[str, Any]:
        """
        Load a dataset from the UCR/UEA Time Series Classification Archive.
        
        Args:
            dataset_name: Name of the dataset
            force_download: If True, download even if the files exist locally
            
        Returns:
            Dictionary containing train and test data as Polars DataFrames
        """
        # File paths
        data_dir = self.data_dir / "UCR" / dataset_name
        cache_file = self.cache_dir / f"ucr_{dataset_name}.pkl"
        
        # If cache exists and we're not forcing a download, load from cache
        if cache_file.exists() and not force_download:
            try:
                import pickle
                with open(cache_file, 'rb') as f:
                    result = pickle.load(f)
                
                # Convert pandas DataFrames to polars if needed
                if 'train' in result and not isinstance(result['train'], pl.DataFrame):
                    result['train'] = pl.from_pandas(result['train'])
                if 'test' in result and not isinstance(result['test'], pl.DataFrame):
                    result['test'] = pl.from_pandas(result['test'])
                
                return result
            except Exception as e:
                print(f"Error loading from cache: {e}. Processing raw data.")
        
        # Download if needed
        if not data_dir.exists() or force_download:
            data_dir.mkdir(parents=True, exist_ok=True)
            
            base_url = "https://www.timeseriesclassification.com/aeon-toolkit/Archives/Univariate2018_arff/"
            
            for file_type in ['train', 'test']:
                url = f"{base_url}{dataset_name}/{dataset_name}_{file_type}.arff"
                print(f"Downloading {file_type} data from {url}")
                
                try:
                    response = requests.get(url)
                    if response.status_code != 200:
                        raise ConnectionError(f"Failed to download {file_type} data: {response.status_code}")
                        
                    with open(data_dir / f"{dataset_name}_{file_type}.arff", 'wb') as f:
                        f.write(response.content)
                except Exception as e:
                    print(f"Error downloading {file_type} data: {e}")
                    raise
        
        # Parse ARFF files to Polars DataFrames
        train_data = self._parse_arff_to_polars(data_dir / f"{dataset_name}_train.arff")
        test_data = self._parse_arff_to_polars(data_dir / f"{dataset_name}_test.arff")
        
        result = {
            'train': train_data,
            'test': test_data,
            'name': dataset_name
        }
        
        # Save to cache
        import pickle
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        
        return result

    def _parse_arff_to_polars(self, file_path: Union[str, Path]) -> pl.DataFrame:
        """
        Parse an ARFF file into a Polars DataFrame.
        
        Args:
            file_path: Path to the ARFF file
            
        Returns:
            Polars DataFrame with the data
        """
        data = []
        attributes = []
        data_section = False
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('%'):
                    continue
                
                # Parse attribute definitions
                if line.lower().startswith('@attribute'):
                    # Extract attribute name (remove quotes if present)
                    attr_name = line.split()[1].strip("'\"")
                    attributes.append(attr_name)
                
                # Check for data section
                elif line.lower().startswith('@data'):
                    data_section = True
                
                # Parse data lines
                elif data_section:
                    # Split by comma and convert to appropriate types
                    values = line.split(',')
                    
                    # Try to convert numeric values
                    processed_values = []
                    for val in values:
                        try:
                            # Convert to numeric if possible
                            processed_values.append(float(val))
                        except ValueError:
                            # Keep as string if not numeric
                            processed_values.append(val)
                    
                    data.append(processed_values)
        
        # Create Polars DataFrame
        # Convert data to dict of columns
        if not data:
            return pl.DataFrame()
        
        columns = {attr: [] for attr in attributes}
        for row in data:
            for i, val in enumerate(row):
                if i < len(attributes):
                    columns[attributes[i]].append(val)
        
        # Create DataFrame with appropriate types
        return pl.DataFrame(columns)

    def load_physionet_dataset(self, dataset_id: str, force_download: bool = False) -> pl.DataFrame:
        """
        Load a dataset from PhysioNet.
        
        Args:
            dataset_id: PhysioNet dataset ID
            force_download: If True, download even if the file exists locally
            
        Returns:
            Polars DataFrame with the data
        """
        # File paths
        data_dir = self.data_dir / "PhysioNet" / dataset_id
        cache_file = self.cache_dir / f"physionet_{dataset_id}.parquet"
        
        # If cache exists and we're not forcing a download, load from cache
        if cache_file.exists() and not force_download:
            try:
                return pl.read_parquet(cache_file)
            except Exception as e:
                print(f"Error loading from cache: {e}. Processing raw data.")
        
        # For now, only implement a few specific datasets
        if dataset_id == "challenge-2012":
            df = self._load_physionet_challenge_2012(data_dir, force_download)
        elif dataset_id == "mitdb":
            df = self._load_physionet_mitdb(data_dir, force_download)
        else:
            raise ValueError(f"Dataset {dataset_id} not implemented yet")
        
        # Save to cache
        df.write_parquet(cache_file)
        
        return df

    def _load_physionet_challenge_2012(self, data_dir: Path, force_download: bool) -> pl.DataFrame:
        """
        Load the PhysioNet Challenge 2012 dataset (Mortality prediction).
        
        Args:
            data_dir: Directory to store the data
            force_download: If True, download even if the files exist locally
            
        Returns:
            Polars DataFrame with the data
        """
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # URLs for the training set A
        outcome_url = "https://physionet.org/files/challenge-2012/1.0.0/Outcomes-a.txt"
        records_dir_url = "https://physionet.org/files/challenge-2012/1.0.0/set-a/"
        
        # Download outcome file if needed
        outcome_file = data_dir / "Outcomes-a.txt"
        if not outcome_file.exists() or force_download:
            print(f"Downloading outcome data from {outcome_url}")
            urllib.request.urlretrieve(outcome_url, outcome_file)
        
        # Read outcome data
        outcome_df = pl.read_csv(outcome_file)
        
        # For simplicity, just download a few example records
        sample_records = outcome_df.select('RecordID').head(10).to_series().to_list()
        
        all_dfs = []
        
        for record_id in sample_records:
            record_file = data_dir / f"{record_id}.txt"
            
            # Download record if needed
            if not record_file.exists() or force_download:
                record_url = f"{records_dir_url}{record_id}.txt"
                print(f"Downloading record {record_id} from {record_url}")
                try:
                    urllib.request.urlretrieve(record_url, record_file)
                except Exception as e:
                    print(f"Error downloading record {record_id}: {e}")
                    continue
            
            # Read record data
            try:
                record_df = pl.read_csv(record_file)
                
                # Add record ID
                record_df = record_df.with_columns(pl.lit(record_id).alias('RecordID'))
                
                # Get outcome
                in_hospital_death = outcome_df.filter(pl.col('RecordID') == record_id).select('In-hospital_death').item()
                record_df = record_df.with_columns(pl.lit(in_hospital_death).alias('In-hospital_death'))
                
                all_dfs.append(record_df)
            except Exception as e:
                print(f"Error processing record {record_id}: {e}")
        
        # Combine all records
        if not all_dfs:
            return pl.DataFrame()
        
        combined_df = pl.concat(all_dfs)
        
        return combined_df

    def _load_physionet_mitdb(self, data_dir: Path, force_download: bool) -> pl.DataFrame:
        """
        Load the MIT-BIH Arrhythmia Database.
        
        Args:
            data_dir: Directory to store the data
            force_download: If True, download even if the files exist locally
            
        Returns:
            Polars DataFrame with the data
        """
        # This is a placeholder for now - loading ECG data requires specialized libraries
        # like wfdb which we might want to add as a dependency later
        raise NotImplementedError("MIT-BIH dataset loading not yet implemented - requires wfdb package")

    def list_ucr_datasets(self) -> List[str]:
        """
        List available datasets in the UCR/UEA Time Series Classification Archive.
        
        Returns:
            List of dataset names
        """
        # URL for the dataset list
        url = "https://www.timeseriesclassification.com/dataset.php"
        
        try:
            response = requests.get(url)
            if response.status_code != 200:
                raise ConnectionError(f"Failed to get dataset list: {response.status_code}")
                
            # Parse the HTML to extract dataset names
            # This is a simple approach - might need beautifulsoup for more robust parsing
            datasets = []
            for line in response.text.split('\n'):
                if "dataset=" in line and "<a href" in line:
                    # Extract dataset name from the URL
                    dataset_name = line.split("dataset=")[1].split('"')[0]
                    datasets.append(dataset_name)
            
            return datasets
        except Exception as e:
            print(f"Error getting UCR dataset list: {e}")
            return []

    def get_kaggle_dataset(self, dataset_name: str, force_download: bool = False) -> pl.DataFrame:
        """
        Load a dataset from Kaggle.
        
        Note: Requires kaggle API credentials to be set up.
        See: https://github.com/Kaggle/kaggle-api#api-credentials
        
        Args:
            dataset_name: Kaggle dataset name (e.g., 'titanic')
            force_download: If True, download even if the file exists locally
            
        Returns:
            Polars DataFrame with the data
        """
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
        except ImportError:
            raise ImportError("kaggle package is required but not installed. "
                             "Install it with 'pip install kaggle'")
        except Exception as e:
            raise Exception(f"Error authenticating with Kaggle API: {e}. Make sure you have set up your API credentials.")
        
        # Cache file path
        cache_file = self.cache_dir / f"kaggle_{dataset_name.replace('/', '_')}.parquet"
        
        # If cache exists and we're not forcing a download, load from cache
        if cache_file.exists() and not force_download:
            try:
                return pl.read_parquet(cache_file)
            except Exception as e:
                print(f"Error loading from cache: {e}. Downloading fresh data.")
        
        # Download the dataset
        data_dir = self.data_dir / "Kaggle" / dataset_name
        data_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            api.dataset_download_files(dataset_name, path=data_dir, unzip=True, force=force_download)
        except Exception as e:
            raise Exception(f"Error downloading dataset {dataset_name}: {e}")
        
        # Find CSV files in the directory
        csv_files = list(data_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in dataset {dataset_name}")
        
        # Load the first CSV file
        df = pl.read_csv(csv_files[0])
        
        # Save to cache
        df.write_parquet(cache_file)
        
        return df

    def load_nasa_cmapss(self, force_download: bool = False) -> Dict[str, pl.DataFrame]:
        """
        Load the NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset.
        
        Args:
            force_download: If True, download even if the file exists locally
            
        Returns:
            Dictionary of Polars DataFrames with the data
        """
        # File paths
        data_dir = self.data_dir / "NASA_CMAPSS"
        cache_file = self.cache_dir / "nasa_cmapss.pkl"
        
        # If cache exists and we're not forcing a download, load from cache
        if cache_file.exists() and not force_download:
            try:
                import pickle
                with open(cache_file, 'rb') as f:
                    result = pickle.load(f)
                
                # Convert pandas DataFrames to polars if needed
                for fd in result:
                    for data_type in result[fd]:
                        if not isinstance(result[fd][data_type], pl.DataFrame):
                            result[fd][data_type] = pl.from_pandas(result[fd][data_type])
                
                return result
            except Exception as e:
                print(f"Error loading from cache: {e}. Processing raw data.")
        
        # Download if needed
        if not data_dir.exists() or force_download:
            data_dir.mkdir(parents=True, exist_ok=True)
            
            url = "https://ti.arc.nasa.gov/c/6/"
            print(f"Downloading NASA C-MAPSS dataset from {url}")
            
            try:
                response = requests.get(url)
                if response.status_code != 200:
                    raise ConnectionError(f"Failed to download data: {response.status_code}")
                    
                # Extract from zip
                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    z.extractall(data_dir)
            except Exception as e:
                print(f"Error downloading NASA C-MAPSS dataset: {e}")
                raise
        
        # Process the data
        result = {}
        
        # Column names for the data
        train_cols = ['unit_number', 'time_cycles'] + [f'op_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]
        test_cols = train_cols.copy()
        rul_cols = ['rul']
        
        for fd in range(1, 5):
            # Load training data
            train_file = data_dir / f"train_FD00{fd}.txt"
            train_df = pl.read_csv(train_file, separator=' ', has_header=False, new_columns=train_cols)
            # Drop the last column which is empty due to the trailing space
            train_df = train_df.select(train_df.columns[:-1])
            
            # Load test data
            test_file = data_dir / f"test_FD00{fd}.txt"
            test_df = pl.read_csv(test_file, separator=' ', has_header=False, new_columns=test_cols)
            # Drop the last column
            test_df = test_df.select(test_df.columns[:-1])
            
            # Load RUL (Remaining Useful Life) data
            rul_file = data_dir / f"RUL_FD00{fd}.txt"
            rul_df = pl.read_csv(rul_file, separator=' ', has_header=False, new_columns=rul_cols)
            # Drop the last column
            rul_df = rul_df.select(rul_df.columns[:-1])
            
            # Add RUL to training data
            # Group by unit_number, find max time_cycles, then calculate RUL
            max_cycles = train_df.group_by('unit_number').agg(pl.max('time_cycles').alias('max_cycles'))
            train_df = train_df.join(max_cycles, on='unit_number')
            train_df = train_df.with_columns((pl.col('max_cycles') - pl.col('time_cycles')).alias('rul')).drop('max_cycles')
            
            # Store the data
            result[f'FD00{fd}'] = {
                'train': train_df,
                'test': test_df,
                'rul': rul_df
            }
        
        # Save to cache
        import pickle
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        
        return result


def get_yahoo_data(symbol: str, 
                  start_date: Union[str, datetime], 
                  end_date: Optional[Union[str, datetime]] = None) -> pl.DataFrame:
    """
    Convenience function to load Yahoo Finance data.
    
    Args:
        symbol: Stock/index symbol
        start_date: Start date for data retrieval
        end_date: End date for data retrieval (defaults to today)
        
    Returns:
        Polars DataFrame with OHLCV data
    """
    loader = TimeSeriesDataLoader()
    return loader.get_yahoo_data(symbol, start_date, end_date)


def list_available_datasets() -> Dict[str, List[str]]:
    """
    List all available datasets from various sources.
    
    Returns:
        Dictionary with dataset sources and available datasets
    """
    loader = TimeSeriesDataLoader()
    result = {}
    
    # UCR datasets
    try:
        result['UCR'] = loader.list_ucr_datasets()
    except Exception as e:
        print(f"Error getting UCR datasets: {e}")
        result['UCR'] = []
    
    # Built-in datasets
    result['Built-in'] = [
        'uci_air_quality',
        'nasa_cmapss'
    ]
    
    # Financial data
    result['Financial'] = [
        'Yahoo Finance (requires symbol)',
        'FRED (requires series_id and API key)',
        'Quandl (requires dataset_code)'
    ]
    
    # Medical data
    result['Medical'] = [
        'physionet_challenge_2012',
        'physionet_mitdb'
    ]
    
    return result