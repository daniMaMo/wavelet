"""
September the 9th, 2020
CDMX
F. Sagols.

Trading in Interactive Brokers Project
This is an example to show the use of time_series program
"""

# You should restructure the imports and install the required libraries
# into the environment.
# In time_series.py eliminate any invocation to methods in timestamp_executions
# module and any method using adj_downloader.

# Before running this program copy the file bash_profile into your root
# directory in "~/.bash_profile", do not forget the dot before "bash".  It
# contains the credentials to access the IBDb (Interactive Brokers database).
# Restart your computer Before running this program for the first time.

import datetime
import pandas as pd
import time_series
import csv


def main():
    """
    Examples of the use the time series module.
    """
    # To display full pandas dataframes.
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    print('Last 10 one day AAPL prices ending at 2022-12-30')
    aapl = time_series.read_ts_from_ibdb('AAPL', '1 day', None, datetime.datetime(2022, 12, 30), last=10)
    print(aapl)
    print()

    print('Last 10 one minute AAPL prices ending at 2022-12-30')
    aapl = time_series.read_ts_from_ibdb(
        'AAPL', '1 min', None, datetime.datetime(2022, 12, 30), last=10)
    print(aapl)
    print()

    print('Apple prices between 2022-10-1 and 2022-11-1')
    aapl = time_series.read_ts_from_ibdb(
        'AAPL', '1 day', datetime.datetime(2022, 10, 1),
        datetime.datetime(2022, 11, 1))
    print(aapl)
    print()

    print('Symbols catalogs')
    catalogs = time_series.get_catalogs()
    print('\n'.join(catalogs))
    print()

    print('Symbols in the sp500 catalog')
    sp500 = time_series.symbols_short_dictionary(
        ('sp500',), ('STK',))
    print('\n'.join(sp500.keys()))

    # You may find useful other time_series methods.

    # Download data.
    data = time_series.read_ts_from_ibdb('AAPL', '1 day', None, '2023-08-31', last=1000)
    data_adj_close = data[0]['adj_close']

    with open('AAPL.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(data_adj_close)


if __name__ == "__main__":
    main()
