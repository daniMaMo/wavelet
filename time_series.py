"""
October the 6th, 2020
CDMX
F. Sagols.
M. Sagols.

Trading in Interactive Brokers Project
Time series management
"""

import argparse
import datetime
import math
import os
import subprocess

import http
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import sql
import psycopg2.extras
import time
import yfinance as yf

from ibdb_connection import connection_tunnel
# import matplotlib
# from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import common
# from platind import common, timestamp_executions
# from series import adj_downloader
# matplotlib.use('TkAgg')
# matplotlib.use('Qt5Agg')

# Global variables
# List of bar sizes that will be downloaded with this program.
TS_EXPORT = "/home/fsagols/Desktop/time-series"
TS_DROPBOX = "/home/fsagols/Dropbox/ts_export"

BAR_SIZES = ['1 hour', '1 day']
# Directory where the time series will be downloaded for interchange


class InvalidBarSizeOrTicker(Exception):
    """ Exception raised when some bar size or symbol is unappropriated.

        ATTRIBUTES
        ----------
        bar_size : str
            Identifier of the wrong bar size
        symbol : str
            Identifier or the wrong symbol
        description: str
            A description of the error

    """
    def __init__(self, bar_size, symbol, description):
        Exception.__init__(self)
        self.bar_size = bar_size
        self.symbol = symbol
        self.description = description


def bar_sizes_dictionary():
    """
    Get the bar_sizes dictionary from the database.

    Parameters
    ----------

    Returns
    -------
    A dictionary. The keys correspond to bar_sizes. The value is a dictionary
    with the attributes:
    bar_size: The size of the bars: '1 secs', '5 secs' and so on.
    download_size: For '1 seconds' bars the maximum number of bars allowed by
        IB in a reqHistoricalDataIB is 900. The other values have a similar
        meaning.
    strconv: Format used by strftime and strptime to convert dates for the
        bar_size.
    dlsfb: Daylight saving first bar.
    dlslb: Daylight saving last bar.
    ndlsfb: Non-daylight saving first bar.
    ndlslb: Non-daylight saving last bar.
    """
    conn = connection_tunnel()
    cur = conn.cursor()
    cur.execute("select * from bar_sizes;")
    bar_sizes = {
        a[0]: {
            'download_size': a[1],
            'strconv': a[2],
            'dlsfb': a[3],
            'dlslb': a[4],
            'ndlsfb': a[5],
            'ndlslb': a[6],
            'alpha_vantage_bs': a[7]
        }
        for a in cur.fetchall()
    }
    conn.commit()
    return bar_sizes


def get_catalogs():
    """
    Returns the market catalogs lists.
    """
    conn = connection_tunnel()
    cur = conn.cursor()
    cur.execute("select catalog_name from catalog_names;")
    catalogs = cur.fetchall()
    conn.commit()
    return [a[0] for a in catalogs]


def get_available_symbols(min_records):
    """
    It returns all available symbols in time_series table

    Parameters
    ----------
    min_records : int
        Retrieve only symbols having more than this number of records.

    Examples
    --------
    >>> get_available_symbols(700)
    Returns
    -------
        list
    """
    conn = connection_tunnel()
    cur = conn.cursor()
    cur.execute(
        """
        select foo.symbol
        from (select count(*), symbol from time_series group by symbol) as foo
        where foo.count > %(min_records)s
        order by foo.symbol;
        """, {'min_records': min_records})
    answer = cur.fetchall()
    conn.commit()
    return [a[0] for a in answer]


def commissions_from_ibdb(logger):
    """ It produces a relation of the commissions paid for buying or selling
    assets at IB and recorded into the table orders of the IBdb database.
    The relation comes in the dictionary 'commissions'. The tickers
    are the keys, and the values are sub-dictionaries with keys 'BUYC'
    (buy call commission), 'SELLC', 'BUYP' and 'SELLP' for each ticker.
    It also produces a dictionary 'avg_std' with keys 'BUYC', 'SELLC',
    'BUYP', 'SELLP', 'BUYS' (stock buy commission) and 'SELLS'  and their
    respective average and standard deviation values over all available
    instruments.

    PARAMETERS
    ----------
    logger : Logger
        of the module.

    RETURNS
    -------
    The  tuple (commissions, avg-commissions)
    """
    conn = connection_tunnel()
    cur = conn.cursor()

    avg_commissions = {}
    # Option commissions download
    cur.execute("""
        select action_, right_, avg(order_commission/filled),
            stddev(order_commission)
        from orders
        where sectype = 'OPT' and order_status = 'Filled'
        group by sectype, action_, right_;
        """)
    answer = cur.fetchall()
    conn.commit()
    for ans in answer:
        avg_commissions[ans[0] + ans[1]] = \
            dict(avg=ans[2], std=ans[3])
    # Stock commissions download.
    cur.execute("""
        select action_, avg(order_commission/filled),
            stddev(order_commission/filled) from orders
        where sectype = 'STK' and order_status = 'Filled'
        group by action_;
        """)
    answer = cur.fetchall()
    conn.commit()
    for ans in answer:
        avg_commissions[ans[0] + 'S'] = {'avg': ans[1], 'std': ans[2]}

    # Detailed option commissions download.
    cur.execute("""
        select symbol, action_, right_, avg(order_commission/filled),
            stddev(order_commission/filled)
        from orders
        where sectype = 'OPT' and order_status = 'Filled'
        group by symbol, sectype, action_, right_
        order by symbol, action_, right_;
        """)
    answer = cur.fetchall()
    conn.commit()
    commissions = {}
    for ans in answer:
        avg = avg_commissions[ans[1] + ans[2]]['avg'] if \
            ans[3] is None or ans[3] == 0.0 else ans[3]
        std = avg_commissions[ans[1] + ans[2]]['std'] if \
            ans[4] is None or ans[4] == 0.0 else ans[4]
        if ans[0] not in commissions:
            commissions[ans[0]] = {}
        commissions[ans[0]][ans[1] + ans[2]] = avg + std
    # Detailed stock commissions download.
    cur.execute("""
        select symbol, action_, avg(order_commission/filled),
            stddev(order_commission/filled)
        from orders
        where sectype = 'STK' and order_status = 'Filled'
        group by symbol, action_ order by symbol, action_;
        """)
    answer = cur.fetchall()
    conn.commit()
    for ans in answer:
        avg = avg_commissions[ans[1] + 'S']['avg'] if \
            ans[2] is None or ans[2] == 0.0 else ans[2]
        std = avg_commissions[ans[1] + 'S']['std'] if \
            ans[3] is None or ans[3] == 0.0 else ans[3]
        if ans[0] not in commissions:
            commissions[ans[0]] = {}
        commissions[ans[0]][ans[1] + 'S'] = avg  # TODO + std

    logger.info('The commissions were updated in the symbols dictionary')
    return commissions, avg_commissions


def complete_commissions(symbols, logger):
    """ Set the values 'SELLC' (sell call option commission), 'BUYC', 'SELLP',
        and 'BUYP' in the  symbol's dictionary to the average values obtained
        from orders table in the IBdb database.

        PARAMETERS
        ----------
        symbols : dict
            Financial symbol's dictionary.
        logger :
            Program logger.

        RETURNS
        -------
        True: If the process was successful.
        False: Otherwise
    """

    if not symbols:
        return
    commissions, avg_std = commissions_from_ibdb(logger)
    for sym in symbols:
        for a_r in ['BUYC', 'BUYP', 'SELLC', 'SELLP', 'SELLS', 'BUYS']:
            if sym in commissions and a_r in commissions[sym] and\
               commissions[sym][a_r] > 0:
                symbols[sym][a_r] = commissions[sym][a_r]
            else:
                symbols[sym][a_r] = \
                    avg_std[a_r]['avg'] + avg_std[a_r]['std'] if \
                    a_r in avg_std else 0.0


def multipliers_from_ibdb(logger):
    """ Produces a dictionary (symbol: multiplier) from orders table in IBdb.

    PARAMETERS
    ----------
    logger : Logger
        of the module.

    RETURNS
    -------
    The dictionary (symbol, multiplier)
    """
    conn = connection_tunnel()
    cur = conn.cursor()
    cur.execute("""
        select distinct symbol, multiplier
        from orders where order_status = 'Filled' and
            sectype = 'OPT';
        """)
    answer = cur.fetchall()
    conn.commit()
    multipliers = {}
    for ans in answer:
        if ans[0] not in multipliers:
            multipliers[ans[0]] = float(ans[1])
        else:
            if float(ans[1]) != multipliers[ans[0]]:
                raise ValueError(
                    "There are two different option multiplier values")
    logger.info("Multipliers added to the symbols dictionary.")
    return multipliers


def symbols_short_dictionary(catalog_tuple, sect_type_tuple, from_list=None) \
        -> dict:
    """
    Builds the short symbol's dictionary.

    PARAMETERS
    ----------
    catalog_tuple : tuple
        List of catalog names (like 'sp500', 'nasdaq' and so on) whose
        symbols will be included in the dictionary. If the list contains the
        word 'all' then all available symbols will be returned.
    sect_type_tuple : tuple
        It contains the type of assets that should be included. The
        possibilities include STK, OPT, INDEX, ETF, and so on.
    from_list : list
        If the symbols are taken from this list instead and the other parameters
        are ignored

    RETURNS
    -------
        dict
    The short symbol's dictionary. Each entry is a dictionary with just one
    attribute:
    ib_id: The symbols' contract id.
    """
    assert (catalog_tuple is None and from_list is not None) or \
           (catalog_tuple is not None and from_list is None), \
           "Only one of catalog_tuple or from_list should be None"
    conn = connection_tunnel()
    cur = conn.cursor()
    if catalog_tuple is None or catalog_tuple[0] != 'all':
        if from_list is None:
            cur.execute(
                """
                select c.symbol, s.ib_id
                from market_catalogs c, symbols s
                where c.symbol = s.symbol and
                      c.catalog_name in %(catalog_list)s and
                      sectype in %(sect_type_tuple)s order by c.weight desc;
                """, {
                    'catalog_list': catalog_tuple,
                    'sect_type_tuple': sect_type_tuple
                })
        else:
            cur.execute(
                """
                select symbol, ib_id
                from symbols
                where symbol in %(from_list)s                
                """, {
                    "from_list": tuple(from_list)
                }
            )
    else:
        cur.execute("select symbol, ib_id from symbols;")
    sym_buffer = cur.fetchall()
    conn.commit()
    symbols = {a[0]: {'ib_id': a[1]} for a in sym_buffer}

    return symbols


def symbols_dictionary(catalog_tuple, sect_type_tuple, logger, from_list=None):
    """ Build the symbol's dictionary.

    PARAMETERS
    ----------
    catalog_tuple : tuple
        It contains the catalog names(like 'sp500', 'nasdaq' and so on). If
        catalog name is 'all' then all available symbols should be returned.
    sect_type_tuple : tuple
        It contains the type of assets that should be included. The
        possibilities include STK, OPT, INDEX, ETF, and so on.
    logger :
        Program's logger.
    from_list : list
        If a list of symbols is provided then the dictionary on that list is
        returned ignoring the catalog_tuple and sect_type parameter.

    RETURNS
    -------
    The symbol's dictionary. It contains the symbols in catalog_name as keys.
    Each entry is dictionary with the attributes:
    ib_ib: Interactive Broker's id of the symbol contract.
    multiplier: Multiplier used in option trading operations.
    price: Last price. To be filled.
    forecasted_price: Forecasted price. To be filled.
    shortable: How easy the stock could be shorted. To be filled
    dividends: Dividend payment information. To be filled.

    See the commission method for some additional attributes.
    """
    symbols = symbols_short_dictionary(
        catalog_tuple, sect_type_tuple, from_list)

    multipliers = multipliers_from_ibdb(logger)
    for sym in symbols:
        if sym in multipliers:
            symbols[sym]['multiplier'] = multipliers[sym]
        else:
            symbols[sym]['multiplier'] = 100
        symbols[sym]['price'] = 0.0
        symbols[sym]['forecasted_price'] = 0.0
        symbols[sym]['expires'] = None
        symbols[sym]['strikes'] = None
        symbols[sym]['shortable'] = None
        symbols[sym]['dividends'] = None

    complete_commissions(symbols, logger)

    return symbols


def read_catalog_from_ibdb(catalog):
    """
    Read a catalog from the market catalogs table.

    PARAMETERS
    ----------
    catalog : str
        Name of the catalog.
    """
    conn = connection_tunnel()
    cur = conn.cursor()
    cur.execute(
        """
        select c.catalog_name, s.company, c.symbol, c.weight
        from market_catalogs c, symbols s
        where c.symbol = s.symbol and c.catalog_name = %(catalog)s
        order by c.weight desc;
        """, {'catalog': catalog})
    answer = cur.fetchall()
    conn.commit()
    if answer is None:
        return None
    answer = [list(a) for a in answer]
    result = pd.DataFrame(data=answer,
                          columns=['index', 'company', 'symbol', 'weight'])
    return result


def last_record_bar(symbol, bar_size, imfs=False):
    """
    Returns the last bar time recorded in time_seres for the symbol.

    Parameters
    ----------
    symbol : str
        Name of the symbol
    bar_size : str
        Size of the bar.
    imfs : bool
        True if and only if we need the last bar for which the Ceemdan has been
        computed.

    Returns
    -------
        datetime.py
        The last bar in time_series or None if none is there.
    """
    conn = connection_tunnel()
    cur = conn.cursor()
    conn.commit()
    cur.execute(
        f"""
        select date_
        from time_series
        where symbol = %(symbol)s and bar_size = %(bar_size)s
              {'and imf1 is not null' if imfs else ''}
        order by date_
              desc limit 1;
        """, {
            'symbol': symbol,
            'bar_size': bar_size
        })
    answer = cur.fetchone()
    conn.commit()
    if answer is not None:
        return answer[0]
    time.sleep(5)
    return None


def get_price(symbol, bar_size, date, price):
    """
    Get a price.

    Parameters
    ----------
    symbol : str
        Financial instrument ticker.
    bar_size : str
        Size of the bar.
    date : datetime.pyi
        Date of the bar
    price : str
        Name of the price to return: 'open_', 'high', 'low', 'close_'.

    Examples
    --------
    >>> get_price('AAPL', '1 day', '2022-12-08', 'close_')
    150.0
    Returns
    -------
        The required price.
    """
    conn = connection_tunnel()
    cur = conn.cursor()
    # TODO Without this commit the program fails. We do not understand
    # why. If we don't use the Identifier price everything works fine.
    # Probable error in module psycopg2.
    conn.commit()
    cur.execute(
        sql.SQL(
            """
            select {}
            from time_series 
            where symbol = %s and bar_size = %s and 
                  date_ = %s;
             """).format(sql.Identifier(price)), [symbol, bar_size, date])
    answer = cur.fetchone()
    conn.commit()
    return answer[0]


# def get_closing_price(symbol, date, download=True):
#     """
#     This method retrieves from IBdb.time_series the price of {symbol} at {
#     date}. If it is not available then it retrieves the closest one before
#     date. If no one is available then it downloads the time series and
#     repeats the process. Finally, if no one was found it returns None.
#
#     PARAMETERS
#     --------
#     symbol : str
#         Name of the symbol.
#     date : datetime
#         Asked date.
#     download : bool
#         True if and only if the time series is downloaded when the price
#         is unavailable for date and symbol
#
#     Returns
#     -------
#         float
#         The last available price for symbol or None if no one is availalble.
#
#     Examples
#     --------
#     >>> get_closing_price('COTY', datetime.datetime(2021, 9, 3))
#     70.2
#     >>> get_closing_price('HFC', datetime.datetime(2021, 8, 13))
#     30.39
#     >>> get_closing_price('NOASSET', datetime.datetime(2021, 8, 13))
#     20.0
#     """
#     conn = connection_tunnel()
#     cur = conn.cursor()
#     # Try to get the price from IBdb.time_series.
#     for attempt in range(2):
#         if attempt == 0 and download:
#             cur.execute(
#                 """
#                 select close_, date_ from time_series
#                 where symbol = %(symbol)s and bar_size = '1 day'
#                       and date_ = %(date)s order by date_ desc limit 1;
#                 """, {
#                     'symbol': symbol,
#                     'date': date
#                 })
#         else:
#             cur.execute(
#                 """
#                 select close_, date_ from time_series
#                 where symbol = %(symbol)s and bar_size = '1 day'
#                       and date_ <= %(date)s order by date_ desc limit 1;
#                 """, {
#                     'symbol': symbol,
#                     'date': date
#                 })
#         answer = cur.fetchone()
#         conn.commit()
#         if answer is not None and answer[0] is not None:
#             return answer[0]
#         if attempt > 0:
#             raise ValueError(
#                 f"It was not possible to get the price or {symbol}"
#                 f" at {str(date)}.")
#         try:
#             if last_record_bar(symbol, '1 day') < date:
#                 adj_downloader.download_ts('1 day', symbol, 'FeliuAVC')
#         except ValueError as error:
#             print(error)
#     return None


def read_ts_from_fs(symbol,
                    bar_size,
                    start_bar,
                    end_bar,
                    interpolate=False,
                    last=0,
                    imf=False):
    """
    Read a time series from a file.

    PARAMETERS
    ----------
    symbol : str
        Name of the stock
    bar_size : str
        Size of the bar
    start_bar : datetime.pyi
        Left end of the time interval to be read.
    end_bar : datetime.pyi
        Right end of the time interval to be read.
    interpolate : bool
        If the missing bars in the time series should be interpolated or not.
    last : int
        Last records number to read.
    imf : bool
        Boolean given to decide whether to include the imf data in the result.

    RETURNS
    -------
    A Pandas data frame containing the series.

    Examples
    --------
    >>> read_ts_from_fs('AAPL', '1 day', None, datetime.datetime(2021, 12,
    ... 4), last=1000)

    """
    cols = range(13) if imf else range(7)
    data_frame = pd.read_csv("./series/time_series/" + symbol + ".csv",
                             header=0,
                             index_col=0,
                             usecols=cols,
                             parse_dates=True)
    if start_bar is not None:
        data_frame = data_frame.loc[start_bar:]
    if end_bar is not None:
        data_frame = data_frame.loc[:end_bar]
    if 0 < last < len(data_frame):
        data_frame = data_frame.iloc[-last:]
    return data_frame


def read_ts_from_ibdb(symbol,
                      bar_size,
                      start_bar,
                      end_bar,
                      interpolate=False,
                      last=0,
                      imf=False,
                      with_info=False,
                      in_dataframe=True):
    """
    Read a time series from the database. If it is not accessible then the
    time series is recovered from the file system.

    PARAMETERS
    ----------
    symbol : str
        Name of the stock
    bar_size : str
        Size of the bar
    start_bar : datetime.pyi
        Left end of the time interval to be read. It could be None.
    end_bar : datetime.pyi
        Right end of the time interval to be read. It could be None.
    interpolate : bool
        If the missing bars in the time series should be interpolated or not.
    last : int
        Last records number to read.
    imf : bool
        Boolean given to decide whether to include the imf data in the result.
    with_info : bool
        True if and only if the field info is required at the end
    in_dataframe : bool
        If the result should be return in a pandas dataframe

    RETURNS
    -------
    A Pandas data frame containing the series.
    The number of imf's in the response.

    Examples
    --------
    >>> read_ts_from_ibdb('AAPL', '1 day', None, datetime.datetime(2022, 12,
    ... 4), last=1000)

    """
    assert bar_size in [
        '1 day', '1M', '1Q', '1 hour', '1 min', '5 mins', '15 mins', '30 mins'
    ], f"Unknown bar size {bar_size}"
    conn = None
    while True:
        try:
            conn = connection_tunnel()
            break
        except psycopg2.OperationalError:
            # return read_ts_from_fs(symbol, bar_size, start_bar, end_bar,
            #                        interpolate, last, imf), None
            print("Connection error loading %s." % symbol)
            continue
    cur = conn.cursor()
    query = ("select date_ as base_date, open_ as open, high, low, "
             "close_ as close, adj_close, volume")
    if with_info:
        query += ", info"
    if imf is True:
        query += ", imf1, imf2, imf3, imf4, imf5, imf6, imf7, imf8, imf9"
    query +=  \
        " from time_series " + \
        "where  symbol = %(symbol)s and bar_size = %(bar_size)s " + \
        ("" if start_bar is None else "and %(start_bar)s <= date_ ") + \
        ("" if end_bar is None else "and date_ <= %(end_bar)s ") + \
        ("" if bar_size == '1 day' else
         "and ((extract(hours from date_) >= 9 and extract(hours from date_) "
         "< 16) or (extract(hours from date_) = 8 and extract(minutes from "
         "date_) >= 30)) ") + \
        "order by date_ desc " + \
        ("" if last == 0 else "limit %(limit)s")
    # CODE: The extract(... from date_) are used to eliminate non-business hours
    # bars.
    cur.execute(
        query + ";", {
            "symbol": symbol,
            "bar_size": bar_size,
            "start_bar": start_bar,
            "end_bar": end_bar,
            "limit": last
        })
    answer = cur.fetchall()
    conn.commit()
    if not in_dataframe:
        answer.reverse()
        return answer, 0
    answer.reverse()
    if answer is None or not answer:
        return pd.DataFrame(), None
    answer = [list(a) for a in answer]
    if not answer:
        return pd.DataFrame(), None
    cols = ['base_date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
    if with_info:
        cols += ['info']
    if imf is True:
        for i in range(1, 10):
            cols += ['imf' + str(i)]
    # CODE: Data frame construction from sql query.
    result = pd.DataFrame(data=answer, columns=cols)
    n_imfs = None
    if imf is True:
        for i in range(1, 10):
            if not result['imf' + str(i)].notnull().values.any():
                del result['imf' + str(i)]
            else:
                n_imfs = i
    result.set_index('base_date', inplace=True)
    if interpolate:
        result.interpolate(limit=5, inplace=True)
    return result, n_imfs


def write_tss_to_fs(bar_sizes, catalogs, logger):
    """
    Writes all time_seres to the file system and to time-series repository.

    bar_sizes : tuple
        Containing the bar_sizes of the instruments to be writen.
    catalogs : tuple
        Containing the catalogs to be written.
    logger :
        Program's logger.
    """
    logger.info(
        "Starting the transference of time series from the database to %s.",
        TS_EXPORT)
    for directory in [TS_EXPORT, TS_DROPBOX]:
        common.ensure_path_existence(directory + '/')
        os.system("rm -rf " + directory + "/* 2> /dev/null")

    for cat in catalogs:
        symbols = symbols_short_dictionary((cat, ),
                                           ('STK', 'INDEX', 'FOREX', 'ETF'))
        for bar_size in bar_sizes:
            series_processed = 0
            symbols_print = []
            for symbol in symbols:
                data, _ = read_ts_from_ibdb(symbol,
                                            bar_size,
                                            datetime.datetime(1900, 1, 1),
                                            datetime.datetime(3000, 1, 1),
                                            True,
                                            imf=True)
                if data.empty:
                    continue
                series_processed += 1
                symbols_print += [symbol]
                if not os.path.exists(TS_EXPORT + '/' + cat):
                    os.mkdir(TS_EXPORT + '/' + cat)
                if not os.path.exists(TS_EXPORT + '/' + cat + '/' + bar_size):
                    os.mkdir(TS_EXPORT + '/' + cat + '/' + bar_size)
                data.to_csv(TS_EXPORT + '/' + cat + '/' + bar_size + '/' +
                            symbol + '.csv')
                if len(symbols_print) == 20:
                    logger.info("%s-%s: %s", cat, bar_size, str(symbols_print))
                    symbols_print = []
            if symbols_print:
                logger.info("%s-%s: %s", cat, bar_size, str(symbols_print))
            if series_processed == 0:
                continue
            logger.info("%s-%s: Finished", cat, bar_size)
        data = read_catalog_from_ibdb(cat)
        if data is not None:
            data.to_csv(TS_EXPORT + '/' + cat + '_symbols' + '.csv')
        logger.info("%s: index written in .csv", cat)

    # Tar file backup. It was disabled at 2022-04-13 because the dropbox
    # storage was full.
    # command = ("tar -czf " + TS_EXPORT + '/time_series.tar.gz *')
    # logger.info("Time series tar file being generated.")
    # os.system(command)
    # logger.info("Tar file generation completed.")
    # os.rename(TS_EXPORT + "/time_series.tar.gz",
    #           TS_DROPBOX + "/time_series.tar.gz")

    # subprocess.run(command, shell=True)
    # ts_tar = tarfile.open(TS_EXPORT + "/time_series.tar.gz", "w:gz")
    # for entry in os.scandir(TS_EXPORT):
    #     if entry.is_dir() or entry.name == ts_tar.name.split('/')[-1]:
    #         continue
    #     ts_tar.add(entry)
    #     if entry.name[-3:-1] + entry.name[-1] != 'csv':
    #         os.remove(entry)
    # ts_tar.close()

    os.chdir(TS_EXPORT)
    try:
        subprocess.run(["git", "add", "-A"],
                       stdout=subprocess.PIPE,
                       check=True)
        subprocess.run([
            "git", "commit", "-m", "Time series"
            " updated as of " +
            datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S")
        ],
                       check=True)
        # subprocess.run(
        #     ["git", "push", f"git@github.com:{user}/time-series.git", "main"],
        #     check=True)
        os.system('git push origin main')
    except subprocess.CalledProcessError as sbp_error:
        print(sbp_error.output)
    logger.info(
        "Transference of the time series from the database to %s"
        " and to %s finished.", TS_EXPORT, TS_DROPBOX)


def write_ts_to_ibdb(symbol, bar_size, data):
    """
    Writes data into time_series table.
    Actually this method is used to write adjusted data.

    PARAMETERS
    ----------
    symbol : str
        Name of the financial instrument
    bar_size: str
        Size of the bars.
    data : Pandas dataframe
        It contains the time series.
    """
    if bar_size == '1 day':
        records = \
            [[symbol, bar_size, str(k), data.loc[k]['1. open'],
              data.loc[k]['2. high'], data.loc[k]['3. low'],
              data.loc[k]['4. close'], data.loc[k]['5. adjusted close'],
              data.loc[k]['6. volume'], data.loc[k]['7. dividend amount'],
              data.loc[k]['8. split coefficient'],
              data.loc[k]['5. adjusted close']]
             for k in data.to_dict(orient='index').keys()]
    elif bar_size in ['1 hour', '1 min', '5 mins', '15 mins', '30 mins']:
        records = \
            [[symbol, bar_size, str(k), data.loc[k]['1. open'],
              data.loc[k]['2. high'], data.loc[k]['3. low'],
              data.loc[k]['4. close'], None, data.loc[k]['5. volume']]
             for k in data.to_dict(orient='index').keys()]
    else:
        raise ValueError("Unknown bar size " + bar_size + ".")
    query1day = ("""
              INSERT INTO time_series VALUES
              %s
              ON CONFLICT (symbol, bar_size, date_) DO
              UPDATE SET adj_close = excluded.adj_close;
              """)
    query_intraday = ("""
              INSERT INTO time_series VALUES
              %s
              ON CONFLICT (symbol, bar_size, date_) DO NOTHING;
              """)
    conn = connection_tunnel()
    cur = conn.cursor()
    # CODE executemany method iterates over execute, and hence, is really slow.
    # Here, execute_values has been placed instead and is 14x faster. The
    # queryes had to change, so they only use one %s and use excluded reference
    # instead of explicit value reference in the UPDATE statement.
    if bar_size == '1 day':
        psycopg2.extras.execute_values(cur, query1day, records)
    elif bar_size in ['1 hour', '1 min', '5 mins', '15 mins', '30 mins']:
        psycopg2.extras.execute_values(cur, query_intraday, records)
    conn.commit()


def write_index_to_ibdb(symbol, bar_size, data):
    """
    Writes data into idexes table.

    PARAMETERS
    ----------
    symbol : str,
        Time series id.
    bar_size : str
        Size of the bars.
    data : Pandas dataframe
        It contains the index.
    """
    records = [[
        symbol, bar_size, data.index[k], None, None, None, None,
        float(data.iloc[k, 0]), None, None, None
    ] for k in range(len(data)) if data.iloc[k, 0] != 0.0]
    conn = connection_tunnel()
    cur = conn.cursor()
    cur.executemany(
        """
        INSERT into time_series
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (symbol, bar_size, date_) DO NOTHING;
        """, records)
    conn.commit()


def symbols_sectors_update(logger):
    """
    Updates the symbol table with sector/category info of every symbol.
    To avoid being banned from Yahoo Finance only 200 records are daily updated.

    Parameters
    ----------
    logger :
        Program's logger.
    """
    logger.info("Updating the category/sector symbols information.")
    conn = connection_tunnel()
    cur = conn.cursor()
    catg, sect = [None, None]
    query = ("UPDATE symbols set (category, sector)= (%s, %s) where "
             "symbol=%s")
    syms_ = symbols_short_dictionary(('sp500', 'nasdaq', 'dowjones', 'etf'),
                                     ('STK', 'INDEX', 'FOREX', 'ETF'))
    processed = 0
    for _, sym_ in common.persistent_generate(syms_,
                                              "symbols_sectors_update.txt"):
        try:
            catg = yf.Ticker(sym_).info['category']
        except (KeyError, http.client.HTTPException):
            pass
        try:
            sect = yf.Ticker(sym_).info['sector']
        except (KeyError, http.client.HTTPException):
            pass
        cur.execute(query, [catg, sect, sym_])
        conn.commit()
        logger.info(
            "Category (%s) or sector (%s) updated for symbol %s."
            " %d/%d", catg, sect, sym_, [*syms_].index(sym_), len(syms_))
        processed += 1
        if processed >= 200:
            break
    logger.info("The daily sector update was finished (%d instruments "
                "updated).", processed)


def write_forex_to_ibdb(data):
    """
    Writes euro-forex data into the database. Tables market_catalogs,
    time_series and symbols are modified.

    PARAMETERS
    ----------
    data : dataframe
        It contains the index.
    """
    data.drop('Unnamed: 42', axis=1, inplace=True)
    data.columns = [c + '-CUR' for c in data.columns]
    records = [[
        data.columns[c], '1 day', data.index[r], None, None, None,
        None if math.isnan(data.iloc[r, c]) else data.iloc[r, c], None, None,
        None, None
    ] for r in range(len(data)) for c in range(len(data.columns))
               if not math.isnan(data.iloc[r, c])]
    currencies = [[
        data.columns[c], None, 'Pending', 'FOREX', 'EURO FOREX', None, None,
        None
    ] for c in range(len(data.columns))]
    catalog = [['euro forex', data.columns[c], None]
               for c in range(len(data.columns))]
    conn = connection_tunnel()
    cur = conn.cursor()
    cur.executemany(
        """
        INSERT INTO symbols
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (symbol) DO NOTHING;
        """, currencies)
    conn.commit()

    cur.executemany(
        """
        INSERT INTO market_catalogs
        VALUES (%s, %s, %s)
        ON CONFLICT (catalog_name, symbol) DO NOTHING;
        """, catalog)
    conn.commit()
    psycopg2.extras.execute_batch(
        cur,
        """
        INSERT into time_series
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (symbol, bar_size, date_) DO NOTHING;
        """, records)
    conn.commit()


def ts_plot(ts_df_arg, fields, logger):
    """
    Plots the time series in the pandas dataframe ts_df.

    PARAMETERS
    ----------
    ts_df_arg : pd.dataframe
        Pandas data frame containing the time series. It is exactly as returned
        from funcion time_series.read_ts_from_ibdb.
    fields : list
        Column names in ts_df_arg to be ploted. Its elements should be in {
        open, high, low, close, adj_close, imf0, imf1, ... imf9}.
    logger :
        Program's logger.

    EXAMPLES
    --------
    >>> ts_plot(read_ts_from_ibdb(
    ...    'IBM', "1 day", None, None, last=1000, imf=True)[0], [
    ...    "open", "close", "volume", "high", "low", "imf1", "imf2", "imf3",
    ...    "imf4", "imf5", "imf6", "imf7", "imf8"],
    ...    common.define_logger('time_series.log'))

    """
    ts_df = ts_df_arg.reset_index()
    j = 0
    try:
        dates_np = np.squeeze(
            np.asarray(pd.DataFrame(ts_df['base_date']).to_numpy()))
        plt.figure(figsize=(12, 9))
        for field in fields:
            field_np = np.squeeze(
                np.asarray(pd.DataFrame(ts_df[field]).to_numpy()))
            plt.subplot(6, 1, j + 1)
            plt.plot(dates_np, field_np, 'b')
            plt.ylabel(field)
            plt.locator_params(axis='y', nbins=5)
            plt.xlabel("Time")
            plt.tight_layout()
            if j >= 5:
                plt.figure(figsize=(12, 9))
                j = 0
            j += 1
    except KeyError:
        logger.error("Error in some field in " + str(fields) +
                     " for the columns of the df " + str(ts_df.columns))
    else:
        plt.show()


def symbol_sec_type(symbol):
    """
    Returns the symbol sector type.

    Parameters
    ----------
    symbol : str
        Name of the symbol.

    Returns
    -------
        symbol.sec_type where symbol = %(symbol)s

    """
    conn = connection_tunnel()
    cur = conn.cursor()

    cur.execute(
        """
        select sectype
        from symbols
        where symbol = %(symbol)s;
        """, {'symbol': symbol})
    conn.commit()
    result = cur.fetchone()
    if result is None:
        return 'STK'
    return result[0]


def get_adj_close(symbol, bar_size, date):
    """
    Get the adj_close price for the symbol at base_date.

    Parameters
    ----------
    symbol : str
        Name of the financial instrument.
    bar_size : int
        Bar size.
    date : datetime.pyi
        Date to retrieve the price

    Returns
    -------
        The required price or None.
    """
    conn = connection_tunnel()
    cur = conn.cursor()
    cur.execute(
        """
        select adj_close from time_series
        where symbol = %(symbol)s and bar_size = %(bar_size)s and
              date_ = %(base_date)s;
        """, {
            'symbol': symbol,
            'bar_size': bar_size,
            'base_date': date
        })
    answer = cur.fetchone()
    conn.commit()
    if answer is None:
        return None
    return answer[0]


def ts_export(logger):
    """
    Export all time series into time_series to dropbox and GitHub.

    Parmeters
    ---------
    logger :
        Program's logger.
    """
    # Stores all time series into dropbox directory for interchange.
    write_tss_to_fs(['1 hour', '1 day', '1M', '1Q'], [
        'Industry Performance', 'sp500', 'nasdaq', 'dowjones',
        'BofA Merrill Lynch Total Bond Return Index Values', 'euro forex',
        'Volatility Index', 'Stock Market Index', 'Interests Rates'
    ], logger)


def main():
    """
    Main program.
    """
    logger = common.define_logger('time_series.log')

    parser = argparse.ArgumentParser(
        "Time series processing program. Under no program option the daily"
        "time series backup process is run")
    parser.add_argument("-s",
                        "--sectors_update",
                        help="Updates sector/category attributes in "
                        "symbols",
                        action='store_true')
    args = parser.parse_args()

    # if args.sectors_update:
    #     time_exec = timestamp_executions.ProgramExecution(
    #         'time_series_sectors', 24 * 60 * 7)
    #     time_exec.start()
    #     symbols_sectors_update(logger)
    # else:
    #     time_exec = timestamp_executions.ProgramExecution(
    #         'time_series_export', 24 * 60)
    #     time_exec.start()
    #     ts_export(logger)
    # time_exec.end()


if __name__ == "__main__":
    main()
