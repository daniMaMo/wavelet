"""
January 21st, 2022
CDMX
F. Sagols.

Trading in Interactive Brokers Project
Common tools for any algorithmic trading project.
"""

import glob
import json
import subprocess
import sys
import re
import holidays
import math
import logging
import select
from logging.handlers import TimedRotatingFileHandler
import numpy as np
from os import path
import os
import datetime
from datetime import timedelta
from itertools import chain, combinations
import pandas as pd


class Error:
    """ Error management

    ATTRIBUTES
    ----------
    errors_list : list
        errors_list[0] is the last errors list. That is the list of errors
        reported from the last invocation to reset_last.
        errors_list[1] is the global errors list. That is, all the errors
        reported along the session.
    attended_errors : list
        Errors codes that the system takes care of.
    unattended_errors: set
        List of errors happened along this session that the program did not
        process.

    METHODS
    -------
    reset_last:
        Reset the last errors list
    reset_global:
         Reset the global errors list
    report_error:
        Keep track of an error in last errors and global errors lists
    dump_last:
        Send to the standard output the last errors list
    dump_global:
        Send to the standard output the global errors list.
    """
    def __init__(self, attended_errors, logger):
        """ Class constructor

        PARAMETERS
        ----------
        attended_errors :
            List of error numbers that the class manages.
        logger :
            System logger. To dispatch error and info messages.
            See https://docs.python.org/3/howto/logging.html
        """
        self.logger = logger
        self.errors_list = [{}, {}]
        # See https://interactivebrokers.github.io/tws-api/message_codes.html
        self.attended_errors = attended_errors
        self.unattended_errors = set()

    def reset_last(self):
        """ Reset last errors list.
        """
        self.errors_list[0] = {}

    def reset_global(self):
        """ Reset global errors list.
        """
        self.errors_list[1] = {}

    def report_error(self, req_id, error_code, error_str):
        """ Report an error into the last and global errors lists.

        PARAMETERS
        ----------
        req_id : int
            Requirement identifier of the IB operation that produced the error.
        error_code : int
            IB identifier of the error.
        error_str : str
            Text description of the error.
        """
        for e_l in [0, 1]:
            if error_code in self.errors_list[e_l]:
                self.errors_list[e_l][error_code]['count'] += 1
                self.errors_list[e_l][error_code]['req_ids'].add(req_id)
            else:
                self.errors_list[e_l][error_code] = \
                    {'string': error_str, 'count': 1, 'req_ids': {req_id}}
        if error_code not in self.attended_errors:
            self.unattended_errors.add(error_code)

    def dump_last(self):
        """ Dumps the last error list to the standard output.
        """
        self.logger.error("Last errors dictionary")
        for code in [*self.errors_list[0]]:
            self.logger.error(f"Code: {code}: {self.errors_list[1][code]}")

    def dump_global(self):
        """ Dumps the global error list to the standard output.
        """
        self.logger.error("Global errors dictionary")
        for code in [*self.errors_list[1]]:
            self.logger.error(f"Code: {code}: {self.errors_list[1][code]}")
        print(f"Unattended errors: {self.unattended_errors}")


def powerset(iterable):
    """
    Computes the power set of one iterable.

    Parameters
    ----------
    iterable : iterable
        Representation of some set.

    Returns
    -------
        Iterable power set
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    set_size = list(iterable)
    return chain.from_iterable(
        combinations(set_size, comb_size)
        for comb_size in range(len(set_size) + 1))


def assign_tasks(tasks, processor, processors):
    """
    Given a list of tasks this method returns the sub-list of tasks that should
    be performed by processor number 'processor'. The total number of
    processors is 'processors'. The assignation is fair.

    Parameters
    ----------
    tasks : list
        Tasks to be performed
    processor : int
        Number of processor.
    processors : int
        Total number of processors

    Returns
    -------
        The sub-lists of tasks for 'processor''.
    """
    tasks_for_processor = math.ceil(len(tasks) / processors)
    start = tasks_for_processor * processor
    end = start + tasks_for_processor
    return tasks[start:end]


class Dates:
    """
    Dates iterator. You define an initial and final base_date, and it iterates
    the time period in one day steps.
    """
    def __init__(self, initial_date, end_date, step=1):
        """
        Creates a dates iterator.

        PARAMETERS
        -----------
        initial_date : datetime.pyi
            Initial
        end_date : datetime.pyi
            End base_date.
        """
        assert initial_date is not None, "initial_date cannot be None"
        self.step = step
        if initial_date and end_date:
            assert initial_date <= end_date, \
                'Starting base_date must be lower or equal than the ending one'
        self.initial_date = initial_date
        self.end_date = end_date
        if not end_date or (initial_date and end_date):
            self.date = self.initial_date
        else:
            self.date = self.end_date

    def __iter__(self):
        """
        Returns the initial base_date
        """
        return self

    def __next__(self):
        """
        Returns the next base_date
        """
        aux = self.date
        if self.date > self.end_date:
            raise StopIteration
        self.date += timedelta(days=self.step)
        return aux.replace(hour=0, minute=0, second=0, microsecond=0)


def date_in_spanish(date):
    """
    Translates a string base_date to spanish. That is, all references to months
    abbreviations like 'Jan', 'Feb', 'Mar' and so on are changed to 'Ene',
    'Feb', 'Mar', respectively.

    Parameters
    ----------
    date : str
        Date to be translated.

    Returns
    ------
        str
        The translated base_date.

    Examples
    --------
    >>> date_in_spanish("23-Apr-2021")
    23-Abr-2021
    >>> date_in_spanish("Dec-24-2020")
    Dic-24-2020
    """
    month_trans = {
        'Jan': 'Ene',
        'Feb': 'Feb',
        'Mar': 'Mar',
        'Apr': 'Abr',
        'May': 'May',
        'Jun': 'Jun',
        'Jul': 'Jul',
        'Aug': 'Ago',
        'Sep': 'Sep',
        'Oct': 'Oct',
        'Nov': 'Nov',
        'Dec': 'Dic'
    }
    for month, value in month_trans.items():
        date = re.sub(month, value, date)
    return date


def play_beep():
    """ Plays an alert sound.
        It is used when a fatal error condition is found, o when the connection
        to TWS is broken and the program is restarted.
    """
    os.system('play -nq -t alsa synth 1 sine 180')


def ensure_path_existence(test_path):
    """
    Given a valid file system path this method verifies its existence.
    Otherwise, it creates the path. If we want to prove the existence of a
    directory we should append a '/' at the end of test_path.

    PARAMETERS
    ----------
    test_path : str
        The path referred above.

    Examples
    --------
    >>> ensure_path_existence(
    ... "./ib_activity_reports/DU2876788/DU2876788_20210215_20210219.csv")
    True
    >>> ensure_path_existence(
    ... "./logs/")
    True
    """
    elements = test_path.split("/")
    if not elements:
        raise ValueError("Empty path detected")

    sub_path = ''
    for index, element in enumerate(elements):
        sub_path = sub_path + element + '/'
        if index == len(elements) - 1:
            sub_path = sub_path[0:-1]
            if element == '':
                return True
            return os.path.isfile(sub_path)
        if sub_path != '' and not path.exists(sub_path):
            # print("%s does not exist. Creating it." % sub_path)
            os.mkdir(sub_path)
    return True


def check_date_validity(date):
    """
    Check if a string format is valid.

    PARAMETERS
    ----------
    date : str
        Date in string format: "YYYY-MM-DD'
    """
    year, month, day = date.split("-")
    try:
        datetime.datetime(int(year), int(month), int(day))
    except ValueError:
        return False
    return True


def monday_and_friday(date):
    """
    Returns the monday and friday dates of the week containing base_date.

    Parameters
    ----------
    date : datetime.pyi

    Examples
    --------
    >>> str(monday_and_friday(datetime.datetime(2021, 11, 28)))
    '(datetime.datetime(2021, 11, 29), datetime.datetime(2021, 12, 3, 0, 0))'
    >>> str(monday_and_friday(datetime.datetime(2021, 11, 29)))
    '(datetime.datetime(2021, 11, 29), datetime.datetime(2021, 12, 3, 0, 0))'
    >>> str(monday_and_friday(datetime.datetime(2021, 11, 30)))
    '(datetime.datetime(2021, 11, 29), datetime.datetime(2021, 12, 3, 0, 0))'
    >>> str(monday_and_friday(datetime.datetime(2021, 12, 1)))
    '(datetime.datetime(2021, 11, 29), datetime.datetime(2021, 12, 3, 0, 0))'
    >>> str(monday_and_friday(datetime.datetime(2021, 12, 2)))
    '(datetime.datetime(2021, 11, 29), datetime.datetime(2021, 12, 3, 0, 0))'
    >>> str(monday_and_friday(datetime.datetime(2021, 12, 3)))
    '(datetime.datetime(2021, 11, 29), datetime.datetime(2021, 12, 3, 0, 0))'
    >>> str(monday_and_friday(datetime.datetime(2021, 12, 4)))
    '(datetime.datetime(2021, 11, 29), datetime.datetime(2021, 12, 3, 0, 0))'

    Returns
    -------
         (datetime.pyi, datetime.pyi)
         The required monday and friday dates.
    """
    day_of_week = date.weekday()
    if day_of_week == 6:
        day_of_week = -1
    monday = date - datetime.timedelta(days=day_of_week)
    friday = date + datetime.timedelta(days=4 - day_of_week)
    return monday, friday


def first_day_in_period(period, date):
    """
    Returns the base_date of the first day in period if 'base_date' is the last
    business day in period. Otherwise, it returns None. For instance, if the
    period is "daily" it returns 'base_date' if it is a business day, otherwise
    returns None. If period is "weekly" and 'base_date' is a friday it returns
    the previous monday base_date.
    If period is "monthly" and 'base_date' is the last business day in period
    then it returns the base_date of the first monday in the month.

    PARAMETERS
    ----------
    period : str
        Name of the period. It could be "daily", "weekly" or "monthly".
    date : datetime.pyi
        Date to be considered.

    RETURNS
    -------
        datetime.pyi
        The first base_date in the period ending in 'base_date'.

    EXAMPLES
    --------
    >>> first_day_in_period(
    ...    'monthly', datetime.datetime(2021, 10, 30))

    >>> first_day_in_period(
    ...    'weekly', datetime.datetime(2021, 5, 18))

    >>> first_day_in_period(
    ...    'weekly', datetime.datetime(2021, 7, 1))

    >>> first_day_in_period(
    ...    'daily', datetime.datetime(2021, 9, 19))

    >>> first_day_in_period(
    ...    'monthly', datetime.datetime(2021, 8, 16))

    >>> first_day_in_period(
    ...    'monthly', datetime.datetime(2021, 8, 31)).strftime("%Y%m%d")
    '20210802'
    >>> first_day_in_period(
    ...    'weekly', datetime.datetime(2021, 8, 20)).strftime("%Y%m%d")
    '20210816'
    >>> first_day_in_period(
    ...    'weekly', datetime.datetime(2021, 3, 23))

    >>> first_day_in_period(
    ...    'daily', datetime.datetime(2021, 8, 15))

    >>> first_day_in_period(
    ...    'daily', datetime.datetime(2021, 8, 16)).strftime("%Y%m%d")
    '20210816'
    """
    if date.weekday() > 4:
        return None
    if period == 'daily':
        return date
    if period == 'weekly':
        if date.weekday() == 4:
            return date + datetime.timedelta(days=-4)
        return None
    next_day = date
    while True:
        next_day += datetime.timedelta(days=1)
        if next_day.day > date.day and next_day.weekday() < 5:
            return None
        if next_day.day <= date.day:
            first_day = date
            first_day = first_day.replace(day=1)
            if first_day.weekday() < 5:
                return first_day
            if first_day.weekday() == 5:
                return first_day + datetime.timedelta(days=2)
            return first_day + datetime.timedelta(days=1)


def next_date(last_date, periodicity, day, time_):
    """
    From last base_date this method looks for the next base_date in which the
    time periodicity is met. For instance, if last_date where '2021-01-08' and
    the periodicity where 'monthly', the day 23, and the time_ 8:30 then the
    next base_date we are looking for would be '2021-01-23 08:30:00'. If it were
    '2021-01-24' then the next base_date would be '2021-02-23 08:30:00'.

    PARAMETERS
    ----------
    last_date : class datetime.datetime
        The last base_date.
    periodicity : str
        The time periodicity. Possible values are 'daily', 'weekly', 'monthly',
        'yearly', 'never'.
    day : integer
        Number of the day into the period selected:
        For 'daily it has no meaning.
        For 'weekly' it is the day of the week. (0-sunday, 1-monday and so on).
        For 'monthly' it is the day of the month (starting in 1).
        For 'yearly' ot is the day in the year (starting in 1).
    time_ : class datetime.datetime
        It is the hour.
    """
    if periodicity == 'never':
        return datetime.datetime(3000, 1, 1)
    # last base_date day for week
    ld_day_of_week = int(last_date.strftime("%w"))
    ld_day_of_year = int(last_date.strftime("%j"))
    if periodicity == 'daily':
        # Last base_date excess from start
        ld_offset = datetime.timedelta(hours=last_date.hour,
                                       minutes=last_date.minute,
                                       seconds=last_date.second)
        # excess from start for period
        p_offset = datetime.timedelta(hours=time_.hour,
                                      minutes=time_.minute,
                                      seconds=time_.second)
        result = last_date - ld_offset + p_offset
        if ld_offset >= p_offset:
            result = result + datetime.timedelta(days=1)
    elif periodicity == 'weekly':
        ld_offset = datetime.timedelta(days=ld_day_of_week,
                                       hours=last_date.hour,
                                       minutes=last_date.minute,
                                       seconds=last_date.second)
        p_offset = datetime.timedelta(days=day,
                                      hours=time_.hour,
                                      minutes=time_.minute,
                                      seconds=time_.second)
        result = last_date - ld_offset + p_offset
        if ld_offset >= p_offset:
            result = result + datetime.timedelta(days=7)
    elif periodicity == 'monthly':
        ld_offset = datetime.timedelta(days=last_date.day,
                                       hours=last_date.hour,
                                       minutes=last_date.minute,
                                       seconds=last_date.second)
        p_offset = datetime.timedelta(days=day,
                                      hours=time_.hour,
                                      minutes=time_.minute,
                                      seconds=time_.second)
        result = last_date - ld_offset + p_offset
        if ld_offset >= p_offset:
            year = result.year
            month = result.month
            day = result.day
            if month < 12:
                month += 1
            else:
                month = 1
                year += 1
            while not check_date_validity(
                    str(year) + '-' + str(month) + '-' + str(day)):
                day -= 1
            result = result.replace(year=year, month=month, day=day)
        elif day >= 28 and result.day <= 4:
            while result.day <= 4:
                result = result - datetime.timedelta(days=1)
    elif periodicity == 'yearly':
        ld_offset = datetime.timedelta(days=ld_day_of_year,
                                       hours=last_date.hour,
                                       minutes=last_date.minute,
                                       seconds=last_date.second)
        p_offset = datetime.timedelta(days=day,
                                      hours=time_.hour,
                                      minutes=time_.minute,
                                      seconds=time_.second)
        result = last_date - ld_offset + p_offset
        if ld_offset >= p_offset:
            result = result.replace(year=result.year + 1)
    else:
        raise ValueError(f"Wrong periodicity value: {periodicity}")

    return result


def previous_business_days(date, days=1):
    """
    Returns the date before a given number of business days.

    Parameters
    ----------
    date : datetime.pyi
        Reference date
    days : int
        Number of business days before.

    Returns
    -------
        date
        The required date.

    Examples
    --------
    >>> previous_business_days(datetime.datetime(2022, 9, 19), 5)
    datetime.datetime(2022, 9, 12, 0, 0)
    >>> previous_business_days(datetime.datetime(2022, 9, 19))
    datetime.datetime(2022, 9, 16, 0, 0)
    """
    us_holidays = holidays.UnitedStates()
    local_date = date.replace(hour=0,
                              minute=0,
                              second=0,
                              microsecond=0)
    while days > 0:
        local_date -= datetime.timedelta(days=1)
        if local_date not in us_holidays and local_date.weekday() not in [5, 6]:
            days -= 1
    return local_date


def is_today_a_business_day(date):
    """
    Returns true if and only if today is a holiday.

    Parameters
    ----------
    date : datetime.py
        Date to be tested.

    Examples
    --------
    >>> is_today_a_business_day(datetime.datetime.today)
    True
    """
    us_holidays = holidays.UnitedStates()
    local_date = date.replace(hour=0,
                              minute=0,
                              second=0,
                              microsecond=0)
    return not (local_date in us_holidays or date.weekday() in [5, 6])


def get_answer(prompt) -> str:
    """
    Reads a user answer.

    Parameters
    ----------
    prompt : str
        Message displayed to the user.

    Returns
    -------
        str
        User's answer
    """
    return input(prompt)


def get_yes_no_answer(prompt):
    """
    Makes a question to the user and waits for a yes/no answer.

    PARAMETERS
    ----------
    prompt: str
        Questions to be asked to the user

    Returns
    -------
        str
        The 'yes' or 'no' answer. In lower case.
    """
    while True:
        answer = input(prompt).lower()
        if answer not in ['yes', 'no']:
            print("Answer 'yes' or 'no' please.")
            continue
        break
    return answer


def input_with_timeout(message, auto_response, timeout=60):
    """ Wait some time for a user answer. If it does not arrive after
    the timeout then an automatic response is yield.

    PARAMETERS
    ----------
    message : str
        Text displayed at standard output.
    auto_response : str
        Automatic response yield when the user did not answer anything.
    timeout : integer
        Seconds to wait before the automatic answer is generated.
    """
    print(message)
    user_response, _, _ = select.select([sys.stdin], [], [], timeout)
    if user_response:
        return sys.stdin.readline()
    print(f"Since you did not answer we assume '{auto_response}'.")
    return auto_response


def incrementing_filename(base, ext):
    """
    Returns the string base+{cons}+'.'+extension of the next available filane in
    the file system where 'cons' is the lowest possible integer.
    Parameters
    ----------
    base : str
        First part of the file name.
    ext : str
        File name extension.

    Returns
    -------
        str
        The required file name.

    Examples
    --------
    >>> incrementing_filename('./pickle/20210104-20210212', 'pck')
    './pickle/20210104-20210212-1.pck'
    """
    ind = 1
    while True:
        file_name = base + '-' + str(ind) + '.' + ext
        if not os.path.exists(file_name):
            break
        ind += 1
    return file_name


def define_logger(logger_file):
    """
    Returns a logger. See module logging.
    The path '.logs' must exit. Otherwise, it is created.

    Parameters
    ----------
    logger_file : str
        File name of the logger
    Returns
    -------
        logger
    """
    ensure_path_existence('./logs/')
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    log_handler = TimedRotatingFileHandler('./logs/' + logger_file,
                                           when="midnight")
    log_formatter = logging.Formatter(
        '%(asctime)s-%(name)s-%(levelname)s-%(pathname)s-%(lineno)s-  '
        '%(message)s')
    log_handler.setFormatter(log_formatter)
    logger = logging.getLogger(logger_file)
    logger.addHandler(log_handler)
    logger.addHandler(console)
    logger.setLevel(logging.INFO)
    return logger


class JSONLocalStorage:
    """
    This class is for storing json data that doesn't fit in IBdb
    (e.g. a list of the best parameters for a model)
    """
    def __init__(self, relative_file_path):
        """
        Set the path where it will store the json file
        PARAMETERS
        ----------
        relative_file_path: str
            File's desired relative path
        """
        self.relative_file_path = relative_file_path

    def path_exists(self):
        """
        Check if the file exists already
        RETURNS
        -------
        Boolean
        """
        return path.exists(self.relative_file_path)

    def write(self, dictionary):
        """
        Overwrites the existing data
        PARAMETERS
        -------
        dictionary: dict
            Python dictionary to be stored
        """
        with open(self.relative_file_path, 'w') as f_write:
            json.dump(dictionary, f_write)
        pass

    def read(self):
        """
        Reads existing data
        RETURNS
        -------
        The stored dictionary if the file exists or an empty dictionary if not.
        """
        if self.path_exists():
            with open(self.relative_file_path, 'r') as f_read:
                load = json.load(f_read)
            return load
        else:
            return {}


class CloserMarkerJSONMixin:
    """Use a class that writes json files to mark them,
     so it can be known when a process has been completed"""
    def is_closed(self):
        """
        Check if there is a closing mark in the file.
        RETURNS
        -------
        bool
        """
        if super().path_exists():
            return False if super().read().get('closed') is None else True
        else:
            return False

    def close(self):
        """Marks the file as closed"""
        load = super().read()
        load['closed'] = True
        self.write(load)

    def read(self):
        """
        Deletes closed mark from the retrieved data
        RETURNS
        -------

        """
        load = super().read()
        if self.is_closed():
            del load['closed']
            return load
        return load


class RuntimeJSONTracker(CloserMarkerJSONMixin, JSONLocalStorage):
    """
    No documentation available
    """
    pass


class PendingWork:
    """
    Any method could use this procedure to store pending work information in
    some specific file into the './pending_work' directory. It is used when
    a program crashes to continue just in the point in which the crash
    occurred.
    """
    def __init__(self, file_name):
        """
        Returns a PendingWork class instance where the recover information
        will be saved into './pending_work/{file_name}

        PARAMETERS
        ----------
        file_name : str
            File to store the pending work.
        """
        ensure_path_existence("./pending_work/")
        self._file_name = file_name

    def write(self, data):
        """
        Stores recovery data.
        :param data: str
            Recovery data
        :return: None
        """
        with open('./pending_work/' + self._file_name, 'w',
                  encoding='utf-8') as f_out:
            f_out.write(data)

    def read(self):
        """
        Reads the recovery data.

        RETURNS
        -------
        None if the file does not exist, otherwise the recovery data.
        """
        if not path.exists('./pending_work/' + self._file_name):
            return None
        with open('./pending_work/' + self._file_name, 'r',
                  encoding='utf-8') as f_open:
            data = f_open.readline()
            if data != '' and data[-1] == '\n':
                data = data[:len(data) - 1]
        return data

    def file_name(self):
        """
        Returns
        -------
            str
            The class instance storage file name.
        """
        return './pending_work/' + self._file_name

    def clear(self):
        """
        Deletes the file associated to the class instance.
        :return: None
        """
        if path.exists('./pending_work/' + self._file_name):
            os.system('rm ' + './pending_work/' + self._file_name)
        else:
            raise FileExistsError("File " + './pending_work/' +
                                  self._file_name + " does not exist.")


def persistent_generate(elements, name):
    """
    This generator is used to download time series but the program is prone to
    crash, and it is a waste of time to start from the beginning. To allow
    recovery between crashes a PendingWork class instance is used.

    PARAMETERS
    ----------
    elements : list
        Elements to be generated (i.e. time series names).
    name : str
        File name of the PendingWork instance.
        Carefully avoid the use of the name of other PendingWork instance with
        the same name.

    """
    pending_work = PendingWork(name)
    initial_element = pending_work.read()
    for ind, element in enumerate(elements):
        if initial_element is not None and \
           not isinstance(initial_element, datetime.date) and \
           isinstance(element, datetime.date):
            initial_element = datetime.datetime.strptime(
                initial_element, "%Y-%m-%d %H:%M:%S")
        if initial_element is not None and initial_element != element:
            continue
        initial_element = None
        pending_work.write(str(element))
        yield ind, element
    pending_work.clear()


def delete_files_on_path(path_, file_name_pattern, days):
    """
    Delete the files in path which are older than a specific number of days.

    Parameters
    ----------
    path_ : str
        Path to look for files.
    file_name_pattern :
        Text pattern to locate files.
    days :
        Files older than this number of days are deleted.

    Returns
    -------
        The list of deleted files names.
    """
    old_directory = os.getcwd()
    os.chdir(path_)
    ref_date = datetime.datetime.today().replace(
        hour=0, minute=0, second=0, microsecond=0) - \
        datetime.timedelta(days=days)
    logs_fs = glob.glob(file_name_pattern)
    deleted_files = []
    for log in logs_fs:
        creation_date = datetime.datetime.fromtimestamp(os.path.getctime(log))
        if creation_date < ref_date:
            subprocess.run(['rm', log])
            deleted_files += [log]
    os.chdir(old_directory)
    return deleted_files


def ib_option_name(symbol, expire, strike, right):
    """
    Yields a string (using Interactive Brokers style) with the name of the
    option in the parameters.

    Parameters
    ----------
    symbol : str
        Option underlying.
    expire : datetime.pyi
        Expiration base_date.
    strike : float
        Option strike
    right : str
        Put ('P') or call ('C').

    Returns
    -------
        str
        An Interactive Brokers style text string with the option name.

    Examples
    --------
    >>> ib_option_name('ALB',datetime.datetime(2021, 2, 5),150.0,'C')
    'ALB 05FEB21 150.0 C'
    >>> ib_option_name('ALB',datetime.datetime(2021, 2, 19),150.0,'P')
    'ALB 19FEB21 150.0 P'
    >>> ib_option_name('D',datetime.datetime(2021, 2, 19),65.2,'C')
    'D 19FEB21 65.2 C'
    >>> ib_option_name('AAPL',datetime.datetime(2021, 10, 10),150.0,'P')
    'AAPL 10OCT21 150.0 P'
    """
    expire_str = expire.strftime("%d%b%y").upper().replace('ENE', 'JAN').\
        replace('ABR', 'APR').replace('AGO', 'AUG').\
        replace('DIC', 'DEC')
    symbol = symbol.replace('0', '').replace('1', '').replace('2', '').\
        replace('3', '').replace('4', '').replace('5', '').replace('6', '').\
        replace('7', '').replace('8', '').replace('9', '')
    return symbol + ' ' + expire_str + " " + f"{strike:.1f}" + " " + \
        right


def from_standard_equity_option_convention(code: str) -> dict:
    """
    Transform a standard equity option convention code to record representation.

    Parameters
    ----------
    code : str
        Standard equity option convention code (see
        https://en.wikipedia.org/wiki/Option_naming_convention).

    Returns
    -------
        dict
        A dictionary containing:
        'symbol': Symbol name
        'expire': Option expiration base_date
        'right': Put (P) or Call (C).
        'strike': Option strike

    Examples:
    >>> from_standard_equity_option_convention('YHOOC15041600030000')
    {'symbol': 'YHOO', 'expire': '20150416', 'right': 'C', 'strike': 30.0}
    """
    option = {}
    parts = re.search('([A-Z]+)([0-9]+)([CP])([0-9]+)', code)
    option['symbol'] = parts.group(1)
    expire = parts.group(2)
    option['expire'] = '20' + expire[0:2] + expire[2:4] + expire[4:6]
    option['right'] = parts.group(3)
    option['strike'] = float(parts.group(4)) / 1000.0
    return option


class NpEncoder(json.JSONEncoder):
    """
    This class is used iby 'json.dumps' to allow costumers type conversion.
    """
    def default(self, o):
        """
        Changes the codification of jason values when if needed.

        Parameters
        ----------
        o :
            python object to be transformed.

        Examples
        --------
        >>> json.dumps({'a': 5,
        ...             'b': datetime.datetime(2021, 3, 30, 8, 30)},
        ...            cls=NpEncoder)
        '{"a": 5, "b": "2021-03-30 08:30:00"}'
        """
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, datetime.datetime):
            return o.strftime("%Y-%m-%d %H:%M:%S")
        return super().default(o)  # super(common.NpEncoder, self).default(o)


# Very slow for many data points.  Fastest for many costs, most readable
def is_pareto_efficient_dumb(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is
    Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        is_efficient[i] = np.all(np.any(costs[:i] < c, axis=1)) and \
                          np.all(np.any(costs[i+1:] < c, axis=1))
    return is_efficient


# Fairly fast for many data points, less fast for many costs, somewhat readable
def is_pareto_efficient_simple(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is
    Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(
                costs[is_efficient] >= c,
                axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient


# Faster than is_pareto_efficient_simple, but less readable.
def is_pareto_efficient(costs, return_mask=True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(costs):
        nondominated_point_mask = np.any(costs >= costs[next_point_index],
                                         axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[
            nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(
            nondominated_point_mask[:next_point_index]) + 1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype=bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient


def map_range(int_0, int_1, x_0):
    """
    Given two intervals int_0 ant int_1 and a real number x_0 in int_0
    returns the real value x_1 such that (x_0 - int_0[0])/(int_0[1] - int_0[0])
    =  (x_1 - int_1[0])/(int_1[1] - int_1[0])

    Parameters
    ----------
    int_0 : tuple
        Containing the lower and upper interval bounds.
    int_1 : tuple
        Idem
    x_0 : float
        A real number in int_0

    Returns
    -------
        float
        x_1

    Examples
    --------
    >>> map_range((0.5, 1.0), (10.0, 20.0), 0.75)
    15.0

    >>> map_range((1.0, 4.0), (0.0, 100.0), 2.0)
    33.33333333333333
    """
    if x_0 < int_0[0]:
        x_0 = int_0[0]
    if x_0 > int_0[1]:
        x_0 = int_0[1]
    return ((x_0 - int_0[0])/(int_0[1] - int_0[0])) * (int_1[1] - int_1[0]) + \
        int_1[0]


def delayed_round(value, prev_value, delay):
    """
    It works like round but with the memory of the previos invocation result.

    Parameters
    ----------
    value : float
        Value to compute the delayed round.
    prev_value : int
        Previous computed dealyed value.
    delay : float
        Sub-interval around 0.5 in which the memory of the evaluation is
        considered. It mus be in [0, 0.5].

    Returns
    -------
        The delayed round of 'value'.

    Examples
    --------
    >>> delayed_round(5.1, 5, 0.2)
    5
    >>> delayed_round(5.2, 5, 0.2)
    5
    >>> delayed_round(5.3, 5, 0.2)
    5
    >>> delayed_round(5.4, 5, 0.2)
    5
    >>> delayed_round(5.5, 5, 0.2)
    5
    >>> delayed_round(5.6, 5, 0.2)
    5
    >>> delayed_round(5.7, 5, 0.2)
    5
    >>> delayed_round(5.8, 5, 0.2)
    6
    >>> delayed_round(5.9, 6, 0.2)
    6
    >>> delayed_round(6.0, 5, 0.2)
    6
    >>> delayed_round(5.1, 6, 0.2)
    5
    >>> delayed_round(5.2, 6, 0.2)
    6
    >>> delayed_round(5.3, 6, 0.2)
    6
    >>> delayed_round(5.4, 6, 0.2)
    6
    >>> delayed_round(5.5, 6, 0.2)
    6
    >>> delayed_round(5.6, 6, 0.2)
    6
    >>> delayed_round(5.7, 6, 0.2)
    6
    >>> delayed_round(5.8, 6, 0.2)
    6
    >>> delayed_round(5.9, 6, 0.2)
    6
    >>> delayed_round(6.0, 6, 0.2)
    6
    >>> delayed_round(6.3, 8, 0.2)
    6
    """
    assert 0 <= delay <= 0.5, 'Delay not in the appropriate range.'
    if prev_value is None:
        return round(value)
    min_value = int(value)
    possible_results = {min_value, min_value + 1}
    if min_value <= value < min_value + delay:
        return min_value
    if min_value + 1 - delay <= value:
        return min_value + 1
    if min_value + delay <= value <= min_value + 1.0 - delay and \
       prev_value in possible_results:
        return prev_value
    return round(value)


def ema_compute(series, alpha, window):
    """
    Computes the exponencial moving average of a time series with parameters
    alpha and window

    Returns
    -------
       DataFrame
       Containing the ema time series.
    """
    # CODE: To print full data frames
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.width', None)
    # pd.set_option('display.max_colwidth', -1)

    sma = series.rolling(window=window, min_periods=window).mean()[:window]
    rest = series[window:]
    # CODE: Pandas dataframes concatenation.
    # CODE: Exponential Weighted Mean.
    ewm1 = pd.concat([sma, rest]).ewm(alpha=alpha, adjust=False).mean()
    return ewm1
