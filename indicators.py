import numpy as np

###### Pensando el current_day como el número de días, en un intervalo de 1 a 64
def moving_average(price, current_day, window_size):
    window = price[(current_day - window_size): current_day]
    return sum(window) / window_size

def exponential_moving_average(price, alpha, current_day, window_size):
    ema = price[-window_size]  # Initializes the first value of the EMA with the first date in the window
    for t in range(len(price[:current_day]) - window_size+1, current_day):
        ema = alpha * price[t] + (1 - alpha) * ema  # EMA formula
    return ema

def rsi(price, current_day, window_size):
    price = price[(current_day - window_size): current_day]
    U = []
    D = []

    diff_price = np.diff(price)

    U = np.append(U, diff_price[diff_price > 0])
    D = np.append(D, -diff_price[diff_price < 0])

    gain = U.mean()
    loss = D.mean()

    rs = gain / loss
    print(gain)
    print(loss)
    rsi = 100 - (100 / (1 + rs))

    return rsi

def moving_std(price, current_day, window_size):
    window_price = price[current_day - window_size: current_day]
    std = np.std(window_price)
    return std
def bollinger_bands(price, current_day, window_size, threshold):
    middle_band = moving_average(price, current_day, window_size)
    upper_band = middle_band + threshold * moving_std(price, current_day, window_size)
    lower_band = middle_band - threshold * moving_std(price, current_day, window_size)
    return upper_band, lower_band

def obv(prices, volumes, window_period, fixed_day):
    # Calculate price change and the sign of the change
    price_change = np.diff(prices)
    sign_change = np.sign(price_change)

    # Calculate OBV for the fixed day using the window period
    obv = volumes[fixed_day - window_period :fixed_day ] * sign_change[
                                                                 fixed_day - window_period - 1:fixed_day - 1]
    obv = np.sum(obv) / np.sum(volumes[fixed_day - window_period:fixed_day])

    return obv

def macd(price, current_day, short_window=12, long_window=26, signal_window=9):
    short_alpha = 2 / (short_window + 1)
    short_ema = exponential_moving_average(price, short_alpha, current_day, short_window)
    long_alpha = 2 / (long_window + 1)
    long_ema = exponential_moving_average(price, long_alpha, current_day, long_window)

    macd_line = short_ema - long_ema

    # Calculate Signal line (EMA of MACD line)
    signal_alpha = 2 / (signal_window + 1)
    signal_line = exponential_moving_average(macd_line, signal_alpha, current_day, signal_window)

    # Calculate MACD Histogram
    macd_histogram = macd_line - signal_line

    return macd_line, signal_line, macd_histogram

def stochastic_oscillator(close,current_day, window_size):  ### usually window_k = 14
    closes = close[current_day - (window_size+1):current_day]

    highest_close = np.max(closes)
    lowest_close = np.min(closes)

    current_close = closes[-1]

    porcent_k = (current_close - lowest_close) / (highest_close - lowest_close) * 100

    return porcent_k

def os(close,current_day, window_d, window_k): ### usually window_k= 14
    k = []
    for i in range(window_d):
        k.append(stochastic_oscillator(close, current_day-i, window_k))
    return stochastic_oscillator(close, current_day-i, window_size=14), np.sum(k)/window_d


