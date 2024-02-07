import numpy as np

###### Pensando el current_day como el número de días, en un intervalo de 1 a 64
def moving_average(price, current_day, window_size):
    window = price[(current_day - window_size): current_day]
    return sum(window) / window_size

def exponential_moving_average(price, alpha, current_day):
    ema = price[0]  # Inicializar el primer valor del EMA como el primer dato
    for t in range(1, current_day):
        ema = alpha * price[t] + (1 - alpha) * ema  # Fórmula del EMA
    return ema

