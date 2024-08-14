import pandas as pd


def linear_interpolate(last_interpolated_reading, reading1, reading2, offset):
    time_gap = reading2['timestamp'] - reading1['timestamp']
    dist_from_reading_1 = last_interpolated_reading['timestamp'] - reading1['timestamp']
    if dist_from_reading_1 >= 0:
        offset = offset + dist_from_reading_1
    else:
        offset = 30 + dist_from_reading_1
    mu = offset / time_gap
    new_timestamp = last_interpolated_reading['timestamp']+30
    new_x = round(reading1['x'] * (1 - mu) + reading2['x'] * mu, 3)
    new_y = round(reading1['y'] * (1 - mu) + reading2['y'] * mu, 3)
    new_z = round(reading1['z'] * (1 - mu) + reading2['z'] * mu, 3)
    d = {'timestamp': new_timestamp, 'x': new_x, 'y': new_y, 'z': new_z}
    return pd.DataFrame(data=d, index=[0])
