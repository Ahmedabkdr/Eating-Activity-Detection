import numpy as np
from matplotlib.ticker import MultipleLocator
from numpy.fft import fftfreq
from scipy import signal

import pandas as pd
from matplotlib import pyplot as plt
from scipy.fft import fft
from scipy.signal import freqz, find_peaks

from interpolate import linear_interpolate


def slice_view(title, axis, df_median_filtered, df_low_pass, df_band_pass, df_interpolated):
    # create a figure and axis object
    fig, ax = plt.subplots()

    # plot the first data frame
    # ax.plot(df['timestamp'], df['x'], label='x')
    # ax.plot(df['timestamp'], df['y'], label='y')
    # ax.plot(df['timestamp'][-50:], df['z'][-50:], label='z')
    # plot the second data frame on the same axis
    # ax.plot(df_interpolated['timestamp'], df_interpolated['x'], label='x interpolated')
    # ax.plot(df_interpolated['timestamp'], df_interpolated['y'], label='y interpolated')
    # ax.plot(df_interpolated['timestamp'][-50:], df_interpolated['z'][-50:], label='z interpolated')
    # ax.plot(df_median_filtered['timestamp'], df_median_filtered['x'], label='x median')
    # ax.plot(df_median_filtered['timestamp'], df_median_filtered['y'], label='y median')
    ax.plot(df_median_filtered['timestamp'][-50:], df_median_filtered[f'{axis}'][-50:], label=f'{axis} median')
    ax.plot(df_low_pass['timestamp'][-50:], df_low_pass[f'{axis}'][-50:], label=f'{axis} low pass')
    ax.plot(df_band_pass['timestamp'][-50:], df_band_pass[f'{axis}'][-50:], label=f'{axis} band pass')
    # set major ticks every 3rd datapoint of df_interpolated
    major_ticks = df_interpolated['timestamp'][::6]
    ax.set_xticks(major_ticks)
    # set minor ticks every datapoint of df2
    # minor_ticks = df_interpolated['timestamp']
    # ax.set_xticks(minor_ticks, minor=True)
    ax.set_xlim([df_interpolated['timestamp'].iloc[-50], df_interpolated['timestamp'].iloc[-1]])
    ax.legend()
    ax.set_title(f'{title}: {axis.upper()}-Acceleration vs Time')
    ax.set_xlabel('Timestamp (ms)')
    ax.set_ylabel(f'{axis.upper()}-Axis Acceleration (milli-g)')
    ax.grid(axis='x', which='both')

    # show the plot
    fig.subplots_adjust(left=0.15)
    plt.show()


def full_view(title, axis, df_median_filtered, df_low_pass, df_band_pass, df_interpolated):
    # create a figure and axis object
    fig, ax = plt.subplots()

    # plot the first data frame
    # ax.plot(df['timestamp'], df['x'], label='x')
    # ax.plot(df['timestamp'], df['y'], label='y')
    # ax.plot(df['timestamp'], df['z'], label='z')
    # plot the second data frame on the same axis
    # ax.plot(df_interpolated['timestamp'], df_interpolated['x'], label='x interpolated')
    # ax.plot(df_interpolated['timestamp'], df_interpolated['y'], label='y interpolated')
    # ax.plot(df_interpolated['timestamp'], df_interpolated['z'], label='z interpolated')
    # ax.plot(df_median_filtered['timestamp'], df_median_filtered['x'], label='x median')
    # ax.plot(df_median_filtered['timestamp'], df_median_filtered['y'], label='y median')
    ax.plot(df_median_filtered['timestamp'], df_median_filtered[f'{axis}'], label=f'{axis} median')
    ax.plot(df_low_pass['timestamp'], df_low_pass[f'{axis}'], label=f'{axis} low pass')
    ax.plot(df_band_pass['timestamp'], df_band_pass[f'{axis}'], label=f'{axis} band pass')
    ax.set_xlim([df_interpolated['timestamp'].iloc[1], df_interpolated['timestamp'].iloc[-1]])
    # add minor gridlines
    ax.set_xticks(range(int(df_interpolated['timestamp'].iloc[0]), int(df_interpolated['timestamp'].iloc[-1]), 15000),
                  minor=True)
    ax.legend()
    ax.set_title(f'{title}: {axis.upper()}-Acceleration vs Time')
    ax.set_xlabel('Timestamp (ms)')
    ax.set_ylabel(f'{axis.upper()}-Axis Acceleration (milli-g)')
    ax.grid(axis='x', which='both')

    # show the plot
    fig.subplots_adjust(left=0.15)
    plt.show()


def plot_segments(segments, title):
    for segment_num, segment in enumerate(segments):
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))
        for i, col in enumerate(['x', 'y', 'z']):
            # Convert timestamp to seconds and subtract the first timestamp to get relative time
            t = (segment['timestamp'] - segment['timestamp'][0]) / 1000
            axes[i].plot(t, segment[col])
            axes[i].set_xlabel('Time (s)')
            axes[i].set_ylabel(col.upper() + '-Axis Acceleration (milli-g)')
            axes[i].set_xlim([0, 15])  # Set the x-axis limits to 0 to 15 seconds
            axes[i].xaxis.set_ticks(np.arange(0, 16, 5))  # Set major ticks at every 5 seconds
            axes[i].xaxis.set_ticks(np.arange(0, 16, 1), minor=True)  # Set minor ticks at every second
            axes[i].grid(which='both', axis='x')  # Show grid lines for both major and minor ticks
        axes[1].set_title(title + f" - Segment {segment_num + 1}")
        plt.tight_layout()  # Automatically adjust subplot parameters to give specified padding
        plt.show()


# segments data into 15 second segments with a 3-second overlap
def segment_data(dataframe, is_eating_df=None):
    window_size = 15000  # ms
    overlap = 3000
    sampling_period = 30
    datapoints_per_window = int(window_size / sampling_period)
    datapoints_per_overlap = int(overlap / sampling_period)
    segments = []
    is_eating_list = []
    index = 0
    # this while loop condition ensures that segments are created till the end of the dataframe and that the last
    # segment is not less than window_size long
    if is_eating_df is None:
        while index + datapoints_per_window < len(dataframe.index):
            new_segment = pd.DataFrame(dataframe.iloc[index:index + datapoints_per_window + 1])
            new_segment.reset_index(drop=True, inplace=True)
            segments.append(new_segment)
            index = index + (datapoints_per_window - datapoints_per_overlap)
        return segments
    else:
        index = 1
        is_eating_df_index = -1
        is_eating = 0
        while index + datapoints_per_window < len(dataframe.index):
            # while loop assumes there is always an accelerometer measurement at or after each is_eating timestamp
            while is_eating_df_index+1 < len(is_eating_df) and int(dataframe["timestamp"][index])+window_size/2 >= \
                    int(is_eating_df["timestamp"][is_eating_df_index+1]):
                is_eating_df_index = is_eating_df_index + 1
                is_eating = int(is_eating_df["is_eating"][is_eating_df_index])
            new_segment = pd.DataFrame(dataframe.iloc[index:index + datapoints_per_window + 1])
            new_segment.reset_index(drop=True, inplace=True)
            segments.append(new_segment)
            is_eating_list.append(is_eating)
            index = index + (datapoints_per_window - datapoints_per_overlap)
        return segments, is_eating_list


def extract_features(df_list):
    x_means = pd.DataFrame(columns=["timestamp", "x-mean"])
    x_variances = pd.DataFrame(columns=["timestamp", "x-variance"])
    x_skews = pd.DataFrame(columns=["timestamp", "x-skewness"])
    y_means = pd.DataFrame(columns=["timestamp", "y-mean"])
    y_variances = pd.DataFrame(columns=["timestamp", "y-variance"])
    y_skews = pd.DataFrame(columns=["timestamp", "y-skewness"])
    z_means = pd.DataFrame(columns=["timestamp", "z-mean"])
    z_variances = pd.DataFrame(columns=["timestamp", "z-variance"])
    z_skews = pd.DataFrame(columns=["timestamp", "z-skewness"])
    xy_covariances = pd.DataFrame(columns=["timestamp", "xy-covariance"])
    yz_covariances = pd.DataFrame(columns=["timestamp", "yz-covariance"])
    zx_covariances = pd.DataFrame(columns=["timestamp", "zx-covariance"])
    for i, dataframe in enumerate(df_list):
        timestamp = dataframe['timestamp'][0]
        x_means.loc[i] = [timestamp, round(dataframe['x'].mean(), 5)]
        x_variances.loc[i] = [timestamp, round(dataframe['x'].var(), 5)]
        x_skews.loc[i] = [timestamp, round(dataframe['x'].skew(), 5)]
        y_means.loc[i] = [timestamp, round(dataframe['y'].mean(), 5)]
        y_variances.loc[i] = [timestamp, round(dataframe['x'].var(), 5)]
        y_skews.loc[i] = [timestamp, round(dataframe['y'].skew(), 5)]
        z_means.loc[i] = [timestamp, round(dataframe['z'].mean(), 5)]
        z_variances.loc[i] = [timestamp, round(dataframe['x'].var(), 5)]
        z_skews.loc[i] = [timestamp, round(dataframe['z'].skew(), 5)]
        xy_covariances.loc[i] = [timestamp, round(dataframe['x'].cov(dataframe['y']), 5)]
        yz_covariances.loc[i] = [timestamp, round(dataframe['y'].cov(dataframe['z']), 5)]
        zx_covariances.loc[i] = [timestamp, round(dataframe['z'].cov(dataframe['x']), 5)]
    x_means["timestamp"] = x_means["timestamp"].astype(int)
    x_variances["timestamp"] = x_variances["timestamp"].astype(int)
    x_skews["timestamp"] = x_skews["timestamp"].astype(int)
    y_means["timestamp"] = y_means["timestamp"].astype(int)
    y_variances["timestamp"] = y_variances["timestamp"].astype(int)
    y_skews["timestamp"] = y_skews["timestamp"].astype(int)
    z_means["timestamp"] = z_means["timestamp"].astype(int)
    z_variances["timestamp"] = z_variances["timestamp"].astype(int)
    z_skews["timestamp"] = z_skews["timestamp"].astype(int)
    xy_covariances["timestamp"] = xy_covariances["timestamp"].astype(int)
    yz_covariances["timestamp"] = yz_covariances["timestamp"].astype(int)
    zx_covariances["timestamp"] = zx_covariances["timestamp"].astype(int)

    return x_means, y_means, z_means, x_variances, y_variances, z_variances, x_skews, y_skews, z_skews, \
        xy_covariances, yz_covariances, zx_covariances


def plot_spectrum(title, axis, df_median_filtered):
    # Extract the 'z' signal from the dataframe
    median_filtered_signal = df_median_filtered[f'{axis}']
    # Compute the Fourier transform of the signal
    n = len(median_filtered_signal)
    yf = np.fft.fft(median_filtered_signal)
    xf = np.linspace(0.0, 1.0 / (2.0 * (1 / 33.33)), n // 2)
    peaks, _ = find_peaks(2.0 / n * np.abs(yf[:n // 2]), distance=10, height=0.4)
    peak_frequencies = xf[peaks]
    # print("Peak Frequencies: ", peak_frequencies)
    # Plot the frequency spectrum
    fig, az = plt.subplots()
    az.plot(xf, 2.0 / n * np.abs(yf[:n // 2]))
    az.plot(xf[peaks], 2.0 / n * np.abs(yf[:n // 2])[peaks], "x")
    az.set_title(f'{title}: {axis.upper()}-Axis Frequency Spectrum')
    az.set_xlabel('Frequency (Hz)')
    az.set_ylabel('Amplitude')

    # Set the major ticks
    major_ticks = np.arange(0, xf.max(), 1)
    az.set_xticks(major_ticks)

    # Set the minor ticks
    minor_ticks = np.arange(0, xf.max(), 0.1)
    az.set_xticks(minor_ticks, minor=True)

    plt.show()


def accel_to_observations(accel_file, is_eating_file, title="Activity"):
    try:
        df_original = pd.read_csv(accel_file, names=["timestamp", "x", "y", "z"])
        df = df_original.iloc[0:1]
        is_eating_df = pd.read_csv(is_eating_file, names=["timestamp", "is_eating"])
        is_eating = []
        # ensure readings are roughly 30ms apart
        # drop readings with a timestamp within 5ms after the previous reading
        index = 1
        max_index = len(df_original.index)
        max_is_eating_index = len(is_eating_df.index)
        while index < max_index:
            print(df_original.loc[index, 'timestamp'])
            if df_original.loc[index, 'timestamp'] > df_original.loc[index - 1, 'timestamp'] + 5:
                df = pd.concat([df, df_original.iloc[index:index + 1]], ignore_index=True)
            index += 1

        df_interpolated = df.iloc[0:1]
        index = 1
        max_index = len(df.index)
        target_period = 30  # ms
        # note following loop fails (runs forever) if there is a huge time gap (disconnected from watch)
        while index < max_index:
            # number of seconds from the last interpolated datapoint to the datapoint of df at this index
            time_gap = df.loc[index, 'timestamp'] - df_interpolated.loc[len(df_interpolated) - 1, 'timestamp']

            # number of seconds from the interpolated datapoint to be generated, to the datapoint of df at this index
            offset = time_gap - target_period

            # if the offset is less than 0, the datapoint of df at this index comes before the point to be generated
            # interpolate between the next 2 points instead
            if offset < 0:
                index = index + 1

            # if the offset is greater than 0, the point to be generated does lie between the point at this index and
            # the point preceding it - interpolation proceeds
            elif offset > 0:
                df_interpolated = pd.concat([df_interpolated,
                                             linear_interpolate(
                                                 df_interpolated.loc[len(df_interpolated) - 1],
                                                 df.loc[index - 1],
                                                 df.loc[index],
                                                 offset
                                             )],
                                            ignore_index=True
                                            )
                # only if another point cannot be fit between the newly generated point and the datapoint of df at this
                # index, will the loop move on to the next point of df
                if offset < 30:
                    index = index + 1

            # if the offset is exactly 0, append the datapoint of df at this index to df_interpolated
            else:
                df_interpolated = pd.concat([df_interpolated, df.iloc[index:index + 1]], ignore_index=True)
                index = index + 1

        df_interpolated['timestamp'] = df_interpolated['timestamp'].astype(int)

        df_median_filtered = pd.DataFrame()
        df_median_filtered['timestamp'] = df_interpolated['timestamp']
        for col in df.columns[1:]:
            df_median_filtered[col] = df_interpolated[col].rolling(3, center=True).median()
        df_median_filtered.dropna(inplace=True)

        # plot_spectrum(title, "x", df_median_filtered)
        # plot_spectrum(title, "y", df_median_filtered)
        # plot_spectrum(title, "z", df_median_filtered)

        fc = 1  # Cut-off frequency of the filter
        fs = 33.33  # Sampling rate of the signal
        b, a = signal.butter(10, fc, 'low', fs=fs, output='ba')

        # Plot the frequency response.
        # w, h = freqz(b, a, fs=fs, worN=8000)
        # plt.subplot(2, 1, 1)
        # plt.plot(w, np.abs(h), 'b')
        # plt.plot(fc, 0.5 * np.sqrt(2), 'ko')
        # plt.axvline(fc, color='k')
        # plt.xlim(0, 0.5 * fs)
        # plt.title("Lowpass Filter Frequency Response")
        # plt.xlabel('Frequency [Hz]')
        # plt.xticks(np.arange(0, 0.5 * fs + 1, 1))
        # plt.gca().xaxis.set_minor_locator(MultipleLocator(0.5))
        # plt.grid(which='major', linestyle=':', linewidth='0.5', color='black')
        # plt.grid()

        df_low_pass = signal.filtfilt(b, a, df_median_filtered, axis=0)
        df_low_pass = pd.DataFrame(data=df_low_pass, columns=df_median_filtered.columns)
        df_low_pass['timestamp'] = df_interpolated['timestamp']
        df_low_pass['x'] = round(df_low_pass['x'], 3)
        df_low_pass['y'] = round(df_low_pass['y'], 3)
        df_low_pass['z'] = round(df_low_pass['z'], 3)

        f_low = 5  # Low cut-off frequency of the filter
        f_high = 10  # High cut-off frequency of the filter
        fs = 33.33  # Sampling rate of the signal
        order = 10  # Order of the filter

        b, a = signal.butter(order, [2 * f_low / fs, 2 * f_high / fs], 'bandpass', output='ba')

        # Plot the frequency response
        # w, h = freqz(b, a, fs=fs, worN=8000)
        # plt.subplot(2, 1, 1)
        # plt.plot(w, np.abs(h), 'b')
        # plt.plot([f_low, f_high], [0.5 * np.sqrt(2), 0.5 * np.sqrt(2)], 'ko')
        # plt.axvline(f_low, color='k')
        # plt.axvline(f_high, color='k')
        # plt.xlim(0, 0.5 * fs)
        # plt.title("Bandpass Filter Frequency Response")
        # plt.xlabel('Frequency [Hz]')
        # plt.xticks(np.arange(0, 0.5 * fs + 1, 1))
        # plt.gca().xaxis.set_minor_locator(MultipleLocator(0.5))
        # plt.grid(which='major', linestyle=':', linewidth='0.5', color='black')

        df_band_pass = signal.filtfilt(b, a, df_median_filtered, axis=0)
        df_band_pass = pd.DataFrame(data=df_band_pass, columns=df_median_filtered.columns)
        df_band_pass['timestamp'] = df_interpolated['timestamp']
        df_band_pass['x'] = round(df_band_pass['x'], 3)
        df_band_pass['y'] = round(df_band_pass['y'], 3)
        df_band_pass['z'] = round(df_band_pass['z'], 3)

        # Plotting
        # slice_view(title, "x", df_median_filtered, df_low_pass, df_band_pass, df_interpolated)
        # slice_view(title, "y", df_median_filtered, df_low_pass, df_band_pass, df_interpolated)
        # slice_view(title, "z", df_median_filtered, df_low_pass, df_band_pass, df_interpolated)
        full_view(title, "x", df_median_filtered, df_low_pass, df_band_pass, df_interpolated)
        full_view(title, "y", df_median_filtered, df_low_pass, df_band_pass, df_interpolated)
        full_view(title, "z", df_median_filtered, df_low_pass, df_band_pass, df_interpolated)

        # Segmentation
        median_filtered_segments, is_eating = segment_data(df_median_filtered, is_eating_df)
        lp_filtered_segments = segment_data(df_low_pass)
        bp_filtered_segments = segment_data(df_band_pass)
        # plot_segments(bp_filtered_segments[0:15], "Lowpass Filtered Data")

        x_means_mf, y_means_mf, z_means_mf, \
            x_variances_mf, y_variances_mf, z_variances_mf, \
            x_skew_mf, y_skew_mf, z_skew_mf, \
            xy_covariances_mf, yz_covariances_mf, zx_covariances_mf = extract_features(median_filtered_segments)

        x_means_lp, y_means_lp, z_means_lp, \
            x_variances_lp, y_variances_lp, z_variances_lp, \
            x_skew_lp, y_skew_lp, z_skew_lp, \
            xy_covariances_lp, yz_covariances_lp, zx_covariances_lp = extract_features(lp_filtered_segments)

        x_means_bp, y_means_bp, z_means_bp, \
            x_variances_bp, y_variances_bp, z_variances_bp, \
            x_skew_bp, y_skew_bp, z_skew_bp, \
            xy_covariances_bp, yz_covariances_bp, zx_covariances_bp = extract_features(bp_filtered_segments)

        features = [x_means_mf, y_means_mf, z_means_mf, x_variances_mf, y_variances_mf, z_variances_mf, x_skew_mf,
                    y_skew_mf, z_skew_mf, xy_covariances_mf, yz_covariances_mf, zx_covariances_mf, x_means_lp,
                    y_means_lp, z_means_lp, x_variances_lp, y_variances_lp, z_variances_lp, x_skew_lp, y_skew_lp,
                    z_skew_lp, xy_covariances_lp, yz_covariances_lp, zx_covariances_lp, x_means_bp, y_means_bp,
                    z_means_bp, x_variances_bp, y_variances_bp, z_variances_bp, x_skew_bp, y_skew_bp, z_skew_bp,
                    xy_covariances_bp, yz_covariances_bp, zx_covariances_bp]

        combined_df = pd.DataFrame()
        for feature in features:
            combined_df = pd.concat([combined_df, feature.iloc[:, 1]], axis=1)
        is_eating_column = {'is_eating': is_eating}
        is_eating_df = pd.DataFrame(is_eating_column)
        combined_df = pd.concat([combined_df, is_eating_df], axis=1)
        return combined_df
    except IOError:
        print("IOError - opening file")
