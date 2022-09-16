# Last change 2022.08.31
import numpy as np
import math
import matplotlib.pyplot as plt
import warnings
import pywt
from scipy import signal
from sklearn import preprocessing
import numpy.fft as fft

global zc_threshold, ssc_threshold, wap_threshold, fs_global
zc_threshold_global = 0.000003
ssc_threshold_global = 0.0000001
wap_threshold_global = 0.0001
fs_global = 1920


#   EMG features
def frequency_features_used(frequency_methods, methods):
    flag = 0
    for method_check in frequency_methods:
        if method_check in methods:
            flag = 1
            break
    return flag


def get_emg_features_from_examples(examples_emg_data, methods='all'):
    # methods=[a list]: 'all', 'mav', 'rms', 'wl', 'zc', 'ssc', 'kf', 'integrated', 'ssi', 'var', 'log', 'tm3', 'wap', 'mnf', 'mdf', 'rkf', 'mnp', 'ttp', 'sm1', 'sm2', 'sm3'
    examples_quantity = np.size(examples_emg_data, 0)
    emg_features = []
    for example_check in range(examples_quantity):
        emg_features = add_elements_in_list(list_name=emg_features, element_name=[get_emg_features_from_an_example(emg_data_channels=examples_emg_data[example_check], methods=methods)])
    return emg_features


def get_emg_features_from_an_example(emg_data_channels, methods='all', fs=fs_global):
    # methods=[a list]: 'all', 'mav', 'rms', 'wl', 'zc', 'ssc', 'kf', 'integrated', 'ssi', 'var', 'log', 'tm3', 'wap', 'mnf', 'mdf', 'rkf', 'mnp', 'ttp', 'sm1', 'sm2', 'sm3'
    emg_channels_quantity = np.min([np.size(emg_data_channels, 0), np.size(emg_data_channels, 1)])
    if 'all' in methods:
        methods = ['mav', 'rms', 'wl', 'zc', 'ssc', 'kf', 'integrated', 'ssi', 'var', 'log', 'tm3', 'wap', 'mnf', 'mdf', 'rkf', 'mnp', 'ttp', 'sm1', 'sm2', 'sm3']

    if 'NTDF' in methods:
        methods = ['LogAbsSSI', 'LogAbsSSIMinusNRS1', 'LogAbsSSIMinusNRS2', 'LogAbsMLK', 'LogAbsMDiff2', 'LogAbsMSR']

    frequency_features = ['mnf', 'mdf', 'rkf', 'mnp', 'ttp', 'sm1', 'sm2', 'sm3']

    if frequency_features_used(frequency_features, methods):
        data_frequency = []
        data_pow = []
        T = 1 / fs
        for channel_check in range(emg_channels_quantity):
            data = emg_data_channels[channel_check]
            N = np.size(data, 0)
            frequency = fft.fftfreq(N, T)
            pow = np.abs(fft.fft(data))
            pow = pow[frequency > 0]
            frequency = frequency[frequency > 0]
            data_frequency = add_elements_in_list(data_frequency, [frequency])
            data_pow = add_elements_in_list(data_pow, [pow])

    features = []
    for channel_check in range(emg_channels_quantity):
        features_channel = []
        for methods_check in methods:
            if methods_check == 'mav':
                features_channel = add_elements_in_list(features_channel, [get_emg_feature_mav(emg_data_channels[channel_check])])
            if methods_check == 'rms':
                features_channel = add_elements_in_list(features_channel, [get_emg_feature_rms(emg_data_channels[channel_check])])
            if methods_check == 'wl':
                features_channel = add_elements_in_list(features_channel, [get_emg_feature_wl(emg_data_channels[channel_check])])
            if methods_check == 'zc':
                features_channel = add_elements_in_list(features_channel, [get_emg_feature_zc(emg_data_channels[channel_check])])
            if methods_check == 'ssc':
                features_channel = add_elements_in_list(features_channel, [get_emg_feature_ssc(emg_data_channels[channel_check])])
            if methods_check == 'kf':
                features_channel = add_elements_in_list(features_channel, [get_emg_feature_kf(emg_data_channels[channel_check])])
            if methods_check == 'integrated':
                features_channel = add_elements_in_list(features_channel, [get_emg_feature_integrated(emg_data_channels[channel_check])])
            if methods_check == 'ssi':
                features_channel = add_elements_in_list(features_channel, [get_emg_feature_ssi(emg_data_channels[channel_check])])
            if methods_check == 'var':
                features_channel = add_elements_in_list(features_channel, [get_emg_feature_var(emg_data_channels[channel_check])])
            if methods_check == 'log':
                features_channel = add_elements_in_list(features_channel, [get_emg_feature_log(emg_data_channels[channel_check])])
            if methods_check == 'tm3':
                features_channel = add_elements_in_list(features_channel, [get_emg_feature_tm3(emg_data_channels[channel_check])])
            if methods_check == 'wap':
                features_channel = add_elements_in_list(features_channel, [get_emg_feature_wap(emg_data_channels[channel_check])])
            if methods_check == 'mnf':
                features_channel = add_elements_in_list(features_channel, [get_emg_feature_mnf(window_data_frequency=data_frequency[channel_check], window_data_pow=data_pow[channel_check])])
            if methods_check == 'mdf':
                features_channel = add_elements_in_list(features_channel, [get_emg_feature_mdf(window_data_frequency=data_frequency[channel_check], window_data_pow=data_pow[channel_check])])
            if methods_check == 'rkf':
                features_channel = add_elements_in_list(features_channel, [get_emg_feature_rkf(window_data_frequency=data_frequency[channel_check], window_data_pow=data_pow[channel_check])])
            if methods_check == 'mnp':
                features_channel = add_elements_in_list(features_channel, [get_emg_feature_mnp(window_data_frequency=data_frequency[channel_check], window_data_pow=data_pow[channel_check])])
            if methods_check == 'ttp':
                features_channel = add_elements_in_list(features_channel, [get_emg_feature_ttp(window_data_frequency=data_frequency[channel_check], window_data_pow=data_pow[channel_check])])
            if methods_check == 'sm1':
                features_channel = add_elements_in_list(features_channel, [get_emg_feature_sm1(window_data_frequency=data_frequency[channel_check], window_data_pow=data_pow[channel_check])])
            if methods_check == 'sm2':
                features_channel = add_elements_in_list(features_channel, [get_emg_feature_sm2(window_data_frequency=data_frequency[channel_check], window_data_pow=data_pow[channel_check])])
            if methods_check == 'sm3':
                features_channel = add_elements_in_list(features_channel, [get_emg_feature_sm3(window_data_frequency=data_frequency[channel_check], window_data_pow=data_pow[channel_check])])

            if methods_check == 'LogAbsSSI':
                features_channel = add_elements_in_list(features_channel, [get_emg_feature_NTDF_LogAbsSSI(emg_data_channels[channel_check])])
            if methods_check == 'LogAbsSSIMinusNRS1':
                features_channel = add_elements_in_list(features_channel, [get_emg_feature_NTDF_LogAbsSSIMinusNRS1(emg_data_channels[channel_check])])
            if methods_check == 'LogAbsSSIMinusNRS2':
                features_channel = add_elements_in_list(features_channel, [get_emg_feature_NTDF_LogAbsSSIMinusNRS2(emg_data_channels[channel_check])])
            if methods_check == 'LogAbsMLK':
                features_channel = add_elements_in_list(features_channel, [get_emg_feature_NTDF_LogAbsMLK(emg_data_channels[channel_check])])
            if methods_check == 'LogAbsMDiff2':
                features_channel = add_elements_in_list(features_channel, [get_emg_feature_NTDF_LogAbsMDiff2(emg_data_channels[channel_check])])
            if methods_check == 'LogAbsMSR':
                features_channel = add_elements_in_list(features_channel, [get_emg_feature_NTDF_LogAbsMSR(emg_data_channels[channel_check])])


        features = add_elements_in_list(features, [features_channel])
    return features


def get_emg_feature_mav(window_data):
    # Mean Absolute Value
    return np.mean(np.abs(window_data))


def get_emg_feature_rms(window_data):
    # Root Mean Square
    return np.sqrt(np.mean([pow(x, 2) for x in window_data]))


def get_emg_feature_wl(window_data):
    # Waveform Length
    diff = list(map(lambda x, y: x-y, window_data[1:], window_data[:-1]))
    return np.sum(np.abs(diff))


def get_emg_feature_zc(window_data, threshold=zc_threshold_global):
    # Zero Crossing
    if threshold == 'None':
        # threshold =
        # print('zc:', threshold)
        threshold = 0.0005

    threshold = np.mean(window_data) + threshold

    data_cross_threshold = 0
    for i in range(np.size(window_data, 0) - 1):
        if window_data[i] >= threshold > window_data[i + 1]:
            data_cross_threshold = data_cross_threshold + 1
        if window_data[i] <= threshold < window_data[i + 1]:
            data_cross_threshold = data_cross_threshold + 1

    return data_cross_threshold


def get_emg_feature_ssc(window_data, threshold=ssc_threshold_global):
    # Slope Sign Change
    if threshold == 'None':
        # delta = np.abs(list(map(lambda first, medium, last: (medium - first) * (medium - last), window_data[0:-2], window_data[1:-1], window_data[2:])))
        # print('ssc max:', np.max(delta), '  ssc min:', np.min(delta))
        threshold = 0.0005
    data_ssc = 0
    for i in range(np.size(window_data, 0) - 2):
        delta_first = window_data[i+1] - window_data[i]
        delta_last = window_data[i] - window_data[i+2]
        if delta_first * delta_last > threshold:
            data_ssc = data_ssc + 1
    return data_ssc


def get_emg_feature_kf(window_data):
    # Kurtosis Factor
    D = np.mean(list(map(lambda x: math.pow(x, 2), window_data - np.mean(window_data))))
    data_kf = np.mean(list(map(lambda x: math.pow(x, 4), window_data - np.mean(window_data)))) / D - 3
    return data_kf


def get_emg_feature_integrated(window_data):
    return np.sum(np.abs(window_data))


def get_emg_feature_ssi(window_data):
    # Simple Square Integrate
    return np.sum(list(map(lambda x: math.pow(x, 2), window_data)))


def get_emg_feature_var(window_data):
    # Variance Of EMG
    return np.var(window_data)


def get_emg_feature_log(window_data):
    # Log Detector
    data = window_data[:]
    for data_check in range(np.size(data)):
        if data[data_check] == 0:
            data[data_check] = 0.0001
    return math.exp(np.mean(list(map(lambda x: math.log(x), np.abs(data)))))


def get_emg_feature_tm3(window_data):
    # 3rd Temporal Moment
    return np.abs(np.mean(list(map(lambda x: math.pow(x, 3), window_data))))


def get_emg_feature_wap(window_data, threshold=wap_threshold_global):
    # Willison Smplitude
    if threshold == 'None':
        # delta = np.abs(list(map(lambda first, last: first - last, window_data[0:-1], window_data[1:])))
        # threshold = np.min(delta) + (np.max(delta) - np.min(delta)) * 0.05
        # print('wap max:', np.max(delta), '  wap min:', np.min(delta))
        threshold = 0.0005
    data_wap = 0
    for i in range(np.size(window_data, 0)-1):
        delta = np.abs(window_data[i+1] - window_data[i])
        if delta >= threshold:
            data_wap = data_wap + 1
    return data_wap


def get_emg_feature_mnf(window_data=[], fs=fs_global, window_data_frequency=[], window_data_pow=[]):
    # mean frequency
    if len(window_data_frequency) * len(window_data_pow) == 0:
        T = 1 / fs
        N = np.size(window_data, 0)
        window_data_frequency = fft.fftfreq(N, T)
        window_data_pow = np.abs(fft.fft(window_data))
        window_data_pow = window_data_pow[window_data_frequency > 0]
        window_data_frequency = window_data_frequency[window_data_frequency > 0]

    return np.sum(list(map(lambda x, y: x * y, window_data_frequency, window_data_pow))) / np.sum(window_data_pow)


def get_emg_feature_mdf(window_data=[], fs=fs_global, window_data_frequency=[], window_data_pow=[]):
    # median frequency
    if len(window_data_frequency) * len(window_data_pow) == 0:
        T = 1 / fs
        N = np.size(window_data, 0)
        window_data_frequency = fft.fftfreq(N, T)
        window_data_pow = np.abs(fft.fft(window_data))
        window_data_pow = window_data_pow[window_data_frequency > 0]
        window_data_frequency = window_data_frequency[window_data_frequency > 0]

    return np.sum(window_data_pow) / 2


def get_emg_feature_rkf(window_data=[], fs=fs_global, window_data_frequency=[], window_data_pow=[]):
    # Peak frequency
    if len(window_data_frequency) * len(window_data_pow) == 0:
        T = 1 / fs
        N = np.size(window_data, 0)
        window_data_frequency = fft.fftfreq(N, T)
        window_data_pow = np.abs(fft.fft(window_data))
        window_data_pow = window_data_pow[window_data_frequency > 0]
        window_data_frequency = window_data_frequency[window_data_frequency > 0]

    return np.max(window_data_pow)


def get_emg_feature_mnp(window_data=[], fs=fs_global, window_data_frequency=[], window_data_pow=[]):
    # mean power
    if len(window_data_frequency) * len(window_data_pow) == 0:
        T = 1 / fs
        N = np.size(window_data, 0)
        window_data_frequency = fft.fftfreq(N, T)
        window_data_pow = np.abs(fft.fft(window_data))
        window_data_pow = window_data_pow[window_data_frequency > 0]
        window_data_frequency = window_data_frequency[window_data_frequency > 0]

    return np.mean(window_data_pow)


def get_emg_feature_ttp(window_data=[], fs=fs_global, window_data_frequency=[], window_data_pow=[]):
    # Total power
    if len(window_data_frequency) * len(window_data_pow) == 0:
        T = 1 / fs
        N = np.size(window_data, 0)
        window_data_frequency = fft.fftfreq(N, T)
        window_data_pow = np.abs(fft.fft(window_data))
        window_data_pow = window_data_pow[window_data_frequency > 0]
        window_data_frequency = window_data_frequency[window_data_frequency > 0]

    return np.sum(window_data_pow)


def get_emg_feature_sm1(window_data=[], fs=fs_global, window_data_frequency=[], window_data_pow=[]):
    if len(window_data_frequency) * len(window_data_pow) == 0:
        T = 1 / fs
        N = np.size(window_data, 0)
        window_data_frequency = fft.fftfreq(N, T)
        window_data_pow = np.abs(fft.fft(window_data))
        window_data_pow = window_data_pow[window_data_frequency > 0]
        window_data_frequency = window_data_frequency[window_data_frequency > 0]

    return np.sum(list(map(lambda x, y: x * y, window_data_frequency, window_data_pow)))


def get_emg_feature_sm2(window_data=[], fs=fs_global, window_data_frequency=[], window_data_pow=[]):
    if len(window_data_frequency) * len(window_data_pow) == 0:
        T = 1 / fs
        N = np.size(window_data, 0)
        window_data_frequency = fft.fftfreq(N, T)
        window_data_pow = np.abs(fft.fft(window_data))
        window_data_pow = window_data_pow[window_data_frequency > 0]
        window_data_frequency = window_data_frequency[window_data_frequency > 0]

    return np.sum(list(map(lambda x, y: x * x * y, window_data_frequency, window_data_pow)))


def get_emg_feature_sm3(window_data=[], fs=fs_global, window_data_frequency=[], window_data_pow=[]):
    if len(window_data_frequency) * len(window_data_pow) == 0:
        T = 1 / fs
        N = np.size(window_data, 0)
        window_data_frequency = fft.fftfreq(N, T)
        window_data_pow = np.abs(fft.fft(window_data))
        window_data_pow = window_data_pow[window_data_frequency > 0]
        window_data_frequency = window_data_frequency[window_data_frequency > 0]

    return np.sum(list(map(lambda x, y: x * x * x * y, window_data_frequency, window_data_pow)))


def get_emg_feature_NTDF_LogAbsSSI(window_data):
    AbsSSI = np.abs(np.sum(list(map(lambda x: math.pow(x, 2), window_data))))
    if AbsSSI == 0:
        AbsSSI = 0.0001
    return np.log(AbsSSI)


def get_emg_feature_NTDF_LogAbsSSIMinusNRS1(window_data):
    SSI = np.sum(list(map(lambda x: math.pow(x, 2), window_data)))
    DerHOM_1 = list(map(lambda x, y: x - y, window_data[1:], window_data[:-1]))
    NRS1 = np.sqrt(np.mean([pow(x, 2) for x in DerHOM_1]))
    SSIMinusNRS1 = SSI - NRS1
    if SSIMinusNRS1 == 0:
        SSIMinusNRS1 = 0.0001
    return np.log(np.abs(SSIMinusNRS1))


def get_emg_feature_NTDF_LogAbsSSIMinusNRS2(window_data):
    SSI = np.sum(list(map(lambda x: math.pow(x, 2), window_data)))
    DerHOM_1 = list(map(lambda x, y: x - y, window_data[1:], window_data[:-1]))
    DerHOM_2 = list(map(lambda x, y: x - y, DerHOM_1[1:], DerHOM_1[:-1]))
    NRS2 = np.sqrt(np.mean([pow(x, 2) for x in DerHOM_2]))
    SSIMinusNRS2 = SSI - NRS2
    if SSIMinusNRS2 == 0:
        SSIMinusNRS2 = 0.0001
    return np.log(np.abs(SSIMinusNRS2))


def get_emg_feature_NTDF_LogAbsMLK(window_data):
    MLK = np.mean(np.abs(window_data))
    if MLK == 0:
        MLK = 0.0001
    return np.log(np.abs(MLK))


def get_emg_feature_NTDF_LogAbsMDiff2(window_data):
    DerHOM_1 = list(map(lambda x, y: x - y, window_data[1:], window_data[:-1]))
    DerHOM_2 = list(map(lambda x, y: x - y, DerHOM_1[1:], DerHOM_1[:-1]))
    MDiff2 = np.mean(np.abs(DerHOM_2))
    if MDiff2 == 0:
        MDiff2 = 0.0001
    return np.log(np.abs(MDiff2))


def get_emg_feature_NTDF_LogAbsMSR(window_data):
    MSR = np.mean([pow(x, 0.5) for x in window_data])
    if MSR == 0:
        MSR = 0.0001
    return np.log(np.abs(MSR))





# EMG signal processing
def get_emg_signal_processed(data, signal_processing_setup):
    outputs = data[:]
    method = signal_processing_setup['method']
    channels_with_rows = signal_processing_setup['channels_with_rows']
    for method_check in method:
        if 'baseline' in method_check:
            outputs = adjust_emg_baseline(emg_data=outputs, channels_with_rows=channels_with_rows)
        if 'butter_worth' in method_check:
            low_cut = signal_processing_setup['butter_worth_low_cut']
            high_cut = signal_processing_setup['butter_worth_high_cut']
            fs = signal_processing_setup['butter_worth_fs']
            order = signal_processing_setup['butter_worth_order']
            outputs = get_emg_butter_worth(emg_data=outputs, low_cut=low_cut, high_cut=high_cut, fs=fs, order=order, channels_with_rows=channels_with_rows)
        if 'wavelet_packet' in method_check:
            threshold = signal_processing_setup['wavelet_packet_threshold']
            threshold_mode = signal_processing_setup['wavelet_packet_threshold_mode']
            wavelet_type = signal_processing_setup['wavelet_packet_wavelet_type']
            dev_level = signal_processing_setup['wavelet_packet_dev_level']
            outputs = get_emg_wavelet_packet(emg_data=outputs, threshold=threshold, threshold_mode=threshold_mode, wavelet_type=wavelet_type, dev_level=dev_level, channels_with_rows=channels_with_rows)
        if 'notch_filter' in method_check:
            frequency_removed = signal_processing_setup['notch_filter_frequency_removed']
            quality_factor = signal_processing_setup['notch_filter_quality_factor']
            fs = signal_processing_setup['notch_filter_fs']
            outputs = get_emg_notch_filtered(emg_data=outputs, frequency_removed=frequency_removed, quality_factor=quality_factor, fs=fs, channels_with_rows=channels_with_rows)

    outputs = np.array(outputs).tolist()
    return outputs


def adjust_emg_baseline(emg_data, channels_with_rows='None'):
    data_adjusted = emg_data[:]
    if channels_with_rows == 1:
        channels_quantities = np.size(data_adjusted, 0)
    elif channels_with_rows == 0:
        channels_quantities = np.size(data_adjusted, 1)
    else:
        channels_quantities = np.min([np.size(data_adjusted, 0), np.size(data_adjusted, 1)])

    for channel in range(channels_quantities):
        data_adjusted[channel] = data_adjusted[channel] - np.mean(data_adjusted[channel])
    return data_adjusted


def get_emg_butter_worth(emg_data, low_cut, high_cut, fs, order=4, channels_with_rows='None'):
    if channels_with_rows == 1:
        channels_quantities = np.size(emg_data, 0)
    elif channels_with_rows == 0:
        channels_quantities = np.size(emg_data, 1)
    else:
        channels_quantities = np.min([np.size(emg_data, 0), np.size(emg_data, 1)])

    nyq = 0.5 * fs
    low_cut_normalized = low_cut / nyq
    high_cut_normalized = high_cut / nyq
    b, a = signal.butter(N=order, Wn=[low_cut_normalized, high_cut_normalized], btype='band', output="ba")
    emg_butter_worth_data = []
    for channel in range(channels_quantities):
        y = signal.lfilter(b, a, emg_data[channel])
        if len(emg_butter_worth_data):
            emg_butter_worth_data = emg_butter_worth_data + [y]
        else:
            emg_butter_worth_data = [y]
    return emg_butter_worth_data


def get_emg_notch_filtered(emg_data, frequency_removed=[50], quality_factor=[0], fs=1920, channels_with_rows='None'):
    data_filtered = emg_data[:]
    if channels_with_rows == 1:
        channels_quantities = np.size(emg_data, 0)
    elif channels_with_rows == 0:
        channels_quantities = np.size(emg_data, 1)
    else:
        channels_quantities = np.min([np.size(emg_data, 0), np.size(emg_data, 1)])

    for frequency_check in range(np.size(frequency_removed, 0)):
        frequency_removed_check = frequency_removed[frequency_check]
        quality_factor_check = quality_factor[frequency_check]
        if quality_factor_check == 0:
            quality_factor_check = frequency_removed_check / 2

        b, a = signal.iirnotch(frequency_removed_check, quality_factor_check, fs)
        emg_notch_filtered_data = []
        for channel in range(channels_quantities):
            y = signal.filtfilt(b, a, data_filtered[channel])
            if len(emg_notch_filtered_data):
                emg_notch_filtered_data = emg_notch_filtered_data + [y]
            else:
                emg_notch_filtered_data = [y]
        data_filtered = emg_notch_filtered_data

    return data_filtered


def get_emg_wavelet_packet(emg_data, threshold=0.04, threshold_mode='soft', wavelet_type='db8', dev_level='max', channels_with_rows='None'):
    # https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-wavelet-transform.html
    if channels_with_rows == 1:
        channels_quantities = np.size(emg_data, 0)
    elif channels_with_rows == 0:
        channels_quantities = np.size(emg_data, 1)
    else:
        channels_quantities = np.min([np.size(emg_data, 0), np.size(emg_data, 1)])

    emg_wavelet_packet_data = []
    for channel in range(channels_quantities):
        w = pywt.Wavelet(wavelet_type)
        if dev_level == 'max':
            dev_level = pywt.dwt_max_level(len(emg_data[channel]), w.dec_len)
        coeffs = pywt.wavedec(emg_data[channel], wavelet_type, level=dev_level)

        for i in range(1, len(coeffs)):
            coeffs[i] = pywt.threshold(data=coeffs[i], value=threshold * max(coeffs[i]), mode=threshold_mode, substitute=0)

        datarec = pywt.waverec(coeffs, wavelet_type)
        datarec = datarec[0:np.size(emg_data[channel])]

        if len(emg_wavelet_packet_data):
            emg_wavelet_packet_data = emg_wavelet_packet_data + [datarec]
        else:
            emg_wavelet_packet_data = [datarec]
    return emg_wavelet_packet_data


# Standardization
def z_score_scale(data, across_what='row'):
    # across_what:  'row',  'col',  'all'
    data_input = data[:]

    if across_what == 'col':
        data_input = np.transpose(data_input)
    if across_what == 'all':
        raw_row = np.size(data_input, 0)
        raw_col = np.size(data_input, 1)
        data_input = data_input.reshape(raw_row * raw_col, 1)

    scale_method = preprocessing.StandardScaler()
    data_scaled = scale_method.fit_transform(data_input)

    if across_what == 'col':
        data_scaled = np.transpose(data_scaled)
    if across_what == 'all':
        data_scaled = np.array(data_scaled).reshape(raw_row, raw_col)

    return data_scaled


def min_max_scale(data, across_what='row'):
    # across_what:  'row',  'col',  'all'
    data_input = np.array(data)

    if across_what == 'col':
        data_input = np.transpose(data_input)
    if across_what == 'all':
        raw_row = np.size(data_input, 0)
        raw_col = np.size(data_input, 1)
        data_input = data_input.reshape(raw_row * raw_col, 1)

    scale_method = preprocessing.MinMaxScaler()
    data_scaled = scale_method.fit_transform(data_input)

    if across_what == 'col':
        data_scaled = np.transpose(data_scaled)
    if across_what == 'all':
        data_scaled = np.array(data_scaled).reshape(raw_row, raw_col)

    return data_scaled


def positive_negative_one_scale(data, across_what='row'):
    # across_what:  'row',  'col',  'all'
    data_input = np.array(data)

    if across_what == 'col':
        data_input = np.transpose(data_input)
    if across_what == 'all':
        raw_row = np.size(data_input, 0)
        raw_col = np.size(data_input, 1)
        data_input = data_input.reshape(raw_row * raw_col, 1)

    scale_method = preprocessing.MinMaxScaler()
    data_scaled = scale_method.fit_transform(data_input) * 2 - 1

    if across_what == 'col':
        data_scaled = np.transpose(data_scaled)
    if across_what == 'all':
        data_scaled = np.array(data_scaled).reshape(raw_row, raw_col)

    return data_scaled


def max_abs_scale(data, across_what='row'):
    # across_what:  'row',  'col',  'all'
    data_input = data[:]

    if across_what == 'col':
        data_input = np.transpose(data_input)
    if across_what == 'all':
        raw_row = np.size(data_input, 0)
        raw_col = np.size(data_input, 1)
        data_input = data_input.reshape(raw_row * raw_col, 1)

    scale_method = preprocessing.MaxAbsScaler()
    data_scaled = scale_method.fit_transform(data_input)

    if across_what == 'col':
        data_scaled = np.transpose(data_scaled)
    if across_what == 'all':
        data_scaled = np.array(data_scaled).reshape(raw_row, raw_col)

    return data_scaled


def robust_scale(data, across_what='row'):
    # across_what:  'row',  'col',  'all'
    data_input = data[:]

    if across_what == 'col':
        data_input = np.transpose(data_input)
    if across_what == 'all':
        raw_row = np.size(data_input, 0)
        raw_col = np.size(data_input, 1)
        data_input = data_input.reshape(raw_row * raw_col, 1)

    scale_method = preprocessing.RobustScaler()
    data_scaled = scale_method.fit_transform(data_input)

    if across_what == 'col':
        data_scaled = np.transpose(data_scaled)
    if across_what == 'all':
        data_scaled = np.array(data_scaled).reshape(raw_row, raw_col)

    return data_scaled


def data_normalize(data, across_what='row', norm='l1'):
    # norm: 'l1', 'l2', 'max'
    # across_what:  'row',  'col',  'all'
    data_input = data[:]

    if across_what == 'row':
        axis = 0
    if across_what == 'col':
        axis = 1
    if across_what == 'all':
        axis = 0
        raw_row = np.size(data_input, 0)
        raw_col = np.size(data_input, 1)
        data_input = data_input.reshape(raw_row * raw_col, 1)

    data_normalized = preprocessing.normalize(data_input, norm=norm, axis=axis)

    if across_what == 'all':
        data_normalized = np.array(data_normalized).reshape(raw_row, raw_col)

    return data_normalized


def standardization(data, across_what='row', method='min_max', setup={}):
    # across_what(str):  'row',  'col',  'all'
    # method(str): 'z_score', 'min_max', 'max_abs', 'robust', 'normalize', 'positive_negative_one'
    # setup(dic):   when method='normalize', setup could be: 'norm':'l1', 'l2', 'max'
    if 'z_score' in method:
        output = z_score_scale(data=data, across_what=across_what)
    if 'min_max' in method:
        output = min_max_scale(data=data, across_what=across_what)
    if 'positive_negative_one' in method:
        output = positive_negative_one_scale(data=data, across_what=across_what)
    if 'max_abs' in method:
        output = max_abs_scale(data=data, across_what=across_what)
    if 'robust' in method:
        output = robust_scale(data=data, across_what=across_what)
    if 'normalize' in method:
        norm = setup['norm']
        output = data_normalize(data=data, across_what=across_what, norm=norm)

    return np.array(output).tolist()


def standardization_pre_channel(data, across_what='row', method='min_max', setup={}, channels_quantity=9):
    # across_what(str):  'row',  'col',  'all'
    # method(str): 'z_score', 'min_max', 'max_abs', 'robust', 'normalize', 'positive_negative_one'
    # setup(dic):   when method='normalize', setup could be: 'norm':'l1', 'l2', 'max'
    outputs = data[:]
    examples_quantity = int(np.size(data, 0) / channels_quantity)
    for channel_check in range(channels_quantity):
        channel_checked_sub = range(channel_check, (examples_quantity-1)*channels_quantity + channel_check + 1, channels_quantity)
        channel_checked_data = np.array([data[data_check] for data_check in channel_checked_sub])
        channel_checked_data_Standardization = standardization(data=channel_checked_data, across_what=across_what, method=method, setup=setup)
        for data_check in range(np.size(channel_checked_sub, 0)):
                outputs[channel_checked_sub[data_check]] = channel_checked_data_Standardization[data_check]

    return outputs


def standardization_pre_example(data, across_what='row', method='min_max', setup={}):
    # across_what(str):  'row',  'col',  'all'
    # method(str): 'z_score', 'min_max', 'max_abs', 'robust', 'normalize', 'positive_negative_one'
    # setup(dic):   when method='normalize', setup could be: 'norm':'l1', 'l2', 'max'
    outputs = []
    for example_check in range(np.size(data, 0)):
        output = standardization(data=data[example_check], across_what=across_what, method=method, setup=setup)
        outputs = outputs + [output]

    return outputs


def standardization_pre_object(data, target_detail=[], details={}, across_what='None', method='None', setup='None', standardization_setup={}):
    # The range of standardization is multiple combinations of the target_detail
    # The order of the data does not change
    # The values in standardization_setup have no priority
    # example:
    #       target_detail = ['subject', 'label']
    #       details = {'subject':[corresponding subject of the data, row elements], 'label':[label of the data, row elements], 'group':[row elements, will not be used in this example]}
    #       Then the data with the same subject and label will be processed at same time.
    # across_what(str):  'row',  'col',  'all'
    # method(str): 'z_score', 'min_max', 'max_abs', 'robust', 'normalize', 'positive_negative_one'
    # setup(dic):   when method='normalize', setup could be: 'norm':'l1', 'l2', 'max'
    outputs = data[:]

    standardization_setup_list = [key for key in standardization_setup.keys()]
    if target_detail == [] and 'target_detail' in standardization_setup_list:
        target_detail = standardization_setup['target_detail']
    if across_what == 'None' and 'across_what' in standardization_setup_list:
        across_what = standardization_setup['across_what']
    if method == 'None' and 'method' in standardization_setup_list:
        method = standardization_setup['method']
    if setup == 'None' and 'setup' in standardization_setup_list:
        setup = standardization_setup['setup']

    if across_what == 'None':
        across_what = 'row'
    if method == 'None':
        method = 'min_max'
    if setup == 'None':
        setup = {}

    per_channel_flag = 0
    if 'channels' in target_detail:
        per_channel_flag = 1
        target_detail.remove('channels')
        sample = data[0]
        channels_quantity = np.size(sample, 0)

    detail_sequence_quantity = np.size(target_detail, 0)
    search_flag = 1

    if detail_sequence_quantity == 0 or 'examples' in target_detail:
        search_flag = 0

    else:
        useful_details = []
        for detail_check in target_detail:
            useful_detail = list(details[detail_check])
            useful_details = add_elements_in_list(list_name=useful_details, element_name=[useful_detail])
        useful_details = np.transpose(useful_details)
        useful_details_types, useful_details_type_quantities, types_quantities = get_elements_types_quantities_in_a_list(useful_details)

        if types_quantities == np.size(outputs, 0):
            search_flag = 0

    if search_flag == 1:
        for details_check in useful_details_types:
            target_sub, nobody = np.where(useful_details == details_check)
            target_data = [outputs[data_check] for data_check in target_sub]

            if across_what == 'row':
                row_connect = 0
            if across_what == 'col' or across_what == 'all':
                row_connect = 1

            target_data, shaped_details = examples_reshape(examples=target_data, row_connect=row_connect)

            if per_channel_flag == 1:
                target_data_standard = standardization_pre_channel(data=target_data, across_what=across_what, method=method, setup=setup, channels_quantity=channels_quantity)
            else:
                target_data_standard = standardization(data=target_data, across_what=across_what, method=method, setup=setup)

            target_data_standard = examples_shape_recovery(examples_reshaped=target_data_standard, reshaped_detail=shaped_details)

            for data_check in range(np.size(target_sub, 0)):
                outputs[target_sub[data_check]] = target_data_standard[data_check]

    if search_flag == 0:
        outputs = standardization_pre_example(data=outputs, across_what=across_what, method=method, setup=setup)

    return outputs


# Other tools
def add_elements_in_list(list_name, element_name):
    # example2: last_name=[] element_name=[a], output=[[a]]
    # example1: last_name=[a] element_name=[b], output=[[a], [b]]
    if len(list_name):
        list_name = list_name + element_name
    else:
        list_name = element_name
    return list_name


def plot_emg_channels(time_data='None', emg_data='None', figure_num=1, line_type='b-', channels_with_rows='None'):
    if channels_with_rows == 1:
        data_length = np.size(emg_data, 1)
        channels_quantities = np.size(emg_data, 0)
    if channels_with_rows == 0:
        data_length = np.size(emg_data, 0)
        channels_quantities = np.size(emg_data, 1)
    else:
        data_length = np.max([np.size(emg_data, 0), np.size(emg_data, 1)])
        channels_quantities = np.min([np.size(emg_data, 0), np.size(emg_data, 1)])

    if time_data == 'None':
        time_data = list(range(data_length))

    if emg_data == 'None':
        warnings.warn('No emg data to plot!')
    else:
        plt.figure(figure_num)
        x_limit_setup = (0, np.max(time_data))
        y_limit_setup = (np.min(np.float32(emg_data)), np.max(np.float32(emg_data)))
        for channel in range(channels_quantities):
            plt.subplot(channels_quantities, 1, channel + 1)
            plt.plot(time_data, emg_data[channel], line_type)
            plt.xlim(x_limit_setup)
            plt.ylim(y_limit_setup)


def plot_emg_frequency_channels(frequency_data='None', pow_data='None', figure_num=1, line_type='b-', channels_with_rows='None'):
    if channels_with_rows == 1:
        data_length = np.size(pow_data, 1)
        channels_quantities = np.size(pow_data, 0)
    if channels_with_rows == 0:
        data_length = np.size(pow_data, 0)
        channels_quantities = np.size(pow_data, 1)
    else:
        data_length = np.max([np.size(pow_data, 0), np.size(pow_data, 1)])
        channels_quantities = np.min([np.size(pow_data, 0), np.size(pow_data, 1)])

    if frequency_data == 'None':
        time_data = list(range(data_length))

    if pow_data == 'None':
        warnings.warn('No emg data to plot!')
    else:
        plt.figure(figure_num)
        x_limit_setup = (0, np.max(frequency_data))
        y_limit_setup = (np.min(np.float32(pow_data)), np.max(np.float32(pow_data)))
        for channel in range(channels_quantities):
            plt.subplot(channels_quantities, 1, channel + 1)
            plt.plot(frequency_data[channel], pow_data[channel], line_type)
            plt.xlim(x_limit_setup)
            plt.ylim(y_limit_setup)


def get_target_examples(examples, examples_details='None', target_details='None'):
    if target_details in examples_details:
        target_sub = examples_details.index(target_details)
        target_examples = np.array(examples)[target_sub]
    else:
        target_examples = []
        warnings.warn('target_details:' + str(target_details) + '  can not be found in examples_details')
    return target_examples


def get_elements_types_quantities_in_a_list(data):
    # input: a list  output:types, elements' quantities in each type
    # example: input = ['a', 'a', 'b']      outputs = ['a', 'b']    [2, 1]    OR   ['b', 'a']   [1, 2]
    # Attention: No sequence!
    data_quantities = {}
    data_types = {}
    for data_check in data:
        data_quantities[str(data_check)] = data_quantities.get(str(data_check), 0) + 1
        data_types[str(data_check)] = data_check

    data_types_dic = [key for key in data_types.keys()]
    elements_quantities = [data_quantities[i] for i in data_types_dic]
    types_names = [data_types[i] for i in data_types_dic]

    return types_names, elements_quantities, np.size(types_names, 0)


def examples_reshape(examples=[], row_connect=1):
    examples = np.array(examples).tolist()
    shape_check = examples
    num_list = 0
    examples_shape_note = {'0': np.size(shape_check, 0)}

    while isinstance(shape_check[0], list):
        num_list = num_list + 1
        shape_check = shape_check[0]
        examples_shape_note[str(num_list)] = np.size(shape_check, 0)
    if row_connect == 1 and num_list == 2:
        channel_size = examples_shape_note['1']
        features_size = examples_shape_note['0'] * examples_shape_note['2']
        examples_reshaped = np.empty(shape=(channel_size, features_size))
        for example_check in range(examples_shape_note['0']):
            for channel_check in range(examples_shape_note['1']):
                for feature_check in range(examples_shape_note['2']):
                    new_feature_sub = feature_check + example_check * examples_shape_note['2']
                    examples_reshaped[channel_check][new_feature_sub] = examples[example_check][channel_check][feature_check]

    if row_connect != 1 and num_list == 2:
        channel_size = examples_shape_note['0'] * examples_shape_note['1']
        features_size = examples_shape_note['2']
        examples_reshaped = np.empty(shape=(channel_size, features_size))
        for example_check in range(examples_shape_note['0']):
            for channel_check in range(examples_shape_note['1']):
                for feature_check in range(examples_shape_note['2']):
                    new_channel_sub = channel_check + example_check * examples_shape_note['1']
                    examples_reshaped[new_channel_sub][feature_check] = examples[example_check][channel_check][feature_check]

    if num_list != 2:
        examples_reshaped = examples

    reshaped_detail = examples_shape_note
    reshaped_detail['row_connect'] = row_connect
    reshaped_detail['num_list'] = num_list

    return examples_reshaped, reshaped_detail


def examples_shape_recovery(examples_reshaped=[], reshaped_detail={}):
    examples = np.empty(shape=(reshaped_detail['0'], reshaped_detail['1'], reshaped_detail['2']))
    if reshaped_detail['num_list'] == 2:
        if reshaped_detail['row_connect'] == 1:
            for example_check in range(reshaped_detail['0']):
                for channel_check in range(reshaped_detail['1']):
                    for feature_check in range(reshaped_detail['2']):
                        new_feature_sub = feature_check + example_check * reshaped_detail['2']
                        examples[example_check][channel_check][feature_check] = examples_reshaped[channel_check][new_feature_sub]
        if reshaped_detail['row_connect'] != 1:
            for example_check in range(reshaped_detail['0']):
                for channel_check in range(reshaped_detail['1']):
                    for feature_check in range(reshaped_detail['2']):
                        new_channel_sub = channel_check + example_check * reshaped_detail['1']
                        examples[example_check][channel_check][feature_check] = examples_reshaped[new_channel_sub][feature_check]
    if reshaped_detail['num_list'] != 2:
        examples = examples_reshaped

    return examples


def emg_fft(emg, fs=fs_global):
    T = 1 / fs
    N = np.size(emg, 0)
    data_frequency = fft.fftfreq(N, T)
    data_pow = np.abs(fft.fft(emg))
    data_pow = data_pow[data_frequency > 0]
    data_frequency = data_frequency[data_frequency > 0]
    return data_frequency, data_pow


def emg_channels_fft(emg, fs=fs_global):
    channel_size = np.size(emg, 0)
    data_frequency = []
    data_pow = []
    for channel_check in range(channel_size):
        emg_frequency, emg_pow = emg_fft(emg=emg[channel_check], fs=fs)
        data_frequency = add_elements_in_list(data_frequency, [emg_frequency])
        data_pow = add_elements_in_list(data_pow, [emg_pow])
    return data_frequency, data_pow
