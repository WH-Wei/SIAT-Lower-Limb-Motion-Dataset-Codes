# Last change 2022.08.31
import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from multiprocessing import Pool
import time
import xlrd
import math
import scipy.io
import EMGbox_WWH

# What you want to do?
flag_load_emg_data = 1
flag_emg_signal_processing = 1
flag_get_emg_windows = 1
flag_emg_standardization = 1
flag_get_emg_features_from_windows = 1
flag_features_standardization = 1

# path setup
path_data_set_folder_global = ".../SIAT_LLMD/"  # Don't miss '/' in the end
path_save_results_to_folder = "....save_folder_path...."
path_input_data_folder = '....save_folder_path..../windows/20220831172024/'  # If above flag = 0 happened, this path should be given!

# auto path setup
path_save_emg_data_to_folder_global = path_save_results_to_folder + "raw_emg/"
path_save_emg_signal_processed_to_folder_global = path_save_results_to_folder + "emg_filtered/"
path_save_emg_windows_to_folder_global = path_save_results_to_folder + "windows/"
path_save_emg_standardization_to_folder_global = path_save_results_to_folder + "emg_standard/"
path_save_emg_features_to_folder_global = path_save_results_to_folder + "emg_features/"
path_save_features_standard_to_folder_global = path_save_results_to_folder + "features_standard/"

# emg data setup
# movements_list_global: 'KLFT', 'TPTO', 'LLB', 'LLF', 'KLCL', 'LLS', 'HS', 'LUGF', 'LUGB', 'TO', 'STDUP', 'SITDN',
subjects_list_global = list(range(11, 51, 1))
movements_list_global = ['KLFT', 'TPTO', 'LLB', 'LLF', 'KLCL', 'LLS', 'HS', 'LUGF', 'LUGB', 'TO', 'STDUP', 'SITDN']


label_type_global = 'all'  # 'num', 'one_hot', 'all'

EMG_channels_quantities_global = 9
sampling_frequency_global = 1920

subjects_quantity_global = np.size(subjects_list_global, 0)
movements_quantity_global = np.size(movements_list_global, 0)

# window setup
window_size_global = 150
window_overlap_size_global = 50

# data drop setup
# when the status change('from A to R' OR 'from R to A'), drop some data
data_drop_time_setup = 0  # ms
data_drop_global = math.floor(data_drop_time_setup / 1000 * sampling_frequency_global)  # do not change

# EMG signal processing setup
# method[a list]: could be 'baseline', 'butter_worth', 'notch_filter', 'wavelet_packet'
# channels_with_rows: 1, 0, default:1
signal_processing_setup_global = {'method': ['butter_worth'],
                                  'butter_worth_low_cut': 15, 'butter_worth_high_cut': 400, 'butter_worth_fs': 1920, 'butter_worth_order': 7,
                                  'wavelet_packet_threshold': 0.08, 'wavelet_packet_threshold_mode': 'soft', 'wavelet_packet_wavelet_type': 'db7', 'wavelet_packet_dev_level': 9,
                                  'notch_filter_frequency_removed': [50], 'notch_filter_quality_factor': [100], 'notch_filter_fs': 1920,
                                  'channels_with_rows': 1}

# Standardization setup
# example:
#       target_detail = ['subject', 'movement', 'group', 'examples', 'channels']
#       details = {'subject':[corresponding subject of the data, row elements], 'label':[label of the data, row elements], 'group':[row elements, will not be used in this example]}
#       Then the data with the same subject and label will be processed at same time.
# across_what(str):  'row',  'col',  'all'
# method(str): 'z_score', 'min_max', 'max_abs','positive_negative_one', 'robust', 'normalize'
# setup(dic):   when method='normalize', setup could be: 'norm':'l1', 'l2', 'max'
EMG_standardization_setup_global = {'method': 'positive_negative_one', 'target_detail': ['subject'], 'across_what': 'col', 'setup': {'norm': 'l1'}}
features_standardization_setup_global = {'method': 'min_max', 'target_detail': ['subject', 'channels'], 'across_what': 'row', 'setup': {'norm': 'l1'}}

# features setup
# 'all', 'mav', 'rms', 'wl', 'zc', 'ssc', 'kf', 'integrated', 'ssi', 'var', 'log', 'tm3', 'wap', 'mnf', 'mdf', 'rkf', 'mnp', 'ttp', 'sm1', 'sm2', 'sm3','NTDF'(Novel time-domain feature set, use it alone)
# features_type_global = ['integrated', 'var', 'wl', 'zc', 'ssc', 'wap']      # Du’s feature set
# features_type_global = ['mav', 'wl', 'zc', 'ssc']      # Hudgins’s feature set
features_type_global = ['NTDF']   # Novel time-domain feature set (NTDF)
# features_type_global = ['mav']
# features_type_global = ['mav', 'rms', 'wl', 'zc', 'ssc', 'kf', 'integrated', 'ssi', 'var', 'log', 'tm3', 'wap', 'mnf', 'mdf', 'rkf', 'mnp', 'ttp', 'sm1', 'sm2', 'sm3']

# other setup
multi_progress_global = 0  # 0 : close multi-progress      -1 : create all progress        number>1 : create number>1 progress
if multi_progress_global == -1:
    multi_progress_global = os.cpu_count() - 1

angle_name = ['left hip adduction', 'left hip flexion', 'left knee flexion', 'left ankle flexion', 'right hip adduction', 'right hip flexion', 'right knee flexion', 'right ankle flexion']
torque_name = ['left hip adduction torque', 'left hip flexion torque', 'left knee flexion torque', 'left ankle flexion torque', 'right hip adduction torque', 'right hip flexion torque', 'right knee flexion torque', 'right ankle flexion torque']
EMG_name = ['tensor fascia lata', 'rectus femoris', 'vastus medialis', 'semimembranosus', 'upper tibialis anterior', 'lower tibialis anterior', 'lateral gastrocnemius', 'medial gastrocnemius', 'soleus']


def get_the_tasks_of_each_progress(all_tasks=[], multi_progress=multi_progress_global):
    tasks_quantities = np.size(all_tasks, 0)
    tasks_average_quantities = math.floor(tasks_quantities / multi_progress)
    tasks = []
    begin_sub = 0
    for progress_check in range(multi_progress):
        if progress_check != multi_progress - 1:
            sub_tasks = all_tasks[begin_sub:begin_sub + tasks_average_quantities]
            begin_sub = begin_sub + tasks_average_quantities
        if progress_check == multi_progress - 1:
            sub_tasks = all_tasks[begin_sub:tasks_quantities]
        tasks = add_elements_in_list(tasks, [sub_tasks])

    return tasks


def create_results_folder():
    files_list = os.listdir(path_save_results_to_folder)
    if "raw_emg" not in files_list:
        os.mkdir(path_save_emg_data_to_folder_global)
    if "emg_filtered" not in files_list:
        os.mkdir(path_save_emg_signal_processed_to_folder_global)
    if "windows" not in files_list:
        os.mkdir(path_save_emg_windows_to_folder_global)
    if "emg_standard" not in files_list:
        os.mkdir(path_save_emg_standardization_to_folder_global)
    if "emg_features" not in files_list:
        os.mkdir(path_save_emg_features_to_folder_global)
    if "features_standard" not in files_list:
        os.mkdir(path_save_features_standard_to_folder_global)


def get_time():
    year_finished = time.strftime('%Y')
    month_finished = time.strftime('%m')
    day_finished = time.strftime('%d')
    hour_finished = time.strftime('%H')
    min_finished = time.strftime('%M')
    sec_finished = time.strftime('%S')
    time_finished = str(year_finished) + str(month_finished) + str(day_finished) + str(hour_finished) + str(min_finished) + str(sec_finished)
    return time_finished


def get_folder_path(first_flag):
    path_input = []
    if first_flag == 1 and flag_load_emg_data * flag_emg_signal_processing * flag_get_emg_windows * flag_emg_standardization * flag_get_emg_features_from_windows * flag_features_standardization == 0:
        path_input = path_input_data_folder

    else:
        newest = 0
        files_list = os.listdir(path_save_results_to_folder + "raw_emg/")
        if files_list:
            newest_folder = int(np.max(np.double(files_list)))
            if newest_folder > newest:
                newest = newest_folder
                path_input = path_save_results_to_folder + "raw_emg/" + str(newest_folder) + '/'

        files_list = os.listdir(path_save_results_to_folder + "emg_filtered/")
        if files_list:
            newest_folder = int(np.max(np.double(files_list)))
            if newest_folder > newest:
                newest = newest_folder
                path_input = path_save_results_to_folder + "emg_filtered/" + str(newest_folder) + '/'

        files_list = os.listdir(path_save_results_to_folder + "windows/")
        if files_list:
            newest_folder = int(np.max(np.double(files_list)))
            if newest_folder > newest:
                newest = newest_folder
                path_input = path_save_results_to_folder + "windows/" + str(newest_folder) + '/'

        files_list = os.listdir(path_save_results_to_folder + "emg_standard/")
        if files_list:
            newest_folder = int(np.max(np.double(files_list)))
            if newest_folder > newest:
                newest = newest_folder
                path_input = path_save_results_to_folder + "emg_standard/" + str(newest_folder) + '/'

        files_list = os.listdir(path_save_results_to_folder + "emg_features/")
        if files_list:
            newest_folder = int(np.max(np.double(files_list)))
            if newest_folder > newest:
                newest = newest_folder
                path_input = path_save_results_to_folder + "emg_features/" + str(newest_folder) + '/'

        files_list = os.listdir(path_save_results_to_folder + "features_standard/")
        if files_list:
            newest_folder = int(np.max(np.double(files_list)))
            if newest_folder > newest:
                path_input = path_save_results_to_folder + "features_standard/" + str(newest_folder) + '/'

    return path_input


def create_results_folder_with_time_name(data_path):
    folder_name = get_time()
    path_new_folder = data_path + folder_name + '/'
    os.mkdir(path_new_folder)
    time.sleep(1)
    return path_new_folder


def get_FileFullName_in_folder_with_SameString_and_extension(path_folder, same_string, extension, flag_create_none_file=0):
    files_list = os.listdir(path_folder)
    flag_success = 0
    file_full_name = 'None'
    for file_checked in files_list:
        if same_string in file_checked:
            if os.path.splitext(file_checked)[1] == extension:
                file_full_name = file_checked
                flag_success = 1
        if flag_success == 1:
            break
    if flag_success == 0:
        if flag_create_none_file == 0:
            print('Error[from:get_FileFullName_in_folder_with_SameString_and_extension]: Can''t find [', same_string, '], the [', extension, '] file, in the folder: [', path_folder, '], return:', file_full_name)
        else:
            file_full_name = same_string + extension
            path_create_none_file = path_folder + '/' + file_full_name
            open(path_create_none_file, 'w')
            print('Error[from:get_FileFullName_in_folder_with_SameString_and_extension]: Can''t find [', same_string, '], the [', extension, '] file, in the folder: [', path_folder, '], create a new file:', file_full_name)

    return file_full_name


def get_all_raw_data_path(path_data_set_folder, subjects_list, movements_list):
    if "Static" in movements_list:
        movements_list.remove("Static")
        print("Notice[from:get_all_raw_data_path]: The path of [Static] in movements_list is already deleted.")

    length_path_data_set_folder = len(path_data_set_folder)
    length_target_path = length_path_data_set_folder + 100

    subjects_quantity = np.size(subjects_list)
    movements_quantity = np.size(movements_list)
    path_raw_label = np.empty(shape=(subjects_quantity, movements_quantity), dtype=(str, length_target_path))
    path_raw_data = np.empty(shape=(subjects_quantity, movements_quantity), dtype=(str, length_target_path))
    for subjects_num in range(subjects_quantity):
        path_subject_raw_label_folder = path_data_set_folder + "Subject" + str(subjects_list[subjects_num]) + "/Labels/"
        path_subject_raw_data_folder = path_data_set_folder + "Subject" + str(subjects_list[subjects_num]) + "/Data/"
        for movements_num in range(movements_quantity):
            path_raw_label[subjects_num, movements_num] = path_subject_raw_label_folder + get_FileFullName_in_folder_with_SameString_and_extension(path_folder=path_subject_raw_label_folder, same_string=movements_list[movements_num],
                                                                                                                                                   extension='.xlsx')
            path_raw_data[subjects_num, movements_num] = path_subject_raw_data_folder + get_FileFullName_in_folder_with_SameString_and_extension(path_folder=path_subject_raw_data_folder, same_string=movements_list[movements_num], extension='.xlsx')
    return path_raw_data, path_raw_label


def get_label_from_target_list(target_list=[], label_type='all'):
    values = np.array(target_list[:])
    data_num_label_list = LabelEncoder().fit_transform(values)
    data_OneHot_label_list = OneHotEncoder(sparse=False).fit_transform(data_num_label_list.reshape((-1, 1)))
    if label_type == 'num':
        return data_num_label_list
    elif label_type == 'one_hot':
        return data_OneHot_label_list
    elif label_type == 'all':
        return data_num_label_list, data_OneHot_label_list
    else:
        print("Error[from:get_label_from_target_list]: label_type is wrong")


def read_excel(path_excel='None', read_locations_begin=[], read_locations_end=[], read_locations_sheet=[]):
    read_excel_return = []
    locations_begin_quantity = np.size(read_locations_begin, 0)
    locations_end_quantity = np.size(read_locations_end, 0)
    locations_sheet_quantity = np.size(read_locations_sheet, 0)

    # check the errors
    if locations_begin_quantity != locations_end_quantity:
        print("Error[from:read_excel]: The number of [read_locations_begin] and [read_locations_end] is different.")
    else:
        locations_quantity = locations_begin_quantity

    if locations_sheet_quantity != locations_quantity:
        locations_sheet = [0 for _ in range(locations_quantity)]

    # read the excel file and sheet list
    workbook = xlrd.open_workbook(filename=path_excel, on_demand=True)
    sheet_name_list = workbook.sheet_names()

    # read the target data
    excel_data = []
    for locations_check in range(locations_quantity):
        target_data_sheet_name = sheet_name_list[locations_sheet[locations_check]]
        target_data_sheet = workbook.sheet_by_name(target_data_sheet_name)
        locations_begin = read_locations_begin[locations_check]
        locations_end = read_locations_end[locations_check]
        locations_data = []
        for cols_check in range(locations_begin[1], locations_end[1] + 1):
            if locations_data:
                locations_data = locations_data + [target_data_sheet.col_values(colx=cols_check, start_rowx=locations_begin[0], end_rowx=locations_end[0])]
            else:
                locations_data = [target_data_sheet.col_values(colx=cols_check, start_rowx=locations_begin[0], end_rowx=locations_end[0])]
        if len(excel_data):
            excel_data = excel_data + locations_data
        else:
            excel_data = locations_data
    return excel_data


def get_emg_window_data(emg_data, label_status, label_group, window_size=150, window_overlap_size=50):
    data_length = np.size(label_status)
    counter = 0
    begin_group = -1
    label_status_begin = []
    data_active_window_data = []
    data_active_window_group = []
    data_rest_window_data = []
    data_rest_window_group = []

    for data_check in range(data_length):
        if len(label_status_begin) == 0:
            label_status_begin = label_status[data_check]

        if label_status[data_check] == label_status_begin:
            if counter == 0:
                active_begin = data_check
                begin_group = label_group[data_check]
                counter = counter + 1
            else:
                if label_group[data_check] == begin_group:
                    counter = counter + 1
                else:
                    counter = 1
                    begin_group = label_group[data_check]
                    active_begin = data_check

            if counter == window_size:
                active_end = data_check
                target_data = [x[active_begin:active_end + 1] for x in emg_data]

                if label_status_begin == 'A':
                    if len(data_active_window_data):
                        data_active_window_data = data_active_window_data + [target_data]
                        data_active_window_group = data_active_window_group + [begin_group]
                    else:
                        data_active_window_data = [target_data]
                        data_active_window_group = [begin_group]

                if label_status_begin == 'R':
                    if len(data_rest_window_data):
                        data_rest_window_data = data_rest_window_data + [target_data]
                        data_rest_window_group = data_rest_window_group + [begin_group]
                    else:
                        data_rest_window_data = [target_data]
                        data_rest_window_group = [begin_group]

                active_begin = active_begin + window_size - window_overlap_size
                counter = window_overlap_size
        else:
            counter = 1
            label_status_begin = label_status[data_check]
            begin_group = label_group[data_check]
            active_begin = data_check
            continue

    return data_active_window_data, data_rest_window_data, data_active_window_group, data_rest_window_group


def add_elements_in_list(list_name, element_name):
    # example2: last_name=[] element_name=[a], output=[a]
    # example1: last_name=[a] element_name=[b], output=[[a], [b]]
    if len(list_name):
        list_name = list_name + element_name
    else:
        list_name = element_name
    return list_name


def load_data(label_path, data_path):
    [raw_label_time, raw_label_status, raw_label_group] = read_excel(path_excel=label_path, read_locations_begin=[[1, 0], [1, 1], [1, 2]], read_locations_end=[[None, 0], [None, 1], [None, 2]])
    raw_data_emg = read_excel(path_excel=data_path, read_locations_begin=[[1, 17]], read_locations_end=[[None, 25]])
    raw_data_angle = read_excel(path_excel=data_path, read_locations_begin=[[1, 1]], read_locations_end=[[None, 8]])
    raw_data_torque = read_excel(path_excel=data_path, read_locations_begin=[[1, 9]], read_locations_end=[[None, 16]])
    return raw_data_emg, raw_label_time, raw_label_status, raw_label_group, raw_data_angle, raw_data_torque


def get_data_information(path_data_set_folder, subjects_list, movements_list, label_type):
    path_raw_data, path_raw_label = get_all_raw_data_path(path_data_set_folder=path_data_set_folder, subjects_list=subjects_list[:], movements_list=movements_list[:])
    movements_label_list = movements_list[:] + ['R']
    num_label_list, OneHot_label_list = get_label_from_target_list(target_list=movements_label_list[:], label_type=label_type)
    data_quantity = np.size(path_raw_data)
    data_inform = {}

    for data_check in range(data_quantity):
        row_sub = math.floor(data_check / movements_quantity_global)
        col_sub = data_check - row_sub * movements_quantity_global

        data_path = path_raw_data[row_sub, col_sub]
        label_path = path_raw_label[row_sub, col_sub]
        subject = subjects_list[row_sub]
        movement = movements_list[col_sub]
        num_label = num_label_list[col_sub]
        OneHot_label = OneHot_label_list[col_sub]
        rest_num_label = num_label_list[-1]
        rest_OneHot_label = OneHot_label_list[-1]

        data_inform['data' + str(data_check)] = {'data_path': data_path, 'label_path': label_path,
                                                 'subject': subject, 'movement': movement,
                                                 'num_label': num_label, 'OneHot_label': OneHot_label,
                                                 'rest_num_label': rest_num_label, 'rest_OneHot_label': rest_OneHot_label}

    data_inform_name = [key for key in data_inform.keys()]

    return data_inform, data_inform_name


def data_drop(raw_data_status, drop_quantity):
    drop_data_check = 0
    data_length = np.size(raw_data_status) - 1
    while drop_data_check < data_length:
        if raw_data_status[drop_data_check] != raw_data_status[drop_data_check + 1]:
            for i in range(np.min([drop_data_check, drop_data_check - drop_quantity + 1]), np.min([drop_data_check + drop_quantity + 1, data_length])):
                raw_data_status[i] = 'N'
            drop_data_check = drop_data_check + drop_quantity
        drop_data_check = drop_data_check + 1
    return raw_data_status


def get_emg_data_from_one_data(single_data_inform):
    data_path = single_data_inform['data_path'][:]
    label_path = single_data_inform['label_path']
    subject = single_data_inform['subject']
    movement = single_data_inform['movement']
    num_label = single_data_inform['num_label']
    OneHot_label = single_data_inform['OneHot_label']
    rest_num_label = single_data_inform['rest_num_label']
    rest_OneHot_label = single_data_inform['rest_OneHot_label']

    if multi_progress_global != 0:
        path_save_emg_data_to_folder = single_data_inform['save_path']
    else:
        path_save_emg_data_to_folder = path_save_emg_data_to_folder_global[:]

    raw_data_emg, raw_label_time, raw_label_status, raw_label_group, raw_data_angle, raw_data_torque = load_data(label_path=label_path, data_path=data_path)

    emg_data = {'emg_data': raw_data_emg, 'EMG_name': EMG_name, 'time_data': raw_label_time,
                'label_status': raw_label_status, 'label_group': raw_label_group,
                'subject': subject, 'movement': movement, 'num_label': num_label, 'OneHot_label': OneHot_label,
                'rest_num_label': rest_num_label, 'rest_OneHot_label': rest_OneHot_label,
                'angle_name': angle_name, 'raw_data_angle': raw_data_angle,
                'torque_name': torque_name, 'raw_data_torque': raw_data_torque}

    files_name = 'sub_' + str(subject) + '_mov_' + str(movement)
    files_save_path = path_save_emg_data_to_folder + files_name + '.mat'
    scipy.io.savemat(files_save_path, emg_data)

    subject_sub = subjects_list_global.index(subject)
    movement_sub = movements_list_global.index(movement)
    # print('Loading EMG data:' + str(subject_sub * movements_quantity_global + movement_sub + 1) + ' / ' + str(subjects_quantity_global * movements_quantity_global))
    print('Loading EMG data: Subject ' + str(subject) + '   Movement ' + str(movement))

    return emg_data


def signal_processing_for_one_emg_data(data_path):
    if multi_progress_global != 0:
        path_save_emg_signal_processed_to_folder = data_path[1]
        data_load_path = data_path[0]
    else:
        path_save_emg_signal_processed_to_folder = path_save_emg_signal_processed_to_folder_global[:]
        data_load_path = data_path

    emg_signal_processed = scipy.io.loadmat(data_load_path)
    raw_data_emg = np.array(emg_signal_processed['emg_data']).tolist()
    data_emg = EMGbox_WWH.get_emg_signal_processed(data=raw_data_emg, signal_processing_setup=signal_processing_setup_global)
    emg_signal_processed['emg_data'] = data_emg

    subject = list(emg_signal_processed['subject'])[0][0]
    movement = list(emg_signal_processed['movement'])[0]
    file_name = 'sub_' + str(subject) + '_mov_' + str(movement) + '_filtered'
    files_save_path = path_save_emg_signal_processed_to_folder + file_name + '.mat'
    scipy.io.savemat(files_save_path, emg_signal_processed)

    subject_sub = subjects_list_global.index(subject)
    movement_sub = movements_list_global.index(movement)
    # print('Signal_processing:' + str(subject_sub * movements_quantity_global + movement_sub + 1) + ' / ' + str(subjects_quantity_global * movements_quantity_global))
    print('Signal_processing: Subject ' + str(subject) + '   Movement ' + str(movement))

    return emg_signal_processed


def get_emg_windows_from_one_data(data_path):
    if multi_progress_global != 0:
        path_save_emg_windows_to_folder = data_path[1]
        data_path = data_path[0]
    else:
        path_save_emg_windows_to_folder = path_save_emg_windows_to_folder_global[:]

    data = scipy.io.loadmat(data_path)

    data_emg = np.array(data['emg_data']).tolist()
    time_data = list(data['time_data'][0])
    label_status = list(data['label_status'])
    label_group = list(data['label_group'][0])
    subject = list(data['subject'])[0][0]
    movement = list(data['movement'])[0]
    num_label = list(data['num_label'])[0][0]
    OneHot_label = np.array(data['OneHot_label'][0]).tolist()
    data_angle = np.array(data['raw_data_angle']).tolist()
    data_torque = np.array(data['raw_data_torque']).tolist()

    rest_num_label = list(data['rest_num_label'])[0][0]
    rest_OneHot_label = np.array(data['rest_OneHot_label'][0]).tolist()

    label_status = data_drop(raw_data_status=label_status, drop_quantity=data_drop_global)

    data_emg_active, data_emg_rest, data_active_group, data_rest_group = get_emg_window_data(emg_data=data_emg, label_status=label_status, label_group=label_group, window_size=window_size_global,
                                                                                             window_overlap_size=window_overlap_size_global)
    data_angle_active, data_angel_rest, useless1, useless2 = get_emg_window_data(emg_data=data_angle, label_status=label_status, label_group=label_group, window_size=window_size_global,
                                                                                 window_overlap_size=window_overlap_size_global)
    data_torque_active, data_torque_rest, useless1, useless2 = get_emg_window_data(emg_data=data_torque, label_status=label_status, label_group=label_group, window_size=window_size_global,
                                                                                   window_overlap_size=window_overlap_size_global)

    data_active_num_labels = [num_label] * np.size(data_active_group, 0)
    data_active_OneHot_labels = [OneHot_label] * np.size(data_active_group, 0)
    data_active_details = [[subject] * np.size(data_active_group, 0), [movement] * np.size(data_active_group, 0), data_active_group]
    data_rest_num_labels = [rest_num_label] * np.size(data_rest_group, 0)
    data_rest_OneHot_labels = [rest_OneHot_label] * np.size(data_rest_group, 0)
    data_rest_details = [[subject] * np.size(data_rest_group, 0), [movement] * np.size(data_rest_group, 0), data_rest_group]

    save_path = path_save_emg_windows_to_folder
    files_name = 'sub_' + str(subject) + '_mov_' + str(movement) + '_windows'
    active_files_name = files_name + '_active.mat'
    rest_files_name = files_name + '_rest.mat'

    active_windows = {'windows': data_emg_active, 'EMG_name': EMG_name, 'angle_name': angle_name, 'angle_windows': data_angle_active, 'torque_name': torque_name, 'torque_windows': data_torque_active, 'num_labels': data_active_num_labels,
                      'OneHot_labels': data_active_OneHot_labels,
                      'subject': data_active_details[0], 'movement': data_active_details[1], 'group': data_active_details[2]}
    rest_windows = {'windows': data_emg_rest, 'EMG_name': EMG_name, 'angle_name': angle_name, 'angle_windows': data_angel_rest, 'torque_name': torque_name, 'torque_windows': data_torque_rest, 'num_labels': data_rest_num_labels,
                    'OneHot_labels': data_rest_OneHot_labels,
                    'subject': data_rest_details[0], 'movement': data_rest_details[1], 'group': data_rest_details[2]}

    scipy.io.savemat(save_path + active_files_name, active_windows)
    scipy.io.savemat(save_path + rest_files_name, rest_windows)

    subject_sub = subjects_list_global.index(subject)
    movement_sub = movements_list_global.index(movement)
    # print('Getting EMG windows:' + str(subject_sub * movements_quantity_global + movement_sub + 1) + ' / ' + str(subjects_quantity_global * movements_quantity_global))
    print('Getting EMG windows: Subject ' + str(subject) + '   Movement ' + str(movement))

    return active_windows, rest_windows


def get_all_emg_windows(path_folder):
    save_name = 'emg_windows.mat'
    windows = []
    angle_windows = []
    torque_windows = []
    num_labels = []
    OneHot_labels = []
    details_subject = []
    details_movement = []
    details_group = []
    for subject_name in subjects_list_global:
        for movement_check in range(movements_quantity_global):
            movement_name = movements_list_global[movement_check]

            active_windows_path = path_folder + 'sub_' + str(subject_name) + '_mov_' + str(movement_name) + '_windows_active.mat'
            rest_windows_path = path_folder + 'sub_' + str(subject_name) + '_mov_' + str(movement_name) + '_windows_rest.mat'
            active_windows = scipy.io.loadmat(active_windows_path)
            rest_windows = scipy.io.loadmat(rest_windows_path)

            windows_emg_active = list(active_windows['windows'][:])
            windows_angle_active = list(active_windows['angle_windows'][:])
            windows_torque_active = list(active_windows['torque_windows'][:])
            windows_num_labels_active = list(active_windows['num_labels'][0][:])
            windows_OneHot_label_active = list(active_windows['OneHot_labels'][:])
            windows_subject_active = list(active_windows['subject'][0][:])
            windows_movement_active = list(active_windows['movement'][:])
            windows_group_active = list(active_windows['group'][0][:])

            windows_emg_rest = list(rest_windows['windows'][:])
            windows_angle_rest = list(rest_windows['angle_windows'][:])
            windows_torque_rest = list(rest_windows['torque_windows'][:])
            windows_num_labels_rest = list(rest_windows['num_labels'][0][:])
            windows_OneHot_label_rest = list(rest_windows['OneHot_labels'][:])
            windows_subject_rest = list(rest_windows['subject'][0][:])
            windows_movement_rest = list(rest_windows['movement'][:])
            windows_group_rest = list(rest_windows['group'][0][:])

            windows = windows + windows_emg_active + windows_emg_rest
            angle_windows = angle_windows + windows_angle_active + windows_angle_rest
            torque_windows = torque_windows + windows_torque_active + windows_torque_rest
            num_labels = num_labels + windows_num_labels_active + windows_num_labels_rest
            OneHot_labels = OneHot_labels + windows_OneHot_label_active + windows_OneHot_label_rest
            details_subject = details_subject + windows_subject_active + windows_subject_rest
            details_movement = details_movement + windows_movement_active + windows_movement_rest
            details_group = details_group + windows_group_active + windows_group_rest

    angle_labels = get_movement_data_to_label(np.array(angle_windows).tolist())
    torque_labels = get_movement_data_to_label(np.array(torque_windows).tolist())

    movements_label_list = movements_list_global[:] + ['R']
    num_label_list, OneHot_label_list = get_label_from_target_list(target_list=movements_label_list[:], label_type=label_type_global)

    outputs = {'examples': np.array(windows).tolist(), 'EMG_name': EMG_name, 'angle_name': angle_name, 'angle_labels': angle_labels, 'torque_name': torque_name, 'torque_labels': torque_labels,
               'num_labels': num_labels, 'OneHot_labels': np.array(OneHot_labels).tolist(),
               'details_subject': details_subject, 'details_movement': details_movement, 'details_group': details_group,
               'movements_label_list': movements_label_list, 'num_label_list': num_label_list, 'OneHot_label_list': OneHot_label_list}
    scipy.io.savemat(path_folder + save_name, outputs)

    return outputs


def get_movement_data_to_label(movement_data_windows):
    windows_quantity = np.size(movement_data_windows, 0)
    joints_quantity = np.size(movement_data_windows, 1)

    label_windows = np.empty(shape=(windows_quantity, joints_quantity), dtype=float)
    for windows_check in range(windows_quantity):
        for joint_check in range(joints_quantity):
            label_windows[windows_check, joint_check] = np.mean(movement_data_windows[windows_check][joint_check])

    return label_windows.tolist()


def get_emg_standard(data_path):
    data = scipy.io.loadmat(data_path)
    examples = np.array(data['examples'][:]).tolist()
    num_labels = list(data['num_labels'][0][:])
    OneHot_labels = np.array(data['OneHot_labels'][:]).tolist()
    subject = list(data['details_subject'][0][:])
    movement = list(data['details_movement'][:])
    group = list(data['details_group'][0][:])
    angle_labels = np.array(data['angle_labels'][:]).tolist()
    torque_labels = np.array(data['torque_labels'][:]).tolist()
    movements_label_list = list(data['movements_label_list'][:])
    num_label_list = np.array(data['num_label_list'][0][:])
    OneHot_label_list = np.array(data['OneHot_label_list'][:]).tolist()

    details_dic = {'subject': subject, 'movement': movement, 'group': group}

    emg_standard = EMGbox_WWH.standardization_pre_object(data=examples, details=details_dic, standardization_setup=EMG_standardization_setup_global)

    outputs = {'examples': emg_standard, 'num_labels': num_labels, 'OneHot_labels': OneHot_labels, 'EMG_name': EMG_name, 'angle_name': angle_name, 'angle_labels': angle_labels, 'torque_name': torque_name, 'torque_labels': torque_labels,
               'details_subject': subject, 'details_movement': movement, 'details_group': group, 'movements_label_list': movements_label_list, 'num_label_list': num_label_list, 'OneHot_label_list': OneHot_label_list}

    save_path = path_save_emg_standardization_to_folder_global
    save_name = 'emg_examples.mat'
    scipy.io.savemat(save_path + save_name, outputs)

    return outputs


def get_features_multi_progress(tasks):
    examples = tasks
    results = EMGbox_WWH.get_emg_features_from_examples(examples_emg_data=examples, methods=features_type_global)
    return results


def get_features(data_path):
    data = scipy.io.loadmat(data_path)
    examples = np.array(data['examples']).tolist()[:]
    num_labels = list(data['num_labels'][0][:])
    OneHot_labels = np.array(data['OneHot_labels'][:]).tolist()
    subject = list(data['details_subject'][0][:])
    movement = list(data['details_movement'][:])
    group = list(data['details_group'][0][:])
    angle_labels = np.array(data['angle_labels'][:]).tolist()
    torque_labels = np.array(data['torque_labels'][:]).tolist()
    movements_label_list = list(data['movements_label_list'][:])
    num_label_list = np.array(data['num_label_list'][0][:])
    OneHot_label_list = np.array(data['OneHot_label_list'][:]).tolist()

    if multi_progress_global == 0:
        features_examples = EMGbox_WWH.get_emg_features_from_examples(examples_emg_data=examples, methods=features_type_global)
    else:
        tasks = get_the_tasks_of_each_progress(examples)
        results = pool.map_async(get_features_multi_progress, tasks)
        pool.close()
        pool.join()
        features_examples = []
        for results in results.get():
            features_examples = features_examples + results

    outputs = {'examples': features_examples, 'num_labels': num_labels, 'OneHot_labels': OneHot_labels, 'EMG_name': EMG_name, 'angle_name': angle_name, 'angle_labels': angle_labels, 'torque_name': torque_name, 'torque_labels': torque_labels,
               'details_subject': subject, 'details_movement': movement, 'details_group': group, 'movements_label_list': movements_label_list, 'num_label_list': num_label_list, 'OneHot_label_list': OneHot_label_list}

    save_path = path_save_emg_features_to_folder_global
    save_name = 'emg_features_examples.mat'
    scipy.io.savemat(save_path + save_name, outputs)

    return outputs


def get_features_standard(data_path):
    data = scipy.io.loadmat(data_path)
    examples = np.array(data['examples']).tolist()[:]
    num_labels = list(data['num_labels'][0][:])
    OneHot_labels = np.array(data['OneHot_labels'][:]).tolist()
    subject = list(data['details_subject'][0][:])
    movement = list(data['details_movement'][:])
    group = list(data['details_group'][0][:])
    angle_labels = np.array(data['angle_labels'][:]).tolist()
    torque_labels = np.array(data['torque_labels'][:]).tolist()
    movements_label_list = list(data['movements_label_list'][:])
    num_label_list = np.array(data['num_label_list'][0][:])
    OneHot_label_list = np.array(data['OneHot_label_list'][:]).tolist()

    details_dic = {'subject': subject, 'movement': movement, 'group': group}

    features_standard = EMGbox_WWH.standardization_pre_object(data=examples, details=details_dic, standardization_setup=features_standardization_setup_global)

    outputs = {'examples': features_standard, 'num_labels': num_labels, 'OneHot_labels': OneHot_labels, 'EMG_name': EMG_name, 'angle_name': angle_name, 'angle_labels': angle_labels, 'torque_name': torque_name, 'torque_labels': torque_labels,
               'details_subject': subject, 'details_movement': movement, 'details_group': group, 'movements_label_list': movements_label_list, 'num_label_list': num_label_list, 'OneHot_label_list': OneHot_label_list}

    save_path = path_save_features_standard_to_folder_global
    save_name = 'emg_features_standard.mat'
    scipy.io.savemat(save_path + save_name, outputs)

    return outputs


def get_head_file(head={}, flag_name='None', output_path='None'):
    begin_time = output_path[-15:-1]
    if flag_name == 'flag_load_emg_data':
        files_list = os.listdir(output_path)
        files_path = [output_path + file_name for file_name in files_list]

        sub_log = str(begin_time + '     Load emg data.')
        logs = [sub_log]

        save_path = output_path + 'Head.mat'

        head['log'] = logs
        head['path'] = files_path
        head['subjects_list'] = subjects_list_global
        head['movements_list'] = movements_list_global
        head['label_type'] = label_type_global

        scipy.io.savemat(save_path, head)

    if flag_name == 'flag_emg_signal_processing':
        files_list = os.listdir(output_path)
        files_path = [output_path + file_name for file_name in files_list]

        logs = head['log']
        sub_log = str(begin_time + '     Emg signal processing.')
        logs = np.append(logs, sub_log)

        path_input = head['path']

        head['log'] = logs
        head['path'] = files_path
        head['path_input'] = path_input
        head['signal_processing_setup'] = signal_processing_setup_global

        save_path = output_path + 'Head.mat'
        scipy.io.savemat(save_path, head)

    if flag_name == 'flag_get_emg_windows':
        files_path = [output_path + 'emg_windows.mat']

        logs = head['log']
        sub_log = str(begin_time + '     Get EMG windows.')
        logs = np.append(logs, sub_log)

        path_input = head['path']

        head['log'] = logs
        head['path'] = files_path
        head['path_input'] = path_input
        head['window_size'] = window_size_global
        head['window_overlap_size'] = window_overlap_size_global
        head['data_drop_time'] = data_drop_time_setup

        save_path = output_path + 'Head.mat'
        scipy.io.savemat(save_path, head)

    if flag_name == 'flag_emg_standardization':
        files_list = os.listdir(output_path)
        files_path = [output_path + file_name for file_name in files_list]

        logs = head['log']
        sub_log = str(begin_time + '     EMG standardization.')
        logs = np.append(logs, sub_log)

        path_input = head['path']

        head['log'] = logs
        head['path'] = files_path
        head['path_input'] = path_input
        head['EMG_standardization_setup'] = EMG_standardization_setup_global

        save_path = output_path + 'Head.mat'
        scipy.io.savemat(save_path, head)

    if flag_name == 'flag_get_emg_features_from_windows':
        files_list = os.listdir(output_path)
        files_path = [output_path + file_name for file_name in files_list]

        logs = head['log']
        sub_log = str(begin_time + '     Get EMG features.')
        logs = np.append(logs, sub_log)

        path_input = head['path']

        head['log'] = logs
        head['path'] = files_path
        head['path_input'] = path_input
        head['features_types'] = features_type_global

        save_path = output_path + 'Head.mat'
        scipy.io.savemat(save_path, head)

    if flag_name == 'flag_features_standardization':
        files_list = os.listdir(output_path)
        files_path = [output_path + file_name for file_name in files_list]

        logs = head['log']
        sub_log = str(begin_time + '     Features standardization.')
        logs = np.append(logs, sub_log)

        path_input = head['path']

        head['log'] = logs
        head['path'] = files_path
        head['path_input'] = path_input
        head['features_standardization_setup'] = features_standardization_setup_global

        save_path = output_path + 'Head.mat'
        scipy.io.savemat(save_path, head)


def get_results_summary():
    files_list = os.listdir(path_save_results_to_folder)
    file_name = 'results_summary.txt'
    file_path = path_save_results_to_folder + file_name

    if file_name not in files_list:
        summary_file = open(file_path, 'w')
        summary_file.write('History Results:\n')
        summary_file.close()
    summary_file = open(file_path, 'a')

    last_head_folder_path = get_folder_path(first_flag=0)
    last_head_file_path = last_head_folder_path + 'Head.mat'
    last_head_file = scipy.io.loadmat(last_head_file_path)

    last_head_file_keys = [key for key in last_head_file.keys()]

    summary_file.write('\n' + str(get_time()) + ': \n')

    summary_file.write('    Log: \n')
    for log_check in last_head_file['log']:
        summary_file.write('        ' + str(log_check) + '\n')

    if 'subjects_list' in last_head_file_keys:
        summary_file.write('    Subjects: ')
        for subject_check in last_head_file['subjects_list'][0]:
            summary_file.write(str(subject_check) + '; ')
        summary_file.write('\n')

    if 'movements_list' in last_head_file_keys:
        summary_file.write('    Movements: ')
        for movement_check in last_head_file['movements_list']:
            summary_file.write(str(movement_check) + '; ')
        summary_file.write('\n')

    if 'label_type' in last_head_file_keys:
        summary_file.write('    Label types: ')
        summary_file.write(str(last_head_file['label_type'][0]))
        summary_file.write('\n')

    if 'signal_processing_setup' in last_head_file_keys:
        summary_file.write('    Signal processing:\n')
        summary_file.write('        Method: ')
        for method_check in last_head_file['signal_processing_setup']['method'][0][0]:
            summary_file.write(str(method_check) + '; ')
        summary_file.write('\n')
        summary_file.write('        Butter worth setup: \n')
        summary_file.write('            Low cut: ' + str(last_head_file['signal_processing_setup']['butter_worth_low_cut'][0][0][0][0]) + '\n')
        summary_file.write('            High cut: ' + str(last_head_file['signal_processing_setup']['butter_worth_high_cut'][0][0][0][0]) + '\n')
        summary_file.write('            Fs: ' + str(last_head_file['signal_processing_setup']['butter_worth_fs'][0][0][0][0]) + '\n')
        summary_file.write('            Order: ' + str(last_head_file['signal_processing_setup']['butter_worth_order'][0][0][0][0]) + '\n')

        summary_file.write('        Wavelet packet setup: \n')
        summary_file.write('            Threshold: ' + str(last_head_file['signal_processing_setup']['wavelet_packet_threshold'][0][0][0][0]) + '\n')
        summary_file.write('            Threshold mode: ' + str(last_head_file['signal_processing_setup']['wavelet_packet_threshold_mode'][0][0][0]) + '\n')
        summary_file.write('            Wavelet type: ' + str(last_head_file['signal_processing_setup']['wavelet_packet_wavelet_type'][0][0][0]) + '\n')
        summary_file.write('            Dev level: ' + str(last_head_file['signal_processing_setup']['wavelet_packet_dev_level'][0][0][0]) + '\n')

        summary_file.write('        Notch filter setup: \n')
        summary_file.write('            Frequency removed: ' + str(last_head_file['signal_processing_setup']['notch_filter_frequency_removed'][0][0][0]) + '\n')
        summary_file.write('            Quality factor: ' + str(last_head_file['signal_processing_setup']['notch_filter_quality_factor'][0][0][0]) + '\n')
        summary_file.write('            Fs: ' + str(last_head_file['signal_processing_setup']['notch_filter_fs'][0][0][0][0]) + '\n')

    if 'window_size' in last_head_file_keys and 'window_overlap_size' in last_head_file_keys:
        summary_file.write('    Window setup: \n')
        summary_file.write('        Window size: ' + str(last_head_file['window_size'][0][0]) + '\n')
        summary_file.write('        Window overlap size: ' + str(last_head_file['window_overlap_size'][0][0]) + '\n')
        summary_file.write('        data drop(ms): ' + str(last_head_file['data_drop_time'][0][0]) + '\n')

    if 'EMG_standardization_setup' in last_head_file_keys:
        summary_file.write('    EMG standardization setup: \n')
        summary_file.write('        Method:' + str(last_head_file['EMG_standardization_setup']['method'][0][0][0]) + '\n')
        summary_file.write('        Target details: ')

        for detail_check in last_head_file['EMG_standardization_setup']['target_detail'][0][0]:
            summary_file.write(str(detail_check) + '; ')
        summary_file.write('\n')

        summary_file.write('        Across :' + str(last_head_file['EMG_standardization_setup']['across_what'][0][0][0]) + '\n')

    if 'features_types' in last_head_file_keys:
        summary_file.write('    Features types: ')

        for feature_check in last_head_file['features_types']:
            summary_file.write(str(feature_check) + '; ')
        summary_file.write('\n')

    if 'features_standardization_setup' in last_head_file_keys:
        summary_file.write('    Features standardization setup: \n')
        summary_file.write('        Method:' + str(last_head_file['features_standardization_setup']['method'][0][0][0]) + '\n')
        summary_file.write('        Target details: ')

        for detail_check in last_head_file['features_standardization_setup']['target_detail'][0][0]:
            summary_file.write(str(detail_check) + '; ')
        summary_file.write('\n')

        summary_file.write('        Across :' + str(last_head_file['features_standardization_setup']['across_what'][0][0][0]) + '\n')

    summary_file.write('\n')
    summary_file.close()
    # summary_file = scipy.io.loadmat(file_path)
    # summary_file[results_summary['time_finished']] = results_summary
    # scipy.io.savemat(file_path, summary_file)


if __name__ == '__main__':
    time_start = time.time()

    if multi_progress_global != 0:
        pool = Pool(multi_progress_global)
    flag_first_task = 1
    create_results_folder()

    if flag_load_emg_data == 1:
        print('\nBegin[loading emg data]')
        path_save_emg_data_to_folder_global = create_results_folder_with_time_name(path_save_emg_data_to_folder_global)
        print('Save path: ' + path_save_emg_data_to_folder_global)
        flag_first_task = 0
        data_information, data_information_name = get_data_information(path_data_set_folder=path_data_set_folder_global[:], subjects_list=subjects_list_global[:], movements_list=movements_list_global[:], label_type=label_type_global)

        if multi_progress_global == 0:
            for data_check in range(np.size(data_information_name, 0)):
                get_emg_data_from_one_data(data_information[data_information_name[data_check]])
        else:
            pool_data_information = [data_information[data_information_name[data_check]] for data_check in range(np.size(data_information_name, 0))]
            for data_check in range(np.size(data_information_name, 0)):
                pool_data_information[data_check]['save_path'] = path_save_emg_data_to_folder_global
            pool.map(get_emg_data_from_one_data, pool_data_information)

        get_head_file(flag_name='flag_load_emg_data', output_path=path_save_emg_data_to_folder_global)
        print('Finish[loading emg data]\n')

    if flag_emg_signal_processing == 1:
        print('\nBegin[signal processing]')
        input_path = get_folder_path(first_flag=flag_first_task)
        path_save_emg_signal_processed_to_folder_global = create_results_folder_with_time_name(path_save_emg_signal_processed_to_folder_global)
        print('Save path: ' + path_save_emg_signal_processed_to_folder_global)
        flag_first_task = 0
        emg_data_head_file = scipy.io.loadmat(input_path + 'Head.mat')
        emg_data_path = emg_data_head_file['path']

        if multi_progress_global == 0:
            for data_check in range(np.size(emg_data_path)):
                signal_processing_for_one_emg_data(data_path=emg_data_path[data_check])
        else:
            inputs = []
            for data_check in range(np.size(emg_data_path, 0)):
                inputs = add_elements_in_list(inputs, [[str(emg_data_path[data_check]), str(path_save_emg_signal_processed_to_folder_global)]])
            pool.map(signal_processing_for_one_emg_data, inputs)

        get_head_file(head=emg_data_head_file, flag_name='flag_emg_signal_processing', output_path=path_save_emg_signal_processed_to_folder_global)
        print('Finish[signal processing]\n')

    if flag_get_emg_windows == 1:
        print('\nBegin[getting emg windows]')
        input_path = get_folder_path(first_flag=flag_first_task)
        path_save_emg_windows_to_folder_global = create_results_folder_with_time_name(path_save_emg_windows_to_folder_global)
        print('Save path: ' + path_save_emg_windows_to_folder_global)
        flag_first_task = 0
        emg_data_head_file = scipy.io.loadmat(input_path + 'Head.mat')
        emg_data_path = emg_data_head_file['path']

        if multi_progress_global == 0:
            for data_check in range(np.size(emg_data_path)):
                get_emg_windows_from_one_data(data_path=emg_data_path[data_check])
        else:
            inputs = []
            for data_check in range(np.size(emg_data_path, 0)):
                inputs = add_elements_in_list(inputs, [[str(emg_data_path[data_check]), str(path_save_emg_windows_to_folder_global)]])
            pool.map(get_emg_windows_from_one_data, inputs)

        input_path = get_folder_path(first_flag=0)
        print('Summary all EMG windows...')
        get_all_emg_windows(path_folder=input_path)

        get_head_file(head=emg_data_head_file, flag_name='flag_get_emg_windows', output_path=path_save_emg_windows_to_folder_global)
        print('Finish[getting emg windows]\n')

    if flag_emg_standardization == 1:
        print('\nBegin[emg standardization]')
        input_path = get_folder_path(first_flag=flag_first_task)
        path_save_emg_standardization_to_folder_global = create_results_folder_with_time_name(path_save_emg_standardization_to_folder_global)
        print('Save path: ' + path_save_emg_standardization_to_folder_global)
        flag_first_task = 0
        emg_data_head_file = scipy.io.loadmat(input_path + 'Head.mat')
        emg_data_path = emg_data_head_file['path']

        for data_check in range(np.size(emg_data_path)):
            get_emg_standard(data_path=emg_data_path[data_check])

        get_head_file(head=emg_data_head_file, flag_name='flag_emg_standardization', output_path=path_save_emg_standardization_to_folder_global)
        print('Finish[emg standardization]\n')

    if flag_get_emg_features_from_windows == 1:
        print('\nBegin[getting emg features]')
        input_path = get_folder_path(first_flag=flag_first_task)
        path_save_emg_features_to_folder_global = create_results_folder_with_time_name(path_save_emg_features_to_folder_global)
        print('Save path: ' + path_save_emg_features_to_folder_global)
        flag_first_task = 0
        emg_data_head_file = scipy.io.loadmat(input_path + 'Head.mat')
        emg_data_path = emg_data_head_file['path']

        for data_check in range(np.size(emg_data_path)):
            get_features(data_path=emg_data_path[data_check])

        get_head_file(head=emg_data_head_file, flag_name='flag_get_emg_features_from_windows', output_path=path_save_emg_features_to_folder_global)
        print('Finish[getting emg features]\n')

    if flag_features_standardization == 1:
        print('\nBegin[features standardization]')
        input_path = get_folder_path(first_flag=flag_first_task)
        path_save_features_standard_to_folder_global = create_results_folder_with_time_name(path_save_features_standard_to_folder_global)
        print('Save path: ' + path_save_features_standard_to_folder_global)
        flag_first_task = 0
        emg_data_head_file = scipy.io.loadmat(input_path + 'Head.mat')
        emg_data_path = emg_data_head_file['path']

        for data_check in range(np.size(emg_data_path)):
            get_features_standard(data_path=emg_data_path[data_check])
            print('features_standardization: ' + str(data_check + 1) + ' / ' + str(np.size(emg_data_path)))

        get_head_file(head=emg_data_head_file, flag_name='flag_features_standardization', output_path=path_save_features_standard_to_folder_global)
        print('Finish[features standardization]\n')

    if multi_progress_global != 0:
        pool.close()
        pool.join()

    get_results_summary()

    time_end = time.time()
    time_used = time_end - time_start
    print('Finish!')
    print('Time used: %0.0f s' % time_used)
