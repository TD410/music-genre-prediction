import numpy as np
import tensorflow as tf
import random as rn
import os

RANDOM_SEED = 12345
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
rn.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)

from keras.models import *
import math
import librosa
import csv
import sklearn

RAW_AUDIO_DIR = "Data/genres/"
MELSPEC_DIR = "Melspec/genres/"
FEATURE_DIR = "Feature/genres/"
FEATURE_NORM_DIR = "Feature/genresNorm/"
FEATURE_5_DIR = "Feature/genres5features/"
FEATURE_5_NORM_DIR = "Feature/genres5featuresNorm/"
COMBINED_INPUT_DIR = 'CombinedInput/genres/'
BEST_MODEL_DIR = "Models/BestModel/"
AUDIO_FILE_EXTENSION = ".au"
AUDIO_DURATION = 29.12

OUTLIER_THRESHOLD = 6


'''Generate melspectrogram'''

def generate_all_melspec():
    for directories in os.listdir(RAW_AUDIO_DIR):
        num_files = 0
        src = os.path.join(RAW_AUDIO_DIR, directories)
        des = os.path.join(MELSPEC_DIR, directories)
        for filename in os.listdir(src):
            if filename.endswith(AUDIO_FILE_EXTENSION):
                num_files += 1
                try:
                    log_S = get_melspec_from_audio(os.path.join(src, filename))
                    np.save(os.path.join(des, filename), log_S)
                except:
                    continue
        print(str(num_files) + " files are completed at " + src)

def get_melspec_from_audio(path, get_full_melspec=False):
    if get_full_melspec:
        y, sr = librosa.load(path)
    else:
        y, sr = librosa.load(path, duration=AUDIO_DURATION)
    y_12k = librosa.resample(y, sr, 12000)
    S = librosa.feature.melspectrogram(y_12k, sr=12000, n_mels=96, hop_length=256)
    log_S = librosa.power_to_db(S, ref=np.max)
    return log_S


'''Generate feature vector'''

def generate_all_feature_vector():
    failed_files = []
    for directories in os.listdir(RAW_AUDIO_DIR):
        num_files = 0
        src = os.path.join(RAW_AUDIO_DIR, directories)
        des = os.path.join(FEATURE_DIR, directories)
        for filename in os.listdir(src):
            if filename.endswith(AUDIO_FILE_EXTENSION):
                num_files += 1
                try:
                    audio_file_path = os.path.join(src, filename)
                    if num_files > 70:
                        melspec_file_path = os.path.join(MELSPEC_DIR + '/test/' + directories, filename + '.npy')
                    else:
                        melspec_file_path = os.path.join(MELSPEC_DIR + '/train/' + directories, filename + '.npy')
                    features = extract_feature_vector(audio_file_path, melspec_file_path)
                    np.save(os.path.join(des, filename), features)
                except IOError:
                    failed_files.append(filename)
                    print('Failed ' + filename)

        print(str(num_files) + " files are completed at " + des)
    print(failed_files)

def extract_feature_vector(audio_path, spectrogram_path):
    y, sr = librosa.load(audio_path)
    s = np.load(spectrogram_path)
    s = np.absolute(s)

    list_x = []
    range_i = int(s.shape[1] / 1366)
    sample_range = librosa.time_to_samples(29.12, sr)

    for i in range(range_i):
        concat_s = s[:, i * 1366:i * 1366 + 1366]
        concat_y = y[i * sample_range:i * sample_range + sample_range]

        spectral_centroid = librosa.feature.spectral_centroid(S=concat_s)[0]
        roll_off = librosa.feature.spectral_rolloff(S=concat_s)[0]
        zcr = librosa.feature.zero_crossing_rate(y=concat_y)[0]
        flux = librosa.onset.onset_strength(S=concat_s)

        energy = librosa.feature.rmse(S=concat_s)[0]
        mean_energy = np.mean(energy)
        low_energy = (energy < mean_energy).sum() / len(energy)

        tempo, beats = librosa.beat.beat_track(y=concat_y, sr=sr, units='frames')
        beats_strength = np.zeros(len(beats))
        for i in range(len(beats)):
            if beats[i] < len(energy):
                beats_strength[i] = np.abs(energy[beats[i]])

        beat_strength_ratio = np.mean(beats_strength) / mean_energy

        x = [np.mean(spectral_centroid), np.var(spectral_centroid),
                np.mean(roll_off), np.var(roll_off),
                np.mean(flux), np.var(flux),
                np.mean(zcr), np.var(zcr),
                np.mean(beats_strength), np.var(beats_strength),
                beat_strength_ratio,
                low_energy, tempo]
        list_x.append(x)

    mean_x = np.array(list_x).mean(0)
    return mean_x


'''Generate 5 feature vector'''

def generate_all_5_feature_vector():
    for directories in os.listdir(FEATURE_DIR):
        src = os.path.join(FEATURE_DIR, directories)
        des = os.path.join(FEATURE_5_DIR, directories)
        for filename in os.listdir(src):
            if filename.endswith('.npy'):
                feature = extract_5_feature_vector(os.path.join(src, filename))
                np.save(os.path.join(des, filename), feature)

def extract_5_feature_vector(feature_path):
    s = np.load(feature_path)
    return [s[2], s[3], s[6], s[9], s[12]]


'''Normalize feature vector'''

def normalize(x, min, max):
    return (x-min) / (max-min)

def normalize_feature_vector(exclude_outlier=False):
    failed_read_files = []
    failed_write_files = []
    all_features = []
    all_features_norm_dir = []
    outliers = []
    if exclude_outlier:
        outliers = read_outlier_csv()
    for directories in os.listdir(FEATURE_DIR):
        num_files = 0
        src = os.path.join(FEATURE_DIR, directories)
        des = os.path.join(FEATURE_NORM_DIR, directories)
        for filename in os.listdir(src):
            if filename in outliers:
                continue
            num_files += 1
            try:
                feature = np.load(os.path.join(src, filename))
                all_features.append(feature)
                all_features_norm_dir.append(os.path.join(des, filename))
            except IOError:
                failed_read_files.append(filename)
                print('Failed to read ' + filename)

        print(str(num_files) + " files are read at " + src)

    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    all_features_norm = min_max_scaler.fit_transform(all_features)
    print("Data min:")
    print(min_max_scaler.data_min_)
    print("Data max:")
    print(min_max_scaler.data_max_)

    for i in range(len(all_features_norm)):
        try:
            np.save(all_features_norm_dir[i], all_features_norm[i])
        except IOError:
            failed_write_files.append(all_features_norm_dir[i])
            print('Failed to write ' + all_features_norm_dir[i])

    print(failed_read_files)
    print(failed_write_files)

def read_outlier_csv():
    outlier_list = []
    outlier_number = np.zeros(13)
    with open ("Outliers.csv", "rt") as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            genre = row[0]
            idx = row[1]
            outlier = row[2].split(';')
            if len(outlier) >= OUTLIER_THRESHOLD:
                filename = genre + "." + idx.zfill(5) + ".au.npy"
                outlier_list.append(filename)
                outlier_number[len(outlier)] += 1

    print(str(len(outlier_list)) + " outlier files.")

    for i in range(13):
        n = outlier_number[i]
        if n > 0:
            print(str(n) + " files with " + str(i) + " outliers.")

    return outlier_list


'''Make predictions and write to csv'''

def create_predict_result_file_list(path):
    file_list = []
    idx = 0
    for dir in os.listdir(path):
        folder = os.path.join(path, dir)
        for filename in os.listdir(folder):
            if filename.endswith('.npy'):
                filename = os.path.join(folder, filename)
                file_list.append([filename, idx])
        idx += 1
    return file_list

def create_predict_result_csv(path, model):
    file_list = create_predict_result_file_list(path)
    textfile = open('trainByPredict.csv', 'a+')
    textfile.write('Filename, Real fact, Predict, Is Correct, Blues, Classical, Country, Disco, Hiphop, Jazz, Metal, Pop, Reggae, Rock\n')
    for file in file_list:
        narray = np.load(file[0])
        narray = narray.reshape((1,) + narray.shape)
        result = model.predict(narray)
        max_result = model.predict_classes(narray)
        #string = np.array_str(result, precision=5, suppress_small=True)
        isCorrect = 1 if file[1] == max_result[0] else 0
        textfile.write(file[0] + ','
                       + str(file[1]) + ','
                       + str(max_result[0]) + ','
                       + str(isCorrect) + ','
                       + str(result[0][0]) + ','
                       + str(result[0][1]) + ','
                       + str(result[0][2]) + ','
                       + str(result[0][3]) + ','
                       + str(result[0][4]) + ','
                       + str(result[0][5]) + ','
                       + str(result[0][6]) + ','
                       + str(result[0][7]) + ','
                       + str(result[0][8]) + ','
                       + str(result[0][9])
                       + '\n')

    textfile.close()

def manual_test_cnn_model(model, audio_path, melspec_path=None):
    if melspec_path is None:
        melspec_path = audio_path + '.npy'

    if not os.path.isfile(melspec_path):
        x = get_melspec_from_audio(audio_path)
        np.save(audio_path, x)
    else:
        x = np.load(melspec_path)

    list_y = []
    range_i = int(x.shape[1] / 1366)
    for i in range (range_i):
        concat_x = x[:,i*1366:i*1366 + 1366]
        concat_x = concat_x.reshape((1,) + concat_x.shape + (1,))
        y = model.predict(concat_x)
        list_y.append(y)

    print('Predict result by CNN Model:')
    mean_y = np.array(list_y).mean(0)
    print(np.array_str(mean_y, precision=5, suppress_small=True))
    return mean_y

def manual_test_feature_model(model, audio_path, melspec_path=None):
    if melspec_path is None:
        melspec_path = audio_path + '.npy'

    if not os.path.isfile(melspec_path):
        S = get_melspec_from_audio(audio_path)
        np.save(audio_path, S)

    vector = extract_feature_vector(audio_path, melspec_path)
    norm_vector = np.zeros(13)

    min_vector = [5.59803988e+03, 4.38158818e+03, 9.25651422e+03, 2.80036453e+03,
                 3.82178955e-01, 1.07484543e-02, 2.16969832e-02, 4.40867877e-05,
                 2.45619258e+01, 8.17447162e-01, 7.70444102e-01, 3.44802343e-01,
                 5.49783910e+01]
    max_vector = [6.58959577e+03, 1.17003701e+05, 1.00941152e+04, 7.48560448e+04,
                1.64673759e+00, 3.69819273e+00, 2.74630825e-01, 2.87184039e-02,
                6.31066910e+01, 2.10971800e+02, 1.06877205e+00, 7.61346999e-01,
                2.34907670e+02]

    for i in range(13):
        min = min_vector[i]
        max = max_vector[i]
        norm_vector[i] = (vector[i] - min)/(max - min)

    feature = norm_vector.reshape((1,) + norm_vector.shape)
    y = model.predict(feature)
    print('Predict result by Feature Model:')
    print(np.array_str(y, precision=5, suppress_small=True))
    return y

feature_model = None
cnn_model = None
combine_model = None

def load_all_model():
    global feature_model, cnn_model, combine_model
    feature_model = load_model(BEST_MODEL_DIR + "featureModel2.h5")
    cnn_model = load_model(BEST_MODEL_DIR + "cnnModelA2Dropout80.h5")
    combine_model = load_model(BEST_MODEL_DIR + "combineModel1.h5")

def manual_test(audio_path, melspec_path=None):
    if feature_model is None or cnn_model is None or combine_model is None:
        load_all_model()

    if melspec_path is None:
        melspec_path = audio_path + '.npy'

    if not os.path.isfile(melspec_path):
        S = get_melspec_from_audio(audio_path)
        np.save(audio_path, S)

    feature_y = manual_test_feature_model(feature_model, audio_path, melspec_path)
    cnn_y = manual_test_cnn_model(cnn_model, audio_path, melspec_path)
    combine_x = np.concatenate((feature_y[0], cnn_y[0]))
    combine_y = combine_model.predict(combine_x.reshape((1,) + combine_x.shape))
    print('Predict result by Combine Model:')
    print(np.array_str(combine_y, precision=5, suppress_small=True))
    return feature_y[0], cnn_y[0], combine_y[0]