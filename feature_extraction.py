import os
import librosa

from statistics import  mean, stdev

def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn

DESTINATION_FOLDER = "data/"

def percentage(part, whole):
  return 100 * float(part)/float(whole)

## Extract MFCC features from audio dataset
with open(os.path.join(DESTINATION_FOLDER, "all_filepaths.txt")) as audio_paths, open(os.path.join("mfcc_features.txt"), 'a') as features_file:
    entries = audio_paths.readlines()
    number_of_samples = len(entries)
    number_of_parsed_samples = 23874
    for audio_path in entries[number_of_parsed_samples-1:]:
        print("parsing sample %i/%i (%.2f%%)" % (number_of_parsed_samples, number_of_samples, percentage(number_of_parsed_samples, number_of_samples)))
        y, sr = librosa.load(audio_path.strip())
        mfccs = librosa.feature.mfcc(y=y, sr=sr)
        first_derivative = librosa.feature.delta(mfccs, order=1)
        second_derivative = librosa.feature.delta(mfccs, order=2)
        mffc_features = []

        for the_mfccs in [mfccs, first_derivative, second_derivative]:
            for coef in mfccs:
                coef = map(float, coef)
                mffc_features.extend([mean(coef), stdev(coef)])

        features_file.write('%s\n' % ' '.join([str(round(i, 5)) for i in mffc_features]))
        number_of_parsed_samples+=1
