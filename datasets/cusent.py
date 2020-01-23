from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from scipy.io import wavfile
import os
# from hparams import train_hparams as hparams
# from util import audio
import pdb
from shutil import copyfile
import re

_max_out_length = 2500


def linelist_transform(spk_full_dir, line_list, spk_index):
    '''transform raw text information to our system filelist format

        Returns:
            A list of tuple which will be written as input information
    '''
    results = []

    for line in line_list:
        wav_name, text = re.split('\s+', line.strip())
        wav_path = os.path.join(spk_full_dir, wav_name)
        # futures.append(executor.submit(partial(_process_utterance, ou
        # futures.append(executor.submit(
        # partial(_process_utterance, out_dir, index, wav_path, text)))
        results.append(_process_utterance(wav_path, text, spk_index))
    return results


def build_from_path(in_dir, num_workers=1, tqdm=lambda x: x):
    '''Preprocesses the LJ Speech dataset from a given input path into a given output directory.

      Args:
        in_dir: The directory where you have downloaded the LJ Speech dataset
        num_workers: Optional number of worker processes to parallelize across
        tqdm: You can optionally pass tqdm to get a nice progress bar

      Returns:
        A list of tuples describing the training examples. This should be written to train.txt
    '''

    # We use ProcessPoolExecutor to parallize across processes. This is just an optimization and you
    # can omit it and just call _process_utterance on each input if you want.
    # executor = ProcessPoolExecutor(max_workers=num_workers)

    # futures = []
    index = 1
    spk_index = 0
    spk_id_dict = dict()
    all_T_train = []
    all_T_valid = []
    all_T_test = []
    for _, spk_dir in enumerate(os.listdir(in_dir)):
        # we only take female speaker
        # if spk_dir[-1] == "m":
            # continue
        # spk id
        spk_index = spk_index + 1
        spk_id_dict[spk_dir] = spk_index
        spk_full_dir = os.path.join(in_dir, spk_dir)
        phonetic_path = os.path.join(spk_full_dir, 'annotate/phonetic')
        with open(phonetic_path, 'r') as fid:
            transcriptions = fid.readlines()
        results = linelist_transform(spk_full_dir, transcriptions, spk_index)
        T_train, T_test = train_test_split(
            results, test_size=0.2, random_state=42)
        # split T_test as two part as T_valid and T_test
        split_T_test = len(T_test) // 2
        T_valid = T_test[0:split_T_test]
        T_test = T_test[split_T_test:]
        all_T_train.extend(T_train)
        all_T_valid.extend(T_valid)
        all_T_test.extend(T_test)
        index = index + 1
    return all_T_train, all_T_valid, all_T_test, spk_id_dict

    # results = [future.result() for future in tqdm(futures)]
    # return [r for r in results if r is not None]


def _process_utterance(wav_path, text, spk_index):
    '''Preprocesses a single utterance audio/text pair.

    This writes the mel and linear scale spectrograms to disk and returns a tuple to write
    to the train.txt file.

    Args:
      out_dir: The directory to write the spectrograms into
      index: The numeric index to use in the spectrogram filenames.
      wav_path: Path to the audio file containing the speech input
      text: The text spoken in the input audio file

    Returns:
      A (wav_path, text, spk_id) tuple to write to train.txt
    '''

    # Return a tuple describing this training example:
    sr, waveform = wavfile.read(wav_path)
    seconds = len(waveform) / sr
    return (text, wav_path, spk_index, seconds)


def generate_speaker_info():
    with open("./cusent-full-filelists/cusent_audio_text_train_filelist.txt", 'r') as tfid:
        tfiles = tfid.readlines()
    wfid = open("./cusent-full-filelists/cusent_speakerinfo.txt", 'w')
    pre_spk_id = '1'
    pre_gener = 'M'
    all_time = 0
    for tn, tfile in enumerate(tfiles):
        text, audio_path, spk_id, audio_length = re.split("\|", tfile.strip())
        gender = os.path.dirname(audio_path)[-1].upper()

        if pre_spk_id != spk_id or tn == len(tfiles) - 1:
            # start a new spk
            pre_infos = '{0} | {1} | {2}\n'.format(
                pre_spk_id, pre_gener, str(datetime.timedelta(seconds=int(all_time))))
            wfid.write(pre_infos)
            all_time = 0
        pre_spk_id = spk_id
        pre_gener = gender
        all_time += float(audio_length)
    wfid.close()


generate_speaker_info()
