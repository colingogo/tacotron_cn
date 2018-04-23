#!/usr/bin/env python
# -*- coding:utf-8 -*-
from utils import audio
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial


def build_from_path(input_dir="/media/btows/SDB/tts_dataset/train_dataset/sangzhujuan",
                    out_dir="/media/btows/SDB/tts_dataset/train_dataset/mels1", n_jobs=4, tqdm=lambda x: x):
    """
    Preprocesses the Lj speech dataset from a gven input path to a given output directory
    Args:
        - in_dir: input directory that contains the files to prerocess
        - out_dir: output drectory of the preprocessed Lj dataset
        - n_jobs: Optional, number of worker process to parallelize across
        - tqdm: Optional, provides a nice progress bar

    Returns:
        - A list of tuple describing the train examples. this should be written to train.txt
    """

    # We use ProcessPoolExecutor to parallelize across processes, this is just for
    # optimization purposes and it can be omited
    executor = ProcessPoolExecutor(max_workers=n_jobs)
    futures = []
    index = 1
    with open(os.path.join(input_dir, 'wavs.txt'), encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('<------>')
            wav_path = os.path.join(input_dir, 'wavs', '{}.wav'.format(parts[0]))
            futures.append(executor.submit(partial(_process_utterance, out_dir, parts[0], wav_path)))
            index += 1
    return [future.result() for future in tqdm(futures)]


def _process_utterance(out_dir, filename, wav_path):
    wav = audio.load_wav(wav_path)

    # Compute the mel scale spectrogram from the wav
    mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)
    n_frames = mel_spectrogram.shape[1]

    # Write the spectrogram to disk
    mel_filename = str(filename) + ".npy"
    np.save(os.path.join(out_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)

    # Return a tuple describing this training example
    return (mel_filename, n_frames)


if __name__ == "__main__":
    build_from_path()
