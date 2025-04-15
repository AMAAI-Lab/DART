import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepspeaker import embedding
from ge2e.inference import load_model
import librosa
import numpy as np

class PreDefinedEmbedder(nn.Module):
    """ Speaker Embedder Wrapper """

    def __init__(self, config):
        super(PreDefinedEmbedder, self).__init__()
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.win_length = config["preprocessing"]["stft"]["win_length"]
        self.embedder_type = config["preprocessing"]["speaker_embedder"]
        self.embedder_cuda = config["preprocessing"]["speaker_embedder_cuda"]
        self.embedder = self._get_speaker_embedder()

    def _get_speaker_embedder(self):
        embedder = None
        if self.embedder_type == "DeepSpeaker":
            embedder = embedding.build_model(
                "./deepspeaker/pretrained_models/ResCNN_triplet_training_checkpoint_265.h5"
            )
        elif self.embedder_type == "GE2E":
            # device = torch.device('cuda:0')
            embedder = load_model('./ge2e/encoder.pt')
        else:
            raise NotImplementedError
        return embedder

    def forward(self, audio):
        if self.embedder_type == "DeepSpeaker":
            spker_embed = embedding.predict_embedding(
                self.embedder,
                audio,
                self.sampling_rate,
                self.win_length,
                self.embedder_cuda
            )
        elif self.embedder_type == "GE2E":
            #wav to mel
            _device=torch.device('cuda:0')
            mel_window_length = 25  # In milliseconds
            mel_window_step = 10    # In milliseconds
            mel_n_channels = 40
            sampling_rate = 16000
            frames = librosa.feature.melspectrogram(
                audio,
                sampling_rate,
                n_fft=int(sampling_rate * mel_window_length / 1000),
                hop_length=int(sampling_rate * mel_window_step / 1000),
                n_mels=mel_n_channels
            )
            frames = frames.astype(np.float32).T

            frames = torch.from_numpy(frames).to(_device)
            frames = frames.unsqueeze(0)
            spker_embed = self.embedder.forward(frames).detach().cpu().numpy()
                # embed_frames_batch
        return spker_embed
