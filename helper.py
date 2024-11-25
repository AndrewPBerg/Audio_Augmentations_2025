from audiomentations.core.transforms_interface import BaseWaveformTransform
from numpy.typing import NDArray
import random
import numpy as np
from audiomentations.core.utils import calculate_rms
import torchaudio
import IPython.display as ipd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import torchaudio.transforms as T
from audiomentations import (
    AddGaussianNoise,
    AddGaussianSNR,
    Gain,
    GainTransition,
    HighPassFilter,
    LowPassFilter,
    BandPassFilter,
    BandStopFilter,
    ClippingDistortion,
    Clip,
    Normalize,
    PitchShift,
    PolarityInversion,
    Resample,
    Reverse,
    Shift,
    TanhDistortion,
    TimeMask,
    TimeStretch,
    Trim,
    Limiter,
    Mp3Compression,

)

def load_wav(path):
    waveform, sample_rate = torchaudio.load(path)
    return waveform.numpy(), sample_rate

def plot_waveforms_comparison(original_audio, augmented_audio, sr, title1="Original", title2="Augmented", color1='blue', color2='red'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Calculate the maximum amplitude across both audio signals
    max_amplitude = max(np.max(np.abs(original_audio)), np.max(np.abs(augmented_audio)))

    # Plot original audio
    librosa.display.waveshow(original_audio, sr=sr, ax=ax1, color=color1)
    ax1.set_title(title1)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Amplitude')
    ax1.set_ylim(-max_amplitude, max_amplitude)

    # Plot augmented audio
    librosa.display.waveshow(augmented_audio, sr=sr, ax=ax2, color=color2)
    ax2.set_title(title2)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Amplitude')
    ax2.set_ylim(-max_amplitude, max_amplitude)

    plt.tight_layout()
    plt.show()

def plot_waveforms_overlay(original_audio, augmented_audio, sr, title="Original vs Augmented"):
    fig, ax = plt.subplots(figsize=(10, 4))

    # Calculate the maximum amplitude across both audio signals
    max_amplitude = max(np.max(np.abs(original_audio)), np.max(np.abs(augmented_audio)))

    # Plot original audio
    librosa.display.waveshow(original_audio, sr=sr, ax=ax, color='blue', alpha=0.7, label='Original')

    # Plot augmented audio
    librosa.display.waveshow(augmented_audio, sr=sr, ax=ax, color='red', alpha=0.5, label='Augmented')

    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    ax.set_ylim(-max_amplitude, max_amplitude)
    ax.legend()

    plt.tight_layout()
    plt.show()

def apply_and_visualize_transform(transform, file_name=FILE_NAME):
    # Load audio and sample rate (sr) and display
    audio, sr = load_wav(file_name)
    print("Original Audio:")
    display(ipd.Audio(audio, rate=sr))

    # Transform audio and display
    augmented_audio = transform(audio, sample_rate=int(sr))
    print("Augmented Audio:")
    display(ipd.Audio(augmented_audio, rate=sr))

    # Cross correlate audio samples
    max_corr, lag = cross_correlation(audio, augmented_audio)

    print(f"Maximum correlation: {max_corr:.4f}")
    print(f"Lag: {lag} samples")
    print(f"Lag Time: {lag/sr:.4f} seconds")

    # Plot the two .wav samples
    plot_waveforms_comparison(audio, augmented_audio, sr, color1='navy', color2='darkred')

    # Plot the overlay comparison
    plot_waveforms_overlay(audio, augmented_audio, sr)

def cross_correlation(audio1, audio2):
    """
    Compute the cross-correlation between two audio signals.

    :param audio1: First audio signal (numpy array)
    :param audio2: Second audio signal (numpy array)
    :return: Maximum correlation value and the corresponding lag
    """
    # Ensure both arrays are 1D
    audio1 = audio1.flatten()
    audio2 = audio2.flatten()

    # Compute cross-correlation
    correlation = np.correlate(audio1, audio2, mode='full')

    # Find the index of maximum correlation
    max_corr_index = np.argmax(np.abs(correlation))

    # Calculate the lag
    lag = max_corr_index - (len(audio1) - 1)

    # Normalize the maximum correlation value
    max_corr_value = correlation[max_corr_index] / (np.linalg.norm(audio1) * np.linalg.norm(audio2))

    return max_corr_value, lag




class SinDistortion(BaseWaveformTransform):

    supports_multichannel = True

    def __init__(
        self, min_distortion: float = 0.01, max_distortion: float = 0.7, p: float = 0.5
    ):
        """
        :param min_distortion: Minimum amount of distortion (between 0 and 1)
        :param max_distortion: Maximum amount of distortion (between 0 and 1)
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        assert 0 <= min_distortion <= 1
        assert 0 <= max_distortion <= 1
        assert min_distortion <= max_distortion
        self.min_distortion = min_distortion
        self.max_distortion = max_distortion

    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters["distortion_amount"] = random.uniform(
                self.min_distortion, self.max_distortion
            )

    def apply(self, samples: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:
        # Find out how much to pre-gain the audio to get a given amount of distortion
        percentile = 100 - 99 * self.parameters["distortion_amount"]
        threshold = np.percentile(np.abs(samples), percentile)
        gain_factor = 0.5 / (threshold + 1e-6)

        # Distort the audio
        distorted_samples = np.sin(gain_factor * samples)

        # Scale the output so its loudness matches the input
        rms_before = calculate_rms(samples)
        if rms_before > 1e-9:
            rms_after = calculate_rms(distorted_samples)
            post_gain = rms_before / rms_after
            distorted_samples = post_gain * distorted_samples

        return distorted_samples


