import os
import pydub
import sounddevice as sd
import numpy as np
import tkinter as tk
from tkinter import filedialog
import scipy.io.wavfile as wavfile
from functools import partial
import librosa
import scipy.signal as sig
import psola
import matplotlib.pyplot as plt
import threading
import time

def play_background_music(audio, sample_rate):
    print("Playing background music...")
    sd.play(audio, samplerate=sample_rate)

def record_voice(duration=40, sample_rate=44100):
    # Load the background music
    print("Loading background music...")
    bg_music_file = get_instrumental_file()
    bg_music, bg_sr = librosa.load(bg_music_file, sr=sample_rate, mono=True)
    bg_music_duration = librosa.get_duration(y=bg_music)
    print("Background music loaded.")

    print(f"Recording voice for {duration} seconds...")

    # Start playing the background music
    bg_music_thread = threading.Thread(target=play_background_music, args=(bg_music, sample_rate))
    bg_music_thread.start()

    # Record the voice
    voice = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    
    # Stop the background music if it's still playing
    sd.stop()

    print("Finished recording.")

    output_file = "recorded_voice.wav"
    wavfile.write(output_file, sample_rate, voice)

    return voice

selected_instrumental_file = None

def get_instrumental_file():
    global selected_instrumental_file

    if selected_instrumental_file is not None:
        return selected_instrumental_file

    # Open a file dialog box for the user to select an instrumental file
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select instrumental file", filetypes=[("Audio files", "*.mp3;*.wav")])

    # Save the file path for future use
    selected_instrumental_file = file_path

    # Load the instrumental file using pydub
    instrumental = pydub.AudioSegment.from_file(file_path)

    # Test the instrumental file by playing it back
    instrumental.export("test_instrumental.wav", format="wav")
    return file_path

# def get_instrumental_file():
#     # Use the hardcoded instrumental file
#     file_path = "imyours.mp3"

#     # Load the instrumental file using pydub
#     instrumental = pydub.AudioSegment.from_file(file_path)

#     # Test the instrumental file by playing it back
#     instrumental.export("test_instrumental.wav", format="wav")

# def get_instrumental_file():
#     # Open a file dialog box for the user to select an instrumental file
#     root = tk.Tk()
#     root.withdraw()
#     file_path = filedialog.askopenfilename(title="Select instrumental file", filetypes=[("Audio files", "*.mp3;*.wav")])

#     # Load the instrumental file using pydub
#     instrumental = pydub.AudioSegment.from_file(file_path)

#     # Test the instrumental file by playing it back
#     instrumental.export("test_instrumental.wav", format="wav")

SEMITONES_IN_OCTAVE = 12

def degrees_from(scale: str):
    """Return the pitch classes (degrees) that correspond to the given scale"""
    degrees = librosa.key_to_degrees(scale)
    # To properly perform pitch rounding to the nearest degree from the scale, we need to repeat
    # the first degree raised by an octave. Otherwise, pitches slightly lower than the base degree
    # would be incorrectly assigned.
    degrees = np.concatenate((degrees, [degrees[0] + SEMITONES_IN_OCTAVE]))
    return degrees

def closest_pitch(f0):
    """Round the given pitch values to the nearest MIDI note numbers"""
    midi_note = np.around(librosa.hz_to_midi(f0))
    # To preserve the nan values.
    nan_indices = np.isnan(f0)
    midi_note[nan_indices] = np.nan
    # Convert back to Hz.
    return librosa.midi_to_hz(midi_note)

def closest_pitch_from_scale(f0, scale):
    """Return the pitch closest to f0 that belongs to the given scale"""
    # Preserve nan.
    if np.isnan(f0):
        return np.nan
    degrees = degrees_from(scale)
    midi_note = librosa.hz_to_midi(f0)
    # Subtract the multiplicities of 12 so that we have the real-valued pitch class of the
    # input pitch.
    degree = midi_note % SEMITONES_IN_OCTAVE
    # Find the closest pitch class from the scale.
    degree_id = np.argmin(np.abs(degrees - degree))
    # Calculate the difference between the input pitch class and the desired pitch class.
    degree_difference = degree - degrees[degree_id]
    # Shift the input MIDI note number by the calculated difference.
    midi_note -= degree_difference
    # Convert to Hz.
    return librosa.midi_to_hz(midi_note)


def aclosest_pitch_from_scale(f0, scale):
    """Map each pitch in the f0 array to the closest pitch belonging to the given scale."""
    sanitized_pitch = np.zeros_like(f0)
    for i in np.arange(f0.shape[0]):
        sanitized_pitch[i] = closest_pitch_from_scale(f0[i], scale)
    # Perform median filtering with a larger kernel to additionally smooth the corrected pitch.
    smoothed_sanitized_pitch = sig.medfilt(sanitized_pitch, kernel_size=15)
    # Remove the additional NaN values after median filtering.
    smoothed_sanitized_pitch[np.isnan(smoothed_sanitized_pitch)] = \
        sanitized_pitch[np.isnan(smoothed_sanitized_pitch)]
    return smoothed_sanitized_pitch


def autotune(audio, sr, correction_function, plot=False):
    # Set some basis parameters.
    frame_length = 2048
    hop_length = frame_length // 4
    fmin = librosa.note_to_hz('C2')
    fmax = librosa.note_to_hz('C7')

    # Pitch tracking using the PYIN algorithm.
    f0, voiced_flag, voiced_probabilities = librosa.pyin(audio,
                                                         frame_length=frame_length,
                                                         hop_length=hop_length,
                                                         sr=sr,
                                                         fmin=fmin,
                                                         fmax=fmax)

    # Apply the chosen adjustment strategy to the pitch.
    corrected_f0 = correction_function(f0)

    if plot:
        # Plot the spectrogram, overlaid with the original pitch trajectory and the adjusted
        # pitch trajectory.
        stft = librosa.stft(audio, n_fft=frame_length, hop_length=hop_length)
        time_points = librosa.times_like(stft, sr=sr, hop_length=hop_length)
        log_stft = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        fig, ax = plt.subplots()
        img = librosa.display.specshow(log_stft, x_axis='time', y_axis='log', ax=ax, sr=sr, hop_length=hop_length, fmin=fmin, fmax=fmax)
        fig.colorbar(img, ax=ax, format="%+2.f dB")
        ax.plot(time_points, f0, label='original pitch', color='cyan', linewidth=2)
        ax.plot(time_points, corrected_f0, label='corrected pitch', color='orange', linewidth=1)
        ax.legend(loc='upper right')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [M:SS]')
        plt.savefig('pitch_correction.png', dpi=300, bbox_inches='tight')

    # Pitch-shifting using the PSOLA algorithm.
    return psola.vocode(audio, sample_rate=int(sr), target_pitch=corrected_f0, fmin=fmin, fmax=fmax)

def auto_tune_voice(voice_file, instrumental_file, output_file, correction_method="closest", scale=None, plot=False):
    voice, sr = librosa.load(voice_file, sr=None, mono=True)
    correction_function = closest_pitch if correction_method == "closest" else \
        partial(aclosest_pitch_from_scale, scale=scale)
    pitch_corrected_voice = autotune(voice, sr, correction_function, plot)
    wavfile.write(output_file, sr, pitch_corrected_voice.astype(np.float32))

    # New function to overlay the autotuned voice with the instrumental
    combined_output_file = "combined_output.wav"
    overlay_autotuned_voice_with_instrumental(output_file, instrumental_file, combined_output_file)

# New function to overlay the autotuned voice with the instrumental
def overlay_autotuned_voice_with_instrumental(voice_file, instrumental_file, output_file):
    voice = pydub.AudioSegment.from_wav(voice_file)
    instrumental = pydub.AudioSegment.from_wav(instrumental_file)
    combined = voice.overlay(instrumental)
    combined.export(output_file, format="wav")

def overlay_original_voice_with_instrumental(voice_file, instrumental_file, output_file):
    voice = pydub.AudioSegment.from_wav(voice_file)
    instrumental = pydub.AudioSegment.from_wav(instrumental_file)
    combined = voice.overlay(instrumental)
    combined.export(output_file, format="wav")

def save_file(voice, instrumental, output_file):
    # Function to save the combined file into the same directory as the Python app
    pass

def main():
    # Record user's voice
    print("Starting voice recording...")
    voice = record_voice()
    print("Finished voice recording.")

    # Retrieve an instrumental file
    print("Getting instrumental file...")
    instrumental = get_instrumental_file()
    print("Got instrumental file.")

    # Add a wait time before starting the autotune process
    print("Waiting for 5 seconds before starting the auto-tuning process...")
    time.sleep(5)  # Wait for 5 seconds

    # Auto-tune the voice to the instrumental
    print("Starting auto-tuning process...")
    voice_file = "recorded_voice.wav"
    instrumental_file = "test_instrumental.wav"
    output_file = "output.wav"
    auto_tune_voice(voice_file, instrumental_file, output_file)
    print("Finished auto-tuning process.")

    # New call to overlay the original voice with the instrumental
    print("Creating combined_output_with_original.wav file...")
    combined_output_with_original_file = "combined_output_with_original.wav"
    overlay_original_voice_with_instrumental(voice_file, instrumental_file, combined_output_with_original_file)
    print("Finished creating combined_output_with_original.wav file.")
    print("All done!")

if __name__ == "__main__":
    main()