from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
from pyannote.core import Segment
import torch

# 1. visit hf.co/pyannote/speaker-diarization and accept user conditions
# 2. visit hf.co/pyannote/segmentation and accept user conditions
# 3. visit hf.co/settings/tokens to create an access token
# 4. instantiate pretrained speaker diarization pipeline
from pyannote.audio import Pipeline

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

# Function to extract and analyze MFCC for each speaker
def extract_mfcc(y, sr, start, end, speaker):
    # Extract segment of the speaker's speech
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    y_segment = y[start_sample:end_sample]

    # Compute MFCCs
    mfccs = librosa.feature.mfcc(y=y_segment, sr=sr, n_mfcc=13)

    # Plot MFCC
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time', sr=sr)
    plt.colorbar(label="MFCC Coefficients")
    plt.title(f"MFCCs for {speaker} (Segment {start:.2f}s - {end:.2f}s)")
    plt.xlabel("Time")
    plt.ylabel("MFCC Coefficients")
    plt.show()


# Load pre-trained speaker diarization model
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token="PLACEHOLDER")

# Apply diarization to an audio file
WAV_PATH="ADReSSo_2020/ADReSS-IS2020-data-train/train/Full_wave_enhanced_audio/cd/"
audio_file = WAV_PATH + "S079.wav"

# Load full audio file
y, sr = librosa.load(audio_file, sr=None)


diarization = pipeline(audio_file)

# Store speaker segments
speaker_segments = {}
for turn, _, speaker in diarization.itertracks(yield_label=True):
    if speaker not in speaker_segments:
        speaker_segments[speaker] = []
    speaker_segments[speaker].append((turn.start, turn.end))

# Print detected speaker timestamps
for speaker, segments in speaker_segments.items():
    print(f"Speaker {speaker}:")
    for start, end in segments:
        print(f"  Speaks from {start:.2f}s to {end:.2f}s")
        extract_mfcc(y, sr, start, end, speaker)
    
    for idx, (start, end) in enumerate(segments):
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        y_segment = y[start_sample:end_sample]

        output_filename = f"speaker_{speaker}_segment_{idx}.wav"
        sf.write(output_filename, y_segment, sr)
        print(f"Saved: {output_filename}")
