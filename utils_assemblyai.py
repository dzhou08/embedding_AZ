# `pip3 install assemblyai` (macOS)
# `pip install assemblyai` (Windows)

import os
from dotenv import load_dotenv
import re

import assemblyai as aai

def process_transcript_string(transcript):
    """
    Process a transcript to return all words if there is only one speaker;
    for more than one speaker, return the concatenated words for the second speaker.

    Parameters:
        transcript (str): The input transcript with speaker labels.

    Returns:
        str: Concatenated words for all speakers if one speaker exists,
             or concatenated words of the second speaker if multiple speakers exist.
    """
    # Parse the transcript into speaker and spoken words
    speaker_words = {}
    lines = transcript.splitlines()

    for line in lines:
        match = re.match(r"^(Speaker \w+): (.+)$", line)
        if match:
            speaker = match.group(1)
            words = match.group(2)
            if speaker not in speaker_words:
                speaker_words[speaker] = []
            speaker_words[speaker].append(words)

    # Check if there's only one speaker
    if len(speaker_words) == 1:
        return " ".join(next(iter(speaker_words.values())))

    # If multiple speakers, return words for the second speaker
    if len(speaker_words) > 1:
        second_speaker = list(speaker_words.keys())[1]
        return " ".join(speaker_words[second_speaker])

    return ""  # Return empty if no speakers are found

def process_audio_file(file_url):
    # get the key from .env file
    #load_dotenv()
    aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
    transcriber = aai.Transcriber()

    #transcript = transcriber.transcribe("https://assembly.ai/news.mp4")
    #file_url="ADReSSo_2020/ADReSS-IS2020-data-train/train/Full_wave_enhanced_audio/cd/S082.wav"
    #file_url="ADReSSo_2020/ADReSS-IS2020-data-train/train/Full_wave_enhanced_audio/cc/S005.wav"

    config = aai.TranscriptionConfig(speaker_labels=True)

    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(
        file_url,
        config=config
    )

    transcript_string = ''
    for utterance in transcript.utterances:
        print(f"Speaker {utterance.speaker}: {utterance.text}")
        transcript_string += f"Speaker {utterance.speaker}: {utterance.text}\n"

    # Process the transcript and return the first speaker's words
    result = process_transcript_string(transcript_string)
    print("Concatenated Words:", result)
    return result
