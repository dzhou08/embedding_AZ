
#from utils.utils import fetch_audio_files, df_to_csv, add_train_scores
from pathlib import Path
import whisper
import os
import codecs
import pandas as pd
import utils_cha_files as cha
import utils_assemblyai as uaai

import os
from dotenv import load_dotenv

import logging
import ssl
import urllib.request


# Load environment variables from .env file
load_dotenv()
logger = logging.getLogger()


# Create an unverified SSL context
ssl._create_default_https_context = ssl._create_unverified_context

# Your existing code
logger.info("Transcription done.")

def fetch_audio_files(data_path, file_extension=".wav"):

    project_path = Path(__file__).parent.resolve()
    data_dir = (project_path / data_path).resolve()

    print(f"data_dir {data_dir}")

    audio_files = []
    for root, dirs, files in os.walk(data_dir):
        for f in sorted(files):
            print(f)
            if f.endswith(file_extension):
                audio_files.append(os.path.join(root, f))
    logger.info(f"Successfully fetched {len(audio_files)} (.cha) audio files! From {data_dir}")
    return audio_files


def transcribe(method):
    logger.info("Start transcription:")

    if method == "cha":
        # path to CHA files
        train_audio_files = fetch_audio_files(os.getenv("TRAINING_CHA_DATA_PATH"), ".cha")
    elif method == "whisper":
        # path to WAV files
        train_audio_files = fetch_audio_files(os.getenv("TRAINING_WAV_DATA_PATH"), ".wav")

    #test_audio_files = fetch_audio_files(os.getenv("TESTING_DATA_PATH"))

    # Write transcriptions files
    write_transcription(train_audio_files, os.getenv("TRAINING_TRANSCRIPT_PATH"), method)
    #write_transcription(test_audio_files, os.getenv("TESTING_TRANSCRIPT_PATH"), method)

    '''
    # Scrape all transcriptions and save it to a csv file
    train_df = transcription_to_df(config.diagnosis_train_transcription_dir)
    train_df = add_train_scores(train_df)

    test_df = transcription_to_df(config.diagnosis_test_transcription_dir)

    df_to_csv(train_df, config.train_scraped_path)
    df_to_csv(test_df, config.test_scraped_path)
    '''
    logger.info("Transcription done.")


def write_transcription(audio_files, transcription_dir, method):
    print(f"audio_files {audio_files}")
    # Loop over all the audio files in the folder
    for audio_file in audio_files:
        # Get base filename
        filename = Path(audio_file).stem
        transcription_file = Path(transcription_dir) / filename
        transcription_file = transcription_file.resolve()
        print(f"transcription_file {transcription_file}")

        # Do not transcribe again if the transcription exists already
        if not transcription_file.exists():
            if method == "cha":
                #transcription using pylangacq on the .cha files
                transcription_str = cha.get_char_transcript(audio_file, 'PAR')
            elif method == "whisper":
                #transcription using WHISPER
                whisper_model_name = os.getenv("WHISPER_MODEL_NAME")
                if not whisper_model_name:
                    raise ValueError("WHISPER_MODEL_NAME environment variable is not set")
                print(whisper_model_name)
                whisper_model = whisper.load_model(whisper_model_name)
                
                print(f"audio_file {audio_file}")
                result = whisper_model.transcribe(audio_file, fp16=False)
                transcription_str = str(result["text"])
            elif method == "assemblyai":
                # transcription using AssemblyAI
                transcription_str = uaai.process_audio_file(audio_file)

            # Create subdirs if not existent
            transcription_file.parent.mkdir(parents=True, exist_ok=True)

            transcription_file.write_text(transcription_str)
            logger.info(f"Transcribed {transcription_file}...")


def transcription_to_df(data_dir):
    """
    Transforms transcriptions from text files into a DataFrame.

    Parameters:
        data_dir (str): The directory containing transcription files.

    Returns:
        pd.DataFrame: A DataFrame with columns 'addressfname' and 'transcript'.
    """
    texts = []

    # Traverse through the directory to fetch transcription files
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            # Read the content of the file
            with codecs.open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
                texts.append((file, text))

    # Create a DataFrame from the list of texts
    df = pd.DataFrame(texts, columns=['addressfname', 'transcript'])

    # Clean up the transcript column by removing newlines and extra spaces
    df['transcript'] = df['transcript'].str.replace('\n', ' ').replace('\\n', ' ').replace('  ', ' ')

    # Sort the DataFrame by the 'addressfname' column in ascending order
    df = df.sort_values(by='addressfname')

    # Reset the index
    df = df.reset_index(drop=True)

    # Debugging: Print the resulting DataFrame
    logger.debug(df)

    return df
