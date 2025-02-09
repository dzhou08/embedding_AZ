

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
import re
from sklearn.utils import resample
from matplotlib import pyplot as plt



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
            if f.endswith(file_extension):
                print(f)
                audio_files.append(os.path.join(root, f))
    logger.info(f"Successfully fetched {len(audio_files)} (.cha) audio files! From {data_dir}")
    return audio_files


def transcribe(method):
    logger.info(f"Start {method} transcription :")

    traing_data_path = ''
    testing_data_path = ''
    training_transcription_data_path = ''
    testing_transcription_data_path = ''
    model_training_path = ''

    # audio file extension
    extension = ""
    if method == "cha":
        extension = ".cha"
    elif method == "whisper":
        extension = ".wav"

    if method == "cha":
        # path to CHA files
        training_data_path = os.getenv("TRAINING_CHA_DATA_PATH")
        testing_data_path = os.getenv("TESTING_CHA_DATA_PATH")
        training_transcription_data_path = os.getenv("TRAINING_CHA_TRANSCRIPT_PATH")
        testing_transcription_data_path = os.getenv("TESTING_CHA_TRANSCRIPT_PATH")
        model_training_path = os.getenv("MODEL_TRANING_PATH_CHA")
        model_testing_path = os.getenv("MODEL_TESTING_PATH_CHA")


    elif method == "whisper":
        # path to WAV files
        training_data_path = os.getenv("TRAINING_WAV_DATA_PATH")
        testing_data_path = os.getenv("TESTING_WAV_DATA_PATH")
        training_transcription_data_path = os.getenv("TRAINING_WAV_TRANSCRIPT_PATH")
        testing_transcription_data_path = os.getenv("TESTING_WAV_TRANSCRIPT_PATH")
        model_training_path = os.getenv("MODEL_TRANING_PATH_WAV")
        model_testing_path = os.getenv("MODEL_TESTING_PATH_WAV")
    elif method == "assemblyai":
        # path to WAV files
        training_data_path = os.getenv("TRAINING_WAV_DATA_PATH")
        testing_data_path = os.getenv("TESTING_WAV_DATA_PATH")
        training_transcription_data_path = os.getenv("TRAINING_WAV_TRANSCRIPT_PATH")
        testing_transcription_data_path = os.getenv("TESTING_WAV_TRANSCRIPT_PATH")
        model_training_path = os.getenv("MODEL_TRANING_PATH_ASSEMBLYAI")
        model_testing_path = os.getenv("MODEL_TESTING_PATH_ASSEMBLYAI")

    
    train_audio_files = fetch_audio_files(training_data_path, extension)
    test_audio_files = fetch_audio_files(testing_data_path, extension)
    
    # Write transcriptions files
    #write_transcription(train_audio_files, training_transcription_data_path, method)
    #write_transcription(test_audio_files, testing_transcription_data_path, method)

    # Scrape all transcriptions and save it to a csv file
    train_df = transcription_to_df(training_transcription_data_path)
    print(train_df)
    train_df = add_train_scores(train_df)
    
    test_df = transcription_to_df(testing_transcription_data_path)

    project_path = Path(__file__).parent.resolve()
    model_training_path = (project_path / model_training_path).resolve()
    model_testing_path = (project_path / model_testing_path).resolve()
    # Ensure the directories exist
    model_training_path.mkdir(parents=True, exist_ok=True)
    model_testing_path.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(model_training_path / 'training.csv', index=False)
    test_df.to_csv(model_testing_path / 'testing.csv', index=False)

    # duplicate both csv files to the general folder
    general_model_training_path = os.getenv("MODEL_TRANING_PATH_GENERAL")
    general_model_testing_path = os.getenv("MODEL_TESTING_PATH_GENERAL")
    general_model_training_path = (project_path / general_model_training_path).resolve()
    general_model_testing_path = (project_path / general_model_testing_path).resolve()
    # Ensure the directories exist
    general_model_training_path.mkdir(parents=True, exist_ok=True)
    general_model_testing_path.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(general_model_training_path / 'training.csv', index=False)
    test_df.to_csv(general_model_testing_path / 'testing.csv', index=False)

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
    print(data_dir)
    # Traverse through the directory to fetch transcription files
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            # Read the content of the file
            with codecs.open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
                # remove extension from the file name
                file_stem = os.path.splitext(file)[0]
                texts.append((file_stem, text))

    # Create a DataFrame from the list of texts
    df = pd.DataFrame(texts, columns=['addressfname', 'transcript'])

    # Clean up the transcript column by removing newlines and extra spaces
    # write regular expression to remove newlines and extra spaces
    df['transcript'] = df['transcript'].astype(str).apply(lambda x: re.sub(r'\s+', ' ', x))

    # Sort the DataFrame by the 'addressfname' column in ascending order
    df = df.sort_values(by='addressfname')

    # Reset the index
    df = df.reset_index(drop=True)

    # Debugging: Print the resulting DataFrame
    logger.debug(df)

    return df


def add_train_scores(df):
    # reading two csv files
    text_data = df
    print(text_data.head())
    scores_df = pd.read_csv(os.getenv("TRAINING_SCORE_FILE"))
    # Rename columns for consistency
    scores_df = scores_df.rename(columns={'adressfname': 'addressfname', 'dx': 'diagnosis'})
    scores_df = binarize_labels(scores_df)
    print(scores_df.head())

    # using merge function by setting how='inner'
    output = pd.merge(text_data,
                      scores_df[['addressfname', 'mmse', 'diagnosis']],  # We don't want the key column here
                      on='addressfname',
                      how='inner')

    print(output)
    return output


def binarize_labels(df):
    # Transform into binary classification
    df['diagnosis'] = [1 if label == 'ad' else 0 for label in df['diagnosis']]
    # How many data points for each class?
    # print(df.dx.value_counts())
    # Understand the data
    # sns.countplot(x='dx', data=df)  # 1 - diagnosed   0 - control group

    ### Balance data by down-sampling majority class
    # Separate majority and minority classes
    df_majority = df[df['diagnosis'] == 1]  # 87 ad datapoints
    df_minority = df[df['diagnosis'] == 0]  # 79 cn datapoints
    # print(len(df_minority))
    # Undersample majority class
    df_majority_downsampled = resample(df_majority,
                                       replace=False,  # sample without replacement
                                       n_samples=len(df_minority),  # to match minority class
                                       random_state=42)  # reproducible results

    # Combine undersampled majority class with minority class
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])
    # Display new class counts
    # print(df_downsampled.dx.value_counts())
    # sns.countplot(x='dx', data=df_downsampled)  # 1 - diagnosed   0 - control group
    plt.show()
    return df_downsampled
