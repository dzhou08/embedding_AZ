import pandas as pd


import transcribe_audio as tr
'''
import classification
import config
import embedding
from config import logger
'''
#rom utils.utils import get_user_input, df_to_csv


import transcribe_audio as tr

def main():

    #tokenizer = config.set_up()

    prompt = "Please indicate your choice by typing numbers: \n 1: Transcribe audio files \n 2: Build Vector Embeddings \n 3: Build Classification Models \n 4: Exit \n"
    choice = input(prompt)

    if choice == "1":
        print("Transcribe audio files")
        prompt = "Please indicate your transcription method choice: \n 1: Use the Transcription in .cha files \n 2: OpenAI Whisper \n 3: AssemblyAI\n"
        transcription_choice = input(prompt)
        if transcription_choice == "1":
            print("Transcription in .cha files")
            tr.transcribe("cha")
        elif transcription_choice == "2":
            print("OpenAI Whisper")
            tr.transcribe("whisper")
        elif transcription_choice == "3":
            print("AssemblyAI")
            tr.transcribe("assemblyai")
    elif choice == "2":
        print("Build Vector Embeddings")
    elif choice == "3":
        print("Build Classification Models")
    elif choice == "4":
        print("Exit")


if __name__ == "__main__":
    main()
