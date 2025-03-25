import pandas as pd


#import classification
import generate_embeddings as embedding

import generate_transcription as tr

import generate_classification as cl

import os

def main():

    prompt = "Please indicate your choice by typing numbers: \n 1: Transcribe audio files \n 2: Build Vector Embeddings \n 3: Build Classification Models \n 4: Build Regression Models \n 5: Exit \n"
    choice = input(prompt)

    if choice == "1":
        go_transcription()
    elif choice == "2":
        print("Build Vector Embeddings")
        go_embedding()
    elif choice == "3":
        print("Build Classification Models")
        go_classification()
    elif choice == "4":
        print("Build Regression Models")
        go_regression()
    else:
        print("Exit")

def go_transcription():
    """
    Prompts the user to choose a transcription method and performs the transcription based on the user's choice.

    The available transcription methods are:
    1. Transcription in .cha files
    2. OpenAI Whisper
    3. AssemblyAI

    The function prints the chosen transcription method and calls the appropriate transcription function.

    Returns:
        None
    """
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

def go_embedding():
    """
    Prompts the user to choose a vector embedding method and prints the selected method.
    The function displays a prompt to the user with two options for vector embedding methods:
    1. text-embedding-3-small
    2. text-embedding-3-large
    Based on the user's input, it prints the selected embedding method. If the user selects
    "1", it prints "text-embedding-3-small" and calls the `create_embeddings` function with
    "bert" and `tokenizer` as arguments. If the user selects "2", it prints "text-embedding-3-large".
    """
    
    print("Build Vector Embeddings")
    prompt = "Please indicate your vector embedding method choice: \n 1: text-embedding-3-small \n 2: text-embedding-3-large\n"
    embedding_choice = input(prompt)
    embedding_model = ""   
    if embedding_choice == "1":
        embedding_model = "text-embedding-3-small"
    elif embedding_choice == "2":
        embedding_model = "text-embedding-3-large"
    
    embedding.create_embeddings(embedding_model)

def go_classification():
    MODEL_TRANING_PATH_GENERAL = os.getenv("MODEL_TRANING_PATH_GENERAL")
    MODEL_TESTING_PATH_GENERAL = os.getenv("MODEL_TESTING_PATH_GENERAL")

    train_embeddings_array = cl.embeddings_to_array(MODEL_TRANING_PATH_GENERAL+ "/training_embeddings.csv")
    test_embeddings_array = cl.embeddings_to_array(MODEL_TESTING_PATH_GENERAL + '/testing_embeddings.csv')
    #print(test_embeddings_array)

    cl.classify_embedding(train_embeddings_array, test_embeddings_array, int(os.getenv("N_SPLITS")))

def go_regression():
    MODEL_TRANING_PATH_GENERAL = os.getenv("MODEL_TRANING_PATH_GENERAL")
    MODEL_TESTING_PATH_GENERAL = os.getenv("MODEL_TESTING_PATH_GENERAL")

    train_embeddings_array = cl.embeddings_to_array(MODEL_TRANING_PATH_GENERAL+ "/training_embeddings.csv")
    test_embeddings_array = cl.embeddings_to_array(MODEL_TESTING_PATH_GENERAL + '/testing_embeddings.csv')
    #print(test_embeddings_array)

    cl.regression_embedding(train_embeddings_array, test_embeddings_array, int(os.getenv("N_SPLITS")))

if __name__ == "__main__":
    main()
