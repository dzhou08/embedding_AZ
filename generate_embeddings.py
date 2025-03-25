from matplotlib import pyplot as plt
import pandas as pd
import openai
import numpy as np
import generate_linguistic_features
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

import logging
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
logger = logging.getLogger()

def create_embeddings(embedding_model):
    """
    Create embeddings for a DataFrame's 'transcript' column using OpenAI's embedding API.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame containing a 'transcript' column.
        embedding_model (str): The embedding model to use (e.g., "text-embedding-ada-002").
        
    Returns:
        pd.DataFrame: DataFrame with a new 'embedding' column and the 'transcript' column removed.
    """
    # Load the DataFrame from test and training csv files
    MODEL_TRANING_PATH_GENERAL = os.getenv("MODEL_TRANING_PATH_GENERAL")
    MODEL_TESTING_PATH_GENERAL = os.getenv("MODEL_TESTING_PATH_GENERAL")
    train_df = pd.read_csv(MODEL_TRANING_PATH_GENERAL+ '/training.csv')
    create_embeddings_for_dataframe(train_df, embedding_model, MODEL_TRANING_PATH_GENERAL + '/training_embeddings.csv')

    test_df = pd.read_csv(MODEL_TESTING_PATH_GENERAL+ '/testing.csv')
    create_embeddings_for_dataframe(test_df, embedding_model, MODEL_TESTING_PATH_GENERAL + '/testing_embeddings.csv')


def create_embeddings_for_dataframe(df, embedding_model, save_file_path):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    embeddings = []
    # read dataframe from save_file_path
    if os.path.exists(save_file_path):
        df = pd.read_csv(save_file_path)
    else:
        df['embedding'] = None
        df['linguistic_features'] = None
        df.to_csv(save_file_path, index=False)

    # iterate through rows in the DataFrame
    for index, row in df.iterrows():
        # skip if embedding already exists, not none and not empty and not nan
        if not pd.isnull(row['embedding']) and row['embedding'] != '[]':
            continue

        transcript = row['transcript']
        logger.info(f"before combining for {row['addressfname']}")
        embedding_vector = []
        try:
            embedding_vector = openai.embeddings.create(input=transcript, model=embedding_model).data[0].embedding
        except Exception as e:
            logger.error(f"An error occurred while creating embedding for input: {transcript}, error: {e}")
        # combine with linguistic features
        # Step 1: Extract linguistic features
        linguistic_vector = generate_linguistic_features.extract_features(transcript, embedding_model)
        # Step 2: Convert features to a vector
        linguistic_feature_vector = np.array([
            linguistic_vector["lexical_diversity"],
            linguistic_vector["syntactic_complexity"],
            linguistic_vector["semantic_coherence"],
            linguistic_vector["speech_errors"]
        ])
        
        # Step 3: Concatenate feature vector and embedding vector
        '''combined_vector = np.concatenate([feature_vector, embedding_vector])

        embeddings.append(combined_vector)


        logger.info(f"after combining {len(combined_vector)}")'''

        # Step 4: 
        # Normalize GPT embedding (L2 normalization)
        normalized_embedding = embedding_vector / np.linalg.norm(embedding_vector)
        #row['embedding'] = normalized_embedding
        row['embedding'] = "[" + ", ".join(map(str, normalized_embedding)) + "]"

        # Standardize smaller features
        linguistic_feature_vector = linguistic_feature_vector.reshape(-1, 1)
        scaler = StandardScaler()
        standardized_linguistic_feature_vector = scaler.fit_transform(linguistic_feature_vector).flatten()
        #row['linguistic_features'] = standardized_linguistic_feature_vector
        row['linguistic_features'] = "[" + ", ".join(map(str, standardized_linguistic_feature_vector)) + "]"
        print(f"{row['addressfname']} embedding length: {len(normalized_embedding)}  linguistic_features: {len(standardized_linguistic_feature_vector)}")
        
        
        # save the row to the DataFrame, inplace
        df.loc[index] = row
        df.to_csv(save_file_path, index=False)

    
    # save numpy array to a column in the DataFrame
    # Convert NumPy arrays to lists
    #embeddings_as_lists = [vector.tolist() for vector in embeddings]
    #df['embedding'] = embeddings_as_lists
    #df = df.drop('transcript', axis=1)
    #df.to_csv(save_file_path, index=False)