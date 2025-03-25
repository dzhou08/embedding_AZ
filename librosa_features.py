import librosa
import numpy as np
import pandas as pd
import os

import generate_classification as cl

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd


def extract_audio_features(wav_path):
    """
    Extract acoustic features from a WAV file for Alzheimer's detection

    Pitch Features: Statistics of fundamental frequency (F0)
    MFCCs: Capture vocal tract configuration
    Spectral Features:
    Centroid (brightness of sound)
    Rolloff (shape of spectrum)
    Bandwidth (spread of spectrum)
    Zero Crossing Rate: Related to voice quality
    RMS Energy: Related to volume/intensity
    Speech Rate: Estimated using zero crossings
    """
    # Load the audio file
    y, sr = librosa.load(wav_path, duration=None)
    
    # Initialize feature dictionary
    features = {}
    
    # 1. Pitch (F0) statistics
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitches_cleaned = pitches[pitches > 0]
    features['pitch_mean'] = np.mean(pitches_cleaned)
    features['pitch_std'] = np.std(pitches_cleaned)
    features['pitch_max'] = np.max(pitches_cleaned)
    features['pitch_min'] = np.min(pitches_cleaned)
    
    # 2. MFCCs (Mel-frequency cepstral coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
        features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
    
    # 3. Spectral features
    # Spectral Centroid
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features['spectral_centroid_mean'] = np.mean(spec_cent)
    features['spectral_centroid_std'] = np.std(spec_cent)
    
    # Spectral Rolloff
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features['spectral_rolloff_mean'] = np.mean(spec_rolloff)
    features['spectral_rolloff_std'] = np.std(spec_rolloff)
    
    # Spectral Bandwidth
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features['spectral_bandwidth_mean'] = np.mean(spec_bw)
    features['spectral_bandwidth_std'] = np.std(spec_bw)
    
    # 4. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features['zero_crossing_rate_mean'] = np.mean(zcr)
    features['zero_crossing_rate_std'] = np.std(zcr)
    
    # 5. Root Mean Square Energy
    rms = librosa.feature.rms(y=y)[0]
    features['rms_mean'] = np.mean(rms)
    features['rms_std'] = np.std(rms)
    
    # 6. Speech rate estimation (using zero crossings as proxy)
    features['speech_rate'] = len(librosa.zero_crossings(y))/len(y)
    
    return features

def parse_audio_features(folder_path_list, output_file_name):
    all_features = []

    for folder_path in folder_path_list:
        # loop through all the files in the folder
        for file in os.listdir(folder_path):
            if file.endswith(".wav"):
                audio_file = folder_path + file
                print(audio_file)
                features = extract_audio_features(audio_file)
                # remove ".wav" extension from the file name
                file_name = file.split(".")[0]
                features['ID'] = file_name
                if "/cd/" in folder_path:
                    features['diagnose'] = 1 
                elif "/cc/" in folder_path:
                    features['diagnose'] = 0
                all_features.append(features)
                # pretty print the features
                '''for key, value in features.items():
                    print(f"{key}: {value}")'''
    
    pd.DataFrame(all_features).to_csv(output_file_name, index=False)

def build_logistic_model(df, target_col='diagnose', test_size=0.2, random_state=42):
    # Remove ID column and separate features from target
    X = df.drop(['ID', target_col], axis=1)
    y = df[target_col]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegression(random_state=random_state, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    
    # Print results
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return model, scaler

def predict_new_data(model, scaler, test_file_path):
    # Load test data
    test_df = pd.read_csv(test_file_path)
    
    # Prepare features (exclude ID column)
    X_new = test_df.drop('ID', axis=1)
    
    # Scale features using the same scaler
    X_new_scaled = scaler.transform(X_new)
    
    # Make predictions
    predictions = model.predict(X_new_scaled)
    
    # Add predictions to the original dataframe
    test_df['predicted_diagnose'] = predictions
    
    # Save results
    output_path = 'predictions_output.csv'
    test_df.to_csv(output_path, index=False)
    
    return predictions, test_df

if __name__ == "__main__":
    '''
    # parse training data
    train_folder_path_list=["ADReSSo_2020/ADReSS-IS2020-data-train/train/Full_wave_enhanced_audio/cd/", 
                      "ADReSSo_2020/ADReSS-IS2020-data-train/train/Full_wave_enhanced_audio/cc/"]
    parse_audio_features(train_folder_path_list, "audio_features_train.csv")
    # parse the test data
    test_folder_path_list=["ADReSSo_2020/ADReSS-IS2020-data-test/test/Full_wave_enhanced_audio/"]
    parse_audio_features(test_folder_path_list, "audio_features_test.csv")
    '''

    # train model
    df_train = pd.read_csv('audio_features_train.csv')  # Load your dataset
    # Remove ID column and separate features from target
    X_train = df_train.drop(['ID', 'diagnose'], axis=1)
    y_train = df_train['diagnose']

    df_test = pd.read_csv('audio_features_test.csv')  # Load your dataset
    # Remove ID column and separate features from target
    X_test = df_test.drop(['ID'], axis=1)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(X_train_scaled)

    print(X_test_scaled)

    # Evaluate performance on test data
    # the ground truth real value for test diagnose is in the dx column
    test_ground_truth = pd.read_csv(os.getenv("GROUND_TRUTH_FILE"))
    # join the test_ground_truth and model_test_results on the ID and addressfname column
    # add test_ground_truth value into the model_test_results dataframe
    test_data = pd.merge(df_test, test_ground_truth, left_on='ID', right_on='ID')
    y_test = test_data['dx']#.to_list()

    # get current folder
    current_folder = os.getcwd()
    cl.training_classifier(
        X_train_scaled, 
        y_train, 
        X_test_scaled, 
        y_test, 
        int(os.getenv("N_SPLITS")), 
        current_folder+"/librosa_results/",
        "Acoustic Features")

    