import numpy as np
import librosa

def extract(filepath, mfcc, chroma, mel):
    try:
        y,sr = librosa.load(filepath, duration=2.5, offset=0.5)
        #sample_rate=22050
        result = np.array([])
        stft = np.abs(librosa.stft(y))
        if mfcc:
            mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfcc))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
            result = np.hstack((result, mel))
        return result
    
    except Exception as e:
        print(f"Error processing {filepath}:e")
        return np.array([])
    

print(np.__version__)