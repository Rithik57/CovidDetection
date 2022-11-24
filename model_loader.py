# code for the python script in the website is tested here
print("preparing imports ... ")
import tensorflow as tf
import os
import librosa
import numpy as np
print("loading model ... ")
model = tf.keras.models.load_model('Models/model1.h5')
print(model.summary())
print("loading audio file ... ")
y,sr = librosa.load('test_sound.wav', mono=True, duration=5)
print("converting to spectrogram ... ")
import matplotlib.pyplot as plt
cmap = plt.get_cmap('inferno')
plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB')
plt.axis('off')
plt.savefig(f'test_spectrogram.png')
header = 'chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' label'
header = header.split()
print("loading features ... ")
y,sr = librosa.load('test_sound.wav', mono=True, duration=5)
chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
rmse = librosa.feature.rms(y=y)
spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
zcr = librosa.feature.zero_crossing_rate(y)
mfcc = librosa.feature.mfcc(y=y, sr=sr)
x_test = f'{np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
for e in mfcc:
    x_test += f' {np.mean(e)}'
x_test = x_test.split()
for i in range(len(x_test)) :
    x_test[i] = float(x_test[i])
print(x_test)
print("running model ... ")
preds = model.predict([x_test])
print(np.argmax(preds))