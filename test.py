print("preparing imports ... ")
import tensorflow as tf
import os
import librosa
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
