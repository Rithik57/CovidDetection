# code for the python script in the website is tested here
print("preparing imports ... ")
import sklearn
import pickle
import librosa
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import os
# suppress all warnings
import warnings
warnings.filterwarnings("ignore")

print("loading model ... ")
clf = pickle.load(open("Models/Model4_GBM.sav",'rb'))
print("model ready for sampling ...")
print("listening for cough ...")
freq=44100 # sampling frequency for the incoming audio
duration=5 # incoming audio clipped to 5 seconds as per the dataset
recording = sd.rec(int(duration*freq),samplerate=freq,channels=2) #specify the recording details
sd.wait() # wait for audio to be recorded
write("recording0.wav",freq,recording)
age = int(input("Enter age : "))
respiratory = int(input("Existing respiratory conditions (1/0) : "))
path = 'recording0.wav'
y,sr = librosa.load(path, mono=True, duration=5)
chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
rmse = librosa.feature.rms(y=y)
spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
zcr = librosa.feature.zero_crossing_rate(y)
mfcc = librosa.feature.mfcc(y=y, sr=sr)
to_append = f'{np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
for e in mfcc:
    to_append += f' {np.mean(e)}'
to_append += ' '+str(age)
to_append += ' '+str(1)
to_append += ' '+str(0)
to_append = to_append.split()
x_test = to_append
for i in range(len(x_test)) :
    x_test[i] = float(x_test[i])
preds = clf.predict([x_test])
if preds[0] == 0:
    print("Covid negative")
else :
    print("Covid Positive")
os.remove('recording0.wav')
