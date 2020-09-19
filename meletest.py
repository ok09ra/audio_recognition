"""
メル周波数ケプストラム係数実装
20200919-
by Sota Okuda

main ref = https://aidiary.hatenablog.com/entry/20120225/1330179868
librosa ver ref = https://www.wizard-notes.com/entry/music-analysis/insts-timbre-with-mfcc#MFCC-%E3%81%A8%E3%81%AF
"""
import sys
import numpy as np
import librosa
import matplotlib.pyplot as plt
import scipy.io.wavfile
import librosa.display
import pandas as pd
import scipy.stats as sp

#load audio file 
args = sys.argv #インライン引数を取得する。sys[0]は実行しているファイル
wav_filename = args[1]
rate, data = scipy.io.wavfile.read(wav_filename)

print("rate is ", rate)
print("datashape is ", data.shape)
print(data.shape[1],"channels")
print("datatype is ", type(data[1][1]), "(intのbit数が，量子化のビット数)")

#16bitの音声ファイルのデータを-1から1に正規化
data = data[:,1]/ 32768
# フレーム長
fft_size = 1024             
# フレームシフト長 
hop_length = int(fft_size / 4) 



D = np.abs(librosa.stft(data))
D_dB = librosa.amplitude_to_db(data, ref=np.max)

# メルスペクトログラムを算出
S = librosa.feature.melspectrogram(S=D, sr=rate)
S_dB = librosa.amplitude_to_db(S, ref=np.max)

# MFCCを算出
mfcc = librosa.feature.mfcc(S=S_dB, n_mfcc=20, dct_type=3)

#standarization
mfcc = sp.stats.zscore(mfcc, axis = 1)

# グラフ表示
librosa.display.specshow(mfcc, sr=rate, hop_length=hop_length, x_axis='time', y_axis='hz', cmap='magma')
plt.colorbar()  
plt.show()          
