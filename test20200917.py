"""
beginning of frequency analysis 
20200917-
by Sota Okuda

main ref = https://jorublog.site/python-voice-analysis/
scipy.io.wavfile ref = https://water2litter.net/rum/post/python_scipy_io_wavfile_read/
DFT ref = http://leo.ec.t.kanazawa-u.ac.jp/staffs/nakayama/edu/file/signal_proc_ch3.pdf
DFT ref = https://www.youtube.com/watch?v=xvX8aaok0PE&ab_channel=KenyuUehara

"""
import sys
import numpy as np
import librosa
import matplotlib.pyplot as plt
import scipy.io.wavfile
import librosa.display
import pandas as pd


#load audio file 
args = sys.argv #インライン引数を取得する。sys[0]は実行しているファイル
wav_filename = args[1]
rate, data = scipy.io.wavfile.read(wav_filename)

print("rate is ", rate)
print("datashape is ", data.shape)
print(data.shape[1],"channels")
print("datatype is ", type(data[1][1]), "(intのbit数が，量子化のビット数)")
"""
サンプリング周波数
アナログ信号をデジタル信号に変換する時に1秒あたりに抽出する点の数

サンプリングの定理
最高周波数の2倍のサンプリング周波数でいい感じにアナログ信号に変換できる。
人間の可聴域は - 20kHz
→44.1kHzくらいでのサンプリングがちょうどいい。

※実際の周波数の2倍よりも低いサンプリング周波数でサンプリングした時に，小さい周波数に復元される。
→ハイパスフィルタをかけたりする。

量子化→
サンプリング点を抽出する時の細かさ。
16bitだったら，2^16 = 65536諧調に分けられる。
→つまり，振幅は65536/2 = 32768が最大値

data　の内部構造
({データのサンプリングした点},{チャンネル数})
サンプリングした点の数　＝秒数*サンプリング周波数
"""

#16bitの音声ファイルのデータを-1から1に正規化
data = data[:,1]/ 32768
# フレーム長
fft_size = 1024             
# フレームシフト長 
hop_length = int(fft_size / 4) 



# 短時間フーリエ変換実行
amplitude = np.abs(librosa.core.stft(data, n_fft=fft_size, hop_length=hop_length))

# 振幅をデシベル単位に変換
log_power = librosa.core.amplitude_to_db(amplitude)

# グラフ表示
librosa.display.specshow(log_power, sr=rate, hop_length=hop_length, x_axis='time', y_axis='hz', cmap='magma')
plt.colorbar(format='%+2.0f dB')  
plt.show()          
