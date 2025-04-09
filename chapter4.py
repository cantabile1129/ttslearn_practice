#%%
print("hello")
# %%
!pip install jupyterlab
#%%
!pip install ttslearn

#%%
#code4.1
from ttslearn.dnntts import DNNTTS
from IPython.display import Audio

engine = DNNTTS()
wav, sr = engine.tts("日本語音声合成のデモです．")
Audio(wav, rate = sr)
#%%
#code4.6
from scipy.io import wavfile
import ttslearn
sr, wav = wavfile.read(ttslearn.util.example_audio_file())

print(sr)
print(wav.shape)
print(len(wav)/sr)
print(wav)
print(type(wav))
#%%
#code4.7

import librosa.display
import matplotlib.pyplot as plt
import numpy as np

#音声ファイルの読み込み
sr, x = wavfile.read(ttslearn.util.example_audio_file())

fig, ax = plt.subplots(figsize=(8, 2))
librosa.display.waveplot(x.astype(np.float32), sr, ax=ax)

ax.set_xlabel("Time [sec]")
ax.set_ylabel("Amplitude")
#%%
#code4.8

#音声ファイルの読み込み
sr , x = wavfile.read(ttslearn.util.example_audio_file())

#code4.12と13でのエラーを解消するため．
x = x.astype(np.float32) / np.iinfo(x.dtype).max


#振幅スペクトル
X = np.abs(np.fft.rfft(x))
#対数振幅スペクトル
logX = 20 * np.log10(X)
#1, 2はグラフの行数列数
fig, ax = plt.subplots(1, 2, figsize = (10, 4), sharex = True)
freq = np.arange(len(X)) / 2 / len(X) * sr
ax[0].plot(freq, X)
ax[0].set_title("Amplitude spectrum")
ax[0].set_xlim(0, sr // 2)
ax[0].set_xlabel("Frequency [Hz]")
ax[0].set_ylabel("Amplitude")

ax[1].plot(freq, logX)
ax[1].set_title("Log amplitude spectrum")
ax[1].set_xlabel("Frequency [Hz]")
ax[1].set_ylabel("Amplitude [dB]")
#%%
#code4.9 NumPyを用いた短時間フーリエ変換の実装

def hanning(N):
  n = np.arange(N)
  w = 0.5 - 0.5 * np.cos(2*np.pi * n / N)
  return w

def stft(x, N, S):
  #窓関数（簡単のため，窓幅とフレーム長Nは同じとする）
  w = hanning(N)
  #短時間フーリエ変換のフレーム数
  M = (len(x) - N) // S + 1
  #短時間フーリエ変換の結果格納用の2次元配列．datatypeが複素数で，周期性や対称性により半分（以上）になっている．
  X = np.zeros((M, N//2 + 1), dtype = np.complex128)
  #音声をずらして切り出し，フーリエ変換
  for m in range(M):
    x_m = w * x[m*S:m*S+N]
    #:は0:N//2+1のことだが，表記を簡略化．
    X[m, :] = np.fft.rfft(x_m)
  return X
#%%
#code4.10 短時間フーリエ変換の結果の可視化

#5ミリ秒のフレームシフト
frame_shift = int(sr * 0.005)
n_fft = 2048
#スペクトログラム
X = stft(x.astype(np.float32), n_fft, frame_shift)
#対数振幅に変換．librosa.amplitude_to_db() は 10 * log10(振幅 / ref) を計算する関数．np.abs(X)で実部と虚部の平方２乗和すなわちスペクトルを取得している．refはreference valueの略でdefaultは振幅の値が1.0．この最大を0とすることで他の振幅がそれよりどの程度小さいかを示している．
logX = librosa.amplitude_to_db(np.abs(X), ref = np.max)

fig , ax = plt.subplots(1, 1, figsize = (8, 4), sharex = True)
#周波数数×フレーム数に変換するため転置を取る
img = librosa.display.specshow(logX.T, hop_length = frame_shift, sr = sr, x_axis = "time", y_axis = "hz", ax = ax)

#色の判例を決定している
fig.colorbar(img, ax = ax, format = "% + 2.f dB")
#音声のパワーは低域に集中するため，8000Hzまで表示
ax.set_ylim(0, 8000)

ax.set_xlabel("Time [sec]")
ax.set_ylabel("Frequency [Hz]")
#%%
#code4.11 librosaを利用した短時間フーリエ変換

import librosa
# n_fft: 2048, frame_shift: 240

#center = Falseとして，ウィンドウの中心ではなくフレームの左端を基準にする（ゼロパディングなし）
X = librosa.stft(x.astype(np.float32), n_fft = n_fft, win_length = n_fft, hop_length =frame_shift, window = "hann", center = False).T
#%%
#code4.12 音声の短時間フーリエ変換およびその逆変換
import IPython.display

#STFT
X = librosa.stft(x, n_fft=n_fft, win_length=n_fft, hop_length=frame_shift, window="hann")

#ISTFT
x_hat = librosa.istft(X, win_length=n_fft, hop_length=frame_shift, window="hann")

#オーディオプレイヤーその表示
IPython.display.display(Audio(x.astype(np.float32), rate=sr))
IPython.display.display(Audio(x_hat.astype(np.float32), rate=sr))

#%%
#code4.13メルスペクトログラムの計算

#スペクトログラム
X = librosa.stft(x, n_fft=n_fft, hop_length=frame_shift)

#80次元のメルスペクトログラム
n_mels = 80
melfb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
melspec = librosa.amplitude_to_db(np.dot(melfb, np.abs(X)), ref=np.max)
# %%
