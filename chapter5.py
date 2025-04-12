#%%
#code5.1&5.2
from nnmnkwii.io import hts
import ttslearn
labels = hts.load(ttslearn.util.example_label_file(mono=False)) 
for start_time, end_time, context in labels[:6]:
   print(f"{start_time} {end_time} {context}") 

"""
^...前の音素
-...現在の音素
+...次の音素
=...次の次の音素

1. A:-2+1+3
音素レベルの位置情報
-2: 現在の音素の 前のモーラ位置
+1: 現在の音素の モーラ内での位置
+3: 次の音素のモーラ内での位置

2. B:xx-xx_xx
音素の前後関係
xx-xx_xx: 現在の音素の 前後の音素カテゴリ

3. C:02_xx+xx
音素がどの 音節 (syllable) に属するか
02: 現在の音素が属する音節の位置
xx+xx: 次の音節との関係

4. D:13+xx_xx
アクセント境界の情報
13: 現在の音素のアクセント位置
xx_xx: 次のアクセントの情報

5. E:xx_xx!xx_xx-xx
ピッチパターン（F0 の変化）
xx_xx: 現在の音素の音調情報
!xx_xx: アクセントの変化位置
-xx: 次の音素の音調情報

6. F:3_3#0_xx@1_5|1_23
モーラ単位の情報
3_3: 現在のモーラの音節位置
#0_xx: 語のモーラ数
@1_5: モーラ内の音素位置
|1_23: 次のモーラとの関係

7. G:7_2%0_xx_1
単語（word）の情報
7_2: 単語の音節数
%0_xx_1: 単語のイントネーション情報

8. H:xx_xx
文レベルの情報
xx_xx: 文の中の音節情報

9. I:5-23@1+1&1-5|1+23
フレーズ単位の情報
5-23: 現在のフレーズ内の位置
@1+1: 前後のフレーズとの関係
&1-5: フレーズのアクセント情報
|1+23: 次のフレーズの情報

10. J:xx_xx
プロソディ情報
xx_xx: 文のイントネーションパターン

11. K:1+5-23
文全体のリズム・ピッチ情報
1+5-23: 文章レベルの韻律情報
"""   
# %%
#code5.3
import pyopenjtalk
pyopenjtalk.g2p("今日もいい天気ですね", kana=True)
pyopenjtalk.g2p("今日もいい天気ですね", kana=False)

labels = pyopenjtalk.extract_fullcontext("今日")
for label in labels:
   print(label)
#%%
#code5.6
from nnmnkwii.io import hts
import ttslearn

binary_dict, numeric_dict = hts.load_question_set(ttslearn.util.example_qst_file())

print("二値特徴量の数:", len(binary_dict))
print("数値特徴量の数:", len(numeric_dict))
print("1つ目の質問:", binary_dict[0], binary_dict[1])
# %%
#code5.7
from nnmnkwii.frontend import merlin as fe

labels = hts.load(ttslearn.util.example_label_file())
feats = fe.linguistic_features(labels, binary_dict, numeric_dict)
print("言語特徴量のサイズ:", feats.shape)
# %%
#code5.8
feats_phoneme = fe.linguistic_features(labels, binary_dict, numeric_dict, add_frame_features=False)
feats_frame = fe.linguistic_features(labels, binary_dict, numeric_dict, add_frame_features=True)
print("言語特徴量のサイズ:", feats_phoneme.shape)
print("言語特徴量のサイズ:", feats_frame.shape)
# %%
#code5.9
from scipy.io import wavfile
import numpy as np
import pyworld
from nnmnkwii.preprocessing.f0 import interp1d

#基本周波数を対数基本周波数へ変換する関数
def f0_to_lf0(f0):
   lf0 = f0.copy()
   nonzero_indices = np.nonzero(f0)
   lf0[nonzero_indices] = np.log(f0[nonzero_indices])
   return lf0

#音声ファイルの読み込み
sr, x = wavfile.read(ttslearn.util.example_audio_file())
x = x.astype(np.float64)

#DIO法で基本周波数推定
f0 , timeaxis = pyworld.dio(x, sr)

#基本周波数を対数基本周波数に変換)
lf0 = f0_to_lf0(f0)

#対数基本周波数を線形補間
clf0 = interp1d(lf0, kind = "linear")
# %%
#code5.10
#DIOによる基本周波数推定
#時間軸の配列とそれに対応する基本周波数の配列を取得
f0, timeaxis = pyworld.dio(x, sr)

#有声・無声フラグの計算
vuv = (f0>0).astype(np.float32)


# %%
#code5.11
import pysptk
#DIOによる基本周波数推定
f0, timeaxis = pyworld.dio(x, sr)

#CheapTrickによるスペクトル包絡の推定
#返り値はパワースペクトル（振幅の2乗）
spectrogram = pyworld.cheaptrick(x, f0, timeaxis, sr)

#線形周波数軸をメル周波数尺度に伸縮し，その後ケプストラムに変換．
#alphaは周波数軸の伸縮のパラメータを表す．
alpha = pysptk.util.mcepalpha(sr)
#FFT長はサンプリング周波数が48kHzの場合は2048．
fftlen = pyworld.get_cheaptrick_fft_size(sr)
#メルケプストラムの次元数はmgc_order + 1となる．
mgc_order = 59
mgc = pysptk.sp2mc(spectrogram, mgc_order, alpha)
#メルケプストラムから元のスペクトル包絡を復元
#スペクトルの次元数は，fftlen//2+1=1025
spectrogram_reconstructed = pysptk.mc2sp(mgc, alpha, fftlen)


# %%
#code5.12
#DIOによる基本周波数の推定
f0, timeaxis = pyworld.dio(x, sr)

#D4C法による被周期性指標の推定
aperiodicity = pyworld.d4c(x, f0, timeaxis, sr)

#帯域別の非周期性指標に圧縮
bap = pyworld.code_aperiodicity(aperiodicity, sr)
# %%
#code5.13
#xは[時間フレーム数, 特徴量次元数]の行列，wは窓（重み係数）．
def compute_delta(x, w):
   #xと同じ形・データ型の配列を0で初期化して作成．
   y = np.zeros_like(x)
   #特徴量の次元ごとに動的特徴量を計算
   for d in range(x.shape[1]):
      #sameは入力長と出力長は同じになるように中心寄せをしている．
      y[:, d] = np.correlate(x[:, d], w, mode="same")
   return y
# %%
#code5.14
from nnmnkwii.preprocessing import delta_features
#WORLDによる音声パラメータの推定
f0, timeaxis = pyworld.dio(x, sr)
spectrogram = pyworld.cheaptrick(x, f0, timeaxis, sr)
aperiodicity = pyworld.d4c(x, f0, timeaxis, sr)

#スペクトル包絡をメルケプストラムに変換
mgc_order = 59
alpha = pysptk.util.mcepalpha(sr)
mgc = pysptk.sp2mc(spectrogram, mgc_order, alpha)

#有声無声フラグの計算
vuv = (f0>0).astype(np.float32)

#連続対数基本周波数系列
lf0 = interp1d(f0_to_lf0(f0), kind="linear")

#帯域非周期性指標
bap = pyworld.code_aperiodicity(aperiodicity, sr)

#基本周波数と有声無声フラグを2次元の行列にしておく．
#モデル入力に合わせて，一次元配列→二次元配列にreshape（shape:[T, 1]）
lf0 = lf0[:, np.newaxis] if len(lf0.shape) == 1 else lf0
vuv = vuv[:, np.newaxis] if len(vuv.shape) == 1 else vuv

#動的特徴量を計算するための窓
windows = [
      [1.0],   #静的特徴量に対する窓
      [-0.5, 0.0, 0.5], #1次動的特徴量に対する窓
      [1.0, -2.0, 1.0],
]

#静的特徴量と動的特徴量を結合した特徴量の計算
mgc = delta_features(mgc, windows)
lf0 = delta_features(lf0, windows)
bap = delta_features(bap, windows)

#すべての特徴量を結合した特徴量を作成
feats = np.hstack([mgc, lf0, vuv, bap])

print(f"メルケプストラムの次元数: {mgc.shape[1]}")
print(f"連続対数基本周波数の次元数: {lf0.shape[1]}")
print(f"有声無声フラグの次元数: {vuv.shape[1]}")
print(f"帯域非周期性指標の次元数: {bap.shape[1]}")
print(f"結合された音響特徴量の次元数: {feats.shape[1]}")
# %%
#code5.15
from nnmnkwii.paramgen import mlpg
from IPython.display import Audio
import IPython
from ttslearn.dnntts.multistream import get_windows, split_streams
from ttslearn.dsp import world_spss_params

sr, x = wavfile.read(ttslearn.util.example_audio_file())
x = x.astype(np.float64)

#音響特徴量抽出のパラメータ
mgc_order = 59
alpha = pysptk.util.mcepalpha(sr)
fftlen = pyworld.get_cheaptrick_fft_size(sr)

#音響特徴量の抽出
feats = world_spss_params(x, sr, mgc_order)

#パラメータ生成に必要な特徴量の分散
#chapter6.pyで解説するが，実際には学習データ全体に対して計算する．
feats_var = np.var(feats, axis=1)

#結合された特徴量から各特徴量の分離．p126参照．
stream_sizes = [(mgc_order+1)*3, 3, 1, pyworld.get_num_aperiodicities(sr)*3]
mgc, lf0, vuv, bap = split_streams(feats, stream_sizes)

#i番目のストリームの開始インデックスと終了インデックス
#hstackは連結するための関数で，cumsumは累積和（序数はその手前のサイズ分加算される）．
start_ind = np.hstack(([0], np.cumsum(stream_sizes[:-1])))
end_ind = np.cumsum(stream_sizes)

#パラメータ生成に必要な動的特徴量の計算に利用した窓（"num_window="は不要）．静的，一階差分，二階差分を表す．
windows = get_windows(3)

#パラメータ生成
mgc = mlpg(mgc, feats_var[start_ind[0]:end_ind[0]], windows)
lf0 = mlpg(lf0, feats_var[start_ind[1]:end_ind[1]], windows)
bap = mlpg(bap, feats_var[start_ind[3]:end_ind[3]], windows)

#メルケプストラムからスペクトル包絡への変換
spectrogram = pysptk.mc2sp(mgc, alpha, fftlen)

#連続対数基本周波数から基本周波数への変換
f0 = lf0.copy()
#DNNの実運用では0か1のバイナリであるとは限らない．
f0[vuv < 0.5] = 0
f0[np.nonzero(f0)] = np.exp(f0[np.nonzero(f0)])

#帯域非周期指標から非周期性指標への変換
aperiodicity = pyworld.decode_aperiodicity(bap.astype(np.float64), sr, fftlen)

#WORLDによる音声波形の合成
y = pyworld.synthesize(f0.flatten().astype(np.float64), spectrogram.astype(np.float64), aperiodicity.astype(np.float64), sr)

#オーディオプレイヤーの表示
IPython.display.display(Audio(x.astype(np.float32), rate=sr))
IPython.display.display(Audio(y.astype(np.float32), rate=sr))
# %%
