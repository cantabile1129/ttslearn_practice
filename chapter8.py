#%%
#code8.1
from ttslearn.wavenet import WaveNetTTS
from tqdm.notebook import tqdm
from IPython.display import Audio

engine = WaveNetTTS()
wav, sr = engine.tts("ウェーブネットにチャレンジしましょう！", tqdm = tqdm)
Audio(wav, rate=sr)

# %%
#code8.stage-1
import subprocess
import os

# 実行したいシェルスクリプトのコマンド
command = ["./run.sh", "--stage", "-1", "--stop-stage", "-1"]
#確実に絶対パスで指定
working_dir = "/home/takamichi-lab-pc05/ドキュメント/B4/Pythonで学ぶ音声合成/ttslearn/recipes/wavenet"

# 実行
result = subprocess.run(command, cwd=working_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# 標準出力と標準エラー出力の確認
print("標準出力:")
print(result.stdout)

print("\n標準エラー:")
print(result.stderr)
# %%
#code8.stage0
import subprocess
import os

# 実行したいシェルスクリプトのコマンド
command = ["./run.sh", "--stage", "0", "--stop-stage", "0"]
#確実に絶対パスで指定
working_dir = "/home/takamichi-lab-pc05/ドキュメント/B4/Pythonで学ぶ音声合成/ttslearn/recipes/wavenet"

# 実行
result = subprocess.run(command, cwd=working_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# 標準出力と標準エラー出力の確認
print("標準出力:")
print(result.stdout)

print("\n標準エラー:")
print(result.stderr)


# %%
#code8.stage1
import subprocess
import os

# 実行したいシェルスクリプトのコマンド
command = ["./run.sh", "--stage", "1", "--stop-stage", "1"]
#確実に絶対パスで指定
working_dir = "/home/takamichi-lab-pc05/ドキュメント/B4/Pythonで学ぶ音声合成/ttslearn/recipes/wavenet"

# 実行
result = subprocess.run(command, cwd=working_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# 標準出力と標準エラー出力の確認
print("標準出力:")
print(result.stdout)

print("\n標準エラー:")
print(result.stderr)


#%%
#code8.2
import numpy as np
import pyworld

from nnmnkwii.preprocessing import delta_features
from nnmnkwii.preprocessing import interp1d
from ttslearn.dsp import f0_to_lf0

#さすがに引数(x)ではなくね？
def world_log_f0_vuv(x, sr):
   #フレーム数だけの配列があり，timeaxisはタイムスタンプ（ex.)[0.000, 0.005, ...]）
   f0, timeaxis=pyworld.dio(x, sr)
   vuv=(f0>0).astype(np.float32)
   
   #連続対数基本周波数
   lf0=f0_to_lf0(f0)
   #interp1dは，Unvoicedが0になりlog(0)がNaNになるので，線形補間している．
   lf0=interp1d(lf0)
   
   #連続対数基本周波数と有声/無声フラグを2次元の行列の形にしておく．delta_features()やnp.hstack()のために[T, 1]の形にする．
   lf0=lf0[:, np.newaxis] if len(lf0.shape)==1 else lf0
   vuv=vuv[:, np.newaxis] if len(vuv.shape)==1 else vuv
   
   #動的特徴量の計算
   windows=[
      [1.0], #静的特徴量に対する窓
      [-0.5, 0.0, 0.5], #1次動的特料に対する窓
      [1.0, -2.0, 1.0], #2次動的特徴量に対する窓
   ]
   lf0=delta_features(lf0, windows)
   
   #全ての特徴量を結合．lf0の[T, 3]とvuv[T, 1]を結合して[T, 4]の行列にする．
   feats=np.hstack([lf0, vuv]).astype(np.float32)
   
   return feats
#%%
#code8.3
# 必要なライブラリのインポート
import numpy as np
import librosa
from scipy.io import wavfile

import ttslearn
from nnmnkwii.frontend import merlin as fe
from nnmnkwii.io import hts


#HTS形式の質問ファイルを読み込み
binary_dict, numeric_dict = hts.load_question_set(ttslearn.util.example_qst_file())

#フルコンテキストラベルの読み込み
labels = hts.load(ttslearn.util.example_label_file())

#フレーム単位の言語特徴量を抽出
#code6.3と6.4で既出だけど，音素単位の言語特徴量を，音素内フレーム位置の特徴量を付加してフレーム単位に展開している．
in_feats = fe.linguistic_features(labels, binary_dict, numeric_dict, add_frame_features=True, subphone_features="coarse_coding")

#音声ファイルの読み込み
sr=16000
_sr, x = wavfile.read(ttslearn.util.example_audio_file())
x=(x/32768).astype(np.float64)
x = librosa.resample(x, orig_sr=_sr, target_sr=sr)

#連続対数基本周波数と有声/無声フラグを結合した特徴量の計算
out_feats=world_log_f0_vuv(x.astype(np.float64), sr)

#フレーム数の調整
#理論上は長さ同じだけど，整数丸めやリサンプリング誤差で微妙にずれることがある．
minL=min(in_feats.shape[0], out_feats.shape[0])
in_feats, out_feats=in_feats[:minL], out_feats[:minL]

#冒頭と末尾の非音声区間の長さを調整
assert "sil" in labels.contexts[0] and "sil" in labels.contexts[-1]
#フレームシフト0.005秒
start_frame=int(labels.start_times[1]/50000)
end_frame=int(labels.end_times[-2]/50000)

#冒頭50ミリ秒，末尾100ミリ秒だけ非音声区間を残している．
#0になるのを防ぐ．
start_frame=max(0, start_frame-int(0.050/0.005))
#minLを超えてバッファ外参照になるのを防ぐ．
end_frame=min(minL, end_frame+int(0.100/0.005))

in_feats=in_feats[start_frame:end_frame]
out_feats=out_feats[start_frame:end_frame]


# %%
#code8.stage2
import subprocess
import os

# 実行したいシェルスクリプトのコマンド
command = ["./run.sh", "--stage", "2", "--stop-stage", "2"]
#確実に絶対パスで指定
working_dir = "/home/takamichi-lab-pc05/ドキュメント/B4/Pythonで学ぶ音声合成/ttslearn/recipes/wavenet"

# 実行
result = subprocess.run(command, cwd=working_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# 標準出力と標準エラー出力の確認
print("標準出力:")
print(result.stdout)

print("\n標準エラー:")
print(result.stderr)

#%%
#code8.4
from ttslearn.dsp import mulaw_quantize
#from ttslearn.dsp import world_log_fo_vuv

#HTS形式の質問ファイルを読み込み
binary_dict, numeric_dict = hts.load_question_set(ttslearn.util.example_qst_file())

#フルコンテキストラベルの読み込み
labels = hts.load(ttslearn.util.example_label_file())

#フレーム単位の言語特徴量を抽出
in_feats = fe.linguistic_features(labels, binary_dict, numeric_dict, add_frame_features=True, subphone_features="coarse_coding")

#音声ファイルの読み込み
sr=16000
_sr, x = wavfile.read(ttslearn.util.example_audio_file())
x=(x/32768).astype(np.float64)
x=librosa.resample(x, orig_sr=_sr, target_sr=sr)

#連続大数基本周波数と有声/無声フラグを結合した特徴量の計算
log_f0_vuv=world_log_f0_vuv(x.astype(np.float64), sr)

#フレーム数の調整
minL=min(in_feats.shape[0], log_f0_vuv.shape[0])
in_feats, log_f0_vuv=in_feats[:minL], log_f0_vuv[:minL]

#冒頭と末尾の非音声区間の長さを調整
assert "sil" in labels.contexts[0] and "sil" in labels.contexts[-1]
start_frame=int(labels.start_times[1]/50000)
end_frame=int(labels.end_times[-2]/50000)

#冒頭50ミリ秒，末尾100ミリ秒だけ非音声区間を残している．
start_frame=max(0, start_frame-int(0.050/0.005))
end_frame=min(minL, end_frame+int(0.100/0.005))

in_feats=in_feats[start_frame:end_frame]
log_f0_vuv=log_f0_vuv[start_frame:end_frame]

#言語特徴量と連続大数基本周波数を結合
in_feats=np.hstack([in_feats, log_f0_vuv])

#時間領域で音声の長さを調整
#0以上start_frame未満を切り落とし．
x=x[int(start_frame*0.005*sr):]
#波形長さ（サンプル数）を計算．
length=int(sr*0.005)*in_feats.shape[0]
#xの長さをin_featsに対応させるように，不足していたら0-padding，過剰なら末尾の切り落とし．
x=pad_1d(x, length)if len(x)<length else x[:length]

#mu-law量子化
quantized_x=mulaw_quantize(x)

#条件付特徴量のアップサンプリングを考えるため，音声波形の長さはフレームシフトで割り切れることを確認．
assert len(quantized_x) % int(sr*0.005) == 0


# %%
#code8.stage3
import subprocess
import os

# 実行したいシェルスクリプトのコマンド
command = ["./run.sh", "--stage", "3", "--stop-stage", "3"]
#確実に絶対パスで指定
working_dir = "/home/takamichi-lab-pc05/ドキュメント/B4/Pythonで学ぶ音声合成/ttslearn/recipes/wavenet"

# 実行
result = subprocess.run(command, cwd=working_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# 標準出力と標準エラー出力の確認
print("標準出力:")
print(result.stdout)

print("\n標準エラー:")
print(result.stderr)
# %%
#code8.stage4
import subprocess
import os

# 実行したいシェルスクリプトのコマンド
command = ["./run.sh", "--stage", "4", "--stop-stage", "4"]
#確実に絶対パスで指定
working_dir = "/home/takamichi-lab-pc05/ドキュメント/B4/Pythonで学ぶ音声合成/ttslearn/recipes/wavenet"

# 実行
result = subprocess.run(command, cwd=working_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# 標準出力と標準エラー出力の確認
print("標準出力:")
print(result.stdout)

print("\n標準エラー:")
print(result.stderr)
# %%
