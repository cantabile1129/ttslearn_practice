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
command = ["./run.sh", "--stage", "-1", "--stop-stage", "8"]
#確実に絶対パスで指定
working_dir = "/home/takamichi-lab-pc05/ドキュメント/B4/Pythonで学ぶ音声合成/ttslearn/recipes/dnntts"

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
command = ["./run.sh", "--stage", "0", "--stop-stage", "8"]
#確実に絶対パスで指定
working_dir = "/home/takamichi-lab-pc05/ドキュメント/B4/Pythonで学ぶ音声合成/ttslearn/recipes/dnntts"

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
command = ["./run.sh", "--stage", "1", "--stop-stage", "8"]
#確実に絶対パスで指定
working_dir = "/home/takamichi-lab-pc05/ドキュメント/B4/Pythonで学ぶ音声合成/ttslearn/recipes/dnntts"

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

def world_log_f0_vuv(x):
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