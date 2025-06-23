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
#code8.5
#conf/train_dnn/model/duration_rnn.yamlに記載あり．

#%%
#code8.stage5
import subprocess
import os

# 実行したいシェルスクリプトのコマンド
command = ["./run.sh", "--stage", "5", "--stop-stage", "5"]
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
#code8.6
#conf/train_dnn/model/logf0_rnn.yamlに記載あり．

#%%
#code8.stage6
import subprocess
import os

# 実行したいシェルスクリプトのコマンド
command = ["./run.sh", "--stage", "6", "--stop-stage", "6"]
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
#code8.7
import torch

#batchは(条件特徴量/c, 波形/x)のタプルのリスト
#max_time_framesは切り出す最大のフレーム数
#hop_sizeは1フレームに対応する波形のサンプル数（80なら5ms）
#aux_context_windowで時間コンテキストを与える．
def collate_fn_wavenet(batch, max_time_frames=100, hop_size=80, aux_context_window=2):
   #切り出す波形の長さ（サンプル単位）
   max_time_steps=max_time_frames*hop_size
   xs, cs=[b[1] for b in batch], [b[0] for b in batch]
   
   #条件付け特徴量の開始位置をランダムに抽出したあと（start_frames），それにそれに対応する短い音声波形を切り出す．
   #ランダムにすることでデータの「多様な部分」から切り出せるため、汎化性能が向上する．
   c_lengths=[len(c)for c in cs]
   start_frames=np.array([np.random.randint(aux_context_window, cl-aux_context_window-max_time_frames)for cl in c_lengths])
   x_starts=start_frames*hop_size
   x_ends=x_starts+max_time_steps
   c_starts=start_frames-aux_context_window
   c_ends=c_starts+max_time_steps+aux_context_window
   #1ループでx[s:e]をそれぞれ抽出できる．この時点ではx_batchは長さBのリスト．
   """x_batch = [
    xs[0][x_starts[0]:x_ends[0]],  # → 長さ8000
    xs[1][x_starts[1]:x_ends[1]],  # → 長さ8000
    xs[2][x_starts[2]:x_ends[2]]   # → 長さ8000
   ]"""

   x_batch=[x[s:e]for x, s, e in zip(xs, x_starts, x_ends)]
   c_batch=[c[s:e]for c, s, e in zip(cs, c_starts, c_ends)]
   
   #numpy.ndarrayのリスト型からtorch.Tensor型に変換します
   x_batch=torch.tensor(x_batch, dtype=torch.long) #(B, T)
   c_batch=torch.tensor(c_batch, dtype=torch.float).transpose(2, 1) #(B, C, T')
   
   return x_batch, c_batch


# %%
#code8.8
from pathlib import Path
from functools import partial

from chapter8 import collate_fn_wavenet

#in_paths, out_pathsは入力・出力の特徴量のファイルパスのリストです．
dataset=Dataset(in_paths, out_paths)
collate_fn=partial(collate_fn_wavenet, max_time_frames=100, hop_size=80, aux_context_window=0)
data_loader=torch.utils.data.DataLoader(dataset, batch_size=8, collate_fn=collate_fn, num_workers=1)

wavs, feats=next(iter(data_loader))

print("音声波形のサイズ:", tuple(wavs.shape))
print("条件付け特徴量のサイズ:", tuple(feats.shape))
# %%
#code8.9
def moving_average_(model, model_test, beta=0.9999):
   for param, param_test in zip(model.parameters(), model_test.parameters()):
      param_test.data=torch.lerp(param_test.data, param.data, beta)
# %%
#code8.10
from ttslearn.wavenet import WaveNet
from torch import optim
from functools import partial

#動作確認用:層の数を減らした小さなWaveNet
ToyWaveNet=partial(WaveNet, out_channels=256, layers=2, stacks=1, kernel_size=2, cin_channels=333)

model=ToyWaveNet()
#モデルパラメータの指数移動平均
model_ema=ToyWaveNet()
model_ema.load_state_dict(model.state_dict())

optimizer=optim.Adam(model.parameters(), lr=0.01)

#gammaは学習率の減衰係数
lr_scheduler=optim.lr_scheduler.StepLR(optimizer, gamma=0.5, step_size=100000)
# %%
#code8.11
#DataLoaderを用いた未知バッチの作成:ミニバッチごとに処理する．
for x, c in data_loader:
   #順伝播の計算
   x_hat=model(x, c)
   #負の対数尤度の計算
   loss=nn.CrossEntropyLoss()(x_hat[:, :, :-1], x[:, 1:]).mean()
   #optimizerに蓄積された勾配をリセット
   optimizer.zero_grad()
   #誤差の逆伝播の計算
   loss.backward()
   #パラメータの更新
   optimizer.step()
   #指数移動平均の更新
   moving_average_(model, model_ema)
   #学習率スケジューラの更新
   lr_scheduler.step()
   
# %%
#code8.12
#conf/train_wavenet/model/wavenet_sr16k_mulaw256.yamlに記載あり．

#%%
#code8.stage7
import subprocess
import os

# 実行したいシェルスクリプトのコマンド
command = ["./run.sh", "--stage", "7", "--stop-stage", "7"]
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
#code8.13
from ttslearn.dsp import inv_mulaw_quantize
import numpy as np
import torch
from tqdm import tqdm


@torch.no_grad()
def gen_waveform(
   device, #cpu or cuda
   labels, #フルコンテキストラベル
   logf0_vuv, #連続対数基本周波数と有声/無声フラグ
   wavenet_model, #学習済みWaveNet
   wavenet_in_scaler, #条件特徴量の正規化用StandardScaler
   binary_dict, #二値特徴量を抽出する正規表現
   numeric_dict, #数値特徴量を抽出する正規表現
   tqdm=tqdm, #プログレスバー
):
   #フレーム単位の言語特徴量の抽出
   in_feats=fe.linguistic_features(labels, binary_dict, numeric_dict, add_frame_features=True, subphone_features="coarse_coding")
   #フレーム単位の言語特徴量と対数連続基本周波数・有声/無声フラグを結合
   in_feats=np.hstack([in_feats, logf0_vuv])
   
   #特徴量の正規化
   in_feats=wavenet_in_scaler.transform(in_feats)
   
   #条件付け特徴量をnumpy.ndarrayからtorch.Tensorに変換
   c=torch.from_numpy(in_feats).float(),to(device)
   #(B, T, C) -> (B, C, T')に変換
   c=c.view(1, -1, c.size(-1)).transpose(1, 2)
   
   #音声波形の長さを計算
   upsample_scale=np.prod(wavenet_model.upsample_scales)
   
   time_steps=(C.shape[-1]-wavenet_model.aux_context_window*2)*upsample_scale
   #WaveNetによる音声波形の生成
   gen_wave=wavenet_model.inference(c, time_steps=time_steps, tqdm=tqdm)
   
   #One-hotベクトルから1次元の信号に変換
   gen_wave=gen_wave.max(1)[1].float().cpu().numpy().reshape(-1)
   
   #mu-law量子化の逆変換
   gen_wave=inv_mulaw_quantize(gen_wave, wavenet_model.out_channels-1)
   
   return gen_wave
   
# %%
#code8.stage8
import subprocess
import os

# 実行したいシェルスクリプトのコマンド
command = ["./run.sh", "--stage", "8", "--stop-stage", "8"]
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
