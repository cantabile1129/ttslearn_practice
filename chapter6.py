#%%
#code6-0
!pip install jupyterlab
!pip install ttslearn

#%%
#code6.1
from ttslearn.dnntts import DNNTTS
from IPython.display import Audio

dnntts_engine = DNNTTS()
wav, sr = dnntts_engine.tts("深層学習に基づく音声合成システムです。")
Audio(wav, rate=sr)
# %%
#code6.stage-1
#コマンドラインではなくセルで実行できるように調整
#notionの通りステージが-1から6まであり，順番に実行．
import subprocess
import os

# 実行したいシェルスクリプトのコマンド
command = ["./run.sh", "--stage", "-1", "--stop-stage", "-1"]
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
#code6.stage0
import subprocess
import os

# 実行したいシェルスクリプトのコマンド
command = ["./run.sh", "--stage", "0", "--stop-stage", "0"]
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
#code6.2
#1発話の前処理

import ttslearn
from nnmnkwii.io import hts
from nnmnkwii.frontend import merlin as fe

#言語特徴量の抽出に使うための質問ファイル
binary_dict, numeric_dict = hts.load_question_set(ttslearn.util.example_qst_file())

#音声のフルコンテキストラベルの読み込み
labels = hts.load(ttslearn.util.example_label_file())

#継続長モデルの入力:言語特徴量
#binary_dictはTrue/Falseで答えられるいわば決定疑問文．
#numeric_dictは数値で答えられる疑問文で，位置や長さを問う．
in_feats = fe.linguistic_features(labels, binary_dict, numeric_dict)

#継続長モデルの出力:音素継続長
out_feats = fe.duration_features(labels)

print("入力特徴量のサイズ:", in_feats.shape)
print("出力特徴量のサイズ:", out_feats.shape)

# %%
#code6.3
#Jupyterでargparseなしで引数を模倣
from types import SimpleNamespace
from pathlib import Path
#プログレスバーの表示
from tqdm import tqdm
import numpy as np
from nnmnkwii.io import hts
from nnmnkwii.frontend import merlin as fe
#マルチプロセス化
from concurrent.futures import ProcessPoolExecutor

# Jupyter上では argparse を使わずに引数を手動設定
args = SimpleNamespace(
    utt_list="/home/takamichi-lab-pc05/ドキュメント/B4/Pythonで学ぶ音声合成/ttslearn/recipes/dnntts/data/utt_list.txt",
    #code6.stage-1で作成
    lab_root="/home/takamichi-lab-pc05/ドキュメント/B4/Pythonで学ぶ音声合成/ttslearn/recipes/dnntts/downloads/jsut-lab-0.1.1/basic5000/lab",
    qst_file="/home/takamichi-lab-pc05/ドキュメント/B4/Pythonで学ぶ音声合成/ttslearn/recipes/common/qst1.hed",
    #出力先ディレクトリ
    out_dir="/home/takamichi-lab-pc05/ドキュメント/B4/Pythonで学ぶ音声合成/ttslearn/recipes/dnntts/data",
    #並列処理で使うプロセス数
    n_jobs=2
)

def preprocess(lab_file, binary_dict, numeric_dict, in_dir, out_dir):
    labels = hts.load(lab_file)
    in_feats = fe.linguistic_features(labels, binary_dict, numeric_dict)
    out_feats = fe.duration_features(labels)
    utt_id = lab_file.stem
    #in_duration/BASIC5000_0001-feats.npyのように入力
    np.save(in_dir / f"{utt_id}-feats.npy", in_feats.astype(np.float32), allow_pickle=False)
    #out_duration/BASIC5000_0001-feats.npyのように出力
    np.save(out_dir / f"{utt_id}-feats.npy", out_feats.astype(np.int32), allow_pickle=False)

# 処理の流れ
with open(args.utt_list) as f:
   #utt_list を読み込み、BASIC5000_0001 などのIDをリスト化
    utt_ids = [utt_id.strip() for utt_id in f]
    #各IDから .lab ファイルのフルパスを作成
lab_files = [Path(args.lab_root) / f"{utt_id}.lab" for utt_id in utt_ids]

binary_dict, numeric_dict = hts.load_question_set(args.qst_file)

in_dir = Path(args.out_dir) / "in_duration"
out_dir = Path(args.out_dir) / "out_duration"
in_dir.mkdir(parents=True, exist_ok=True)
out_dir.mkdir(parents=True, exist_ok=True)

with ProcessPoolExecutor(args.n_jobs) as executor:
    futures = [executor.submit(preprocess, lab_file, binary_dict, numeric_dict, in_dir, out_dir) for lab_file in lab_files]
    for future in tqdm(futures):
        future.result()
# %%
#code6.4.stage1
#コマンドラインではなくセルで実行できるように調整
#notionの通りステージが-1から6まであり，順番に実行．
import subprocess
import os

# 実行したいシェルスクリプトのコマンド
command = ["./run.sh", "--stage", "1", "--stop-stage", "1"]
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
from ttslearn.dsp import world_spss_params

#音響モデルの入力:言語特徴量
in_feats = fe.linguistic_features(labels, binary_dict, numeric_dict, add_frame_features=True, subphone_features="corase_coding")

#音声の読み込み
_sr, x = wavfile.read(ttslearn.util.example_wav_file())
sr = 16000
x = (x / 32768).astype(np.float64)
x = librosa.resample(x, _sr, sr)

#音響モデルの出力:音響特徴量
out_feats = world_spss_params(x, sr)

#フレーム数の調整
minL = min(in_feats.shape[0], out_feats.shape[0])
in_feats = in_feats[:minL]
out_feats = out_feats[:minL]

#冒頭と末尾の非音声区間の長さを調整
assert "sil" in labels.contexts[0] and "sil" in labels.context[-1]
start_frame = int(labels.start_times[1]/50000)
end_frame = int(labels.end_times[-2]/50000)

#冒頭50ミリ秒，末尾100ミリ秒
start_frame = max(0, start_frame - int(0.050/0.005))
end_frame = min(minL, end_frame + int(0.100/0.005))

in_feats = in_feats[start_frame:end_frame]
out_feats = out_feats[start_frame:end_frame]


# %%
