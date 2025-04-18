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

import argparse
#pythonインタプリタの動作制御（実行環境の状態や挙動を制御，コマンドライン引数を受け取る）
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tqdm import tqdm
import numpy as np

def get_parser():
   parser = argparse.ArgumentParser(description="Preprocess duration models")
   parser.add_argument("utt_list", type=str, help="utternace list")
   parser.add_argument("lab_root", type=str, help="lab_root")
   parser.add_argument("qst_file", type=str, help="HTS Style question file")
   parser.add_argument("out_dir", type=str, help="output directory")
   parser.add_argument("--n_jobs", type=int, default = 1, help="Number of jobs")
   return parser

def preprocess(lab_file, binary_dict, numeric_dict, in_Dir, out_dir):
   labels=hts.loads(lab_file)
   in_feats = fe.linguistic_features(labels, binary_dict, numeric_dict)
   out_feats = fe.duration_features(labels)
   utt_id = lab_file.stem
   np.save(in_dir / f"{utt_id}-feats.npy", in_feats.astype(np.float32), allow_pickle =False)  
   np.save(out_dir / f"{utt_id}-feats.npy", out_feats.astype(np.int32), allow_pickle =False)

if __name__ == "__main__":
   args = get_parser().parse_args(sys.argv[1:])
   
   with open(args.utt_list) as f :
      utt_ids = [utt_id.strip() for utt_id in f]
   lab_root = [Path(args.lab_root) / f"{utt_id}.lab" for utt_id in utt_ids]
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
#code6.3.revised

from types import SimpleNamespace
from pathlib import Path
from tqdm import tqdm
import numpy as np
from nnmnkwii.io import hts
from nnmnkwii.frontend import merlin as fe
from concurrent.futures import ProcessPoolExecutor

# Jupyter上では argparse を使わずに引数を手動設定
args = SimpleNamespace(
    utt_list="data/utt_list.txt",
    lab_root="data/lab",
    qst_file="data/questions.hed",
    out_dir="data/output",
    n_jobs=2
)

def preprocess(lab_file, binary_dict, numeric_dict, in_dir, out_dir):
    labels = hts.load(lab_file)
    in_feats = fe.linguistic_features(labels, binary_dict, numeric_dict)
    out_feats = fe.duration_features(labels)
    utt_id = lab_file.stem
    np.save(in_dir / f"{utt_id}-feats.npy", in_feats.astype(np.float32), allow_pickle=False)
    np.save(out_dir / f"{utt_id}-feats.npy", out_feats.astype(np.int32), allow_pickle=False)

# 処理の流れ
with open(args.utt_list) as f:
    utt_ids = [utt_id.strip() for utt_id in f]
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
