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
#code6.4
#import漏れが多すぎる

import ttslearn
import librosa
from scipy.io import wavfile
import numpy as np
from nnmnkwii.frontend import merlin as fe
from nnmnkwii.io import hts
from ttslearn.dsp import world_spss_params

#音声のフルコンテキストラベルの読み込み
labels = hts.load(ttslearn.util.example_label_file())

#音響モデルの入力:言語特徴量
#add_frame_features=Trueは音素単位の特徴量をフレーム単位に展開しており，subphone_features="coarse_coding"はその際に書く音素内における洗位置情報を付加することを意味している．
in_feats = fe.linguistic_features(labels, binary_dict, numeric_dict, add_frame_features=True, subphone_features="coarse_coding")

#音声の読み込み
#example_wav_file→example_audio_fileに変更
_sr, x = wavfile.read(ttslearn.util.example_audio_file())
sr = 16000
#浮動少数（-1.0〜1.0）に変換
x = (x / 32768).astype(np.float64)
x = librosa.resample(x, orig_sr=_sr, target_sr=sr)

#音響モデルの出力:音響特徴量
out_feats = world_spss_params(x, sr)

#フレーム数の調整で，短い方に合わせる．
minL = min(in_feats.shape[0], out_feats.shape[0])
in_feats = in_feats[:minL]
out_feats = out_feats[:minL]

#冒頭と末尾の非音声区間の長さを調整
#1フレームは50ミリ秒=50000ナノ秒
#最初の音素と最後の音素に無音があるかを確認
assert "sil" in labels.contexts[0] and "sil" in labels.contexts[-1]
#第2番目の音素（先頭のsilの次）の開始時刻
start_frame = int(labels.start_times[1]/50000)
#最後から2番目の音素（末尾のsilの前）の終了時刻
end_frame = int(labels.end_times[-2]/50000)
#実音声部分の開始と終了のフレーム番号

#冒頭50ミリ秒，末尾100ミリ秒の余裕を持たせる．非音声区間の長さを統一して学習データ量の削減をする．
start_frame = max(0, start_frame - int(0.050/0.005))
end_frame = min(minL, end_frame + int(0.100/0.005))

in_feats = in_feats[start_frame:end_frame]
out_feats = out_feats[start_frame:end_frame]


#継続帳モデルより4ずつ増えているが，音素内のフレーム単位の位置特徴量が追加されたことに起因．
print("入力特徴量のサイズ:", in_feats.shape)
print("出力特徴量のサイズ:", out_feats.shape)

from ttslearn.dnntts.multistream import get_static_features

#動的特徴量を除いて，各音響特徴量を取り出す．
mgc, lf0, vuv, bap = get_static_features(out_feats, num_windows=3, stream_sizes=[120, 3, 1, 3], has_dynamic_features=[True, True, False, True])
print("メルケプストラムのサイズ:", mgc.shape)
print("連続対数基本周波数のサイズ:", lf0.shape)
print("有声/無声フラグのサイズ:", vuv.shape)
print("帯域非周期性指標のサイズ:", bap.shape)
# %%
#code6.5.stage2
import subprocess
import os

# 実行したいシェルスクリプトのコマンド
command = ["./run.sh", "--stage", "2", "--stop-stage", "2"]
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
#code6.5
#正規化のための特徴量の計算のコマンドラインプログラム
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

#Jupyterでargparseなしで引数を模倣
from types import SimpleNamespace
from pathlib import Path

from tqdm import tqdm

args = SimpleNamespace(
    utt_list="/home/takamichi-lab-pc05/ドキュメント/B4/Pythonで学ぶ音声合成/ttslearn/recipes/dnntts/data/utt_list.txt",
    in_dir="/home/takamichi-lab-pc05/ドキュメント/B4/Pythonで学ぶ音声合成/ttslearn/recipes/dnntts/data/in_duration",
    out_path="/home/takamichi-lab-pc05/ドキュメント/B4/Pythonで学ぶ音声合成/ttslearn/recipes/dnntts/data/in_scaler.joblib"
)

# 特徴量正規化のスケーラー作成
#ディレクトリをPathオブジェクトに変換
in_dir = Path(args.in_dir)
#平均と標準偏差の記録
scaler = StandardScaler()

# 発話IDのリストを読み込む
#ファイルを開き，fを通して中身を使う
with open(args.utt_list) as f:
    for utt_id in tqdm(f):
        utt_id = utt_id.strip()
        feat_path = in_dir / f"{utt_id}-feats.npy"
        #2次元のはず
        c = np.load(feat_path)
        scaler.partial_fit(c)

# スケーラーを保存
joblib.dump(scaler, args.out_path)
#計算したスケーラーをjoblibで保存
print(f"Saved scaler to: {args.out_path}")
# %%
#code6.6.stage3
import subprocess
import os

# 実行したいシェルスクリプトのコマンド
command = ["./run.sh", "--stage", "3", "--stop-stage", "3"]
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
#code6.6
import torch
import torch.nn as nn
class DNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2):
        #親クラス（nn.Module）を呼び出し，初期設定をする
        super(DNN, self).__init__()
        model = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        #ループ変数は使わないので_にしている
        for _ in range(num_layers):
            model.append(nn.Linear(hidden_dim, hidden_dim))
            model.append(nn.ReLU())
        model.append(nn.Linear(hidden_dim, out_dim))
        #リストで作った以下の層を，順番にまとめて一つのネットワークとして定義．nn.Sequential()は複数のレイヤーを順番に流すためのラッパーで，*modelはリストの中身を展開（アンパック）する．
        """
            model = [
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            ...
            nn.Linear(hidden_dim, out_dim)
            ]
        """
        self.model = nn.Sequential(*model)
        
    def forward(self, x, lens=None):
        return self.model(x)

# %%
#code6.7
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMRNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=1, bidirectional=True, dropout=0.0):
        super(LSTMRNN, self).__init__()
        self.num_direction = 2 if bidirectional else 1
        #入力の形が(batch, time, feature)．bidirectional引数で片方向か双方向か切り替える．
        self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional, batch_first=True, dropout=dropout)
        #LSTMの出力は隠れ状態の数×2（双方向）で，出力次元はout_dim
        self.hidden2out = nn.Linear(hidden_dim * self.num_direction, out_dim)

    def forward(self, seqs, lens):
        # pack_padded_sequenceは，長さの異なる系列をバッチ処理するために必要な前処理．
        # lensに基づきパディングを無視できる圧縮表現に変換する．無駄なゼロをLSTMに見せないことで，効率的に学習できる．
        seqs = pack_padded_sequence(seqs, lens, batch_first=True)
        out, _ = self.lstm(seqs)
        #元の長さ＋パディングありの形状に戻す
        #(batch_size, max_seq_len, hidden_dim * num_direction)
        out, _ = pad_packed_sequence(out, batch_first=True)
        #outはLSTMの出力で，最後の隠れ状態を取り出す．
        out = self.hidden2out(out)
        return out
# %%
#code6.8
#PytorchのDataLoaderを用いるためにDatasetクラスを定義する
import torch
from torch.utils import data as data_utils

class Dataset(data_utils.Dataset):
    #in_pathsは入力特徴量の.npyファイルのパス一覧で，out_pathsは出力特徴量の.npyファイルのパス一覧
    def __init__(self, in_paths, out_paths):
        self.in_paths = in_paths
        self.out_paths = out_paths
        
    # 指定したインデックスの入出力ファイルを読み込み，1ペアのデータ（Numpy配列）として返す．
    def __getitem__(self, idx):
        return np.load(self.in_paths[idx]), np.load(self.out_paths[idx])
    
    # データセットの長さ（サンプル数）を返す．
    def __len__(self):
        return len(self.in_paths)

#%%
#code6.9
#エラー解消できず
from pathlib import Path
from ttslearn.util import pad_2d
import numpy as np

def collate_fn_dnntts(batch):
    lengths = [len(x[0]) for x in batch]
    #paddingの目標長は，バッチ内の系列長の最大値
    max_len = max(lengths)
    #code6.8でペアにしていたように，x[0]は入力で，x[1]は出力
    #パディングした配列をテンソルに変換し，それを3次元テンソル(B, T_max, D)にまとめる
    x_batch = torch.stack([torch.from_numpy(pad_2d(x[0], max_len)) for x in batch])
    y_batch = torch.stack([torch.from_numpy(pad_2d(x[1], max_len)) for x in batch])
    #元の系列長（padding前）をテンソル化
    lengths = torch.tensor(lengths, dtype=torch.long)
    return x_batch, y_batch, lengths

#in_pathsとout_pathsの定義が必要
in_paths = sorted(Path("/home/takamichi-lab-pc05/ドキュメント/B4/Pythonで学ぶ音声合成/ttslearn/recipes/dnntts/dump/jsut_sr16000/norm/dev/in_duration").glob("*.npy"))
out_paths = sorted(Path("/home/takamichi-lab-pc05/ドキュメント/B4/Pythonで学ぶ音声合成/ttslearn/recipes/dnntts/dump/jsut_sr16000/norm/dev/in_duration").glob("*.npy"))

dataset = Dataset(in_paths, out_paths)
#num_worker=0でプロセス並列化しないことを指定

data_loader = data_utils.DataLoader(dataset, batch_size=8, collate_fn=collate_fn_dnntts, num_workers=0)
in_feats, out_feats, lengths = next(iter(data_loader))

print("入力特徴量のサイズ:", tuple(in_feats.size()))
print("出力特徴量のサイズ:", tuple(out_feats.size()))
print("系列長のサイズ:", tuple(lengths.size()))
# %%
#code6.10
from ttslearn.dnntts import DNN
from torch import optim

model = DNN(in_dim=325, hidden_dim=64, out_dim=1, num_layers=2)

#lrは学習率
optimizer = optim.Adam(model.parameters(), lr=0.01)

#gammaは学習率の減衰係数を表す
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# %%
#code6.11
#DataLoaderを用いたミニバッチの作成
import torch.nn as nn

for in_feats, out_feats, lengths in data_loader:
    #順伝播
    pred_out_feats = model(in_feats, lengths)
    #損失
    loss = nn.MSELoss()(pred_out_feats, out_feats)
    #optimizerに蓄積された勾配をリセット
    optimizer.zero_grad()
    #逆伝播
    loss.backward()
    #パラメータの更新
    optimizer.step()

# %%
#code6.12
#/home/takamichi-lab-pc05/ドキュメント/B4/Pythonで学ぶ音声合成/conf_chapter6/config.yaml

#%%
#code6.13
#以下はJupyterでのHydraの使い方だが，今後はpy_chapter6に移動．

import hydra 
from omegaconf import DictConfig, OmegaConf

#hydra.mainをJupyter対応させるAP
from hydra import initialize, compose

#以下Jupyter用
# 1. 初期化：configファイルのディレクトリを指定（相対パス or 絶対パス）
initialize(config_path="conf_chapter6")
# 2. 設定ファイルの読み込み（ファイル名から .yaml は省略する）
# overrideで，Jupyter形式のpy形式やipynb形式ではできないコマンド引数からの設定変更の代替となる．
cfg: DictConfig = compose(config_name="config", overrides=["train.batch_size=32", "model.hidden_dim=256"])
# 3. 表示（YAML形式で見やすく出力）
print(OmegaConf.to_yaml(cfg))

# %%