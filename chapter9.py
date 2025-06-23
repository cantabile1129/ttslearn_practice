#%%
#code9.1
#英語を想定した語彙（文字の集合）の定義，及び文字と数値の相互変換のための辞書の定義
#語彙の定義
characters = "abcdefghijklmnopqrstuvwxyz!'(),-.:;? "
#その他特殊記号
#ポーズや吸気音・呼気音をテキストから予測することは困難．
extra_symbols=[
   "^", #文の先頭を表す<SOS>
   "$", #文の終端を表す<EOS>
]
#パディングを表す
#Tacotron2では学習時に異なる複数の系列からミニバッチを構成する．
#数値に変換すると0になる．
_pad="~" 

symbols=[_pad]+extra_symbols+list(characters)

#文字列⇔数値の相互変換のための辞書
"""
(0, symbols[0])  # _pad → "~"
(1, symbols[1])  # "^"
(2, symbols[2])  # "$"
(3, symbols[3])  # "a"
(4, symbols[4])  # "b"
...
"""
_symbol_to_id={s: i for i, s in enumerate(symbols)}
_id_to_symbol={i: s for i, s in enumerate(symbols)}
# %%
#code9.2
#英語を想定した文字列から数値列への変換
def text_to_sequence(text):
   #簡易のため，大文字と小文字を区別せず，すべての大文字を小文字に変換
   text = text.lower()
   
   #<SOS>
   seq= [_symbol_to_id["^"]]
   #本文
   seq+=[_symbol_to_id[s] for s in text]
   #<EOS>
   seq.append(_symbol_to_id["$"])
   
   return seq

def sequence_to_text(seq):
   return [_id_to_symbol[s] for s in seq]

#%%
#code9.3
seq=text_to_sequence("Hello, world!")
print(f"文字列から数値列への変換: {seq}")
print(f"数値列から文字列への逆変換: {sequence_to_text(seq)}")

#%%
#code9.4
import torch
import torch.nn as nn

class SimplestEncoder(nn.Module):
   #パディングの数値表現に0を割り当てたので（~のこと），0に設定．
   #embed_dimは文字を変換して得られるベクトルの次元数
   def __init__(self, num_vocab=40, embed_dim=256):
      super().__init__()
      self.embed=nn.Embedding(num_vocab, embed_dim, padding_idx=0)
      
   def forward(self, seqs):
      return self.embed(seqs)
   
   
#%%
#code9.5
from ttslearn.util import pad_1d

def get_dummy_input():
   #バッチサイズに2を想定して，適当な文字列を作成する．
   seqs=[
      text_to_sequence("What's your name?"),
      text_to_sequence("My name is Tacotron2."),
   ]
   #long型のテンソルに変換
   in_lens=torch.tensor([len(x) for x in seqs], dtype=torch.long)
   #パディング後の系列長
   max_len=max(len(x) for x in seqs)
   seqs=torch.stack([torch.from_numpy(pad_1d(seq, max_len)) for seq in seqs])
   #seqs:[B, T]で，in_lens:[B]
   return seqs, in_lens

#%%
#code9.6
encoder=SimplestEncoder(num_vocab=40, embed_dim=256)
seqs, in_lens=get_dummy_input()
encoder_outs=encoder(seqs)
print(f"入力のサイズ: {tuple(seqs.shape)}")
print(f"出力のサイズ: {tuple(encoder_outs.shape)}")



#%%
#code9.7
class ConvEncoder(nn.Module):
   def __init__(self, num_vocab=40, embed_dim=256, conv_layers=3, conv_channels=256, conv_kernel_size=5):
      super().__init__()
      self.embed=nn.Embedding(num_vocab, embed_dim, padding_idx=0)
      
      #1次元畳み込みの重ね合わせ:局所的な依存関係のモデル化
      self.convs=nn.ModuleList()
      for layer in range(conv_layers):
         in_channels=embed_dim if layer==0 else conv_channels
         self.convs+=[
            nn.Conv1d(in_channels, conv_channels, conv_kernel_size, padding=(conv_kernel_size-1)//2, bias=False),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
         ]
      self.convs=nn.Sequential(*self.convs)
      
   def forward(self, seqs):
      emb = self.embed(seqs)
      #1次元畳み込みとembeddingでは，入力のサイズが異なるので注意．
      #embのサイズは[B, T, D]で，convsの入力は[B, D, T]なので転置して，また[B, T, D]に戻す．
      out=self.convs(emb.transpose(1, 2)).transpose(1, 2)
      return out
   
#%%
#code9.8
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(ConvEncoder):
   def __init__(self, num_vocab=40, embed_dim=512, hidden_dim=512, conv_layers=3, conv_channels=512, conv_kernel_size=5):
      super().__init__(num_vocab, embed_dim, conv_layers, conv_channels, conv_kernel_size)
      #双方向LSTMによる長期依存関係のモデル化
      self.blstm=nn.LSTM(conv_channels, hidden_dim//2, num_layers=1, batch_first=True, bidirectional=True)
      
      def forward(self, seqs, in_lens):
         emb=self.embed(seqs)
         out=self.convs(emb.transpose(1, 2)).transpose(1, 2)
         #双方向LSTMの計算
         out=pack_padded_sequence(out, in_lens, batch_first=True)
         out, _ = self.blstm(out)
         out, _ = pad_packed_sequence(out, batch_first=True)
         return out

#%%

         