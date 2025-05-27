#%%
#code7.1
#μ-law algorithmの変換
import numpy as np
#np.log1pはln(1+x)だけど，この方が精度が良い．
#入力は-1〜1で，出力も-1〜1
def mulaw(x, mu=255):
   return np.sign(x)*np.log1p(mu*np.abs(x))/np.log1p(mu)

#出力は0〜255を取る．
def quantize(y, mu=255, offset=1):
   #[-1, 1] -> [0, 2] -> [0, 1] -> [0, mu]
   return ((y+offset)/2*mu).astype(np.int64)

def mulax_quantize(x, mu=255):
   return quantize(mulaw(x, mu), mu)

#%%
#code7.2
#μ-law algorithmの逆変換
def inv_mulaw(y, mu=255):
   return np.sign(y)*(1.0/mu)*((1.0+mu)**np.abs(y)-1.0)

def inv_quantize(y, mu):
   #[0, mu] -> [-1, 1]
   return 2*y.astype(np.float32)/mu-1

def inv_mulaw_quantize(y, mu=255):
   return inv_mulaw(inv_quantize(y, mu), mu)

#%%
#code7.3
import torch
import torch.nn as nn
class CausalConv1d(nn.Module):
   def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
      super().__init__()
      #入力長と出力長の差を解消するための片方分．
      self.padding=kernel_size-1
      #**kwargsは，キーワード付き引数（ex.)stride=1）をまとめて受け取り，dict形式で保持する構文．
      #nn.Conv1dは，デフォルトでは左右対称のパディングを行う．
      self.conv=nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, **kwargs)
      
   def forward(self, x):
      y=self.conv(x)
      #因果性を保つために，順方向シフトする．
      if self.padding>0:
         #yは[batch, channel, time]の形状を持つ3次元テンソル．
         #右端のpadding分だけ未来の状態を持つので，それを削除する．
         y=y[:, :, :-self.padding]
      return y
   
#%%
#code7.4

class DilatedCausalConv1d(nn.Module):
   def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
      super().__init__()
      #入力の一番左に着目して，カーネルの一番右以外の個数をdilation factorだけかける．
      self.padding=(kernel_size-1)*dilation
      self.conv=nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation, **kwargs)
      
   def forward(self, x):
      y=self.conv(x)
      if self.padding>0:
         y=y[:, :, :-self.padding]
      return y
   
#%%
#code7.5
class GatedDilatedCausalConv1d(nn.Module):
   def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
      super().__init__()
      self.padding=(kernel_size-1)*dilation
      self.conv=nn.Conv1d(in_channels, out_channels*2, kernel_size, padding=self.padding, dilation=dilation)
      
   def forward(self, x):
      y=self.conv(x)
      if self.padding>0:
         y=y[:, :, :-self.padding]
      
      #チャネル方向の分割．
      #dim=1でチャネル方向に指定し，同じテンソルが二つ繋がっているのでそれを前半と後半に分ける．
      a, b=y.split(y.size(1)//2, dim=1)
      
      y=torch.tanh(a)*torch.sigmoid(b)
      return y
# %%
#code7.6
class RepeatUpSampling(nn.Module):
   #upsample_scalesはリストで，倍率を指定する．
   def __init__(self, upsample_scales):
      super().__init__()
      #np.prodは行方向の積で，[12, 5, 2]なら120．段階的にアップサンプリングする．
      self.upsample=nn.Upsample(scale_factor=np.prod(upsample_scales))
      
   def forward(self, c):
      return self.upsample(c)
   
#%%
#code7.7
from torch.nn import functional as F

class UpsampleNetwork(nn.Module):
   def __init__(self, upsample_scales):
      super().__init__()
      self.upsample_scales=upsample_scales
      #可変個数の子モジュールを持ちたい．PyTorchにこのリスト内のモジュールを追跡・保存・パラメータ管理させるための重要ステップ．
      self.conv_layers=nn.ModuleList()
      #upsample_scalesはリストで，倍率を指定する．
      for scale in upsample_scales:
         #w=(k-1)/2の変形で，k=2w+1．
         kernel_size=(1, scale*2+1)
         #padding=(0, scale)は左右それぞれscale個分だけあるが，右だけパディング．因果性畳み込み．
         #単なる平滑化なのでbias=False．
         #Conv2dで全チャンネルを独立に平滑化でき，時間軸だけの畳込みができる．
         conv=nn.Conv2d(1, 1, kernel_size=kernel_size, padding=(0, scale), bias=False)
         #kernek_size分の入力値を足して平均を取る．単なる平均だけど，平滑化している．
         #カーネルサイズの逆数により初期化する．
         conv.weight.data.fill_(1.0/np.prod(kernel_size))
         self.conv_layers.append(conv)
         
   def forward(self, c):
      #cを[batch, channel, time]の3次元テンソルから，[batch, 1, channel, time]の形状を持つ4次元テンソルにする．
      #これでConv2dで扱えるようにしている．
      c = c.unsqueeze(1)
      #最近傍補間と畳込みの繰り返し
      for idx, scale in enumerate(self.upsample_scales):
         #最近傍補間（F.interpolateはアップサンプリングで，時間軸のみscale倍する．）
         c = F.interpolate(c, scale_factor=(1, scale))
         #畳込み（__init__()で作られた平均フィルタを適用．）
         c = self.conv_layers[idx](c)
      #cを[batch, channel, time]の形状を持つ3次元テンソルに戻す．
      return c.squeeze(1)
   
#%%
#code7.8
class ConvInUpsampleNetwork(nn.Module):
   #cin_channelsは条件つけ特徴量の次元数（80次元のメルスペクトログラム等），aux_context_windowは近傍なんフレームの特徴量を考慮するかを，それぞれ表す．
   def __init__(self, upsample_scales, cin_channels, aux_context_window):
      super(ConvInUpsampleNetwork, self).__init__()
      #aux_context_window（前）+1+aux_context_window（後）=kernel_size
      kernel_size=2*aux_context_window+1
      #文脈平滑化．入力と出力のチャネル数は揃えたい．
      self.conv_in=nn.Conv1d(cin_channels, cin_channels, kernel_size, bias=False)
      #16,000Hzに対して，条件特徴量は80〜100フレーム/sなので，補間して伸ばす．
      self.upsample=UpsampleNetwork(upsample_scales)
      
   def forward(self, c):
      return self.upsample(self.conv_in(c))
# %%
#code7.9
#1×1の畳み込みは，チャンネル数を変換するために使う．
def Conv1d1x1(in_channels, out_channels, bias=True):
   return nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0, dilation=1, bias=bias)

#%%
#code7.10
#各コードがnotionの写真のどこに対応しているかを参照すること．
class ResSkipBlock(nn.Module):
   def __init__(
      self,
      residual_channels, 
      gate_channels, 
      kernel_size, 
      skip_out_channels,
      dilation=1,
      cin_channels=80, #条件特徴量の次元数だけど，見るからに80次元のメルスペクトログラム．
      *args, #リストやタプルのアンパック
      **kwargs #辞書などキーワード引数のアンパック
   ):
      super().__init__()
      self.padding=(kernel_size-1)*dilation
      
      #1次元膨張畳み込み（dilation==1のときは通常の1次元畳み込み）
      self.conv=nn.Conv1d(residual_channels, gate_channels, kernel_size, padding=self.padding, dilation=dilation, *args, **kwargs)
      
      #local conditioning用の1×1 convolution
      self.conv1x1c=Conv1d1x1(cin_channels, gate_channels, bias=False)
      
      #ゲートつき活性化関数のために，1次元畳み込みの出力は2分割されることに注意．
      gate_out_channels=gate_channels//2
      
      self.conv1x1_out=Conv1d1x1(gate_out_channels, residual_channels)
      self.conv1x1_skip=Conv1d1x1(gate_out_channels, skip_out_channels)
      
   def forward(self, x, c):
      #残差接続用に入力を保持する．
      residual=x
      
      #1次元畳み込み．[batch, channel, time]の形状を持つ3次元テンソル．
      splitdim=1
      x=self.conv(x)
      #因果性を保証するために，出力をシフトとする．
      x=x[:, :, :-self.padding]
      
      #チャンネル方向で出力を分割する．
      a, b=x.split(x.size(1)//2, dim=1)
      
      #local conditioning
      #系列長は既に等しいので，チャンネル数を合わせる．
      c = self.conv1x1c(c)
      ca, cb = c.split(c.size(1)//2, dim=1)
      a, b = a+ca, b+cb
      
      #ゲート付き活性化関数
      x=torch.tanh(a)*torch.sigmoid(b)
      
      #スキップ接続用の出力を計算
      #後ほど複数ブロック分化さんするため．
      s=self.conv1x1_skip(x)
      
      #残差接続の要素和を行う前に，次元数を合わせる．
      x=self.conv1x1_out(x)
      
      x = x+residual
      
      return x, s
# %%
#code7.11
import torch
import torch.nn as nn

class WaveNet(nn.Module):
   def __uinit__(
      self,
      out_channels=256, #出力のチャネル数
      layers=30, #レイヤー数
      stacks=3, #畳み込みブロック数
      residual_channels=64, #残差接続のチャネル数
      gate_channels=128, #ゲートのチャネル数
      skip_out_channels=64, #スキップ接続のチャネル数
      kernel_size=2, #1次元畳み込みのカーネルサイズ
      cin_channels=80, #条件付け特徴量のチャネル数
      upsample_scales=None, #アップサンプリングスケール
      aux_context_window=0, #アップサンプリング時に参照する近傍フレーム数
   ):
      super().__init__()
      self.out_channels=out_channels
      self.cin_channels=cin.channels
      self.aux_context_window=aux_context_window
      if upsample_scales is None:
         upsample_scales=[10, 8]
      self.upsample_scales=upsample_scales
      
      self.first_conv=Conv1d1x1(out_channels, residual_channels)
      
      #メインとなる畳み込み層
      self.main_conv_layers=nn.ModuleList()
      layers_per_stack=layers//stacks
      for layer in range(layers):
         dilation=2**(layer%layers_per_stack)
         conv=ResSkipBlock(residual_channels, gate_channels, kernel_size, skip_out_channels, dilation=dilation, cin_channels=cin_channels)
         self.main_conv_layers.append(conv)
         
      #スキップ接続の和から波形への変換
      self.last_conv_layers=nn.ModuleList(
         [
            nn.ReLU(), 
            Conv1d1x1(skip_out_channels, skip_out_channels),
            nn.ReLU(),
            Conv1d1x1(skip_out_channels, out_channels),
         ]
      )
      
      #フレーム単位の特徴量をサンプル単位にアップサンプリング
      self.upsample_net=ConvInUpsampleNetwork(
         upsample_scales, cin_channels, aux_context_window
      )
      
#%%
#code7.12
#WaveNetクラスのメソッド
def forward(self, x, c):
   #量子化された離散値列からone-hotベクトルに変換
   #(B, T) -> (B, T, out_channels) -> (B, out_channels, T)
   x=F.one_hot(x, num_classes=self.out_channels).transpose(1, 2).float()
   
   #条件つけ特徴量のアップサンプリング
   c=self.upsample_net(c)
   
   #One-hotベクトルの次元（out_channels）から隠れ層の次元（residual_channels）に変換
   x=self.first_conv(x)
   #メインの畳み込み層の処理
   #各層におけるスキップ接続の出力を加算して保持
   skip=0
   for f in self.main_conv_layers:
      x, s=f(x, c)
      skip+=s
      
   #スキップ接続の和を入力として，出力を計算
   x=skips
   for f in self.last_conv_layers:
      x=f(x)
      
   #NOTE:出力を確立値として解釈する場合にはsoftmaxが必要だが，学習時にはnn.CrossEntropyLossの計算においてs
# %%
#code7.13
#WaveNetクラスのメソッド
def inference(self, c, num_time_steps=100, tqdm=lambda x: x):
   B=c.shape[0]
   
   #local conditioning
   #(B, C, T)
   c=self.upsample_net(c)
   #(B, C, T) -> (B, T, C)
   c=c.tranpose(1, 2).contiguoius()
   
   outputs=[]
   
   #自己回帰生成における初期値
   current_input=torch.zeros(B, 1, self.out_channels).to(c.device)
   current_input[:, :, int(mulaw_quantize(0))]=1
   
   if tqdm is None:
      ts =range(num_time_steps)
   else:
      ts=tqdm(range(num_time_steps))
      
   #逐次的に生成
   for t in ts:
      #時刻tにおける入力は，時刻t-1における出力
      if t>0:
         current_input=outputs[-1]
         
      #時刻tにおける条件付け特徴量
      ct=c[:, t, :].unsqueeze(1)
      
      x=current_input
      
      x=self.first_conv.incremental_forward(x)
      skips=0
      for f in self.main_conv_layers:
         x, h=f.incremental_forward(x, ct)
         skips += h
      x=skips
      for f in self.last_conv_layers:
         if hasattr(f, "incremental_forward"):
            x=f.incremental_forward(x)
         else:
            x=f(x)
      #Softmaxにより，出力をカテゴリカル分布のパラメータに変換
      x=F.softmax(x.view(B, -1), dim=1)
      #カテゴリカル分布からサンプリング
      x=torch.distributions.OneHotCategorical(x).sample()
      outputs+=[x.data]
      
   #T × B × Cの形状を持つテンソルを返す．
   outputs=torch.stack(outputs)
   #B × C × Tの形状に変換
   outputs=outputs.transpose(0, 1).tranpose(1, 32).contiguous()
   
   return outputs
#%%
