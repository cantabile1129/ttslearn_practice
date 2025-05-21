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

