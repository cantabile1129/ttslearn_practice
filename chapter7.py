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