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

