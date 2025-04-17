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
#code6.2
print("hello, world")
# %%
