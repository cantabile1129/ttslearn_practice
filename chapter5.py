#%%
#code5.1&5.2
from nnmnkwii.io import hts
import ttslearn
labels = hts.load(ttslearn.util.example_label_file(mono=False)) 
for start_time, end_time, context in labels[:6]:
   print(f"{start_time} {end_time} {context}") 

"""
^...前の音素
-...現在の音素
+...次の音素
=...次の次の音素

1. A:-2+1+3
音素レベルの位置情報
-2: 現在の音素の 前のモーラ位置
+1: 現在の音素の モーラ内での位置
+3: 次の音素のモーラ内での位置

2. B:xx-xx_xx
音素の前後関係
xx-xx_xx: 現在の音素の 前後の音素カテゴリ

3. C:02_xx+xx
音素がどの 音節 (syllable) に属するか
02: 現在の音素が属する音節の位置
xx+xx: 次の音節との関係

4. D:13+xx_xx
アクセント境界の情報
13: 現在の音素のアクセント位置
xx_xx: 次のアクセントの情報

5. E:xx_xx!xx_xx-xx
ピッチパターン（F0 の変化）
xx_xx: 現在の音素の音調情報
!xx_xx: アクセントの変化位置
-xx: 次の音素の音調情報

6. F:3_3#0_xx@1_5|1_23
モーラ単位の情報
3_3: 現在のモーラの音節位置
#0_xx: 語のモーラ数
@1_5: モーラ内の音素位置
|1_23: 次のモーラとの関係

7. G:7_2%0_xx_1
単語（word）の情報
7_2: 単語の音節数
%0_xx_1: 単語のイントネーション情報

8. H:xx_xx
文レベルの情報
xx_xx: 文の中の音節情報

9. I:5-23@1+1&1-5|1+23
フレーズ単位の情報
5-23: 現在のフレーズ内の位置
@1+1: 前後のフレーズとの関係
&1-5: フレーズのアクセント情報
|1+23: 次のフレーズの情報

10. J:xx_xx
プロソディ情報
xx_xx: 文のイントネーションパターン

11. K:1+5-23
文全体のリズム・ピッチ情報
1+5-23: 文章レベルの韻律情報
"""   
# %%
#code5.3
import pyopenjtalk
pyopenjtalk.g2p("今日もいい天気ですね", kana=True)
pyopenjtalk.g2p("今日もいい天気ですね", kana=False)

labels = pyopenjtalk.extract_fullcontext("今日")
for label in labels:
   print(label)
#%%
#code5.6
from nnmnkwii.io import hts
import ttslearn

binary_dict, numeric_dict = hts.load_question_set(ttslearn.util.example_qst_file())

print("二値特徴量の数:", len(binary_dict))
print("数値特徴量の数:", len(numeric_dict))
print("1つ目の質問:", binary_dict[0], binary_dict[1])
# %%
#code5.7
from nnmnkwii.frontend import merlin as fe

labels = hts.load(ttslearn.util.example_label_file())
feats = fe.linguistic_features(labels, binary_dict, numeric_dict)
print("言語特徴量のサイズ:", feats.shape)
# %%
#code5.8
feats_phoneme = fe.linguistic_features(labels, binary_dict, numeric_dict, add_frame_features=False)
feats_frame = fe.linguistic_features(labels, binary_dict, numeric_dict, add_frame_features=True)
print("言語特徴量のサイズ:", feats_phoneme.shape)
print("言語特徴量のサイズ:", feats_frame.shape)
# %%
#code5.9