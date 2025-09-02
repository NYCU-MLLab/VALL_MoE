from customs.make_custom_dataset import create_dataset,create_my_dataset
import h5py
import glob
import torch
import numpy as np
import os
import torchaudio
import soundfile as sf
from utils.g2p.symbols import symbols
from utils.g2p import PhonemeBpeTokenizer
from utils.prompt_making import make_prompt, make_transcript
from data.collation import get_text_token_collater
from data.dataset import create_dataloader
import pandas as pd
from data.tokenizer import AudioTokenizer, tokenize_audio
from utils.g2p.mandarin import chinese_to_lazy_ipa

from data.tokenizer import TextTokenizer, tokenize_text
from tokenizers import Tokenizer
from tqdm import tqdm

tsv_path="/home/VALL-E-X-Trainer-by-CustomData/zh_data/train.tsv"
wav_dir="split"




create_dataset(wav_dir,dataloader_process_only=True)


# hdf5_path = "/home/VALL-E-X-Trainer-by-CustomData/zh_data/audio_sum.hdf5"
# ann_path = "/home/VALL-E-X-Trainer-by-CustomData/zh_data/audio_ann_sum.txt"



def check_dataset(h5_path, ann_path, tokenizer_path="./utils/g2p/bpe_1024.json"):
    # 載入 tokenizer 和 collater
    tokenizer = PhonemeBpeTokenizer(tokenizer_path)
    text_collater = get_text_token_collater()

    # 載入註解與 HDF5 資料
    with h5py.File(h5_path, 'r') as h5_file, open(ann_path, 'r', encoding='utf-8') as ann_file:
        lines = [line.strip() for line in ann_file if line.strip()]
        print(f" 正在檢查 {len(lines)} 筆資料...\n")

        for line in tqdm(lines):
            try:
                stem, dur, lang, text = line.split("|")
                audio_tokens = h5_file[stem]["audio"][()]

                # 檢查 NaN
                if np.isnan(audio_tokens).any():
                    print(f" NaN in audio_tokens: {stem}")
                    continue

                # 處理 text → IPA → BPE tokens
                ipa_string = chinese_to_lazy_ipa(text)
                phonemes = ipa_string.split(" ")
                tokenized = tokenizer.tokenizer.encode(" ".join(phonemes))

                if not tokenized.ids:
                    print(f"❌ 無法產出 BPE token: {stem} | 原句: {text}")
                    continue

                # 嘗試通過 collater 組成 batch
                text_tokens, enroll_x_lens = text_collater([tokenized.ids])
                if text_tokens.shape[1] == 0:
                    print(f"⚠️ Empty text token after collate: {stem}")
            except Exception as e:
                print(f"⚠️ 錯誤樣本 {line}: {e}")


# check_dataset(
#     h5_path="/home/VALL-E-X-Trainer-by-CustomData/zh_data/audio_sum.hdf5",
#     ann_path="/home/VALL-E-X-Trainer-by-CustomData/zh_data/audio_ann_sum.txt",
#     tokenizer_path="/home/VALL-E-X-Trainer-by-CustomData/utils/g2p/bpe_1024.json"
# )






# # 初始化 BPE Tokenizer
# tokenizer = PhonemeBpeTokenizer(tokenizer_path="./utils/g2p/bpe_1024.json")

# # IPA 分詞（Lazy IPA）
# ipa_string = chinese_to_lazy_ipa("於是他決定要做生意")
# phonemes = ipa_string.split(" ")

# # 使用 tokenizer.tokenizer 來 encode IPA
# tokenized = tokenizer.tokenizer.encode(" ".join(phonemes))

# # 給 collater 做 batch 整理

# text_collater = get_text_token_collater()
# text_tokens, enroll_x_lens = text_collater([tokenized.ids])

# print("IPA:", ipa_string)
# print("Tokenized IDs:", tokenized.ids)
# print("Tokens:", tokenizer.tokenizer.encode(" ".join(phonemes)).tokens)

# print("text_tokens shape:", text_tokens.shape)
# print("text_tokens dtype:", text_tokens.dtype)
# print("enroll_x_lens:", enroll_x_lens)
# print("Decoded:", tokenizer.tokenizer.decode(tokenized.ids))




# dur_df = pd.read_csv("/home/VALL-E-X-Trainer-by-CustomData/zh_data/clip_durations.tsv", sep="\t")
# tsv_df = pd.read_csv(tsv_path, sep='\t')
# row = tsv_df[tsv_df['path'] == 'common_voice_zh-TW_19352152.mp3']

# print(row)





#data_dir = "/home/VALL-E-X-Trainer-by-CustomData/zh_data/clips_wav"



