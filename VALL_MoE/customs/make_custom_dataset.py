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

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}
from data.tokenizer import (
    AudioTokenizer,
    tokenize_audio,
)

tokenizer_path = "./utils/g2p/bpe_69.json"
tokenizer = PhonemeBpeTokenizer(tokenizer_path)
device = 'cuda' if torch.cuda.is_available() else 'cpu'





def create_my_dataset():
    wav_dir = "/home/VALL-E-X-Trainer-by-CustomData/zh_data/clips_wav"
    tsv_path = "/home/VALL-E-X-Trainer-by-CustomData/zh_data/train.tsv"
    output_dir = "/home/VALL-E-X-Trainer-by-CustomData/zh_data/clips_wav"
    dur_df = pd.read_csv("/home/VALL-E-X-Trainer-by-CustomData/zh_data/clip_durations.tsv", sep="\t")
    dur_lookup = {
        os.path.splitext(row["clip"])[0]: float(row["duration[ms]"]) / 1000.0 for _, row in dur_df.iterrows()
    }

    os.makedirs(output_dir, exist_ok=True)
    tokenizer_path = "./utils/g2p/bpe_1024.json"
    text_tokenizer = PhonemeBpeTokenizer(tokenizer_path)
    text_collater = get_text_token_collater()
    codec = AudioTokenizer('cuda' if torch.cuda.is_available() else 'cpu')

    df = pd.read_csv(tsv_path, sep='\t').dropna(subset=["path", "sentence"])

    h5_output_path = os.path.join(output_dir, 'audio_sum.hdf5')
    ann_output_path = os.path.join(output_dir, 'audio_ann_sum.txt')
    skip_output_path = os.path.join(output_dir, 'skipped.txt')

    with h5py.File(h5_output_path, 'w') as h5_file:
        for idx, row in df.iterrows():
            try:
                mp3_name = row['path']
                wav_name = mp3_name.replace(".mp3", ".wav")
                wav_path = os.path.join(wav_dir, wav_name)
                stem = os.path.splitext(os.path.basename(wav_name))[0]
                sentence = row['sentence'].strip()
                language = 'zh'

                if not os.path.exists(wav_path):
                    print(f"❌ 找不到音訊檔案：{wav_path}")
                    continue

                if not sentence:
                    print(f"⚠️ 空白句子，跳過：{stem}")
                    continue

                ipa_string = chinese_to_lazy_ipa(sentence)
                phonemes = ipa_string.split(" ")
                tokenized = text_tokenizer.tokenizer.encode(" ".join(phonemes))
                if not tokenized.ids:
                    raise ValueError("Empty tokenized ids")

                # text_tokens, _ = text_collater([tokenized.ids])
                text_collater = get_text_token_collater()
                text_tokens, enroll_x_lens = text_collater([tokenized.ids])
                text_tokens = text_tokens.squeeze(0)

                wav_pr, sr = torchaudio.load(wav_path)
                if wav_pr.size(0) == 2:
                    wav_pr = wav_pr.mean(0, keepdim=True)

                encoded_frames = tokenize_audio(codec, (wav_pr, sr))
                audio_tokens = encoded_frames[0][0].transpose(2, 1).cpu().numpy()

                grp = h5_file.create_group(stem)
                grp.create_dataset("audio", data=audio_tokens)

                duration = dur_lookup.get(stem, 0.0)
                with open(ann_output_path, 'a', encoding='utf-8') as ann_file:
                    ann_file.write(f'{stem}|{duration:.2f}|{language}|{sentence}\n')

                print(f"✅ 轉換成功: {stem}")

            except Exception as e:
                print(f"❌ 錯誤於 {row['path']}: {e}")
                with open(skip_output_path, "a", encoding="utf-8") as skip_file:
                    skip_file.write(f"{stem}|general error: {e}\n")








def make_prompts(name, audio_prompt_path, transcript=None):
    text_tokenizer = PhonemeBpeTokenizer(tokenizer_path="./utils/g2p/bpe_1024.json")
    text_collater = get_text_token_collater()
    codec = AudioTokenizer(device)
    wav_pr, sr = torchaudio.load(audio_prompt_path)
    # check length
    if wav_pr.size(-1) / sr > 15:
        raise ValueError(f"Prompt too long, expect length below 15 seconds, got {wav_pr / sr} seconds.")
    if wav_pr.size(0) == 2:
        wav_pr = wav_pr.mean(0, keepdim=True)
    text_pr, lang_pr = make_transcript(name, wav_pr, sr, transcript)
  

    # tokenize audio
    encoded_frames = tokenize_audio(codec, (wav_pr, sr))
    audio_tokens = encoded_frames[0][0].transpose(2, 1).cpu().numpy()

    # tokenize text
    phonemes, langs = text_tokenizer.tokenize(text=f"{text_pr}".strip())
    text_tokens, enroll_x_lens = text_collater(
        [
            phonemes
        ]
    )

    return audio_tokens, text_tokens, langs, text_pr







def create_dataset(data_dir, dataloader_process_only):
    if dataloader_process_only:
        h5_output_path=f"{data_dir}/audio_sum.hdf5"
        ann_output_path=f"{data_dir}/audio_ann_sum.txt"
        #audio_folder = os.path.join(data_dir, 'audio')
        #audio_paths = glob.glob(f"{data_dir}/*.wav")  # Change this to match your audio file extension
        audio_paths = glob.glob(f"{data_dir}/**/*.wav", recursive=True)
        print(f"Found {len(audio_paths)} audio files.")
        # Create or open an HDF5 file
        with h5py.File(h5_output_path, 'w') as h5_file:
            # Loop through each audio and text file, assuming they have the same stem
            for audio_path in audio_paths:
                try:
                    parent_folder = os.path.basename(os.path.dirname(audio_path))  # 获取文件夹名称
                    stem = f"{parent_folder}_{os.path.splitext(os.path.basename(audio_path))[0]}"  # 例如 utt1_1, utt1_2
                    #stem = os.path.splitext(os.path.basename(audio_path))[0]
                    audio_tokens, text_tokens, langs, text = make_prompts(name=stem, audio_prompt_path=audio_path)
                    
                    text_tokens = text_tokens.squeeze(0)
                    # Create a group for each stem
                    grp = h5_file.create_group(stem)
                    # Add audio and text tokens as datasets to the group
                    grp.create_dataset('audio', data=audio_tokens)
                    #grp.create_dataset('text', data=text_tokens)
                    
                    with open(ann_output_path, 'a', encoding='utf-8') as ann_file:
                        audio, sample_rate = sf.read(audio_path)
                        duration = len(audio) / sample_rate
                        #ann_file.write(f'{stem}|{duration}|{langs[0]}|{text}\n')  # 改行を追加
                        ann_file.write(f'{stem}|{duration}|{langs[0]}|{text.strip()}\n')
                except Exception as e:
                    print(f"An error occurred: {e}")
    else:
        dataloader = create_dataloader(data_dir=data_dir, max_size=10, max_duration=30)
        return dataloader





























# def create_dataset():
#     wav_dir="/home/VALL-E-X-Trainer-by-CustomData/zh_data/clips_wav"
#     tsv_path="/home/VALL-E-X-Trainer-by-CustomData/zh_data/train.tsv"
#     output_dir="/home/VALL-E-X-Trainer-by-CustomData/zh_data"
#     dur_df = pd.read_csv("/home/VALL-E-X-Trainer-by-CustomData/zh_data/clip_durations.tsv", sep="\t")
#     dur_lookup = {
#         os.path.splitext(row["clip"])[0]: float(row["duration[ms]"]/ 1000.0) 
#         for _, row in dur_df.iterrows()
#     }
#     os.makedirs(output_dir, exist_ok=True)
#     tokenizer_path = "./utils/g2p/bpe_69.json"
#     text_tokenizer = PhonemeBpeTokenizer(tokenizer_path)
#     text_collater = get_text_token_collater()
#     codec = AudioTokenizer('cuda' if torch.cuda.is_available() else 'cpu')

#     df = pd.read_csv(tsv_path, sep='\t')
#     df = df.dropna(subset=["path", "sentence"])

   
#     h5_output_path = os.path.join(output_dir, 'audio_sum.hdf5')
#     ann_output_path = os.path.join(output_dir, 'audio_ann_sum.txt')
#     skip_output_path = os.path.join(output_dir, 'skipped.txt')

#     with h5py.File(h5_output_path, 'w') as h5_file:
#         for idx, row in df.iterrows():
#             try:
#                 mp3_name = row['path']
#                 wav_name = mp3_name.replace(".mp3", ".wav")
#                 wav_path = os.path.join(wav_dir, wav_name)
#                 stem = os.path.splitext(os.path.basename(wav_name))[0]
#                 sentence = row['sentence'].strip()
#                 language = 'zh'  # 固定指定為 zh

#                 # === 檢查音檔是否存在 ===
#                 if not os.path.exists(wav_path):
#                     print(f"❌ 找不到音訊檔案：{wav_path}")
#                     continue
                
#                 if not sentence:
#                     print(f"⚠️ 空白句子，跳過：{stem}")
#                     continue

#                 # === tokenize sentence ===
#                 phonemes, langs = text_tokenizer.tokenize(text=sentence)
#                 if not phonemes:
#                     print(f"⚠️ tokenizer 回傳空 phoneme：{stem}，sentence: {sentence}")
#                     with open(skip_output_path, "a", encoding="utf-8") as skip_file:
#                         skip_file.write(f"{stem}|{sentence}\n")
#                     continue

#                 text_tokens, enroll_x_lens = text_collater([phonemes])
#                 text_tokens = text_tokens.squeeze(0)



#                 # === 讀取 wav、處理為 mono ===
#                 wav_pr, sr = torchaudio.load(wav_path)
#                 if wav_pr.size(0) == 2:
#                     wav_pr = wav_pr.mean(0, keepdim=True)

#                 # === 語音 token 化 ===
#                 encoded_frames = tokenize_audio(codec, (wav_pr, sr))
#                 audio_tokens = encoded_frames[0][0].transpose(2, 1).cpu().numpy()

#                 # === 文本 token 化（BPE + phoneme）===
#                 phonemes, langs = text_tokenizer.tokenize(text=sentence.strip())
#                 text_tokens, enroll_x_lens = text_collater([phonemes])
#                 text_tokens = text_tokens.squeeze(0)

#                 # === 寫入 hdf5 ===
#                 grp = h5_file.create_group(stem)
#                 grp.create_dataset("audio", data=audio_tokens)
#                 # grp.create_dataset("text", data=text_tokens)  # optional

#                 # === 寫入標註 txt ===
#                 duration = dur_lookup.get(stem, 0.0)
#                 with open(ann_output_path, 'a', encoding='utf-8') as ann_file:
#                     ann_file.write(f'{stem}|{duration:.2f}|{language}|{sentence}\n')

#                 print(f"✅ 轉換成功: {stem}")

#             except Exception as e:
#                 print(f"❌ 錯誤於 {row['path']}: {e}")

























# def create_dataset():
#     wav_dir = "/home/VALL-E-X-Trainer-by-CustomData/zh_data/clips_wav"
#     tsv_path = "/home/VALL-E-X-Trainer-by-CustomData/zh_data/train.tsv"
#     output_dir = "/home/VALL-E-X-Trainer-by-CustomData/zh_data"
#     dur_df = pd.read_csv("/home/VALL-E-X-Trainer-by-CustomData/zh_data/clip_durations.tsv", sep="\t")
#     dur_lookup = {
#         os.path.splitext(row["clip"])[0]: float(row["duration[ms]"]) / 1000.0
#         for _, row in dur_df.iterrows()
#     }

#     os.makedirs(output_dir, exist_ok=True)
#     tokenizer_path = "./utils/g2p/bpe_1024.json"
#     text_tokenizer = PhonemeBpeTokenizer(tokenizer_path)
#     text_collater = get_text_token_collater()
#     codec = AudioTokenizer('cuda' if torch.cuda.is_available() else 'cpu')

#     df = pd.read_csv(tsv_path, sep='\t')
#     df = df.dropna(subset=["path", "sentence"])

#     h5_output_path = os.path.join(output_dir, 'audio_sum.hdf5')
#     ann_output_path = os.path.join(output_dir, 'audio_ann_sum.txt')
#     skip_output_path = os.path.join(output_dir, 'skipped.txt')

#     with h5py.File(h5_output_path, 'w') as h5_file:
#         for idx, row in df.iterrows():
#             try:
#                 mp3_name = row['path']
#                 wav_name = mp3_name.replace(".mp3", ".wav")
#                 wav_path = os.path.join(wav_dir, wav_name)
#                 stem = os.path.splitext(os.path.basename(wav_name))[0]
#                 sentence = row['sentence'].strip()
#                 language = 'zh'

#                 if not os.path.exists(wav_path):
#                     print(f"❌ 找不到音訊檔案：{wav_path}")
#                     continue

#                 if not sentence:
#                     print(f"⚠️ 空白句子，跳過：{stem}")
#                     continue

#                 # === tokenizer 主流程 + fallback ===
#                 try:
#                     phonemes, langs = text_tokenizer.tokenize(text=sentence)
                    
#                     if not phonemes:
#                         raise ValueError("Empty phoneme")
#                 except Exception as e:
#                     print(f"⚠️ tokenizer 回傳空 → 使用 fallback G2P：{stem}")
#                     print(chinese_to_lazy_ipa("於是他決定要做生意"))
#                     with open(skip_output_path, "a", encoding="utf-8") as skip_file:
#                         skip_file.write(f"{stem}|fallback used|{sentence}\n")


#                     # fallback 改用 mandarin lazy ipa
#                     fallback_ipa = chinese_to_lazy_ipa(sentence)
#                     phonemes, langs = text_tokenizer.tokenize(text=fallback_ipa)


#                 text_tokens, enroll_x_lens = text_collater([phonemes])
#                 text_tokens = text_tokens.squeeze(0)

#                 wav_pr, sr = torchaudio.load(wav_path)
#                 if wav_pr.size(0) == 2:
#                     wav_pr = wav_pr.mean(0, keepdim=True)

#                 encoded_frames = tokenize_audio(codec, (wav_pr, sr))
#                 audio_tokens = encoded_frames[0][0].transpose(2, 1).cpu().numpy()

#                 grp = h5_file.create_group(stem)
#                 grp.create_dataset("audio", data=audio_tokens)

#                 duration = dur_lookup.get(stem, 0.0)
#                 with open(ann_output_path, 'a', encoding='utf-8') as ann_file:
#                     ann_file.write(f'{stem}|{duration:.2f}|{language}|{sentence}\n')

#                 print(f"✅ 轉換成功: {stem}")

#             except Exception as e:
#                 print(f"❌ 錯誤於 {row['path']}: {e}")
#                 with open(skip_output_path, "a", encoding="utf-8") as skip_file:
#                     skip_file.write(f"{stem}|general error: {e}\n")














# def create_dataset(data_dir, dataloader_process_only):
#     if dataloader_process_only:
#         h5_output_path=f"{data_dir}/audio_sum.hdf5"
#         ann_output_path=f"{data_dir}/audio_ann_sum.txt"
#         #audio_folder = os.path.join(data_dir, 'audio')
#         #audio_paths = glob.glob(f"{data_dir}/*.wav")  # Change this to match your audio file extension
#         audio_paths = glob.glob(f"{data_dir}/**/*.wav", recursive=True)
#         print(f"Found {len(audio_paths)} audio files.")
#         # Create or open an HDF5 file
#         with h5py.File(h5_output_path, 'w') as h5_file:
#             # Loop through each audio and text file, assuming they have the same stem
#             for audio_path in audio_paths:
#                 try:
#                     parent_folder = os.path.basename(os.path.dirname(audio_path))  # 获取文件夹名称
#                     stem = f"{parent_folder}_{os.path.splitext(os.path.basename(audio_path))[0]}"  # 例如 utt1_1, utt1_2
#                     #stem = os.path.splitext(os.path.basename(audio_path))[0]
#                     audio_tokens, text_tokens, langs, text = make_prompts(name=stem, audio_prompt_path=audio_path)
                    
#                     text_tokens = text_tokens.squeeze(0)
#                     # Create a group for each stem
#                     grp = h5_file.create_group(stem)
#                     # Add audio and text tokens as datasets to the group
#                     grp.create_dataset('audio', data=audio_tokens)
#                     #grp.create_dataset('text', data=text_tokens)
                    
#                     with open(ann_output_path, 'a', encoding='utf-8') as ann_file:
#                         audio, sample_rate = sf.read(audio_path)
#                         duration = len(audio) / sample_rate
#                         #ann_file.write(f'{stem}|{duration}|{langs[0]}|{text}\n')  # 改行を追加
#                         ann_file.write(f'{stem}|{duration}|{langs[0]}|{text.strip()}\n')
#                 except Exception as e:
#                     print(f"An error occurred: {e}")
#     else:
#         dataloader = create_dataloader(data_dir=data_dir, max_size=10, max_duration=30)
#         return dataloader