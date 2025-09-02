from utils.generation import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
from IPython.display import Audio
from utils.prompt_making import make_prompt
import torch


#make_prompt(name="nycuka2", audio_prompt_path="prompts/nycuka2.wav")

# def get_model_file_size(path):
#     model_state = torch.load(path, map_location='cpu' ,weights_only=False)

#     if 'model' in model_state:
#         model_state = model_state['model']  # icefall 格式通常是這樣

#     total_params = sum(p.numel() for p in model_state.values())
#     total_bytes = sum(p.numel() * p.element_size() for p in model_state.values())
#     print(f"🔢 Total parameters: {total_params:,}")
#     print(f"📦 Total size: {total_bytes / (1024 ** 2):.2f} MB")

# # 範例
# get_model_file_size("/home/VALL-E-X-Trainer-by-CustomData/exp/valle/best-train-loss.pt")

## EN prompt:en2zh_tts_4

# zh_prompt =  """
# 豆腐為什麼能打傷人
# """

# text_prompt = """
#     [ZH]這個[ZH] 
#     [EN]restaurant[EN] 
#     [ZH]很有名[ZH]
#     [ZH]很多人都來吃[ZH]
#     """

###############################################


#######################################################
# #china
# preload_models("/home/VALL-E-X-Trainer-by-CustomData/exp/valle/epoch-500000.pt")

# # generate audio from text

# audio_array = generate_audio(text_prompt,prompt="zh2en_tts_2",language="mix")

# # save audio to disk
# write_wav("vallex_premix.wav", SAMPLE_RATE, audio_array)

# # play text in notebook|
# Audio(audio_array, rate=SAMPLE_RATE)

#######################################################################################



mix_prompt = """
    [ZH]這個[ZH]
    [EN]language model[EN]
    [ZH]很好用功能很多[ZH]
    """

one_prompt =  """   
He honours whatever he recognizes in himself, such morality equals self glorification.
"""

#######################################################################################
####with用js，finetune用newtai 

# taiwan_validloss
preload_models("exp/valle/epoch-500150.pt")

# generate audio from text en2zh_tts_3

########################with用js，finetune用newtai ###########################

###zh or en
audio_array = generate_audio(one_prompt,prompt="en2zh_tts_3")

##mix
#audio_array = generate_audio(mix_prompt,ˊ="newtai",language="mix")



# save audio to disk
write_wav("MoEnycuka.wav", SAMPLE_RATE,audio_array)

# play text in notebook
Audio(audio_array, rate=SAMPLE_RATE)

##############################################################################################################



#######################################################################################
# # Taiwan_ephco
# preload_models("/home/VALL-E-X-Trainer-by-CustomData/exp/valle/epoch-500275.pt")

# # generate audio from text



# audio_array = generate_audio(text_prompt,prompt="zh2en_tts_2",language="mix")

# # save audio to disk
# write_wav("vallex_tai_epoch_mix.wav", SAMPLE_RATE, audio_array)

# # play text in notebook
# Audio(audio_array, rate=SAMPLE_RATE)

##############################################################################################################




# import h5py
# import os
# import numpy as np




# import soundfile as sf

# h5_path = "/home/VALL-E-X-Trainer-by-CustomData/taiwna_data/audio_sum.hdf5"
# ann_path = "/home/VALL-E-X-Trainer-by-CustomData/taiwna_data/audio_ann_sum.txt"
# audio_folder = "/home/VALL-E-X-Trainer-by-CustomData/taiwna_data/"  # 你的音檔資料夾

# # 讀取標註中的音訊時長
# ann_durations = {}
# with open(ann_path, "r", encoding="utf-8") as f:
#     for line in f:
#         parts = line.strip().split("|")
#         ann_durations[parts[0]] = float(parts[1])

# # 檢查 HDF5 內的音訊長度
# with h5py.File(h5_path, "r") as f:
#     for key in f.keys():
#         try:
#             # 解析 uttX_Y → uttX 資料夾 & Y 檔名
#             utt_folder, file_id = key.split("_")  # e.g., utt1_2 → utt1, 2
#             audio_path = os.path.join(audio_folder, utt_folder, file_id + ".wav")
            
#             if os.path.exists(audio_path):
#                 # 讀取原始音檔
#                 audio, sr = sf.read(audio_path)
#                 duration = len(audio) / sr

#                 # 比對標註時長
#                 if abs(duration - ann_durations[key]) > 0.2:  # 容許 0.2 秒誤差
#                     print(f"⚠️ {key} 時長不匹配: 標註 {ann_durations[key]}s，實際 {duration}s")
#                 else:
#                     print(f"✅ {key} 時長匹配")
#             else:
#                 print(f"❌ 找不到對應音檔: {audio_path}")

#         except Exception as e:
#             print(f"❌ {key} 發生錯誤: {e}")