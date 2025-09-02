import os
import torch
import logging
from macros import *
from data.tokenizer import (
    AudioTokenizer,
    tokenize_audio,
)
from models.vallex import VALLE
from vocos import Vocos
from pathlib import Path  # 修改這裡
import platform
import pathlib

plt = platform.system()
print("Operating System:", plt)

if plt == 'Linux':
    pathlib.WindowsPath = pathlib.PosixPath

def get_model(device):
    url = 'https://huggingface.co/Plachta/VALL-E-X/resolve/main/vallex-checkpoint.pt'

    checkpoints_dir = "./checkpoints"

    model_checkpoint_name = "vallex-checkpoint.pt"

    if not os.path.exists(checkpoints_dir): os.mkdir(checkpoints_dir)
    if not os.path.exists(os.path.join(checkpoints_dir, model_checkpoint_name)):
        import wget
        print("3")
        try:
            logging.info(
                "Downloading model from https://huggingface.co/Plachta/VALL-E-X/resolve/main/vallex-checkpoint.pt ...")
            wget.download(url, out=os.path.join(checkpoints_dir, model_checkpoint_name), bar=wget.bar_adaptive)
        except Exception as e:
            logging.info(e)
            raise Exception(
                "\n Model weights download failed, please go to 'https://huggingface.co/Plachta/VALL-E-X/resolve/main/vallex-checkpoint.pt'"
                "\n manually download model weights and put it to {} .".format(os.getcwd() + "\checkpoints"))
    # VALL-E
    model = VALLE(
        N_DIM,
        NUM_HEAD,
        NUM_LAYERS,
        norm_first=True,
        add_prenet=False,
        prefix_mode=PREFIX_MODE,
        share_embedding=True,
        nar_scale_factor=1.0,
        prepend_bos=True,
        num_quantizers=NUM_QUANTIZERS,
    ).to(device)


    ###pretrain#######
    # checkpoint_path = Path(checkpoints_dir) / model_checkpoint_name  # 修改這裡
    # checkpoint = torch.load(checkpoint_path, map_location='cpu')
    # missing_keys, unexpected_keys = model.load_state_dict(
    #     checkpoint["model"], strict=True
    # )
    # assert not missing_keys
    ###pretrain#######

    
    # Encodec
    codec = AudioTokenizer(device)
    
    vocos = Vocos.from_pretrained('charactr/vocos-encodec-24khz').to(device)
    
    return model, codec, vocos
