import cv2
import os
import numpy as np
from tqdm import tqdm

path_list = ['data/CLEVRER/train_vid/', 'data/CLEVRER/val_vid/']
save_path_list = ['data/CLEVRER/train', 'data/CLEVRER/val' ]

for i in range(2)[1:]:
    path = path_list[i]
    save_path = save_path_list[i]

    dir_list = os.listdir(path)
    mp4s = [d for d in dir_list if '.mp4' in d]

    if not os.path.exists(f'{save_path}'):
        os.mkdir(f'{save_path}')

    for mp4 in tqdm(mp4s):
        if not os.path.exists(f'{save_path}/{mp4[:-4]}'):
            os.mkdir(f'{save_path}/{mp4[:-4]}')

        vidcap = cv2.VideoCapture(f'{path}/{mp4}')
        success, image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite(f"{save_path}/{mp4[:-4]}/frame{str(count).zfill(5)}.png", image)
            success, image = vidcap.read()
            count +=1
