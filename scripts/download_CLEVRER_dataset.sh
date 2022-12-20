#!/bin/bash
mkdir data/CLEVRER
curl http://data.csail.mit.edu/clevrer/videos/train/video_train.zip -o data/CLEVRER/video_train.zip
curl http://data.csail.mit.edu/clevrer/videos/validation/video_validation.zip -o data/CLEVRER/video_validation.zip

unzip data/CLEVRER/video_train.zip -d ./data/CLEVRER/
unzip data/CLEVRER/video_validation.zip -d ./data/CLEVRER/

mkdir data/CLEVRER/train_vid/
mkdir data/CLEVRER/val_vid/

mv data/CLEVRER/video_0*/*.mp4 data/CLEVRER/train_vid/
mv data/CLEVRER/video_1*/*.mp4 data/CLEVRER/val_vid/


mkdir data/CLEVRER/train
mkdir data/CLEVRER/val

# extract frames from videos
python scripts/preprocess_CLEVRER_dataset.py

# clear
rm -rf data/CLEVRER/video_0*/
rm -rf data/CLEVRER/video_1*/