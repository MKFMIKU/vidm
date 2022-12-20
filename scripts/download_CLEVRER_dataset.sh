#!/bin/bash
mkdir data/CLEVRER
# curl http://data.csail.mit.edu/clevrer/videos/train/video_train.zip -o data/CLEVRER/video_train.zip
curl http://data.csail.mit.edu/clevrer/videos/validation/video_validation.zip -o data/CLEVRER/video_validation.zip

# unzip data/CLEVRER/video_train.zip -d ./data/CLEVRER/
unzip data/CLEVRER/video_validation.zip -d ./data/CLEVRER/