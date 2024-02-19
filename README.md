# Automatic Chord Recognition Codes (for External)

This repo contains implementation of 2 models for Automatic Chord Recognition(ACR) task:
**Bi-Directional Transformer** and **CRNN**(Convolutional Recurrent Neural Network).

These models are used as baselines in the following paper we have written:
Hikaru Yamaga, Toshifumi Momma, Kazunori Kojima, Yoshiaki Itoh, "Ensemble of Transformer and Convolutional Recurrent Neural Network for Improving Discrimination Accuracy in Automatic Chord Recognition.", APSIPA ASC, pp.2299-2305, 2023

## Overview
- Genre: Sequence Labeling Task
- Input: Audio File(.wav)
- Feature: CQT-Spectrogram(2-dimension, TimeAxis:108 * FreqAxis:192)
- Output: Sequence of Predicted Chords

## Requirements
- Python (>= 3.8.10)
- pytorch
- numpy
- pandas
- librosa
- sklearn
- mir_eval
- matplotlib

## Usage
### Train
1. Store audio files(.wav format) in "./02_audiofiles/"
2. Store annotation files(.lab format) in "./02_chordfiles/"
3. Execute `python ./gen_features.py` (create CQT-Spectrogram features)
4. Execute `python ./gen_split_indices.py` (split data for model training)
5. Execute `python ./train.py --model [BTC/CRNN] --index [1-5]`

### Test
1. Move 5 model data(.pth format) to "04_[BTC/CRNN]model/saved_models/"
2. Execute `python ./train.py --model [BTC/CRNN]`

### Inference
1. Execute `python ./inference.py` --path [path of target audio file(.wav format)]
2. Check result file under "./09_result/"

## Code Descriptions
- `01_utils/gen_features.py` : Generates CQT-Spectrogram features based on audio and chord data stored in "02_audiofiles", "02_chordfiles".
- `01_utils/gen_split_indices.py`: Splits Data for Train/Test and 5-Fold Cross Validation.
- `01_utils/torch_utils.py` : Defines classes and functions to be used when training and testing models.
- `04_BTCmodel/model.py` : Defines Bi-Directional Transformer model.
- `04_BTCmodel/modules.py` : Defines Modules for Bi-Directional Transformer model with Pytorch.
- `04_CRNNmodel/model.py` : Defines CRNN model with Pytorch.
- `train.py` : for training model.
- `test.py` : for testing model.
- `inference.py` : infers Chords from Audio File.

## Reference
[BTC]
Jonggwon Park et al., "A Bi-Directional Transformer for Musical Chord Recognition", 20th ISMIR, pp. 620-627, 2019.

[CRNN]
Junyan Jiang et al., "Large-vocabulary Chord Transcription Via Chord Structure Decomposition", 20th ISMIR, pp. 644-651, 2019.
