# WaveRNN

This is a Pytorch implementation of [WaveRNN](
https://arxiv.org/abs/1802.08435v1)
provided:

## Preparation

### Requirements

* Python 3.6 or newer
* PyTorch with CUDA enabled

### Preparing data

1. Set parameters in `utils/audio.py`, In particular, you should set `sample_rate, hop_length, win_length`
2. `python process.py --wav_dir='wavs' --output='data'`

## Training

`train.py` is the entry point:

```
$ python train.py
```

Trained models are saved under the `logdir` directory.

## Generating

`generate.py` is the entry point:

```
$ python generate.py --resume="ema_logdir"
```

audios are saved under the `out` directory.

# Reference

1. [fatchord/WaveRNN](https://github.com/fatchord/WaveRNN).
2. [mkotha/WaveRNN](https://github.com/mkotha/WaveRNN).
