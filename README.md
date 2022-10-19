# PiMOD

PiMOD (Pitch Estimation by Multiple Octave Decoders) is a neural-network-based pitch estimation method.



Yael Segal(segal.yael@campus.technion.ac.il)\
Joseph Keshet (jkeshet@technion.ac.il)



PiMOD is a software package for automatic estimation of pitch. We propose a neural-based architecture composed of CNN as fearues extractor and a linear layers as multiple octave decoders. 

The model was present in the paper [Pitch Estimation by Multiple Octave Decoders](https://ieeexplore.ieee.org/document/9501499).
If you find our work useful please cite :
```
@article{segal2021pitch,
  title={Pitch Estimation by Multiple Octave Decoders},
  author={Segal, Yael and Arama-Chayoth, May and Keshet, Joseph},
  journal={IEEE Signal Processing Letters},
  volume={28},
  pages={1610--1614},
  year={2021},
  publisher={IEEE}
}
```


## Installation instructions:
Python 3.9+

Download the code:

git clone https://github.com/YaelSegal/PiMOD

```
python3 -m pip install --user virtualenv
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```


## Training:

Place your .wav files and labels file (.txt) in the same directory.
The code assumes a main directory and folds, where you can use each fold for train/val/test. 
The  data structure is as follows: 

```
SOME_PATH/DATA_PATH
        └───1
        │   │   1.wav
        │   │   1.txt
        │   │   2.wav
        │   │   2.txt
        └───2
        │   │   3.wav
        │   │   3.txt
        │   │   4.wav
        │   │   4.txt
```

Example of training command:
```
python run.py --data SOME_PATH/DATA_PATH --train_folds 1_2_3 --val_folds 4 --test_folds 5 --experiment_name my_exp --cuda --neptune
```

Some highlights:
1. The code supporting logging to [neptune](https://neptune.ai/) using the flag --neptune.
2. To run on GPU, use the flag --cuda
3. Look at the run.py file to see all the script hyper-parameters

## Inference:

```
python predict.py --data SOME_PATH/DATA_PATH/FILE.wav --cuda --model ./model/6_decoders_MDB_KEELE.pth --batch_size 32 --outpath OUTPUT_DIRECTORY
```
Some highlights:
1. The prediction script can run on a directory (with multiple files) or a single file. 
2. To run on GPU, use the flag --cuda
3. The default model is ./model/6_decoders_MDB_KEELE.pth, which was train on MDB-STEM-Synth[1] and Keele[2] datasets.

Example of prediction file:
```
time,frequency,confidence
.000, 0.000, 1.000
0.010, 0.000, 1.000
0.020, 0.000, 1.000
0.030, 0.000, 1.000
0.040, 0.000, 1.000
0.050, 0.000, 1.000
0.060, 0.000, 1.000
0.070, 0.000, 1.000
0.080, 0.000, 1.000
0.090, 0.000, 1.000
0.100, 0.000, 1.000
0.110, 231.109, 0.730
0.120, 217.145, 1.000
0.130, 212.794, 1.000
0.140, 209.846, 1.000
0.150, 210.985, 1.000
0.160, 210.410, 1.000
0.170, 213.200, 1.000
0.180, 215.494, 1.000
0.190, 217.457, 1.000
```


## Please Note
* The current version only supports WAV files as input.
The model is trained on 16 kHz audio, so if the input audio has a different sample rate, it will be first resampled to 16 kHz using librosa (for training- this will make the training slower).

* Prediction/ Training is significantly faster if you run on GPU.


## References

[1] J. W. Kim, J. Salamon, P. Li, and J. P. Bello, “CREPE: A convolutional representation for pitch estimation,” in Proc. IEEE Int. Conf. Acoust., Speech, Signal Process., 2018, pp. 161–165.

[2] F. Plante, G. F. Meyer, and W. A. Ainsworth, “A pitch extraction refer- ence database,” in Proc. 4th Eur. Conf. Speech Commun. Technol., 1995, pp. 837–840.
