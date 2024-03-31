# Contrast-Phys and Contrast-Phys+

**Update: The paper for [Contrast-Phys+](https://ieeexplore.ieee.org/document/10440521) has been accepted by TPAMI. Please check the `contrast-phys+` folder for the code.**

This is the official code repository of our ECCV 2022 paper "Contrast-Phys: Unsupervised Video-based Remote Physiological Measurement via Spatiotemporal Contrast" and our paper "Contrast-Phys+: Unsupervised and Weakly-supervised Video-based Remote Physiological Measurement via Spatiotemporal Contrast". 

We incorporate prior knowledge about remote photoplethysmography (rPPG) into contrastive learning to achieve unsupervised rPPG training. The method does not require any ground truth for training.

[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720488.pdf), [Poster](https://github.com/zhaodongsun/contrast-phys/releases/download/aux/0205.pdf), [Video](https://github.com/zhaodongsun/contrast-phys/releases/download/aux/0205.mp4)

![](https://github.com/zhaodongsun/contrast-phys/releases/download/aux/all.png)

## Demo

Please go to [Demo](/demo) folder for More details.

## Prerequisite

Please check `requirement.txt` for the required Python libraries.

## Dataset Preprocessing

需要将PURE的图片数据变成无损的.avi数据，可以在python中使用如下代码：（To convert PURE image data to lossless.avi data, use the following code in python:）

```
!ffmpeg -framerate 30 -pattern_type glob -i {png_path} -c:v ffv1 {output_path}
```

The original videos are firstly preprocessed to crop the face. Facial landmarks are generated using [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace). I use the following OpenFace commend to get the facial landmark (.csv file) from each video. For more details, you could check [here](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Command-line-arguments).

If you are using OpenFace on Linux, you may use the following command in Bash to get the landmark from one video. You can use Bash For loop to get landmarks from multiple videos.
```
./FeatureExtraction -f <VideoFileName> -out_dir <dir> -2Dfp
```

If you are using OpenFace on Windows, you may use the following For loop in Python to get the landmarks for multiple videos. You can download the Windows Version of OpenFace [here](https://1drv.ms/u/s!AtCpzthip8c9_RsyovL9Ngfd6OKq?e=b1HtC4).
```
for v in video_list:
    os.system('.\\openface\\FeatureExtraction.exe -f %s -out_dir %s -2Dfp'%(v, landmarks_folder))
```

openface 2安装推荐使用windows，只需要下载最新的visual studio和github仓库中的zip文件以及模型参数文件即可，2024年3月在win 10上成功部署。（Windows is recommended for openface 2 installation. All you need to do is download latest visual studio, github repository and model parameter file. The deployment was successful on win 10 in March 2024.）

在windows系统中也可以参照openface.bat对视频执行特征提取操作。（In windows system, you can also refer to openface.bat to perform feature extraction operations on videos.）

We first get the minimum and maximum horizontal and vertical coordinates of the landmarks to locate the central facial point for each frame. The bounding box size is 1.2 times the vertical coordinate range of landmarks from the first frame and is fixed for the following frames. After getting the central facial point of each frame and the size of the bounding box, we crop the face from each frame. The cropped faces are resized to $128 \times 128$, which are ready to be fed into our model. Video frames in a video should be stored in a .h5 file. Please refer to `preprocessing.py` for more details. For example, for UBFC-rPPG dataset, the processed dataset should like

```
  dataset
  ├── 1.h5
  ├── 3.h5
  .
  .
  ├── 48.h5
  ├── 49.h5

```

For each .h5 file
```
  X.h5
  ├── imgs  # face videos with cropped faces, shape [N, 128, 128, C]. N is the temporal length. C is the color channel, for RGB videos, C=3, for NIR videos, C=1.
  ├── bvp   # ground truth PPG signal, shape [N]. You don't need to include it for the training set.
```
 
 Since Contrast-Phys is an unsupervised method, `bvp` is not needed for training. You may not include `bvp` in the training set, but `bvp` is needed in the test set for performance evaluation. You can download one .h5 file example [here](https://1drv.ms/u/s!AtCpzthip8c9-xlaJwlaK2zU6sfn?e=OH9klk).


## Training and testing

### Training
Please make sure your dataset is processed as described above. You only need to modify the code in a few places in `train.py` and `utils_data.py` to start your training. I have marked `TODO` and made comments in `train.py` and `utils_data.py` where you might need to modify. After modifying the code, you can directly run

```
python train.py
```
When you first run `train.py`, the training recording including model weights and some metrics will be saved at `./results/1`

### Testing

After training, you can test the model on the test set. Please make sure .h5 files in test set have `bvp`. You can directly run
```
python test.py with train_exp_num=1
```
The predicted rPPG signals and ground truth PPG signals are saved in `./results/1/1`. You can filter the rPPG signals by `butter_bandpass` function with lowcut=0.6 and highcut=4 and get heart rates by `hr_fft` function in `utils_sig.py`. To get the ground truth heart rates, you should first filter ground truth PPG signals by `butter_bandpass` function with lowcut=0.6 and highcut=4 and get ground truth heart rates by `hr_fft` function.

## Citation

```
@article{sun2024,
  title={Contrast-Phys+: Unsupervised and Weakly-supervised Video-based Remote Physiological Measurement via Spatiotemporal Contrast},
  author={Sun, Zhaodong and Li, Xiaobai},
  journal={TPAMI},
  year={2024}
}

@inproceedings{sun2022contrast,
  title={Contrast-Phys: Unsupervised Video-based Remote Physiological Measurement via Spatiotemporal Contrast},
  author={Sun, Zhaodong and Li, Xiaobai},
  booktitle={European Conference on Computer Vision},
  year={2022},
}
```
