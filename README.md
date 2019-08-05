TensorFlow年龄识别演示例子
==========================================================

## 代码源
https://github.com/yJl7JT/opencv-estimate-age

### 模型下载
You can find a pre-trained age checkpoint for inception here:
https://drive.google.com/drive/folders/0B8N1oYmGLVGWbDZ4Y21GLWxtV1E
因为被墙，在这提供百度网盘下载链接，下载后解压到本地工程里面的22801目录
http://pan.baidu.com/s/1mhLBIHy
### 运行步骤
```
python2.7 guess.py --model_type inception --model_dir ./22801 --filename test1.jpg
```
test1.jpg是拍的人像数据
需要安装python2.7、openCV3.0、tensorFlow1.0环境。
Ubuntu环境为14.04LTS
如果没有安装openCV环境，则会出现以下问题：
```
  Traceback (most recent call last):
      File "guess.py", line 12, in <module>
      from utils import ImageCoder, make_batch, FaceDetector
      File "/home/david/work/tensorflow/rude-carnie/utils.py", line 7, in <module>
      import cv2
      ImportError: No module named cv2
```

###安装openCV
下载地址https://codeload.github.com/Itseez/opencv/zip/3.0.0
主要安装步骤参考http://www.cnblogs.com/asmer-stone/p/4592421.html


###运行结果
```
~/work/tensorflow/rude-carnie$ python2.7 guess.py --model_type inception --model_dir ./22801 --filename   test1.jpg 
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE3 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
Executing on /cpu:0
selected (fine-tuning) inception model
./22801/checkpoint-14999
Running file test1.jpg
Running multi-cropped image
Guess @ 1 (4, 6), prob = 0.99
Guess @ 2 (8, 12), prob = 0.01
```
