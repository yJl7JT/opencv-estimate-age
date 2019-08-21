TensorFlow年龄识别演示例子
==========================================================

## 代码源
https://github.com/dpressel/rude-carnie
https://github.com/nanpian/rude-carnie

### age_demo.py 是对guess.py精简后的
saver.restore 修改模型路劲
tf.gfile.FastGFile 修改加载图片的路劲

### 目录ckpt
saver.save(sess, "ckpt/model.ckpt") 从新保存模型

### 目录pb 是模型进行转换

### 目录skil 
是把模型部署SKIL平台，通过python发送请求
```bash
pip install skil_client
```