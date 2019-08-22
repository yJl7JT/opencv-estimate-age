import tensorflow as tf
import  numpy as np
class ImageCoder(object):
    """
    Helper class that provides TensorFlow image coding utilities.
    """

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def png_to_jpeg(self, image_data):
        # Convert the image data from png to jpg
        return self._sess.run(self._png_to_jpeg,
                              feed_dict={self._png_data: image_data})

    def decode_jpeg(self, image_data):
        # Decode the image data as a jpeg image
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3, "JPEG needs to have height x width x channels"
        assert image.shape[2] == 3, "JPEG needs to have 3 channels (RGB)"
        return image

RESIZE_FINAL=227
def _process_image(filename, coder):
    with tf.gfile.FastGFile(filename, 'rb') as f:
        image_data = f.read()

    # Convert any PNG to JPEG's for consistency.
    image = coder.decode_jpeg(image_data)

    crops = []
    print('Running multi-cropped image')
    h = image.shape[0]
    w = image.shape[1]
    hl = h - RESIZE_FINAL
    wl = w - RESIZE_FINAL

    crop = tf.image.resize_images(image, (RESIZE_FINAL, RESIZE_FINAL))

    crops.append(tf.image.per_image_standardization(crop)) #对图像进行标准化，转化成亮度均值为0，方差为1.
    crops.append(tf.image.flip_left_right(crop))  #左右翻转

    corners = [(0, 0), (0, wl), (hl, 0), (hl, wl), (int(hl / 2), int(wl / 2))]
    for corner in corners:
        ch, cw = corner
        cropped = tf.image.crop_to_bounding_box(image, ch, cw, RESIZE_FINAL, RESIZE_FINAL)
        crops.append(tf.image.per_image_standardization(cropped))
        flipped = tf.image.flip_left_right(cropped)
        crops.append(tf.image.per_image_standardization(flipped))

    image_batch = tf.stack(crops)

    print("一共多少图片",np.array(image_batch).shape)
    return image_batch

'''
crop_to_bounding_box 参数：

image：形状为[batch, height, width, channels]的4-D张量,或形状为[height, width, channels]的3-D张量.
offset_height：输入中结果左上角的垂直坐标.
offset_width：输入中结果左上角的水平坐标.
target_height：结果的高度.
target_width：结果的宽度.
返回值：

如果image是四维,则返回形状为[batch, target_height, target_width, channels]的四维浮动张量；
如果image是三维的,则返回形状为[target_height, target_width, channels]的三维浮动张量.

'''