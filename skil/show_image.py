best = 7
label = "小人" if best < 3  else "大人"

import cv2
from PIL import ImageFont, ImageDraw, Image
import numpy as np

bk_img = cv2.imread("D:/tmp/227/122222.jpg")
bk_img = cv2.resize(bk_img,(500,500))
# 设置需要显示的字体
fontpath = "font/simsun.ttc"   # 32为字体大小
font = ImageFont.truetype(fontpath, 32)
img_pil = Image.fromarray(bk_img)
draw = ImageDraw.Draw(img_pil)
# 绘制文字信息<br># (100,300/350)为字体的位置，(255,255,255)为白色，(0,0,0)为黑色
cv2.rectangle(bk_img, (20,30), (250,78), (255,255,255), -1)
draw.text((230, 50), label, font=font, fill=(0,0,255))
# draw.text((100, 350), "你好", font=font, fill=(255, 255, 255))
bk_img = np.array(img_pil)

cv2.imshow(" ", bk_img)
cv2.waitKey()

