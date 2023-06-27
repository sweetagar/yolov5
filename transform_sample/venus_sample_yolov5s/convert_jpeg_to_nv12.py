import cv2
import numpy as np

def convert_jpeg_to_nv12(image_path):
    # 读取JPEG图像
    image = cv2.imread(image_path)

    # 将图像转换为YUV颜色空间
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    # 将YUV图像转换为NV12格式
    nv12_image = np.concatenate((yuv_image[:, :, 0], yuv_image[:, :, 1:]), axis=1)

    return nv12_image

def display_nv12_image(nv12_image, width, height):
    # 将NV12图像解析为YUV图像
    yuv_image = np.zeros((int(height * 1.5), width), dtype=np.uint8)
    yuv_image[:, :width] = nv12_image[:, :width]
    yuv_image[:, width:] = np.repeat(nv12_image[:, width:], 2, axis=1)

    # 将YUV图像转换为BGR格式
    bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV12)

    # 显示BGR图像
    cv2.imshow("NV12 Image", bgr_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

image_path = "./bus.jpg"
image = cv2.imread(image_path)

height, width, _ = image.shape

nv12_image = convert_jpeg_to_nv12(image_path)
display_nv12_image(nv12_image, width, height)
