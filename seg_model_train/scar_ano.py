
import re
import cv2
import numpy as np
import glob
from stain_anomaly import add_stain
import random
def show_img(image):
    #image=cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4))
    cv2.imshow("image",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def create_random_strip(image, length_range, width_range, brightness_change_range, small_lenth_threshold,
                        ):
    height, width = image.shape[:2]

    # 在给定范围内随机选择长度和宽度
    strip_length = np.random.randint(length_range[0], length_range[1])
    strip_width = np.random.randint(width_range[0], width_range[1])

    # 随机选择长条的起点
    start_x = np.random.randint(0, width - strip_length)
    start_y = np.random.randint(0, height - strip_width)

    # 提取原图像相应区域并计算平均亮度
    image_strip_region = image[start_y:start_y + strip_width, start_x:start_x + strip_length]
    mean_brightness = np.mean(image_strip_region)

    # 生成一个随机长条（亮度变化可正可负：正=比背景亮，负=比背景深）
    if strip_length < small_lenth_threshold:
        # 很短的长条：随机深色(0)或浅色(255)
        strip_val = 0 if random.random() < 0.5 else 255
        strip = np.full_like(image_strip_region, strip_val, dtype=np.uint8)
    else:
        brightness_change = np.random.randint(brightness_change_range[0], brightness_change_range[1])
        strip_val = np.clip(mean_brightness + brightness_change, 0, 255)
        strip = np.full_like(image_strip_region, strip_val, dtype=np.uint8)

    # 使用高斯模糊平滑长条的边缘
    strip = cv2.GaussianBlur(strip, (21, 21), 0)

    return strip, start_x, start_y, strip_width, strip_length


def rotate_strip(strip, angle):
    height, width = strip.shape[:2]
    center = (width // 2, height // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_strip = cv2.warpAffine(strip, M, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return rotated_strip


def modify_image_with_strip(image, strip, start_x, start_y, strip_width, strip_length):
    # 提取原图像相应区域
    image_strip = image[start_y:start_y + strip_width, start_x:start_x + strip_length]

    # 将生成的长条与原图像区域融合
    blended_strip = cv2.addWeighted(image_strip, 0.7, strip, 0.3, 0)

    # 将融合后的区域放回原图像
    image[start_y:start_y + strip_width, start_x:start_x + strip_length] = blended_strip

    return image







def scar_creat(image,
               length_range=(1, 20),
               width_range=(600, 1024),
               brightness_change_range=(-35, 40),  # 负=深色划痕，正=浅色划痕
               small_length_threshold=5,
               ):

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 生成并修改长条
    strip, start_x, start_y, strip_width, strip_length = create_random_strip(image, length_range, width_range,
                                                                             brightness_change_range,
                                                                             small_length_threshold,
                                                                             )
    print(strip_width, strip_length)#strip = np.repeat(strip[:, :, np.newaxis], 3, axis=2)
    # 融合旋转后的长条到原图像
    result_image = modify_image_with_strip(image, strip, start_x, start_y, strip_width, strip_length)
    # result_image = cv2.cvtColor(result_image, cv2.COLOR_GRAY2BGR)
    return result_image


if __name__ == '__main__':
    images_path=r"image_all\image_data_01_26B\CAM3\train\good\*"
    # 获取文件夹中所有图片的路径
    image_files = glob.glob(images_path)
    # 按数字顺序排序文件名
    def numerical_sort(value):
        """
        用于按数字顺序对文件名进行排序的函数。
        提取文件名中的数字并进行比较。
        """
        numbers = re.findall(r'\d+', value)
        return int(numbers[-1]) if numbers else float('inf')
    sorted_files_images_path = sorted(image_files, key=numerical_sort)

    for image_path in sorted_files_images_path:
        image = cv2.imread(image_path)
        image_2 =image.copy()
        #corrupt_image = add_stain(image, size='0.1-2', color='200-255', irregularity=0.5, blur=0)

        # 融合旋转后的长条到原图像
        result_image = scar_creat(image_2)
        result_image = cv2.resize(result_image,(256,256))
        image = cv2.resize(image, (256, 256))
        # 绘制绿色的矩形框
        #cv2.rectangle(result_image, (start_x-15, start_y-15), (start_x + strip_length+15, start_y + strip_width+15), (0, 255, 0), 2)

        # 显示结果图像
        #cv2.imshow('Original Image', image_bgr)
        cv2.imshow('Image with Modified Strip', result_image)
        cv2.imshow('Image raw', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()




