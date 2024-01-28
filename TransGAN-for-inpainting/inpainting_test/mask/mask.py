# 自定义Mask
from PIL import Image

# 定义图像的尺寸
width = 512
height = 512
rgb_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)]
T = 48
# 创建一个空白的RGB图像对象
image = Image.new("RGB", (width, height))
# 遍历RGB元组列表，将每个元组对应的颜色绘制到图像上
for i in range(width):
    for j in range(height):
        if width / 2 - T <= i <= width / 2 + T and height / 2 - T <= j <= height / 2 + T:
            rgb = rgb_list[0]
            image.putpixel((i, j), rgb)  # 绘制像素颜色到图像上
        elif width / 2 - T <= i <= width / 2 + T and height / 2 - T <= j <= height / 2 + T:
            rgb = rgb_list[1]
            image.putpixel((i, j), rgb)  # 绘制像素颜色到图像上
        elif width / 2 - T <= i <= width / 2 + T and height / 2 - T <= j <= height / 2 + T:
            rgb = rgb_list[2]
            image.putpixel((i, j), rgb)  # 绘制像素颜色到图像上
        elif width / 2 - T <= i <= width / 2 + T and height / 2 - T <= j <= height / 2 + T:
            rgb = (255, 255, 255)
            image.putpixel((i, j), rgb)  # 绘制像素颜色到图像上
        else:
            rgb = (0, 0, 0)
            image.putpixel((i, j), rgb)  # 绘制像素颜色到图像上
# 保存图像到本地文件
image.save("mask.png", "PNG")
