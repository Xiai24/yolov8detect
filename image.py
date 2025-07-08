import cv2
from ultralytics import YOLO

# 打开图像
img_path = "./labeltest/labels/teimg.jpg" # 修改为你的图像路径
img = cv2.imread(filename=img_path)

# 加载模型
model = YOLO(model="yolov8tennis.pt") # 修改为你的模型路径

# 进行推理
res = model(img)

# 绘制推理结果
annotated_img = res[0].plot()

# 显示图像
cv2.imshow(winname="YOLOV8", mat=annotated_img)
cv2.waitKey(delay=10000)

# 保存结果
cv2.imwrite(filename="result.jpg", img=annotated_img)