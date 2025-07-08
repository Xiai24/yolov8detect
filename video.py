import cv2
from ultralytics import YOLO

# 加载模型
model = YOLO(model="yolov8tennis.pt") # 修改为你的模型路径

# 打开视频
video_path = "video.mp4" # 修改为你的视频路径
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    res, frame = cap.read()
    if res:
        # 进行推理
        results = model(frame)

    # 绘制结果
    annotated_frame = results[0].plot()

    # 显示图像
    cv2.imshow(winname="YOLOV8", mat=annotated_frame)

    # 按ESC退出
    if cv2.waitKey(1) == 27:
        break
    else:
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()