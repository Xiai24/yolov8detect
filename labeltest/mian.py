import cv2
import os
import glob
from ultralytics import YOLO


def save_yolo_label(results, img_path, label_dir="labels"):
    """保存YOLO格式标签文件（单个图片）"""
    os.makedirs(label_dir, exist_ok=True)
    img_height, img_width = results[0].orig_shape
    # 生成与图片同名的标签文件路径
    label_name = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
    label_path = os.path.join(label_dir, label_name)

    with open(label_path, 'w') as f:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # 左上角和右下角坐标
            class_id = int(box.cls)  # 类别ID

            # 计算归一化坐标（YOLO格式要求）
            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height

            # 写入标签（保留6位小数精度）
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    print(f"标签已保存: {label_path}")


def process_single_image(img_path, model, output_img_dir="output_images", output_label_dir="output_labels"):
    """处理单张图片（内部调用，供批量处理使用）"""
    # 读取图片
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ 跳过无效图片: {img_path}")
        return

    # 模型推理（统一参数）
    results = model(img, conf=0.25, imgsz=640)  # 置信度和输入尺寸可调整

    # 保存标签（无论是否检测到目标，都生成空文件避免后续遗漏）
    os.makedirs(output_label_dir, exist_ok=True)
    save_yolo_label(results, img_path, output_label_dir)

    # 生成并保存标注后的图片
    os.makedirs(output_img_dir, exist_ok=True)
    annotated_img = results[0].plot()  # 即使无检测结果，也生成原始图带文字标注
    output_img_path = os.path.join(output_img_dir, os.path.basename(img_path))
    cv2.imwrite(output_img_path, annotated_img)
    print(f"✅ 处理完成: {output_img_path}")


def process_folder(img_folder, model):
    """批量处理文件夹中的所有图片"""
    # 支持的图片格式（可根据需要添加）
    img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
    img_paths = []
    for ext in img_extensions:
        img_paths.extend(glob.glob(os.path.join(img_folder, ext)))

    if not img_paths:
        print(f"❌ 未在文件夹 {img_folder} 中找到图片")
        return

    print(f"📁 找到 {len(img_paths)} 张图片，开始批量处理...")
    for i, img_path in enumerate(img_paths, 1):
        print(f"\n处理第 {i}/{len(img_paths)} 张:")
        process_single_image(img_path, model)
    print("\n📊 批量处理完成！")


def process_camera(model, camera_id=0):
    """摄像头实时检测"""
    cap = cv2.VideoCapture(camera_id)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("❌ 无法读取摄像头画面，退出")
            break

        # 实时推理
        results = model(frame, conf=0.25, imgsz=640)
        annotated_frame = results[0].plot()

        # 显示画面
        cv2.imshow("YOLOv8 实时检测", annotated_frame)
        if cv2.waitKey(1) == 27:  # 按ESC键退出
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 初始化模型（替换为你的模型路径）
    model = YOLO("yolov8tennis.pt")

    # 选择处理模式
    print("请选择处理模式：")
    print("1 - 单张图片处理")
    print("2 - 摄像头实时检测")
    print("3 - 文件夹批量处理")
    mode = input("输入模式编号 (1/2/3): ")

    if mode == "1":
        img_path = input("请输入图片路径: ").strip()
        # 单张图片处理时，输出到默认文件夹
        process_single_image(img_path, model, "single_output", "single_labels")
        # 显示结果（单张模式下额外显示）
        output_img = cv2.imread(os.path.join("single_output", os.path.basename(img_path)))
        if output_img is not None:
            cv2.imshow("检测结果", output_img)
            cv2.waitKey(5000)  # 显示5秒后关闭
            cv2.destroyAllWindows()

    elif mode == "2":
        print("📹 启动摄像头实时检测（按ESC退出）...")
        process_camera(model)

    elif mode == "3":
        img_folder = input("请输入图片文件夹路径: ").strip()
        process_folder(img_folder, model)

    else:
        print("❌ 无效的模式选择，请输入 1/2/3")