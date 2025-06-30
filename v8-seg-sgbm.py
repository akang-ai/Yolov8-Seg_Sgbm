# -----------------------------------------------------------------------#
#   predict.py将单张图片预测、摄像头检测、FPS测试和目录遍历检测等功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
#   添加了CUDA加速和多线程异步处理
#   添加了红通道补偿预处理功能
#   水下专用
# -----------------------------------------------------------------------#
import time
import math
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import os
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

# 检查CUDA可用性
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device.upper()}")
if device == "cuda":
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")

# -------------------------------------双目相机参数-----------------------------------------
""""
#陆上标定
left_camera_matrix = np.array([[427.0117, 0, 341.0647], [0, 426.6854, 261.4323], [0, 0, 1]])
left_distortion = np.array([[0.0324, -0.0313, 0.00019638, -0.00058164, 0]])

right_camera_matrix = np.array([[427.7178, 0, 340.9055], [0, 427.4384, 257.0497], [0, 0, 1]])
right_distortion = np.array([[0.0322, -0.0309, 0.000038662, 0.00016210, 0]])

R = np.array([[0.9991, -0.0029, 0.0414],
              [0.0028, 1, 0.0037],
              [-0.0414, -0.0036, 0.9991]])
T = np.array([-58.8666, -0.0245, 1.1520])
"""
# 水下标定
left_camera_matrix = np.array([[528.5528, 0, 348.6572], [0, 527.7025, 254.4125], [0, 0, 1]])
left_distortion = np.array([[0.3406, 0.0973, -0.0084, 0.0108, 0]])

right_camera_matrix = np.array([[529.5957, 0, 347.7455], [0, 528.7845, 249.9668], [0, 0, 1]])
right_distortion = np.array([[0.3368, 0.1032, -0.0090, 0.0030, 0]])

R = np.array([[0.9994, -0.0021, 0.0346],
              [0.0020, 1, 0.0031],
              [-0.0346, -0.0030, 0.9994]])
T = np.array([-58.6681, -0.1043, 0.6831])
size = (640, 480)

# 立体校正
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
    left_camera_matrix, left_distortion,
    right_camera_matrix, right_distortion,
    size, R, T
)

# 校正映射
left_map1, left_map2 = cv2.initUndistortRectifyMap(
    left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2
)
right_map1, right_map2 = cv2.initUndistortRectifyMap(
    right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2
)


# SGBM参数
def get_sgbm():
    stereo = cv2.StereoSGBM_create(
        minDisparity=1,
        numDisparities=64,
        blockSize=5,
        P1=8 * 3 * 5 ** 2,
        P2=32 * 3 * 5 ** 2,
        disp12MaxDiff=-1,
        preFilterCap=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=100,
        mode=cv2.STEREO_SGBM_MODE_HH
    )
    return stereo


# 初始化YOLOv8-seg模型（使用CUDA加速）
yolo_model = YOLO('runs/segment/train_640/weights/best.pt').to(device)  # 使用分割模型并移至GPU

# 相机参数（根据实际标定调整）
focal_length = 528.5528  # 左相机焦距
baseline = 0.0587  # 基线距离（米）


def calculate_depth(disparity):
    """将视差值转换为深度（单位：米）"""
    with np.errstate(divide='ignore'):  # 忽略除以0的警告
        depth = (focal_length * baseline) / (disparity + 1e-6)
    return depth


def red_channel_compensation(image, red_gain=1.25):
    """
    对图像进行红通道补偿，以改善水下图像的色彩平衡
    :param image: 输入图像 (BGR格式)
    :param red_gain: 红色通道增益因子
    :return: 补偿后的图像
    """
    # 分离通道
    b, g, r = cv2.split(image.astype(np.float32))

    # 应用红通道增益
    r = np.clip(r * red_gain, 0, 255)

    # 平衡其他通道 (可选)
    b = np.clip(b , 0, 255)
    g = np.clip(g , 0, 255)

    # 合并通道
    compensated = cv2.merge([b, g, r])
    compensated = np.clip(compensated, 0, 255).astype(np.uint8)

    return compensated


def sgbm_worker(left_gray, right_gray, result_queue):
    """SGBM视差计算工作线程"""
    try:
        # 立体校正
        img1_rectified = cv2.remap(left_gray, left_map1, left_map2, cv2.INTER_LINEAR)
        img2_rectified = cv2.remap(right_gray, right_map1, right_map2, cv2.INTER_LINEAR)

        # 计算视差
        stereo = get_sgbm()
        disparity = stereo.compute(img1_rectified, img2_rectified)

        # 归一化视差图
        disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        disp_color = cv2.applyColorMap(disp, cv2.COLORMAP_JET)

        result_queue.put((disparity, disp_color))
    except Exception as e:
        print(f"SGBM计算错误: {e}")
        result_queue.put((None, None))


def yolo_inference_worker(frame, result_queue):
    """YOLO推理工作线程"""
    try:
        # 使用YOLOv8-seg进行实例分割
        results = yolo_model(frame, verbose=False)  # 禁用详细输出提高速度

        # 收集检测结果
        detections = []
        for result in results:
            # 检查是否有检测结果
            if result.boxes is None or result.masks is None:
                continue

            # 获取掩码和边界框
            masks = result.masks
            boxes = result.boxes

            # 确保boxes和masks长度一致
            min_len = min(len(boxes), len(masks))

            for i in range(min_len):
                box = boxes[i]
                mask = masks[i]

                # 只处理置信度大于0.4的目标
                if box.conf[0] > 0.4:
                    # 获取类别ID和名称
                    cls_id = int(box.cls[0])
                    cls_name = yolo_model.names[cls_id]

                    # 获取边界框坐标
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

                    # 获取分割掩码
                    obj_mask = mask.data[0].cpu().numpy().astype(np.uint8)

                    detections.append({
                        'cls_name': cls_name,
                        'confidence': float(box.conf[0]),
                        'box': (x1, y1, x2, y2),
                        'mask': obj_mask
                    })

        result_queue.put(detections)
    except Exception as e:
        print(f"YOLO推理错误: {e}")
        result_queue.put([])


def process_frame_async(frame1, frame2, is_video=False):
    """异步处理单个帧并返回结果图像和视差图"""
    # 进行红通道补偿预处理
    frame1_compensated = red_channel_compensation(frame1)
    frame2_compensated = red_channel_compensation(frame2)

    # 创建结果队列
    sgbm_queue = queue.Queue(maxsize=1)
    yolo_queue = queue.Queue(maxsize=1)

    # 转换为灰度图用于SGBM
    left_gray = cv2.cvtColor(frame1_compensated, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(frame2_compensated, cv2.COLOR_BGR2GRAY)

    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=2) as executor:
        # 提交SGBM任务
        sgbm_future = executor.submit(sgbm_worker, left_gray, right_gray, sgbm_queue)

        # 提交YOLO任务 (使用补偿后的彩色图像)
        yolo_future = executor.submit(yolo_inference_worker, frame1_compensated, yolo_queue)

        # 等待结果
        sgbm_future.result()
        yolo_future.result()

    # 获取SGBM结果
    disparity, disp_color = sgbm_queue.get()

    # 获取YOLO结果
    detections = yolo_queue.get()

    # 处理检测结果
    result_frame = frame1_compensated.copy()

    for detection in detections:
        cls_name = detection['cls_name']
        confidence = detection['confidence']
        x1, y1, x2, y2 = detection['box']
        obj_mask = detection['mask']

        # 在视差图中提取目标区域
        if disparity is not None:
            # 确保视差图和掩码尺寸一致
            if disparity.shape[:2] != obj_mask.shape[:2]:
                obj_mask = cv2.resize(obj_mask, (disparity.shape[1], disparity.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)

            # 直接使用掩码索引获取视差值（保持原始精度）
            obj_disparity = disparity[obj_mask > 0]

            # 计算有效视差区域（忽略无效值）
            valid_disp = obj_disparity[obj_disparity > 0]
            if valid_disp.size > 0:
                # 计算平均视差（保持原始精度）
                avg_disparity = np.mean(valid_disp) / 16.0  # SGBM返回16倍整数值

                # 计算深度
                depth = calculate_depth(avg_disparity)

                # 计算中心点坐标
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # 绘制边界框
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

                # 绘制分割轮廓
                contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(result_frame, contours, -1, (0, 255, 0), 2)

                # 显示深度信息
                label = f"{cls_name} {confidence:.2f} dis={depth:.2f}m"
                cv2.putText(result_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return result_frame, disp_color, frame1_compensated


if __name__ == "__main__":
    # ----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'           表示单张图片预测
    #   'video'             表示视频检测
    #   'fps'               表示测试fps
    #   'dir_predict'       表示遍历文件夹进行检测并保存。
    # ----------------------------------------------------------------------------------------------------------#
    mode = "video"
    # -------------------------------------------------------------------------#
    #   crop                指定了是否在单张图片预测后对目标进行截取
    #   count               指定了是否进行目标的计数
    # -------------------------------------------------------------------------#
    crop = False
    count = False
    # ----------------------------------------------------------------------------------------------------------#
    #   video_path          用于指定视频的路径，当video_path=0时表示检测摄像头
    #   video_save_path     表示视频保存的路径
    #   video_fps           用于保存的视频的fps
    # ----------------------------------------------------------------------------------------------------------#
    video_path = "004.mp4"
    video_save_path = ""
    video_fps = 15.0
    # ----------------------------------------------------------------------------------------------------------#
    #   test_interval       用于指定测量fps的时候，图片检测的次数
    #   fps_image_path      用于指定测试的fps图片
    # ----------------------------------------------------------------------------------------------------------#
    test_interval = 100
    fps_image_path = "img/street.jpg"
    # -------------------------------------------------------------------------#
    #   dir_origin_path     指定了用于检测的图片的文件夹路径
    #   dir_save_path       指定了检测完图片的保存路径
    # -------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path = "img_out/"

    # 目标分辨率 - 使用双目相机的原生分辨率
    TARGET_WIDTH = 640
    TARGET_HEIGHT = 480

    if mode == "predict":
        while True:
            img_path = input('输入图片路径: ')
            try:
                image = cv2.imread(img_path)
                if image is None:
                    print('图片打开错误! 请重试!')
                    continue

                # 分割左右图像 - 假设输入是拼接的左右图像
                height, width = image.shape[:2]
                frame1 = image[:, :width // 2]  # 左图像
                frame2 = image[:, width // 2:]  # 右图像

                # 显示原始图像
                cv2.imshow("Original", frame1)

                # 处理帧
                start_time = time.time()
                result_frame, disp_color, compensated_frame = process_frame_async(frame1.copy(), frame2.copy())
                process_time = time.time() - start_time

                # 显示补偿后的图像
                cv2.imshow("Compensated", compensated_frame)

                # 显示处理时间
                cv2.putText(result_frame, f"Process: {process_time * 1000:.1f}ms",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # 显示结果
                cv2.imshow("Result", result_frame)
                cv2.imshow("Disparity", disp_color)

                # 等待按键
                key = cv2.waitKey(0)
                if key == 27:  # ESC键退出
                    break

            except Exception as e:
                print(f'错误: {e}')
                continue

        cv2.destroyAllWindows()

    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        # 获取视频信息
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)

        print(f"视频信息: {frame_width}x{frame_height}, {fps:.2f} FPS, 总帧数: {total_frames}")

        # 创建两个视频写入器（结果视频和视差视频）
        out_result = None
        out_disp = None
        if video_save_path != "":
            # 创建结果视频路径
            base_name, ext = os.path.splitext(video_save_path)
            result_video_path = f"{base_name}_result.avi"
            disp_video_path = f"{base_name}_disp.avi"
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out_result = cv2.VideoWriter(result_video_path, fourcc, video_fps, (frame_width // 2, frame_height))
            out_disp = cv2.VideoWriter(disp_video_path, fourcc, video_fps, (frame_width // 2, frame_height))

        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

        # 性能统计
        fps = 0.0
        frame_count = 0
        total_process_time = 0.0
        min_process_time = float('inf')
        max_process_time = 0.0

        # 预热
        print("预热模型...")
        for _ in range(5):
            height, width = frame.shape[:2]
            frame1 = frame[:, :width // 2]  # 左图像
            frame2 = frame[:, width // 2:]  # 右图像
            _, _, _ = process_frame_async(frame1.copy(), frame2.copy(), is_video=True)

        print("开始处理视频...")
        while True:
            t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()
            if not ref:
                break

            frame_count += 1

            # 分割左右图像
            height, width = frame.shape[:2]
            frame1 = frame[:, :width // 2]  # 左图像
            frame2 = frame[:, width // 2:]  # 右图像

            # 处理帧
            process_start = time.time()
            result_frame, disp_color, compensated_frame = process_frame_async(frame1.copy(), frame2.copy(),
                                                                              is_video=True)
            process_time = time.time() - process_start

            # 更新性能统计
            total_process_time += process_time
            min_process_time = min(min_process_time, process_time)
            max_process_time = max(max_process_time, process_time)

            # 计算FPS
            fps = (fps + (1. / (time.time() - t1))) / 2

            # 显示性能信息
            cv2.putText(result_frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 显示结果
            cv2.imshow("Original", frame1)
            cv2.imshow("Compensated", compensated_frame)
            cv2.imshow("YOLOv8-Seg + SGBM (Async)", result_frame)
            cv2.imshow("Disparity", disp_color)

            # 保存结果到两个视频
            if out_result is not None and out_disp is not None:
                out_result.write(result_frame)
                out_disp.write(disp_color)

            # 退出检测
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 打印性能报告
        avg_process_time = total_process_time / frame_count
        print("\n视频处理完成!")
        print(f"处理帧数: {frame_count}/{total_frames}")
        print(f"最小处理时间: {min_process_time * 1000:.1f}ms")
        print(f"最大处理时间: {max_process_time * 1000:.1f}ms")
        print(f"平均FPS: {1 / avg_process_time:.2f}")

        capture.release()
        if out_result is not None:
            out_result.release()
        if out_disp is not None:
            out_disp.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        # FPS测试模式
        image = cv2.imread(fps_image_path)
        if image is None:
            print(f"无法打开图像: {fps_image_path}")
        else:
            # 分割左右图像
            height, width = image.shape[:2]
            frame1 = image[:, :width // 2]  # 左图像
            frame2 = image[:, width // 2:]  # 右图像

            # 预热GPU
            print("预热模型...")
            for _ in range(5):
                _, _, _ = process_frame_async(frame1.copy(), frame2.copy())

            # 测试FPS
            print("开始FPS测试...")
            start_time = time.time()
            process_times = []
            for i in range(test_interval):
                process_start = time.time()
                _, _, _ = process_frame_async(frame1.copy(), frame2.copy())
                process_time = time.time() - process_start
                process_times.append(process_time)

                # 显示进度
                if (i + 1) % 10 == 0:
                    print(f"已处理 {i + 1}/{test_interval} 帧")

            end_time = time.time()

            # 计算性能指标
            total_time = end_time - start_time
            avg_time = total_time / test_interval
            min_time = min(process_times)
            max_time = max(process_times)

            print("\nFPS测试完成!")
            print(f"总时间: {total_time:.4f} 秒")
            print(f"最小帧时间: {min_time * 1000:.1f}ms")
            print(f"最大帧时间: {max_time * 1000:.1f}ms")
            print(f"FPS: {1 / avg_time:.2f}")

    elif mode == "dir_predict":
        import os
        from tqdm import tqdm

        if not os.path.exists(dir_save_path):
            os.makedirs(dir_save_path)

        img_names = [f for f in os.listdir(dir_origin_path)
                     if f.lower().endswith(
                ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff'))]

        # 性能统计
        total_time = 0.0
        min_time = float('inf')
        max_time = 0.0

        print(f"开始批量处理 {len(img_names)} 张图像...")
        for img_name in tqdm(img_names):
            image_path = os.path.join(dir_origin_path, img_name)
            image = cv2.imread(image_path)

            if image is None:
                print(f"无法读取图像: {image_path}")
                continue

            # 分割左右图像
            height, width = image.shape[:2]
            frame1 = image[:, :width // 2]  # 左图像
            frame2 = image[:, width // 2:]  # 右图像

            # 处理帧
            start_time = time.time()
            result_frame, _, _ = process_frame_async(frame1.copy(), frame2.copy())
            process_time = time.time() - start_time

            # 更新性能统计
            total_time += process_time
            min_time = min(min_time, process_time)
            max_time = max(max_time, process_time)

            # 保存结果
            save_path = os.path.join(dir_save_path, img_name)
            cv2.imwrite(save_path, result_frame)

        # 打印性能报告
        avg_time = total_time / len(img_names)
        print("\n目录处理完成!")
        print(f"总时间: {total_time:.2f} 秒")
        print(f"最小处理时间: {min_time * 1000:.1f}ms")
        print(f"最大处理时间: {max_time * 1000:.1f}ms")

    else:
        raise AssertionError("请指定正确的模式: 'predict', 'video', 'fps', 'dir_predict'.")