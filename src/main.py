import os, sys

# 项目根与源码目录，保证可从任意工作目录启动
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
for sub_dir in ["perception", "communication", "serial_examples", "third_party"]:
    sys.path.insert(0, os.path.join(SRC_DIR, sub_dir))

from LivoxCtypesInterface import LivoxInterface
from Camera import Camera
from CoordinateConversions import CoordinateConverter
import threading
import cv2
from ultralytics import YOLO
from PIL import Image
from YOLOResultsFilter import remove_overlapping_boxes
from CommunicationAndGuess_ModifiedFromPFA import Communicator

team_color = "RED"

yolo_cls_to_communicator_cls = {
    0: "B1",
    1: "B2",
    2: "B3",
    3: "B4",
    4: "B5",
    5: "B7",
    6: "R1",
    7: "R2",
    8: "R3",
    9: "R4",
    10: "R5",
    11: "R7",
    12: "Empty"
}

def main():
    # 保证 finally 中即使初始化中途失败也能安全判断
    livoxInterface = None
    livoxThread = None
    camera = None
    communicator = None
    try:
        livoxInterface = LivoxInterface()
        livoxInterface.pyif_init()
        #livoxInterface.test()
        #livoxInterface.pyif_uninit()
        livoxThread = threading.Thread(target=livoxInterface.test, daemon=True)
        livoxThread.start()

        camera = Camera(camera_index=0)
        # 设置相机参数
        camera.set_exposure_time(10000)  # 设置曝光时间为10000微秒
        camera.set_gain(16.0)  # 设置增益为10dB
        camera.start_grabbing()

        #car_model_path = os.path.join(BASE_DIR, "model", "rmCar_yolov12n_int8_openvino_model", "rmCar_yolov12n.pt")
        car_model_path = os.path.join(BASE_DIR, "model", "rmCar_yolov12n_int8_openvino_model")
        car_conf_threshold = 0.25
        car_model = YOLO(car_model_path)
        #car_model.export(format="openvino",int8=True)

        #armor_model_path = os.path.join(BASE_DIR, "model", "rmArmor_yolov12n_int8_openvino_model", "rmArmor_yolov12n.pt")
        armor_model_path = os.path.join(BASE_DIR, "model", "rmArmor_yolov12n_int8_openvino_model")
        armor_conf_threshold = 0.5
        armor_model = YOLO(armor_model_path)
        #armor_model.export(format="openvino",int8=True)

        communicator = Communicator(state=("R" if team_color == "RED" else "B"), visualize_map=True, visualize_information=True, allow_no_serial=True)

        coordinateConverter = CoordinateConverter(team_color = team_color, debugFlags = {"debugPosition": False, "debugFunction": True})

        while True:

            img = camera.get_image()
            if img is not None:
                car_model_results = car_model(img, conf=car_conf_threshold, verbose=False)

                # 去除重叠检测框，保留较大的
                keep_indices = remove_overlapping_boxes(car_model_results[0].boxes)
                car_model_results[0].boxes = car_model_results[0].boxes[keep_indices]

                # 处理每个检测框
                custom_car_boxes = []
                for carBox in car_model_results[0].boxes:
                    custom_car_boxes.append({"box": carBox})
                    x1, y1, x2, y2 = map(int, carBox.xyxy[0])
                    # 提取边界框区域
                    car_box_region = img[y1:y2, x1:x2]
                    # 在这里进行图像处理
                    armor_model_results = armor_model(car_box_region, conf=armor_conf_threshold, verbose=False)
                    annotated_img_armor = armor_model_results[0].plot()
                    clsCount = [0]*12
                    for armorBox in armor_model_results[0].boxes:
                        clsCount[int(armorBox.cls)] += 1
                    if sum(clsCount) == 0:
                        custom_car_boxes[-1]["cls"] = 12
                    else:
                        custom_car_boxes[-1]["cls"] = clsCount.index(max(clsCount))
                    # 将处理后的区域放回原图
                    img[y1:y2, x1:x2] = annotated_img_armor

                annotated_img = car_model_results[0].plot()

                img_radar = livoxInterface.image2dResult[0]
                img_radar_mask = livoxInterface.image2dResult[1]
                img_map_to_rad = cv2.resize(coordinateConverter.map_to_rad(annotated_img), (1024, 1024))
                cv2.copyTo(img_radar, img_radar_mask, img_map_to_rad)

                position_results = coordinateConverter.get_target_positions_and_draw(img_map_to_rad, livoxInterface, custom_car_boxes)

                for yolo_cls in range(13):
                    cls_results = position_results[yolo_cls]
                    for cls_result in cls_results:
                        communicator_cls = yolo_cls_to_communicator_cls[yolo_cls]
                        cls_global_position = cls_result["global_position"]
                        communicator.add_data(communicator_cls, cls_global_position[0], cls_global_position[1])

                #annotated_img = cv2.resize(annotated_img, (1006, 759))
                #cv2.imshow("Camera", annotated_img)
                img_map_to_rad = cv2.resize(img_map_to_rad, (800, 800))
                cv2.imshow("Camera mapToRad", img_map_to_rad)

            # 所有 cv2 窗口统一由主线程渲染（其他线程只负责生成图像），
            # 避免多线程同时调用 cv2.imshow/waitKey 导致 OpenCV 高GUI 卡死/崩溃
            map_visualization = communicator.get_latest_map_visualization()
            if map_visualization is not None:
                map_title, map_image = map_visualization
                cv2.imshow(map_title, map_image)
            info_image = communicator.get_latest_info_image()
            if info_image is not None:
                cv2.imshow("information_ui", info_image)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # 按下 ESC 或 q 退出
                break
    except KeyboardInterrupt:
        print("\n[main] 收到 Ctrl+C，正在安全退出...")
    finally:
        # 无论正常退出（ESC/q）还是异常退出（如 Ctrl+C），都按初始化相反的顺序释放资源,保证安全退出
        if communicator is not None:
            try:
                communicator.stop()
            except Exception as e:
                print(f"[main] 通信器释放失败: {e}")
        if camera is not None:
            try:
                camera.stop_grabbing()
            except Exception as e:
                print(f"[main] 相机释放失败: {e}")
        cv2.destroyAllWindows()
        if livoxInterface is not None:
            # 先停止 Livox 后台线程再 Uninit，避免线程与 C 侧资源释放竞争导致卡死/崩溃
            try:
                livoxInterface.stop()
            except Exception as e:
                print(f"[main] Livox 线程停止失败: {e}")
            if livoxThread is not None:
                try:
                    livoxThread.join(timeout=2)
                except Exception as e:
                    print(f"[main] Livox 线程等待失败: {e}")
            try:
                livoxInterface.pyif_uninit()
            except Exception as e:
                print(f"[main] Livox 释放失败: {e}")


if __name__ == "__main__":
    main()
