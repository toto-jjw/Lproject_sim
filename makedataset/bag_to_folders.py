import os
import numpy as np
import cv2
import rclpy
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from sensor_msgs.msg import Image
from tf2_msgs.msg import TFMessage

bag_path = "my_bag"
out_root = "sorted_by_timestamp2"
os.makedirs(out_root, exist_ok=True)

rclpy.init()
reader = SequentialReader()
storage_options = StorageOptions(uri=bag_path, storage_id='mcap')
converter_options = ConverterOptions('', '')
reader.open(storage_options, converter_options)

# 프레임별 파일명 저장용 dict
frame_files = {}

while reader.has_next():
    topic, data, t = reader.read_next()

    msg_sec = None
    msg_nsec = None
    if topic in ['/stereo/left/rgb_throttled', '/stereo/right/rgb_throttled', '/front_camera/depth/depth_throttled']:
        msg = deserialize_message(data, Image)
        msg_sec = msg.header.stamp.sec
        msg_nsec = msg.header.stamp.nanosec
        frame_key = f"{msg_sec}_{msg_nsec}"
        if frame_key not in frame_files:
            frame_files[frame_key] = {}
        if topic == '/stereo/left/rgb_throttled':
            fname = f"left_{frame_key}.png"
            frame_files[frame_key]['left'] = fname
            if msg.encoding == 'rgb8':
                img_np = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
                cv2.imwrite(os.path.join(out_root, fname), img_np)
            elif msg.encoding == 'mono8':
                img_np = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width))
                cv2.imwrite(os.path.join(out_root, fname), img_np)
        elif topic == '/stereo/right/rgb_throttled':
            fname = f"right_{frame_key}.png"
            frame_files[frame_key]['right'] = fname
            if msg.encoding == 'rgb8':
                img_np = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
                cv2.imwrite(os.path.join(out_root, fname), img_np)
            elif msg.encoding == 'mono8':
                img_np = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width))
                cv2.imwrite(os.path.join(out_root, fname), img_np)
        elif topic == '/front_camera/depth/depth_throttled':
            if msg.encoding == '16UC1':
                img_np = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width))
                fname = f"depth_{frame_key}.npy"
                frame_files[frame_key]['depth'] = fname
                np.save(os.path.join(out_root, fname), img_np)
            elif msg.encoding == '32FC1':
                img_np = np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width))
                fname = f"depth_{frame_key}.npy"
                frame_files[frame_key]['depth'] = fname
                np.save(os.path.join(out_root, fname), img_np)
            elif msg.encoding == 'mono8':
                img_np = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width))
                fname = f"depth_{frame_key}.png"
                frame_files[frame_key]['depth'] = fname
                cv2.imwrite(os.path.join(out_root, fname), img_np)
    elif topic == '/tf_throttled':
        msg = deserialize_message(data, TFMessage)
        if len(msg.transforms) > 0:
            msg_sec = msg.transforms[0].header.stamp.sec
            msg_nsec = msg.transforms[0].header.stamp.nanosec
            frame_key = f"{msg_sec}_{msg_nsec}"
            fname = f"tf_{frame_key}.txt"
            frame_files.setdefault(frame_key, {})['tf'] = fname
            with open(os.path.join(out_root, fname), "w") as f:
                f.write(str(msg))

# 프레임별 파일명 정리 텍스트 파일 생성
with open(os.path.join(out_root, "frame_index.txt"), "w") as f:
    for frame_key in sorted(frame_files.keys()):
        files = frame_files[frame_key]
        line = f"{frame_key}: " + ", ".join([files.get(k, "-") for k in ['left', 'right', 'depth', 'tf']])
        f.write(line + "\n")

rclpy.shutdown()