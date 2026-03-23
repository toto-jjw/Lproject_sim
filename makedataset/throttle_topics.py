import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from tf2_msgs.msg import TFMessage
from message_filters import Subscriber, ApproximateTimeSynchronizer


import bisect

class SyncThrottleNode(Node):
    def __init__(self):
        super().__init__('sync_throttle_node')
        self.left_sub = Subscriber(self, Image, '/stereo/left/rgb')
        self.right_sub = Subscriber(self, Image, '/stereo/right/rgb')
        self.depth_sub = Subscriber(self, Image, '/front_camera/depth/depth')

        self.left_pub = self.create_publisher(Image, '/stereo/left/rgb_throttled', 10)
        self.right_pub = self.create_publisher(Image, '/stereo/right/rgb_throttled', 10)
        self.depth_pub = self.create_publisher(Image, '/front_camera/depth/depth_throttled', 10)
        self.tf_pub = self.create_publisher(TFMessage, '/tf_throttled', 10)

        self.sync = ApproximateTimeSynchronizer(
            [self.left_sub, self.right_sub, self.depth_sub], queue_size=10, slop=0.05)
        self.sync.registerCallback(self.sync_callback)

        # TF 메시지 버퍼 (timestamp, idx, msg) 리스트
        self.tf_buffer = []  # (timestamp, idx, msg)
        self.tf_buffer_maxlen = 1000
        self.tf_buffer_idx = 0  # timestamp가 같을 때 정렬용
        self.create_subscription(TFMessage, '/tf', self.tf_callback, 10)

        # 1Hz publish 제어용
        self.last_pub_time = None
        # publish period in seconds (0.1 -> 10 Hz)
        self.publish_period = 0.1

    def get_odom_base_link_tf(self, ts, window=0.1):
        # ts와 window(±0.1초) 내의 TFMessage transforms 중 odom→base_link만 반환
        for t, idx, msg in self.tf_buffer:
            if abs(t - ts) <= window:
                for tf in msg.transforms:
                    if (tf.header.frame_id == 'odom' and tf.child_frame_id == 'base_link'):
                        filtered = TFMessage()
                        filtered.transforms = [tf]
                        return filtered
        return None

    def sync_callback(self, left, right, depth):
        # 동기화된 이미지의 timestamp 사용
        ts = left.header.stamp.sec + left.header.stamp.nanosec * 1e-9
        # 1Hz로만 publish
        if self.last_pub_time is None or ts - self.last_pub_time >= self.publish_period:
            self.left_pub.publish(left)
            self.right_pub.publish(right)
            self.depth_pub.publish(depth)
            # odom→base_link 변환만 publish
            tfmsg = self.get_odom_base_link_tf(ts)
            if tfmsg is not None:
                self.tf_pub.publish(tfmsg)
            self.last_pub_time = ts

    def get_merged_tf(self, ts, window=0.1):
        # ts와 window(±0.1초) 내의 모든 TFMessage transforms를 합쳐서 반환
        transforms = []
        for t, idx, msg in self.tf_buffer:
            if abs(t - ts) <= window:
                transforms.extend(msg.transforms)
        if transforms:
            merged = TFMessage()
            merged.transforms = transforms
            return merged
        return None

    def tf_callback(self, msg):
        # 버퍼에 저장 (가장 첫 transform의 timestamp 기준)
        if len(msg.transforms) > 0:
            t = msg.transforms[0].header.stamp
            ts = t.sec + t.nanosec * 1e-9
            bisect.insort(self.tf_buffer, (ts, self.tf_buffer_idx, msg))
            self.tf_buffer_idx += 1
            # 버퍼 크기 제한
            if len(self.tf_buffer) > self.tf_buffer_maxlen:
                self.tf_buffer.pop(0)

    def get_closest_tf(self, ts):
        # 버퍼에서 ts와 가장 가까운 TF 메시지 반환
        if not self.tf_buffer:
            return None
        # idx는 0, msg는 2
        idx = bisect.bisect_left(self.tf_buffer, (ts, -1))
        if idx == 0:
            return self.tf_buffer[0][2]
        if idx == len(self.tf_buffer):
            return self.tf_buffer[-1][2]
        before = self.tf_buffer[idx-1]
        after = self.tf_buffer[idx]
        if abs(before[0] - ts) <= abs(after[0] - ts):
            return before[2]
        else:
            return after[2]

def main(args=None):
    rclpy.init(args=args)
    node = SyncThrottleNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
