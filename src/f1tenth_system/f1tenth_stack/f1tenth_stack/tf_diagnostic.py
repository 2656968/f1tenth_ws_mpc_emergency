#!/usr/bin/env python3  
import rclpy  
from rclpy.node import Node  
from tf2_ros import TransformException  
from tf2_ros.buffer import Buffer  
from tf2_ros.transform_listener import TransformListener  
import tf_transformations  
import numpy as np  
import time  
  
class TFDiagnostic(Node):  
    def __init__(self):  
        super().__init__('tf_diagnostic')  
        self.tf_buffer = Buffer()  
        self.tf_listener = TransformListener(self.tf_buffer, self)  
          
        # 진단할 TF 체인  
        self.tf_chains = [  
            ('map', 'odom'),  
            ('odom', 'base_link'),  
            ('base_link', 'laser'),  
            ('base_link', 'imu_link'),  
            ('map', 'base_link'),  
            ('map', 'laser')  
        ]  
          
        # 2초 대기 후 진단 시작  
        self.timer = self.create_timer(2.0, self.diagnose_once)  
          
    def diagnose_once(self):  
        self.timer.cancel()  
          
        print("\n" + "="*80)  
        print("TF 진단 시작")  
        print("="*80 + "\n")  
          
        # 1. 모든 TF 체인 확인  
        print("1. TF 체인 상태 확인:")  
        print("-" * 80)  
          
        for parent, child in self.tf_chains:  
            try:  
                trans = self.tf_buffer.lookup_transform(  
                    parent, child, rclpy.time.Time())  
                  
                # Translation  
                t = trans.transform.translation  
                print(f"\n✓ {parent} → {child}:")  
                print(f"  Translation: [{t.x:.3f}, {t.y:.3f}, {t.z:.3f}]")  
                  
                # Rotation (쿼터니언 → 오일러)  
                r = trans.transform.rotation  
                euler = tf_transformations.euler_from_quaternion([r.x, r.y, r.z, r.w])  
                roll_deg = np.degrees(euler[0])  
                pitch_deg = np.degrees(euler[1])  
                yaw_deg = np.degrees(euler[2])  
                  
                print(f"  Quaternion: [{r.x:.3f}, {r.y:.3f}, {r.z:.3f}, {r.w:.3f}]")  
                print(f"  Euler (deg): Roll={roll_deg:.1f}, Pitch={pitch_deg:.1f}, Yaw={yaw_deg:.1f}")  
                  
                # 경고 체크  
                if abs(roll_deg) > 5 or abs(pitch_deg) > 5:  
                    print(f"  ⚠️  경고: Roll/Pitch가 5도 이상 기울어져 있습니다!")  
                  
            except TransformException as ex:  
                print(f"\n✗ {parent} → {child}: 변환 없음")  
                print(f"  오류: {ex}")  
          
        # 2. 라이다 스캔 방향 분석  
        print("\n\n2. 라이다 스캔 설정 확인:")  
        print("-" * 80)  
          
        from sensor_msgs.msg import LaserScan  
        scan_received = [False]  
          
        def scan_callback(msg):  
            if not scan_received[0]:  
                scan_received[0] = True  
                print(f"  Frame ID: {msg.header.frame_id}")  
                print(f"  Angle Min: {msg.angle_min:.3f} rad ({np.degrees(msg.angle_min):.1f}°)")  
                print(f"  Angle Max: {msg.angle_max:.3f} rad ({np.degrees(msg.angle_max):.1f}°)")  
                print(f"  Angle Increment: {msg.angle_increment:.6f} rad")  
                  
                if msg.angle_increment > 0:  
                    print(f"  스캔 방향: 반시계방향 (CCW)")  
                else:  
                    print(f"  스캔 방향: 시계방향 (CW)")  
                  
                fov = msg.angle_max - msg.angle_min  
                print(f"  FOV: {np.degrees(fov):.1f}°")  
                  
                if abs(fov - 2*np.pi) < 0.1:  
                    print(f"  ✓ 360도 스캔 (RPLIDAR S3)")  
                elif abs(fov - 4.71) < 0.1:  
                    print(f"  ✓ 270도 스캔 (Hokuyo)")  
          
        scan_sub = self.create_subscription(LaserScan, '/scan', scan_callback, 1)  
          
        # 스캔 데이터 대기  
        timeout = 3.0  
        start = time.time()  
        while not scan_received[0] and (time.time() - start) < timeout:  
            rclpy.spin_once(self, timeout_sec=0.1)  
          
        if not scan_received[0]:  
            print("  ✗ 라이다 스캔 데이터를 받지 못했습니다!")  
          
        # 3. 오도메트리 설정 확인  
        print("\n\n3. 오도메트리 설정 확인:")  
        print("-" * 80)  
          
        from nav_msgs.msg import Odometry  
        odom_received = [False]  
          
        def odom_callback(msg):  
            if not odom_received[0]:  
                odom_received[0] = True  
                print(f"  Frame ID: {msg.header.frame_id}")  
                print(f"  Child Frame ID: {msg.child_frame_id}")  
                  
                pos = msg.pose.pose.position  
                print(f"  Position: [{pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}]")  
                  
                ori = msg.pose.pose.orientation  
                euler = tf_transformations.euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])  
                yaw_deg = np.degrees(euler[2])  
                print(f"  Yaw: {yaw_deg:.1f}°")  
                  
                vel = msg.twist.twist.linear  
                print(f"  Linear Velocity: [{vel.x:.3f}, {vel.y:.3f}, {vel.z:.3f}]")  
          
        odom_sub = self.create_subscription(Odometry, '/odom', odom_callback, 1)  
          
        # 오도메트리 데이터 대기  
        start = time.time()  
        while not odom_received[0] and (time.time() - start) < timeout:  
            rclpy.spin_once(self, timeout_sec=0.1)  
          
        if not odom_received[0]:  
            print("  ✗ 오도메트리 데이터를 받지 못했습니다!")  
          
        # 4. 파티클 필터 설정 확인  
        print("\n\n4. 파티클 필터 설정 확인:")  
        print("-" * 80)  
          
        pf_odom_received = [False]  
          
        def pf_odom_callback(msg):  
            if not pf_odom_received[0]:  
                pf_odom_received[0] = True  
                print(f"  Frame ID: {msg.header.frame_id}")  
                  
                pos = msg.pose.pose.position  
                print(f"  Estimated Position: [{pos.x:.3f}, {pos.y:.3f}]")  
                  
                ori = msg.pose.pose.orientation  
                euler = tf_transformations.euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])  
                yaw_deg = np.degrees(euler[2])  
                print(f"  Estimated Yaw: {yaw_deg:.1f}°")  
          
        pf_sub = self.create_subscription(Odometry, '/pf/pose/odom', pf_odom_callback, 1)  
          
        # 파티클 필터 데이터 대기  
        start = time.time()  
        while not pf_odom_received[0] and (time.time() - start) < timeout:  
            rclpy.spin_once(self, timeout_sec=0.1)  
          
        if not pf_odom_received[0]:  
            print("  ✗ 파티클 필터 데이터를 받지 못했습니다!")  
          
        # 5. 문제 요약  
        print("\n\n5. 문제 요약 및 권장 사항:")  
        print("="*80)  
          
        # TF 체인 완전성 체크  
        missing_tfs = []  
        for parent, child in [('map', 'odom'), ('odom', 'base_link'), ('base_link', 'laser')]:  
            try:  
                self.tf_buffer.lookup_transform(parent, child, rclpy.time.Time())  
            except:  
                missing_tfs.append(f"{parent} → {child}")  
          
        if missing_tfs:  
            print(f"\n⚠️  누락된 TF: {', '.join(missing_tfs)}")  
            print("   → VESC publish_tf 설정 또는 파티클 필터 child_frame_id 확인 필요")  
          
        # 라이다 방향 체크  
        try:  
            trans = self.tf_buffer.lookup_transform('base_link', 'laser', rclpy.time.Time())  
            r = trans.transform.rotation  
            euler = tf_transformations.euler_from_quaternion([r.x, r.y, r.z, r.w])  
            yaw_deg = np.degrees(euler[2])  
              
            if abs(yaw_deg) > 10:  
                print(f"\n⚠️  라이다 yaw 회전: {yaw_deg:.1f}°")  
                print(f"   → bringup_launch.py에서 yaw 보정 필요")  
        except:  
            pass  
          
        print("\n진단 완료!")  
        print("="*80 + "\n")  
          
        # 종료  
        self.destroy_node()  
        rclpy.shutdown()  
  
def main():  
    rclpy.init()  
    node = TFDiagnostic()  
    rclpy.spin(node)  
  
if __name__ == '__main__':  
    main()