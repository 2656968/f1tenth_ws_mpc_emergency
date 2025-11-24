#!/usr/bin/env python3  
import rclpy  
from rclpy.node import Node  
from sensor_msgs.msg import LaserScan  
import numpy as np  
  
class LidarZeroFinder(Node):  
    def __init__(self):  
        super().__init__('lidar_zero_finder')  
        self.scan_sub = self.create_subscription(  
            LaserScan, '/scan', self.scan_callback, 10)  
        self.scan_received = False  
          
    def scan_callback(self, msg):  
        if self.scan_received:  
            return  
        self.scan_received = True  
          
        print("\n" + "="*80)  
        print("RPLIDAR S3 0도 기준점 찾기")  
        print("="*80 + "\n")  
          
        # 기본 정보  
        print("1. 라이다 스캔 기본 정보:")  
        print("-" * 80)  
        print(f"  Frame ID: {msg.header.frame_id}")  
        print(f"  Angle Min: {msg.angle_min:.4f} rad ({np.degrees(msg.angle_min):.1f}°)")  
        print(f"  Angle Max: {msg.angle_max:.4f} rad ({np.degrees(msg.angle_max):.1f}°)")  
        print(f"  Angle Increment: {msg.angle_increment:.6f} rad ({np.degrees(msg.angle_increment):.3f}°)")  
        print(f"  Total Points: {len(msg.ranges)}")  
        print(f"  Range Min: {msg.range_min:.2f}m")  
        print(f"  Range Max: {msg.range_max:.2f}m\n")  
          
        # 각도 배열 생성  
        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))  
        ranges = np.array(msg.ranges)  
          
        # 유효한 측정값만 필터링  
        valid_mask = (ranges > msg.range_min) & (ranges < msg.range_max)  
        valid_angles = angles[valid_mask]  
        valid_ranges = ranges[valid_mask]  
          
        print("2. 가장 가까운 물체 찾기 (0도 기준점 추정):")  
        print("-" * 80)  
          
        if len(valid_ranges) == 0:  
            print("  ✗ 유효한 측정값이 없습니다!")  
            self.destroy_node()  
            rclpy.shutdown()  
            return  
          
        # 가장 가까운 물체의 각도 찾기  
        min_idx = np.argmin(valid_ranges)  
        closest_angle = valid_angles[min_idx]  
        closest_distance = valid_ranges[min_idx]  
          
        print(f"  가장 가까운 물체:")  
        print(f"    거리: {closest_distance:.3f}m")  
        print(f"    각도: {closest_angle:.4f} rad ({np.degrees(closest_angle):.1f}°)")  
        print(f"    인덱스: {np.where(angles == closest_angle)[0][0]}/{len(msg.ranges)}\n")  
          
        # 0도 근처 물체 찾기 (±10도 범위)  
        zero_range = 10 * np.pi / 180  # 10도를 라디안으로  
        near_zero_mask = (valid_angles >= -zero_range) & (valid_angles <= zero_range)  
          
        if np.any(near_zero_mask):  
            near_zero_angles = valid_angles[near_zero_mask]  
            near_zero_ranges = valid_ranges[near_zero_mask]  
              
            print("3. 0도 근처(±10°) 물체:")  
            print("-" * 80)  
            for i, (angle, distance) in enumerate(zip(near_zero_angles, near_zero_ranges)):  
                print(f"    [{i+1}] 각도: {angle:.4f} rad ({np.degrees(angle):.1f}°), 거리: {distance:.3f}m")  
            print()  
          
        # 각 사분면별 최소 거리  
        print("4. 각 방향별 가장 가까운 물체:")  
        print("-" * 80)  
          
        directions = [  
            ("정면 (0°)", -10, 10),  
            ("왼쪽 (90°)", 80, 100),  
            ("뒤쪽 (180° 또는 -180°)", 170, 180),  
            ("오른쪽 (-90°)", -100, -80)  
        ]  
          
        for name, start_deg, end_deg in directions:  
            start_rad = start_deg * np.pi / 180  
            end_rad = end_deg * np.pi / 180  
              
            if start_deg > end_deg:  # 뒤쪽 처리  
                mask = (valid_angles >= start_rad) | (valid_angles <= end_rad)  
            else:  
                mask = (valid_angles >= start_rad) & (valid_angles <= end_rad)  
              
            if np.any(mask):  
                dir_ranges = valid_ranges[mask]  
                dir_angles = valid_angles[mask]  
                min_idx = np.argmin(dir_ranges)  
                print(f"  {name}:")  
                print(f"    거리: {dir_ranges[min_idx]:.3f}m")  
                print(f"    각도: {dir_angles[min_idx]:.4f} rad ({np.degrees(dir_angles[min_idx]):.1f}°)")  
            else:  
                print(f"  {name}: 측정값 없음")  
          
        print("\n" + "="*80)  
        print("5. 0도 기준점 찾는 방법:")  
        print("="*80)  
        print("  1) 라이다 정면에 박스나 벽을 약 0.5~1m 거리에 두세요")  
        print("  2) 이 스크립트를 다시 실행하세요")  
        print("  3) '가장 가까운 물체'의 각도를 확인하세요")  
        print("  4) 그 각도가 RPLIDAR의 0도 기준점입니다")  
        print("  5) TF yaw 보정값 = -(측정된 각도)")  
        print(f"\n  예: 측정 각도가 {np.degrees(closest_angle):.1f}°라면")  
        print(f"      TF yaw = {-closest_angle:.4f} rad ({-np.degrees(closest_angle):.1f}°)")  
        print("="*80 + "\n")  
          
        self.destroy_node()  
        rclpy.shutdown()  
  
def main():  
    rclpy.init()  
    node = LidarZeroFinder()  
    rclpy.spin(node)  
  
if __name__ == '__main__':  
    main()