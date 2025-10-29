#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist

class SimpleTeleop(Node):
    def __init__(self):
        super().__init__('simple_teleop')
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.subscription = self.create_subscription(
            Joy,
            'joy',
            self.joy_callback,
            10)

    def joy_callback(self, msg):
        twist = Twist()
        twist.linear.x = msg.axes[1]
        twist.angular.z = msg.axes[0]
        self.publisher_.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    simple_teleop = SimpleTeleop()
    rclpy.spin(simple_teleop)
    simple_teleop.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
