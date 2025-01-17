#!/usr/bin/env python3
import sys
import signal
import Hobot.GPIO as GPIO
import time

def signal_handler(signal, frame):
    sys.exit(0)

# 定义使用的GPIO通道：
# 36号作为输出，可以点亮一个LED
# 38号作为输入，可以接一个按钮
triger_pin1 = 29 # BOARD 编码 29
echo_pin1 = 31 # BOARD 编码 31
triger_pin2 = 13
echo_pin2 = 11

# 禁用警告信息
GPIO.setwarnings(False)

def main():
    # 设置管脚编码模式为硬件编号 BOARD
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(triger_pin1, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(echo_pin1, GPIO.IN)
    GPIO.setup(triger_pin2, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(echo_pin2, GPIO.IN)
    # 防抖配置
    window_size = 5  # 设置滑动窗口大小，即缓存最近的5个距离值
    distance_history1 = []  # 用于存储最近的距离值
    distance_history2 = []  # 用于存储最近的距离值

   # print("Starting demo now! Press CTRL+C to exit\n")
    try:
            # print("echo_pin:",GPIO.input(echo_pin),"\n")
            GPIO.output(triger_pin1, GPIO.HIGH)
            time.sleep(0.000015)
            GPIO.output(triger_pin1, GPIO.LOW)
            # print("echo_pin:",GPIO.input(echo_pin),"\n")
            while GPIO.input(echo_pin1) == 0:
                continue
            # print("wait_for_edge0:",GPIO.input(echo_pin),"\n")
            t1 = time.time_ns() # time start ns
            while GPIO.input(echo_pin1) == 1:
                continue
            # print("wait_for_edge1:",GPIO.input(echo_pin),"\n")
            t2 = time.time_ns() # time start ns
            GPIO.output(triger_pin2, GPIO.HIGH)
            time.sleep(0.000015)
            GPIO.output(triger_pin2, GPIO.LOW)
            while GPIO.input(echo_pin2) == 0:
                continue
            # print("wait_for_edge0:",GPIO.input(echo_pin),"\n")
            t3 = time.time_ns() # time start ns
            while GPIO.input(echo_pin2) == 1:
                continue
            t4 = time.time_ns() # time start ns
            last1 = (t2 - t1)/1000000
            last2 = (t4 - t3)/1000000
            # print("last:",last,"ms\n")
            Distance1=(last1*0.346)/2 # m
            Distance1_cm=Distance1 * 100 # cm						
            # print("distance:",Distance_cm,"cm\n")
            # 将当前的距离值加入历史数据列表
            distance_history1.append(Distance1_cm)

            # 保证历史数据不超过设定的窗口大小
            if len(distance_history1) > window_size:
                distance_history1.pop(0)

            # 计算滑动窗口中的平均值（去噪声）
            smoothed_distance1 = sum(distance_history1) / len(distance_history1)
            Distance2=(last2*0.346)/2 # m
            Distance2_cm=Distance2 * 100 # cm						
            # print("distance:",Distance_cm,"cm\n")
            # 将当前的距离值加入历史数据列表
            distance_history2.append(Distance2_cm)

            # 保证历史数据不超过设定的窗口大小
            if len(distance_history2) > window_size:
                distance_history2.pop(0)

            # 计算滑动窗口中的平均值（去噪声）
            smoothed_distance2 = sum(distance_history2) / len(distance_history2)

            # 打印平滑后的距离值
            print(f"{smoothed_distance1:.2f} {smoothed_distance2:.2f}")
            # print(f"{smoothed_distance2:.2f}")
            #print(Distance_cm)
            time.sleep(0.05)
    finally:
        GPIO.cleanup()  # cleanup all GPIOs



if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    main()
