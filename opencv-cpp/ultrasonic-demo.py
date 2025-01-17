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
triger_pin = 13 # BOARD 编码 29
echo_pin = 11 # BOARD 编码 31

# 禁用警告信息
GPIO.setwarnings(False)

def main():
    # 设置管脚编码模式为硬件编号 BOARD
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(triger_pin, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(echo_pin, GPIO.IN)

    # 防抖配置
    window_size = 5  # 设置滑动窗口大小，即缓存最近的5个距离值
    distance_history = []  # 用于存储最近的距离值

   # print("Starting demo now! Press CTRL+C to exit\n")
    try:
        while True:
            # print("echo_pin:",GPIO.input(echo_pin),"\n")
            GPIO.output(triger_pin, GPIO.HIGH)
            time.sleep(0.000015)
            GPIO.output(triger_pin, GPIO.LOW)
            # print("echo_pin:",GPIO.input(echo_pin),"\n")
            while GPIO.input(echo_pin) == 0:
                continue
            # print("wait_for_edge0:",GPIO.input(echo_pin),"\n")
            t1 = time.time_ns() # time start ns
            while GPIO.input(echo_pin) == 1:
                continue
            # print("wait_for_edge1:",GPIO.input(echo_pin),"\n")
            t2 = time.time_ns() # time start ns
            last = (t2 - t1)/1000000
            # print("last:",last,"ms\n")
            Distance=(last*0.346)/2 # m
            Distance_cm=Distance * 100 # cm						
            # print("distance:",Distance_cm,"cm\n")
            # 将当前的距离值加入历史数据列表
            distance_history.append(Distance_cm)

            # 保证历史数据不超过设定的窗口大小
            if len(distance_history) > window_size:
                distance_history.pop(0)

            # 计算滑动窗口中的平均值（去噪声）
            smoothed_distance = sum(distance_history) / len(distance_history)

            # 打印平滑后的距离值
            print(f"{smoothed_distance:.2f}")
            #print(Distance_cm)
            time.sleep(0.05)
    finally:
        GPIO.cleanup()  # cleanup all GPIOs



if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    main()
