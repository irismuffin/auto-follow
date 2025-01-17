#!/bin/bash

# 串口设备名称
SERIAL_PORT="/dev/ttyUSB0"  # 根据你的串口设备进行修改，如ttyUSB0等

# 检查串口设备是否存在
if [ -e "$SERIAL_PORT" ]; then
    echo "串口设备 $SERIAL_PORT 存在。"

    # 检查串口是否正在使用
    if fuser "$SERIAL_PORT" > /dev/null 2>&1; then
        echo "串口 $SERIAL_PORT 已打开。"
    else
        echo "串口 $SERIAL_PORT 没有被打开。"
    fi
else
    echo "串口设备 $SERIAL_PORT 不存在。"
fi

