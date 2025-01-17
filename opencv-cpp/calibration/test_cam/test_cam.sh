#!/bin/bash

# 设置保存图片的目录
SAVE_DIR="pics"
# 设置保存的图片数量
NUM_IMAGES=10
# 设置保存图片的文件格式
IMG_FORMAT="jpg"
# 图片文件列表的 XML 文件
XML_FILE="pics_list.xml"

# 创建保存图片的目录
mkdir -p $SAVE_DIR

# 初始化 XML 文件
echo '<?xml version="1.0"?>' > $XML_FILE
echo '<opencv_storage>' >> $XML_FILE
echo '  <images>' >> $XML_FILE

# 使用 ffmpeg 从摄像头捕获图片
for ((i=1; i<=NUM_IMAGES; i++))
do
    # 生成图片文件名
    IMAGE_PATH="${SAVE_DIR}/image_${i}.${IMG_FORMAT}"

    # 使用 ffmpeg 从 /dev/video8 捕获一帧并保存为 jpg 格式
    ffmpeg -f v4l2 -framerate 30 -video_size 640x480 -i /dev/video8 -vframes 1 "$IMAGE_PATH"

    # 检查图片是否成功保存
    if [[ -f "$IMAGE_PATH" ]]; then
        echo "Captured image: $IMAGE_PATH"
        # 将图片路径添加到 XML 文件
        echo "    <image>$IMAGE_PATH</image>" >> $XML_FILE
    else
        echo "Error: Failed to capture image $i"
    fi
done

# 结束 XML 文件
echo '  </images>' >> $XML_FILE
echo '</opencv_storage>' >> $XML_FILE

# 完成输出
echo "Capture complete. $NUM_IMAGES images saved to '$SAVE_DIR' and picture list saved to '$XML_FILE'."

