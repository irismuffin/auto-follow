#!/bin/bash

# 指定图片目录和输出的 XML 文件名
IMAGE_DIR="/home/edu/10.29/picture"
OUTPUT_FILE="picture.xml"


# 创建 XML 文件并写入头部
echo '<?xml version="1.0"?>' > "$OUTPUT_FILE"
echo '<opencv_storage>' >> "$OUTPUT_FILE"
echo '    <images>' >> "$OUTPUT_FILE"

# 遍历所有图片文件
for img in "$IMAGE_DIR"/*.{jpg,jpeg,png,bmp,gif}; do
    if [ -f "$img" ]; then
        # 写入图片路径
        echo "        \"$img\"" >> "$OUTPUT_FILE"
    fi
done

# 结束 XML 标签
echo '    </images>' >> "$OUTPUT_FILE"
echo '</opencv_storage>' >> "$OUTPUT_FILE"

