# OCR ROI 评估工具

一个基于 **PyQt5 + PaddleOCR** 的本地图片文本识别工具，支持：

- 图片 ROI 框选识别
- 实时 OCR 识别结果显示
- 表格显示文字及置信度
- 放大/缩小/滚动查看图片
- 双击鼠标缩放（左键放大，右键缩小）
- 多次 ROI 框选评估不同区域 OCR 效果

---

## 安装

```bash
git clone <repo-url>
cd <repo-dir>
pip install pyqt5 opencv-python paddleocr numpy

## 安装Paddle框架
https://www.paddleocr.ai/latest/version3.x/installation.html
