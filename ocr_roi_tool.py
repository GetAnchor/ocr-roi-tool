import os
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'  # 禁止 PaddleOCR 模型来源检查
import sys
import cv2
import numpy as np
from paddleocr import PaddleOCR

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton,
    QFileDialog, QHBoxLayout, QVBoxLayout, QWidget,
    QTableWidget, QTableWidgetItem, QScrollArea
)
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen

# ---------------- OCR Engine ----------------
# 初始化 PaddleOCR，本地模型
ocr_engine = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

# 根据置信度返回框的颜色
def confidence_color(conf):
    if conf >= 0.9:
        return Qt.green
    elif conf >= 0.7:
        return Qt.yellow
    return Qt.red

# ---------------- ImageLabel ----------------
# 自定义 QLabel 用于显示图片、ROI 框选、缩放、绘制标注
class ImageLabel(QLabel):
    def __init__(self, main_window, scroll_area=None):
        super().__init__(main_window)
        self.main_window = main_window
        self.scroll_area = scroll_area
        self.setMouseTracking(True)

        # 原始图片和 ROI 信息
        self.image = None
        self.rois = []

        self.scale = 1.0  # 当前缩放比例
        self.start_pos = None  # 框选起点
        self.end_pos = None    # 框选终点
        self.setAlignment(Qt.AlignTop | Qt.AlignLeft)

    # 设置图片
    def set_image(self, img):
        self.image = img
        self.rois.clear()
        self.scale = 1.0
        self.update_display()

    # 清除 ROI
    def clear_rois(self):
        self.rois.clear()
        self.update_display()

    # 刷新显示
    def update_display(self):
        if self.image is None:
            return

        h, w = self.image.shape[:2]
        scaled_w = int(w * self.scale)
        scaled_h = int(h * self.scale)

        # OpenCV 转 QPixmap
        rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w*3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            scaled_w, scaled_h, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        # 创建画布并绘制图片
        canvas = QPixmap(pixmap.size())
        canvas.fill(Qt.black)
        painter = QPainter(canvas)
        painter.drawPixmap(0, 0, pixmap)

        # 绘制 ROI 框和文本
        for roi in self.rois:
            img_rect = roi["img_rect"]
            view_rect = QRect(
                int(img_rect.left() * self.scale),
                int(img_rect.top() * self.scale),
                int(img_rect.width() * self.scale),
                int(img_rect.height() * self.scale)
            )
            pen = QPen(confidence_color(roi["conf"]), 2)
            painter.setPen(pen)
            painter.drawRect(view_rect)
            painter.drawText(view_rect.left(), view_rect.top()-5,
                             f"{roi['text']} ({roi['conf']:.2f})")

        # 绘制正在框选的临时框
        if self.start_pos and self.end_pos:
            r = QRect(self.start_pos, self.end_pos).normalized()
            painter.setPen(QPen(Qt.blue, 1, Qt.DashLine))
            painter.drawRect(r)

        painter.end()
        self.setPixmap(canvas)
        self.resize(pixmap.size())

    # 鼠标按下：开始框选
    def mousePressEvent(self, event):
        if self.image is None:
            return
        if event.button() == Qt.LeftButton:
            self.start_pos = event.pos()
            self.end_pos = event.pos()

    # 鼠标移动：更新框选
    def mouseMoveEvent(self, event):
        if self.start_pos:
            self.end_pos = event.pos()
            self.update_display()

    # 鼠标释放：完成框选并 OCR
    def mouseReleaseEvent(self, event):
        if self.image is None or not self.start_pos:
            return

        self.end_pos = event.pos()
        rect = QRect(self.start_pos, self.end_pos).normalized()

        # 坐标映射：视图 → 图片原始坐标
        scale_x = self.image.shape[1] / self.pixmap().width()
        scale_y = self.image.shape[0] / self.pixmap().height()

        x1 = int(rect.left() * scale_x)
        y1 = int(rect.top() * scale_y)
        x2 = int(rect.right() * scale_x)
        y2 = int(rect.bottom() * scale_y)

        if x2 - x1 < 5 or y2 - y1 < 5:
            self.start_pos = None
            self.update_display()
            return

        roi_img = self.image[y1:y2, x1:x2]
        if roi_img.size == 0:
            self.start_pos = None
            self.update_display()
            return

        # OCR
        result = ocr_engine.predict(roi_img)[0]
        rec_texts = result.get("rec_texts", [])
        rec_scores = result.get("rec_scores", [])
        text = "\n".join(rec_texts)
        conf = float(np.min(rec_scores)) if rec_scores else 0.0

        self.rois.append({
            "img_rect": QRect(x1, y1, x2-x1, y2-y1),
            "text": text,
            "conf": conf
        })

        self.start_pos = None
        self.update_display()
        self.main_window.update_table()

    # 缩放，保持可视区域中心
    def zoom(self, factor):
        if self.image is None or self.scroll_area is None:
            return
        hbar = self.scroll_area.horizontalScrollBar()
        vbar = self.scroll_area.verticalScrollBar()
        viewport_width = self.scroll_area.viewport().width()
        viewport_height = self.scroll_area.viewport().height()

        center_x_img = (hbar.value() + viewport_width/2) / self.scale
        center_y_img = (vbar.value() + viewport_height/2) / self.scale

        self.scale *= factor
        self.scale = max(0.2, min(5.0, self.scale))

        hbar.setValue(int(center_x_img * self.scale - viewport_width/2))
        vbar.setValue(int(center_y_img * self.scale - viewport_height/2))

        self.update_display()

    # 双击缩放：左键放大，右键缩小
    def mouseDoubleClickEvent(self, event):
        if self.image is None or self.scroll_area is None:
            return

        click_pos = event.pos()
        x_img = click_pos.x() / self.scale
        y_img = click_pos.y() / self.scale

        if event.button() == Qt.LeftButton:
            factor = 1.25
        elif event.button() == Qt.RightButton:
            factor = 0.8
        else:
            return

        self.scale *= factor
        self.scale = max(0.2, min(5.0, self.scale))

        viewport_width = self.scroll_area.viewport().width()
        viewport_height = self.scroll_area.viewport().height()
        hbar = self.scroll_area.horizontalScrollBar()
        vbar = self.scroll_area.verticalScrollBar()

        hbar.setValue(int(x_img * self.scale - viewport_width/2))
        vbar.setValue(int(y_img * self.scale - viewport_height/2))
        self.update_display()

# ---------------- MainWindow ----------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OCR ROI 评估工具")

        # 图片显示区域
        self.scroll_area = QScrollArea()
        self.image_label = ImageLabel(self, scroll_area=self.scroll_area)
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(True)

        # OCR 结果表格
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["ID","Text","Confidence","Area(px)"])
        self.table.setWordWrap(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setTextElideMode(Qt.ElideNone)

        # 按钮
        open_btn = QPushButton("打开图片")
        clear_btn = QPushButton("清除标注")
        zoom_in_btn = QPushButton("放大")
        zoom_out_btn = QPushButton("缩小")

        open_btn.clicked.connect(self.open_image)
        clear_btn.clicked.connect(self.clear_rois)
        zoom_in_btn.clicked.connect(lambda: self.image_label.zoom(1.25))
        zoom_out_btn.clicked.connect(lambda: self.image_label.zoom(0.8))

        # 布局
        left = QVBoxLayout()
        left.addWidget(self.scroll_area)
        left.addWidget(open_btn)
        left.addWidget(clear_btn)
        left.addWidget(zoom_in_btn)
        left.addWidget(zoom_out_btn)

        main = QHBoxLayout()
        main.addLayout(left, 3)
        main.addWidget(self.table, 2)

        container = QWidget()
        container.setLayout(main)
        self.setCentralWidget(container)

    # 打开图片
    def open_image(self):
        path,_ = QFileDialog.getOpenFileName(self,"选择图片","","Images (*.png *.jpg *.jpeg)")
        if not path: return
        img = cv2.imread(path)
        self.image_label.set_image(img)
        self.table.setRowCount(0)

    # 清除标注
    def clear_rois(self):
        self.image_label.clear_rois()
        self.table.setRowCount(0)

    # 更新表格
    def update_table(self):
        self.table.setRowCount(len(self.image_label.rois))
        for i, roi in enumerate(self.image_label.rois):
            rect = roi["img_rect"]
            area = rect.width()*rect.height()
            self.table.setItem(i,0,QTableWidgetItem(str(i+1)))
            self.table.setItem(i,1,QTableWidgetItem(roi["text"]))
            self.table.setItem(i,2,QTableWidgetItem(f"{roi['conf']:.3f}"))
            self.table.setItem(i,3,QTableWidgetItem(str(area)))
        self.table.resizeRowsToContents()

# ---------------- Run ----------------
if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1200,800)
    win.show()
    sys.exit(app.exec_())
