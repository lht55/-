import time, cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QFileDialog, QMainWindow
from PyQt5.QtCore import QTimer, QDateTime
from PyQt5.Qt import QThread, pyqtSignal, QMutex
import qimage2ndarray
from queue import Queue
from Project import Ui_Form
from infer import *

video_steam = Queue()

# Create an argument parser to handle command-line argumentss
def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='best.onnx', help='Input your ONNX model.')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    args = parser.parse_args()
    # Check the requirements and select the appropriate backend (CPU or GPU)
    check_requirements('onnxruntime-gpu' if torch.cuda.is_available() else 'onnxruntime')
    return args

class Thread1(QThread):        # 线程1
    # 使用自定义信号，一定要记得信号是类变量，必须在类中定义，不能在实例方法中定义，
    thread1_signal2 = pyqtSignal(object)   #定义信号,定义参数为object类型

    def __init__(self):
        super(Thread1, self).__init__()
        self.t = 0
        self.Image_thread1 = None
        self.mutex = QMutex()            # 创建线程锁
        self._isPause = False

    def run(self):
        while True:
            self.mutex.lock()            # 加锁
            time.sleep(0.001)            # 休眠
            _, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.Image_thread1 = frame
            self.thread1_signal2.emit(self.Image_thread1)    # 传信号
            self.mutex.unlock()          # 解锁


class Thread2(QThread):         # 线程2
    thread1_signal3 = pyqtSignal(object)

    def __init__(self):
        super(Thread2, self).__init__()
        self.t = 0
        self.Image_thread1 = None
        self.mutex = QMutex()
        self._isPause = False
        self.video_flag = 0

    def run(self):
        while True:
            self.mutex.lock()
            time.sleep(0.001)
            # time.sleep(1 / self.a)  在这里改帧率
            _, frame = cap.read()
            args = getArgs()
            # Create an instance of the Yolov8 class with the specified arguments
            detection = Yolov8(args.model, frame, args.conf_thres, args.iou_thres)
            # Perform object detection and obtain the output image
            frame = detection.main()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.video_flag == 0:
                video_steam.put(frame)
            self.Image_thread1 = frame
            self.thread1_signal3.emit(self.Image_thread1)
            self.mutex.unlock()


class PyQtMainEntry(QMainWindow, Ui_Form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.mark = 0
        self.btnReadImage.clicked.connect(self.btnReadImage_Clicked)
        self.btnShowCamera.clicked.connect(self.btnOpenCamera_Clicked)
        self.btnStartLabel.clicked.connect(self.startRecognize)
        self.btnSaveResult.clicked.connect(self.resultsave)
        self.time()


    def btnReadImage_Clicked(self):
        self.mark = 0
        filename,  _ = QFileDialog.getOpenFileName(self, '打开')
        if filename:
            self.captured = cv2.imread(str(filename))
            self.captured = cv2.cvtColor(self.captured, cv2.COLOR_BGR2RGB)
            rows, cols, channels = self.captured.shape
            bytesPerLine = channels * cols
            QImg = QImage(self.captured.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
            self.Videolabel.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.Videolabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.Videolabel.setScaledContents(True)


    def btnOpenCamera_Clicked(self):
        self.mark = 1
        self.process_thread = Thread2()
        self.process_thread.thread1_signal3.connect(self.thread2_work2)
        self.preview_thread = Thread1()
        self.preview_thread.thread1_signal2.connect(self.thread1_work2)
        self.preview_thread.start()

    def thread1_work2(self, img):
        self.Image = img
        qimg = qimage2ndarray.array2qimage(img)
        self.Videolabel.setPixmap(QPixmap(qimg))
        self.Videolabel.show()
        self.Videolabel.setScaledContents(True)

    def thread2_work2(self, img):
        self.Image = img
        qimg = qimage2ndarray.array2qimage(img)
        self.DetectImagelabel.setPixmap(QPixmap(qimg))
        self.DetectImagelabel.show()
        self.DetectImagelabel.setScaledContents(True)
        self.Videolabel_2.setPixmap(QPixmap(qimg))
        self.Videolabel_2.show()
        self.Videolabel_2.setScaledContents(True)

    def startRecognize(self):
        if self.mark == 0:
            img = self.captured
            args = getArgs()
            # Create an instance of the Yolov8 class with the specified arguments
            detection = Yolov8(args.model, img, args.conf_thres, args.iou_thres)
            # Perform object detection and obtain the output image
            draw_1 = detection.main()
            self.result = draw_1
            draw_2 = qimage2ndarray.array2qimage(draw_1)
            self.DetectImagelabel.setPixmap(QPixmap(draw_2))
            self.DetectImagelabel.setScaledContents(True)
            self.DetectImagelabel.show()
        else:
            self.process_thread.start()


    # 显示时间
    def showCurrentTime(self, timeLabel):
        time = QDateTime.currentDateTime()
        self.timeDisplay = time.toString('yyyy-MM-dd hh:mm:ss dddd')
        timeLabel.setText(self.timeDisplay)

    def time(self):
        self.timer = QTimer()
        self.timer.timeout.connect(lambda: self.showCurrentTime(self.label_2))
        self.timer.start()

    def resultsave(self):
        path_filename = QFileDialog.getExistingDirectory(self, '结果保存')
        if path_filename:
            self.saveImage = cv2.cvtColor(self.result, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path_filename + '/' + self.timeDisplay[:10]
                       + '_' + str(10) + '.png', self.saveImage)
        # 显示路径
        self.PathLineEdit.setText(path_filename)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    cap = cv2.VideoCapture(0)
    window = PyQtMainEntry()
    window.show()
    sys.exit(app.exec_())
