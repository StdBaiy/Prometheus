#!/usr/bin/env python3
"""
An example that uses TensorRT's Python api to make inferences.
"""
import ctypes
import sys
import os
path = sys.path[0]
path = path + '/siam_rpn_lib/'
sys.path.insert(0,'/opt/ros/' + os.environ['ROS_DISTRO'] + '/lib/python3/dist-packages/')
from cv_bridge import CvBridge
sys.path.append(path)
import random
import sys
import threading
import time
import math
import socket
import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import argparse
import torch
import torchvision
from threading import Lock

from net import SiamRPNvot
from run_SiamRPN import SiamRPN_init, SiamRPN_track
from siam_utils import get_axis_aligned_bbox, cxy_wh_2_rect
import argparse

# 此处接受启动命令的参数
parser = argparse.ArgumentParser()
parser.add_argument("--image_topic", default=None, help="from ROS image topic get image data")
parser.add_argument('--no_tcp', action='store_const',
                    const=False, default=True,
                    help='whether to send tcp data')

args = parser.parse_args()

g_image = None
getim = None
image_lock = Lock()
if args.image_topic:
    def callback(data):
        global g_image, getim, image_lock
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
        with image_lock:
            g_image = cv_image   
            getim = True

    import rospy
    from sensor_msgs.msg import Image
    rospy.init_node("yolov5_tracker", anonymous=True)
    rospy.Subscriber(args.image_topic, Image, callback)

ROS_SERVER_ON = args.no_tcp
NAMES_TXT = 'coco'

# 预设的图片信息
INPUT_W = 608
INPUT_H = 608
CONF_THRESH = 0.25
IOU_THRESHOLD = 0.4
global g_x, g_y, g_clicked, track_start
g_x = 0
g_y = 0
g_clicked = False
track_start = False


# 开始监听
if ROS_SERVER_ON:
    socket_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    socket_server.bind(('', 9091))
    socket_server.listen(5)
    print('Start listening on port 9091 ...')
    client_socket, client_address = socket_server.accept()
    print('Got my client')


def on_mouse(event, x, y, flags, param):
    global g_x, g_y, g_clicked, track_start
    if event == cv2.EVENT_LBUTTONDOWN:
        # print("x: {}, y: {}".format(x, y))
        g_x = x
        g_y = y
        g_clicked = True
    if event == cv2.EVENT_RBUTTONDOWN:
        track_start = False

# 新建一个窗口，播放摄像头数据，并显示识别框，提供点击功能
WINDOW_NAME = 'tensorrt-yolov5'
cv2.namedWindow(WINDOW_NAME)  # flags = cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO
cv2.setMouseCallback(WINDOW_NAME, on_mouse)


def load_class_desc(dataset='coco'):
    """
    载入class_desc文件夹中的类别信息，txt文件的每一行代表一个类别
    :param dataset: str 'coco'
    :return: list ['cls1', 'cls2', ...]
    """
    desc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'class_desc')
    desc_names = []
    for f in os.listdir(desc_dir):
        if f.endswith('.txt'):
            desc_names.append(os.path.splitext(f)[0])
    # 如果类别描述文件存在，则返回所有类别名称，否则会报错
    cls_names = []
    if dataset in desc_names:
        with open(os.path.join(desc_dir, dataset + '.txt')) as f:
            for line in f.readlines():
                if len(line.strip()) > 0:
                    cls_names.append(line.strip())
    else:
        raise NameError('{}.txt not exist in "class_desc"'.format(dataset))
    # 类别描述文件不能为空，否则会报错
    if len(cls_names) > 0:
        return cls_names
    else:
        raise RuntimeError('{}.txt is EMPTY'.format(dataset))


def plot_one_box(x, img, color=(0,255,0), label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    param: 
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return

    """
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


class YoLov5TRT(object):
    """
    description: A YOLOv5 class that warps TensorRT ops, preprocess and postprocess ops.
    """

    def __init__(self, engine_file_path):
        # Create a Context on this device,
        self.cfx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.fps = 0
        self.tc = 0.

        print('loading SiamPRN model...........')
        self.siamnet = SiamRPNvot()
        self.siamnet.load_state_dict(torch.load(path + 'SiamRPNVOT.model'))
        self.siamnet.eval().cuda()
        z = torch.Tensor(1, 3, 127, 127)
        #self.siamnet.temple(z.cuda())
        x = torch.Tensor(1, 3, 271, 271)
        #self.siamnet(x.cuda())
        self.cls_names = load_class_desc(NAMES_TXT)
        self.frame_cnt = 0

    # 预测函数，输入图像帧，把预测结果发给客户端
    def infer(self, input_image_frame):
        global g_x, g_y, g_clicked, track_start, g_label
        t0 = time.time()
        
        threading.Thread.__init__(self)
        # Make self the active context, pushing it on top of the context stack.
        self.cfx.push()
        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings

        sent_it = False
        self.frame_cnt += 1
        if not track_start:
            ##############这部分是接受图像到返回检测框和预测类别+概率的整个过程###################
            # Do image preprocess
            input_image, image_raw, origin_h, origin_w = self.preprocess_image(
                input_image_frame
            )
            # Copy input image to host buffer
            np.copyto(host_inputs[0], input_image.ravel())
            # Transfer input data  to the GPU.
            cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
            # Run inference.
        
            context.execute_async(bindings=bindings, stream_handle=stream.handle)
            # Transfer predictions back from the GPU.
            cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
            # Synchronize the stream
            stream.synchronize()
            # Remove any context from the top of the context stack, deactivating it.
            self.cfx.pop()
            # Here we use the first row of output in that batch_size = 1
            output = host_outputs[0]
            # Do postprocess
            result_boxes, result_scores, result_classid = self.post_process(
                output, origin_h, origin_w
            )
            ##############################################################################

            box_clicked = []
            dist = 1e9
            # Draw rectangles and labels on the original image
            # 逐个检测预测框
            for i in range(len(result_boxes)):
                box = result_boxes[i]
                if box[0] < 0: box[0] = 0
                if box[1] < 0: box[1] = 0
                if box[2] < 0: box[2] = 0
                if box[3] < 0: box[3] = 0
                plot_one_box(
                    box,
                    image_raw,
                    label="{}:{:.2f}".format(self.cls_names[int(result_classid[i])], result_scores[i]),
                )
                if ROS_SERVER_ON:
                    # FrameID, 是否检测到目标(0/1,>1:num-of-objs), obj-order, 类别, x (0-1), y (0-1), w (0-1), h (0-1), 置信度, 0:detecting-mode
                    # 按照如下格式发送数据，实际上是字符串
                    client_socket.send('{:08d},{:03d},{:03d},{:03d},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:04d},{:04d},0'.format(
                        self.frame_cnt,
                        len(result_boxes),
                        i,
                        int(result_classid[i]),
                        box[0] / image_raw.shape[1],
                        box[1] / image_raw.shape[0],
                        (box[2]-box[0]) / image_raw.shape[1],
                        (box[3]-box[1]) / image_raw.shape[0],
                        result_scores[i],
                        int((box[0]+box[2])/2),
                        int((box[1]+box[3])/2)).encode('utf-8'))
                    sent_it = True

                # 如果点击了该预测框
                if g_clicked and (g_x>box[0] and g_y>box[1] and g_x<box[2] and g_y<box[3]):
                    cx = (box[0] + box[2]) / 2
                    cy = (box[1] + box[3]) / 2
                    d = math.sqrt((cx-g_x)**2 + (cy-g_y)**2)
                    if d < dist:
                        dist = d
                        # 输入被点击的预测框
                        box_clicked = box
                        g_label = int(result_classid[i])

            if g_clicked:
                if dist < 1e8:
                    cx = (box_clicked[0] + box_clicked[2]) / 2
                    cy = (box_clicked[1] + box_clicked[3]) / 2
                    print("x: {}, y: {}, box-cx: {}, box-cy: {}".format(g_x, g_y, cx, cy))
                    target_pos = np.array([int(cx), int(cy)])
                    target_sz = np.array([int(box_clicked[2]-box_clicked[0]), int(box_clicked[3]-box_clicked[1])])
                    self.state = SiamRPN_init(input_image_frame, target_pos, target_sz, self.siamnet)
                    track_start = True
                g_clicked = False

            if ROS_SERVER_ON and not sent_it:  # 发送无检测信息标志位
                client_socket.send('{:08d},000,000,000,0.000,0.000,0.000,0.000,0.000,0000,0000,0'.format(self.frame_cnt).encode('utf-8'))

        else:
            self.state = SiamRPN_track(self.state, input_image_frame)  # track
            res = cxy_wh_2_rect(self.state['target_pos'], self.state['target_sz'])
            res = [int(l) for l in res]
            image_raw = input_image_frame
            cv2.rectangle(image_raw, (res[0], res[1]), (res[0] + res[2], res[1] + res[3]), (0, 255, 255), 2)
            if ROS_SERVER_ON:
                # FrameID, 是否检测到目标(0/1), 类别, x (0-1), y (0-1), w (0-1), h (0-1), 置信度, 1:tracking-mode
                client_socket.send('{:08d},001,000,{:03d},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:04d},{:04d},1'.format(
                    self.frame_cnt,
                    int(g_label),
                    res[0] / image_raw.shape[1],
                    res[1] / image_raw.shape[0],
                    res[2] / image_raw.shape[1],
                    res[3] / image_raw.shape[0],
                    self.state['score'],
                    int(res[0]+res[2]/2),
                    int(res[1]+res[3]/2)).encode('utf-8'))
                sent_it = True

            if ROS_SERVER_ON and not sent_it:  # 发送无检测信息标志位
                client_socket.send('{:08d},000,000,000,0.000,0.000,0.000,0.000,0.000,0000,0000,1'.format(self.frame_cnt).encode('utf-8'))

        cv2.imshow(WINDOW_NAME, image_raw)
        cv2.waitKey(1)
        self.fps += 1
        self.tc += (time.time() - t0)
        if self.tc >= 1.:
            print('FPS:{}'.format(self.fps))
            self.fps = 0
            self.tc = 0.

        # parent, filename = os.path.split(input_image_path)
        # save_name = os.path.join(parent, "output_" + filename)
        # # 　Save image
        # cv2.imwrite(save_name, image_raw)

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.cfx.pop()

    def preprocess_image(self, input_image_frame):
        """
        description: Read an image from image path, convert it to RGB,
                     resize and pad it to target size, normalize to [0,1],
                     transform to NCHW format.
        param:
            input_image_path: str, image path
        return:
            image:  the processed image
            image_raw: the original image
            h: original height
            w: original width
        """
        image_raw = input_image_frame # cv2.imread(input_image_path)
        h, w, c = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        # Calculate widht and height and paddings
        r_w = INPUT_W / w
        r_h = INPUT_H / h
        if r_h > r_w:
            tw = INPUT_W
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((INPUT_H - th) / 2)
            ty2 = INPUT_H - th - ty1
        else:
            tw = int(r_h * w)
            th = INPUT_H
            tx1 = int((INPUT_W - tw) / 2)
            tx2 = INPUT_W - tw - tx1
            ty1 = ty2 = 0
        # Resize the image with long side while maintaining ratio
        image = cv2.resize(image, (tw, th))
        # Pad the short side with (128,128,128)
        image = cv2.copyMakeBorder(
            image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
        )
        image = image.astype(np.float32)
        # Normalize to [0,1]
        image /= 255.0
        # HWC to CHW format:
        image = np.transpose(image, [2, 0, 1])
        # CHW to NCHW format
        image = np.expand_dims(image, axis=0)
        # Convert the image to row-major order, also known as "C order":
        image = np.ascontiguousarray(image)
        return image, image_raw, h, w

    def xywh2xyxy(self, origin_h, origin_w, x):
        """
        description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        param:
            origin_h:   height of original image
            origin_w:   width of original image
            x:          A boxes tensor, each row is a box [center_x, center_y, w, h]
        return:
            y:          A boxes tensor, each row is a box [x1, y1, x2, y2]
        """
        y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
        r_w = INPUT_W / origin_w
        r_h = INPUT_H / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - (INPUT_H - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - (INPUT_H - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - (INPUT_W - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - (INPUT_W - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h

        return y

    # 后处理，把预测的结果分割成有意义的数据
    def post_process(self, output, origin_h, origin_w):
        """
        description: postprocess the prediction
        param:
            output:     A tensor likes [num_boxes,cx,cy,w,h,conf,cls_id, cx,cy,w,h,conf,cls_id, ...] 
            origin_h:   height of original image
            origin_w:   width of original image
        return:
            result_boxes: finally boxes, a boxes tensor, each row is a box [x1, y1, x2, y2]
            result_scores: finally scores, a tensor, each element is the score correspoing to box
            result_classid: finally classid, a tensor, each element is the classid correspoing to box
        """
        # Get the num of boxes detected
        num = int(output[0])
        # Reshape to a two dimentional ndarray
        pred = np.reshape(output[1:], (-1, 6))[:num, :]
        # to a torch Tensor
        pred = torch.Tensor(pred).cuda()
        # Get the boxes
        boxes = pred[:, :4]
        # Get the scores
        scores = pred[:, 4]
        # Get the classid
        classid = pred[:, 5]
        # Choose those boxes that score > CONF_THRESH
        si = scores > CONF_THRESH
        boxes = boxes[si, :]
        scores = scores[si]
        classid = classid[si]
        # Trandform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
        boxes = self.xywh2xyxy(origin_h, origin_w, boxes)
        # Do nms
        indices = torchvision.ops.nms(boxes, scores, iou_threshold=IOU_THRESHOLD).cpu()
        result_boxes = boxes[indices, :].cpu()
        result_scores = scores[indices].cpu()
        result_classid = classid[indices].cpu()
        # 返回检测框，检测概率和检测类别
        return result_boxes, result_scores, result_classid


class myThread(threading.Thread):
    def __init__(self, func, args):
        threading.Thread.__init__(self)
        self.func = func
        self.args = args

    def run(self):
        self.func(*self.args)


if __name__ == "__main__":
    # load custom plugins
    PLUGIN_LIBRARY = "build/libmyplugins.so"
    ctypes.CDLL(PLUGIN_LIBRARY)
    engine_file_path = "build/yolov5s.engine"

    # load coco labels
    categories = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"]

    # a  YoLov5TRT instance
    yolov5_wrapper = YoLov5TRT(engine_file_path)
    print("Loading YoloV5 Model")

    # from https://github.com/ultralytics/yolov5/tree/master/inference/images
    # input_image_paths = ["T_20210424205620_000.jpg"]

    # for input_image_path in input_image_paths:
    # 此处判断是接受图像话题还是视频输入
    if args.image_topic:
        while not rospy.is_shutdown():
            if not getim:
                # print("wait for image")
                # 每隔0.01s输入一次图像，加上处理的时间，处理频率约20Hz？
                time.sleep(0.01)
                continue
            with image_lock:
                img = g_image.copy()
                getim = False
            yolov5_wrapper.infer(img)
    else:
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            # create a new thread to do inference
            ret, input_image = cap.read()
            yolov5_wrapper.infer(input_image)

    # destroy the instance
    yolov5_wrapper.destroy()
