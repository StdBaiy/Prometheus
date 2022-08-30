"""
An example that uses TensorRT's Python api to make inferences.
"""
import ctypes
import sys
import os
path = sys.path[0]
path = path + '/siam_rpn_lib/'
print(path)
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
import torch
import torchvision
from deepsort.deep_sort import DeepSort


ROS_SERVER_ON = False
NAMES_TXT = 'coco'

INPUT_W = 608
INPUT_H = 608
CONF_THRESH = 0.25
IOU_THRESHOLD = 0.4
WINDOW_NAME = 'tensorrt-yolov5'



if ROS_SERVER_ON:
    socket_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    socket_server.bind(('', 9091))
    socket_server.listen(5)
    print('Start listening on port 9091 ...')
    client_socket, client_address = socket_server.accept()
    print('Got my client')



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

        self.cls_names = load_class_desc(NAMES_TXT)
        self.frame_cnt = 0

        reid_ckpt = 'deepsort/ckpt.t7'
        self.deepsort = DeepSort(reid_ckpt,
                                 max_dist=0.2, min_confidence=0.3,
                                 nms_max_overlap=0.5, max_iou_distance=0.7,
                                 max_age=70, n_init=3, nn_budget=100,
                                 use_cuda=True)

    def infer(self, input_image_frame):
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

        # print(result_boxes)
        if len(result_boxes) > 0:
            result_boxes = (torch.stack((result_boxes[:,0]+(result_boxes[:,2]-result_boxes[:,0])/2,result_boxes[:,1]+(result_boxes[:,3]-result_boxes[:,1])/2,(result_boxes[:,2]-result_boxes[:,0]),(result_boxes[:,3]-result_boxes[:,1])), dim=1))
            outputs = self.deepsort.update(result_boxes, result_scores, image_raw)
            outputs = torch.tensor(outputs)
            outputs_len = len(outputs)
            if outputs_len > 0:
                result_boxes = outputs[:, :4]
                result_scores = torch.ones(outputs_len)
                result_classid = outputs[:, 4]
            else:
                result_boxes = []
            # print(outputs)

        box_clicked = []
        dist = 1e9
        # Draw rectangles and labels on the original image
        for i in range(len(result_boxes)):
            box = result_boxes[i]
            if box[0] < 0: box[0] = 0
            if box[1] < 0: box[1] = 0
            if box[2] < 0: box[2] = 0
            if box[3] < 0: box[3] = 0
            # outputs = self.deepsort.update((torch.Tensor([box[0],box[1],(box[2]-box[0]),(box[3]-box[1])])), (torch.Tensor(result_scores[i])), im0)
            plot_one_box(
                box,
                image_raw,
                #label="{}:{:.2f}".format(self.cls_names[int(result_classid[i])], result_scores[i]),
                label="{}".format(int(result_classid[i])),
            )
            if ROS_SERVER_ON:
                # FrameID, 是否检测到目标(0/1,>1:num-of-objs), obj-order, 类别, x (0-1), y (0-1), w (0-1), h (0-1), 置信度
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


        if ROS_SERVER_ON and not sent_it:  # 发送无检测信息标志位
            client_socket.send('{:08d},000,000,000,0.000,0.000,0.000,0.000,0.000,0000,0000,0'.format(self.frame_cnt).encode('utf-8'))

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

    # from https://github.com/ultralytics/yolov5/tree/master/inference/images
    # input_image_paths = ["T_20210424205620_000.jpg"]

    # for input_image_path in input_image_paths:
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        # create a new thread to do inference
        ret, input_image = cap.read()
        yolov5_wrapper.infer(input_image)

    # destroy the instance
    yolov5_wrapper.destroy()
