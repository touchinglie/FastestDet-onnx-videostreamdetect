import cv2
import time
import numpy as np
import onnxruntime
import os
from pathlib import Path
import sys
import argparse


# sigmoid函数
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


# tanh函数
def tanh(x):
    return 2.0 / (1 + np.exp(-2 * x)) - 1


# 数据预处理
def preprocess(src_img, size):
    output = cv2.resize(src_img, (size[0], size[1]), interpolation=cv2.INTER_AREA)
    output = output.transpose(2, 0, 1)
    output = output.reshape((1, 3, size[1], size[0])) / 255

    return output.astype("float32")


# nms算法
def nms(dets, thresh=0.3):
    # dets:N*M,N是bbox的个数，M的前4位是对应的（x1,y1,x2,y2），第5位是对应的分数
    # #thresh:0.3,0.5....
    if dets is []:
        return [None]
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 求每个bbox的面积
    order = scores.argsort()[::-1]  # 对分数进行倒排序
    keep = []  # 用来保存最后留下来的bboxx下标

    while order.size > 0:
        i = order[0]  # 无条件保留每次迭代中置信度最高的bbox
        keep.append(i)

        # 计算置信度最高的bbox和其他剩下bbox之间的交叉区域
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 计算置信度高的bbox和其他剩下bbox之间交叉区域的面积
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        # 求交叉区域的面积占两者（置信度高的bbox和其他bbox）面积和的必烈
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # 保留ovr小于thresh的bbox，进入下一次迭代。
        inds = np.where(ovr <= thresh)[0]

        # 因为ovr中的索引不包括order[0]所以要向后移动一位
        order = order[inds + 1]

    output = []
    for i in keep:
        output.append(dets[i].tolist())

    return output


# 检测
def detection(session, img, input_width, input_height, thresh):
    pred = []

    # 输入图像的原始宽高
    H, W, _ = img.shape

    # 数据预处理: resize, 1/255
    data = preprocess(img, [input_width, input_height])

    # 模型推理
    input_name = session.get_inputs()[0].name
    feature_map = session.run([], {input_name: data})[0][0]

    # 输出特征图转置: CHW, HWC
    feature_map = feature_map.transpose(1, 2, 0)
    # 输出特征图的宽高
    feature_map_height = feature_map.shape[0]
    feature_map_width = feature_map.shape[1]

    # 特征图后处理
    for h in range(feature_map_height):
        for w in range(feature_map_width):
            data = feature_map[h][w]

            # 解析检测框置信度
            obj_score, cls_score = data[0], data[5:].max()
            score = (obj_score**0.6) * (cls_score**0.4)

            # 阈值筛选
            if score > thresh:
                # 检测框类别
                cls_index = np.argmax(data[5:])
                # 检测框中心点偏移
                x_offset, y_offset = tanh(data[1]), tanh(data[2])
                # 检测框归一化后的宽高
                box_width, box_height = sigmoid(data[3]), sigmoid(data[4])
                # 检测框归一化后中心点
                box_cx = (w + x_offset) / feature_map_width
                box_cy = (h + y_offset) / feature_map_height

                # cx,cy,w,h => x1, y1, x2, y2
                x1, y1 = box_cx - 0.5 * box_width, box_cy - 0.5 * box_height
                x2, y2 = box_cx + 0.5 * box_width, box_cy + 0.5 * box_height
                x1, y1, x2, y2 = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)

                pred.append([x1, y1, x2, y2, score, cls_index])
    if len(pred) == 0:
        return [None]
    return nms(np.array(pred), opt.nmsthresh)


def run(img):
    # time库里的定时器只有global整个time库类才能正常使用
    global time
    global resframe

    # 目标检测
    start = time.perf_counter()
    # 输入并推理
    bboxes = detection(session, img, input_width, input_height, opt.detectthresh)
    if bboxes == [None]:
        if type(opt.source) == int or opt.source == "video":
            resframe = img
        return img
    end = time.perf_counter()
    usedtime = (end - start) * 1000.0
    print("forward time:%fms" % usedtime)

    print("=================box info===================")
    for b in bboxes:
        print(b)
        obj_score, cls_index = b[4], int(b[5])
        x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])

        # 绘制检测框
        img = img.astype(np.uint8)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(img, "%.2f" % obj_score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)
        cv2.putText(img, names[cls_index], (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)

    if type(opt.source) == int or opt.source == "video":
        resframe = img
    return img
    # 接下来是将处理好的图像外显，可以有以下几种方法：
    # 1、写成图片文件保存到本地
    # cv2.imwrite("result.jpg", img)

    # 2、转成其它格式交由其它程序处理
    """
    img = cv2.cvtColor(
        img, cv2.COLOR_RGB2BGR
    )  # ESP32采集的是RGB格式，要转换为BGR（opencv的格式）
    sucbool, res = cv2.imencode(".jpg", img)
    resimg = res.tobytes()
    return (bboxes, resimg)
    """

    # 3、写成视频保存到本地
    """
    save_path = "ROOT / result.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") # 确定视频被保存后的编码格式
    fps = 20
    w,h=(352,352)
    vid_writer[i] = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
    while (capture.isOpened()): # 设备摄像头被打开后
        retval, frame = capture.read() # 从摄像头中实时读取视频
        if retval = True: # 读取到摄像头视频后
            outputrite(frame) # 在VideoWriter类对象中写入读取到的帧
            cv2.imsw("frame", frame) # 在窗口中显示摄像头视频
        key = cv2.waitKey(1) # 窗口的图像刷新时间为1毫秒
        if key == 27: # 如果按下Esc键
            break
    capture.release() # 关闭设备摄像头
    output.release() 释放VideoWriter类对象
    """

    # 4、用opencv内置的cv2.imshow()显示实时流(外层得套for或者while循环)
    # cv2.imshow(0,img)
    # capture.release() # 关闭摄像头流
    # cv2.destroyAllWindows() # 销毁显示摄像头视频的窗口

    """
    以上四种方法可以写在run里也可写在下面执行的区域
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classes", type=str, default="coco.names", help="The classes' name"
    )
    parser.add_argument(
        "--weight", type=str, default="FastestDet.onnx", help=".weight config"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=0,
        help="The source uses to detect, webcam(The int num of cam)/video(just set it as 'video' and set videofile as well)/pic(absoulute)",
    )
    parser.add_argument(
        "--videofile", type=str, default="", help="Required only for 'video' source"
    )
    parser.add_argument(
        "--nmsthresh", type=int, default=0.25, help="Inference Detect threshold"
    )
    parser.add_argument(
        "--detectthresh", type=int, default=0.7, help="Inference Detect threshold"
    )

    # 保证命令在文件根目录执行，文件名定位正确
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0]  # 确认文件根目录
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))  # 将根目录临时添加到系统环境PATH中
    ROOT = Path(
        os.path.relpath(ROOT, Path.cwd())
    )  # 下面相对地址必须填 ROOT / '(相对地址)'

    # 以下的assert都是确认指定的文件是否存在
    opt = parser.parse_args()
    assert os.path.exists(opt.classes), "请指定正确的配置文件路径"
    assert os.path.exists(opt.weight), "请指定正确的模型路径"

    # 模型输入的宽高
    input_width, input_height = 352, 352
    # 加载模型，后一个provider参数是指定用CPU还是GPU推理，默认CPU
    # 可参考网上改成'CUDAExecutionProvider'来利用CUDA使用GPU推理
    session = onnxruntime.InferenceSession(
        opt.weight, providers=["CPUExecutionProvider"]
    )
    # 加载label names
    names = []
    with open(opt.classes, "r") as f:
        for line in f.readlines():
            names.append(line.strip())

    # 读取source源同样有三种方式：
    # 判断输入是什么类型的数据
    if type(opt.source) != int:
        if opt.source == "video":
            # 视频文件流源
            assert os.path.exists(opt.videofile), "视频文件不存在"
            cap = cv2.VideoCapture(opt.videofile)  # 用opencv打开视频文件，转换为视频流
            print("Opened VideoStream.")
            print(f"Video width:{int(cap.get(3))} Video height:{int(cap.get(4))}")
            # 查看结果文件夹最后的视频序号并生成此次结果视频文件
            folderChecker = os.listdir("resultsave/video/")
            if len(folderChecker) == 0:
                savePath = f"resultsave/video/0.mp4"
            else:
                videoNum = int(sorted(folderChecker)[-1].split(".")[0]) + 1
                savePath = f"resultsave/video/{videoNum}.mp4"
            # 帧率调整
            fps = 25.0
            # 创建结果视频写入器以保存文件
            vw = cv2.VideoWriter(
                savePath,
                cv2.VideoWriter.fourcc(*"mp4v"),
                fps,
                (int(cap.get(3)), int(cap.get(4))),
            )
            while cap.isOpened():
                # 按下q键退出推理
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    vw.release()
                    break
                ret, img = cap.read()  # 获取视频的开启状态和每一帧图片
                if ret:
                    run(img)
                    vw.write(resframe)
                    cv2.imshow("Videostream", resframe)
                else:
                    cv2.waitKey(1)  # 防止推理速度不够快，有空白帧
            cap.release()
        else:
            # 图片源
            assert os.path.exists(opt.source), "图片文件不存在"
            img = cv2.imread(opt.source)
            respic = run(img)
            cv2.imwrite("resultsave/result.jpg", respic)
    else:
        # 摄像头实时流源
        cap = cv2.VideoCapture(opt.source)  # 用opencv打开摄像头，0为默认摄像头
        print("Opened Camera.")
        print(f"Video width:{int(cap.get(3))} Video height:{int(cap.get(4))}")
        # 查看结果文件夹最后的视频序号并生成此次结果视频文件
        folderChecker = os.listdir("resultsave/video/")
        if len(folderChecker) == 0:
            savePath = f"resultsave/video/0.mp4"
        else:
            videoNum = int(sorted(folderChecker)[-1].split(".")[0]) + 1
            savePath = f"resultsave/video/{videoNum}.mp4"
        # 帧率调整
        fps = 25.0
        # 创建结果视频写入器以保存文件
        vw = cv2.VideoWriter(
            savePath,
            cv2.VideoWriter.fourcc(*"mp4v"),
            fps,
            (int(cap.get(3)), int(cap.get(4))),
        )
        while True:
            # 按下q键退出推理
            if cv2.waitKey(1) & 0xFF == ord("q"):
                vw.release()
                break
            ret, img = cap.read()  # 获取视频的开启状态和每一帧图片
            if ret:
                run(img)
                vw.write(resframe)
                cv2.imshow("CameraStream", resframe)
            else:
                cv2.waitKey(1)  # 防止推理速度不够快，有空白帧

    if "cap" in globals():
        if cap.isOpened:
            cap.release()
    if "vw" in globals():
        vw.release()
    cv2.destroyAllWindows()
