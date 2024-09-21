# FastestStreamDet

***A inference implementation that supports multiple input streams in a single file based on FastestDet.***

<img alt="video_demo" src="Streamresult.png" align="right">


# Advance

* Use the ONNX model to infer real-time camera streams, images, and video file streams.
* Simply output inference results in a variety of ways.
* Single file.
* Debugged problem carefully, fixing most bugs.


# How to use

## Install the required dependencies firstly
* Python(3.8 for recommended version) and pip
* opencv-python
* numpy==1.23.0
* onnx_simplifier==0.3.10
* onnxruntime==1.16.0
* pathlib
* argparse

## Demo
* Type in the terminal:
  
  ```
  python detect.py --source datatest/6.jpg
  ```

* Check your result image in the resultsave folder.

* Type in the terminal:
  
  ```
  python detect.py --source video --videofile datatest/21115-315137069_small.mp4
  python detect.py --source video --videofile datatest/2174-155747455_small.mp4
  ```

* The First example video owned by [Jahhoo](https://pixabay.com/users/jahhoo-1418773/)
* The Second example video owned by [RafterJr72](https://pixabay.com/users/rafterjr72-11468402/)

* Check the detect result through the new window created by opencv

* Click "q" to end the detect process.


## CameraStream
* Choose your own camera index then type in terminal:
  
  ```
  python detect.py
  ```
  or use your own camera index:
  ```
  python detect.py {your camera index, it usual in integer type}
  ```

* Check the detect result through the new window created by opencv

* Click "q" to end the detect process.


## Picture
* Type in the terminal:
  
  ```
  python detect.py --source {your Picturefilepath}
  ```

* Check your result image in the resultsave folder.


## VideoStream
* Type in the terminal:
  
  ```
  python detect.py --source video --videofile {your videofilepath}
  ```

* Check the detect result through the new window created by opencv

* Click "q" to end the detect process.

# Todo

* Save the VideoStream detection result as a file
* Write an UI to adapt most usage
* Multi CPU core Inferences
* Use ncnn Framework(almost done)

# Reference

* [FastestDet](https://github.com/dog-qiuqiu/FastestDet)
* [onnx](https://github.com/onnx/onnx)
* [Jahhoo](https://pixabay.com/users/jahhoo-1418773/)
* [RafterJr72](https://pixabay.com/users/rafterjr72-11468402/)
