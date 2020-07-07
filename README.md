---
### A complete implementation of paper [Orientation- and Scale-Invariant Multi-Vehicle Detection and Tracking from Unmanned Aerial Videos](http://jiewang.name/publications/rsmdpi2019) is 
#### 1. Fine-tune the vehicle detector with the dataset [UAV-Vehicle-Detection-Dataset](https://github.com/jwangjie/UAV-Vehicle-Detection-Dataset). 
#### 2. Step by step fine-tuning the vehicle detector [Fine-tune-YOLOv3](https://github.com/jwangjie/Fine-tune-YOLOv3).
#### 3. A multi-vehicle tracking is conducted by [deep_sort_yolov3](https://github.com/jwangjie/deep_sort_yolov3).

---

## Step by step implementation of fine-tuning the UAV vehicle detector

### System requirement: Ubuntu 16.04, OpenCV 3.4.0 and CUDA 9.0

---
### Install dependencies (OpenCV 3.4.0 and CUDA 9.0)
0. update apt-get   
``` bash 
sudo apt-get update
```
   
1. Install apt-get deps  
``` bash
sudo apt-get install openjdk-8-jdk git python-dev python3-dev python-numpy python3-numpy build-essential python-pip python3-pip python-virtualenv swig python-wheel libcurl3-dev curl   
```

2. install nvidia drivers 
``` bash
# download drivers
curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb

# download key to allow installation
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub

# install actual package
sudo dpkg -i ./cuda-repo-ubuntu1604_9.0.176-1_amd64.deb

#  install cuda 
sudo apt-get update
sudo apt-get install cuda-9-0   
```    

  2a. reboot Ubuntu
  ```bash
  sudo reboot
  ```    

  2b. check nvidia driver install 
  ``` bash
  nvidia-smi   

  # you should see a list of gpus printed    
  # if not, the previous steps failed.   
  ``` 

3. Install cudnn 

``` bash
wget https://s3.amazonaws.com/open-source-william-falcon/cudnn-9.0-linux-x64-v7.3.1.20.tgz
sudo tar -xzvf cudnn-9.0-linux-x64-v7.3.1.20.tgz
sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
```    

4. Add these lines to end of ~/.bashrc:   
``` bash
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
export CUDA_HOME=/usr/local/cuda
export PATH="$PATH:/usr/local/cuda/bin"
```   

4a. Reload bashrc     
``` bash 
source ~/.bashrc
```   

5. Install OpenCV
[How to install OpenCV 3.4.0 on Ubuntu 16.04]( https://www.pytorials.com/how-to-install-opencv340-on-ubuntu1604/)

---

### Install YOLOv3

1. **Install YOLOv3:** [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)   
   
   a.	For cuda complie issues: execute this [line](https://github.com/pjreddie/darknet/issues/200#issuecomment-329692411) `export     PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}`, before `make`
   
---

### Test YOLOv3 

1. Download pretrained yolo [weights](https://pjreddie.com/media/files/yolov3.weights), put it inside `darknet` folder    
   
2. Run YOLOv3 by `./darknet detector test ./cfg/coco.data ./cfg/yolov3.cfg ./yolov3.weights ./data/horses.jpg`

   a. if errors such as "The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Carbon support". If you are on Ubuntu or Debian, install libgtk2.0-dev and pkg-config, then re-run cmake or configure script” happens, follow the [steps]( https://stackoverflow.com/questions/14655969/opencv-error-the-function-is-not-implemented?answertab=active#tab-top) as follows
   ``` bash 
   sudo apt-get install libqt4-dev
   cd ~/ opencv-3.4.0
   mkdir build
   cd build
   cmake -D WITH_QT=ON ..
   make
   sudo make install 
   ```   

---

### Use our multiple vehicle detector 

1. Clone this [repository](https://github.com/jwangjie/Fine-tune-YOLOv3) 
   1) replace the [Makefile](https://github.com/jwangjie/Fine-tune-YOLOv3/blob/master/Makefile) in `darknet` folder
   2) add [yolov3_dji.cfg](https://github.com/jwangjie/Fine-tune-YOLOv3/blob/master/cfg/yolov3_dji.cfg) in `cfg` folder
   3) add `dji.data, dji.names, test.txt, train.txt` [files](https://github.com/jwangjie/Fine-tune-YOLOv3/tree/master/data) in `data` folder
2. download our [trained weight](https://drive.google.com/file/d/1xGxTxgevj6UPXTXNsUbt9g3Oogr3pATQ/view)
   
   **Test your video:**  `./darknet detector demo data/dji.data cfg/yolov3_dji.cfg yolov3_dji_final.weights yourVideo.mp4 -out_filename yourVideo.avi`

--- 

### Fine-tune Training using our dataset 

0. In general, follow [How to Train](https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects) 
   
   a. if error `Out of memory` shows, in `.cfg-file`, increase `subdivisions = 16, 32 or 64` following [this](https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L4)

1. Download our [dataset](https://github.com/jwangjie/UAV-Vehicle-Detection-Dataset), put all files in one folder `dji`, and add `dji` in `data` folder
2. **Training:**  `./darknet detector train data/dji.data cfg/yolov3_dji.cfg darknet53.conv.74`

---  

## Reference
Please kindly cite this paper in your publications if this helps your research:

```
@article{wang2019orientation,
  title={Orientation-and Scale-Invariant Multi-Vehicle Detection and Tracking from Unmanned Aerial Videos},
  author={Wang, Jie and Simeonova, Sandra and Shahbazi, Mozhdeh},
  journal={Remote Sensing},
  volume={11},
  number={18},
  pages={2155},
  year={2019},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```
