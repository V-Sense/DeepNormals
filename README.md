# [Deep Normals](https://www.scss.tcd.ie/~hudonm/publication/deep-normal-estimation-for-automatic-shading-of-hand-drawn-characters/)
![Example result](images/Carrot.png?raw=true "Example result of the provided model." )
Example result of a deep normal estimation from sketch. Original creation from David Revoy, only non-commercial research usage is allowed.

## Overview

This code provides pre-trained models used in the following research paper:
```
   "Deep Normal Estimation for Automatic Shading of Hand-Drawn Characters"
   Matis Hudon, Rafael Pagés, Mairéad Grogan, Aljosa Smolić
   Geometry Meets Deep Learning ECCV 2018 Workshop
```
Please refer to our [project webpage](https://www.scss.tcd.ie/~hudonm/publication/deep-normal-estimation-for-automatic-shading-of-hand-drawn-characters/) for more detailed information. 
Please also see our [video of results](https://www.youtube.com/watch?v=1tZ-y0PzV8g&t=3s).

## Prerequisites

###With Docker

If you are familiar with docker [this image](https://hub.docker.com/r/matishudon/dockerdeepn/) has everything installed to run the code (please install and run with [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)):
	```
	docker pull matishudon/dockerdeepn
	```

This code was tested under ubuntu 16.04.

###Locally

This code was tested under ubuntu 16.04.

- Cuda 9

- python3

- [Numpy](http://www.numpy.org/)
	```
	pip3 install numpy
	```
	
- [Tensorflow](https://www.tensorflow.org/)
	```
	pip3 install -U tensorflow-gpu
	```
- opencv-python 
	```
	pip3 install opencv-python
	```
	
- [tqdm](https://github.com/tqdm/tqdm)
	```
	pip3 install tqdm
	```
	
-  [TFlearn](http://tflearn.org/)
	```
	pip3 install tflearn
	```
	
## Usage
### Generate Normal maps from line drawings

Before using the code, the model has to be downloaded from [here](https://v-sense.scss.tcd.ie/Datasets/DeepNormalsModel.zip) Unzip in the Net/ folder.

####To Run Locally:

You can test the code with:
```
python3 main.py
```
	
You should see a file Normal_Map.png
You can specify your own images (as in the example folder) with:
```
python3 main.py --lineart_path PathToYourImage --mask_path PathToCorrespondingMask
```
Please see main.py for other commands.

####To run with docker:
```
sudo nvidia-docker run -v CodeDirectory/:/container/directory/ -it matishudon/dockerdeepn python3 /container/directory/main.py --docker_path /container/directory/
```
Where **CodeDirectory** is the full path of the directory containing main.py.

### Play with the "renderer"
 We also provide a very simple and slow renderer Interactive_Rendering.py please look into the python file for commands. You need to provide an additional color image for this. See the example folder Pepper/.
 <img src="images/Pepper.gif" width="400" height="300" />
Artistic image from David Revoy, only non-commercial research usage is allowed.
####To Run Locally:
```
python3 Interactive_Rendering.py 
```
####To run with docker:

First run on local machine:
```
xhost +local:docker
```
Then:
```
sudo nvidia-docker run -v CodeDirectory/:/container/directory/ -ti -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --env QT_X11_NO_MITSHM=1 matishudon/dockerdeepn python3 /container/directory/Interactive_Rendering.py --docker_path /container/directory/
```
Where **CodeDirectory** is the full path of the directory containing main.py.

## Dataset

The Dataset used in this work (Training and Testing) can be downloaded [here](https://v-sense.scss.tcd.ie/Datasets/DeepNormalsDataset.zip).

## Citing 
If you use this model, code or dataset please cite our paper:

```
@inproceedings{hudon2018deep,
	title={Deep Normal Estimation for Automatic Shading of Hand-Drawn Characters},
	author={Matis Hudon, Rafael Pagés, Mairéad Grogan, Aljosa Smolić},
	booktitle={ECCV Workshops},
	year={2018}
}
```

## Acknowledgements
The authors would like to thank David Revoy and Ester Huete, for sharing their original creations. This publication has emanated from research conducted with the financial support of Science Foundation Ireland (SFI) under the Grant Number 15/RP/2776. We gratefully acknowledge the support of NVIDIA Corporation with the donation of the Titan Xp GPU used for this research.

## License
Copyright (c) 2018 Matis Hudon, Trinity College Dublin

**Please read carefully the following terms and conditions and any accompanying documentation before you download and/or use this software and associated documentation files (the "Software").**

The authors hereby grant you a non-exclusive, non-transferable, free of charge right to copy, modify, merge, publish, distribute, and sublicense the Software for the sole purpose of performing non-commercial scientific research, non-commercial education, or non-commercial artistic projects.

Any other use, in particular any use for commercial purposes, is prohibited. This includes, without limitation, incorporation in a commercial product, use in a commercial service, or production of other artefacts for commercial purposes.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

You understand and agree that the authors are under no obligation to provide either maintenance services, update services, notices of latent defects, or corrections of defects with regard to the Software. The authors nevertheless reserve the right to update, modify, or discontinue the Software at any time.

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. You agree to cite the Deep Normal Estimation for Automatic Shading of Hand-Drawn Characters paper in documents and papers that report on research using this Software.

![](images/v-sense.jpg) 