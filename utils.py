import os
import cv2
import numpy as np 
from tqdm import tqdm

def load_linedrawing(Path):
	#print('loading' + Path)
	img = cv2.imread(Path, cv2.IMREAD_GRAYSCALE)
	img = cv2.bitwise_not(img) #invert image
	ret,thresh1 = cv2.threshold(img,24,255,cv2.THRESH_BINARY)
	return thresh1

def load_image(Path):
	print('loading' + Path)
	img = cv2.imread(Path, cv2.IMREAD_GRAYSCALE)
	#img = cv2.bitwise_not(img) #invert image not for rendering
	return img

def load_mask(Path):
	#print('loading' + Path)
	mask = cv2.imread(Path, cv2.IMREAD_GRAYSCALE)
	return mask

def load_color(Path):
	#print('loading' + Path)
	return cv2.imread(Path, cv2.IMREAD_UNCHANGED)

def load_normal(Path):
	print('loading' + Path)
	imgN = cv2.imread(Path)
	return imgN

def neighbours_vec(image):
    return image[2:,1:-1], image[2:,2:], image[1:-1,2:], image[:-2,2:], image[:-2,1:-1],     image[:-2,:-2], image[1:-1,:-2], image[2:,:-2]

def transitions_vec(P2, P3, P4, P5, P6, P7, P8, P9):
    return ((P3-P2) > 0).astype(int) + ((P4-P3) > 0).astype(int) + \
    ((P5-P4) > 0).astype(int) + ((P6-P5) > 0).astype(int) + \
    ((P7-P6) > 0).astype(int) + ((P8-P7) > 0).astype(int) + \
    ((P9-P8) > 0).astype(int) + ((P2-P9) > 0).astype(int)

def zhangSuen_vec(image, iterations):
    for iter in tqdm(range (1, iterations)):
        #print(iter)
        # step 1    
        P2,P3,P4,P5,P6,P7,P8,P9 = neighbours_vec(image)
        condition0 = image[1:-1,1:-1]
        condition4 = P4*P6*P8
        condition3 = P2*P4*P6
        condition2 = transitions_vec(P2, P3, P4, P5, P6, P7, P8, P9) == 1
        condition1 = (2 <= P2+P3+P4+P5+P6+P7+P8+P9) * (P2+P3+P4+P5+P6+P7+P8+P9 <= 6)
        cond = (condition0 == 1) * (condition4 == 0) * (condition3 == 0) * (condition2 == 1) * (condition1 == 1)
        changing1 = np.where(cond == 1)
        image[changing1[0]+1,changing1[1]+1] = 0
        # step 2
        P2,P3,P4,P5,P6,P7,P8,P9 = neighbours_vec(image)
        condition0 = image[1:-1,1:-1]
        condition4 = P2*P6*P8
        condition3 = P2*P4*P8
        condition2 = transitions_vec(P2, P3, P4, P5, P6, P7, P8, P9) == 1
        condition1 = (2 <= P2+P3+P4+P5+P6+P7+P8+P9) * (P2+P3+P4+P5+P6+P7+P8+P9 <= 6)
        cond = (condition0 == 1) * (condition4 == 0) * (condition3 == 0) * (condition2 == 1) * (condition1 == 1)
        changing2 = np.where(cond == 1)
        image[changing2[0]+1,changing2[1]+1] = 0
    return 255*image

def BorderHandle(x, size_2, lenn):
	if (x - size_2) < 0:
		xm = 0
		Xm = size_2 - x
	else:
		xm = (x - size_2)
		Xm = 0

	if(x + size_2) > lenn:
		xM = lenn
		XM = size_2 + (lenn - x)
	else:
		xM = x + size_2
		XM = 2*size_2

	return (xm, xM, Xm, XM)


def CropMultiScale_ZeroPadding_2(x, y, image, image_2, image_4, size):
	img_blank = np.zeros((size, size, 3), np.float32)
	x1 = int(x / 2) + size + 1
	y1 = int(y / 2)	+ size + 1
	x2 = int(x / 4) + size + 1
	y2 = int(y / 4) + size + 1
	x = x + size + 1
	y = y + size + 1
	size = int(size / 2)

	xm, xM, Xm, XM = BorderHandle(x1, size, image_2.shape[1])
	ym, yM, Ym, YM = BorderHandle(y1, size, image_2.shape[0])
	img_blank[Ym:YM,Xm:XM,1] = image_2[ym:yM, xm:xM]

	xm, xM, Xm, XM = BorderHandle(x2, size, image_4.shape[1])
	ym, yM, Ym, YM = BorderHandle(y2, size, image_4.shape[0])
	img_blank[Ym:YM,Xm:XM,2] = image_4[ym:yM, xm:xM]

	xm, xM, Xm, XM = BorderHandle(x, size, image.shape[1])
	ym, yM, Ym, YM = BorderHandle(y, size, image.shape[0])
	img_blank[Ym:YM,Xm:XM,0] = image[ym:yM, xm:xM]

	img_blank = img_blank / 127.5 - 1.0
	return img_blank

def PrepareMultiScale(img):
	size = 256
	img_pad = np.zeros((img.shape[0] + 2 * size, img.shape[1] + 2 * size), np.float32)
	img_pad[size + 1:(img.shape[0]+size + 1), size + 1:(img.shape[1]+size + 1)] = img
	
	#resized version of image for global view
	img_2tmp = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation= cv2.INTER_LINEAR)
	img_2 = np.zeros((img_2tmp.shape[0] + 2 * size, img_2tmp.shape[1] + 2 * size), np.float32)
	img_2[size + 1:(img_2tmp.shape[0]+size + 1), size + 1:(img_2tmp.shape[1]+size + 1)] = img_2tmp
	
	img_4tmp = cv2.resize(img_2tmp, None, fx=0.5, fy=0.5, interpolation= cv2.INTER_LINEAR)
	img_4 = np.zeros((img_4tmp.shape[0] + 2 * size, img_4tmp.shape[1] + 2 * size), np.float32)
	img_4[size + 1:(img_4tmp.shape[0]+size + 1), size + 1:(img_4tmp.shape[1]+size + 1)] = img_4tmp

	return img_pad, img_2, img_4

def MaskToInput(img, Mask):
	NonzeroPointsMask1, NonzeroPointsMask2 = np.nonzero(Mask)
	#print(np.shape(NonzeroPointsMask2))
	Non= np.array((NonzeroPointsMask1, NonzeroPointsMask2))
	for l in range(len(NonzeroPointsMask2)):
		x = Non[1][l]
		y = Non[0][l]
		if(img[y, x] != 255.0):
			img[y, x] = 160
	return img

def CleanWithMask(img, Mask):
	height, width = Mask.shape
	t1 = np.stack((Mask, Mask, Mask), axis = 2)
	t1 = t1 / 255.0
	t2 = 1.0 - t1
	img[0:height, 0:width, :] = img[0:height, 0:width, :] * t1 + t2* 255.0
	return img

def Normalize(x):
	Norm = np.sqrt(x[0]*x[0] + x[1]*x[1] +x[2]*x[2])
	if Norm == 0.0:
		return x
	return x / Norm