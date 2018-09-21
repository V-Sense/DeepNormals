import argparse
import os
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import l2_normalize
from tflearn.metrics import Top_k, R2
import cv2
import sys

from model import *
from utils import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('--docker_path', dest='P', default='./', help='Path of shared docker directory, default is ./ for using without docker')
parser.add_argument('--lineart_path', dest='lineart_path', default='Pepper/Lines.jpg', help='Path of the linedrawing (grayscale lines)')
parser.add_argument('--mask_path', dest='mask_path', default='Pepper/Mask.jpg', help='Path of the mask')
parser.add_argument('--save_path', dest ='save_path', default='RES/', help='Path of the save folder')
parser.add_argument('--model_name', dest='model_name', default='Net/DeepNormals/DeepNormals', help='Path of the Model')
parser.add_argument('--thining_it', dest='thining_it', type=int, default=10, help='Number of iterations for ZhangSuen thining algorithm')
parser.add_argument('--nb_grids', dest='nb_grids', type=int, default=40, help='Number of tile grids for reconstruction')
args = parser.parse_args()

# Add docker prefix if needed
args.lineart_path = args.P + args.lineart_path
args.mask_path = args.P + args.mask_path
args.model_name = args.P + args.model_name
args.save_path = args.P + args.save_path

#Check if images exist
if not os.path.isfile(args.lineart_path):
	sys.exit("Error, couldn't read lineart image, file doesn't exist")
if not os.path.isfile(args.mask_path):
	sys.exit("Error, couldn't read mask, file doesn't exist")

#First prepare input data
img = load_linedrawing(args.lineart_path)

#Load Mask
Mask = load_mask(args.mask_path)

#Thining algorithm
print('Thining Input image: ')
img = zhangSuen_vec(img/255.0, args.thining_it)

#print("Apply mask correction to image") *! Different from Paper (minor change)!*
img = MaskToInput(img, Mask)

#Padding and preparing multiscale images
img_pad, img_2, img_4 = PrepareMultiScale(img)

height, width = img.shape
size= 256

#Generate and Load model
model = GenerateNet()
#Load Network
MODEL_NAME = args.model_name
with tf.Session() as sess:
	if os.path.exists('{}.meta'.format(MODEL_NAME)):
		print('model: ' + MODEL_NAME + ' loading!')
		model.load(MODEL_NAME)
		print('model: ' + MODEL_NAME + ' loaded!')
	else:
		sys.exit('Error: ' + MODEL_NAME + ' Not Found')


ind =0.0
recfin = np.zeros((height + 600, width + 600, 3)).astype(float)
recTrim = []
print('Predicting grids:')
for offset in tqdm(range(0, 256, int(256/args.nb_grids))):
	SubBatch = []
	Pos = []
	index = 0.0

	for j in range(int(height / 256) + 2):
		y = j * 256 + offset -128
		for i in range(int(width / 256) + 2):
			x = i* 256 + offset -128
			#st = time.time()
			Sub = CropMultiScale_ZeroPadding_2(x, y, img_pad, img_2, img_4, size)
			#Sub[Sub<0.3]=0
			#et = time.time()-st
			#print("Cropp: " + str(et))
			#cv2.imshow('Sub', Sub)
			#cv2.waitKey(0)
			SubBatch.append(Sub)
			index = index + 1.0
			Pos.append([x,y])

	S = np.array(SubBatch)
	predN = model.predict({'input' : S})
	rec = np.zeros((height + 900, width + 900, 3)).astype(float)
	off = 260
	s = int(size/2)
	ind += 1.0
	for i in range(int(index)):
		x = off + Pos[i][0]
		y = off + Pos[i][1]
		rec[(y-s):(y+s), (x-s):(x+s),:] += predN[i]
	recfin[0: height, 0:width] += rec[260: height + 260, 260:width+260]



#Average Tiles
recfin = (recfin/(ind)) *127.5 + 127.5

#Clean result
recfin = CleanWithMask(recfin, Mask)

#Remove Padding
final = np.zeros((height, width,3))
final = recfin[0:height, 0:width, :]

#Write result
cv2.imwrite(args.save_path +'Normal_Map.png', final)
print('result normal map saved in: ' +args.save_path +'Normal_Map.png')





