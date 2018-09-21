import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import l2_normalize
from tflearn.metrics import Top_k, R2

def GenerateNet():
	LeakyReluSlope = 0.3
	#Defining the network 
	# input image 3 multiscale layers
	
	# Current size: 256x256x3
	NetworkIN = input_data(shape=[None, 256, 256, 3], name='input')
	
	# Current size: 256x256x3
	Network128 = tflearn.layers.conv.conv_2d(NetworkIN, 6, 4, strides=2, activation=lambda x: tflearn.activations.leaky_relu(x, alpha=LeakyReluSlope), name="e1")
	#Network = tflearn.layers.normalization.batch_normalization(Network128)
	
	# Current size: 128x128x6
	Network64 = tflearn.layers.conv.conv_2d(Network128, 12, 4, strides=2, activation=lambda x: tflearn.activations.leaky_relu(x, alpha=LeakyReluSlope), name="e1")
	#Network = tflearn.layers.normalization.batch_normalization(Network64)
	
	# Current size: 64x64x12
	Network32 = tflearn.layers.conv.conv_2d(Network64, 24, 4, strides=2, activation=lambda x: tflearn.activations.leaky_relu(x, alpha=LeakyReluSlope), name="e1")
	#Network = tflearn.layers.normalization.batch_normalization(Network32)
	
	# Current size: 32x32x24
	Network16 = tflearn.layers.conv.conv_2d(Network32, 48, 4, strides=2, activation=lambda x: tflearn.activations.leaky_relu(x, alpha=LeakyReluSlope), name="e1")
	#Network = tflearn.layers.normalization.batch_normalization(Network16)
	
	# Current size: 16x16x48
	Network8 = tflearn.layers.conv.conv_2d(Network16, 96, 4, strides=2, activation=lambda x: tflearn.activations.leaky_relu(x, alpha=LeakyReluSlope), name="e1")
	#Network = tflearn.layers.normalization.batch_normalization(Network8)
	
	# Current size: 8x8x96
	Network4 = tflearn.layers.conv.conv_2d(Network8, 192, 4, strides=2, activation=lambda x: tflearn.activations.leaky_relu(x, alpha=LeakyReluSlope), name="e1")
	#Network = tflearn.layers.normalization.batch_normalization(Network4)
	
	# Current size: 4x4x192
	Network2 = tflearn.layers.conv.conv_2d(Network4, 384, 4, strides=2, activation=lambda x: tflearn.activations.leaky_relu(x, alpha=LeakyReluSlope), name="e1")
	#Network = tflearn.layers.normalization.batch_normalization(Network2)
	
	#===================== UPSAMPLING DECODER ==========================
	
	#current size: 2x2x384
	#Network = tflearn.layers.core.dropout(Network2, 0.5)
	
	#------------ 2x2 -> 4x4 ----------------------------------------------------------|
	#current size: 2x2x384
	Network = tflearn.layers.conv.upsample_2d(Network2, 2)
	Network = tflearn.layers.conv.conv_2d(Network, 192, 4, strides=1, activation=lambda x: tflearn.activations.leaky_relu(x, alpha=LeakyReluSlope), name="e1")
	#Network = tflearn.layers.normalization.batch_normalization(Network)
	#current size: 4x4x192
	Network = tflearn.layers.merge_ops.merge((Network4, Network), mode='concat', axis=3)
	#current size: 4x4x384
	Network = tflearn.layers.conv.conv_2d(Network, 192, 4, strides=1, activation=lambda x: tflearn.activations.leaky_relu(x, alpha=LeakyReluSlope), name="e1")
	#Network = tflearn.layers.normalization.batch_normalization(Network)
	
	#Network = tflearn.layers.core.dropout(Network, 0.5)
	
	#------------ 4x4 -> 8x8 ----------------------------------------------------------|
	#current size: 4x4x192
	Network = tflearn.layers.conv.upsample_2d(Network, 2)
	Network = tflearn.layers.conv.conv_2d(Network, 96, 4, strides=1, activation=lambda x: tflearn.activations.leaky_relu(x, alpha=LeakyReluSlope), name="e1")
	#Network = tflearn.layers.normalization.batch_normalization(Network)
	#current size: 8x8x96
	Network = tflearn.layers.merge_ops.merge((Network8, Network), mode='concat', axis=3)
	#current size: 8x8x192
	Network = tflearn.layers.conv.conv_2d(Network, 96, 4, strides=1, activation=lambda x: tflearn.activations.leaky_relu(x, alpha=LeakyReluSlope), name="e1")
	#Network = tflearn.layers.normalization.batch_normalization(Network)
	
	#Network = tflearn.layers.core.dropout(Network, 0.5)
	
	
	#------------ 8x8 -> 16x16 ----------------------------------------------------------|
	#current size: 8x8x96
	Network = tflearn.layers.conv.upsample_2d(Network, 2)
	Network = tflearn.layers.conv.conv_2d(Network, 48, 4, strides=1, activation=lambda x: tflearn.activations.leaky_relu(x, alpha=LeakyReluSlope), name="e1")
	#Network = tflearn.layers.normalization.batch_normalization(Network)
	#current size: 16x16x48
	Network = tflearn.layers.merge_ops.merge((Network16, Network), mode='concat', axis=3)
	#current size: 16x16x96
	Network = tflearn.layers.conv.conv_2d(Network, 48, 4, strides=1, activation=lambda x: tflearn.activations.leaky_relu(x, alpha=LeakyReluSlope), name="e1")
	#Network = tflearn.layers.normalization.batch_normalization(Network)
	
	#------------ 16x16 -> 32x32 ----------------------------------------------------------|
	#current size: 16x16x48
	Network = tflearn.layers.conv.upsample_2d(Network, 2)
	Network = tflearn.layers.conv.conv_2d(Network, 24, 4, strides=1, activation=lambda x: tflearn.activations.leaky_relu(x, alpha=LeakyReluSlope), name="e1")
	#Network = tflearn.layers.normalization.batch_normalization(Network)
	#current size: 32x32x24
	Network = tflearn.layers.merge_ops.merge((Network32, Network), mode='concat', axis=3)
	#current size: 32x32x48
	Network = tflearn.layers.conv.conv_2d(Network, 24, 4, strides=1, activation=lambda x: tflearn.activations.leaky_relu(x, alpha=LeakyReluSlope), name="e1")
	#Network = tflearn.layers.normalization.batch_normalization(Network)
	
	
	#------------ 32x32 -> 64x64 ----------------------------------------------------------|
	#current size: 32x32x24
	Network = tflearn.layers.conv.upsample_2d(Network, 2)
	Network = tflearn.layers.conv.conv_2d(Network, 12, 4, strides=1, activation=lambda x: tflearn.activations.leaky_relu(x, alpha=LeakyReluSlope), name="e1")
	#Network = tflearn.layers.normalization.batch_normalization(Network)
	#current size: 64x64x12
	Network = tflearn.layers.merge_ops.merge((Network64, Network), mode='concat', axis=3)
	#current size: 64x64x24
	Network = tflearn.layers.conv.conv_2d(Network, 12, 4, strides=1, activation=lambda x: tflearn.activations.leaky_relu(x, alpha=LeakyReluSlope), name="e1")
	#Network = tflearn.layers.normalization.batch_normalization(Network)
	
	
	#------------ 64x64 -> 128x128 ----------------------------------------------------------|
	#current size: 64x64x12
	Network = tflearn.layers.conv.upsample_2d(Network, 2)
	Network = tflearn.layers.conv.conv_2d(Network, 6, 4, strides=1, activation=lambda x: tflearn.activations.leaky_relu(x, alpha=LeakyReluSlope), name="e1")
	#Network = tflearn.layers.normalization.batch_normalization(Network)
	#current size: 128x128x6
	Network = tflearn.layers.merge_ops.merge((Network128, Network), mode='concat', axis=3)
	#current size: 128x128x12
	Network = tflearn.layers.conv.conv_2d(Network, 6, 4, strides=1, activation=lambda x: tflearn.activations.leaky_relu(x, alpha=LeakyReluSlope), name="e1")
	#Network = tflearn.layers.normalization.batch_normalization(Network)
	
	#------------ 128x128 -> 256x256 ----------------------------------------------------------|
	#current size: 128x128x6
	Network = tflearn.layers.conv.upsample_2d(Network, 2)
	Network = tflearn.layers.conv.conv_2d(Network, 3, 4, strides=1, activation=lambda x: tflearn.activations.leaky_relu(x, alpha=LeakyReluSlope), name="e1")
	#Network = tflearn.layers.normalization.batch_normalization(Network)
	#current size: 256x256x3
	#Network = tflearn.layers.merge_ops.merge((NetworkIN, Network), mode='concat', axis=3)
	split0, _,_ = tf.split(NetworkIN, num_or_size_splits=3, axis=3)
	Network = tflearn.layers.merge_ops.merge((split0, Network), mode='concat', axis=3)
	#current size: 256x256x6
	Network = tflearn.layers.conv.conv_2d(Network, 3, 4, strides=1, activation='tanh')
	#Network = tflearn.layers.normalization.batch_normalization(Network)
	Network = tf.nn.l2_normalize(Network, dim = 3)
	
	#Network = regression(Network, optimizer='adam', learning_rate=LR, loss ='normal_loss_Masked', name = 'targets', batch_size=10, metric =None)
	return tflearn.DNN(Network)
