import PIL.Image as image
import tensorflow as tf
import os
import numpy as np
import scipy
#import two mainly used


#下面这个狗屎是用来记录你图片读取的文件夹的
filePath="root1"

#right down is to def the function to open a file into image

# 下面这个是用打开图片的，讲图片转化成pil 的image 对象，filepath是用来记录文件夹路劲的
# 使用yield 方便循环遍历（其实yield 本质就是返回一个tuple 数据）
def openFIle(path=filePath):
#对每个在path 中的文件
	for file_name in os.listdir(path):
		# 一旦它的后缀是png就把它转成image
		if file_name.endswith('.png'):
		   file_path=os.path.join(path,file_name)
		   im=image.open(file_path).convert('L')
		   yield im

#下面这个是转成numpyarray，将openfile中的每个pil image 转成 np array
def returnArray(path=filePath):
	arr=[]
	arr2=[]
	# 预设空list 用于容纳img 对象中的像素值
	for im in openFIle(path):
# 便利大小便于全部输出
		for i in range(im.size[0]):
			for j in range(im.size[1]):
				#print((i,j))
				
				pix=1.0 - float(im.getpixel((i, j)))/255.0

				arr.append(pix)
		# 转numpy
		img_obtain=np.array(arr).reshape((im.size[0],im.size[1]))
		arr=[]
		arr2.append(img_obtain)
	#arr2 stores all the images
	return arr2

# 下面这玩意是用于补方的
def resize_and_normalize_image(img):  
    # 补方  
   # print(img.shape)
   # 需要补的大小，是两者差的一半，一半的目的是一边补一半
    pad_size = abs(img.shape[0]-img.shape[1]) // 2  
    if img.shape[0] < img.shape[1]:  
        pad_dims = ((pad_size, pad_size), (0, 0))  
    else:  
        pad_dims = ((0, 0), (pad_size, pad_size))  

#下面这个函数可以去np查一下
    img = np.lib.pad(img, pad_dims, mode='constant', constant_values=0)  
    # 缩放  
    img = scipy.misc.imresize(img, (64 - 4*2, 64 - 4*2))  
    img = np.lib.pad(img, ((4, 4), (4, 4)), mode='constant', constant_values=0)  
    # 用于报错的
    assert img.shape == (64, 64)  
 
    return img  


# 把前面导出的未补方的转到这里补方后转到
def returnNormaliezedArray(arr):
	
	arr2=[]
	for i in range(len(arr)):
		
		img=resize_and_normalize_image(arr[i])

		arr2.append(img)
	return arr2
# 这里就是输出正则化的图片。由于正则化或则这里的图片有点问题，导致图片显示为纯黑色
# 注意im与img区别
def reloadPic(arr):
	index=0
	for im in arr:
		index=index+1


		im=im.reshape([64,64])
		img = image.new('L', (64, 64), 255) 
		for i in range(64): 
			for j in range(64): 
				img.putpixel((i, j), 255 - int(im[i][j] * 255.0)) 
		# 下面这句话是关键的存储
		img.save((str)(index)+".png")

# 这里是把arr里的图片扁掉，[64,64]的转化成[64*64]
def flatArr(arr):
	arr2=[]
	for y in arr:
		y.reshape([-1,64*64])
		arr2.append(y)
	# arr2转化成[?,64*64]形状的，为了之后好放进去
	arr2=np.array(arr2).reshape([-1,64*64])
	return arr2





# 这个testImg是测试集中随便挑选的

testImg=returnNormaliezedArray(returnArray("test"))
# 下面得到的就是图片的拉扁后的np array
# arr 是训练集中是的
arr=flatArr(returnNormaliezedArray(returnArray(filePath)))
# arr2是训练集中不是的
arr2=flatArr(returnNormaliezedArray(returnArray("root1_test")))


# 下面整体就是一个两层卷积，第一层卷积核大小为3*3 步长为1
# 第二层卷积核大小为4*4 步长为1
#池化都选择的max_pool， 步长和窗口大小都选择的2*2

x = tf.placeholder(tf.float32, shape=[None,64*64])

y_=tf.placeholder(tf.float32, shape=[None,1])
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.075)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
#right down is to define the functions
#the shape down is used as the input
def conv2d(input,filter):
	return tf.nn.conv2d(input,filter,strides=[1,1,1,1],padding="VALID")
def max_pooling_1(input):
	return tf.nn.max_pool(input,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
# 输出为20个通道数
w1=weight_variable([3,3,1,20])
b1=bias_variable([20])
x_image=tf.reshape(x,[-1,64,64,1])
convd1=conv2d(x_image,w1)
pooling1=max_pooling_1(tf.nn.relu(convd1+b1))
# 第二层输出选择是60个通道数
w2=weight_variable([4,4,20,60])
b2=bias_variable([60])
convd2=conv2d(pooling1,w2)
pooling2=max_pooling_1(tf.nn.relu(convd2+b2))

# 最后由于选择的是padding 是valid，所以最后图片的大小事14*14
w_fc=weight_variable([14*14*60,1024])
b_fc=bias_variable([1024])


pooling_flat=tf.reshape(pooling2,[-1,14*14*60])
out=tf.nn.relu(tf.matmul(pooling_flat,w_fc)+b_fc)
W_fc2 = weight_variable([1024, 1])
b_fc2 = bias_variable([1])
yprev=tf.matmul(out, W_fc2) + b_fc2
# 由于是二分类问题所以选择的正则化
# sigmoid是在二分类问题中我认为比较优秀的函数。可是其太可怕了，最后的斜率太小导致无法正常训练。
# 抽时间将标签换为0.75 与0.25试试
y_conv=tf.sigmoid(tf.matmul(out, W_fc2) + b_fc2)

sess=tf.Session()
# 损失函数，使用的平方差
loss = tf.reduce_sum(abs((y_*y_)-(y_conv*y_conv)))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
correct_prediction = (y_- y_conv<0.5)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# 由于标签标签与
all=sess.run(tf.global_variables_initializer())

# tem1里面就是用来放标签1用的一个np array
tem1=[]
for i in range(len(testImg)):

	tempY=[]
	tempY.append(1)
	tem1.append(tempY)

tem1=np.array(tem1).reshape([-1,1])

tem=[]
# tem 同理
for i in range(len(arr)):

		tempY=[]
		tempY.append(1)
		tem.append(tempY)
tem=np.array(tem).reshape([-1,1])
tem3=[]
for i in range(len(arr2)):

		tempY=[]
		tempY.append(0)
		tem3.append(tempY)
tem3=np.array(tem3).reshape([-1,1])

batch_xs, batch_ys =arr,tem


# print(sess.run(yprev,feed_dict={x: arr2, y_: tem3}))
# 下面两句话都是在用数据集跑
sess.run(train_step, feed_dict={x: arr2, y_: tem3})
sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})	

print("later")
# 下面的两句话都是在用数据集跑前面的数字出来
# yprev就是未用sigmoid 之前的数字
print(sess.run(accuracy,feed_dict={x: batch_xs, y_: batch_ys}))
print(sess.run(yprev,feed_dict={x:arr2, y_: tem3}))
print(sess.run(yprev,feed_dict={x:flatArr(testImg),y_:tem1}))







	





