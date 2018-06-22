import PIL.Image as image
import tensorflow as tf
import os
import numpy as np
import scipy
#import two mainly used


#下面这个狗屎是用来记录你图片读取的文件夹的
trainingPath="trainingSet"
testPath="testSet"

#right down is to def the function to open a file into image

# 下面这个是用打开图片的，讲图片转化成pil 的image 对象，filepath是用来记录文件夹路劲的
# 使用yield 方便循环遍历（其实yield 本质就是返回一个tuple 数据）
def openFIle(path):
#对每个在path 中的文件
	for file_name in os.listdir(path):
		# 一旦它的后缀是png就把它转成image
		tag=18
		count=0
		if file_name.endswith('.png'):
			tempFileName=file_name
			file_path=os.path.join(path,tempFileName)
			im=image.open(file_path).convert('L')
			if file_name.endswith('00.png'):
				count=count+1

				tag=0
			elif file_name.endswith('01.png'):
				count=count+1
				# tag=0
				tag=1

			elif file_name.endswith('02.png'):
				count=count+1
				tag=2
			elif file_name.endswith('03.png'):
				count=count+1
				tag=3
			elif file_name.endswith('04.png'):
				count=count+1
				tag=4
			elif file_name.endswith('05.png'):
				count=count+1
				tag=5
				# tag=1
			elif file_name.endswith('06.png'):
				count=count+1
				tag=6
			elif file_name.endswith('07.png'):
				count=count+1
				tag=7
			elif file_name.endswith('08.png'):
				count=count+1
				tag=8

			elif file_name.endswith('10.png'):
				count=count+1
				tag=9
			elif file_name.endswith('11.png'):
				count=count+1
				tag=10
			elif file_name.endswith('12.png'):
				count=count+1
				tag=11
			elif file_name.endswith('13.png'):
				tag=12
			elif file_name.endswith('14.png'):
				count=count+1
				tag=13
			elif file_name.endswith('15.png'):
				count=count+1
				tag=14
			elif file_name.endswith('16.png'):
				count=count+1
				tag=15

			yield im,tag
# 下面这玩意是用来跑tag number 转化为tag one-hot的
def transferTagNumber(tag):
	arr=[]
	for i in range(16):
		arr.append(0)
	arr[tag]=1
	arr=np.array(arr).reshape([-1,1])
	return arr

#下面这个是转成numpyarray，将openfile中的每个pil image 转成 np array
def returnArray(path):
	arr=[]
	arr2=[]
	tagArr=[]
	# 预设空list 用于容纳img 对象中的像素值
	for im,tag in openFIle(path):
# 便利大小便于全部输出
		for i in range(im.size[0]):
			for j in range(im.size[1]):
				#print((i,j))
				
				pix=float(im.getpixel((i, j)))

				arr.append(pix)
		tagArr.append(transferTagNumber(tag))
		# 转numpy
		img_obtain=np.array(arr).reshape((im.size[0],im.size[1]))
		arr=[]
		arr2.append(img_obtain)
	#arr2 stores all the images
	tagArr=np.array(tagArr).reshape([-1,16])
	return arr2,tagArr

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
    img = np.lib.pad(img, pad_dims, mode='constant', constant_values=255.0)  
    # 缩放  
    img = scipy.misc.imresize(img, (64 - 4*2, 64 - 4*2))  
    img = np.lib.pad(img, ((4, 4), (4, 4)), mode='constant', constant_values=255.0)  
    img=(255-img)/255.0

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
				img.putpixel((i, j), (255 - int(im[i][j] * 255.0))) 
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
# traningset
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.01)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
trainingSetImg=flatArr(returnNormaliezedArray(returnArray(trainingPath)[0]))
trainingSetLabel=returnArray(trainingPath)[1]

testSetImg=flatArr(returnNormaliezedArray(returnArray(testPath)[0]))
testSetLabel=returnArray(testPath)[1]
import minst.input_data as input_data
mnist=input_data.read_data_sets("MNIST_data/", one_hot=True)
x=tf.placeholder(tf.float32,[None,28*28])
y_=tf.placeholder(tf.float32,[None,10])

w=weight_variable([28*28,10000])
b=bias_variable([10000])
y1=tf.nn.relu(tf.matmul(x,w) + b)
w2=weight_variable([10000,30000])
b2=bias_variable([30000])
y2 = tf.nn.relu(tf.matmul(y1,w2) + b2)
w3=weight_variable([30000,10])
b3=bias_variable([10])
y=tf.nn.softmax(tf.matmul(y2,w3)+b3)
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess=tf.Session()
sess.run(tf.global_variables_initializer())
# sess.run(train_step, feed_dict={x: , y_: trainingSetLabel})
# print(sess.run(accuracy,feed_dict={x: trainingSetImg, y_: trainingSetLabel}))
# 下面的两句话都是在用数据集跑前面的数字出来
# yprev就是未用sigmoid 之前的数字
# print(sess.run(accuracy,feed_dict={x: testSetImg, y_: testSetLabel}))
for i in range(600):
  batch_xs, batch_ys = mnist.train.next_batch(50)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  print(sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))
 
   
print("the final number is")

