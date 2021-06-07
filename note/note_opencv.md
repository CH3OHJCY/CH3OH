# PYTHON-OPENCV 学习
CH3OH的学习笔记 - 2021.5.6
## OpenCV基本操作(笔记补充)
<font size = 5 >1. 图像读取</font>

<font color=blue>dst =  cv2.imread()</font>

<font color=red>可以在图像后面加是 “1，0，-1” 分别代表彩色图像，灰度图，alpha通道图(透明度)</font>

----------

<font size = 5 >2. 图像显示:</font>

cv.imshow:  (彩色图像是BGR)

<font color=blue>cv2.imshow(“名字”,dst)</font>

<font color=red>cv.waitKey(0),使用了imshow后一定要使用这个来给inshow时间画图</font>  

matplotlib: (彩色图像是RGB)

<font color=blue>from matplotlib import pyplot as plt</font>

<font color=blue>plt.imshow(img) (因为颜色通道顺序不同所有要翻转)</font>

翻转方法:

```	

	b,g,r=cv2.split(messi) #通道的拆分
	messi_rgb=cv2.merge((r,g,b)) #通道的融合
	plt.imshow(messi_rgb),plt.title('messi_rgb_plt')
	plt.imshow(img[:,:,::-1])
	:,:,:所有的像素  :-1是翻转
```

----------

<font size = 5 >3. 保存图像</font>

<font color=blue>cv2.imwrite(“名字”,dst)</font>

----------

<font size = 5 >4. 绘制几何图形</font>

(1)直线：  

<font color=blue>cv.line(img,start,end,color,thickness)</font>

img:要绘制的图

start,end:起点，终点

(2)圆:

<font color=blue>cv.circle(img,centerpoint,r,color,thickness)</font>

当thickness为-1会填充颜色

(3)矩阵

<font color=blue>cv.rectangle(ing,leftupper,rightdown,color,thickness)</font>

(4)添加文字

<font color=blue>cv.putText(img,text,station,font,fontsize,color,thickness,cv.LINE_AA)</font>

text:文本内容

station:位置

font:字体

fontsize:字体大小

eg：cv.putText(img,'23333',(10,50),font,4,(255,255,255),2,cv_LINE_AA)

(5)图像属性

```

	def get_image_info(image):
	    print (type(image)) #image的类型
	    print (image.shape) #图像的高，宽，通道数目
	    print (image.size)  #图像的大小
	    print (image.dtype) #字节位数
	    pixel_data = np.array(image)
	    print (pixel_data)  #输出图片的矩阵
```


(6)色彩空间的改变
<font color=blue>dst = cv.cvtColor(img,flag)</font>

flag:
cv.COLOR_BGR2HSV cv.COLOR_BGR2GRAY

----------

<font size = 5 >5. 几何变换</font>

(1)图像缩放:

<font color=blue>cv.resize(src,dsize,fx,fy,interpolation)</font>

dsize:绝对尺寸，直接体哦阿正图像大小

fx,fy:相对尺寸，将dsize设为None，然后设置fx,fy的比例因子

interpolation:插值方法

cv.INTER_LINEAR    双线性插值法

cv.INTER_NEAREST   最近邻插法

cv.INTER_AREA      像素区域重采样(默认)

cv.INTER_CUBIC     双三次插值

```

	# 2.1 绝对尺寸
	rows,cols = img1.shape[:2]
	res = cv.resize(img1,(2*cols,2*rows),interpolation=cv.INTER_CUBIC)
	# 2.2 相对尺寸
	res1 = cv.resize(img1,None,fx=0.5,fy=0.5,interpolation=cv.INTER_CUBIC)
```


(2)图像平移:

<font color=blue>cv.warpAffine(img,M,dsize)</font>

img: 输入图像

M： 2*∗3移动矩阵

对于(x,y)处的像素点，要把它移动到(x+tx,y+ty),M矩阵应如下设置：M = [1 0 tx ; 0 1 ty]

dsize: 输出图像的大小

```
	rows,cols = img1.shape[:2]
	M = M = np.float32([[1,0,100],[0,1,50]])# 平移矩阵
	dst = cv.warpAffine(img1,M,(cols,rows))
```

(3)图像旋转

<font color=blue>cv2.getRotationMatrix2D(center, angle, scale)</font>

center：旋转中心

angle：旋转角度

scale：缩放比例

返回值:M：旋转矩阵,调用cv.warpAffine完成图像的旋转

```

	# 2 图像旋转
	rows,cols = img.shape[:2]
	# 2.1 生成旋转矩阵
	M = cv.getRotationMatrix2D((cols/2,rows/2),90,1)
	# 2.2 进行旋转变换
	dst = cv.warpAffine(img,M,(cols,rows))
	#(cols,rows)这个是旋转后的图像大小规定
```

-----------

## 图像处理
<font size = 5 >1. 方框滤波</font>

<font color=blue >函数：处理结果 = cv2.boxFliter(原始图像, 目标图像深度, 核大小, normalize)</font> 

<font color=red>目标图像深度</font>：一般情况下我们将目标图像深度设置为-1，表示与原始图像一致。

<font color=red>核大小</font>：方框滤波的方框的行数和列数。

 例子：

核大小：<font color=red>(5,5)</font> ---- K = (1/25)[一个5·5的1矩阵]

核大小：<font color=red>(3,3)</font> ---- K = (1/9)[一个3·3的1矩阵]

<font color=red>normalize</font>：是否对目标图像进行归一化操作。

	(1)当normalize = 1时，方框滤波的效果和均值滤波的效果是一致的，要用邻域像素的和除以核里面像素的总个数，得到中心像素的灰度值。
	(2)当normalize = 0时，表示不进行归一化操作，此时要计算核内所有像素的和代替原始图像核中心像素的大小，需要注意的是这种方式很容易造成溢出现象，例如当图像的像素用8位表示时候，如果所求和大于255则大于255的部分舍去，直接取255(白色)

----------

<font size = 5 >2. 均值滤波 --- 抗随机噪音</font>

<font color=blue>函数：处理结果 = cv2.cv2.blur(原始图像, 滤波核) </font> (均值滤波是对中心点的邻域求算术平均和)

<font color=red>滤波核</font>：大小是5×5时，则取其自身和周围24个像素值的均值来代替当前像素值
```

    img_mean = cv2.blur(img, (5,5))
```

----------

<font size = 5 >3. 高斯滤波 --- 抗正态分布的噪声(高斯噪声)</font>

<font color=red >使用高斯公式来计算权重即为加权平均</font>

高斯滤波是对这个九个数求加权平均，其中心思想是邻域中每个点离中心点的距离不一样，不应该像均值滤波一样每个点的权重一样，<font color=red >而是离中心点越近，权值越大。</font>而每个点的权重就是高斯分布。

<font color=blue>函数：处理结果 = cv2.GaussianBlur(原始图像, ksize, sigmaX) </font>

(1)ksize为卷积核大小，即邻域大小，比如ksize为(3,3)，则对以中心点为中心点3×3的邻域做操作

(2)sigmaX为高斯核函数在X方向的标准偏差

------------

<font size = 5 >4. 中值滤波 --- 抗椒盐噪声和斑点噪声</font>

<font color=red >在规定的函数窗里按照强度值来排列像素点，选择排序像素季的中值来作为新值 --- 故常使用奇数点来进行中值计算</font>

<font color=blue>函数：处理结果=cv2.medianBlur(src, ksize)</font>
(1)ksize为卷积核大小，取一个数  

例:dst = cv2.medianBlur(image,3)

----------

<font size = 5 >5. 双边滤波</font>

<font color=blue>函数：dst = cv2.bilateralFilter(src=image, d, sigmaColor, sigmaSpace) </font>

(1)d：像素的邻域直径，如果d是非正数可有sigmaColor和sigmaSpace计算可得；

(2)sigmaColor：颜色空间的标准方差(一般尽可能的大)，这个值越大,就是该像素领域内就会有越宽广的颜色混到一起；

(3)sigmaSpace：坐标空间的标准方差(像素单位)(一般尽可能的小),越大表示越远的像素点之间会相互影响，从而使得更大的区域中足够相似的颜色获得相同的颜色

----------

<font size = 5 >6. 形态学滤波</font>

```

    import cv2 as cv
    import numpy as np 
    img = cv.imread('j.png',0)
    kernel = np.ones((5,5),np.uint8)  
    erosion = cv.erode(img,kernel,iterations = 1)

```

(1)腐蚀 - 以黑色(0)为中心把kernel内的变为黑色---一个把白变黑的过程
    
<font color=blue>函数：erosion = cv2.erode(img,kernel,iterations) </font>

    a.第一个参数：img指需要腐蚀的图

    b.第二个参数：kernel指腐蚀操作的内核，默认是一个简单的3X3矩阵

    c.第三个参数：iterations指的是腐蚀次数，省略是默认为1
   
<font color=red >形态学滤波中的kernel都是kernel = np.ones((5,5),np.uint8)这样的</font>  

(2)膨胀 - 以白色(1)为中心把kernel内的变为白色---一个把黑变白的过程

<font color=blue>函数：dilation = cv2.dilate(img,kernel,iterations) </font>

    a.第一个参数：img指需要膨胀的图

    b.第二个参数：kernel指膨胀操作的内核，默认是一个简单的3X3矩阵

    c.第三个参数：iterations指的是膨胀次数，省略是默认为1  

(3)开运算 - 先进行腐蚀再进行膨胀就叫做开运算 - 用来去除噪声

    <font color=blue>函数：opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) </font>  

(4)闭运算 - 先膨胀再腐蚀 - 用来填充前景物体中的小洞，或者前景物体上的小黑点
    <font color=blue>函数：closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel) </font>  

(5)形态学梯度 -  膨胀图和腐蚀图之差 - 腐蚀是去掉了图像的边缘信息，膨胀加强了边缘信息，那么当用膨胀后的图像减去腐蚀后的图像，就得到了图像的边缘信息

<font color=red >相对于一个描边的过程</font>

<font color=blue>函数：dst = cv.morphologyEx(src, cv.MORPH_GRADIENT,, kernel) </font>  

(6)顶帽 - 是原图像与开操作之间的差值图像 - 帽运算往往用来分离比邻近点亮一些的斑块。当一幅图像具有大幅的背景的时候，而微小物品比较有规律的情况下，可以使用顶帽运算进行背景提取

<font color=blue>函数：dst = cv.morphologyEx(source, cv.MORPH_BLACKHAT, kernel) </font>  

(7)黑帽 - 为闭运算的结果图与原图像之差 - 黑帽运算后的效果图突出了比原图轮廓周围的区域更暗的区域，所以黑帽运算用来分离比邻近点暗一些的斑块

----------

<font size = 5 >7. 漫水填充</font>

<font color=blue>函数：floodFill(image, mask, seedPoint, newVal, loDiff=None, upDiff=None, flags=None) </font>

(1)image:输入图像，可以是一通道或者是三通道。

(2)mask:该版本特有的掩膜。 单通道，8位，<font color=red >在长宽上都比原图像image多2个像素点。</font>另外，当flag为FLOORFILL_MAK_ONLY时，只会填充mask中数值为0的区域。

(3)seedPoint:漫水填充的种子点，即起始点。

(4)newVal:被填充的像素点新的像素值

(5)loDiff：表示当前的观察点像素值与其相邻区域像素值或待加入该区域的像素之间的亮度或颜色之间负差的最大值。

(6)upDiff:表示当前的观察点像素值与其相邻区域像素值或待加入该区域的像素之间的亮度或颜色之间负差的最小值。

(7)flag：

    当为CV_FLOODFILL_FIXED_RANGE 时，待处理的像素点与种子点作比较，在范围之内，则填充此像素 。（改变图像）

    CV_FLOODFILL_MASK_ONLY 此位设置填充的对像， 若设置此位，则mask不能为空，此时，函数不填充原始图像img，而是填充掩码图像.

eg:
```

	#FIXED_RANGE:
	import numpy as np
	import cv2 as cv
	img = np.zeros((512,512,3), np.uint8)
	img[:, :, : ] = 255
	cv.rectangle(img,(10,10),(502,502),(255,0,0),3)
	cv.circle(img,(230,256), 15, (100,100,100),-1)
	cv.circle(img,(282,256), 15, (100,100,100),-1)
	copyIma = img.copy()
	h, w = img.shape[:2]
	print(h, w)
	mask = np.zeros([h+2, w+2], np.uint8)
	cv.floodFill(copyIma, mask, (100, 100), (20, 0, 99), (200, 200, 200), (0, 0, 0), cv.FLOODFILL_FIXED_RANGE)  
	cv.imshow("img_1",copyIma)
	cv.imshow("img_2",img)
	cv.waitKey(0)
 
	#MASK_ONLY:
	import numpy as np
	import cv2 as cv
	image = np.zeros([400, 400, 3], np.uint8)
	image[100:300, 100:300, : ] = 255
	cv.imshow("fill_binary", image)
	cv.waitKey(10)
	mask = np.ones([402, 402, 1], np.uint8)
	mask[151:251, 151:251] = 0
	cv.floodFill(image, mask, (200, 200), (255,255,0), cv.FLOODFILL_MASK_ONLY)
	cv.imshow("img",image)
	cv.waitKey(0)

```

----------

<font size = 5 >8. 图像金字塔与图片尺寸缩放</font>

向上取样：

<font color=blue>函数：dst = cv2.pyrUp(src) </font>

    原理：在每个方向上扩大原来的2倍，新增的行列用0

向下取样：

<font color=blue>函数：dst = cv2.pyrDown(src) </font>

    原理：使用高斯核卷积后删除所有偶数行,列

高斯金字塔：使用向下取样，构成的图像金字塔结构为高斯金字塔

<font color=blue> 拉普拉斯金字塔：L = G - PyrUp(PyrDown(G)) </font>

G:原始图像  
L:拉普拉斯金字塔图像

```

	import cv2 as cv
	img = cv.imread("F:\\CISDI\\1st\\test_image\\2.jfif")
	img_0 = cv.pyrDown(img)
	img_up_0 = cv.pyrUp(img_0)
	img_la_0 = img - img_up_0
	img_1 = cv.pyrDown(img_0)
	img_up_1 = cv.pyrUp(img_1)
	img_la_1 = img_0 - img_up_1
	cv.imshow("img",img)
	cv.imshow("img_0",img_0)
	cv.imshow("img_1",img_1)
	cv.imshow("img_la_0",img_la_0)
	cv.imshow("img_la_1",img_la_1)
	cv.waitKey(0)
```

----------

<font size = 5>9. 阈值化</font>

阈值化图像其实就是对灰度图像进行二值化操作，根本原理是利用设定的阈值判断图像像素为0还是255，所以在图像二值化中阈值的设置很重要。

(1)全局阈值化：

<font color=blue>函数：retval,dst = cv2.threshold(src, thresh, maxval, type) </font>

<font color=red>该函数有两个返回值，第一个retVal（得到的阈值），第二个就是阈值化后的图像</font>

src参数表示输入图像（多通道，8位或32位浮点）

thresh参数表示阈值

maxval参数表示与THRESH_BINARY和THRESH_BINARY_INV阈值类型一起使用设置的最大值。

type参数表示阈值类型。

阈值类型:

二值阈值化 —— 像素值大于阈值的设为最大值，小于阈值的设为0    <font color=red>cv2.THRESH_BINARY </font>

反向二值阈值化 —— 像素值大于阈值的设为0，小于阈值的设为最大值    <font color=red>cv2.THRESH_BINARY_INV </font>

截断阈值化 —— 像素值大于阈值的设为阈值，小于阈值的保持原来的像素值    <font color=red>cv2.THRESH_TRUNC </font>

超过阈值被置0 —— 像素值大于阈值的置为0，小于阈值的保持原来的像素值    <font color=red>cv2.THRESH_TOZERO </font>

低于阈值被置0 —— 像素值大于阈值的保持原来的像素值，小于阈值的置为0<font color=red>cv2.THRESH_TOZERO_INV </font>

```

	import cv2 as cv
	img = cv.imread("F:\\CISDI\\1st\\test_image\\diff_1_1.jpg",0)
	retval,dst = cv.threshold(img, 178, 256,cv.THRESH_TRUNC) #这个时候“256”没有实际用途因为阈值化的方法是cv.THRESH_TRUNC
	print([type(dst)])
	cv.imshow("A",dst)
	cv.waitKey(0)
  
	import cv2 as cv
	img = cv.imread("F:\\CISDI\\1st\\test_image\\diff_1_1.jpg",0)
	retval,dst = cv.threshold(img, 178,100,cv.THRESH_BINARY) #如果这个100设为255，就是把图片进行二值化
	print([type(dst)])
	cv.imshow("A",dst)
	cv.waitKey(0)
```


(2)局部阈值化
局部阈值（局部自适应）则是根据像素的邻域块的像素值分布来确定该像素位置上的二值化阈值，自适应阈值是图像的不同的区域使用不同的阀值，而阀值是对这个区域计算得来的

常用的局部自适应阈值有：

1）局部邻域块的均值:<font color=red>cv2.ADAPTIVE_THRESH_MEAN_C</font>

2）局部邻域块的高斯加权和<font color=red>cv2.ADAPTIVE_THRESH_GAUSSIAN_C</font>

<font color=blue>函数：dst = cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C) </font>

+ src参数表示输入图像（8位单通道图像）。
+ maxValue参数表示使用 THRESH_BINARY 和 THRESH_BINARY_INV 的最大值。
+ adaptiveMethod参数表示自适应阈值算法
+ thresholdType参数表示阈值类型，必须为THRESH_BINARY或THRESH_BINARY_INV的阈值类型。
+ blockSize参数表示块大小（奇数且大于1，比如3，5，7… ）。
+ C参数是常数，表示从平均值或加权平均值中减去的数。 通常情况下，这是正值，但也可能为零或负值。阈值极为通过平均和高斯加权所计算的值在减去C。

```

	import cv2 as cv
	src = cv.imread("F:\\CISDI\\1st\\test_image\\diff_1_1.jpg",0)
	dst = cv.adaptiveThreshold(src, 200, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 9, 4)
	cv.imshow("A",dst)
	cv.waitKey(0)
```

----------

## 图像变换
<font size = 5>1. 边缘检测</font>

(1)Canny算子

<font color=blue>函数：edges = cv2.Canny(image,threshold1,threshold2)</font>

edges:边界图像

image:原始图像

threshold1:阈值1 - minVal阈值1

threshold2:阈值2 - maxVal阈值2

阈值 - 用来控制边界信息的丰富程度，这俩越小，边界信息越丰富，反之亦然。

```

	import cv2 as cv
	src = cv.imread("F:\\CISDI\\1st\\test_image\\diff.jpg",0)
	dst = cv.Canny(src,20,30)
	cv.imshow("A",dst)
	cv.waitKey(0)
```

(2)Sobel算子 --- 计算不同方向的梯度

<font color=blue>函数：dst=cv2.Sobel(src,ddepth,dx,dy,kszie)</font>

src:原始图像

ddpepth:处理图像的深度.-1表示与原始图像一样

dx:x轴方向,计算x轴方向,dx=1,dy=0

dy:y轴方向,计算y轴,dx=0,dy=1

ksize:核大小，默认3

```

	import cv2 
	img = cv2.imread("F:\\CISDI\\1st\\test_image\\diff.jpg", 0)
	x = cv2.Sobel(img,cv2.CV_64F,1,0)
	y = cv2.Sobel(img,cv2.CV_64F,0,1)
	absX = cv2.convertScaleAbs(x)   # 转回uint8
	absY = cv2.convertScaleAbs(y)
	dst = cv2.addWeighted(absX,0.5,absY,0.5,0) #两个值相加来合成一个图的数据。这个0.5是系数及个	体的权重。0代表的是修正值。
	cv2.imshow("absX", absX)
	cv2.imshow("absY", absY)
	cv2.imshow("Result", dst)
	cv2.waitKey(0)
	cv2.destroyAllWindows() 
```

<font color=red>在Sobel函数的第二个参数这里使用了cv2.CV_64F。即Sobel函数求完导数后会有负值，还有会大于255的值。而原图像是uint8，即8位无符号数，所以Sobel建立的图像位数不够，会有截断。因此要改变类型  </font>

absX = cv2.convertScaleAbs(src) #将原始图像转为256色位图


(3)Scharr算子 --- 和sobel相似只是卷积核不同

<font color=red>精确度比sobel算子高</font>
```

	import cv2 
	img = cv2.imread("F:\\CISDI\\1st\\test_image\\diff.jpg", 0)
	x = cv2.Scharr(img,cv2.CV_64F,1,0)
	y = cv2.Scharr(img,cv2.CV_64F,0,1)
	absX = cv2.convertScaleAbs(x)   # 转回uint8
	absY = cv2.convertScaleAbs(y)
	dst = cv2.addWeighted(absX,0.5,absY,0.5,0) #两个值相加来合成一个图的数据。这个0.5是系数及个体的权重。0代表的是修正值。
	cv2.imshow("absX", absX)
	cv2.imshow("absY", absY)
	cv2.imshow("Result", dst)
	cv2.waitKey(0)
	cv2.destroyAllWindows() 
```


(4)Laplacian算子  --- 两次sobel，即二阶段

<font color=blue>函数：dst = cv2.Laplacian(src, ddepth)</font>

<font color=red>使用方法类似于sobel</font>

```

	import cv2 
	img = cv2.imread("F:\\CISDI\\1st\\test_image\\diff.jpg", 0)
	x = cv2.Laplacian(img,cv2.CV_64F)
	res = cv2.convertScaleAbs(x)   # 转回uint8
	cv2.imshow("res", res)
	cv2.waitKey(0)
	cv2.destroyAllWindows() 
```

-----------

<font size = 5>2. 霍夫变换</font>

(1)标准霍夫变换 --- 用来检测图像中的直线
<font color=blue>line = cv.HoughLines(img, rho, theta, threshold)</font>

rho:半径的长度

theta:角的步距

threshold:阈值，要大于这个阈值才算直线

霍夫线检测后，输出的结构为[rho,theta]的一个矩阵

```

	#一小段矩阵结构解释
	import numpy as np
	a = np.zeros((2,3))
	print(a)
	b = a[0]
	print(b)
```

	import numpy as np
	import cv2 as cv
	img = cv.imread('F:\\CISDI\\1st\\test_image\\diff.jpg')
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	edges = cv.Canny(gray, 50, 150)
	lines = cv.HoughLines(edges, 0.8, np.pi / 180, 150)
	#画图开始
	for line in lines:
	    print(line)
	    rho, theta = line[0]
	    a = np.cos(theta)
	    b = np.sin(theta)
	    x0 = a * rho
	    y0 = b * rho
	    x1 = int(x0 + 1000 * (-b))
	    y1 = int(y0 + 1000 * (a))
	    x2 = int(x0 - 1000 * (-b))
	    y2 = int(y0 - 1000 * (a))
	    cv.line(img, (x1, y1), (x2, y2), (255, 255, 255))
	cv.imshow("out", img)
	cv.waitKey(0)
 ```


(2) 霍夫圆检测
<font color=blue>circles = cv.HoughCircles(image, method, dp, minDist, param1, param2, minRadius,maxRadius )</font>

method：使用霍夫变换圆检测的算法，它的参数是cv.HOUGH_GRADIENT

dp：霍夫空间的分辨率，dp=1时表示霍夫空间与输入图像空间的大小一致，dp=2时霍夫空间是输入图像空间的一半，以此类推

minDist为圆心之间的最小距离，如果检测到的两个圆心之间距离小于该值，则认为它们是同一个圆心

param1：边缘检测时使用Canny算子的高阈值，低阈值是高阈值的一半。

param2：检测圆心和确定半径时所共有的阈值

minRadius和maxRadius为所检测到的圆半径的最小值和最大值
```

	import cv2 as cv
	import numpy
	
	
	def hough_circle_demo(image):
	    # 霍夫圆检测对噪声敏感，边缘检测消噪
	    dst = cv.pyrMeanShiftFiltering(image, 10, 100)  # 边缘保留滤波EPF
	    gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
	    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
	    circles = numpy.uint16(numpy.around(circles))  #把circles包含的圆心和半径的值变成整数
	    for i in circles[0,:]:
	        cv.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 2)
	        cv.circle(image, (i[0], i[1]), 2, (255, 0, 0), 2)
	    cv.imshow("circle image", image)
	
	
	src = cv.imread("F:\\CISDI\\1st\\test_image\\1_m.jpg")
	cv.namedWindow("coins", cv.WINDOW_AUTOSIZE)
	cv.imshow("coins", src)
	hough_circle_demo(src)
	cv.waitKey(0)
	cv.destroyAllWindows()
```

----------


<font size = 5>3. 重映射</font>

<font color=blue>remap(src, map1, map2, interpolation)</font>
+ src是源图像数据
+ map1是用于插值的X坐标
+ map2是用于插值的Y坐标
+ interpolation是采用的插值算法  

支持的插值算法有：
+ INTER_NEAREST - 最近邻插值
+ INTER_LINEAR – 双线性插值（默认值）
+ INTER_CUBIC – 双三次样条插值（逾4×4像素邻域内的双三次插值）
+ INTER_LANCZOS4 -Lanczos插值（逾8×8像素邻域的Lanczos插值）

```

	#上下颠倒
	import cv2
	import numpy as np
	src = cv2.imread("F:/CISDI/1st/test_image/1.jpg")
	rows,cols,channels = src.shape
	img_x = np.zeros((rows,cols),np.float32)
	img_y = np.zeros((rows,cols),np.float32)
	#坐标映射
	for y in range(rows):
	    for x in range(cols):
	        img_y[y,x] = rows - y
	        img_x[y,x] = cols - x
	        
	dst = cv2.remap(src,img_x,img_y,cv2.INTER_LINEAR)
	cv2.imshow('src',src)
	cv2.imshow('dst',dst)
	cv2.waitKey()
	cv2.destroyAllWindows()
```

----------

<font size = 5>4. 仿射变换 --- 图像角度变换</font>

<font color=blue>dst = cv2.warpAffine(img, M, (cols, rows))</font>  

就是以三个点为基准来变换

```

	img = cv2.imread('Rachel.jpg')
	rows, cols, ch = img.shape
	pts1 = np.float32([[0, 0], [10, 0], [0, 10]])
	pts2 = np.float32([[10, 0], [20, 0], [0, 10]])
	# pts1是原图中的3的参考点，pts2是变化的。后面的坐标分别对于在新旧图中的坐标
	M = cv2.getAffineTransform(pts1, pts2)
	# 把pts1/2构建成矩阵
	dst = cv2.warpAffine(img, M, (cols, rows))
	cv2.imshow('image', dst)
	k = cv2.waitKey(0)
	if k == ord('s'):
	    cv2.imwrite('Rachel1.jpg', dst)
	    cv2.destroyAllWindows()
```

---------

5. 掩膜

```

	#1.创建蒙版
	mask = np.zeros(img.shape[:2], np.uint8)
	mask[400:650, 200:500] = 255
	#2.掩模
	masked_img = cv.bitwise_and(img,img,mask = mask)
```

---------

<font size = 5>6. 直方图</font>

(1)直方图提取
<font color=blue>dst = cv2.calcHist(images,channels,mask,histSize,ranges)</font>
+ images: 原图像。当传入函数时应该用中括号 [] 括起来，例如：[img]。
+ channels: 如果输入图像是灰度图，它的值就是 [0]；如果是彩色图像的话，传入的参数可以是 [0]，[1]，[2] 它们分别对应着通道 B，G，R。 　　
+ mask: 掩模图像。要统计整幅图像的直方图就把它设为 None。但是如果你想统计图像某一部分的直方图的话，你就需要制作一个掩模图像，并使用它。（后边有例子）
+ histSize:BIN 的数目(灰度级的个数)。也应该用中括号括起来，例如：[256]。 　
+ ranges: 像素值范围，通常为 [0，256]

```

	import numpy as np
	import cv2 as cv
	from matplotlib import pyplot as plt
	# 1 直接以灰度图的方式读入
	img = cv.imread('./image/cat.jpeg',0)
	# 2 统计灰度图
	histr = cv.calcHist([img],[0],None,[256],[0,256])
	# 3 绘制灰度图
	plt.figure(figsize=(10,6),dpi=100)
	plt.plot(histr)
	plt.grid()
	plt.show()
```

(2)直方图均衡化

<font color=blue>dst = cv.equalizeHist(img)</font>

<font color=red>img:是灰度图像</font>

(3)自适应的直方图均衡化

<font color=blue>dst = cv.createCLAHE(clipLimit, tileGridSize)</font>

clipLimit: 对比度限制，默认是40

tileGridSize: 分块的大小，默认为8*8

```
	
	clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

```
---------

## 图像轮廓
<font size = 5>1. findContours --- 寻找轮廓</font>

<font color=blue>contours,hierarchy = cv2.findContours(image, mode, method)</font>  

<font color=red>opencv2返回两个值：contours：hierarchy。注:opencv3会返回三个值,分别是img, countours, hierarchy</font>  

(1)image:寻找轮廓的图像；

(2)mode:第二个参数表示轮廓的检索模式，有四种(本文介绍的都是新的cv2接口)

(3)method:为轮廓的近似办法  

mode的分类:

(1)cv2.RETR_EXTERNAL表示只检测外轮廓

(2)cv2.RETR_LIST检测的轮廓不建立等级关系

(3)cv2.RETR_CCOMP建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。如果内孔内还有一个连通物体，这个物体的边界也在顶层。

(4)cv2.RETR_TREE建立一个等级树结构的轮廓。  

第三个参数method为轮廓的近似办法

(1)cv2.CHAIN_APPROX_NONE存储所有的轮廓点，相邻的两个点的像素位置差不超过1，即max（abs（x1-x2），abs（y2-y1））==1

(2)cv2.CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素，
只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息

(3)cv2.CHAIN_APPROX_TC89_L1,CV_CHAIN_APPROX_TC89_KCOS使用teh-Chinl chain 近似算法  

<font color=blue>返回值</font>

这个函数的返回值有两个(其中一个是可选的)

(1)contour:返回一个list，list中每个元素都是图像中的一个轮廓

(2)hierarchy:其中的元素个数和轮廓个数相同，每个轮廓contours[i]对应4个hierarchy元素hierarchy[i][0] ~hierarchy[i][3]，分别表示后一个轮廓、前一个轮廓、父轮廓、内嵌轮廓的索引编号，如果没有对应项，则该值为负数

<font size = 5>2. drawContours --- 轮廓绘制</font>

<font color=blue>cv2.drawContours(image, contours, contourIdx, color)</font>

第一个参数是指明在哪幅图像上绘制轮廓；

第二个参数是轮廓本身，在Python中是一个list。

第三个参数指定绘制轮廓list中的哪条轮廓，如果是-1，则绘制其中的所有轮廓。

也可以输入数字来选择画那一个轮廓(如父轮廓，子轮廓等)

```

	import numpy as np
	import cv2 as cv
	img = np.zeros((512,512,3), np.uint8)
	img[:, :, : ] = 255
	cv.rectangle(img,(10,10),(502,502),(255,0,0),3)
	cv.circle(img,(230,256), 15, (255,0,0),-1)
	cv.circle(img,(282,256), 15, (255,0,0),-1)
	gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
	ret, binary = cv.threshold(gray,127,255,cv.THRESH_BINARY)
	contours, hierarchy = cv.findContours(gray,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
	cv.drawContours(img,contours,3,(255,255,255),3) #这个3是线条粗细
	cv.imshow("img", img)
	cv.waitKey(0)
```

<font size = 5>3. convexHull --- 寻找凸包函数</font>

<font color=blue>convexHull(points, hull=None, clockwise=None, returnPoints=None)</font>

points：轮廓

hull：返回值，为凸包角点。可以理解为多边形的点坐标，或索引。

clockwise：布尔类型，为True时，凸包角点将按顺时针方向排列；为False时，为逆时针。

returnPoints：布尔类型，默认值True，函数返回凸包角点的x/y坐标；为False时，函数返回轮廓中凸包角点的索引。

```

	import cv2
	#读取图片并转至灰度模式
	img = cv2.imread('F:\\CISDI\\1st\\test_image\\2333.PNG')
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#二值化
	ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
	#图片轮廓
	contours, hierarchy = cv2.findContours(thresh, 2, 1)
	cnt = contours[0] -- 0 指contours中的父子结构
	print(contours)
	#寻找凸包并绘制凸包（轮廓）
	hull = cv2.convexHull(cnt)
	length = len(hull)
	for i in range(len(hull)):
	    #tuple - 元组，这里的作用是把[]变为()来表示坐标
	    cv2.line(img, tuple(hull[i][0]), tuple(hull[(i+1)%length][0]), (0,255,0), 2)
	#显示图片
	cv2.imshow('line', img)
	cv2.waitKey()
```

----------

## 直方图与匹配
<font size = 5>1. 计算直方图</font>

(1) calcHist函数
<font color=blue>dst = cv2.calcHist(images,channels,mask,histSize,ranges)</font>
+ images: 原图像。当传入函数时应该用中括号 [] 括起来，例如：[img]。
+ channels: 如果输入图像是灰度图，它的值就是 [0]；如果是彩色图像的话，传入的参数可以是 [0]，[1]，[2] 它们分别对应着通道 B，G，R。 　　
+ mask: 掩模图像。要统计整幅图像的直方图就把它设为 None。但是如果你想统计图像某一部分的直方图的话，你就需要制作一个掩模图像，并使用它。（后边有例子）
+ histSize:BIN 的数目(灰度级的个数)。也应该用中括号括起来，例如：[256]
+ ranges: 像素值范围，通常为 [0，256]  
  
(2) minMaxLoc函数 --- 寻找全局最大，最小值及其位置
<font color=blue>a,b,c,d = cv2.minMaxLoc(data)</font>

```

	import numpy as np
	import cv2
	a=np.array([[1,2,3,4],[5,67,8,9]])
	min_val,max_val,min_indx,max_indx=cv2.minMaxLoc(a)
	print(min_val,max_val,min_indx,max_indx)

```

	out:
	1.0 67.0 (0, 0) (1, 1)

对比RGB(彩色图片)

```

	import cv2 as cv
	from matplotlib import pyplot as plt
	img = cv.imread('F:\\CISDI\\1st\\test_image\\diff.webp')
	color = ('b', 'g', 'r')
	for i, col in enumerate(color):
	    hist = cv.calcHist([img], [i], None, [256], [0, 256])
	    plt.plot(hist, color=col)
	    plt.xlim([0, 256])
	plt.show()
```

----------

<font size = 5>2. 直方图对比</font>

用途:如果我们有两张图像，并且这两张图像的直方图一样，或者有极高的相似度，那么在一定程度上，我们可以认为这两幅图是一样的，这就是直方图比较的应用之一。

<font color=blue>dst = cv2.compareHist(H1, H2, method)</font>
H1，H2 分别为要比较图像的直方图
method - 比较方式
比较方式（method）
+ 相关性比较 (method=cv.HISTCMP_CORREL) 值越大，相关度越高，最大值为1，最小值为0
+ 卡方比较(method=cv.HISTCMP_CHISQR 值越小，相关度越高，最大值无上界，最小值0
+ 巴氏距离比较(method=cv.HISTCMP_BHATTACHARYYA) 值越小，相关度越高，最大值为1，最小值为0

```

	import cv2 as cv
	from matplotlib import pyplot as plt
	src_1 = cv.imread('F:\\CISDI\\1st\\test_image\\2_1.jfif')
	src_2 = cv.imread('F:\\CISDI\\1st\\test_image\\2_2.jfif')
	dst_1 = cv.calcHist([src_1],[0],None,[256],[0,256])
	dst_2 = cv.calcHist([src_2],[0],None,[256],[0,256])
	plt.plot(dst_1, label='A', color='b')
	plt.plot(dst_2, label='B', color='r')
	match1 = cv.compareHist(dst_1, dst_2, cv.HISTCMP_BHATTACHARYYA)
	match2 = cv.compareHist(dst_1, dst_2, cv.HISTCMP_CORREL)
	match3 = cv.compareHist(dst_1, dst_2, cv.HISTCMP_CHISQR)
	print("巴氏距离：%s, 相关性：%s, 卡方：%s" %(match1, match2, match3))
```

----------

<font size = 5>3. 反向投影</font>

就是计算图像为某一特征的直方图模型，然后使用模型去寻找图像中存在的该特征  


<font color=blue>dst = cv2.calcBackProject(images, channels, hist, ranges, scale)</font>

images ：输入图像，注意加 []；

channels：通道，通道数必须与直方图维度相匹配，

hist：图象的直方图；

ranges：直方图的变化范围；

scale：输出反投影的可选比例因子

```

	import cv2 as cv
	from matplotlib import pyplot as plt
	def back_projection_demo():
	    # 读取图片
	    test = cv.imread("test1.jpg")
	    target = cv.imread("target.jpeg")
	    # 转换为 HSV 格式
	    roi_hsv = cv.cvtColor(test, cv.COLOR_BGR2HSV)
	    target_hsv = cv.cvtColor(target, cv.COLOR_BGR2HSV)
	    cv.imshow("sample", test)
	    cv.imshow("target", target)
	    # 计算直方图
	    roiHist = cv.calcHist([roi_hsv], [0, 1], None, [64, 64], [0, 180, 0, 256])
	    # 获取直方图的反向投影
	    dst = cv.calcBackProject([target_hsv], [0, 1],
	                             roiHist, [0, 180, 0, 256], 1)
	    cv.imshow("back_projection_demo", dst)
	back_projection_demo()
	cv.waitKey(0)
	cv.destroyAllWindows()
```

----------

<font size = 5>4. 通道复制</font>

----------

<font size = 5>5. 模板匹配</font>

通过 滑动, 我们的意思是图像块一次移动一个像素 (从左往右,从上往下). 在每一个位置, 都进行一次度量计算来表明它是 “好” 或 “坏” 地与那个位置匹配 (或者说块图像和原图像的特定区域有多么相似).
对于 T 覆盖在 I 上的每个位置,你把度量值 保存 到 结果图像矩阵 (R) 中. 在 R 中的每个位置 (x,y) 都包含匹配度量值:

<font color=blue>result = cv2.matchTemplate(source, template, match_method)</font>

source:大图

template:目标截取图

match_method:匹配算法  

6种模板匹配算法:

(1)平方差匹配法CV_TM_SQDIFF（最好匹配为0.匹配越差,匹配值越大）

(2)归一化平方差匹配法CV_TM_SQDIFF_NORMED

(3)相关匹配法CV_TM_CCORR
这类方法采用模板和图像间的乘法操作,所以较大的数表示匹配程度较高,0标识最坏的匹配效果

(4)归一化相关匹配法CV_TM_CCORR_NORMED

(5)相关系数匹配法CV_TM_CCOEFF
这类方法将模版对其均值的相对值与图像对其均值的相关值进行匹配,1表示完美匹配,-1表示糟糕的匹配,0表示没有任何相关性(随机序列)

(6)归一化相关系数匹配法CV_TM_CCOEFF_NORMED

+ 随着从简单的测量(平方差)到更复杂的测量(相关系数),我们可获得越来越准确的匹配(同时也意味着越来越大的计算代价).  

```

	import cv2 as cv
	import numpy as np
	def template_demo():
	    tp1 =cv.imread("F:\\CISDI\\1st\\test_image\\diff_1_1.jpg")
	    target = cv.imread("F:\\CISDI\\1st\\test_image\\diff.jpg")
	    cv.imshow("tpl",tp1)
	    cv.imshow("target",target)
	    methods =[cv.TM_SQDIFF_NORMED,cv.TM_CCORR_NORMED,cv.TM_CCOEFF_NORMED]
	    th ,tw=tp1.shape[:2]
	    for md in methods:
	        print(md)
	        result =cv.matchTemplate(target,tp1,md)
	        #来找画方框的起点
	        min_val,max_val,min_loc,max_loc=cv.minMaxLoc(result)
	        if md==cv.TM_SQDIFF_NORMED:
	            tl=min_loc
	        else:
	            tl=max_loc
	        br =(tl[0]+tw,tl[1]+th)
	        cv.rectangle(target,tl,br,(0,0,255),2)
	        cv.imshow("match"+np.str(md),target)
	template_demo()
	cv.waitKey(0)
	cv.destroyAllWindows()
```

----------

## 图像轮廓于图像修补(笔记补充 - 承接凸包)
<font size = 5>1. 使用多边形来包围轮廓</font>
<font color=red>这些输入图都是二值化的图</font>

(1) 外部矩阵边界 
<font color=blue>x, y, w, h = cv2.boundingRect(cnt)</font>
(x,y)是矩阵的左上点坐标
(x+w,y+h)是矩阵的右下点坐标

```

	import cv2
	#用绿色(0, 255, 0)来画出最小的矩形框架
	x, y, w, h = cv2.boundingRect(cnt)
	cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
```
  

(2)最小包围矩阵
<font color=blue>
rect = cv2.minAreaRect(cnt) # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）</font>
<font color=blue>
box = cv2.boxPoints(rect) # 获取最小外接矩形的4个顶点坐标(ps: cv2.boxPoints(rect) for OpenCV 3.x)</font>

```

	import cv2
	import numpy as np
	img = cv2.imread('F:\\CISDI\\1st\\test_image\\2333.PNG')
	cnt = np.array([[200,250], [250,300], [300, 270], [270,200], [120, 240]], np.int32)必须是array数组的形式 -- (这里的例子是一个四边形)
	rect = cv2.minAreaRect(cnt) # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
	box = cv2.boxPoints(rect) # 获取最小外接矩形的4个顶点坐标(ps: cv2.boxPoints(rect) for OpenCV 3.x)
	box = np.int0(box) #把浮点数变为整数
	#画出来
	cv2.drawContours(img, [box], 0, (255, 255, 0), 2) #这个box是个四点轮廓list
	cv2.imshow("good",img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
```
  
(3)最小包围圆形
<font color=blue>
((x, y), radius) = cv2.minEnclosingCircle(c)）</font>

```

	import cv2
	img = cv2.imread('F:\\CISDI\\1st\\test_image\\2333.PNG')
	thresh = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	((x, y), radius) = cv2.minEnclosingCircle(contours[0])
	print(contours[0])
	((x, y), radius) = cv2.minEnclosingCircle(contours[0])
	img = cv2.circle(img, (int(x), int(y)), int(radius),(255, 255, 0), 2)
	cv2.imshow("good",img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
```
  
(4)椭圆多边形二维点集
<font color=blue>ellipse = cv2.fitEllipse(cnt)</font>
ellipse = [(x,y),(a,b),angle]
(x,y)代表椭圆中心点的位置
(a,b)代表长短轴长度，应注意a、b为长短轴的直径，而非半径
angle 代表了中心旋转的角度
<font color=blue>cv2.ellipse(src, ellipse, (color), width)</font>

```

	import cv2
	img = cv2.imread('F:\\CISDI\\1st\\test_image\\2333.PNG')
	thresh = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	ellipse = cv2.fitEllipse(contours[0])
	cv2.ellipse(img, ellipse, (255,255,0),3)
	cv2.imshow("good",img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

```
  
(5)逼近多边形曲线
<font color=blue>approx = cv2.approxPolyDP(contours,epsilon,True)</font>
epsilon:阈值，若大于这个阈值则该点变为断点，这个值越小越准确
True:线段会闭合 若改为False则为开

```

	import cv2
	img = cv2.imread('F:\\CISDI\\1st\\test_image\\2333.PNG')
	thresh = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	approx = cv2.approxPolyDP(contours[0], 4, True)
	cv2.polylines(img, [approx], True, (255, 0, 0), 2)
	cv2.imshow("good",img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
```
  
----------
<font size = 5>2. 图像的矩</font>

----------
<font size = 5>3. 分水岭算法</font>

<font color=blue>dst = cv2.inpaint（src，mask, inpaintRadius，flags）</font>
流程:
1，输入图像 2，变换为灰度图像 3，二值化图像 4，距离变换 5，寻找种子点 6，生成marker 7，分水岭变换 8，输出图像


```

	import numpy as np
	import cv2 as cv
	
	
	#将图片进行二值化
	img = cv.imread('F:\\CISDI\\1st\\test_image\\water.png')
	gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
	ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
	
	
	#减噪处理
	kernel = np.ones((3,3),np.uint8)
	opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
	
	# 形态学膨胀
	sure_bg = cv.dilate(opening,kernel,iterations=3)
	
	# Finding sure foreground area
	#函数distanceTransform()用于计算图像中每一个非零点像素与其最近的零点像素之间的距离，输出的是保存每一个非零点与最近零点的距离信息
	#可以根据距离变换的这个性质，经过简单的运算，用于细化字符的轮廓和查找物体质心(中心)
	dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
	
	ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
	
	# Finding unknown region
	sure_fg = np.uint8(sure_fg)
	unknown = cv.subtract(sure_bg,sure_fg)
	
	# Marker labelling
	ret, markers = cv.connectedComponents(sure_fg)
	# Add one to all labels so that sure background is not 0, but 1
	markers = markers+1
	# Now, mark the region of unknown with zero
	markers[unknown==255] = 0
	
	
	markers = cv.watershed(img,markers)
	img[markers == -1] = [255,0,0]
	
	cv.imshow("good",img)
	cv.imshow("goood",unknown)
	cv.waitKey(0)
	cv.destroyAllWindows()
```

----------
<font size = 5>4. 图像修补</font>

<font color=blue>dst = cv2.inpaint（src，mask, inpaintRadius，flags）</font>
src：输入8位1通道或3通道图像。
inpaintMask：修复掩码，8位1通道图像。非零像素表示需要修复的区域。
dst：输出与src具有相同大小和类型的图像。
inpaintRadius：算法考虑的每个点的圆形邻域的半径。
flags：(两种方法)
+ cv2.INPAINT_TELEA
+ cv2.INPAINT_NS
----------

## 角点检测
<font size = 5>1. Harris 角点检测</font>

<font color=blue>dst = cv.cornerHarris(src,blockSize,ksize,k)</font>
src:输入的类型要是float32
blockSize:角点检测中要考虑的邻域大小
ksize:soble求导使用的核大小
k:角点检测方程中的自由参数，取值参数为:[0.04,0.06]

```

	import cv2 as cv
	import numpy as np
	import matplotlib.pyplot as plt
	img = cv.imread('F:\\CISDI\\1st\\test_image\\2_1.jfif')
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	
	gray = np.float32(gray)
	
	dst = cv.cornerHarris(gray,4,5,0.04)
	
	#变量a的阈值为0.01 * dst.max()，如果dst的图像值大于阈值，那么该图像的像素点设为True，否则为False
	img[dst>0.001*dst.max()] = [255,255,0]
	a = dst>0.01 * dst.max() #a是一个由1，0构成的bool的矩阵

	plt.figure(figsize = (10,8),dpi=100)
	cv.imshow("good",img)
	cv.waitKey(0)
```

----------
<font size = 5>2. Shi-Tomasi 角点检测</font>

<font color=blue> corners = cv2.goodFeaturesToTrack(src,maxcorners,qualityLevel,minDistance)</font>
src:灰度图
maxcorners:获得角点的数目
qualityLevel:最低可接受角点质量水平，在[0，1]之间
minDistance:角点之间的最小欧式距离
返回:搜索到的角点

```

	#numpy 中ravel函数
	i = [[132,200]]
	x,y = np.ravel(i)
	#输出:x是132，y是200。这个函数的作用是把多维拆开
	#for loop
	for i in cor:
	    x,y = np.ravel(i) #或者x,y = i.ravel()

```

	import cv2 as cv
	import numpy as np
	import matplotlib.pyplot as plt
	img = cv.imread('F:\\CISDI\\1st\\test_image\\2_1.jfif')
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	
	cor = cv.goodFeaturesToTrack(gray,100,0.4,10)
	
	for i in cor:
	    x,y = np.ravel(i)
	    cv.circle(img,(x,y),2,(255,255,0),-1)
	
	cv.imshow("good",img)
	cv.waitKey(0)

----------  
  
  
<font size = 5>2. SIFT/SURF 算法 --- 这两个玩意都是受到版权保护的，用不了</font>

<font color=red>实例化sift</font>
gray = cv.imread('F:\\CISDI\\1st\\test_image\\2_1.jfif',0)

<font color=blue>sift = cv.xfeatures2d.SIFT_create()</font>
创建sift对象

<font color=blue>kp,des = sift.detectAndCompute(gray,None)</font>
检测观察点并计算    
gray 是进行关键点检索的图，是灰度图
kp 是关键点信息，包括位置，尺寸，方向信息
des 是关键点描述符，每个关键点对应128各梯度信息的特征向量

<font color=blue>cv.drawKeypoints(image, keypoints, outImage,color，flags)</font>
把观察点画在图上
image 是原始图像
keypoints 是关键点信息，将其绘制在图像上
outImage 是输出图像可以是原始图
color 是颜色参数
flags 是绘图功能标识设置
+ cv.DRAW_MATCHS_FLAGS_DEFAULT 创建图像矩阵，对每一个关键点只绘制中间点
+ cv.DRAW_MATCHS_FLAGS_DRAW_RICH_KEYPOINTS 对每一个特征点绘制带大小和方向的关键点图形

```

	import cv2 as cv
	import numpy as np
	import matplotlib.pyplot as plt
	
	#实例化sift
	
	img = cv.imread('F:\\CISDI\\1st\\test_image\\2_1.jfif')
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	
	sift = cv.xfeatures2d.SIFT_create() #创建sift对象
	kp,des = sift.detectAndCompute(gray,None)
	cv.drawKeypoints(img, kp,img,cv.DRAW_MATCHS_FLAGS_DRAW_RICH_KEYPOINTS )
	
	cv.imshow("good",img)
	cv.waitKey(0)
```

<font color=red>实例化suft</font>

```

	import cv2 
	import numpy as np 
	
	img = cv2.imread('F:\\CISDI\\1st\\test_image\\2_1.jfif')
	
	#参数为hessian矩阵的阈值
	surf = cv2.xfeatures2d.SURF_create(400)
	#找到关键点和描述符
	key_query,desc_query = surf.detectAndCompute(img,None)
	#把特征点标记到图片上
	img=cv2.drawKeypoints(img,key_query,img)
	
	cv2.imshow('sp',img)
	cv2.waitKey(0)
```

<font size = 5>3.FAST 算法</font>

<font color=red>实例化fast</font>
<font color=blue>fast = cv.FastFeatureDetector_create(threshold,nonmaxSuppresion)</font>
threshold 阈值t，默认值为10
nonmaxSuppression 是否进行非极大值抑制，默认值为True
fast 创建的FastFeatureDetector对象

<font color=blue>kp = fast.detect(grayImg,None)</font>
kp 是关键点信息，包括位置，尺度，方向信息

```

	import cv2 as cv
	import numpy as np
	import matplotlib.pyplot as plt
	
	img = cv.imread('F:\\CISDI\\1st\\test_image\\2_1.jfif')
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	
	fast = cv.FastFeatureDetector_create(30) 
	kp = fast.detect(gray,None)
	cv.drawKeypoints(img, kp,img,(255,255,0))
	cv.imshow("good",img)
	cv.waitKey(0)
```

<font size = 5>4.ORB 算法</font>

<font color = blue>orb = cv.xfeatures2d.orb_create(nfeatures)</font>
nfeatures 特征点的最大数量
<font color = blue>kp,des = orb.detectAndCompute(grat,None)</font>
kp 关键点信息
des 关键点描述，每个关键点BRIEF特征向量，二进制字符串
cv.drawKeypoints(image，keypoints，outputimage，color)

```

	import cv2 as cv
	import numpy as np
	import matplotlib.pyplot as plt
	
	img = cv.imread('F:\\CISDI\\1st\\test_image\\2_1.jfif')
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	
	fast = cv.ORB_create(50) 
	kp,des = fast.detectAndCompute(gray,None)
	
	img2 = cv.drawKeypoints(img, kp,img,(255,255,0))
	cv.imshow("good",img2)
	cv.waitKey(0)
```
-----------

##参考:
[黑马程序员人工智能教程_10小时学会图像处理OpenCV入门教程](https://www.bilibili.com/video/BV1Fo4y1d7JL?p=14) 

[CH3OH的CSDN-OpenCV收藏夹](https://i.csdn.net/#/user-center/collection-list?type=1&spm=1000.2115.3001.4506)