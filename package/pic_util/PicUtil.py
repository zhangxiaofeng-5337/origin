import cv2 as cv
import numpy as np


def img_resolve(img_src="/Users/xiaofengzhang/Downloads/opencv_file/sample1.jpg",
                img_out_src='/Users/xiaofengzhang/Downloads/opencv_file/temp/',
                thresh=150,
                img_temp_size_in=24,
                img_temp_size_out=28):
    img_src = "/Users/xiaofengzhang/Downloads/opencv_file/sample1.jpg"
    thresh = 150
    # 设置切图不含边框的size
    img_temp_size_in = 24
    # 设置切图含边框的size
    img_temp_size_out = 28
    # 画面输出路径
    img_out_src = '/Users/xiaofengzhang/Downloads/opencv_file/temp/'

    '''
    === 1.图片对象灰阶处理 ===
    OpenCV:imread
    参数1:path  文件路径
    参数2:Flags 图片的颜色类型
    IMREAD_ANYCOLOR = 4  任何可能的颜色格式读取图像
    IMREAD_ANYDEPTH = 2  输入具有相应深度时返回16位/ 32位图像，否则将其转换为8位
    IMREAD_COLOR = 1     3通道BGR彩色图像
    IMREAD_GRAYSCALE = 0 单通道灰度图像
    '''
    img_blue = cv.imread(img_src, cv.IMREAD_GRAYSCALE)
    '''
    === 2.获取图片的长宽 ===
    '''
    h, w = img_blue.shape

    '''
    === 3.针对灰度图进行二值化 ===
    OpenCV:threshold 灰阶图片二值化
    参数1:image  图片对象
    参数2:thresh 用于二值化的界定值
    参数3:maxval 根据二值化类型设置的最大色值
    参数4:二值化类型
         THRESH_BINARY:    大于thresh变maxval，小于等于thresh变0
         THRESH_BINARY_INV:大于thresh变0，小于等于thresh变maxval
         THRESH_TRUNC:     大于thresh变thresh，小于等于thresh不变
         THRESH_TOZERO:    大于thresh不变，小于等于thresh置0
         THRESH_TOZERO_INV:大于thresh置0，小于等于thresh不变
    返回值1:ret   与参数thresh一致
    返回值2:image 二值化结果图像
    '''
    ret, image_thresh_binary = cv.threshold(img_blue, thresh, 255, cv.THRESH_BINARY_INV)
    '''
    === 4.获取二值图的轮廓线 ===
    OpenCV:findContours 获取图片轮廓线
    参数1:image  图片对象
    参数2:mode   轮廓的检索模式，有四种
         RETR_EXTERNAL 表示只检测外轮廓
         RETR_LIST     检测的轮廓不建立等级关系
         RETR_CCOMP    建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息
         RETR_TREE     建立一个等级树结构的轮廓
    参数3:method 轮廓的近似办法
         CHAIN_APPROX_NONE   存储所有的轮廓点
         CHAIN_APPROX_SIMPLE 压缩水平方向，垂直方向，对角线方向的元素
         CHAIN_APPROX_TC89_L1
         CV_CHAIN_APPROX_TC89_KCOS 使用teh-Chinl chain 近似算法
    返回值1:contours  轮廓本身
    返回值2:hierarchy 每条轮廓对应的属性
    '''
    contours, hierarchy = cv.findContours(image_thresh_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    '''
    === 5.绘制轮廓线 ===
    OpenCV:drawContours 轮廓线绘制
    参数1:image 图片对象
    参数2:contours 轮廓本身（是一个list）
    参数3:contourIdx 绘制哪条轮廓 -1 绘制全部
    参数4:color 颜色
    参数5:thickness 轮廓线宽度
    参数6:lineType 轮廓线类型
    参数6:hierarchy
    参数7:maxLevel
    参数8:offset
    '''
    cv.drawContours(image_thresh_binary, contours, -1, 255, 1)
    '''
    === 6.获取各条轮廓线的极值(返回值为左，上，右，下四个极值的坐标范围) ===
    '''
    contours_most = []
    for i in range(len(contours)):
        pentagram = contours[i]
        left_most = tuple(pentagram[:, 0][pentagram[:, :, 0].argmin()])[0]
        right_most = tuple(pentagram[:, 0][pentagram[:, :, 0].argmax()])[0]
        top_most = tuple(pentagram[:, 0][pentagram[:, :, 1].argmin()])[1]
        bottom_most = tuple(pentagram[:, 0][pentagram[:, :, 1].argmax()])[1]
        contours_most.append([left_most, top_most, right_most, bottom_most])
    print(contours_most)
    '''
    === 7.根据极值范围进行图片剪切,另存 ===
    '''
    # 循环轮廓线
    for i in range(len(contours_most)):
        item = contours_most[i]
        img_temp = image_thresh_binary[item[1]:item[3], item[0]:item[2]]
        h, w = img_temp.shape
        # 如果轮廓图在100像素以下则忽略
        if h * w < 100:
            continue
        # 针对切图进行切图不含边框的resize
        fx = w / h
        if fx < 1:
            w = int(img_temp_size_in * fx)
            h = img_temp_size_in
        else:
            w = img_temp_size_in
            h = int(img_temp_size_in / fx)
        img_temp = cv.resize(img_temp, (w, h))
        # 获取背景图片对象（含边框）
        img_out = np.zeros([img_temp_size_out, img_temp_size_out], img_temp.dtype)
        # 融合背景图 和 切图
        rh = int((img_temp_size_out - h) / 2)
        rw = int((img_temp_size_out - w) / 2)
        roi = img_out[rh:h + rh, rw:w + rw]
        dst = cv.addWeighted(img_temp, 1, roi, 0, 0)
        img = img_out.copy()
        img[rh:h + rh, rw:w + rw] = dst
        cv.imwrite(img_out_src + 'temp' + str(i) + '.png', img)


# # 图片处理预览
# cv.namedWindow('Image')
# # cv.imshow('Image', dst)
# cv.imshow('Image', image_thresh_binary)
# # cv.imshow('Image', img_blue)
# cv.waitKey(0)
# cv.destroyWindow('Image')
