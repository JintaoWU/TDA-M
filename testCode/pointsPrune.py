"""
Author: Jintao Wu
Time:2020-04-03
test for Pruning
"""
from matplotlib import pyplot as plt
import numpy as  np
import cv2
from testCode.hullTest import testPoint
import math
import skimage
import time

def draw(points,result):
    p=np.array(points)
    x=p[:,0]
    y=p[:,1]

    # s为点的大小
    plt.scatter(x, y, c='r', s=5,linewidths=None)

    length = len(result)
    for i in range(0, length - 1):
        plt.plot([result[i][0], result[i + 1][0]], [result[i][1], result[i + 1][1]], c='g')
    plt.plot([result[0][0], result[length - 1][0]], [result[0][1], result[length - 1][1]], c='g')

    plt.show()

#最大联通区域
def largestConnectComponent(bw_img):
    labeled_img, num = skimage.measure.label(bw_img, neighbors=4, background=0, return_num=True)
    if num==1:
        lcc = (labeled_img == 1)
        return lcc
    max_label = 0
    max_num = 0
    for i in range(1, num+1): # 这里从1开始，防止将背景设置为最大连通域
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i
    lcc = (labeled_img == max_label)

    return lcc

#生成单棵树的RGB图片
def treeRGB(points):
    img = np.zeros((224, 224, 3), np.uint8)
    for p in points:
        img[p[0],p[1]]=[255,255,255]
    return img

#计算距离
def computeDist(points):
    p1=points[0]
    p2=points[1]
    return math.sqrt(math.pow(p1[0]-p2[0],2)+math.pow(p1[1]-p2[1],2))

def computePDist(p1,p2):
    return math.sqrt(math.pow(p1[0]-p2[0],2)+math.pow(p1[1]-p2[1],2))

#提取边缘信息，为了减少凸包算法的时间
def Edge(img_gray):
    #plt.imshow(gray)
    #plt.show()

    tmp=np.zeros((len(img_gray),len(img_gray[0])))

    points=[]
    for i in range(len(img_gray)):
        flag=img_gray[i][0]
        for j in range(1, len(img_gray[i])):
            if img_gray[i][j]!=flag:
                points.append([i,j])
                flag=img_gray[i][j]
                tmp[i][j]=1


    #print(len(points))

    #plt.imshow(tmp)
    #plt.show()
    return points

def Furthest(linePoints):
    p=[]
    for line,points in linePoints.items():
        if len(points)==1:
            p.append(points[0])
        elif len(points)==2:
            p.append(points[0])
            p.append(points[1])
        elif len(points)>2:
            tmp=[points[0],points[1]]
            dist=-1
            num=len(points)
            for i in range(num-1):
                for j in range(i+1,num):
                    p1=points[i]
                    p2=points[j]
                    d=computePDist(p1,p2)
                    if d>dist:
                        tmp=[p1,p2]
                        dist=d
            p.append(tmp[0])
            p.append(tmp[1])
    return p

#更新距离最大
def updateDist(keyPoints,point):
    dist1=computePDist(keyPoints[0],point)
    dist2=computePDist(keyPoints[1],point)

    if dist1>dist2:
        if dist1>keyPoints[2]:
            keyPoints[1][0]=point[0]
            keyPoints[1][1]=point[1]
            keyPoints[2]=dist1
    else:
        if dist2>keyPoints[2]:
            keyPoints[0][0]=point[0]
            keyPoints[0][1]=point[1]
            keyPoints[2]=dist2

#剪枝,a角度
def Pruning(points):
    keyPoints={}
    for point in points:
        key=0
        if point[0]==0:
            key=90
        else:
            key=(point[1])/(point[0])

            #转为角度（-90～90）
            key=math.degrees(math.atan(key))
            key=int(key)

        if key not in keyPoints:
            keyPoints[key]=[]
            keyPoints[key].append(point)
        else:
            if len(keyPoints[key])<2:
                keyPoints[key].append(point)
                keyPoints[key].append(computeDist(points))
            else:
                updateDist(keyPoints[key], point)

    points=[]
    for k,v in keyPoints.items():
        if len(v)==3:
            points.append(v[0])
            points.append(v[1])
        else:
            for ele in v:
                points.append(ele)
    #print(len(points))
    #for k,v in keyPoints.items():
        #print(k)
    return points
def Pruningbad(points):
    a=1
    keyPoints={}
    for point in points:
        key=0
        if point[0]==0:
            key=int(90/a)
        else:
            key=(point[1])/(point[0])
            #转为角度（-90～90）
            key=math.degrees(math.atan(key))

            key=key/90*(int(90/a))
            key=int(key)
            print(key)

        if key not in keyPoints:
            keyPoints[key]=[]
            keyPoints[key].append(point)
        else:
            if len(keyPoints[key])<2:
                keyPoints[key].append(point)
                keyPoints[key].append(computeDist(points))
            else:
                updateDist(keyPoints[key], point)

    points=[]
    for k,v in keyPoints.items():
        if len(v)==3:
            points.append(v[0])
            points.append(v[1])
        else:
            for ele in v:
                points.append(ele)
    #print(len(points))
    #for k,v in keyPoints.items():
        #print(k)
    return points

#剪枝,a角度
def Pruning2(points):
    a=1
    keyPoints={}
    for point in points:
        key=0
        if point[0]==0:
            key=int(90/a)
        else:
            key=int((((point[1])/(point[0]))/90)*(int(90/a)))

            #转为角度（-90～90）
            key=math.degrees(math.atan(key))
            key=int(key)

        if key not in keyPoints:
            keyPoints[key]=[]
            keyPoints[key].append(point)
        else:
            keyPoints[key].append(point)

    points=Furthest(keyPoints)
    return points

#计算周长P
def computePerimeter(points):
    sum=computePDist(points[0],points[len(points)-1])
    for i in range(len(points)-1):
        d=computePDist(points[i],points[i+1])
        sum=sum+d
    return sum

#计算宽度w,高度h
def computeWH(points):
    tmp=np.array(points)
    X=tmp[:,0]
    maxX=max(X)
    minX=min(X)
    width=maxX-minX
    Y=tmp[:,1]
    maxY=max(Y)
    minY=min(Y)
    heigh=maxY-minY

    return width,heigh

#计算面积CPA
def computeCPA(tRGB):
    #cv2.imshow("hgjh", tRGB)
    #cv2.waitKey()
    gray = cv2.cvtColor(tRGB, cv2.COLOR_BGR2GRAY)
    CPA=np.sum((gray > 125) * 1)
    return CPA

#计算asymmetry index
def computeAI(width,heigh):
    return (heigh/width)*1.0

#计算roundness index
def computeRI(Perimeter,CPA):
    return (CPA/Perimeter)*1.0

#计算compactness index
def computeCI(Perimeter,CPA):
    return 4*(math.pi)*CPA/(Perimeter*Perimeter)

#计算RMSE
def computeRMSE(g,p):
    s=0.0
    for i in range(len(g)):
        tmp=g[i]-p[i]
        tmp2=math.pow(tmp,2)
        s=s+tmp2
    s=s/len(g)
    return math.sqrt(s)

#计算MAE
def computeMAE(g,p):
    return np.sum(g-p)/len(g)

#计算精度
def computeAcc(g,p):
    tmp=abs(g-p)
    tmp2=tmp/g
    tmp3=np.sum(tmp2)/len(g)
    tmp4=1-tmp3
    return tmp4


def processGround(imgPath,re):
    # 读入图片，并转为二值图
    #Pdir = "/media/idiot/Flying/农户数码/果树检测分割数据（已处理）/test/test/orderGround/"
    image = cv2.imread(imgPath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = (gray > 125) * 255

    # 取最大联通区域
    l = largestConnectComponent(gray) * 1
    # 属于单棵树的像素个数
    treeArea = np.sum(l)
    #plt.figure(), plt.imshow(l, 'gray')
    #plt.show()


    # 统计单棵树的坐标
    #points = []
    points = Edge(l)
    """
    for i in range(len(l)):
        for j in range(len(l[0])):
            if l[i][j] > 0:
                points.append([i, j])
    """
    # 生成单棵树的RGB图像，供填充多边形所用
    tRGB = treeRGB(points)
    # 计算凸包
    points = testPoint(points)
    #print(points)
    Perimeter=computePerimeter(points)
    width, heigh=computeWH(points)

    # points=[[134, 4], [161, 20], [179, 50], [202, 105], [214, 174], [215, 182], [205, 216], [149, 210], [146, 209], [5, 87], [4, 85], [4, 84], [5, 83]]
    # 转换坐标系
    for ele in points:
        tmp = ele[0]
        ele[0] = ele[1]
        ele[1] = tmp
    #print(len(points))

    # 转换
    points = np.array(points, np.int32)

    # 填充多边形
    cv2.fillConvexPoly(tRGB, points, (255, 255, 255))
    CPA=computeCPA(tRGB)

    AI=computeAI(width,heigh)
    RI=computeRI(Perimeter,CPA)
    CI=computeCI(Perimeter,CPA)
    #print(Perimeter, width, heigh, CPA, AI, RI, CI)
    s=str(Perimeter)+"\t"+str(width)+"\t"+str(heigh)+"\t"+str(CPA)+"\t"+str(AI)+"\t"+str(RI)+"\t"+str(CI)+"\t"+str(treeArea)+"\n"
    re.write(s)
    name=imgPath.split('/')[-1]
    cv2.imwrite('./groundCon/'+name, tRGB, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

def processPredicted(imgPath):
    # 读入图片，并转为二值图
    image = cv2.imread(imgPath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = (gray > 100) * 255
    #plt.imshow(gray, 'gray')
    #plt.show()

    # 取最大联通区域
    l = largestConnectComponent(gray) * 1

    pointsA=[]
    for i in range(len(l)):
        for j in range(len(l[0])):
            if l[i][j]>0:
                pointsA.append([i,j])

    #不进行任何修建
    start_NP = time.time()
    conA=testPoint(pointsA)
    end_NP = time.time()
    t_NP = (end_NP - start_NP)*1000
    #print(len(pointsA))
    draw(pointsA,conA)

    #利用关键点提取方法
    # 统计单棵树的坐标
    start_K = time.time()
    pointsE = Edge(l)
    M_K1 = time.time()
    #print(len(pointsE))
    #conA = testPoint(pointsE)
    #draw(pointsE, conA)

    pointsP = Pruning(pointsE)
    #print(len(pointsP))
    M_K2 = time.time()
    conA = testPoint(pointsP)
    end_K = time.time()
    t_k = (end_K - start_K)*1000

    et=(M_K1 - start_K)*1000
    pt=(M_K2 - M_K1)*1000
    hull=t_k-et-pt
    s="("+"EPE time:"+str(et)+"ms\t\t"+"Pruning time:"+str(pt)+"ms\t\t"+"convex time:"+str(hull)+"ms)"
    draw(pointsP, conA)

    print("Number of points (without any pruning):\t"+str(len(pointsA))+"\t\t\tTime:\t"+str(t_NP)+'ms\n')
    print("Number of points (EPE + Pruning):\t"+str(len(pointsP))+"\t\t\tTime:\t"+str(t_k)+'ms'+"\t"+s+"\n")



def forPredicted():
    testDir = "../TDS-dataset/2-Segmentation/test/"
    for i in range(0,1):
        print("image index:\t"+str(i))
        imageP = testDir + str(i) + '_predict.png'
        processPredicted(imageP)


print("process predicted!!!")
forPredicted()





