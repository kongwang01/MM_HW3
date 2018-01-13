#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from pylab import *

from PIL import ImageTk, Image

import numpy
import scipy.fftpack

#======  搜尋dataset中的圖片數量  =============
import sys

fileList = []
fileSize = 0
folderCount = 0
rootdir = './dataset'

for root, subFolders, files in os.walk(rootdir):
    folderCount += len(subFolders)
    for file in files:
        f = os.path.join(root,file)
        fileSize = fileSize + os.path.getsize(f)
        #print(f)
        fileList.append(f)

print("Total Size is {0} bytes".format(fileSize))
print("Total Files ", len(fileList))
print("Total Folders ", folderCount)


images_quantities = len(fileList)
#=================================================

all_images_dictionary = {}

#======  Q3  =============
from scipy.cluster.vq import kmeans,vq, whiten
images_start_quantities_for_SIFT = 0
images_quantities_for_SIFT = len(fileList)

kMeans_k = 20 #設定kmeans要分類成幾群
#======  Q3  =============

import pickle

import colorsys

#Q1
for i in range(0, images_quantities):
    temp_str = str(i)
    temp_str_fillZero = temp_str.zfill(4)
    i_th_fileName = "./dataset/ukbench0"+ temp_str_fillZero +".jpg"
    
    if i not in all_images_dictionary:
        all_images_dictionary[i] = {}
        all_images_dictionary[i]["fileName"] = i_th_fileName
        
    all_images_dictionary[i]["color_histogram_R_list"] = list()
    all_images_dictionary[i]["color_histogram_G_list"] = list()
    all_images_dictionary[i]["color_histogram_B_list"] = list()
    
    for j in range(0, 256):
        all_images_dictionary[i]["color_histogram_R_list"].append(0)
        all_images_dictionary[i]["color_histogram_G_list"].append(0)
        all_images_dictionary[i]["color_histogram_B_list"].append(0)
        
    #HSV
    all_images_dictionary[i]["color_histogram_H_list"] = list()
    all_images_dictionary[i]["color_histogram_S_list"] = list()
    all_images_dictionary[i]["color_histogram_V_list"] = list()
    
    for j in range(0, 360):
        all_images_dictionary[i]["color_histogram_H_list"].append(0)
        all_images_dictionary[i]["color_histogram_S_list"].append(0)
        all_images_dictionary[i]["color_histogram_V_list"].append(0)
    
    
    #讀入Color Histogram    
    im = Image.open(i_th_fileName).convert('RGB') #讀入圖片
    pixel = im.load()
    width, height = im.size

    for x in xrange(width):    
        for y in xrange(height):
            R, G, B = pixel[x,y]
            all_images_dictionary[i]["color_histogram_R_list"][R]+=1
            all_images_dictionary[i]["color_histogram_G_list"][G]+=1
            all_images_dictionary[i]["color_histogram_B_list"][B]+=1
            H, S, V = colorsys.rgb_to_hsv(R/255., G/255., B/255.)
            all_images_dictionary[i]["color_histogram_H_list"][int(H*360)]+=1
            #all_images_dictionary[i]["color_histogram_H_list"][(int(H*360))/3]+=1
            all_images_dictionary[i]["color_histogram_S_list"][int(S*100)]+=1
            all_images_dictionary[i]["color_histogram_V_list"][int(V*100)]+=1
print 'Q1 complete'
#-------------------------------------------

#Q2
Q2_dict = {} 
for i in range(0, images_quantities):
    Q2_dict.clear()#Q2_dict用來計算每次輸入的image的DCT_Y, DCT_Cb, DCT_Cr，每次匯入新圖時清空
    temp_str = str(i)
    temp_str_fillZero = temp_str.zfill(4)
    i_th_fileName = "./dataset/ukbench0"+ temp_str_fillZero +".jpg"
    
    if i not in all_images_dictionary:
        all_images_dictionary[i] = {}
        all_images_dictionary[i]["fileName"] = i_th_fileName
        
    all_images_dictionary[i]["DCT_Y_list"] = list()
    all_images_dictionary[i]["DCT_Cb_list"] = list()
    all_images_dictionary[i]["DCT_Cr_list"] = list()

    
    #開始計算DCT 
    im = Image.open(i_th_fileName).convert('RGB') #讀入圖片
    pixel = im.load()
    width, height = im.size

    for x in xrange(width):    
        for y in xrange(height):
            if(((width%8) == 0) and ((height%8) == 0)):
                curr_block_key = "block" + str((x/(width/8)))+ str((y/(height/8)))
            elif(((width%8) == 0) and ((height%8) != 0)):
                curr_block_key = "block" + str((x/(width/8)))+ str((y/((height/8)+1)))
            elif(((width%8) != 0) and ((height%8) == 0)):
                curr_block_key = "block" + str((x/((width/8)+1)))+ str((y/(height/8)))
            else:
                curr_block_key = "block" + str((x/((width/8)+1)))+ str((y/((height/8)+1)))
            
            if curr_block_key not in Q2_dict:
                Q2_dict[curr_block_key] = list()
                Q2_dict[curr_block_key].append(0) #Q2_dict[curr_block_key][0] 儲存pixels個數
                Q2_dict[curr_block_key].append(0) #Q2_dict[curr_block_key][1] Representative Color R
                Q2_dict[curr_block_key].append(0) #Q2_dict[curr_block_key][2] Representative Color G
                Q2_dict[curr_block_key].append(0) #Q2_dict[curr_block_key][3] Representative Color B
            
            R, G, B = pixel[x,y]
            Q2_dict[curr_block_key][0] += 1
            Q2_dict[curr_block_key][1] += R
            Q2_dict[curr_block_key][2] += G
            Q2_dict[curr_block_key][3] += B
        
        
    #前一個雙層for只有做加總，這邊將加總完的R,G,B除以該block的總pixels數量        
    for j in Q2_dict:
        Q2_dict[j][1] = Q2_dict[j][1]/Q2_dict[j][0]
        Q2_dict[j][2] = Q2_dict[j][2]/Q2_dict[j][0]
        Q2_dict[j][3] = Q2_dict[j][3]/Q2_dict[j][0]
        
        
    #把im的每個pixels的值改為各自block的顏色平均值(用來print出計算平均顏色的結果對不對)
    """
    for x in xrange(width):    
        for y in xrange(height):
            if(((width%8) == 0) and ((height%8) == 0)):
                curr_block_key = "block" + str((x/(width/8)))+ str((y/(height/8)))
            elif(((width%8) == 0) and ((height%8) != 0)):
                curr_block_key = "block" + str((x/(width/8)))+ str((y/((height/8)+1)))
            elif(((width%8) != 0) and ((height%8) == 0)):
                curr_block_key = "block" + str((x/((width/8)+1)))+ str((y/(height/8)))
            else:
                curr_block_key = "block" + str((x/((width/8)+1)))+ str((y/((height/8)+1)))
                
            pixel[x,y] = (Q2_dict[curr_block_key][1],Q2_dict[curr_block_key][2],Q2_dict[curr_block_key][3])
    """
    
     
    #步驟二：把算完的平均值存進8x8的image
    im_8x8 = Image.open(all_images_dictionary[0]["fileName"])
    im_8x8 = im_8x8.resize((8, 8))
    pixel_8x8 = im_8x8.load()
    width_8x8, height_8x8 = im_8x8.size
    
    for x in xrange(width_8x8):    
        for y in xrange(height_8x8):
            curr_block_key = "block" + str(x)+ str(y)
            pixel_8x8[x,y] = (Q2_dict[curr_block_key][1],Q2_dict[curr_block_key][2],Q2_dict[curr_block_key][3])
    
    
    
    #步驟三：把8x8的RGB轉成8x8的YCbCr
    im_ycbcr = im_8x8.convert('YCbCr')
    pixel_ycbcr = im_ycbcr.load()
    width_ycbcr, height_ycbcr = im_ycbcr.size
    
    #用來print出8x8的im_ycbcr的每個pixel的值
    #for x in xrange(width_ycbcr):    
    #    for y in xrange(height_ycbcr):
    #        Y, Cb, Cr = pixel_ycbcr[x,y]
    #        print Y, Cb, Cr
            
    
    
    #步驟四：對8x8的YCbCr image(im_ycbcr)做DCT運算，Y、Cb、Cr分別運算
    dctSize = im_ycbcr.size[0]
    
    pixels_Y = numpy.array(im_ycbcr.getdata(0), dtype=numpy.float).reshape((dctSize, dctSize))
    pixels_Cb = numpy.array(im_ycbcr.getdata(1), dtype=numpy.float).reshape((dctSize, dctSize))
    pixels_Cr = numpy.array(im_ycbcr.getdata(2), dtype=numpy.float).reshape((dctSize, dctSize))


    # perform 2-dimensional DCT (discrete cosine transform):
    DCT_Y = scipy.fftpack.dct(scipy.fftpack.dct(pixels_Y.T, norm="ortho").T, norm="ortho")
    DCT_Cb = scipy.fftpack.dct(scipy.fftpack.dct(pixels_Cb.T, norm="ortho").T, norm="ortho")
    DCT_Cr = scipy.fftpack.dct(scipy.fftpack.dct(pixels_Cr.T, norm="ortho").T, norm="ortho")
    
    for x in xrange(width_ycbcr):    
        for y in xrange(height_ycbcr):
            all_images_dictionary[i]["DCT_Y_list"].append(DCT_Y[x][y])
            all_images_dictionary[i]["DCT_Cb_list"].append(DCT_Cb[x][y])
            all_images_dictionary[i]["DCT_Cr_list"].append(DCT_Cr[x][y])
print 'Q2 complete'
#-------------------------------------------

#Q3
for i in range(images_start_quantities_for_SIFT, images_quantities_for_SIFT):
    print i
    temp_str = str(i)
    temp_str_fillZero = temp_str.zfill(4)
    i_th_fileName = "./dataset/ukbench0"+ temp_str_fillZero +".jpg"
    
    if i not in all_images_dictionary:
        all_images_dictionary[i] = {}
        all_images_dictionary[i]["fileName"] = i_th_fileName
        
    all_images_dictionary[i]["feature_descriptors_histogram"] = list()
    
    for j in range(0, kMeans_k):
        all_images_dictionary[i]["feature_descriptors_histogram"].append(0)

    outputFilename = './siftset/'+i_th_fileName[-16:-4]+'.sift'
    
    #有些sift檔為空的，要判斷
    if (os.stat(outputFilename).st_size != 0): 
        """ Read feature properties and return in matrix form. """
        f = loadtxt(outputFilename)
        #if((len(f) == 132) and (i != 994)): #只有一個特徵點的，要做特別處理
        #    d1 = np.array([f[4:]]) # feature locations, descriptors
        #else:
        l1,d1 = f[:,:4],f[:,4:] # feature locations, descriptors
            
        
        all_images_dictionary[i]["feature_descriptors_raw"] = d1
        
        if(i == images_start_quantities_for_SIFT):
            BoF = d1
        else:
            BoF = np.concatenate((BoF, d1))
    else:
        all_images_dictionary[i]["feature_descriptors_raw"] = False
        

###==========  運行kmeans，將BoF做分群  ============================

whitened = whiten(BoF)
data = whitened

# computing K-Means with K = kMeans_k (kMeans_k clusters)
centroids,_ = kmeans(data,kMeans_k)

# assign each sample to a cluster
for i in range(images_start_quantities_for_SIFT, images_quantities_for_SIFT):
    if(type(all_images_dictionary[i]["feature_descriptors_raw"]) != bool):#有特徵點的才搜索每個特徵點屬於第幾群
        whitened2 = whiten(all_images_dictionary[i]["feature_descriptors_raw"])
        idx,_ = vq(whitened2,centroids)
        for j in range(len(idx)):
            all_images_dictionary[i]["feature_descriptors_histogram"][idx[j]]+=1
print 'Q3 complete'
#------------------------------------------------------------------------

# Store data to 'HW3_dictData.pickle'
with open('HW3_dictData_kmean20.pickle', 'wb') as handle:
    pickle.dump(all_images_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
