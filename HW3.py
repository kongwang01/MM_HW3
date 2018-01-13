#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from pylab import *

from Tkinter import *
from PIL import ImageTk, Image
import tkMessageBox
import tkFileDialog 
from ttk import Frame, Button, Label, Style

import numpy
import scipy.fftpack

import pickle

import colorsys

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


#images_quantities = 1006
all_images_dictionary = {}

#with open('HW3_dictData_kmean20.pickle', 'rb') as handle:
#    all_images_dictionary = pickle.load(handle)

#======  Q3  =============
from scipy.cluster.vq import kmeans,vq, whiten
images_start_quantities_for_SIFT = 0
#images_quantities_for_SIFT = 1006
images_quantities_for_SIFT = len(fileList)

kMeans_k = 20 #設定kmeans要分類成幾群
#======  Q3  =============


class Tkinter_GUI(Frame):
  
    def __init__(self, parent):
        Frame.__init__(self, parent)   
         
        self.parent = parent
        
        self.initUI()
    
        
    def initUI(self):
        self.parent.title("HW3") 
        self.pack(fill=BOTH, expand=1)

        Button(self, text = "Select File", command = openFile).grid(row=0, column=0, pady=5)
        self.fileName = StringVar()
        Label(self, textvariable=self.fileName).grid(row=0, column=1, columnspan=2, pady=5, sticky=W)
        self.queryImage = []
        self.queryImage.append(Label(self))
        self.queryImage[0].grid(row=0, column=3, pady=5, sticky=W)

        Label(self, text = "Select Mode: ").grid(row=1, column=0, pady=5)
        mode = StringVar(self)
        mode.set("Q1-ColorHistogram")
        om = OptionMenu(self, mode, "Q1-ColorHistogram", "Q2-ColorLayout", "Q3-SIFT Visual Words", "Q4-Visual Words using stop words")
        om.grid(row=1, column=1, pady=5, sticky=W)

        mode3 = StringVar(self)
        mode3.set("RGB")
        om3 = OptionMenu(self, mode3, "RGB", "HSV")
        om3.grid(row=1, column=3, pady=5, sticky=W)
        
        Button(self, text = "SEARCH", command = lambda: startSearching(self.fileName.get(),mode.get(),mode3.get())).grid(row=3, column=0, pady=5)

        
        #用來將top10名的圖片貼到window上
        self.images = []
        for i in range(10):
            self.images.append(Label(self))
            self.images[i].grid(row=(i/5)*2+4, column=i%5, padx=5, pady=20)
            
        #用來將top10名的文字貼到window上
        self.rank_labels = []
        for i in range(10):
            self.rank_labels.append(Label(self))
            self.rank_labels[i].grid(row=(i/5)*2+5, column=i%5, padx=5, pady=10)
            


def openFile ():
    fileName = tkFileDialog.askopenfilename(initialdir = "./dataset")
    app.fileName.set(fileName)
    
    image = Image.open(fileName)
    image = image.resize((50, 50), Image.ANTIALIAS) #The (80, 80) is (height, width)
    photo = ImageTk.PhotoImage(image)
    app.queryImage[0].configure(image=photo)
    app.queryImage[0].image = photo

global Q1_first_time_run
global Q2_first_time_run
global Q3_first_time_run
global Q4_first_time_run
Q1_first_time_run = True##因為有輸出成pickle檔了，所以直接改為False
Q2_first_time_run = True
Q3_first_time_run = True
Q4_first_time_run = True

def startSearching (fileName, mode, mode3):
    #print "Your Code Here."
    
    if mode == "Q1-ColorHistogram":
        print "Q1."
        Query_fileName = fileName
        
        #讀入所有jpg的Color Histogram(只有第一次按Search時需要)
        global Q1_first_time_run
        if (Q1_first_time_run):
            Q1_first_time_run = False
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
        #-------------------------------------------
        
        #計算Query跟所有image個別的distance(利用Eucildean)
        query_dict_key = int(Query_fileName[Query_fileName.find('.')- 5: Query_fileName.find('.')])
        
        distance_QueryWithImages = list()
        
        if mode3 == "HSV":
            for i in range(0, images_quantities):
                distance_QueryWithImages.append(list())
                ith_distance = 0
                ith_H_distance = 0
                ith_S_distance = 0
                ith_V_distance = 0
                for j in range(0, 256):
                    ith_H_distance += ((all_images_dictionary[query_dict_key]["color_histogram_H_list"][j] - all_images_dictionary[i]["color_histogram_H_list"][j])**2)
                    ith_S_distance += ((all_images_dictionary[query_dict_key]["color_histogram_S_list"][j] - all_images_dictionary[i]["color_histogram_S_list"][j])**2)
                    ith_V_distance += ((all_images_dictionary[query_dict_key]["color_histogram_V_list"][j] - all_images_dictionary[i]["color_histogram_V_list"][j])**2)
                 
                #print ith_distance
                ith_H_distance = ith_H_distance ** (0.5)
                ith_S_distance = ith_S_distance ** (0.5)
                ith_V_distance = ith_V_distance ** (0.5)
                ith_distance = ith_H_distance + ith_S_distance + ith_V_distance
                distance_QueryWithImages[i].append(i)
                distance_QueryWithImages[i].append(ith_distance)
                
            distance_QueryWithImages = sorted(distance_QueryWithImages,key=lambda l:l[1], reverse=False) #讓list照distance由小到大排序
        else:
            for i in range(0, images_quantities):
                distance_QueryWithImages.append(list())
                ith_distance = 0
                ith_R_distance = 0
                ith_G_distance = 0
                ith_B_distance = 0
                for j in range(0, 256):
                    ith_R_distance += ((all_images_dictionary[query_dict_key]["color_histogram_R_list"][j] - all_images_dictionary[i]["color_histogram_R_list"][j])**2)
                    ith_G_distance += ((all_images_dictionary[query_dict_key]["color_histogram_G_list"][j] - all_images_dictionary[i]["color_histogram_G_list"][j])**2)
                    ith_B_distance += ((all_images_dictionary[query_dict_key]["color_histogram_B_list"][j] - all_images_dictionary[i]["color_histogram_B_list"][j])**2)
                 
                #print ith_distance
                ith_R_distance = ith_R_distance ** (0.5)
                ith_G_distance = ith_G_distance ** (0.5)
                ith_B_distance = ith_B_distance ** (0.5)
                ith_distance = ith_R_distance + ith_G_distance + ith_B_distance
                distance_QueryWithImages[i].append(i)
                distance_QueryWithImages[i].append(ith_distance)
                
            distance_QueryWithImages = sorted(distance_QueryWithImages,key=lambda l:l[1], reverse=False) #讓list照distance由小到大排序
        #-------------------------------------------
        
        #將前十名的結果放到tkinter視窗上
        for i in range(0,10):
            app.rank_labels[i].configure(text="Rank " + str(i) + " is number " + str(distance_QueryWithImages[i][0]) + ", distance is " + str(int(distance_QueryWithImages[i][1])))
            photo = ImageTk.PhotoImage(Image.open(all_images_dictionary[distance_QueryWithImages[i][0]]["fileName"]))
            app.images[i].configure(image=photo)
            app.images[i].image = photo

                           
    elif mode == "Q2-ColorLayout":
        print "Q2."
        
        Q2_dict = {} 
        Query_fileName = fileName
        
        #計算所有jpg的DCT(只有第一次按Search時需要)
        global Q2_first_time_run
        if (Q2_first_time_run):
            Q2_first_time_run = False
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
                im_8x8 = Image.open(fileName)
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
                
        #-------------------------------------------
             
        #計算Query跟所有image個別的distance(利用Eucildean)
        query_dict_key = int(Query_fileName[Query_fileName.find('.')- 5: Query_fileName.find('.')])
        
        distance_QueryWithImages = list()
        
        for i in range(0, images_quantities):
            distance_QueryWithImages.append(list())
            ith_distance = 0
            ith_Y_distance = 0
            ith_Cb_distance = 0
            ith_Cr_distance = 0
            for j in range(0, 64):
                ith_Y_distance += ((all_images_dictionary[query_dict_key]["DCT_Y_list"][j] - all_images_dictionary[i]["DCT_Y_list"][j])**2)
                ith_Cb_distance += ((all_images_dictionary[query_dict_key]["DCT_Cb_list"][j] - all_images_dictionary[i]["DCT_Cb_list"][j])**2)
                ith_Cr_distance += ((all_images_dictionary[query_dict_key]["DCT_Cr_list"][j] - all_images_dictionary[i]["DCT_Cr_list"][j])**2)
             
            #print ith_distance
            ith_Y_distance = ith_Y_distance ** (0.5)
            ith_Cb_distance = ith_Cb_distance ** (0.5)
            ith_Cr_distance = ith_Cr_distance ** (0.5)
            ith_distance = ith_Y_distance + ith_Cb_distance + ith_Cr_distance
            distance_QueryWithImages[i].append(i)
            distance_QueryWithImages[i].append(ith_distance)
            
            
        distance_QueryWithImages = sorted(distance_QueryWithImages,key=lambda l:l[1], reverse=False) #讓list照distance由小到大排序
        #-------------------------------------------
        
        for i in range(0,10):
            app.rank_labels[i].configure(text="Rank " + str(i) + " is number " + str(distance_QueryWithImages[i][0]) + ", distance is " + str(int(distance_QueryWithImages[i][1])))
            photo = ImageTk.PhotoImage(Image.open(all_images_dictionary[distance_QueryWithImages[i][0]]["fileName"]))
            app.images[i].configure(image=photo)
            app.images[i].image = photo
             

        
    elif mode == "Q3-SIFT Visual Words":
        print "Q3."
        Query_fileName = fileName
        
        #讀入所有jpg的Color Histogram(只有第一次按Search時需要)
        global Q3_first_time_run
        if (Q3_first_time_run):
            Q3_first_time_run = False
            #從dataset讀入image，並利用SIFT找出feature locations, descriptors，取出descriptors(d1)並存入BoF中
            for i in range(images_start_quantities_for_SIFT, images_quantities_for_SIFT):
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
            print len(BoF)
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
        #------------------------------------------------------------------------
        
        #計算Query跟所有image個別的distance(利用Eucildean)
        query_dict_key = int(Query_fileName[Query_fileName.find('.')- 5: Query_fileName.find('.')])
        distance_QueryWithImages = list()
        
        
        for i in range(images_start_quantities_for_SIFT, images_quantities_for_SIFT):
            distance_QueryWithImages.append(list())
            ith_distance = 0

            for j in range(0, kMeans_k):
                ith_distance += ((all_images_dictionary[query_dict_key]["feature_descriptors_histogram"][j] - all_images_dictionary[i]["feature_descriptors_histogram"][j])**2)

             
            #print ith_distance
            ith_distance = ith_distance ** (0.5)

            distance_QueryWithImages[i].append(i)
            distance_QueryWithImages[i].append(ith_distance)
            
        distance_QueryWithImages = sorted(distance_QueryWithImages,key=lambda l:l[1], reverse=False) #讓list照distance由小到大排序
        #-------------------------------------------
        
        if(type(all_images_dictionary[query_dict_key]["feature_descriptors_raw"]) != bool):#如果Query有特徵點才直接print出前十名
            for i in range(0,10):
                app.rank_labels[i].configure(text="Rank " + str(i) + " is number " + str(distance_QueryWithImages[i][0]) + ", distance is " + str(int(distance_QueryWithImages[i][1])))
                photo = ImageTk.PhotoImage(Image.open(all_images_dictionary[distance_QueryWithImages[i][0]]["fileName"]))
                app.images[i].configure(image=photo)
                app.images[i].image = photo
        else:#如果Query沒有特徵點，先print出自己再print出前九名
            app.rank_labels[0].configure(text="Rank " + str(i) + " is number " + str(query_dict_key) + ", distance is 0")
            photo = ImageTk.PhotoImage(Image.open(all_images_dictionary[query_dict_key]["fileName"]))
            app.images[0].configure(image=photo)
            app.images[0].image = photo
            for i in range(1,10):
                app.rank_labels[i].configure(text="Rank " + str(i) + " is number " + str(distance_QueryWithImages[i][0]) + ", distance is " + str(int(distance_QueryWithImages[i][1])))
                photo = ImageTk.PhotoImage(Image.open(all_images_dictionary[distance_QueryWithImages[i][0]]["fileName"]))
                app.images[i].configure(image=photo)
                app.images[i].image = photo
        
                
        
    elif mode == "Q4-Visual Words using stop words":
        print "Q4."
        Query_fileName = fileName
        
        #讀入所有jpg的Color Histogram(只有第一次按Search時需要)
        #global Q3_first_time_run
        if (Q3_first_time_run):
            Q3_first_time_run = False
            #從dataset讀入image，並利用SIFT找出feature locations, descriptors，取出descriptors(d1)並存入BoF中
            for i in range(images_start_quantities_for_SIFT, images_quantities_for_SIFT):
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
                    if((len(f) == 132) and (i != 994)): #只有一個特徵點的，要做特別處理
                        d1 = np.array([f[4:]]) # feature locations, descriptors
                    else:
                        l1,d1 = f[:,:4],f[:,4:] # feature locations, descriptors
                        
                    
                    all_images_dictionary[i]["feature_descriptors_raw"] = d1
                    
                    if(i == images_start_quantities_for_SIFT):
                        BoF = d1
                    else:
                        BoF = np.concatenate((BoF, d1))
                else:
                    all_images_dictionary[i]["feature_descriptors_raw"] = False
                    

            ###==========  運行kmeans，將BoF做分群  ============================
            print len(BoF)
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
        #------------------------------------------------------------------------
        #計算每一群中的features數量
        features_num = dict()
        for j in range(0, kMeans_k):
            features_num[j] = 0
            for i in range(images_start_quantities_for_SIFT, images_quantities_for_SIFT):
                features_num[j] += all_images_dictionary[i]["feature_descriptors_histogram"][j]

        #找出前十百分比features數量最多的群
        toptenpercent = list()
        features_num = sorted(features_num.iteritems(), key=lambda d:d[1], reverse = True)
        #print features_num
        for i in range(0, kMeans_k):
            toptenpercent.append(0)
        for i in range(0, kMeans_k/10):
            toptenpercent[features_num[i][0]] = 1
        #print toptenpercent
            
        #------------------------------------------------------------------------
        
        #計算Query跟所有image個別的distance(利用Eucildean)
        query_dict_key = int(Query_fileName[Query_fileName.find('.')- 5: Query_fileName.find('.')])
        distance_QueryWithImages = list()
        
        
        for i in range(images_start_quantities_for_SIFT, images_quantities_for_SIFT):
            distance_QueryWithImages.append(list())
            ith_distance = 0

            for j in range(0, kMeans_k):
                if toptenpercent[j] == 0:#不是前十百分比才去計算
                    ith_distance += ((all_images_dictionary[query_dict_key]["feature_descriptors_histogram"][j] - all_images_dictionary[i]["feature_descriptors_histogram"][j])**2)

             
            #print ith_distance
            ith_distance = ith_distance ** (0.5)

            distance_QueryWithImages[i].append(i)
            distance_QueryWithImages[i].append(ith_distance)
            
        distance_QueryWithImages = sorted(distance_QueryWithImages,key=lambda l:l[1], reverse=False) #讓list照distance由小到大排序
        #-------------------------------------------
        
        if(type(all_images_dictionary[query_dict_key]["feature_descriptors_raw"]) != bool):#如果Query有特徵點才直接print出前十名
            for i in range(0,10):
                app.rank_labels[i].configure(text="Rank " + str(i) + " is number " + str(distance_QueryWithImages[i][0]) + ", distance is " + str(int(distance_QueryWithImages[i][1])))
                photo = ImageTk.PhotoImage(Image.open(all_images_dictionary[distance_QueryWithImages[i][0]]["fileName"]))
                app.images[i].configure(image=photo)
                app.images[i].image = photo
        else:#如果Query沒有特徵點，先print出自己再print出前九名
            app.rank_labels[0].configure(text="Rank " + str(i) + " is number " + str(query_dict_key) + ", distance is 0")
            photo = ImageTk.PhotoImage(Image.open(all_images_dictionary[query_dict_key]["fileName"]))
            app.images[0].configure(image=photo)
            app.images[0].image = photo
            for i in range(1,10):
                app.rank_labels[i].configure(text="Rank " + str(i) + " is number " + str(distance_QueryWithImages[i][0]) + ", distance is " + str(int(distance_QueryWithImages[i][1])))
                photo = ImageTk.PhotoImage(Image.open(all_images_dictionary[distance_QueryWithImages[i][0]]["fileName"]))
                app.images[i].configure(image=photo)
                app.images[i].image = photo
        
    else:
        print "ELSE."


if __name__ == '__main__':
    root = Tk()
    size = 220, 220

    app = Tkinter_GUI(root)
    #root.geometry("1024x720")
    root.geometry("1280x820")
    root.mainloop()
