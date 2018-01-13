#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
from numpy import array
import sift
from pylab import *

import os
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

for i in range(0, images_quantities):
    temp_str = str(i)
    temp_str_fillZero = temp_str.zfill(5)
    i_th_fileName = "./dataset/ukbench"+ temp_str_fillZero +".jpg"

    #im1 = array(Image.open(i_th_fileName).convert('L'))

    outputFilename = './siftset/'+i_th_fileName[-16:-4]+'.sift'
    
    sift.process_image(i_th_fileName, outputFilename)
    
    print temp_str + " complete"
    

