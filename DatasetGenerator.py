#!/usr/bin/env python3
import os
import numpy as np
from PIL import Image, ImageFile

import torch
from torch.utils.data import Dataset

#-------------------------------------------------------------------------------- 

class DatasetGenerator (Dataset):
    
    #-------------------------------------------------------------------------------- 
    
    def __init__ (self, pathImageDirectory, pathDatasetFile, transform):
    
        self.listImagePaths = []
        self.listImageLabels = []
        self.transform = transform
    
        #---- Open file, get image paths and labels
    
        fileDescriptor = open(pathDatasetFile, "r")

        #---- get into the loop
        line = True
        
        while line:
            try:    
                line = fileDescriptor.readline()
            
            #--- if not empty
                if line:
          
                    lineItems = line.split()
                
                    imagePath = os.path.join(pathImageDirectory, lineItems[0])
                    imageLabel = lineItems[1:]
                    imageLabel = [int(i) for i in imageLabel]
                
                    self.listImagePaths.append(imagePath)
                    self.listImageLabels.append(imageLabel)   
            except IOError:
                print(line)
                print(i)
        fileDescriptor.close()
    
    #-------------------------------------------------------------------------------- 
    
    def __getitem__(self, index):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        imagePath = self.listImagePaths[index]
        
        imageData = Image.open(imagePath).convert('RGB')
        imageLabel= torch.FloatTensor(self.listImageLabels[index])
        
        if self.transform != None: imageData = self.transform(imageData)
        
        return imageData, imageLabel
        
    #-------------------------------------------------------------------------------- 
    
    def __len__(self):
        
        return len(self.listImagePaths)
    
 #-------------------------------------------------------------------------------- 
    

