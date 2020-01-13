# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 14:31:07 2019

@author: Paul
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plotIM_BB (image, boudingBox, boudingBox2 = None, bbType = "3d", relativLabels = True, loss = None):
    """
   
    This Function plots an image with its 3d or 2d Bounding Box
    
    :param image:           numpy array of an RGB Image with size (x,y,3)
    :param boudingBox:      numpy array of the Bounding Box points with size (5,1) for 2D or (17,1) for 3D
    :param bbType:          Type Bounding Box 2d or 3d 
    :param relativLabels:   deterin  if bb points are relativ, ranging from 0 to 1 or are absolut pixel values 
    :return:                fig and ax object
    """
    if bbType.upper() == "2D":
        if boudingBox.shape[0] != 5   :
            print("Error : size of 2D boundingbox is " + str(boudingBox.shape) + " but expected to be (5,)" )
            return

        # Create figure and axes
        fig,ax = plt.subplots(1)
        # Display the image
        ax.imshow(image)
        # Create a Rectangle patch
        xImgS = image.shape[0]
        yImgS = image.shape[1]
        x = boudingBox[1] * xImgS
        y = boudingBox[0] * yImgS
        w = boudingBox[3] * xImgS - x
        h = boudingBox[2] * yImgS - y
        rect = patches.Rectangle((x,y),w,h,linewidth=2,edgecolor='r',facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
        if boudingBox2 is not None:
            xImgS = image.shape[0]
            yImgS = image.shape[1]
            x = boudingBox2[1] * xImgS
            y = boudingBox2[0] * yImgS
            w = boudingBox2[3] * xImgS - x
            h = boudingBox2[2] * yImgS - y
            rect = patches.Rectangle((x,y),w,h,linewidth=2,edgecolor='b',facecolor='none')    
            ax.add_patch(rect)
        plt.show()
        
    if bbType.upper() == "3D":
        if boudingBox.shape[0] != 17:
            print("Error : size of 3D boundingbox is " + str(boudingBox.shape) + " but expected to be (17,)" )
            return;        
        
        if relativLabels:
            xImgS = image.shape[0]
            yImgS = image.shape[1] 
        else:
            xImgS = 1
            yImgS = 1
            
        fig,ax = plt.subplots(1)
        ax.imshow(image)
        
        xyA = (boudingBox[0]*xImgS, boudingBox[1]*yImgS)
        xyB = (boudingBox[2]*xImgS, boudingBox[3]*yImgS)
        con = patches.ConnectionPatch(xyA, xyB, "data", "data", shrinkA=0, shrinkB=0,
                              mutation_scale=20, fc="b",edgecolor='r',linewidth=2)
        ax.add_artist(con)
        
        xyA = (boudingBox[2]*xImgS, boudingBox[3]*yImgS)
        xyB = (boudingBox[4]*xImgS, boudingBox[5]*yImgS)
        con = patches.ConnectionPatch(xyA, xyB, "data", "data", shrinkA=0, shrinkB=0,
                              mutation_scale=20, fc="b",edgecolor='r',linewidth=2)
        ax.add_artist(con)
        
        xyA = (boudingBox[4]*xImgS, boudingBox[5]*yImgS)
        xyB = (boudingBox[6]*xImgS, boudingBox[7]*yImgS)
        con = patches.ConnectionPatch(xyA, xyB, "data", "data", shrinkA=0, shrinkB=0,
                              mutation_scale=20, fc="b",edgecolor='r',linewidth=2)
        ax.add_artist(con)
        
        xyA = (boudingBox[6]*xImgS, boudingBox[7]*yImgS)
        xyB = (boudingBox[0]*xImgS, boudingBox[1]*yImgS)
        con = patches.ConnectionPatch(xyA, xyB, "data", "data", shrinkA=0, shrinkB=0,
                              mutation_scale=20, fc="b",edgecolor='r',linewidth=2)
        ax.add_artist(con)
        
        xyA = (boudingBox[8]*xImgS, boudingBox[9]*yImgS)
        xyB = (boudingBox[10]*xImgS, boudingBox[11]*yImgS)
        con = patches.ConnectionPatch(xyA, xyB, "data", "data", shrinkA=0, shrinkB=0,
                              mutation_scale=20, fc="b",edgecolor='r',linewidth=2)
        ax.add_artist(con)
        
        xyA = (boudingBox[10]*xImgS, boudingBox[11]*yImgS)
        xyB = (boudingBox[12]*xImgS, boudingBox[13]*yImgS)
        con = patches.ConnectionPatch(xyA, xyB, "data", "data", shrinkA=0, shrinkB=0,
                              mutation_scale=20, fc="b",edgecolor='r',linewidth=2)
        ax.add_artist(con)
        
        xyA = (boudingBox[12]*xImgS, boudingBox[13]*yImgS)
        xyB = (boudingBox[14]*xImgS, boudingBox[15]*yImgS)
        con = patches.ConnectionPatch(xyA, xyB, "data", "data", shrinkA=0, shrinkB=0,
                              mutation_scale=20, fc="b",edgecolor='r',linewidth=2)
        ax.add_artist(con)
        
        xyA = (boudingBox[14]*xImgS, boudingBox[15]*yImgS)
        xyB = (boudingBox[8]*xImgS, boudingBox[9]*yImgS)
        con = patches.ConnectionPatch(xyA, xyB, "data", "data", shrinkA=0, shrinkB=0,
                              mutation_scale=20, fc="b",edgecolor='r',linewidth=2)
        ax.add_artist(con)
        
        xyA = (boudingBox[0]*xImgS, boudingBox[1]*yImgS)
        xyB = (boudingBox[8]*xImgS, boudingBox[9]*yImgS)
        con = patches.ConnectionPatch(xyA, xyB, "data", "data", shrinkA=0, shrinkB=0,
                              mutation_scale=20, fc="b",edgecolor='r',linewidth=2)
        ax.add_artist(con)
        
        xyA = (boudingBox[2]*xImgS, boudingBox[3]*yImgS)
        xyB = (boudingBox[10]*xImgS, boudingBox[11]*yImgS)
        con = patches.ConnectionPatch(xyA, xyB, "data", "data", shrinkA=0, shrinkB=0,
                              mutation_scale=20, fc="b",edgecolor='r',linewidth=2)
        ax.add_artist(con)

        xyA = (boudingBox[4]*xImgS, boudingBox[5]*yImgS)
        xyB = (boudingBox[12]*xImgS, boudingBox[13]*yImgS)
        con = patches.ConnectionPatch(xyA, xyB, "data", "data", shrinkA=0, shrinkB=0,
                              mutation_scale=20, fc="b",edgecolor='r',linewidth=2)
        ax.add_artist(con)    
        
        xyA = (boudingBox[6]*xImgS, boudingBox[7]*yImgS)
        xyB = (boudingBox[14]*xImgS, boudingBox[15]*yImgS)
        con = patches.ConnectionPatch(xyA, xyB, "data", "data", shrinkA=0, shrinkB=0,
                              mutation_scale=20, fc="b",edgecolor='r',linewidth=2)
        ax.add_artist(con)
        plt.show()
        
        fig.suptitle("loss_metric=" + str(loss), fontsize=16)
        
        if boudingBox2 is not None:
            xyA = (boudingBox2[0]*xImgS, boudingBox2[1]*yImgS)
            xyB = (boudingBox2[2]*xImgS, boudingBox2[3]*yImgS)
            con = patches.ConnectionPatch(xyA, xyB, "data", "data", shrinkA=0, shrinkB=0,
                                  mutation_scale=20, fc="b",edgecolor='b',linewidth=2)
            ax.add_artist(con)
            
            xyA = (boudingBox2[2]*xImgS, boudingBox2[3]*yImgS)
            xyB = (boudingBox2[4]*xImgS, boudingBox2[5]*yImgS)
            con = patches.ConnectionPatch(xyA, xyB, "data", "data", shrinkA=0, shrinkB=0,
                                  mutation_scale=20, fc="b",edgecolor='b',linewidth=2)
            ax.add_artist(con)
            
            xyA = (boudingBox2[4]*xImgS, boudingBox2[5]*yImgS)
            xyB = (boudingBox2[6]*xImgS, boudingBox2[7]*yImgS)
            con = patches.ConnectionPatch(xyA, xyB, "data", "data", shrinkA=0, shrinkB=0,
                                  mutation_scale=20, fc="b",edgecolor='b',linewidth=2)
            ax.add_artist(con)
            
            xyA = (boudingBox2[6]*xImgS, boudingBox2[7]*yImgS)
            xyB = (boudingBox2[0]*xImgS, boudingBox2[1]*yImgS)
            con = patches.ConnectionPatch(xyA, xyB, "data", "data", shrinkA=0, shrinkB=0,
                                  mutation_scale=20, fc="b",edgecolor='b',linewidth=2)
            ax.add_artist(con)
            
            xyA = (boudingBox2[8]*xImgS, boudingBox2[9]*yImgS)
            xyB = (boudingBox2[10]*xImgS, boudingBox2[11]*yImgS)
            con = patches.ConnectionPatch(xyA, xyB, "data", "data", shrinkA=0, shrinkB=0,
                                  mutation_scale=20, fc="b",edgecolor='b',linewidth=2)
            ax.add_artist(con)
            
            xyA = (boudingBox2[10]*xImgS, boudingBox2[11]*yImgS)
            xyB = (boudingBox2[12]*xImgS, boudingBox2[13]*yImgS)
            con = patches.ConnectionPatch(xyA, xyB, "data", "data", shrinkA=0, shrinkB=0,
                                  mutation_scale=20, fc="b",edgecolor='b',linewidth=2)
            ax.add_artist(con)
            
            xyA = (boudingBox2[12]*xImgS, boudingBox2[13]*yImgS)
            xyB = (boudingBox2[14]*xImgS, boudingBox2[15]*yImgS)
            con = patches.ConnectionPatch(xyA, xyB, "data", "data", shrinkA=0, shrinkB=0,
                                  mutation_scale=20, fc="b",edgecolor='b',linewidth=2)
            ax.add_artist(con)
            
            xyA = (boudingBox2[14]*xImgS, boudingBox2[15]*yImgS)
            xyB = (boudingBox2[8]*xImgS, boudingBox2[9]*yImgS)
            con = patches.ConnectionPatch(xyA, xyB, "data", "data", shrinkA=0, shrinkB=0,
                                  mutation_scale=20, fc="b",edgecolor='b',linewidth=2)
            ax.add_artist(con)
            
            xyA = (boudingBox2[0]*xImgS, boudingBox2[1]*yImgS)
            xyB = (boudingBox2[8]*xImgS, boudingBox2[9]*yImgS)
            con = patches.ConnectionPatch(xyA, xyB, "data", "data", shrinkA=0, shrinkB=0,
                                  mutation_scale=20, fc="b",edgecolor='b',linewidth=2)
            ax.add_artist(con)
            
            xyA = (boudingBox2[2]*xImgS, boudingBox2[3]*yImgS)
            xyB = (boudingBox2[10]*xImgS, boudingBox2[11]*yImgS)
            con = patches.ConnectionPatch(xyA, xyB, "data", "data", shrinkA=0, shrinkB=0,
                                  mutation_scale=20, fc="b",edgecolor='b',linewidth=2)
            ax.add_artist(con)
    
            xyA = (boudingBox2[4]*xImgS, boudingBox2[5]*yImgS)
            xyB = (boudingBox2[12]*xImgS, boudingBox2[13]*yImgS)
            con = patches.ConnectionPatch(xyA, xyB, "data", "data", shrinkA=0, shrinkB=0,
                                  mutation_scale=20, fc="b",edgecolor='b',linewidth=2)
            ax.add_artist(con)    
            
            xyA = (boudingBox2[6]*xImgS, boudingBox2[7]*yImgS)
            xyB = (boudingBox2[14]*xImgS, boudingBox2[15]*yImgS)
            con = patches.ConnectionPatch(xyA, xyB, "data", "data", shrinkA=0, shrinkB=0,
                                  mutation_scale=20, fc="b",edgecolor='b',linewidth=2)
            ax.add_artist(con)            
            
        else:
            print('Error: keyword: ' + bbType + ' unknown')
        
        