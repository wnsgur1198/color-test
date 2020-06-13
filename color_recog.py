# -*- coding: UTF-8 -*-
import numpy as np
import cv2
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from webcolors import rgb_to_name


image_path = "images/0.jpg"
#color_class = ['white','black','red','green','blue','yellow','magenta','cyan']
color_class = ['하얀','검정','빨간','초록','파란','노란','분홍색','민트색', "갈색", "보라색", "주황색", "연두색", "진한회색", "연한회색"]

class ColorRecog:
    def __init__(self):
      #print('color recog')
      pass

    def centroid_histogram(self, clt):
        # grab the number of different clusters and create a histogram
        # based on the number of pixels assigned to each cluster
        numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
        (hist, _) = np.histogram(clt.labels_, bins=numLabels)

        # normalize the histogram, such that it sums to one
        hist = hist.astype("float")
        hist /= hist.sum()

        # return the histogram
        return hist

    def plot_colors(self, hist, centroids):
        clrRecog = ColorRecog()
        
        # initialize the bar chart representing the relative frequency
        # of each of the colors
        bar = np.zeros((50, 300, 3), dtype="uint8")
        startX = 0

        first_color = -1 
        second_color = -1
        color_name = ''

        # loop over the percentage of each cluster and the color of
        # each cluster
        for (percent, color) in zip(hist, centroids):
            
            #print(percent)
            #print(color)
            
            ## Most Color
            if first_color < percent:
                
                first_color = percent
                first_color_list = color
                
                red = round(first_color_list[0])
                green = round(first_color_list[1])
                blue = round(first_color_list[2])
                
                first_color_list_int = (red, green, blue)            
                  
            ## Second-most Color
            elif second_color < first_color and second_color < percent:
                second_color = percent
                second_color_list = color
                
                red = round(second_color_list[0])
                green = round(second_color_list[1])
                blue = round(second_color_list[2])

                second_color_list_int = (red, green, blue)   
            
            # plot the relative percentage of each cluster
            endX = startX + (percent * 300)
            cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                          color.astype("uint8").tolist(), -1)
            startX = endX
            
        ## classify color
        for i in range(len(color_class)):          
            if i == 0:
                for i in range(len(color_class)):            
                    if clrRecog.classify_color(second_color_list_int) == color_class[i]:
                        color_name = color_class[i]        
                        
            elif clrRecog.classify_color(first_color_list_int) == color_class[i]:
                color_name = color_class[i]  
            
        ## print Color Name    
        # print("first: ")
        # print(first_color_list_int)
        # print("second: ")
        # print(second_color_list_int)
        # print(color_name)

        # return the bar chart
        return bar, color_name
        
    # classify color ------------------------------------------------
    def classify_color(self, rgb_tuple):
        # colors = { "white" : (255,255,255),
                   # "black" : (0,0,0),
                   # "red": (255,0,0),
                   # "green" : (0,255,0),
                   # "blue" : (0,0,255),               
                   # "yellow" : (255,255,0),
                   # "magenta" : (255,0,255),
                   # "cyan" : (0,255,255),
                # }
        colors = { "하얀" : (255,255,255),
                   "검정" : (0,0,0),
                   "빨간": (255,0,0),
                   "초록" : (0,255,0),
                   "파란" : (0,0,255),               
                   "노란" : (255,255,0),
                   "분홍색" : (255,0,255),
                   "민트색" : (0,255,255),
                   "갈색" : (128,64,0),
                   "보라색" : (128,0,128),
                   "보라색" : (114,77,188),
                   "주황색" : (255,128,0),
                   "연두색" : (0,255,64),
                   "진한회색" : (128,128,128),
                   "연한회색" : (192,192,192),
                }
                
        manhattan = lambda x,y : abs(x[0] - y[0]) + abs(x[1] - y[1]) + abs(x[2] - y[2]) 
        distances = {k: manhattan(v, rgb_tuple) for k, v in colors.items()}
        color = min(distances, key=distances.get)
        
        return color

    # show image color cluster --------------------------------
    def image_color_cluster(self, image_path, k = 5):
        clrRecog = ColorRecog()
    
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.reshape((image.shape[0] * image.shape[1], 3))
        
        clt = KMeans(n_clusters = k)
        clt.fit(image)

        hist = clrRecog.centroid_histogram(clt)
        bar, color_name = clrRecog.plot_colors(hist, clt.cluster_centers_)

        print(color_name)
        
        plt.figure()
        plt.axis("off")
        plt.imshow(bar)
        plt.show()

        return color_name

# before cluster
image = mpimg.imread(image_path)
plt.imshow(image)

# after cluster
ColorRecog().image_color_cluster(image_path)
