#!/usr/bin/env python3

import cv2
import csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
## Submission by Ujjwal Gupta and Mahdi Ghanei

class ImageProcess:
    def __init__(self):
        # default 
        self.height, self.width = 200, 200
    
    def getImageDir(self, mode='train'):
        '''Return image directory
        '''
        imageDirectory = './2021imgs/' + mode + '_images/'
        return imageDirectory
    
    def readLabels(self, mode='train'):
        ''' Load training images and labels
        '''
        imageDirectory = self.getImageDir(mode)
        with open(imageDirectory + mode + '.txt', 'r') as f:
            reader = csv.reader(f)
            lines = list(reader)
        return lines

    def getMaskImage(self, img):
        '''creates a mask of an image
        '''
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        m, n, _ = np.shape(img)

        # increase the brightness of the middle region (area of interest)
        hsv_img[m//5:4*m//5,:,2] += 50

        # define range of bgr color in HSV
        # get the suitable values by running hsv_bounds.py
        lower_bgr = np.array([0,80,60])
        upper_bgr = np.array([255,255,255])

        # mask coloured signs
        mask = cv2.inRange(hsv_img, lower_bgr, upper_bgr)
        return mask

    def cropImage(self, pre_processed_img, mask, resize1, resize2):
        '''Crops an image based on mask 
        '''
        # default 
        m, n, _ = np.shape(pre_processed_img)

        # coordinates of the center of the sign
        x_shift = 100
        y_shift = 100
        if (mask==0).all(): # if no sign is there
            x_coord, y_coord = x_shift, y_shift
        else:
            x_coord, y_coord = np.mean(np.where(mask>0),axis=1)
        x_coord = int(x_coord)
        y_coord = int(y_coord)

        processed_img = cv2.bitwise_and(pre_processed_img, pre_processed_img, mask=mask)
        
        if x_coord - x_shift > 0 and x_coord + x_shift < m and y_coord - y_shift > 0 and y_coord + y_shift < n:
            processed_img = processed_img[x_coord-x_shift:x_coord+x_shift,y_coord-y_shift:y_coord+y_shift]
        else:
            processed_img = cv2.resize(processed_img,(self.height, self.width),interpolation=cv2.INTER_AREA)

        processed_img = cv2.resize(processed_img,(resize1, resize2),interpolation=cv2.INTER_AREA)
        
        return processed_img

img_process = ImageProcess()
##############################################
### Process training images
lines = img_process.readLabels(mode='train')
imageDirectory = img_process.getImageDir(mode='train')

resize1, resize2 = 32, 32
train = np.array([]).reshape(0,3 * resize1 * resize2)
for i in range(len(lines)):
    img = cv2.imread(imageDirectory +lines[i][0]+".jpg")
    pre_processed_img = img.copy()

    # get mask
    mask = img_process.getMaskImage(pre_processed_img)
    # crop image based on mask
    processed_img = img_process.cropImage(pre_processed_img, mask, resize1, resize2)
    
    processed_img = processed_img.flatten().reshape(1,-1)
    train = np.vstack((train,processed_img))

train_data = train.astype(np.float32)


##############################################
### Train calssifier
# read in training labels
train_labels = np.array([np.int32(lines[i][1]) for i in range(len(lines))])

### knn classifier
knn = cv2.ml.KNearest_create()
knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)
k = 7
print('Training done!')

if(__debug__):
	Title_images = 'Original Image'
	Title_processed = 'Processed Image'
	cv2.namedWindow( Title_images, cv2.WINDOW_AUTOSIZE )

##############################################
### Test on the test set

lines = img_process.readLabels(mode='test')
imageDirectory = img_process.getImageDir(mode='test')

correct = 0.0
confusion_matrix = np.zeros((6,6))
for i in range(len(lines)):
    original_img = cv2.imread(imageDirectory+lines[i][0]+".jpg")
    pre_processed_img = original_img.copy()
    ### process test images
    new_mask = img_process.getMaskImage(pre_processed_img)
    processed_img = img_process.cropImage(pre_processed_img, new_mask, resize1, resize2)

    if(__debug__):
        cv2.imshow(Title_images, original_img)
        cv2.imshow(Title_processed, processed_img)
        key = cv2.waitKey()
        if key==27:    # Esc key to stop
            break

    test_img = processed_img.flatten().reshape(1,-1)
    test_img = test_img.astype(np.float32)
    test_label = np.int32(lines[i][1])

    #classify on the bassis of majority
    ret, results, neighbours, dist = knn.findNearest(test_img, k)
    
    # modify the classification on the basis of weights
    weights = []
    near_neighbours = np.unique(neighbours)
    for j in near_neighbours:
        index = np.where(neighbours == j)
        inv_dist = np.reciprocal(dist[index]+0.1)  # add 0.1 to avoid the cases of divisibility by 0
        weight = np.sum(inv_dist)
        weights.append(weight)
       
    ret = near_neighbours[np.argmax(weights)]

    if test_label == ret:
        print(str(lines[i][0]) + " Correct, " + str(ret))
        correct += 1
        confusion_matrix[np.int32(ret)][np.int32(ret)] += 1
    else:
        confusion_matrix[test_label][np.int32(ret)] += 1
        
        print(str(lines[i][0]) + " Wrong, " + str(test_label) + " classified as " + str(ret))
        print("\tneighbours: " + str(neighbours))
        print("\tdistances: " + str(dist))

print("\n\nTotal accuracy: " + str(correct/len(lines)))
print(confusion_matrix)

