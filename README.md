# DeepLearning
Bacteria features prediction using CRNA (Convolutional Regression Neural Network) 


Technical Stack: Python3, Keras, Pandas, Numpy, TensorFlow, 
 
Install the following packages : 
 
• numpy 
• pandas
• xlrd 
• cv2 
• tensorflow 
• keras 
• re 
• json 
 
Command used to evaluate the  model : 
 
model.eval(loc1,loc2,loc3,loc4); 
 
loc1 = excel sheet path(Please use the excel sheet attached as it has only variables that have to be predicted ). Loc2 = Test Images Path(Images that has to be tested) Loc3 = Model Path(Model Attached in the mail) Loc4 = Weights Path(Weights File attached in the mail) 

Data Augmentation:
Original Dataset – 266 Images

Pre-processing techniques:
1. Rotate at random degrees
2. Flip Left-Right, top-bottom
3. Add gaussian blur
4. Changing color Saturation
5. Compressed and resized to 256x256 pixels

Total Images Generated: 2000 Images
Training Images Sampled: 1600(80%)
Testing Images Sampled: 400 (20%)


Neural Network Implementation:
Network selected: Convolutional Neural Network
Network Architecture:
1. 2D-Convulutuional Layer (32 Filters of Size 3x3), Activation – Relu
2. 2D-Convulutuional Layer (32 Filters of Size 3x3), Activation – Relu
3. 2D-MaxPooling layer (pool_size = (2,2)), Dropout (0.25)
4. 2D-Convulutuional Layer (32 Filters of Size 3x3), Activation – Relu
5. 2D-Convulutuional Layer (32 Filters of Size 3x3), Activation – Relu
6. 2D-MaxPooling layer (pool_size = (2,2)), Dropout (0.25)
7. 2D-Convulutuional Layer (32 Filters of Size 3x3), Activation – Relu
8. 2D-Convulutuional Layer (32 Filters of Size 3x3), Activation – Relu
9. 2D-MaxPooling layer (pool_size = (3,3)), Dropout (0.25)
10. Flatten layer
11. Dense layer (neurons: 512), Activation – Relu, Dropout (0.5)
12. Dense layer (neurons: 256), Activation – Relu, Dropout (0.5)
13. Dense layer (neurons: 29), Activation – Relu, Dropout (0.25)
14. Dense layer (neurons: 18), Activation – Relu, Dropout (0.25)
