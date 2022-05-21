# -*- coding: utf-8 -*-
"""
Created on Sat May 21 19:01:54 2022

@author: neera
"""

#loading the Signature dataset
path_train = "/kaggle/working/UCSD_Anomaly_Dataset_mod/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train/"
dir_list = []
for i in range(1, 16, 1):
  dir_list.append(path_train + "Train" + str(i).zfill(3)+"/")
#dir_list = next(os.walk(path_train))[1]
#dir_list.sort()

#for training and testing data
trainer, trainer2 = [], []
train_output, train_output2 = [], []
#for storing the labels
test_labels = np.array([])

#creating a mix of forged and genuine signatures to create a training and testing dataset

for path in dir_list:
    bunch = []
    for i in range(1, 121, 1):
        file_path = path + str(i).zfill(3)+".tif"
        temp = cv2.imread(file_path)
        #constant= cv2.copyMakeBorder(temp, 1, 1, 1, 1,cv2.BORDER_CONSTANT,value=black)
        temp = cv2.resize(temp, (128, 64))
        bunch.append(temp)
        if (i > 8):
            train_output.append(temp)
    for start in range(0, 112, 1):
        end = start + 8
        trainer.append(bunch[start:end])
    print(np.asarray(trainer).shape)
    #shufflilng the images
#random.shuffle(images)
'''black = [0, 0, 0]
for img in images:
  temp = cv2.imread(img)
  #constant= cv2.copyMakeBorder(temp, 1, 1, 1, 1,cv2.BORDER_CONSTANT,value=black)
  temp = cv2.resize(temp, (128, 64))

  trainer.append(temp)

#horizontal flip
for img in trainer[:1800]:
  temp = cv2.flip(img, 1)
  trainer.append(temp)

#vertical flip
for img in trainer[:1800]:
  temp = cv2.flip(img, 0)
  trainer.append(temp)

#vertical and horizontal flip
for img in trainer[:1800]:
  temp = cv2.flip(img, -1)
  trainer.append(temp)
'''
for image in trainer:
    trainer2.append(image[0])
trainer2 = np.array(trainer2)
print(trainer2.shape)
train_output = np.array(train_output)
trainer = np.array(trainer)
print(trainer.shape)
print(train_output.shape)
plt.subplot(221),
plt.imshow(trainer[1][7], cmap = 'gray')
plt.subplot(222),
plt.imshow(train_output[0], cmap = 'gray')
plt.subplot(223),
plt.imshow(trainer2[0], cmap = 'gray')

trainer = trainer.astype('float32') / 255.0
trainer2 = trainer2.astype('float32') / 255.0
train_output = train_output.astype('float32') / 255.0
plt.imshow(trainer[1][7], cmap = 'gray')