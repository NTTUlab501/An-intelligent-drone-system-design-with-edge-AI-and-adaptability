import cv2
import os
import numpy as np
from IPython.display import clear_output
from tqdm import tqdm, trange

from classDPU import coreDPU
image_path = "/home/petalinux/dev/testimg/"
model_path = "/home/petalinux/dev/KV260_4096.xmodel"

table = np.zeros((6,6),dtype=int)

DPU_Class = coreDPU(model_path)

for i in range(1,7):
    print("Image_Folder_Label:",i)

    image_i_folder = image_path+"{num}/".format(num=i)
    for filename in tqdm(os.listdir(image_i_folder),desc="Image_Folder_{} ".format(i),ncols=100):
        image_location = image_i_folder + filename
        image = cv2.imread(image_location)
        image_resize = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        level = DPU_Class.runDPU(image_resize) - 1
        table[level][i-1] +=1


class model_Evaluate:
    def __init__(self,table):
        df = pd.DataFrame(table)
        df['row_sum']=df.apply(lambda x: x.sum(),axis=1)
        df.loc['col_sum']=df.apply(lambda x: x.sum(),axis=0)
        self.__df = df 
        
    def getAccuracy(self):
        Accuracy = 0
        TP_TN = 0 #所有類別正確數之合
        TP_TN_FP_FN = 0 #圖片總數

        for i in range(6):
            TP_TN += self.__df[i][i]

        TP_TN_FP_FN = self.__df['row_sum'].values[-1]
        Accuracy = (TP_TN/TP_TN_FP_FN)
        return Accuracy
    def getPrecision(self):
        Precision = np.zeros((6,),dtype=float)

        for i in range(6):
            TP = self.__df[i][i]
            TP_FP = self.__df['row_sum'].values[i]
            Precision[i] = TP/TP_FP
        Precision_df = pd.DataFrame(Precision,columns = ['Precision'])
        Precision_df = Precision_df.applymap(lambda x: '%.2f%%' % (x*100))
        return Precision_df['Precision'].tolist()
    
    def getRecall(self):
        Recall = np.zeros((6,),dtype=float)
        for i in range(6):
            TP = self.__df[i][i]
            TP_FN = self.__df.iloc[-1].values[i]
            Recall[i] =  TP/TP_FN
        Recall_df = pd.DataFrame(Recall,columns = ['Recall'])
        Recall_df = Recall_df.applymap(lambda x: '%.2f%%' % (x*100))
        return Recall_df['Recall'].tolist()
        
