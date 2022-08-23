import numpy as np
from matplotlib import pyplot as plt
import xir
import vart
import time 

class coreDPU:
    def __init__(self,model_path):
        g = xir.Graph.deserialize(model_path)
        sub = []
        root = g.get_root_subgraph()

        child_subgraph = root.toposort_child_subgraph()
        sub = [s for s in child_subgraph
              if s.has_attr("device") and s.get_attr("device").upper() == "DPU"]
        self.model = sub
        self.__createDPU()
        
    def __createDPU(self):
        self.__dpu = vart.Runner.create_runner(self.model[0],"run")
        input_tensors = self.__dpu.get_input_tensors() #得模型DPU輸入層
        output_tensors = self.__dpu.get_output_tensors() #得模型運算DPU最終層
        self.__input_dims = tuple(input_tensors[0].dims) #(1,224,224,3)
        self.__output_dims = tuple(output_tensors[0].dims) #(1,6)
    
    def runDPU(self,img) ->int:
        input_data = []
        output_data = []
        input_data = [np.empty(self.__input_dims, dtype =np.float32, order = "C")]
        output_data = [np.empty(self.__output_dims, dtype = np.float32, order = "C")]
        dpu_image = input_data[0]
        dpu_image[0,...] =  img #圖進DPU的BUFFER,

        self.__dpu.execute_async(dpu_image,output_data)
        ans = np.argmax(output_data[0]) +1#分類
        #print(output_data[0])
        #print('AQI_LEVEL:' + str(ans+1))
        return ans