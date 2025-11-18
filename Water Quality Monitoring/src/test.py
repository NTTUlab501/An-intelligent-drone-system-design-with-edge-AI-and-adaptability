#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
import time
import threading
import queue
from tqdm import tqdm
from typing import List

import xir
import vart

# If there's no graphical interface, avoid using notebook version of tqdm or GUI backends
# import matplotlib
# matplotlib.use('Agg')

# Custom spectral transfer tool
from Spectrum import SpectrumTransfer

# Evaluation metrics
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


# --------------------- #
# 1) Parameter Settings
# --------------------- #

# Spectral transfer weight file
SpectrumTransfer_path = 'weight.npz'
# Assuming 5 bands selected => output shape (224,224,5)
band_list = [6, 7, 8, 9, 10]

# Test image folder (subfolders "a_Heavy_pollution", "b_Menium_pollution", "c_Mild_pollution")
datapath = './0417class3train/test/'

# Class label mapping and class names
dicClass = {'a_Heavy_pollution': 0, 'b_Menium_pollution': 1, 'c_Mild_pollution': 2}
class_names = ['Heavy_pollution', 'Menium_pollution', 'Mild_pollution']

norm_size = 224  # Expected input size for xmodel

# xmodel paths (three models)
model_path_hsv   = './0409data_model/xmodel_1600-dual/multimodal_hsv_B1600.xmodel'   # HSV model (input shape: (1,224,224,3))
model_path_hsi   = './0409data_model/xmodel_1600-dual/multimodal_hsi_B1600.xmodel'   # HSI model (input shape: (1,224,224,5))
model_path_merge = './0409data_model/xmodel_1600-dual/multimodal_merge_B1600.xmodel' # Merged model

# Use two worker threads (set based on the FPGA's DPU core count; here, 2 is assumed)
NUM_WORKERS = 2

# Global list to store model loading times from each worker (in seconds)
model_loading_times = [] 
model_loading_lock = threading.Lock()


# --------------------- #
# 2) Define the coreDPU Class
# --------------------- #
class coreDPU:
    def __init__(self, model_path):
        g = xir.Graph.deserialize(model_path)
        subgraphs = self.__get_child_subgraph_dpu(g)
        self.model = subgraphs
        self.__createDPU()

    def __get_child_subgraph_dpu(self, graph: "Graph") -> List["Subgraph"]:
        assert graph is not None, "'graph' should not be None."
        root_subgraph = graph.get_root_subgraph()
        assert root_subgraph is not None, "Failed to get root subgraph."
        if root_subgraph.is_leaf:
            return []
        child_subgraphs = root_subgraph.toposort_child_subgraph()
        assert child_subgraphs and len(child_subgraphs) > 0, "No child subgraph found."
        return [cs for cs in child_subgraphs 
                if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"]

    def __createDPU(self):
        self.dpu = vart.Runner.create_runner(self.model[0], "run")
        input_tensors = self.dpu.get_input_tensors()
        output_tensors = self.dpu.get_output_tensors()
        self.__input_dims = tuple(input_tensors[0].dims)     # e.g., (1,224,224,3) or (1,224,224,5)
        self.__output_dims = tuple(output_tensors[0].dims)   # e.g., (1,8,8,20) or (1,2048)

    def __CPUCalcSoftmax(self, data):
        exps = np.exp(data)
        return exps / np.sum(exps)

    def runDPU(self, img, apply_softmax=False):
        """
        img: Input image tensor (shape must match xmodel's expectation)
        apply_softmax: Whether to apply softmax here
        """
        input_data = [np.empty(self.__input_dims, dtype=np.float32, order="C")]
        output_data = [np.empty(self.__output_dims, dtype=np.float32, order="C")]
        
        input_data[0][0, ...] = img
        job_id = self.dpu.execute_async(input_data, output_data)
        self.dpu.wait(job_id)
        
        out = np.squeeze(output_data[0])
        if apply_softmax:
            return self.__CPUCalcSoftmax(out)
        else:
            return out


# --------------------- #
# 3) Load Test Data
# --------------------- #
def load_test_images(root_path):
    imageList = []
    labelList = []
    sub_folders = os.listdir(root_path)
    print("Detected subfolders:", sub_folders)
    
    for class_name in sub_folders:
        if class_name not in dicClass:
            print(f"Unknown folder: {class_name}, skipping")
            continue
        label_id = dicClass[class_name]
        class_path = os.path.join(root_path, class_name)
        for f in os.listdir(class_path):
            full_path = os.path.join(class_path, f)
            imageList.append(full_path)
            labelList.append(label_id)
    
    print(f"Read a total of {len(imageList)} images")
    return imageList, np.array(labelList)


# --------------------- #
# 4) Define the Local Inference Function Using DPU and Spectral Transfer
# --------------------- #
def process_image_local(img_path, dpu_hsv, dpu_hsi, dpu_merge, transfer_instance):
    # ----- Image Capture Timing -----
    t0 = time.time()
    # Read and decode image
    raw_data = np.fromfile(img_path, dtype=np.uint8)
    image = cv2.imdecode(raw_data, -1)
    if image is None:
        print("Failed to read image:", img_path)
        return None, 0, 0, 0
    # If image has 4 channels, take only the first 3 channels
    if image.shape[2] > 3:
        image = image[:, :, :3]
    # Preprocess A: Convert to HSV (3 channels)
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_img = cv2.resize(hsv_img, (norm_size, norm_size), interpolation=cv2.INTER_LANCZOS4)
    hsv_input = np.expand_dims(hsv_img, axis=0).astype(np.float32)
    # Preprocess B: Convert to HSI (5 channels)
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_img = cv2.resize(rgb_img, (norm_size, norm_size), interpolation=cv2.INTER_LANCZOS4)
    spec_img = transfer_instance.transfer_select(rgb_img, interval=40, band=band_list).astype(np.float32)
    hsi_input = np.expand_dims(spec_img, axis=0)
    t1 = time.time()
    capture_time = (t1 - t0) * 1000  # in milliseconds
    
    # ----- Model Inference Timing -----
    # Run HSV and HSI models concurrently in two threads
    t2 = time.time()
    results = {}
    def run_hsv():
        results['hsv'] = dpu_hsv.runDPU(hsv_input, apply_softmax=False)
    def run_hsi():
        results['hsi'] = dpu_hsi.runDPU(hsi_input, apply_softmax=False)
    
    th_hsv = threading.Thread(target=run_hsv)
    th_hsi = threading.Thread(target=run_hsi)
    th_hsv.start(); th_hsi.start()
    th_hsv.join(); th_hsi.join()
    t3 = time.time()
    
    if 'hsv' not in results or 'hsi' not in results:
        print("Submodel execution failed for:", img_path)
        return None, capture_time, 0, 0
    
    hsv_out = np.squeeze(results['hsv'])
    hsi_out = np.squeeze(results['hsi'])
    # Concatenate outputs from both models
    concat_out = np.concatenate((hsv_out, hsi_out), axis=-1)
    merge_input = concat_out.reshape(1, 8, 8, 32)
    
    # Run merged model and time the output step
    t4 = time.time()
    merge_out = dpu_merge.runDPU(merge_input, apply_softmax=True)
    t5 = time.time()
    output_time = (t5 - t4) * 1000  # in milliseconds
    inference_time = (t3 - t2) * 1000 + output_time  # total model inference time in ms
    
    pred = int(np.argmax(np.squeeze(merge_out)))
    return pred, capture_time, inference_time, output_time


# --------------------- #
# 5) Define Worker Thread Function (each thread initializes its local DPU and spectral transfer)
# --------------------- #
def worker_local(input_queue: queue.Queue, output_queue: queue.Queue, pbar):
    # Measure model loading time for this worker
    load_start = time.time()
    local_dpu_hsv   = coreDPU(model_path_hsv)
    local_dpu_hsi   = coreDPU(model_path_hsi)
    local_dpu_merge = coreDPU(model_path_merge)
    local_transfer  = SpectrumTransfer()
    local_transfer.load(SpectrumTransfer_path)
    load_end = time.time()
    worker_model_load_time = load_end - load_start  # in seconds
    with model_loading_lock:
        model_loading_times.append(worker_model_load_time)
    
    while True:
        try:
            img_path, label = input_queue.get(block=False)
        except queue.Empty:
            break
        
        result = process_image_local(img_path, local_dpu_hsv, local_dpu_hsi, local_dpu_merge, local_transfer)
        if result[0] is not None:
            # result is a tuple: (pred, capture_time, inference_time, output_time)
            output_queue.put((img_path, label, result[0], result[1], result[2], result[3]))
        input_queue.task_done()
        pbar.update(1)
    
    # Release local resources if needed
    del local_dpu_hsv, local_dpu_hsi, local_dpu_merge


# --------------------- #
# 6) Main Program: Initialization, Build Queues, Start Worker Threads, and Aggregate Results
# --------------------- #
if __name__ == '__main__':
    t0_all = time.time()
    
    print("Starting data loading...")
    test_images, test_labels = load_test_images(datapath)
    print("Data loading completed.")
    
    # Build input and output queues
    input_queue = queue.Queue()
    output_queue = queue.Queue()
    
    for img_path, label in zip(test_images, test_labels):
        input_queue.put((img_path, label))
    
    pbar = tqdm(total=len(test_images), desc="Optimized Inference", unit="img")
    
    workers = []
    for _ in range(NUM_WORKERS):
        t = threading.Thread(target=worker_local, args=(input_queue, output_queue, pbar))
        t.start()
        workers.append(t)
    
    input_queue.join()
    for t in workers:
        t.join()
    pbar.close()
    
    t1_all = time.time()
    total_time = t1_all - t0_all
    total_frames = len(test_images)
    fps = total_frames / total_time if total_time > 0 else 0
    
    # Aggregate prediction results and timing metrics
    y_true = []
    y_pred = []
    capture_times = []
    inference_times = []
    output_times = []
    while not output_queue.empty():
        _, label, pred, cap_time, inf_time, out_time = output_queue.get()
        y_true.append(label)
        y_pred.append(pred)
        capture_times.append(cap_time)
        inference_times.append(inf_time)
        output_times.append(out_time)
    
    avg_capture_time = np.mean(capture_times) if capture_times else 0
    avg_inference_time = np.mean(inference_times) if inference_times else 0
    avg_output_time = np.mean(output_times) if output_times else 0
    avg_model_loading_time = np.mean(model_loading_times) if model_loading_times else 0
    
    print(f"\nInferred {total_frames} images, Total time = {total_time:.4f} seconds, FPS = {fps:.2f}")
    print(f"Model Loading Time: {avg_model_loading_time:.6f} s")
    print(f"Image Capture: {avg_capture_time:.6f} ms")
    print(f"Model Inference: {avg_inference_time:.6f} ms")
    print(f"Output: {avg_output_time:.6f} ms")
    
    # Calculate evaluation metrics
    labels = [0, 1, 2]
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    
    print(" " * 32 + "precision    recall    f1-score")
    print("-" * 55)
    for i, lbl in enumerate(labels):
        print(f"{class_names[lbl]:<32}{precision[i]:>10.5f}{recall[i]:>10.5f}{f1[i]:>10.5f}")
    print("-" * 55)
    print(f"Accuracy:{accuracy:>36.5f}")
