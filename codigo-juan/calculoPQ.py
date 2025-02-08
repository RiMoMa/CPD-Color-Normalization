import glob
import os
import numpy as np
import scipy.io as sio
from tqdm import tqdm

# Función para ver los labels únicos
def get_unique_labels(inst_map):
    unique_labels = np.unique(inst_map)
    return unique_labels[unique_labels > 0]

# Cálculo de la métrica
def panoptic_quality(gt, pred):
    gt_labels = get_unique_labels(gt)
    pred_labels = get_unique_labels(pred)
    
    tp = 0  # True Positives
    fp = 0  # False Positives
    fn = 0  # False Negatives
    
    for gt_label in gt_labels:
        if gt_label in pred_labels:
            tp += 1
            pred_labels = np.setdiff1d(pred_labels, [gt_label])  # Remove matched prediction label
        else:
            fn += 1  # If no match, then False Negative
    
    # All remaining predicted labels are False Positives
    fp = len(pred_labels)
    
    if tp + 0.5 * fp + 0.5 * fn == 0:
        return 0.0
    
    pq = tp / (tp + 0.5 * fp + 0.5 * fn)
    return pq

def process_images(gt_folder, pred_folder):
    pq_scores = []
    gt_files = glob.glob(os.path.join(gt_folder, "*.mat"))
    
    for gt_path in tqdm(gt_files, desc=f"Processing {pred_folder}"):
        image_id = os.path.basename(gt_path).replace(".mat", "")
        pred_path = os.path.join(pred_folder, f"{image_id}_inst_map.npy")
        
        if not os.path.exists(pred_path):
            continue
        
        gt_data = sio.loadmat(gt_path)['inst_map']
        pred_data = np.load(pred_path)
        
        pq = panoptic_quality(gt_data, pred_data)
        pq_scores.append(pq)
    
    return pq_scores

# Rutas de datasets
gt_folder = "/kaggle/input/lizard-stardist/GroundTruth"
non_norm_folder = "/kaggle/input/originals-nonnormalized-realones/Original - NonNormalized"
norm1_folder = "/kaggle/input/lizard-stardist/RGB/Stardist-monuseg7130"
norm2_folder = "/kaggle/input/lizard-stardist/RGB/Stardist-monusegA561"

# PQ
pq_non_norm = process_images(gt_folder, non_norm_folder)
pq_norm1 = process_images(gt_folder, norm1_folder)
pq_norm2 = process_images(gt_folder, norm2_folder)

average_score_nonNorm = np.mean(pq_non_norm)
std_dev_nonNorm = np.std(pq_non_norm)

average_score_norm1 = np.mean(pq_norm1)
std_dev_norm1 = np.std(pq_norm1)

average_score_norm2 = np.mean(pq_norm2)
std_dev_norm2 = np.std(pq_norm2)

print(f"Panoptic Quality promedio para imágenes sin normalizar -> {average_score_nonNorm:.4f}")
print(f"Desviacion estándar para imágenes sin normalizar -> {std_dev_nonNorm:.4f}")
print("")
print(f"Panoptic Quality promedio para imágenes normalizadas con Monuseg7130 -> {average_score_norm1:.4f}")
print(f"Desviacion estándar para imágenes normalizadas con Monuseg7130 -> {std_dev_norm1:.4f}")
print("")
print(f"Panoptic Quality promedio para imágenes normalizadas con MonusegA561 -> {average_score_norm2:.4f}")
print(f"Desviacion estándar para imágenes normalizadas con MonusegA561 -> {std_dev_norm2:.4f}")
