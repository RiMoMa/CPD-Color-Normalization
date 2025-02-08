import numpy as np
import os
from skimage import measure
import matplotlib.pyplot as plt
from imageio import imread, imsave
from stardist.models import StarDist2D
from csbdeep.utils import normalize

# Load the pretrained model
model = StarDist2D.from_pretrained('2D_versatile_he')
	
def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            img_path = os.path.join(input_folder, filename)
            img = imread(img_path)

            # Predict instance segmentation map
            labels, _ = model.predict_instances(normalize(img))

            # Save the `inst_map` as a .npy file (NumPy array)
            np.save(os.path.join(output_folder, f'{os.path.splitext(filename)[0]}_inst_map.npy'), labels)

            # Save as a .tiff file (if needed)
            imsave(os.path.join(output_folder, f'{os.path.splitext(filename)[0]}_inst_map.tiff'), labels.astype(np.uint16))

            # Find contours for visualization
            contours = measure.find_contours(labels, level=0.5)

            # Overlay contours on the input image
            fig, ax = plt.subplots()
            ax.imshow(img, cmap='gray')
            ax.axis('off')

            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], linewidth=0.4, color='#52f212')

            # Save the visualization
            output_path = os.path.join(output_folder, f'{os.path.splitext(filename)[0]}.png')
            fig.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
            plt.close(fig)

# Define input and output folders
input_folder = "/media/daedro/Datos/CPD/CPD_Datasets/LizardDataset/archive/datasets/od/TCGA-21-5784-01Z-00-DX1/"
output_folder = "/media/daedro/Datos/CPD/CPD_Datasets/LizardDataset/archive/segmented/maps/od/TCGA-21-5784-01Z-00-DX1/"

# Run processing
process_images(input_folder, output_folder)
