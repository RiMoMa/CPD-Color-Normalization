import time
import torch
import os
import numpy as np
from skimage import io, transform
from segment_anything import sam_model_registry
import torch.nn.functional as F

SAM_MODEL_TYPE = "vit_b"
MedSAM_CKPT_PATH = "work_dir/MedSAM/medsam_vit_b.pth"
MEDSAM_IMG_INPUT_SIZE = 1024

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Freeze seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()
torch.cuda.manual_seed(2023)
np.random.seed(2023)


def medsam_inference(medsam_model, img_embed, box_1024, height, width):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().detach().cpu().numpy()  # Detach before converting to NumPy
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg


def save_mask(mask, output_path):
    # Enhance contrast: Multiply mask values to stretch the range to [0, 255]
    mask = mask * 20  # Scale up the mask to make it more visible
    
    # Ensure it's within the range [0, 255] and convert to uint8
    mask = np.clip(mask, 0, 255).astype(np.uint8)
    
    # Save the mask to an image file
    io.imsave(output_path, mask)


def get_mask_from_image(image_path):
    # Load the BMP image
    img_np = io.imread(image_path)
    if len(img_np.shape) == 2:
        img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)  # Convert to 3-channel if grayscale
    else:
        img_3c = img_np

    # Resize to 1024x1024 for the model
    H, W, _ = img_3c.shape
    img_1024 = transform.resize(
        img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
    ).astype(np.uint8)

    img_1024 = (img_1024 - img_1024.min()) / np.clip(
        img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
    )  # Normalize to [0, 1]

    # Convert to tensor and add batch dimension
    img_1024_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)

    # Load MedSAM model
    medsam_model = sam_model_registry["vit_b"](checkpoint=MedSAM_CKPT_PATH).to(device)
    medsam_model.eval()

    # Get image embedding
    with torch.no_grad():
        embedding = medsam_model.image_encoder(img_1024_tensor)  # (1, 256, 64, 64)

    # Define the bounding box for the whole image (scaled to 1024)
    box_1024 = np.array([[0, 0, W, H]]) / np.array([W, H, W, H]) * 1024  # Box for the full image

    # Perform segmentation inference
    mask = medsam_inference(medsam_model, embedding, box_1024, H, W)

    return mask


def process_images_in_folder(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all BMP files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.bmp'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename.replace('.bmp', '_mask.png'))

            # Get the mask for the current image
            mask = get_mask_from_image(input_path)

            # Save the mask with contrast enhancement
            save_mask(mask, output_path)

            print(f"Processed {filename}, mask saved to {output_path}")


# Example usage
input_folder = "/media/daedro/Datos/CPD/CPD_Datasets/GlasDataset/Vahadane/"
output_folder = "/media/daedro/Datos/CPD/CPD_Datasets/GlaSMasks/vahadane/"

process_images_in_folder(input_folder, output_folder)
