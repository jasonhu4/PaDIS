import pydicom
import numpy as np
import os
from PIL import Image
import imageio.v2 as imageio
from scipy.io import savemat

################################################################################################################
# https://www.aapm.org/grandchallenge/lowdosect/
# 2021 Update: Low Dose CT Challenge Images and Information Available => You can download it from here
#
# https://www.cancerimagingarchive.net/collection/spie-aapm-lung-ct-challenge/
# For dataset mentioned in the paper, Download DICOM file from this website.
# https://www.youtube.com/watch?v=NO48XtdHTic => refer to this tutorial video to download DCM format ct images
#
# The raw dataset should be in DCM format, if not comment out convert_folder.
################################################################################################################
def dcm_to_png(input_path, output_path):
    dicom_data = pydicom.dcmread(input_path)
    
    image_array = dicom_data.pixel_array
    
    image_array = image_array - np.min(image_array)
    image_array = (image_array / np.max(image_array) * 255).astype(np.uint8)
    
    image = Image.fromarray(image_array)
    image.save(output_path)
    print(f"Converted DICOM file saved as PNG: {output_path}")

def convert_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".dcm"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename.replace(".dcm", ".png"))
            dcm_to_png(input_path, output_path)

main_dir = /home/
input_path = r"PATH for DCM"
output_path = f"{main_dir} PATH for png"
convert_folder(input_folder=input_path,output_folder=output_path) #save image into PNG file first, the datas should be in the following Path {main_dir}/ObjectName/.png images. ObjectName = BE001, LC001
image_list = []

for subdir in os.listdir(main_dir):
    subdir_path = os.path.join(main_dir, subdir)
    if os.path.isdir(subdir_path):
        for filename in os.listdir(subdir_path):
            file_path = os.path.join(subdir_path, filename)
            if filename.lower().endswith(('.png')):
                img = imageio.imread(file_path, mode='L')
                image_list.append(img)

images_array = np.stack(image_list, axis=-1)

savemat('ct_images_dataset.mat', {'images': images_array})

print(f"Saved {images_array.shape[2]} images of size {images_array.shape[0]}x{images_array.shape[1]} to ct_images_dataset.mat")
