from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import json
from utils import draw_umich_gaussian
import albumentations as A



def preprocess_image(img, kps=None):
    if kps is None:
        kps = np.array([])

    normalize = A.Compose([
        A.Resize(360, 640, p=1),
        A.Normalize(p=1)
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    norm = normalize(image=img, keypoints=kps)
    inp = np.rollaxis(norm['image'], 2, 0)

    return inp, norm['keypoints']


class courtDataset(Dataset):

    def __init__(self, mode, input_height=720, input_width=1280, scale=2, hp_radius=7, augment=False):
        self.mode = mode
        assert mode in ['train', 'val'], 'incorrect mode'
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = int(input_height / scale)
        self.output_width = int(input_width / scale)
        self.num_joints = 14
        self.hp_radius = hp_radius
        self.scale = scale
        self.augment = augment

        self.path_dataset = './data'
        self.path_images = os.path.join(self.path_dataset, 'images')
        with open(os.path.join(self.path_dataset, 'data_{}.json'.format(mode)), 'r') as f:
            self.data = json.load(f)
        print('mode = {}, len = {}'.format(mode, len(self.data)))

        self.labels = [
            "Top-left corner",
            "Top-right corner",
            "Bottom-left corner",
            "Bottom-right corner",
            "Top-right singles",
            "Bottom-right singles",
            "Top-left singles",
            "Bottom-left singles",
            "Bottom-left service line",
            "Bottom-right service line",
            "Top-left service line",
            "Top-right service line",
            "Center line top",
            "Center line bottom"
        ]

        # Define augmentations
        self.transform = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(gamma_limit=(40, 160), p=0.9),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.CLAHE(p=0.3),
            A.RandomShadow(p=0.3),
            A.RandomSunFlare(p=0.2),
            A.GaussNoise(p=0.3),
            A.MotionBlur(p=0.2),
            A.ImageCompression(quality_range=[60, 80], p=0.3),
            # A.HorizontalFlip(p=0.5, 
                            #  symmetric_keypoints=[(1, 0), (3, 2), (6, 4), (7, 5), (9, 8), (11, 10), (12, 13)]
                            #  ),
            A.Rotate(limit=15, p=0.5),
            # A.RandomCrop(height=540, width=960, p=0.5),
            A.GridDistortion(p=0.3),
            A.RandomScale(p=0.5),
            A.Perspective(scale=(0.05, 0.1), keep_size=True,
                         interpolation=1, p=0.5),
            A.Affine(p=0.3),
            # ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    def __getitem__(self, index):
        img_name = self.data[index]['id'] + '.png'
        raw_kps_data = self.data[index]['kps']

        initial_kps_np = np.full((self.num_joints, 2), -1.0, dtype=np.float32)
        if isinstance(raw_kps_data, list) and len(raw_kps_data) > 0:
            try:
                loaded_kps_array = np.array(raw_kps_data, dtype=np.float32)
                if loaded_kps_array.ndim == 2 and loaded_kps_array.shape[1] == 2:
                    num_loaded = loaded_kps_array.shape[0]
                    num_to_copy = min(num_loaded, self.num_joints)
                    initial_kps_np[:num_to_copy] = loaded_kps_array[:num_to_copy]
            except Exception as e:
                # Optionally log this error if it happens frequently in production
                # print(f"Warning: Error converting raw_kps_data for {img_name}: {e}")
                pass # Keep initial_kps_np as placeholders
        
        img_cv = cv2.imread(os.path.join(self.path_images, img_name))
        
        if img_cv is None:
            print(f"ERROR: Could not read image {img_name}. Returning placeholders.")
            dummy_img = np.zeros((3, self.output_height, self.output_width), dtype=np.float32)
            dummy_hm_hp = np.zeros((self.num_joints, self.output_height, self.output_width), dtype=np.float32)
            dummy_kps = np.full((self.num_joints, 2), -1.0, dtype=np.float32)
            return dummy_img, dummy_hm_hp, dummy_kps, img_name[:-4]

        kps_for_transform = initial_kps_np 
        img_for_preprocess = img_cv

        if self.augment:
            augmented = self.transform(image=img_cv, keypoints=kps_for_transform)
            img_for_preprocess = augmented['image']
            kps_for_transform = augmented['keypoints']
        
        processed_img_np, processed_kps_output = preprocess_image(img_for_preprocess, kps_for_transform)
        
        final_kps_np = np.full((self.num_joints, 2), -1.0, dtype=np.float32)
        if isinstance(processed_kps_output, (list, np.ndarray)) and len(processed_kps_output) > 0:
            try:
                if isinstance(processed_kps_output, list):
                    temp_kps_array = np.array(processed_kps_output, dtype=np.float32)
                else: 
                    temp_kps_array = processed_kps_output.astype(np.float32)

                if temp_kps_array.ndim == 2 and temp_kps_array.shape[1] == 2:
                    num_processed = temp_kps_array.shape[0]
                    num_to_copy = min(num_processed, self.num_joints)
                    if num_to_copy > 0:
                        final_kps_np[:num_to_copy] = temp_kps_array[:num_to_copy]
            except Exception as e:
                # Optionally log this error
                # print(f"Warning: Error processing keypoints for {img_name} after augmentations/preprocess: {e}")
                pass # Keep final_kps_np as placeholders
        
        hm_hp = np.zeros((self.num_joints, self.output_height, self.output_width), dtype=np.float32)
        draw_gaussian = draw_umich_gaussian

        for i in range(self.num_joints):
            x_coord, y_coord = final_kps_np[i][0], final_kps_np[i][1]
            if not (x_coord == -1.0 and y_coord == -1.0):
                if 0 <= x_coord < self.output_width and 0 <= y_coord < self.output_height:
                    x_pt_int = np.rint(x_coord).astype(int)
                    y_pt_int = np.rint(y_coord).astype(int)
                    if 0 <= x_pt_int < self.output_width and 0 <= y_pt_int < self.output_height:
                        draw_gaussian(hm_hp[i], (x_pt_int, y_pt_int), self.hp_radius)

        return processed_img_np, hm_hp, final_kps_np, img_name[:-4]

    def __len__(self):
        return len(self.data)
