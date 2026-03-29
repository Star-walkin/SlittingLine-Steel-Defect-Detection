import glob
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from options2 import TrainOptions
import matplotlib.pyplot as plt
import cv2
from scar_ano import scar_creat
from stain_anomaly import add_stain
from pathlib import Path
import random

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
class traindataset(Dataset):
    def __init__(self, args, data_dir, img_size, mode="train"):
        self.args = args
        self.img_size = img_size

        # 【修改1】 基础变换：只做尺寸调整、转Tensor、归一化
        # 既然是单通道训练，Normalize 只需要一个 mean 和一个 std
        self.base_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            # transforms.Normalize([0.5], [0.5]),  # 单通道归一化
        ])

        # 【修改2】 增强变换：仅用于输入图像 (可选)
        # 如果你想做增强，只对 Input 做。但在初期调试，建议先注释掉 ColorJitter，求稳。
        self.aug_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            # transforms.ColorJitter(brightness=0.1, contrast=0.1), # 先注释掉，确保稳定
            transforms.ToTensor(),
            # transforms.Normalize([0.5], [0.5]),
        ])

        self.images = sorted(glob.glob(os.path.join(data_dir, mode) + '/good/*.*'))
        # if len(self.images) > 400:  # 稍微多采一点样
        #     self.images = random.sample(self.images, 400)

    def __getitem__(self, idx):
        normal_image_path = self.images[idx]

        # 【修改3】 始终保持单通道 (GRAY)
        gray = cv2.imread(normal_image_path, cv2.IMREAD_GRAYSCALE)
        image_name = os.path.basename(normal_image_path)

        # 先固定“正常图”的副本（关键！）
        normal_np = gray.copy()

        thresh = random.uniform(0, 1)
        if thresh < 1:
            anomaly_np = scar_creat(gray.copy())  # 传 copy，避免污染 gray / normal_np

        else:
            anomaly_np = normal_np.copy()

        normal_pil = Image.fromarray(normal_np)
        anomaly_pil = Image.fromarray(anomaly_np)

        normal_image = self.base_transform(normal_pil)
        anomaly_image = self.aug_transform(anomaly_pil)

        return normal_image, anomaly_image, image_name

    def __len__(self):
        return len(self.images)


def Get_traindataloader(args, dataset_name):
    train_dataloader = DataLoader(traindataset(args,
                                               "%s/%s/%s" % (
                                                   args.data_root, args.exp_name, dataset_name),
                                                args.img_size),
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=1,
                                  pin_memory=True,
                                  persistent_workers=True,
                                  prefetch_factor=2,
                                  drop_last=False)
    return train_dataloader

class testdataset(Dataset):
    """ dataset name."""

    def __init__(self, args, data_dir, img_size,mode="test" ):
        """
        Args:
            root_dir (string): Directory with the MVTec AD dataset.
            defect_name (string): defect to load.
            transform: Transform to apply to data
            mode: "train" loads training samples "test" test samples default "train"
        """
        self.args = args
        self.img_size = img_size


        self.transform_test = transforms.Compose([transforms.Resize((self.img_size, self.img_size)),
                                                  transforms.ToTensor(),
                                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                                  ])

        #a=glob.glob(r'G:\image_data\cam3\test\anomaly/*.*')
        self.images = sorted(glob.glob(data_dir + '/*.*'))
        #print(self.images)
    def __getitem__(self, idx):
        filename = self.images[idx]
        class_name=Path(filename).parts[-2]
        basename = os.path.basename(filename).split(".")[0]
        image = Image.open(filename).convert("RGB")
        img = self.transform_test(image)

        return class_name,basename,img

    def __len__(self):
        return len(self.images)


def Get_testdataloader(args, dataset_name):
    test_dataloader = DataLoader(testdataset(args,
                                             "%s\%s\%s/" % (args.data_root, args.exp_name,dataset_name),
                                             args.img_size,),batch_size=1,
                                             shuffle=False, num_workers=1, drop_last=False)
    return test_dataloader

if __name__ == '__main__':
    args = TrainOptions().parse()
    all_types =  ['CAM1','CAM2','CAM3','CAM4',]
    for dataset_name in all_types:

        train_dataloader = Get_traindataloader(args,dataset_name)
        test_dataloader =Get_testdataloader(args,dataset_name)
        print(len(train_dataloader))
        for idx, (normal_image,anomaly_image,img_name) in enumerate(train_dataloader):
            a=normal_image
            b=anomaly_image
            c=a-b
            print(c)
            input=normal_image.permute(0,2,3,1).numpy()
            GT=anomaly_image.permute(0,2,3,1).numpy()
            for j in range(input.shape[0]):
                    #j=denormalize(i[j])
                print(img_name[j])
                plt.subplot(1, 2, 1),plt.imshow(input[j],cmap="gray")
                plt.subplot(1, 2, 2), plt.imshow(GT[j],cmap="gray")
                plt.show(block=True)