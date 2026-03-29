from options2 import TrainOptions
from seg_model_NEW import UNet
from seg_dataset import Get_traindataloader
import torch
import torch.nn as nn
from loss import SSIM
from tqdm import tqdm
import os


def train(args,dataset_name,save_dir):
    train_dataloader = Get_traindataloader(args,dataset_name)
    #test_dataloader = Get_testdataloader(args,dataset_name)
    ssim_loss = SSIM()
    #wavelet_loss=Wavelet(wave=args.wave)
    mse_loss=nn.MSELoss()
    #model=UNet(wave=args.wave).cuda()
    model = UNet().cuda()
    #msgm = MSGMSLoss()

    optimizer =torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00001)
    for epoch in tqdm(range(args.epochs)):
        model.train()
        for g in optimizer.param_groups:
            g['lr'] = 1e-5
        total_batches = len(train_dataloader)
        for idx, (normal_image, anomaly_image) in tqdm(enumerate(train_dataloader), total=total_batches,
                                                       desc=f"Epoch {epoch + 1}"):
            normal_image = normal_image.cuda()
            anomaly_image = anomaly_image.cuda()
            rec_img = model(anomaly_image)
            mse = mse_loss(normal_image, rec_img).cuda()
            ssim = ssim_loss(normal_image, rec_img).cuda()
            loss = mse + ssim
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()

            # ✅ 打印训练进度
            print(f"Epoch [{epoch + 1}/{args.epochs}], Batch [{idx + 1}/{total_batches}], Loss: {loss.item():.6f}")

        if epoch % args.save_interval == 0:
            with open(os.path.join(save_dir, "loss.log"), "a") as f:
                f.write(f"[Epoch {epoch + 1}] Loss: {loss.item():.6f}\n")
            torch.save(model.state_dict(), os.path.join(save_dir, f"epoch_{epoch + 1}.pth"))


if __name__ == '__main__':
    args = TrainOptions().parse()

    # 1) 这里是“总根目录”
    args.data_root = r"image_all"

    # 2) 这里是“实验名文件夹”（里面有 CAM1~CAM4）
    args.exp_name = "image_data_01_24"

    all_types = ["CAM1", "CAM2", "CAM3", "CAM4"]  # 注意和文件夹名一致
    for dataset_name in all_types:
        save_dir = rf".\train-result\{args.exp_name}\{dataset_name}"
        os.makedirs(save_dir, exist_ok=True)

        print(f"train-{dataset_name}")
        train(args, dataset_name, save_dir)



