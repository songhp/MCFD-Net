from utils import *
import matplotlib.pyplot as plt
from torch import nn
import time
import cv2
from torchvision.transforms import ToPILImage
import torchvision
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()


def train(train_loader, model, criterion, optimizer, epoch):
    print('Epoch: %d' % (epoch + 1))
    model.train()
    sum_loss = 0
    for inputs, _ in train_loader:
        inputs = rgb_to_ycbcr(inputs.to(device))[:, 0, :, :].unsqueeze(1) / 255.
        optimizer.zero_grad()
        if use_half_precision_flag:
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs[0], inputs) + criterion(outputs[1], inputs)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs[0], inputs) + criterion(outputs[1], inputs)
            loss.backward()
            optimizer.step()

        sum_loss += loss.item()

    return sum_loss


def plt_show_images(inputs_np, outputs_np, dataset_name, model_name):
    plt.subplot(1, 2, 1)
    plt.imshow(inputs_np, cmap='gray')
    plt.title(dataset_name + model_name + 'Input ')
    plt.subplot(1, 2, 2)
    plt.imshow(outputs_np, cmap='gray')
    plt.title(dataset_name + model_name + 'Output')
    plt.show()


def valid_bsds(valid_loader, model_, criterion, model_name='mcfd', is_test=False):
    sum_psnr = 0
    sum_ssim = 0
    _ssim = SSIM().to(test_device)
    model = model_.eval()
    model.to(test_device)
    with torch.no_grad():
        for iters, (inputs, _) in enumerate(valid_loader):
            inputs = rgb_to_ycbcr(inputs.to(test_device))[:, 0, :, :].unsqueeze(1) / 255.
            outputs = model(inputs)
            mse = F.mse_loss(outputs[0], inputs)
            psnr = 10 * log10(1 / mse.item())
            sum_psnr += psnr
            sum_ssim += ssim(outputs[0].to(test_device), inputs.to(test_device))

            if is_test:
                inputs_np = inputs.cpu().numpy().squeeze()
                outputs_np = outputs[0].cpu().numpy().squeeze()
                plt_show_images(inputs_np, outputs_np, 'bsds', model_name)
    model.to(device)
    return sum_psnr / len(valid_loader), sum_ssim / len(valid_loader)


def valid_set(valid_loader, model_, criterion, model_name='mcfd', is_test=False):
    sum_psnr = 0
    sum_ssim = 0
    _ssim = SSIM().to(test_device)
    model = model_.eval()
    model.to(test_device)
    with torch.no_grad():
        for iters, (inputs, _) in enumerate(valid_loader):
            inputs = rgb_to_ycbcr(inputs.to(test_device))[:, 0, :, :].unsqueeze(1) / 255.
            outputs = model(inputs)
            mse = F.mse_loss(outputs[0], inputs)
            psnr = 10 * log10(1 / mse.item())
            sum_psnr += psnr
            sum_ssim += ssim(outputs[0].to(test_device), inputs.to(test_device))
            if is_test:
                inputs_np = inputs.cpu().numpy().squeeze()
                outputs_np = outputs[0].cpu().numpy().squeeze()
                plt_show_images(inputs_np, outputs_np, 'set', model_name)
    model.to(device)
    return sum_psnr / len(valid_loader), sum_ssim / len(valid_loader)
