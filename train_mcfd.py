import argparse
import random
import warnings
import model.mcfd as mcfd
from loss import *
import torch.optim as optim
from data_processor import *
from trainer import *
from torch import nn

warnings.filterwarnings("ignore")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    global args, model
    setup_seed(random.randint(1, 100))
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        torch.backends.cudnn.benchmark = True

    if args.model == 'mcfdnet':
        model = mcfd.MCFD(sensing_rate=args.sensing_rate)

    if MULTI_GPU:
        model = nn.DataParallel(model, device_ids=device_ids)
    model.to(device)

    criterion = loss_fn
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [60, 90, 120, 150, 180], gamma=0.25, last_epoch=-1)
    if MULTI_GPU:
        optimizer = nn.DataParallel(optimizer, device_ids=device_ids)
        scheduler = nn.DataParallel(scheduler, device_ids=device_ids)
        optimizer = optimizer.module
        scheduler = scheduler.module

    train_loader, test_loader_bsds, test_loader_set5, test_loader_set14 = data_loader(args)

    print('\nModel: %s\n'
          'Sensing Rate: %.6f\n'
          'Epoch: %d\n'
          'Initial LR: %f\n'
          'Dataset Nums: %.0f\n'
          'Model Parameters: %.0f\n'
          % (args.model, args.sensing_rate, args.epochs, args.lr, len(train_loader.dataset), count_parameters(model)))

    print(args.model + '_' + str(args.sensing_rate) + ' Start Training--------------------------------------------')
    losss, psnrs, ssims = [], [], []
    start_time = time.time()
    for epoch in range(args.epochs):
        print('\ncurrent lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        loss = train(train_loader, model, criterion, optimizer, epoch)
        print('current loss:', loss)
        scheduler.step()

        losss.append(loss)

        if epoch % 1 == 0:
            psnr1, ssim1 = valid_bsds(test_loader_bsds, model, criterion, args.model, False)
            print("BSDS--PSNR: %.2f--SSIM: %.4f" % (psnr1, ssim1))
            psnr2, ssim2 = valid_set(test_loader_set5, model, criterion, args.model, False)
            print("Set5--PSNR: %.2f--SSIM: %.4f" % (psnr2, ssim2))
            psnr3, ssim3 = valid_set(test_loader_set14, model, criterion, args.model, False)
            print("Set14--PSNR: %.2f--SSIM: %.4f" % (psnr3, ssim3))
            psnrs.append((psnr1 + psnr2 + psnr3) / 3.00)
            ssims.append((ssim1.item() + ssim2.item() + ssim3.item()) / 3.00)

        end_time = time.time()
        elapsed_time = end_time - start_time
        start_time = end_time
        print("Running time: {:.2f} seconds".format(elapsed_time))

        if epoch > 40 and epoch % 10 == 0:
            if MULTI_GPU:
                torch.save(model.module.state_dict(),
                           './trained_model/' + str(epoch) + '_' + str(args.model) + '_' + str(
                               args.sensing_rate) + '.pth')
            else:
                torch.save(model.state_dict(), './trained_model/' + str(epoch) + '_' + str(args.model) + '_' + str(
                    args.sensing_rate) + '.pth')

    if MULTI_GPU:
        torch.save(model.module.state_dict(),
                   './trained_model/' + str(args.model) + '_' + str(args.sensing_rate) + '.pth')
    else:
        torch.save(model.state_dict(), './trained_model/' + str(args.model) + '_' + str(args.sensing_rate) + '.pth')

    show_loss(args.epochs, losss)
    print(psnrs, ssims)
    print(args.model + '_' + str(args.sensing_rate) + ' Trained finished------------------------------------------')


def show_loss(epochs, losss):
    epochs = list(range(1, epochs + 1))
    plt.plot(epochs, losss, 'bo', label='Training loss')
    plt.title(args.model + '_' + str(args.sensing_rate) + ' Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def show_psnrs_ssims(epochs, psnrs, ssims):
    epochs = list(range(1, epochs + 1))
    plt.plot(epochs, psnrs, 'bo', label='PSNRs')
    plt.plot(epochs, ssims, 'r', label='SSIMs')
    plt.title(args.model + '_' + str(args.sensing_rate) + ' PSNR and SSIM variations')
    plt.xlabel('Epochs')
    plt.ylabel('Values')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    device_nums = len(device_ids)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mcfdnet',
                        choices=['mcfdnet', 'mrccsnet', 'rkccsnet', 'csnet'],
                        help='choose model to train')
    parser.add_argument('--sensing-rate', type=float, default=0.5,
                        choices=[0.50000, 0.25000, 0.12500, 0.06250, 0.03125, 0.015625],
                        help='set sensing rate')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=6 * device_nums, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--block-size', default=32, type=int,
                        metavar='N', help='block size (default: 32)')
    parser.add_argument('--image-size', default=96, type=int,
                        metavar='N', help='image size used for training (default: 96)')
    parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--save-dir', dest='save_dir',
                        help='The directory used to save the trained model',
                        default='save_temp', type=str)

    # main()
    args = parser.parse_args()
    models = ['mcfdnet']
    sensing_rates = [0.50000, 0.25000, 0.12500]
    # sensing_rates = [0.06250, 0.03125, 0.015625]
    for i in range(len(models)):
        for j in range(len(sensing_rates)):
            args.sensing_rate = sensing_rates[j]
            args.model = models[i]
            if MULTI_GPU:
                print(device_nums, 'GPU Train cuda:', device_ids)
            else:
                print('1 Single GPU Train')
            main()

# nohup python train_mcfd.py > MCFD_50000.txt &
# nohup python train_mcfd.py > MCFD_25000.txt &
# nohup python train_mcfd.py > MCFD_12500.txt &
# nohup python train_mcfd.py > MCFD_06250.txt &
# nohup python train_mcfd.py > MCFD_03125.txt &
# nohup python train_mcfd.py > MCFD_015625.txt &
