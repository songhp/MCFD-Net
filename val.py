import argparse
import warnings
import model.mcfd as MCFD
from loss import *
from data_processor import *
from trainer import *

warnings.filterwarnings("ignore")


def main():
    global args, model
    args = parser.parse_args()
    print(args)

    if args.model == 'mcfd':
        model = MCFD.MCFD(sensing_rate=args.sensing_rate)

    model = model.cuda()

    dic = './trained_model/' + str(args.model) + '_' + str(args.sensing_rate) + '.pth'

    model.load_state_dict(
        torch.load(dic, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    criterion = loss_fn

    _, bsds, set5, set14, compare = data_loader(args, is_test=True)
    model.eval()

    psnr1, ssim1 = valid_bsds(bsds, model, criterion, args.model, True)
    print("----BSDS----PSNR: %.2f----SSIM: %.4f" % (psnr1, ssim1))
    psnr2, ssim2 = valid_set(set5, model, criterion, args.model, True)
    print("----Set5----PSNR: %.2f----SSIM: %.4f" % (psnr2, ssim2))
    psnr3, ssim3 = valid_set(set14, model, criterion, args.model, True)
    print("----Set14----PSNR: %.2f----SSIM: %.4f" % (psnr3, ssim3))


if __name__ == '__main__':
    torch.cuda.set_device(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mcfd',
                        help='choose model to train')

    parser.add_argument('--sensing-rate', type=float, default=0.5,
                        choices=[0.50000, 0.25000, 0.12500, 0.06250, 0.03125, 0.015625],
                        help='set sensing rate')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=4, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--block-size', default=32, type=int,
                        metavar='N', help='block size (default: 32)')
    parser.add_argument('--image-size', default=96, type=int,
                        metavar='N', help='image size used for training (default: 96)')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--save-dir', dest='save_dir',
                        help='The directory used to save the trained models',
                        default='trained_model', type=str)

    main()
