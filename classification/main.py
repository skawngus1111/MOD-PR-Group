import argparse

from pytorch_lightning import Trainer

from utils import get_model
from dataset import DataModule

def ArgParser() :
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='mobilenet-v1', help='net type',
                        choices=['mobilenet-v1', 'googlenet', 'inception-v3',
                                 'nin'])
    parser.add_argument('--data_type', type=str, default='cifar10', help='data type', choices=['cifar10', 'cifar100', 'MNIST'])
    parser.add_argument('--input_channel', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--data_path', type=str, default='/media/jhnam19960514/68334fe0-2b83-45d6-98e3-76904bf08127/home/namjuhyeon/Desktop/LAB/common material/Dataset Collection/CIFAR/')
    parser.add_argument('--num_workers', type=int, default=8, help='worker for data loader')
    parser.add_argument('--parallel', action='store_true', default=True)

    # Train Parameter
    parser.add_argument('--epochs', type=int, default=10, help='max epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')

    parser.add_argument('--log-dir', default='./runs/logs/', help='Directory for saving checkpoint models')
    parser.add_argument('--step', default=10, help='print step')

    parser.add_argument('--gpus', type=int, default=1)
    args = parser.parse_args()

    return args

def main(args) :
    print('hello world!')
    data_module = DataModule()
    model = get_model(args.model_name, args.input_channel, args.num_classes)
    trainer = Trainer(gpus=args.gpus)
    trainer.fit(model, data_module)

if __name__=='__main__' :
    args = ArgParser()
    main(args)