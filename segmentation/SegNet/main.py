"""
Trainer.py -> 1. 각 모델에 동일한 가중치로 초기화되도록 초기화 함수 만들기
              2. epoch마다 생성된 영상을 저장하기
"""
import argparse

from trainer import Trainer

def main(args) :
    print("hello world!")

    trainer = Trainer(args.data_path, args.data_type, args.parallel,
                      args.batch_size, args.lr, args.momentum, args.weight_decay, args.num_workers, args.num_classes,
                      args.log_dir, args.step)

    model, history = trainer.fit(args.epochs)

    trainer.save_history(history)
    trainer.save_model(model)

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='COCO', help='data type', choices=['COCO'])
    parser.add_argument('--data_path', type=str, default='/media/jhnam19960514/68334fe0-2b83-45d6-98e3-76904bf08127/home/namjuhyeon/Desktop/LAB/common material/Dataset Collection/COCO')
    parser.add_argument('--num_workers', type=int, default=8, help='worker for data loader')
    parser.add_argument('--parallel', action='store_true', default=True)

    # Train Parameter
    parser.add_argument('--epochs', type=int, default=10, help='max epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--num_classes', type=int, default=21, help='number of class')

    parser.add_argument('--log-dir', default='./runs/logs/', help='Directory for saving checkpoint models')
    parser.add_argument('--step', default=10, help='print step')
    args = parser.parse_args()

    main(args)