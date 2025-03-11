import argparse

parser = argparse.ArgumentParser(description='args_setting')
# Train args
parser.add_argument('--DEVICE', type=str, default='cuda:0')
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--save_epoch', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=64)

parser.add_argument('--task', type=str, default='PET-MRI')
parser.add_argument('--model', type=str, default='UNet_full')

args = parser.parse_args()
