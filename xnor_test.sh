python cifar.py -a xnorresnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/xnorresnet-test20 --gpu 1

python cifar.py -a xnorresnet --depth 26 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/xnorresnet-test28 --gpu 1

python cifar.py -a xnorresnet --depth 32 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/xnorresnet-test32 --gpu 2

python cifar.py -a xnorresnet --depth 38 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/xnorresnet-test38 --gpu 2

python cifar.py -a xnorresnet --depth 44 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/xnorresnet-test44 --gpu 3

python cifar.py -a xnorresnet --depth 50 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/xnorresnet-test50 --gpu 0
