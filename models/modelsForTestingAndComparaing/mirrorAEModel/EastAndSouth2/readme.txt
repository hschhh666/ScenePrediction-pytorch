EastAndSouth2

loss使用的是E-E   S-S   S-E  E-S，数据是平均数据，结果比使用E-E  S-S z-z 的loss要好。
训练过程中观察了z和z的mse，发现两个隐变量的差有由大变小再变大再变小的过程，并且收敛的时候z-z的mse并没有接近于0。