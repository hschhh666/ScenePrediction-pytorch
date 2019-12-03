EastAndSouth3

换了一下数据集，使用单次仿真数据训练，而非平均仿真数据。定义两种loss:
loss1 = E-E + S-S + z-z
loss2 = E-E + S-S
loss1和loss2交替训练，即epoch为偶数时用loss1，为奇数时用loss2。
使用loss1时，E和S对应的M相同，即是使用相同的M生成的数据。

数据集大小：3x100

训练结果是 预测的图片是整个场景的平均，不行