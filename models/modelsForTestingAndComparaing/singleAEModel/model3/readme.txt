model3

这个模型在model2的基础上，增加了全连接层，把隐变量维度降到了20。但是我并没有跑完整的一次实验，因为设置的全连接层参数太多了，某一层输入9*64*64=36864，输出9*32*32=9216，这个参数的量级实在太恐怖，跑了一会发现训练实在太慢了，就停掉了。然后，才有了model4，就是缩减卷积层最后的通道数，同时降低全连接层的维度以减少参数。