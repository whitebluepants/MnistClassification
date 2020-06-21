import d2lzh as d2l
import mxnet as mx
from mxnet import nd, autograd, gluon, init
from mxnet.gluon import data as gdata, loss as gloss, nn
import time
import random
import matplotlib.pyplot as plt

# 数据增强
flig_aug = gdata.vision.transforms.Compose(
    [
        gdata.vision.transforms.Resize(32),
        gdata.vision.transforms.RandomResizedCrop(28, scale=(0.64, 1.0), ratio=(1.0, 1.0)),
        gdata.vision.transforms.RandomFlipLeftRight(),
        gdata.vision.transforms.ToTensor()
    ]
)

no_aug = gdata.vision.transforms.Compose(
    [
        gdata.vision.transforms.ToTensor()
    ]
)


# 模型
class Residual(nn.HybridBlock):
    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1,
                               strides=strides)
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(num_channels, kernel_size=1,
                                   strides=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    def hybrid_forward(self, F, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)


def get_resnet18(num_classes):
    net = nn.HybridSequential()
    net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
            nn.BatchNorm(), nn.Activation('relu'))

    def resnet_block(num_channels, num_residuals, first_block=False):
        blk = nn.HybridSequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                blk.add(Residual(num_channels))
        return blk

    net.add(resnet_block(64, 2, first_block=True),
            resnet_block(128, 2),
            resnet_block(256, 2),
            resnet_block(512, 2))
    net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
    return net


def evaluate_accuracy(data_iter, net, ctx):
    acc_sum, n = nd.array([0], ctx=ctx), 0
    for x, y in data_iter:
        x, y = x.as_in_context(ctx), y.as_in_context(ctx)
        acc_sum += (net(x).argmax(axis=1) == y.astype('float32')).sum()
        n += y.size
    return acc_sum.asscalar() / n

def train(load_params=False):
    batch_size = 128
    lr = 0.01
    num_epochs = 1  # 演示用
    ctx = mx.gpu()
    net = get_resnet18(10)
    net.initialize(init.Normal(sigma=0.01), ctx=ctx)

    if load_params is False:
        train_iter = gdata.DataLoader(gdata.vision.MNIST(train=True).transform_first(flig_aug),
                                      shuffle=True, batch_size=batch_size)
        loss = gloss.SoftmaxCrossEntropyLoss()
        trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
        Train(train_iter, net, loss, trainer, ctx, num_epochs, batch_size)
    else:
        filename = 'Mnist.params'
        net.load_parameters(filename, ctx=ctx)

    return net


def Train(train_iter, net, loss, trainer, ctx, num_epochs, batch_size):
    print('Training on', ctx)
    for epoch in range(num_epochs):
        n = 0
        train_loss, train_acc, test_acc = 0.0, 0.0, 0.0
        start = time.time()
        for x, y in train_iter:
            x, y = x.as_in_context(ctx), y.as_in_context(ctx)
            with autograd.record():
                y_hat = net(x)
                l = loss(y_hat, y)
            l.backward()
            trainer.step(batch_size)
            train_loss += l.sum().asscalar()
            train_acc += (y_hat.argmax(axis=1) == y.astype('float32')).sum().asscalar()
            n += y.size

        print('epoch %d, loss %.4f, train acc %.3f,  '
              'time %.1f sec'
              % (epoch + 1, train_loss / n, train_acc / n,
                 time.time() - start))


def test(net):
    ctx = mx.gpu()
    test_iter = gdata.DataLoader(gdata.vision.MNIST(train=False).transform_first(no_aug),
                                 shuffle=False, batch_size=128)
    test_acc = evaluate_accuracy(test_iter, net, ctx)
    print('test acc %.3f' % test_acc)

    for show_x, show_y in test_iter:
        show_x, show_y = show_x.as_in_context(ctx), show_y.as_in_context(ctx)
        break

    text_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    size = 10
    valid_index = random.sample(range(0, 128), size)
    true_y = [text_labels[int(i)] for i in show_y.asnumpy()[valid_index]]

    pred_y = [text_labels[int(i)] for i in net(show_x).argmax(axis=1).asnumpy()[valid_index]]

    titles = ['True: ' + true + '\n' + 'Pred: ' + pred for true, pred in zip(true_y, pred_y)]
    image = []
    for i in valid_index:
        image.append(show_x[i])
    show_mnist(image, titles[0:size])
    plt.show()


def Predict(net, X):
    y_hat = net(X)

    return y_hat.argmax(axis=1).asnumpy()[0]


def show_mnist(images, labels):
    _, figs = plt.subplots(1, len(images), figsize=(15, 6))
    i = 0
    for f, lbl in zip(figs, labels):
        f.imshow(images[i].reshape(28, 28).asnumpy(), cmap='gray')
        i += 1
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
