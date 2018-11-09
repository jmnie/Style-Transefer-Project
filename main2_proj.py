import time
import random
import os
import mxnet as mx
import numpy as np
np.set_printoptions(precision=2)
from PIL import Image

from mxnet import autograd, gluon
from mxnet.gluon import nn, Block, HybridBlock, Parameter, ParameterDict
import mxnet.ndarray as F
from mxnet.gluon.model_zoo import vision

import net
import utils
from option import Options, program_args
import data
from network import shufflenet_v1

def train(args):
    np.random.seed(args.seed)
    if args.cuda:
        ctx = mx.gpu(0)
    else:
        ctx = mx.cpu(0)
    # dataloader
    transform = utils.Compose([utils.Scale(args.image_size),
                               utils.CenterCrop(args.image_size),
                               utils.ToTensor(ctx),
                               ])
    train_dataset = data.ImageFolder(args.dataset, transform)
    train_loader = gluon.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                         last_batch='discard')
    style_loader = utils.StyleLoader(args.style_folder, args.style_size, ctx=ctx)
    print('len(style_loader):',style_loader.size())

    # models
    #vgg = net.Vgg16()
    #utils.init_vgg_params(vgg, 'models', ctx=ctx)
    
    style_model = net.Net(ngf=args.ngf)
    style_model.initialize(init=mx.initializer.MSRAPrelu(), ctx=ctx)
    if args.resume is not None:
        print('Resuming, initializing using weight from {}.'.format(args.resume))
        style_model.load_params(args.resume, ctx=ctx)
    print('style_model:',style_model)
    # optimizer and loss
    trainer = gluon.Trainer(style_model.collect_params(), 'adam',
                            {'learning_rate': args.lr})
    mse_loss = gluon.loss.L2Loss()

    for e in range(args.epochs):
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            # prepare data
            style_image = style_loader.get(batch_id)
            style_v = utils.subtract_imagenet_mean_preprocess_batch(style_image.copy())
            style_image = utils.preprocess_batch(style_image)

            features_style = vgg(style_v)
            gram_style = [net.gram_matrix(y) for y in features_style]

            xc = utils.subtract_imagenet_mean_preprocess_batch(x.copy())
            f_xc_c = vgg(xc)[1]
            with autograd.record():
                style_model.set_target(style_image)
                y = style_model(x)

                y = utils.subtract_imagenet_mean_batch(y)
                features_y = vgg(y)

                content_loss = 2 * args.content_weight * mse_loss(features_y[1], f_xc_c)

                style_loss = 0.
                for m in range(len(features_y)):
                    gram_y = net.gram_matrix(features_y[m])
                    _, C, _ = gram_style[m].shape
                    gram_s = F.expand_dims(gram_style[m], 0).broadcast_to((args.batch_size, 1, C, C))
                    style_loss = style_loss + 2 * args.style_weight * \
                        mse_loss(gram_y, gram_s[:n_batch, :, :])

                total_loss = content_loss + style_loss
                total_loss.backward()

            trainer.step(args.batch_size)
            mx.nd.waitall()

            agg_content_loss += content_loss[0]
            agg_style_loss += style_loss[0]

            if (batch_id + 1) % args.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.3f}\tstyle: {:.3f}\ttotal: {:.3f}".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                                agg_content_loss.asnumpy()[0] / (batch_id + 1),
                                agg_style_loss.asnumpy()[0] / (batch_id + 1),
                                (agg_content_loss + agg_style_loss).asnumpy()[0] / (batch_id + 1)
                )
                print(mesg)


            if (batch_id + 1) % (4 * args.log_interval) == 0:
                # save model
                save_model_filename = "Epoch_" + str(e) + "iters_" + \
                    str(count) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
                    args.content_weight) + "_" + str(args.style_weight) + ".params"
                save_model_path = os.path.join(args.save_model_dir, save_model_filename)
                style_model.save_params(save_model_path)
                print("\nCheckpoint, trained model saved at", save_model_path)

    # save model
    save_model_filename = "Final_epoch_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
        args.content_weight) + "_" + str(args.style_weight) + ".params"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    style_model.save_params(save_model_path)
    print("\nDone, trained model saved at", save_model_path)


def evaluate(args):
    if args.cuda:
        ctx = mx.gpu(0)
    else:
        ctx = mx.cpu(0)
    # images
    content_image = utils.tensor_load_rgbimage(args.content_image,ctx, size=args.content_size, keep_asp=True)
    style_image = utils.tensor_load_rgbimage(args.style_image, ctx, size=args.style_size)
    style_image = utils.preprocess_batch(style_image)
    # model
    style_model = net.Net(ngf=args.ngf)
    style_model.load_params(args.model, ctx=ctx)
    # forward
    style_model.set_target(style_image)
    output = style_model(content_image)
    #utils.tensor_save_bgrimage(output[0], args.output_image, args.cuda)
    utils.tensor_save_rgbimage(output[0], args.output_image, args.cuda)


def optimize(args):
    """    Gatys et al. CVPR 2017
    ref: Image Style Transfer Using Convolutional Neural Networks
    """
    if args.cuda:
        ctx = mx.gpu(0)
    else:
        ctx = mx.cpu(0)
    # load the content and style target
    content_image = utils.tensor_load_rgbimage(args.content_image,ctx, size=args.content_size, keep_asp=True)
    content_image = utils.subtract_imagenet_mean_preprocess_batch(content_image)
    style_image = utils.tensor_load_rgbimage(args.style_image, ctx, size=args.style_size)
    style_image = utils.subtract_imagenet_mean_preprocess_batch(style_image)
    # load the pre-trained vgg-16 and extract features
    vgg = net.Vgg16()
    utils.init_vgg_params(vgg, 'models', ctx=ctx)
    # content feature
    f_xc_c = vgg(content_image)[1]
    # style feature
    features_style = vgg(style_image)
    gram_style = [net.gram_matrix(y) for y in features_style]
    # output
    output = Parameter('output', shape=content_image.shape)
    output.initialize(ctx=ctx)
    output.set_data(content_image)
    # optimizer
    trainer = gluon.Trainer([output], 'adam',
                            {'learning_rate': args.lr})
    mse_loss = gluon.loss.L2Loss()

    # optimizing the images
    for e in range(args.iters):
        utils.imagenet_clamp_batch(output.data(), 0, 255)
        # fix BN for pre-trained vgg
        with autograd.record():
            features_y = vgg(output.data())
            content_loss = 2 * args.content_weight * mse_loss(features_y[1], f_xc_c)
            style_loss = 0.
            for m in range(len(features_y)):
                gram_y = net.gram_matrix(features_y[m])
                gram_s = gram_style[m]
                style_loss = style_loss + 2 * args.style_weight * mse_loss(gram_y, gram_s)
            total_loss = content_loss + style_loss
            total_loss.backward()

        trainer.step(1)
        if (e + 1) % args.log_interval == 0:
            print('loss:{:.2f}'.format(total_loss.asnumpy()[0]))

    # save the image
    output = utils.add_imagenet_mean_batch(output.data())
    utils.tensor_save_bgrimage(output[0], args.output_image, args.cuda)

def main():
    # figure out the experiments type
    args = Options().parse()

    if args.subcommand is None:
        raise ValueError("ERROR: specify the experiment type")

    if args.subcommand == "train":
        # Training the model
        train(args)

    elif args.subcommand == 'eval':
        # Test the pre-trained model
        evaluate(args)

    elif args.subcommand == 'optim':
        # Gatys et al. using optimization-based approach
        optimize(args)

    else:
        raise ValueError('Unknow experiment type')

def evaluate_test():
    content_image = 'images/content/xjtlu.jpg'
    style_image = 'images/styles/starry_night.jpg'
    output_image = 'test_output.jpg'
    cuda = 0
    eval_args = program_args(content_image,content_image,style_image,128,128,0)

    if cuda == 0:
        ctx = mx.cpu(0)
    else:
        ctx = mx.gpu(0)

    content_image = utils.tensor_load_rgbimage(eval_args.contentImage, ctx, size=eval_args.size, keep_asp=True)
    style_image = utils.tensor_load_rgbimage(eval_args.styleImage, ctx, size=eval_args.size)
    style_image = utils.preprocess_batch(style_image)
    # model
    style_model = net.Net(ngf=eval_args.ngf)
    style_model.load_parameters(eval_args.model, ctx=ctx)
    # forward
    style_model.set_target(style_image)
    output = style_model(content_image)
    print(type(output),output.asnumpy().shape)
    
    np_array = output.asnumpy()
    img_array = np.reshape(np_array,(np_array.shape[2],np_array.shape[3],np_array.shape[1]))
    print(img_array.shape)
    img = Image.fromarray(np.uint8(img_array))  
    img.show()

def test_output():
    style_model = net.Net()
    #print(style_model)
    ctx = mx.cpu(0)
    X = mx.ndarray.random.normal(shape=(20,3,224,224),ctx=ctx)
    vgg = net.Vgg16()
    print(vgg)
    vgg.initialize()

    
    output = vgg.forward(X)
    for item in output:
        print(item.shape)

    # styleNet = net.Net(input_nc=3, output_nc=3, ngf=64,n_blocks=1)
    # print(styleNet)
    # styleNet.initialize()
    # output_2 = styleNet.forward(X)
    # print(output_2)

if __name__ == "__main__":
   #main()
   #evaluate_test()
   test_output()
