import os
import cv2
import numpy as np
import mxnet as mx
import net
from option import Options, program_args
import utils
from PIL import Image
import mxnet.ndarray as F

class camero_args:
    def __init__(self,styleImage,size=50,ngf=128,cuda=0):
        self.ngf = ngf #number of generator filter channels
        self.size = size
        self.styleImage = styleImage
        self.cuda = cuda
        self.model = 'models/models/21styles.params'

def load_image(content_image,ctx,size=None,scale=None, keep_asp=False):
    img = Image.fromarray(np.uint8(content_image))

    if size is not None:
        if keep_asp:
            size2 = int(size * 1.0 / img.size[0] * img.size[1])
            img = img.resize((size, size2), Image.ANTIALIAS)
        else:
            img = img.resize((size, size), Image.ANTIALIAS)

    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    
    img = np.array(img).transpose(2, 0, 1).astype(float)
    img = F.expand_dims(mx.nd.array(img, ctx=ctx), 0)

    ## return this content Image
    return img


def run_demo(eval_args):

    ## load parameters
        
    #content_image = 'images/content/xjtlu.jpg'
    #style_image = 'images/styles/starry_night.jpg'
    #eval_args = program_args(content_image,content_image,style_image,128,128,0)
    #eval_args = camero_args(style_image)

    if eval_args.cuda == 0:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu()

    ## Change the content and style image using Style Loader
    #content_image = utils.tensor_load_rgbimage(eval_args.contentImage, ctx, size=eval_args.size, keep_asp=True)
    style_image = utils.tensor_load_rgbimage(eval_args.styleImage, ctx, size=eval_args.size)
    style_image = utils.preprocess_batch(style_image)

    style_model = net.Net(ngf=eval_args.ngf)
    style_model.load_parameters(eval_args.model, ctx=ctx)
    style_model.set_target(style_image)

    cam = cv2.VideoCapture(0)

    while True:
       ## read frame
       ret, frame = cam.read()
       # read content image (cimg)
       #cimg = img.copy() 
       #img = np.array(img).transpose(2, 0, 1)
       content_img = load_image(frame,ctx,eval_args.size)
       
       output = style_model(content_img)
       tensor = output[0]
       #(b, g, r) = F.split(tensor, num_outputs=3, axis=0)
       #tensor = F.concat(r, g, b, dim=0)
       img = F.clip(tensor, 0, 255).asnumpy()
       img = img.transpose(1, 2, 0).astype('uint8')
       img = Image.fromarray(img)
       image = np.array(img.resize((frame.shape[1], frame.shape[0]), Image.ANTIALIAS))
       #print(frame.shape,image.shape)
       numpy_horizontal = np.hstack((frame, image))
       #cv2.imshow("Content Window",frame)
       #cv2.imshow("Style Window",grey)

       cv2.imshow("Test Window Shape",numpy_horizontal)
       if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cam.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    ## change the size here to increase the quality
    size = 200

    # change the style image
    style_image = 'images/styles/starry_night.jpg'

    # if you have gpu, set cuda = 1
    eval_args = camero_args(style_image,size=size,cuda=0)
    run_demo(eval_args)
