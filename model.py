import collections
import numpy as np
import tensorflow as tf
import tifffile as tiff
import argparse
import glob
import pandas as pd
import cv2
import os

#p_size = 512
EPS = 1e-12

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="path to folder containing images")
parser.add_argument("--mask_dir",help="path to folder containing masks")
parser.add_argument("--save_dir",default="./outputs",help="path to save checkpoints/outputs")
parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")
parser.add_argument("--max_epochs", type=int, default=5,help="number of training epochs")
parser.add_argument("--batch_size", type=int, default=8, help="number of images in batch for training")
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--lr", type=float, default=0.02, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--mode", required=True, choices=["train", "test"])
parser.add_argument("--patch_size",type=int,default=512,help="Patch size of image to train/test")
a = parser.parse_args()
Epochs=a.max_epochs
p_size= a.patch_size
BATCH_SIZE = a.batch_size

def M(image_id):
    # __author__ = amaia
    # https://www.kaggle.com/aamaia/dstl-satellite-imagery-feature-detection/rgb-using-m-bands-example
    filename = os.path.join(a.input_dir, '{}.tiff'.format(image_id))
    img = tiff.imread(filename)
    img = np.rollaxis(img, 0, 3)
    return img

def stretch_n(bands, lower_percent=5, higher_percent=95):
    out = np.zeros_like(bands)
    n = bands.shape[2]
    for i in range(n):
        a = 0  # np.min(band)
        b = 255  # np.max(band)
        c = np.percentile(bands[:, :, i], lower_percent)
        d = np.percentile(bands[:, :, i], higher_percent)
        t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[:, :, i] = t
    return out/255.0

def mask_read(image_id):
    filename = os.path.join(a.mask_dir, 'mask_{}.png'.format(image_id))
    mask_in=cv2.imread(filename,0)
    return mask_in/255

def examples_load(Ids):
    img_bat=[]
    mask_bat=[]
    for i in range(BATCH_SIZE):
        img_name=Ids[0]
        print ("Reading Image: " + img_name)
        Ids=np.roll(Ids,1)
        img=M(img_name)
        img=stretch_n(img)
        mask=mask_read(img_name)
        mask=np.expand_dims(mask,-1)
        x_max=img.shape[0]
        y_max=img.shape[1]
        x_c=np.random.randint(0,x_max-p_size)
        y_c=np.random.randint(0,y_max-p_size)
        img_pat=img[x_c:x_c+p_size,y_c:y_c+p_size,:]
        mask_pat=mask[x_c:x_c+p_size,y_c:y_c+p_size,:]
        img_bat.append(img_pat)
        mask_bat.append(mask_pat)
    return Ids,img_bat,mask_bat

def conv(batch_input, out_channels, stride):
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [4, 4, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")
        return conv


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(input):
    with tf.variable_scope("batchnorm"):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)

        channels = input.get_shape()[3]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized


def deconv(batch_input, out_channels):
    with tf.variable_scope("deconv"):
        batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
        filter = tf.get_variable("filter", [4, 4, out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, out_channels, in_channels]
        #     => [batch, out_height, out_width, out_channels]
        conv = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height * 2, in_width * 2, out_channels], [1, 2, 2, 1], padding="SAME")
        return conv

def pixel_wise_sigmoid(output_map):
    exponential_map = tf.exp(tf.subtract(output_map,tf.reduce_max(output_map)))
    sum_exp = tf.add(exponential_map ,tf.scalar_mul(tf.exp(tf.negative(tf.reduce_max(output_map))),tf.ones_like(exponential_map)))
    return tf.div(exponential_map,sum_exp)

Model = collections.namedtuple("Model", "outputs, gen_loss, train, dice_coeff")

def create_generator(generator_inputs, generator_outputs_channels,mode="train"):
    layers = []

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("encoder_1"):
        output = conv(generator_inputs, a.ngf, stride=2)
        layers.append(output)

    layer_specs = [
        a.ngf * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        a.ngf * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        a.ngf * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        a.ngf * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        a.ngf * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        a.ngf * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        a.ngf * 8, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            #rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            rectified = tf.nn.relu(layers[-1])
            convolved = conv(rectified, out_channels, stride=2)
            output = batchnorm(convolved)
            layers.append(output)

    layer_specs = [
        (a.ngf * 8, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (a.ngf * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (a.ngf * 8, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (a.ngf * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (a.ngf * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (a.ngf * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (a.ngf, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = deconv(rectified, out_channels)
            output = batchnorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output = deconv(rectified, generator_outputs_channels)
        #output = pixel_wise_softmax_2(output)
        #output = tf.sigmoid(output)
        layers.append(output)
    if mode=="train":
        return layers[-1]
    elif mode=="test":
        return tf.sigmoid(layers[-1])

def create_model(inputs, targets):
    with tf.variable_scope("generator") as scope:
        out_channels = int(targets.get_shape()[-1])
        outputs = create_generator(inputs, out_channels)
        out_map= tf.sigmoid(outputs)
    with tf.name_scope("dice_coefficient"):
        eps = 1e-5
        #prediction = pixel_wise_softmax_2(logits)
        intersection = tf.reduce_sum(targets* out_map)
        union =  eps + tf.reduce_sum(targets + out_map)
        dice_coeff = (2 * intersection/ (union))
    with tf.name_scope("entropy_loss"):
        shape_target = tf.shape(targets)
        targets_flat = tf.reshape(targets, [-1, shape_target[1]*shape_target[2]*shape_target[3]])
        outputs_flat = tf.reshape(outputs, [-1, shape_target[1]*shape_target[2]*shape_target[3]])
        #entropy =  tf.negative(tf.reduce_mean(tf.add(tf.multiply(targets_flat,tf.log(outputs_flat)),tf.multiply(one_mat-targets_flat,tf.log(one_mat-outputs_flat)))))
        entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = targets_flat, logits = outputs_flat))
    with tf.name_scope("generator_loss"):
        gen_loss = entropy    
    with tf.name_scope("generator_train"):
            gen_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            gen_train = gen_optim.minimize(gen_loss)

    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        gen_loss=gen_loss ,
        outputs=out_map,
        train=tf.group(incr_global_step, gen_train),
        dice_coeff = dice_coeff,
    )

def main():
    df = pd.read_csv('./train_wkt_v4.csv')
    Ids=df ['ImageId'].unique()
    if a.mode=="test":
        if not os.path.exists(a.save_dir):
            os.mkdir(a.save_dir)
        x_=tf.placeholder(tf.float32,shape=(1,p_size,p_size,3))
        with tf.variable_scope("generator") as scope:
            output_masks=create_generator(generator_inputs=x_,generator_outputs_channels=1,mode="test")
        saver=tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess,a.checkpoint)
            for i in glob.glob(os.path.join(a.input_dir,'*')):
                j=os.path.split(i)[-1]
                k=j.split('.')[0]
                #print k
                img=M(k)
                img=stretch_n(img)
                img=np.expand_dims(img,0)
                img_in=img[:,0:p_size,0:p_size,:]
                mask_out=sess.run(output_masks,feed_dict={x_:img_in})[0,:,:,0]
                mask_out[mask_out >= 0.5] = 255
                mask_out[mask_out < 0.5] = 0
                cv2.imwrite(os.path.join(a.save_dir,'masko_'+str(k)+'.png'),mask_out)
                print("Saved: "+str(os.path.join(a.save_dir,'masko_'+str(k)+'.png')))
    elif a.mode == "train":    
        x_=tf.placeholder(tf.float32,shape=(BATCH_SIZE,p_size,p_size,3))
        y_=tf.placeholder(tf.float32,shape=(BATCH_SIZE,p_size,p_size,1))
        model=create_model(x_,y_)
        saver=tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if a.checkpoint != None:
                saver.restore(sess,a.checkpoint)
            for j in range(Epochs):
                for i in range(Ids.shape[0]/BATCH_SIZE):
                #for i in range(1):
                    #if i==21 or i==20:
                    #    fetches={
                    #    "dice_coefficient" : model.dice_coeff,
                    #    "gen_loss": model.gen_loss
                    #    }
                    #    print "VALIDATION:"    
                    #else:
                    fetches={
                    "train" : model.train ,
                    "dice_coefficient" : model.dice_coeff ,
                    "gen_loss": model.gen_loss,
                    }
                    print("Epoch: %d Iteration: %d" %(j+1,i+1))
                    Ids,x_tr,y_tr=examples_load(Ids)
                    results=sess.run(fetches,feed_dict={x_ : x_tr,y_ : y_tr})
                    print("Dice_coeff: %f\nGenLoss: %f\n"%(results["dice_coefficient"],results["gen_loss"]))
                save_ck_path=saver.save(sess,os.path.join(a.save_path,"checkpoints","model.ckpt"))
                print("Saved Model to %s" %(save_ck_path))

main()