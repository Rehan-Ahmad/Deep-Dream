# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 20:47:43 2018

@author: Rehan
"""
import cv2
import torch
from torch.autograd import Variable
from torch.optim import SGD
from torchvision import models
import numpy as np
import copy

class DeepDream():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, conv_layer_num, filter_num, image_path):
        self.model = model
        self.model.eval()
        self.conv_layer_num = conv_layer_num
        self.filter_num = filter_num
        self.conv_output = 0
        # read the image
        self.created_image = cv2.imread(image_path, 1)
        # Hook the layer to get result of the convolution
        self.hook()

    def hook(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.filter_num]

        # Hook the selected layer
        self.model[self.conv_layer_num].register_forward_hook(hook_function)

    def dream(self, iters):
        # Process image and return variable
        self.processed_image = self.preprocess_image(self.created_image, False)
        # Define optimizer for the image
        # Earlier layers need higher learning rates to visualize whereas later layers need less
        optimizer = SGD([self.processed_image], lr=12,  weight_decay=1e-4)
        for i in range(1, iters):
            optimizer.zero_grad()
            # Assign image to a variable to move forward in the model
            x = self.processed_image
            for index, layer in enumerate(self.model):
                # Forward
                x = layer(x)
                # Only need to forward until we the selected layer is reached
                if index == self.conv_layer_num:
                    break

            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()[0]))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = self.recreate_image(self.processed_image)
            # Save image every 20 iteration
            if i % 20 == 0:
                cv2.imwrite('./dream_l' + str(self.conv_layer_num) +
                            '_f' + str(self.filter_num) + '_iter'+str(i)+'.jpg',
                            self.created_image)

    def preprocess_image(self, cv2im, resize_im=True):
        """
            Processes image for CNNs
    
        Args:
            PIL_img (PIL_img): Image to process
            resize_im (bool): Resize to 224 or not
        returns:
            im_as_var (Pytorch variable): Variable that contains processed float tensor
        """
        # mean and std list for channels (Imagenet)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # Resize image
        if resize_im:
            cv2im = cv2.resize(cv2im, (224, 224))
        im_as_arr = np.float32(cv2im)
        im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
        im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
        # Normalize the channels
        for channel, _ in enumerate(im_as_arr):
            im_as_arr[channel] /= 255
            im_as_arr[channel] -= mean[channel]
            im_as_arr[channel] /= std[channel]
        # Convert to float tensor
        im_as_ten = torch.from_numpy(im_as_arr).float()
        # Add one more channel to the beginning. Tensor shape = 1,3,224,224
        im_as_ten.unsqueeze_(0)
        # Convert to Pytorch variable
        im_as_var = Variable(im_as_ten, requires_grad=True)
        return im_as_var

    def recreate_image(self, im_as_var):
        """
            Recreates images from a torch variable, sort of reverse preprocessing
    
        Args:
            im_as_var (torch variable): Image to recreate
    
        returns:
            recreated_im (numpy arr): Recreated image in array
        """
        reverse_mean = [-0.485, -0.456, -0.406]
        reverse_std = [1/0.229, 1/0.224, 1/0.225]
        recreated_im = copy.copy(im_as_var.data.numpy()[0])
        for c in range(3):
            recreated_im[c] /= reverse_std[c]
            recreated_im[c] -= reverse_mean[c]
        recreated_im[recreated_im > 1] = 1
        recreated_im[recreated_im < 0] = 0
        recreated_im = np.round(recreated_im * 255)
    
        recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
        # Convert RBG to GBR
        recreated_im = recreated_im[..., ::-1]
        return recreated_im

if __name__ == '__main__':

    conv_layer = 34
    filter_num = 94
    iterations = 51
    image_path = 'park.jpg'
    im = cv2.resize(cv2.imread(image_path),(480,380))
    cv2.imwrite('parkresize.jpg', im)   
    image_path = 'parkresize.jpg'
    
    pretrained_model = models.vgg19(pretrained=True).features
    ddream = DeepDream(pretrained_model, conv_layer, filter_num, image_path)
    ddream.dream(iterations)
