from torchvision import transforms
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange, tqdm
from utils import conv_output_shape, scale

class CVAE_hard_coded(nn.Module):
    def __init__(self, img_shape, n_channels=1, latent_dim=2, kernel_size=3, stride=1):
        """
        img_shape is a tuple containing the height and width of the image in that order. img_shape=(12,24)
        """
        super(CVAE_hard_coded, self).__init__()

#         INCREASE THE NUMBER OF OUTPUT CHANNELS TO CREATE MORE FEATURES AND SEE K SUSEDE
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=64, kernel_size=kernel_size, stride=stride),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=kernel_size, stride=stride),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=kernel_size, stride=stride),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=256, out_channels=300, kernel_size=kernel_size, stride=stride),
            nn.ReLU(inplace=True),   
            nn.MaxPool2d(kernel_size=2, stride=2),
##             nn.AdaptiveAvgPool2d((6,6))
        )
        
        pad = 0
        for i in range(8-4): #number of convolutions applied
            if i == 0:
                h, w = conv_output_shape(100, 100, kernel_size=kernel_size, stride=stride, pad=pad, dilation=1)
                # Max pool output:
                h, w = conv_output_shape(h, w, kernel_size=2, stride=2, pad=0, dilation=1)
            else:
                h, w = conv_output_shape(h, w, kernel_size=kernel_size, stride=stride, pad=pad, dilation=1)
                # Max pool output:
                h, w = conv_output_shape(h, w, kernel_size=2, stride=2, pad=0, dilation=1)  
            
        self.last_h, self.last_w = h, w
        self.last_channel = 300
        
        self.encoder_FC = nn.Sequential(
            nn.Linear(self.last_channel * self.last_h * self.last_w, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 500),
            nn.ReLU(inplace=True),
        )
        
        self.enc_mu = nn.Linear(500, latent_dim)
        self.enc_logvar = nn.Linear(500, latent_dim)

        self.encoder_layers = [self.encoder_conv, self.encoder_FC, self.enc_logvar, self.enc_mu]
        

        self.decoder_FC = nn.Sequential(
            nn.Linear(latent_dim, 500),
            nn.ReLU(inplace=True),            
            nn.Linear(500, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, self.last_channel * self.last_h * self.last_w),
            nn.ReLU(inplace=True),
        )

        self.decoder_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(in_channels=300, out_channels=256, kernel_size=kernel_size, stride=stride),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=kernel_size, stride=stride),
            nn.ReLU(inplace=True),
#             nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=kernel_size, stride=stride),
            nn.ReLU(inplace=True),
# #             nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(in_channels=64, out_channels=n_channels, kernel_size=kernel_size, stride=stride),
            nn.Upsample(size=(100,100), mode='nearest')
##             nn.ReLU(inplace=True), No relu at the end (values roughly between -3 and 3)
        )
        self.classifier = None

    def embed(self, x):
        x = self.encoder_conv(x)

        x = x.view(-1, self.last_channel * self.last_h * self.last_w)

        x = self.encoder_FC(x)

        mu = self.enc_mu(x)
        logvar = self.enc_logvar(x)

        # Reparametrization trick
        std = torch.exp(logvar/2)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z, mu, logvar

    def decode(self, z):
        x = self.decoder_FC(z)

        x = x.view(-1, self.last_channel, self.last_h, self.last_w)

        x = self.decoder_conv(x)
        x = torch.sigmoid(x)

        return x


    def forward(self, x):
        z, mu, logvar = self.embed(x)
        return self.decode(z), mu, logvar

    def get_encoder_parameters(self):
        from itertools import chain
        return_params = chain(*[layer.parameters() for layer in self.encoder_layers])
        return return_params


    def train_model_with_validation_with_2_classifiers(self, train_loader, val_loader, args, writer=None, classifier_weights=None, with_dapi=False):
        print("Starting training of the CVAE model:")
        print(f"Reduction = {args['reduction']}")
        beta = args["beta"]
        print(args["n_classes"])

        optimizer = torch.optim.Adam(self.parameters(), lr=args["lr"])

        # Random or not
        self.binary_classifier = sample_classifier(n_classes = 2, n_input = args['n_latent_dims'])
        self.binary_classifier.to(args['device'])
        # First define classifier
        self.classifier = sample_classifier(n_classes = args["n_classes"], n_input = args['n_latent_dims'])
        self.classifier.to(args['device'])

        # seperate optimizer for only the encoding layers, with only its parameters
        optimizer_enc = torch.optim.Adam(self.get_encoder_parameters(), lr=args["lr"])

        losses = {"reconstruction_loss": [], "kl_loss": [], "val_loss":[], "predict_class_loss": [], "classify_embeding_loss": [], "binary_predict_class_loss": [], "binary_classify_embeding_loss": []}

        # Define loss and optimizer for the classifier
        binary_classify_criterion = nn.BCELoss()
        if args["n_classes"] == 2:
            classify_criterion = nn.BCELoss()
        else:
            classify_criterion = nn.CrossEntropyLoss(classifier_weights)

        binary_classify_optimizer = torch.optim.Adam(self.binary_classifier.parameters(), lr=args["class_lr"])
        classify_optimizer = torch.optim.Adam(self.classifier.parameters(), lr=args["class_lr"])

        for epoch in trange(args["epochs"], desc="Training", unit="Epoch"):
        # for epoch in range(args["epochs"]):
            total_reconstruction_loss = 0
            total_kl_loss = 0
            total_val_loss = 0
            total_predict_loss = 0
            total_classify_loss = 0
            total_binary_predict_loss = 0
            total_binary_classify_loss = 0

            self.train()
            for batch, label, binary_label in train_loader:

                batch = batch.to(args["device"])
                binary_label = binary_label.to(args["device"])
                label = label.to(args["device"])

                reconstruction, mu, logvar = self.forward(batch)

                kl_divergence = 0.5 * torch.sum(-1 - logvar + mu.pow(2) + logvar.exp())
                kl_divergence *= beta
                if with_dapi:
                    rec_loss = F.mse_loss(batch[:,0,:,:], reconstruction[:,0,:,:], reduction=args["reduction"])
                    # rec_loss = F.mse_loss(batch, reconstruction, reduction=args["reduction"])
                else:
                    rec_loss = F.mse_loss(batch, reconstruction, reduction=args["reduction"])

                loss = rec_loss + kl_divergence
                # loss = rec_loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad(True)


                total_reconstruction_loss += rec_loss.item()
                total_kl_loss += kl_divergence.item()


                # After losses, train binary classifier
                self.eval()
                self.binary_classifier.train()

                embedding, _, _ = self.embed(batch)

                binary_prediction = self.binary_classifier.forward(embedding)
                binary_predict_loss = binary_classify_criterion(binary_prediction, binary_label)
                binary_predict_loss *= 100

                total_binary_predict_loss += binary_predict_loss.item()

                binary_predict_loss.backward()
                binary_classify_optimizer.step()
                binary_classify_optimizer.zero_grad(True)

                # set classifier to eval
                self.binary_classifier.eval()
                # Set encoder to train for another forward pass
                for enc_layer in self.encoder_layers:
                    enc_layer.train()

                # forward pass AE again:
                # self.forward()

                #classify embeddings
                embedding, _, _ = self.embed(batch)
                binary_prediction = self.binary_classifier.forward(embedding)
                binary_class_loss = binary_classify_criterion(binary_prediction, binary_label)
                binary_class_loss *= 100

                total_binary_classify_loss += binary_class_loss.item()

                binary_class_loss.backward()
                # step only the encoder
                optimizer_enc.step()
                optimizer_enc.zero_grad(True)



                # After losses of AE, train classifier
                self.eval()
                self.classifier.train()
                embedding, _, _ = self.embed(batch)

                prediction = self.classifier.forward(embedding)
                predict_loss = classify_criterion(prediction, label)

                total_predict_loss += predict_loss.item()

                predict_loss.backward()
                classify_optimizer.step()
                classify_optimizer.zero_grad(True)

                # set classifier to eval
                self.classifier.eval()
                # Set encoder to train for another forward pass
                for enc_layer in self.encoder_layers:
                    enc_layer.train()

                # forward pass AE again:
                # self.forward()

                #classify embeddings
                embedding, _, _ = self.embed(batch)
                prediction = self.classifier.forward(embedding)
                class_loss = classify_criterion(prediction, label)
                # class_loss *= 100 #Tried this, didn't change anything

                total_classify_loss += class_loss.item()

                class_loss.backward()
                # step only the encoder
                optimizer_enc.step()
                optimizer_enc.zero_grad(True)

                
            del batch
            torch.cuda.empty_cache()
            self.eval()
            for val_batch, val_label, binary_val_label in val_loader:
                val_batch = val_batch.to(args["device"])
                val_label = val_label.to(args["device"])

                reconstruction, mu, logvar = self.forward(val_batch)

                val_loss = F.mse_loss(val_batch, reconstruction, reduction=args["reduction"])
                total_val_loss += val_loss.item()



            losses["reconstruction_loss"].append(total_reconstruction_loss)
            losses["kl_loss"].append(total_kl_loss)
            losses["val_loss"].append(total_val_loss)
            losses["predict_class_loss"].append(total_predict_loss)
            losses["classify_embeding_loss"].append(total_classify_loss)
            losses["binary_predict_class_loss"].append(total_binary_predict_loss)
            losses["binary_classify_embeding_loss"].append(total_binary_classify_loss)


            if writer is not None:
                writer.add_scalar("Loss/train", total_reconstruction_loss, epoch)
                writer.add_scalar("Loss/validation", total_val_loss, epoch)
                writer.add_scalar("Loss/KL", total_kl_loss, epoch)
                writer.add_scalar("Loss/predict", total_predict_loss, epoch)
                writer.add_scalar("Loss/classify", total_classify_loss, epoch)
                writer.add_scalar("Loss/binary_predict", total_binary_predict_loss, epoch)
                writer.add_scalar("Loss/binary_classify", total_binary_classify_loss, epoch)
                writer.flush()


        return losses
    def transfer_learn_model_with_validation(self, train_loader, val_loader, args, writer=None, with_dapi=False):
        print("Starting training of the CVAE model:")
        print(f"Reduction = {args['reduction']}")
        print(f"beta = {args['beta']}")
        beta = args['beta']

        optimizer = torch.optim.Adam(self.parameters(), lr=args["lr"])

        losses = {"reconstruction_loss": [], "kl_loss": [], "val_loss":[]}

        for epoch in trange(args["epochs"], desc="Training", unit="Epoch"):
            total_reconstruction_loss = 0
            total_kl_loss = 0
            total_val_loss = 0


            # Freeze convolutional layers
            for param in self.encoder_conv.parameters():
                param.requires_grad = False


            # Set enc FC, enc mu and enc logvar to train
            for i, layer in enumerate(self.encoder_layers):
                if i != 0: # since the first one is all the convolutional layers combined
                    layer.train()

            # set decoder FC to train
            self.decoder_FC.train()

            for batch in train_loader:
                batch = batch.to(args["device"])
                reconstruction, mu, logvar = self.forward(batch)

                kl_divergence = 0.5 * torch.sum(-1 - logvar + mu.pow(2) + logvar.exp())
                kl_divergence *= beta
                if with_dapi:
                    rec_loss = F.mse_loss(batch[:,0,:,:], reconstruction[:,0,:,:], reduction=args["reduction"])
                else:
                    rec_loss = F.mse_loss(batch, reconstruction, reduction=args["reduction"])

                loss = rec_loss + kl_divergence

                loss.backward()
                optimizer.step()
                optimizer.zero_grad(True)


                total_reconstruction_loss += rec_loss.item()
                total_kl_loss += kl_divergence.item()
                
            del batch
            torch.cuda.empty_cache()
            self.eval()
            for val_batch in val_loader:
                val_batch = val_batch.to(args["device"])
                reconstruction, mu, logvar = self.forward(val_batch)

                val_loss = F.mse_loss(val_batch, reconstruction, reduction=args["reduction"])
                total_val_loss += val_loss.item()



            losses["reconstruction_loss"].append(total_reconstruction_loss)
            losses["kl_loss"].append(total_kl_loss)
            losses["val_loss"].append(total_val_loss)

            if writer is not None:
                writer.add_scalar("Loss/train", total_reconstruction_loss, epoch)
                writer.add_scalar("Loss/validation", total_val_loss, epoch)
                writer.add_scalar("Loss/KL", total_kl_loss, epoch)
                writer.flush()

        return losses

    def custom_scale(self, tensor, min_value, max_value):
        v_min, v_max = tensor.min(), tensor.max()
        new_min, new_max = min_value, max_value
        v_p = (tensor - v_min)/(v_max - v_min)*(new_max - new_min) + new_min
        return v_p

    # assuming arr.shape = [X,Y]
    def prepare_new_input(self, arr):
        tf = transforms.ToTensor()
        image_tensor = tf(arr).float()
        image_tensor = transforms.Normalize(torch.mean(image_tensor),torch.std(image_tensor))(image_tensor) 
        image_tensor = torch.unsqueeze(image_tensor, dim=0)
        scaled_image = self.custom_scale(image_tensor, 0, 1)
        return scaled_image

    # assumes array is prepared by prepare_new_input
    def classify_new_input(self, arr, binary=False):
        embedding, _, _ = self.embed(arr)
        if binary:
            prediction = self.binary_classifier.forward(embedding)
        else:
            prediction = self.classifier.forward(embedding)
        pred_index = torch.argmax(prediction)
        pred_index = int(pred_index.cpu())
        return pred_index, prediction


class sample_classifier(nn.Module):
    def __init__(self, n_classes, n_input):
        # n_input == latent dimensions
        super(sample_classifier, self).__init__()
        self.n_classes = n_classes
        self.n_input = n_input
        
        self.lin1 = nn.Linear(n_input, n_classes)
        # self.lin2 = nn.Linear(16, n_classes)

    def forward(self, latent):
        x = self.lin1(latent)
        x = torch.sigmoid(x).to(torch.float64)
        return x
