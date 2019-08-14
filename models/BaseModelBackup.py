import os
import cv2
import numpy as np

import torch
import torch.nn as nn

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from IPython.display import Image, display, clear_output

from bokeh.io import output_file, show, export_png
from bokeh.plotting import gridplot, figure, show, output_file, output_notebook


sns.set_style("whitegrid")
sns.set_palette(sns.dark_palette("purple"))

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
    
    # adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L139
    def save(self, state, filepath):
        if hasattr(self, 'p'):
            state['p'] = self.p
        torch.save(state, filepath)           

    def _plot_loss(self, ax, epoch, train_loss, valid_loss):
        ax.set_title("Error")
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Error')

        ax.plot(np.arange(len(train_loss)), train_loss, color="black")
        ax.plot(np.arange(len(valid_loss)), valid_loss, color="gray", linestyle="--")
        ax.legend(['Training error', 'Validation error'])

    def _plot_tsne(self, ax, z_plot, y_plot, classes, colors):
        ax.set_title('Latent space - TSNE')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        tsne = TSNE(n_components=2, perplexity=20)
        z_tsne = tsne.fit_transform(z_plot)
        for i in range(len(classes)):
            ax.scatter(*z_tsne[y_plot == classes[i]].T, c=colors[i], marker='o', label=classes[i], alpha=0.5)

        ax.legend(classes)

    def _plot_pca(self, ax, z_plot, y_plot, classes, colors):
        ax.set_title('Latent space - PCA')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')

        pca = PCA(n_components=2)
        z_pca = pca.fit_transform(z_plot)

        for i in range(len(classes)):
            ax.scatter(*z_pca[y_plot == classes[i]].T, c=colors[i], marker='o', label=classes[i], alpha=0.5)

        ax.legend(classes)

    def _plot_pca_eig(self, ax, z_plot, y_plot, classes, colors):
        ax.set_title('Latent space eigenvalues')
        ax.set_xlabel('Principal Components')
        ax.set_ylabel('Cumulative Explained Variance')


        for i in range(len(classes)):
            pca = PCA(n_components=min(z_plot.shape[1],y_plot[y_plot == classes[i]].shape[0]))
            z_pca = pca.fit_transform(z_plot[y_plot == classes[i]])
            var=np.cumsum(pca.explained_variance_ratio_)
            ax.plot(var, c=colors[i],label=classes[i])

        pca = PCA(n_components=min(z_plot.shape[1],z_plot.shape[0]))
        z_pca = pca.fit_transform(z_plot)
        total_var = np.cumsum(pca.explained_variance_ratio_)
        ax.plot(total_var,color='black',label='total')

        ax.legend()   
        
        
    def _plot_orig(self, ax, outputs, sides, x):         
        ax.set_title('Originals')
        ax.axis('off')

        if 'noisy_x' in outputs.keys():
            x = outputs['noisy_x']
        if x.shape[1] == 1:
            x = x.repeat(1,3,1,1)

        rows = 2
        columns = 2
        w,h = (120,90)

        canvas = np.zeros((h*rows, columns*w,3))
        for i in range(rows):
            for j in range(columns):
                idx = min(i % columns + rows * j, x.size()[0]-1)
                if sides == 'both_in_one':
                    if i == 0:
                        canvas[i*h:(i+1)*h, j*w:(j+1)*w] = cv2.resize(x[idx].cpu().permute((1,2,0)).detach().numpy()[:,:,:3],(w, h))
                    else:
                        canvas[i*h:(i+1)*h, j*w:(j+1)*w] = cv2.resize(x[idx].cpu().permute((1,2,0)).detach().numpy()[:,:,3:],(w, h))
                else:
                    canvas[i*h:(i+1)*h, j*w:(j+1)*w] = cv2.resize(x[idx].cpu().permute((1,2,0)).detach().numpy(),(w, h))

        ax.imshow(canvas)

    def _plot_recon(self, ax, sides, x, x_hat):
        ax.set_title('Reconstructions')
        ax.axis('off')
        
        rows = 2
        columns = 2
        w,h = (120,90)
        if x_hat.shape[1] == 1:
            x_hat = x_hat.repeat(1,3,1,1)

        canvas = np.zeros((h*rows, columns*w,3))
        for i in range(rows):
            for j in range(columns):
                idx = min(i % columns + rows * j, x.size()[0]-1)
                if sides == 'both_in_one':
                    if i == 0:
                        canvas[i*h:(i+1)*h, j*w:(j+1)*w] = cv2.resize(x_hat[idx].cpu().permute((1,2,0)).detach().numpy()[:,:,:3],(w, h))
                    else:
                        canvas[i*h:(i+1)*h, j*w:(j+1)*w] = cv2.resize(x_hat[idx].cpu().permute((1,2,0)).detach().numpy()[:,:,3:],(w, h))
                else:
                    canvas[i*h:(i+1)*h, j*w:(j+1)*w] = cv2.resize(x_hat[idx].cpu().permute((1,2,0)).detach().numpy(),(w, h))

        ax.imshow(canvas)
        
    def _plot_loss(self, train_loss, valid_loss):
        p_loss = figure(width=400, plot_height=400, title="Loss")
        p_loss.line(np.arange(len(train_loss)),train_loss, color="black",legend="Training Loss")
        p_loss.line(np.arange(len(valid_loss)),valid_loss, color="gray",line_dash='dotted', legend='Validation Loss')
        p_loss.xaxis.axis_label = 'Epoch'
        p_loss.yaxis.axis_label = 'Loss'
        return p_loss

    def _plot_tsne(self, z_plot, y_plot, classes, colors):
        tsne = TSNE(n_components=2, perplexity=20)
        z_tsne = tsne.fit_transform(z_plot)

        p_tsne = figure(width=400, plot_height=400, title="Latent Space - TSNE")
        for i in range(len(classes)):
            p_tsne.scatter(*z_tsne[y_plot == classes[i]].T, color=colors[i], marker='o', legend=classes[i], alpha=0.2)

        p_tsne.xaxis.axis_label = 'Dimension 1'
        p_tsne.yaxis.axis_label = 'Dimension 2'
        return p_tsne
    def _plot_pca(self, z_plot, y_plot, classes, colors):
        pca = PCA(n_components=2)
        z_pca = pca.fit_transform(z_plot)

        p_pca = figure(width=400, plot_height=400, title="Latent Space - PCA")
        for i in range(len(classes)):
            p_pca.scatter(*z_pca[y_plot == classes[i]].T, color=colors[i], marker='o', legend=classes[i], alpha=0.2)

        p_pca.xaxis.axis_label = 'Dimension 1'
        p_pca.yaxis.axis_label = 'Dimension 2'
        return p_pca

    def _plot_pca_eig(self, z_plot, y_plot, classes, colors):
        p_eig = figure(width=400, plot_height=400, title="Latent space eigenvalues")
        for i in range(len(classes)):
            pca = PCA(n_components=min(z_plot.shape[1],y_plot[y_plot == classes[i]].shape[0]))
            z_pca = pca.fit_transform(z_plot[y_plot == classes[i]])
            var=np.cumsum(pca.explained_variance_ratio_)
            p_eig.line(np.arange(len(var)),var, color=colors[i],legend=classes[i])

        pca = PCA(n_components=min(z_plot.shape[1],z_plot.shape[0]))
        z_pca = pca.fit_transform(z_plot)
        total_var = np.cumsum(pca.explained_variance_ratio_)
        p_eig.line(np.arange(len(total_var)),total_var,color='black',legend='total')

        p_eig.xaxis.axis_label = 'Principal Components'
        p_eig.yaxis.axis_label = 'Cumulative Explained Variance'
        return p_eig

    def _plot_orig(self, outputs, sides, x):         
        p_orig = figure(width=400, plot_height=400, title="Originals", x_range=(0, 240), y_range=(0, 180))
        p_orig.xaxis.visible = False
        p_orig.yaxis.visible = False

        if 'noisy_x' in outputs.keys():
            x = outputs['noisy_x']
        if x.shape[1] == 1:
            x = x.repeat(1,3,1,1)

        rows = 2
        columns = 2
        w,h = (120,90)

        canvas = np.zeros((h*rows, columns*w,3))
        for i in range(rows):
            for j in range(columns):
                idx = min(i % columns + rows * j, x.size()[0]-1)
                if sides == 'both_in_one':
                    if i == 0:
                        canvas[i*h:(i+1)*h, j*w:(j+1)*w] = cv2.resize(x[idx].cpu().permute((1,2,0)).detach().numpy()[:,:,:3],(w, h))
                    else:
                        canvas[i*h:(i+1)*h, j*w:(j+1)*w] = cv2.resize(x[idx].cpu().permute((1,2,0)).detach().numpy()[:,:,3:],(w, h))
                else:
                    canvas[i*h:(i+1)*h, j*w:(j+1)*w] = cv2.resize(x[idx].cpu().permute((1,2,0)).detach().numpy(),(w, h))
        canvas = cv2.cvtColor((canvas*255).astype('uint8'), cv2.COLOR_RGB2RGBA)
        canvas = canvas[::-1]
        p_orig.image_rgba(image=[canvas], x=0, y=0, dw=120*2, dh=90*2)
        return p_orig

    def _plot_recon(self, sides, x, x_hat):       
        p_recon = figure(width=400, plot_height=400, title="Reconstructions", x_range=(0, 240), y_range=(0, 180))
        p_recon.xaxis.visible = False
        p_recon.yaxis.visible = False

        rows = 2
        columns = 2
        w,h = (120,90)
        if x_hat.shape[1] == 1:
            x_hat = x_hat.repeat(1,3,1,1)

        canvas = np.zeros((h*rows, columns*w,3))
        for i in range(rows):
            for j in range(columns):
                idx = min(i % columns + rows * j, x.size()[0]-1)
                if sides == 'both_in_one':
                    if i == 0:
                        canvas[i*h:(i+1)*h, j*w:(j+1)*w] = cv2.resize(x_hat[idx].cpu().permute((1,2,0)).detach().numpy()[:,:,:3],(w, h))
                    else:
                        canvas[i*h:(i+1)*h, j*w:(j+1)*w] = cv2.resize(x_hat[idx].cpu().permute((1,2,0)).detach().numpy()[:,:,3:],(w, h))
                else:
                    canvas[i*h:(i+1)*h, j*w:(j+1)*w] = cv2.resize(x_hat[idx].cpu().permute((1,2,0)).detach().numpy(),(w, h))

        canvas = cv2.cvtColor((canvas*255).astype('uint8'), cv2.COLOR_RGB2RGBA)
        canvas = canvas[::-1]
        p_recon.image_rgba(image=[canvas], x=0, y=0, dw=120*2, dh=90*2)
        return p_recon

    def plot_preprocess(self, outputs, batch, results, batchSize, cuda):
        x_hat = outputs['x_hat']
        z = outputs['z']

        x, y, _id = batch
        y = np.array(y)

        if results is not None:
            z_train, y_train, z_valid, y_valid = results
            z_plot = z_valid
            y_plot = np.array(y_valid)
        else:
            z_plot = z.cpu().detach().numpy()
            y_plot = y

        classes = np.unique(y_plot)
        if cuda:
            x = x.cpu().detach()
            x_hat = x_hat.cpu().detach()
            z = z.cpu().detach()

        z = z.reshape((batchSize,-1))
        return x, x_hat, z_plot, y_plot, classes
        
    def plot(self,train_loss,valid_loss,outputs,batch,epoch,batchSize,cuda=True,results = None,savefp=False,sides='both'):
        colors = ['blue','red','green','yellow','purple','pink','black','brown','purple','cyan','magenta']
        tmp_img = "tmp_ae_out.png"

        x, x_hat, z_plot, y_plot, classes = self.plot_preprocess(outputs, batch, results, batchSize, cuda)
        assert len(classes) <= len(colors)

        # -- Plotting --
        f, axarr = plt.subplots(3, 2, figsize=(20, 30))

        # Loss
        ax = axarr[0, 0]
        self._plot_loss(ax, epoch, train_loss, valid_loss)

        # Latent space TSNE
        ax = axarr[0, 1]
        self._plot_tsne(ax, z_plot, y_plot, classes, colors)    

        # Latent space PCA
        ax = axarr[1, 0]
        self._plot_pca(ax, z_plot, y_plot, classes, colors)
        
        # Latent space eigenvalues
        ax = axarr[1, 1]
        self._plot_pca_eig(ax, z_plot, y_plot, classes, colors)

        # Inputs
        ax = axarr[2, 0]
        self._plot_orig(ax, outputs, sides, x)

        # Reconstructions
        ax = axarr[2, 1]
        self._plot_recon(ax, sides, x, x_hat)

        
        plt.savefig(tmp_img)
        if savefp:
            plt.savefig(savefp)
        plt.close(f)
        display(Image(filename=tmp_img))
        clear_output(wait=True)

        #os.remove(tmp_img)
        
    def plot(self, train_loss,valid_loss,outputs,batch,batchSize,cuda=True,results = None,savefp=False,sides='both'):
        print('started plotting')
        output_notebook()
        colors = ['blue','red','green','yellow','purple','pink','black','brown','purple','cyan','magenta']
        tmp_img = "tmp_ae_out.png"

        x, x_hat, z_plot, y_plot, classes = self.plot_preprocess(outputs, batch, results, batchSize, cuda)
        assert len(classes) <= len(colors)

        # Loss
        p_loss = self._plot_loss(train_loss, valid_loss)

        # Latent space TSNE
        p_tsne = self._plot_tsne(z_plot, y_plot, classes, colors)    

        # Latent space PCA
        p_pca = self._plot_pca(z_plot, y_plot, classes, colors)

        # Latent space eigenvalues
        p_eig = self._plot_pca_eig(z_plot, y_plot, classes, colors)

        # Inputs
        p_orig = self._plot_orig(outputs, sides, x)

        # Reconstructions
        p_recon = self._plot_recon(sides, x, x_hat)

        p = gridplot([[p_loss, p_tsne], [p_pca, p_eig], [p_orig, p_recon]])

        output_file("tmp_ae_out.html")
        if savefp:
            output_file(savefp)
        # show the results
        show(p)

        export_png(p, filename=tmp_img)
        output_file("tmp_ae_out.html")
        return p