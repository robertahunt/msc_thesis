import os
import cv2
import sys
import matplotlib

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from IPython.display import Image, display, clear_output

sns.set_style("whitegrid")
sns.set_palette(sns.dark_palette("purple"))

global start_time
def start_timer():
    global start_time
    start_time = pd.Timestamp.now()
    
def tick(msg=''):
    global start_time
    print(msg + ', Time Taken: %s'%str(pd.Timestamp.now() - start_time))

def plot(train_loss,valid_loss,outputs,batch,epoch,batchSize,cuda=True,results = None,savefp=False,sides='both'):
    x_hat = outputs['x_hat']
    z = outputs['z']
    
    x, y = batch
    y = np.array(y)
    
    colors = ['blue','red','green','yellow','purple','pink']
    if results is not None:
        z_train, y_train, z_valid, y_valid = results
        z_plot = z_valid
        y_plot = np.array(y_valid)
    else:
        z_plot = z.cpu().detach().numpy()
        y_plot = y
        
    classes = np.unique(y)
    
    if cuda:
        x = x.cpu().detach()
        x_hat = x_hat.cpu().detach()
        z = z.cpu().detach()
    
    z = z.reshape((batchSize,-1))
    
    tmp_img = "tmp_ae_out.png"
    
    # -- Plotting --
    f, axarr = plt.subplots(3, 2, figsize=(20, 30))

    # Loss
    ax = axarr[0, 0]
    ax.set_title("Error")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error')


    ax.plot(np.arange(epoch+1), train_loss, color="black")
    ax.plot(np.arange(epoch+1), valid_loss, color="gray", linestyle="--")
    ax.legend(['Training error', 'Validation error'])

    # Latent space TSNE
    ax = axarr[0, 1]

    ax.set_title('Latent space - TSNE')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    
    tsne = TSNE(n_components=2, perplexity=20)
    z_tsne = tsne.fit_transform(z_plot)
    for i in range(len(classes)):
        ax.scatter(*z_tsne[y_plot == classes[i]].T, c=colors[i], marker='o', label=classes[i])

    ax.legend(classes)
    
    # Latent space PCA
    ax = axarr[1, 0]

    ax.set_title('Latent space - PCA')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    
    
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(z_plot)

    for i in range(len(classes)):
        ax.scatter(*z_pca[y_plot == classes[i]].T, c=colors[i], marker='o', label=classes[i])

    ax.legend(classes)
    
    # Latent space eigenvalues
    ax = axarr[1, 1]

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

    # Inputs
    ax = axarr[2, 0]
    ax.set_title('Originals')
    ax.axis('off')

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
    
    # Reconstructions
    ax = axarr[2, 1]
    ax.set_title('Reconstructions')
    ax.axis('off')

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
    
    plt.savefig(tmp_img)
    if savefp:
        plt.savefig(savefp)
    plt.close(f)
    display(Image(filename=tmp_img))
    clear_output(wait=True)

    os.remove(tmp_img)

    
def check_memory_usage():
    a = []
    for var, obj in locals().items():
        a += [[var, sys.getsizeof(obj)]]
    print(pd.DataFrame(a).sort_values(1))
    
def count_parameters(model):
    total_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param = np.prod(param.size())
            #if param.dim() > 1:
                #print(name, ':', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
            #else:
                #print(name, ':', num_param)
            total_param += num_param
    return total_param


def adjust_range(img):
    return (255*((img - img.min())/(img.max() - img.min()))).astype('uint8')