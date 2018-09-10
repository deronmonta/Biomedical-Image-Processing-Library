
import numpy as np
import matplotlib.pyplot as plt


def merge_img(images,size,resize_factor=0.3):
    """For merging multiple images
    
    Parameters
    ----------
    images : numpy array
        [Height, Width, Channel]
    size : list
        []
    resize_factor : float, optional
         (the default is 0.3)
    
    Returns
    -------
    img:
        merged images
    """
   
    
    h, w = images.shape[1], images.shape[2]
    h = int(h * resize_factor)
    w = int(w * resize_factor)
    
    img = np.zeros((h * size[0], w * size[1],3))
    print(h)
    print(w)
    print(img.shape)
    for idx, image in enumerate(images):
        i = int(idx % size[1])
        j = int(idx / size[1]) 
        image_ = imresize(image,size=(w,h),interp='bicubic')
        img[j * h:j * h + h, i * w:i * w + w,:] = image_
    print(img.shape)
    return img

def plot_gan_results(generator, epoch, test_samples, z_space, save_dir, fig_size=(5,5)):
    """[summary]
    
    Parameters
    ----------
    generator : [type]
        [description]
    epoch : [type]
        [description]
    test_samples : [type]
        [description]
    z_space : [type]
        [description]
    save_dir : [type]
        [description]
    fig_size : tuple, optional
        [description] (the default is (5,5), which [default_description])
    
    """

    
    
    fixed_z = torch.randn(test_samples,z_space).view(-1,z_space,1,1).float().cuda()
    generated_images = generator(fixed_z)

    n_rows = np.sqrt(test_samples).astype(np.int32)
    n_cols = np.sqrt(test_samples).astype(np.int32)

    fig, axes = plt.subplots(n_rows, n_cols, figsize= fig_size)

    for ax , img in zip(axes.flatten(), generated_images):
        img = img.cpu().data.numpy()
        img = np.squeeze(img)
        ax.axis('off')
        #ax.set_adjustable('box-forced')

        ax.imshow(img,cmap='gray',aspect='equal')
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    title = 'Epoch {} '.format(epoch)
    fig.text(0.5,0.04, title, ha='center')
    save_fn = save_dir + 'DCGAN_epoch_{:d}'.format(epoch) + '.png'
    plt.savefig(save_fn)
    plt.close('all')
