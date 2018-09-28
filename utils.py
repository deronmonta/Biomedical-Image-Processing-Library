import numpy as np
import SimpleITK as sitk

import numpy as np
import SimpleITK as sitk 
import pandas as pd
from glob import glob
from tqdm import tqdm # For progress bars
from skimage import measure, morphology
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import convex_hull_image
import glob
import os
import matplotlib
matplotlib.use('Agg')# To run on servers
import matplotlib.pyplot as plt

def volume_to_MIP(np_img, slices_num=5):

    ''' 
    Transform a 3D volume to a Maximun Intensity Projection (MIP)
    create the mip image from original image, slice_num is the number of 
    slices for maximum intensity projection
    '''

    img_shape = np_img.shape
    np_mip = np.zeros(img_shape)
    for i in range(img_shape[0]):
        start = max(0, i-slices_num)
        np_mip[i,:,:] = np.amax(np_img[start:i+1],0)

    return np_mip


def normalize_volume(volume):
    """Clip a CT volume between MIN_BOUND and MAX_BOUND, then normalize between 0 and 255
    
    Args:
        volume (numpy): numpy array to be normalized
    
    Returns:
        volume (numpy): Normalized array
    """
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    volume = (volume - MIN_BOUND) /(MAX_BOUND - MIN_BOUND)
    volume[volume > 1] = 1 #Clip everything larger than 1 and 0
    volume[volume < 0] = 0
    volume = (volume*255).astype('uint8')

    return volume


def load_scans(filename):
    """Load images with .mhd, .raw format
    
    Args:
        filename (string): full path to the .mhd file
    
    Returns:
        img_array: np array containing image data
        origin (list): origina coodinates
        spacing (list): [z,x,y] spacing
    """
    itk_image = sitk.ReadImage(filename)
    img_array = sitk.GetArrayFromImage(itk_image)

    origin = np.array(list(reversed(itk_image.GetOrigin())))# x y z origin in world coordinates
    spacing = np.array(list(reversed(itk_image.GetSpacing())))# Spacing in world coordinates

    return img_array,origin,spacing

def WorldCoord_to_Voxel(world_coord,origin,spacing):

    
    '''
    Convert the world coordinates in the annotation files to voxel coordinates
    Args:
        world_coord: 
        origin: origin coordinates obtained from itk object loader 
        spacing: spacing obtained from image header loader or other sources 
    Output:
        voxel_coordinated: []

    '''
    stretched_voxel_coordinates = np.absolute(world_coord - origin)
    voxel_coordinates = stretched_voxel_coordinates / spacing
    voxel_coordinates = list(map(int,voxel_coordinates))

    return voxel_coordinates

def Voxel_to_WorldCoord(voxel_coord,origin,spacing):
    '''
    Convert the voxel coordinates back to world coordinates
    Args:
        voxel_coord: 
        origin: origin coordinates obtained from itk object loader 
        spacing:
    Output:

    '''
    stretched_voxel_coordinates = voxel_coord * spacing
    world_coordinates = stretched_voxel_coordinates + origin

    return world_coordinates



def coord_to_bbox(coord,diameter):
    '''
    Convert centroid coordinates and diameter to cooresponding bounding box 

    Args:
        coord: [x ,y ,z]
        diameter: float 

    Return:
        bbox =[]
        
    '''
    y1 = coord[0] - int(0.5*diameter)
    y2 = coord[0] + int(0.5*diameter)
    x1 = coord[1] - int(0.5*diameter)
    x2 = coord[1] + int(0.5*diameter)
    bbox = [y1,x1,y2,x2]
    return bbox


def crop_volume(volume,to_keep=64):
    '''
    Crop out pixels that's not in the center 
    Arg:
        volume: 4D numpy array, with dimension [modality, z_len, x_len, y_len ]
        to_keep: pixels to keep in the center
    Outputs:
        cropped_volume: [modality, z_new_len, x_new_len, y_new_len]
    ''' 
    new_width, new_height, new_height_z = to_keep, to_keep, to_keep
    _,z_len,x_len, y_len = volume.shape

    left = np.ceil((x_len - new_width)/2.)
    top = np.ceil((y_len - new_height)/2.)
    top_z = np.ceil((z_len - new_height_z)/2)

    right = np.floor((x_len + new_width)/2.)
    bottom = np.floor((y_len + new_height)/2.)
    bottom_z =np.floor((z_len+ new_height_z)/2.)

    cropped_volume = volume[:,bottom_z:top_z,left:right,bottom:top, 64:192]

    return cropped_volume

def extract_3D_patch(input,patch_shape,xstep=24,ystep=24,zstep=24):
    """Extract 3D patches from a whole volume
    
    Args:
        input (3D numpy array): 
        patch_shape (list): Output shape of the patches 
        xstep (int, optional): Defaults to 24. [description]
        ystep (int, optional): Defaults to 24. [description]
        zstep (int, optional): Defaults to 24. [description]
    
    Returns:
        [type]: [description]
    """
    patches_3D = np.lib.stride_tricks.as_strided(input, ((input.shape[0] - patch_shape[0] + 1) // xstep, (input.shape[1] - patch_shape[1] + 1) // ystep,
                                                  (input.shape[2] - patch_shape[2] + 1) // zstep, patch_shape[0], patch_shape[1], patch_shape[2]),
                                                  (input.strides[0] * xstep, input.strides[1] * ystep,input.strides[2] * zstep, input.strides[0], input.strides[1],input.strides[2]))
    patches_3D= patches_3D.reshape(patches_3D.shape[0]*patches_3D.shape[1]*patches_3D.shape[2], patch_shape[0],patch_shape[1],patch_shape[2])
    return patches_3D


def stuff_3D_patches(patches,out_shape,xstep=24,ystep=24,zstep=24):
    """Stuff the processed 3D patches back to original shape
    
    Args:
        patches (): [description]
        out_shape ([type]): [description]
        xstep (int, optional): Defaults to 24. [description]
        ystep (int, optional): Defaults to 24. [description]
        zstep (int, optional): Defaults to 24. [description]
    
    Returns:
        [type]: [description]
    """
    out = np.zeros(out_shape, patches.dtype)
    denom = np.zeros(out_shape, patches.dtype)
    patch_shape = patches.shape[-3:]
    patches_6D = np.lib.stride_tricks.as_strided(out, ((out.shape[0] - patch_shape[0] + 1) // xstep, (out.shape[1] - patch_shape[1] + 1) // ystep,
                                                  (out.shape[2] - patch_shape[2] + 1) // zstep, patch_shape[0], patch_shape[1], patch_shape[2]),
                                                  (out.strides[0] * xstep, out.strides[1] * ystep,out.strides[2] * zstep, out.strides[0], out.strides[1],out.strides[2]))
    denom_6D = np.lib.stride_tricks.as_strided(denom, ((denom.shape[0] - patch_shape[0] + 1) // xstep, (denom.shape[1] - patch_shape[1] + 1) // ystep,
                                                  (denom.shape[2] - patch_shape[2] + 1) // zstep, patch_shape[0], patch_shape[1], patch_shape[2]),
                                                  (denom.strides[0] * xstep, denom.strides[1] * ystep,denom.strides[2] * zstep, denom.strides[0], denom.strides[1],denom.strides[2]))
    np.add.at(patches_6D, tuple(x.ravel() for x in np.indices(patches_6D.shape)), patches.ravel())
    np.add.at(denom_6D, tuple(x.ravel() for x in np.indices(patches_6D.shape)), 1)
    return out/denom


def get_patch_from_coord(volume, coord, window_size = 10):
    '''
    Extract a 3D patch from a volume with given coord
    Args:
        volume: 3D numpy array with dimension [z, x, y]
        coord: [z,x,y]
        window_size: voxels to keep around the centroid coordinates
    Returns:

    '''
    shape = volume.shape
    patch =   volume[coord[0] - window_size: coord[0] + window_size,
						   coord[1] - window_size: coord[1] + window_size,
						   coord[2] - window_size: coord[2] + window_size]		
    return patch


def rescale_volume(volume, old_dimension, new_dimension, order):
    '''
    Rescale a volume according to new spacing 
    Args:
        volume: 
        old_dimension: []
        new_dimension: [] 
        order: order of the spline interpolation
    '''
    target_shape = np.round(volume.shape * old_dimension / new_dimension)
    true_spacing = old_dimension * volume.shape / target_shape
    resize_factor = target_shape / volume.shape
    rescaled_volume = zoom(volume, resize_factor, mode = 'nearest',order=order)

    return rescaled_volume,true_spacing


def get_binary_volume(volume,spacing, intensity_threshold=-600, area_threshold=30, eccentricity_threshold=0.99,sigma=1):
    '''
    Generate a binary mask from a CT volume of the lungs, Thresholding based on the following criteria:
        area connected > 30mm^2
        eccentricity < 0.99 
    Args:
        volume:
        spacing:
        intensity_threshold:
        area_threshold:
        eccentricity_threshold
        sigma: 
    Output:

    '''
    binary_volume = np.zeros(volume.shape, dtype=bool)
    slice_shape = volume.shape[1]
    #Process 2D slices one by one
    for i in range(volume.shape[0]):
        #First do intensity threshold
       
        binary_slice = gaussian_filter(volume[i].astype('float32'), sigma, truncate=2.0) < intensity_threshold
        valid_label = []

        #Then do eccentricity threshold and area
        label = measure.label(binary_slice)
        properties = measure.regionprops(label)

        # x_axis = np.linspace(-label.shape[1]/2+0.5, label.shape[1]/2-0.5, label.shape[1]) * spacing[1]
        # y_axis = np.linspace(-label.shape[2]/2+0.5, label.shape[2]/2-0.5, label.shape[2]) * spacing[2]
        # x, y = np.meshgrid(x_axis, y_axis)
        # print(x,y)
        for prop in properties:
            if prop.area * spacing[1] * spacing[2] > area_threshold and prop.eccentricity < eccentricity_threshold:
                valid_label.append(prop.label)

        binary_slice = np.in1d(label,valid_label).reshape(label.shape)
        binary_volume[i] = binary_slice

    return binary_volume
