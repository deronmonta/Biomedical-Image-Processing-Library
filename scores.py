import numpy as np


def dice_score(prediction,ground_truth):
    '''
    Return the  dice score for a prediction
    Args:
        prediction, ground_truth: pytorch tensors or numpy array
    '''
    esp = 1e-8
    smooth = 1.
    try:
        prediction = prediction.flatten()
        ground_truth = ground_truth.flatten()
        prediction
        intersection = np.sum((prediction == ground_truth).astype(float))
        union = np.sum(np.sum(prediction != 0),np.sum(ground_truth != 0))
        return ((2. * intersection + smooth) / (union))
        
    except:
        predflat = prediction.contiguous().view(-1)#Stretch to 1d 
        gtflat = ground_truth.contiguous().view(-1)
        intersection = (predflat == gtflat).sum()
        intersection = torch.sum((predflat == gtflat).float())
        return ((2. * intersection + smooth) / (gtflat.sum() + predflat.sum() + smooth))