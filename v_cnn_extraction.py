import torch
from torchvision.io import decode_image


def create_feature_extractor(model):
    """
    Create a feature extractor by removing the final fully connected layer from the model.
    
    Args:
        model (torch.nn.Module): Pre-trained torch model for feature extraction (i.e. ResNet50).

    Returns:
        feature_extractor (torch.nn.Module): Model without the final fully connected layer.
    """
    # Keep all layers except the final fully connected layer
    modules = list(model.children())[:-1]
    feature_extractor = torch.nn.Sequential(*modules)
    
    return feature_extractor


def preprocess_image(image_path, weights):
    """
    Preprocess an image for feature extraction using the specified weights.
    
    Args:
        image_path (str): Path to the image file.
        weights (torchvision.models.Weights): Weights object for the pre-trained torchvision model.
    
    Returns:
        img_tensor (torch.Tensor): Preprocessed image tensor ready for feature extraction.
    """
    # Load image
    img = decode_image(image_path)

    # Initialize the inference transforms
    preprocess = weights.transforms()

    # Apply inference preprocessing transforms
    img_tensor = preprocess(img).unsqueeze(0)
    
    return img_tensor


def get_v_cnn(image, feature_extractor):
    """
    Extract visual features from an image using a pre-trained model (i.e. ResNet50) from torchvision.
    
    Args:
        image (torch.Tensor): Preprocessed image tensor.
        feature_extractor (torch.nn.Module): Pre-trained model for feature extraction.

    Returns:
        v_cnn (torch.Tensor): Extracted visual features as a tensor.
    """
    # Set model to eval mode
    feature_extractor.eval()

    with torch.no_grad():
        # Extract features
        v_cnn = feature_extractor(image)

        # Remove unnecessary dimensions
        v_cnn = v_cnn.flatten()
    
        return v_cnn

