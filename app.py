import torch
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models
import matplotlib.pyplot as plt

# Set device to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the content and style images, ensure they are moved to the GPU
def load_image(image_path, max_size=600):  # Increase image resolution for more details
    image = Image.open(image_path)
    
    # Transform image to Tensor
    transform = transforms.Compose([
        transforms.Resize((max_size, max_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)  # Ensure the image is moved to GPU

# Load the content and style images
content_image = load_image(r"C:\Users\litin\OneDrive\Desktop\Prodigy_Lijin\Task 5\images\download.jpeg")
style_image = load_image(r"C:\Users\litin\OneDrive\Desktop\Prodigy_Lijin\Task 5\images\images (3).jpeg")

# Load VGG19 Model and move it to GPU
vgg = models.vgg19(pretrained=True).features.to(device)

# Freeze the parameters (we don't need to update the model's weights)
for param in vgg.parameters():
    param.requires_grad_(False)

# Function to extract features from the layers of the VGG model
def get_features(image, model):
    layers = {'0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1', '19': 'conv4_1', '21': 'conv4_2', '28': 'conv5_1'}
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

# Calculate the Gram Matrix for the style loss
def gram_matrix(tensor):
    _, d, h, w = tensor.size()  # Get dimensions
    tensor = tensor.view(d, h * w)  # Reshape to d x (h*w)
    gram = torch.mm(tensor, tensor.t())  # Calculate Gram matrix
    return gram

# Neural Style Transfer Function
def run_style_transfer(content_img, style_img, vgg, num_steps=500, style_weight=1e6, content_weight=1e0, learning_rate=0.002):
    # Create a clone of the content image and make it require gradients
    input_img = content_img.clone().requires_grad_(True).to(device)
    
    # Optimizer to update input_img
    optimizer = optim.Adam([input_img], lr=learning_rate)
    
    # Extract features from the content and style images
    content_features = get_features(content_img, vgg)
    style_features = get_features(style_img, vgg)
    
    # Calculate Gram matrices for the style image
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
    
    for i in range(num_steps):
        optimizer.zero_grad()

        # Extract features from the input image
        input_features = get_features(input_img, vgg)

        # Calculate content loss (focus on maintaining content structure)
        content_loss = torch.mean((input_features['conv4_2'] - content_features['conv4_2']) ** 2)

        # Calculate style loss
        style_loss = 0
        for layer in style_features:
            input_feature = input_features[layer]
            input_gram = gram_matrix(input_feature)
            style_gram = style_grams[layer]
            layer_style_loss = torch.mean((input_gram - style_gram) ** 2)
            style_loss += layer_style_loss / input_feature.numel()

        # Total loss
        total_loss = content_weight * content_loss + style_weight * style_loss
        
        # Backpropagate and optimize
        total_loss.backward()
        optimizer.step()
        
        if i % 50 == 0:  # Print every 50 steps
            print(f"Step {i}, Total loss: {total_loss.item()}")
            
    return input_img

# Run the style transfer with more steps for better quality
output_image = run_style_transfer(content_image, style_image, vgg, num_steps=500, style_weight=1e6, content_weight=1e0)

# Convert output to image and save it
def save_image(tensor, path):
    image = tensor.clone().detach().cpu().squeeze(0)  # Move to CPU before saving
    image = transforms.ToPILImage()(image)
    image.save(path)

save_image(output_image, "output_image.jpg")
