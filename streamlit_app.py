import streamlit as st
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz

# Initialize the ResNet50 model
model = resnet50()

# Load the weights from the .pth file in the repository
weights_path = "resnet50-0676ba61.pth"
model.load_state_dict(torch.load(weights_path))
model.eval()

# Streamlit app
st.title("ResNet-50 Explanations with Captum")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image for ResNet50
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(image).unsqueeze(0)
    input_tensor.requires_grad_()

    # Predict with ResNet50
    predictions = model(input_tensor)

    # Get the prediction class
    prediction_class = torch.argmax(predictions).item()
    st.write(f"Predicted Class: {prediction_class}")

    # Explain with Captum
    ig = IntegratedGradients(model)
    attr, _ = ig.attribute(input_tensor, target=prediction_class, return_convergence_delta=True)

    # Visualize the explanations
    attr = np.transpose(attr.squeeze().cpu().detach().numpy(), (1,2,0))
    st.image(attr, caption="Integrated Gradients Explanation", use_column_width=True)

st.write("Upload an image to get started!")
