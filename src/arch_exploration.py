# from modified_mobilenet_gelu2 import MobNet_StartAt_Layer8

# # Create an instance of your model
# model = MobNet_StartAt_Layer8(num_classes=10, pretrain_classes=10)

# # Print the model architecture and the classifier layers
# print(model)

# # Optionally, print the classifier layers explicitly
# for idx, layer in enumerate(model.model.classifier):
#     print(f"Classifier Layer {idx}: {layer}")
import torch
import mobilenet_modified_swav as mobilenet_models

# Create an instance of your model
model = mobilenet_models.__dict__["mobilenet_v3_large"](pretrained=False)

# Print the model architecture
print(model)
# Set the model to evaluation mode
# Sample input tensor (adjust the shape and values as needed)
inp = torch.randn(1, 3, 224, 224)  # Batch size of 1, 3 channels, 224x224 image

# Perform a forward pass through the feature extraction part
with torch.no_grad():
    raw_features = model.forward_backbone(inp)

# # Access the output from the forward_head function
# in_features = output[0]
# out_features = output[1]
# # Print the shape of the output from the forward_head function
# print("in_features Shape:", in_features.shape)
# print("out_features Shape:", out_features.shape)
print("raw_features", raw_features.shape)
