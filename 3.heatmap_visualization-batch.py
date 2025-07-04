import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from torchvision.models import resnet50
import os
import pandas as pd

class Classifier(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.drop_out = nn.Dropout(p=0.5)
        self.linear = nn.Linear(2048, num_class)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.drop_out(x)
        x = self.linear(x)
        x = self.Sigmoid(x)
        return x

#"ResNet50"
class Backbone_ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = resnet50(pretrained=False)
        encoder_layers = list(base_model.children())
        self.backbone = nn.Sequential(*encoder_layers[:9])

    def forward(self, x):
        return self.backbone(x)

def register_hooks(model, target_layer_name):
    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations['value'] = output

    def backward_hook(module, grad_input, grad_output):
        gradients['value'] = grad_output[0]

    # 获取目标层
    target_layer = dict([*model.named_modules()])[target_layer_name]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)

    return activations, gradients

def overlay_heatmap(image, heatmap_in, alpha=0.5):
    heatmap = np.uint8(255 * heatmap_in)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, code=cv2.COLOR_BGR2RGB)

    image = np.array(image)
    if image.shape[-1] == 4:
        image = image[:, :, :3]

    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    superimposed_img = heatmap * alpha + image
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    return Image.fromarray(superimposed_img)

def save_images_with_gradcam(excel_path, image_dir, output_dir, model, target_layer_name, transform, device):

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_excel(excel_path)

    for _, row in df.iterrows():
        image_id = row['image_id']
        predicted_prob = row['predicted_prob']

        image_path = os.path.join(image_dir, image_id)

        if not os.path.exists(image_path):
            print(f"Image {image_id} not found, skipping...")
            continue

        image = Image.open(image_path).convert("RGB")
        transformed_image = transform(image).unsqueeze(0).to(device)

        activations, gradients = register_hooks(model, target_layer_name)

        model.eval()
        output = model(transformed_image)

        model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][0] = 1
        output.backward(gradient=one_hot)

        activations_value = activations['value'].cpu().data.numpy()[0]
        gradients_value = gradients['value'].cpu().data.numpy()[0]

        weights = np.mean(gradients_value, axis=(1, 2))
        cam = np.zeros(activations_value.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations_value[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (image.width, image.height))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        result_image = overlay_heatmap(image, cam)

        combined_image = Image.new('RGB', (image.width * 2, image.height))
        combined_image.paste(image, (0, 0))
        combined_image.paste(result_image, (image.width, 0))

        predicted_prob_percent = f"{predicted_prob * 100:.1f}%"

        output_filename = f"{os.path.splitext(image_id)[0]}_{predicted_prob_percent}.jpg"
        output_path = os.path.join(output_dir, output_filename)

        combined_image.save(output_path)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    backbone = Backbone_ResNet50()
    classifier = Classifier(num_class=1)
    model = nn.Sequential(backbone, classifier)
    model.load_state_dict(torch.load("/ywj/data/xiawei/bone_tumor_adjust/250416-1141_ResNet50_RadImgNet/best_model_seed_666_lr_0.004_bs_128_epc39_inAUC_0.785_exAUC_0.738.pt",
                                     weights_only=True))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    model.eval()

    target_layer_name = "0.backbone.7.2.conv3"

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = "external_test" # internal_test

    folder = ""

    if dataset == "internal_test":
        folder = "jpgs_patches_RJ-adjust"
    if dataset == "external_test":
        folder = "jpgs_patches_BTXRD"

    excel_path = "./ResNet50_RadImgNet/" + dataset + "_predictions.xlsx"
    image_dir = "/data/bone_tumor/" + folder + "/"
    output_dir = "data/bone_tumor/" + folder + "-gradcam-" + dataset + "/"
    save_images_with_gradcam(excel_path, image_dir, output_dir, model, target_layer_name, transform, device)
