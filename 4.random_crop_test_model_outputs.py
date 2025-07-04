import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from torchvision.models import resnet50, inception_v3
from sklearn.metrics import roc_auc_score
from PIL import Image
import math
from monai.transforms import (
    Resize,
    RandSpatialCrop,
    EnsureChannelFirst,
    Compose
)

def randCrop(volume_data, scale):
    num_slices = math.ceil(volume_data.shape[0])
    rand_crop = RandSpatialCrop(roi_size=(int(volume_data.shape[1] * scale), int(volume_data.shape[2] * scale)),
                                random_center=True, random_size=True)
    resize_transform = Resize(spatial_size=(volume_data.shape[1], volume_data.shape[2]))
    transforms = Compose([rand_crop, resize_transform])  # [rand_crop, resize_transform]  # [rand_crop]

    cropped_volumes = []
    ensure_channels_first = EnsureChannelFirst(channel_dim=-1)

    for i in range(num_slices):
        slice_data = volume_data[i, :, :, :]
        slice_tensor = ensure_channels_first(slice_data)
        transformed_slice = transforms(slice_tensor)
        cropped_volumes.append(transformed_slice)

    cropped_volumes_stack = np.stack(cropped_volumes, axis=0)

    cropped_volumes_final = np.moveaxis(cropped_volumes_stack, 1, -1)

    return cropped_volumes_final

def calculate_auc_and_predictions(loader, model, device):

    model.eval()
    all_labels = []
    all_probs = []
    all_image_ids = []
    with torch.no_grad():
        for idx, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            outputs = outputs.squeeze()
            probs = outputs.cpu().numpy()
            probs = np.round(probs, 3)
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

            batch_indices = list(range(idx * loader.batch_size, min((idx + 1) * loader.batch_size, len(loader.dataset))))
            all_image_ids.extend(loader.dataset.df.iloc[batch_indices]['image_id'])
    auc = roc_auc_score(all_labels, all_probs)
    auc = np.round(auc, 3)
    return auc, all_image_ids, all_labels, all_probs


def save_predictions_to_excel(image_ids, labels, predictions, original_df, output_file):

    results_df = pd.DataFrame({
        'image_id': image_ids,
        'true_label': labels,
        'predicted_prob': predictions
    })

    merged_df = pd.merge(original_df, results_df, left_on='image_id', right_on='image_id', how='inner')

    merged_df.to_excel(output_file, index=False)

class BoneTumorDataset(Dataset):
    def __init__(self, image_dir, df, transform=None):
        self.df = df
        self.transform = transform

        self.images = []
        for img_name in df['image_id']:
            img_path = os.path.join(image_dir, img_name)
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            self.images.append(image)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.df.iloc[idx]['label']
        return image, label

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

#"ResNet50"  "InceptionV3"  load pretrained weights from RadImgNet
class Backbone_ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = resnet50(pretrained=False)
        encoder_layers = list(base_model.children())
        self.backbone = nn.Sequential(*encoder_layers[:9])

    def forward(self, x):
        return self.backbone(x)
class Backbone_InceptionV3(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = inception_v3(pretrained=False, aux_logits=False)
        encoder_layers = list(base_model.children())
        self.backbone = nn.Sequential(*encoder_layers[:-1])

    def forward(self, x):
        return self.backbone(x)


def main(model_type, seed, output_dir, trained_model_path):
    torch.manual_seed(seed)
    np.random.seed(seed)
    batch_size = 64

    rj_image_dir = r'F:\bone_tumor\X-ray\RJ\jpgs_patches_RJ'
    btxrd_image_dir = r'F:\bone_tumor\X-ray\BTXRD\jpgs_patches_BTXRD'
    rj_excel_path = './dataset_RJ.xlsx'
    btxrd_excel_path = './dataset_tumor_BTXRD.xlsx'

    rj_df = pd.read_excel(rj_excel_path)
    btxrd_df = pd.read_excel(btxrd_excel_path)

    rj_df['image_id'] = rj_df['image_id'].apply(lambda x: f"{x}.jpg")
    btxrd_df['image_id'] = btxrd_df['image_id'].apply(lambda x: f"{x}.jpeg")

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=seed)
    train_idx, temp_idx = next(splitter.split(rj_df, rj_df['label']))
    train_df = rj_df.iloc[train_idx]
    temp_df = rj_df.iloc[temp_idx]

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_idx, test_idx = next(splitter.split(temp_df, temp_df['label']))
    val_df = temp_df.iloc[val_idx]
    internal_test_df = temp_df.iloc[test_idx]

    # BTXRD
    external_test_df = btxrd_df

    transform = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.ToTensor()
    ])

    if model_type == 'InceptionV3_ImageNet':
        transform = transforms.Compose([
            transforms.RandomCrop(224),
            transforms.Resize(299),
            transforms.ToTensor()
        ])

    model = models.resnet50(pretrained=False)
    if model_type == "ResNet50_ImageNet":
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1)
        model = nn.Sequential(model, nn.Sigmoid())
    elif model_type == "InceptionV3_ImageNet":
        model = models.inception_v3(pretrained=False)
        model.aux_logits = False
        model.fc = nn.Linear(model.fc.in_features, 1)
        model = nn.Sequential(model, nn.Sigmoid())
    elif model_type == "ResNet50_RadImgNet":
        backbone = Backbone_ResNet50()
        classifier = Classifier(num_class=1)
        model = nn.Sequential(backbone, classifier)
    elif model_type == "InceptionV3_RadImgNet":
        backbone = Backbone_InceptionV3()
        classifier = Classifier(num_class=1)
        model = nn.Sequential(backbone, classifier)

    model.load_state_dict(torch.load(trained_model_path, weights_only=True))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    model.eval()
    torch.set_grad_enabled(False)

    results = []

    # test 50 times for random crops
    for ind in range(50):
        internal_test_dataset = BoneTumorDataset(rj_image_dir, internal_test_df, transform=transform)
        external_test_dataset = BoneTumorDataset(btxrd_image_dir, external_test_df, transform=transform)

        internal_test_loader = DataLoader(internal_test_dataset, batch_size=batch_size, shuffle=False)
        external_test_loader = DataLoader(external_test_dataset, batch_size=batch_size, shuffle=False)

        internal_test_auc, internal_test_image_ids, internal_test_labels, internal_test_probs = calculate_auc_and_predictions(
            internal_test_loader, model, device)
        external_test_auc, external_test_image_ids, external_test_labels, external_test_probs = calculate_auc_and_predictions(
            external_test_loader, model, device)

        print(str(ind), f" - Internal Test AUC: {internal_test_auc:.3f}, "f"External Test AUC: {external_test_auc:.3f}")

        results.append([ind, round(internal_test_auc, 3), round(external_test_auc, 3)])

    df = pd.DataFrame(results, columns=['ind', 'internal_test_auc', 'external_test_auc'])
    df.to_excel(output_dir, index=False, engine='openpyxl')

    internal_test_auc_mean = df['internal_test_auc'].mean().round(3)
    external_test_auc_mean = df['external_test_auc'].mean().round(3)

    print(f"Mean Internal Test AUC: {internal_test_auc_mean}", f"Mean External Test AUC: {external_test_auc_mean}")

    print(f"data successfully write in {output_dir}")

if __name__ == "__main__":
    time = "250416-1730"
    seed = 666

    model_dict = {}
    model_dict['ResNet50_RadImgNet'] = "/ResNet50_RadImgNet_trained.pt"
    model_dict['ResNet50_ImageNet'] = '/ResNet50_ImageNet_trained.pt'
    model_dict['InceptionV3_RadImgNet'] = "/InceptionV3_RadImgNet_trained.pt"
    model_dict['InceptionV3_ImageNet'] = '/InceptionV3_ImageNet_trained.pt'

    for model_type, model_path in model_dict.items():
        print(f"Model Type: {model_type}, Model Path: {model_path}")
        output_file = "./random_crop_test_" + model_type + ".xlsx"
        main(model_type, seed, output_file, model_path)
        torch.cuda.empty_cache()