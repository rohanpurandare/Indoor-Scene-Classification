import torch
import torchvision.models as models
from data_loader import create_data_loaders

pretrained_model = models.resnet18(weights="IMAGENET1K_V1")
pretrained_model.fc = torch.nn.Identity()

for param in pretrained_model.parameters():
    param.requires_grad = False

train_loader, test_loader = create_data_loaders()

def extract_embeddings(model, dataloader):
    embeddings = []
    model.eval()
    with torch.no_grad():
        for images, _ in dataloader:
            images = torch.transpose(images, 1, 3).float()
            print(images.shape)
            features = model(images)
            embeddings.append(features)
    return torch.cat(embeddings)

embeddings = extract_embeddings(pretrained_model, train_loader)
torch.save(embeddings, 'train_embeddings.pt')
