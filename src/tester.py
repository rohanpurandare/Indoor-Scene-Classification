from data_loader import create_data_loaders
import os

print("Current working directory:", os.getcwd())


train_loader, test_loader = create_data_loaders()
print("Train DataLoader:")
print(f"Number of batches: {len(train_loader)}")
print(f"Number of total samples: {len(train_loader.dataset)}")

print("\nTest DataLoader:")
print(f"Number of batches: {len(test_loader)}")
print(f"Number of total samples: {len(test_loader.dataset)}")

image, label = next(iter(train_loader))
print("\nImage shape:", image.shape)
print("Label:", label)
