import torch
from pytorch_pretrained_gans import make_gan
from torchvision.utils import save_image, make_grid
import os
import random
import time
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Optional: ImageNet class labels (0-999)
IMAGENET_CLASSES = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
LABELS_PATH = "imagenet_classes.txt"

# Function to download class labels if needed
def download_labels():
    if not os.path.exists(LABELS_PATH):
        import urllib.request
        urllib.request.urlretrieve(IMAGENET_CLASSES, LABELS_PATH)

    with open(LABELS_PATH) as f:
        return [line.strip() for line in f.readlines()]

# Initialize BigGAN model
def initialize_gan():
    G = make_gan(gan_type='biggan')
    return G

# Generate image for a theme index
def generate_image_for_theme(G, theme_idx=None):
    z = G.sample_latent(batch_size=1)
    y = torch.zeros((1, 1000))

    if theme_idx is None:
        theme_idx = random.randint(0, 999)

    y[0][theme_idx] = 1
    img = G(z=z, y=y)
    return img, theme_idx

# Save and show image using matplotlib
def show_and_save_image(img, filename="generated_image.png", title="Generated Image"):
    os.makedirs("outputs", exist_ok=True)
    filepath = f"outputs/{filename}"
    save_image(img, filepath, normalize=True)
    print(f"✅ Image saved as {filepath}")

    # Convert tensor to image array
    img_array = img.squeeze().detach().cpu().numpy()
    img_array = np.transpose(img_array, (1, 2, 0))  # CHW to HWC

    plt.imshow(img_array)
    plt.title(title)
    plt.axis("off")
    plt.show()

# Generate multiple images at once and display a grid
def generate_multiple_images(G, num_images=4):
    images = []
    labels = []
    for _ in range(num_images):
        img, idx = generate_image_for_theme(G)
        images.append(img)
        labels.append(idx)
    grid = make_grid(torch.cat(images), nrow=2, normalize=True)
    timestamp = int(time.time())
    filename = f"grid_{timestamp}.png"
    save_image(grid, f"outputs/{filename}")
    print(f"🧩 Saved image grid as outputs/{filename}")
    # Show grid
    npimg = grid.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title("Image Grid (Random Classes)")
    plt.axis('off')
    plt.show()

def main():
    print("🔄 Initializing BigGAN model...")
    G = initialize_gan()
    print("✅ Model loaded successfully.\n")

    # Load class names
    download_labels()
    class_names = download_labels()

    # Ask for mode
    choice = input("Do you want to input a theme (class index), generate multiple, or random single? (input/multi/random): ").lower()

    if choice == "input":
        theme_idx = int(input("Enter class index for your theme (0-999): "))
        img, theme_idx = generate_image_for_theme(G, theme_idx)
        class_name = class_names[theme_idx] if 0 <= theme_idx < 1000 else "Unknown"
        show_and_save_image(img, f"theme_{theme_idx}_{class_name}.png", f"Theme: {class_name}")
    elif choice == "multi":
        num = int(input("How many images would you like to generate? (e.g., 4): "))
        generate_multiple_images(G, num)
    elif choice == "random":
        img, theme_idx = generate_image_for_theme(G)
        class_name = class_names[theme_idx] if 0 <= theme_idx < 1000 else "Unknown"
        print(f"🎲 Random theme index chosen: {theme_idx} ({class_name})")
        show_and_save_image(img, f"theme_{theme_idx}_{class_name}.png", f"Theme: {class_name}")
    else:
        print("Invalid choice. Exiting...")

if __name__ == "__main__":
    main()
