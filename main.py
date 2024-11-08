import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
import h5py

# Load the UMIST dataset from .mat file
mat_file_path = 'umist_cropped.mat'
mat_data = scipy.io.loadmat(mat_file_path)

# Extract facedat and dirnames
facedat = mat_data['facedat'][0]
dirnames = mat_data['dirnames'][0]

# Create lists to store processed images and labels 
images = []
labels = []

# Process each person's data
for person_idx, person_data in enumerate(facedat):
    for image in person_data:
        if isinstance(image, np.ndarray) and image.size > 0:
            images.append(image)
            labels.append(person_idx)

# Convert labels to numpy array
labels = np.array(labels)

# First split: temporary train (85%) and test (15%)
indices = np.arange(len(images))
idx_temp_train, idx_test, y_temp_train, y_test = train_test_split(
    indices, 
    labels,
    test_size=0.15,
    stratify=labels,
    random_state=42
)

# Second split: final train (70%) and validation (15%)
idx_train, idx_val, y_train, y_val = train_test_split(
    idx_temp_train,
    y_temp_train,
    test_size=0.176,
    stratify=y_temp_train,
    random_state=42
)

# Create the split datasets using indices
X_train = [images[i] for i in idx_train]
X_val = [images[i] for i in idx_val]
X_test = [images[i] for i in idx_test]

print("\nDataset Split Sizes:")
print(f"Training set: {len(X_train)} images ({len(X_train)/len(images)*100:.1f}%)")
print(f"Validation set: {len(X_val)} images ({len(X_val)/len(images)*100:.1f}%)")
print(f"Test set: {len(X_test)} images ({len(X_test)/len(images)*100:.1f}%)")

# Save splits in a way that handles varying dimensions
def save_split(images, labels, prefix):
    # Find the maximum width to pad all images to the same size
    max_width = max(img.shape[1] for img in images)
    height = images[0].shape[0]  # All images have same height (92)
    
    # Create padded array
    padded_images = np.zeros((len(images), height, max_width))
    
    # Fill the padded array
    for i, img in enumerate(images):
        w = img.shape[1]
        padded_images[i, :, :w] = img
    
    # Save as compressed npz file
    np.savez_compressed(
        f'umist_split_{prefix}.npz',
        images=padded_images,
        labels=labels,
        original_widths=[img.shape[1] for img in images]  # Save original widths for later
    )

# Save all splits
save_split(X_train, y_train, 'train')
save_split(X_val, y_val, 'val')
save_split(X_test, y_test, 'test')

print("\nSaved splits successfully!")
print("Files created:")
print("- umist_split_train.npz")
print("- umist_split_val.npz")
print("- umist_split_test.npz")

# Function to load and verify the saved data
def load_and_verify_split(prefix):
    data = np.load(f'umist_split_{prefix}.npz')
    print(f"\nVerifying {prefix} split:")
    print(f"Images shape: {data['images'].shape}")
    print(f"Labels shape: {data['labels'].shape}")
    print(f"Number of original widths: {len(data['original_widths'])}")
    return data

# Verify saved data
train_data = load_and_verify_split('train')
val_data = load_and_verify_split('val')
test_data = load_and_verify_split('test')