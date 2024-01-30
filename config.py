
# model_save_path = "/content/model"
model_save_path = "/content/drive/MyDrive/Colab_models/Brain_MRI_image_synthesis/DCGAN_brain_mri_trained"

# Root directory for the dataset
data_root = 'data/brain_mri'

# Path to folder with individual images
img_folder = f'{data_root}/brain_mri_combined'

# Path to download the dataset to
download_path = f'{data_root}/Brain_MRI.zip'

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

