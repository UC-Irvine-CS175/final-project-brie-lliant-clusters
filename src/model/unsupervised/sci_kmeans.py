import os
import random
import numpy as np
import pyprojroot
root = pyprojroot.find_root(pyprojroot.has_dir(".git"))
import sys
sys.path.append(str(root))

import pandas as pd
import cv2
from tqdm import tqdm

from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow

from PIL import Image

import torch
from dataclasses import dataclass
from torchvision.models import resnet
from torchvision.models.resnet import Bottleneck
from torch.hub import load_state_dict_from_url
from mpl_toolkits.mplot3d import Axes3D
import wandb

from resnet101 import BPSConfig
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from kmeans_pytorch import kmeans as km

from src.dataset.bps_datamodule import BPSDataModule
from src.dataset.augmentation import(
    NormalizeBPS,
    ResizeBPS,
    ToTensor
)
from src.vis_utils import(
    plot_2D_scatter_plot,
    plot_3D_scatter_plot,
)
from pca_tsne import(
    preprocess_images,
    perform_pca,
    perform_tsne,
    create_tsne_cp_df,
)


def main():
    # Initialize a BPSConfig object
    config = BPSConfig()

    # Fix random seed for reproducibility 
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Instantiate BPSDataModule object
    bps_datamodule = BPSDataModule(train_csv_file=config.train_meta_fname, 
                                   train_dir=config.data_dir,
                                   val_csv_file=config.val_meta_fname,
                                   val_dir=config.data_dir,
                                   resize_dims=(64, 64),
                                   batch_size=config.batch_size,
                                   num_workers=config.num_workers)
    
    IMAGE_SHAPE = (64, 64)
    N_ROWS = 5
    N_COLS = 7
    N_COMPONENTS = N_ROWS * N_COLS

    # Set stage to validation
    bps_datamodule.setup(stage='train')

    # Preprocess the images 
    image_stream_1d, all_labels = preprocess_images(lt_datamodule=bps_datamodule.train_dataloader())

    # all_labels = true label of each image
    # Fits PCA to X_flat and performs dimensionality reduction on the data (4096 -> 35)
    pca, X_pca = perform_pca(X_flat=image_stream_1d, n_components = N_COMPONENTS)

    # Create DataFrame based off of dimensionally reduced array
    pca_df = pd.DataFrame(X_pca)

    # Standardize data
    scaler = StandardScaler()
    pca_df.columns = pca_df.columns.astype(str)
    scaler.fit(pca_df)
    X_scaled = scaler.transform(pca_df)
    pca_df_scaled = pd.DataFrame(X_scaled, columns=pca_df.columns)
    print(pca_df_scaled.head())

    # Perform kmeans
    kmeans = KMeans(n_clusters=2, random_state=42)
    pca_labels = kmeans.fit_predict(X_pca)
    pca_df_scaled['cluster'] = pca_labels

    # Perform tsne 
    pca_np = pca_df_scaled.to_numpy()
    n_components = 2
    pca_tsne_2d = perform_tsne(X_reduced_dim=pca_np, n_components=n_components)
    pca_tsne_2d_df = create_tsne_cp_df(X_tsne=pca_tsne_2d, labels=all_labels, num_points=1000)

    # 2D plot
    plot_2D_scatter_plot(pca_tsne_2d_df, "2D Plot for Training")   

    # 3D plot
    n_components = 3
    pca_tsne_3d = perform_tsne(X_reduced_dim=pca_np, n_components=n_components)
    pca_tsne_3d_df = create_tsne_cp_df(X_tsne=pca_tsne_3d, labels=all_labels, num_points=1000)
    plot_3D_scatter_plot(pca_tsne_3d_df, "3D Plot for Training")

    return

if __name__ == "__main__":
    main()