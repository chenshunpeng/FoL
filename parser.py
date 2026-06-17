# -*- coding: UTF-8 -*-
import os
import torch
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Benchmarking Visual Geolocalization",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Training parameters
    parser.add_argument("--freeze", type=int, default=0)
    parser.add_argument("--train_batch_size", type=int, default=60,
                        help="Number of triplets (query, pos, negs) in a batch. Each triplet consists of 12 images")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.0001, help="_")
    parser.add_argument("--optim", type=str, default="adam", help="_", choices=["adam", "sgd", "adamw"])
    parser.add_argument("--epochs_num", type=int, default=5,
                        help="number of epochs to train for")
    parser.add_argument("--infer_batch_size", type=int, default=16,
                        help="Batch size for inference (caching and testing)")
    # Model parameters
    parser.add_argument('--fc_output_dim', type=int, default=None,
                        help="Output dimension of fully connected layer. If None, don't use a fully connected layer.")
    # Initialization parameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to load checkpoint from, for resuming training or testing.")
    # Other parameters
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num_workers", type=int, default=4, help="num_workers for all dataloaders")
    parser.add_argument('--resize', type=int, default=[504, 504], nargs=2, help="Resizing shape for images (HxW).")
    parser.add_argument('--test_method', type=str, default="hard_resize",
                        choices=["hard_resize", "single_query", "central_crop", "five_crops", "nearest_crop", "maj_voting"],
                        help="This includes pre/post-processing methods and prediction refinement")
    parser.add_argument("--majority_weight", type=float, default=0.01, 
                        help="only for majority voting, scale factor, the higher it is the more importance is given to agreement")
    parser.add_argument(
        "--efficient_ram_testing",
        action="store_true",
        help=("Use exact float32 retrieval with a chunked database and disk-backed "
              "local features. This reduces RAM usage without approximate search."),
    )
    parser.add_argument(
        "--efficient_ram_cache_dir",
        type=str,
        default="/data_nvme/VPR/FoL",
        help="Directory used for temporary local-feature caches in low-memory mode.",
    )
    parser.add_argument(
        "--efficient_ram_max_cache_gib",
        type=float,
        default=500.0,
        help="Hard upper limit for one low-memory evaluation cache, in GiB.",
    )
    parser.add_argument(
        "--efficient_ram_min_free_gib",
        type=float,
        default=500.0,
        help="Minimum free space that must remain on the cache filesystem, in GiB.",
    )
    parser.add_argument("--val_positive_dist_threshold", type=int, default=25, help="_")
    parser.add_argument("--train_positives_dist_threshold", type=int, default=10, help="_")
    parser.add_argument('--recall_values', type=int, default=[1, 5, 10, 25, 100], nargs="+",
                        help="Recalls to be computed, such as R@5.")
    # Paths parameters
    parser.add_argument("--eval_datasets_folder", type=str, default=None, help="Path with all datasets")
    parser.add_argument("--eval_dataset_name", type=str, default="pitts30k", help="Relative path of the dataset")
    parser.add_argument("--pca_dataset_folder", type=str, default=None,
                        help="Path with images to be used to compute PCA (ie: pitts30k/images/train")
    parser.add_argument("--save_dir", type=str, default="default",
                        help="Folder name of the current run (saved in ./logs/)")
    parser.add_argument("--dataset_names", type=str, nargs='+', help="Names of datasets to test")
    parser.add_argument("--dataset_name", type=str, help="Names of datasets to test")

    parser.add_argument("--log_dir", type=str, default="default",
                        help="experiment name, output logs will be saved under logs/log_dir")

    args = parser.parse_args()
    
    if args.eval_datasets_folder == None:
        try:
            args.eval_datasets_folder = os.environ['DATASETS_FOLDER']
        except KeyError:
            raise Exception("You should set the parameter --datasets_folder or export " +
                            "the DATASETS_FOLDER environment variable as such \n" +
                            "export DATASETS_FOLDER=../datasets_vg/datasets")
    
    return args
