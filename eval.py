import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import parser
import logging
from os.path import join
from datetime import datetime
import test_FoL
import util
import commons
import datasets
import network_FoL
import warnings
warnings.filterwarnings("ignore")

######################################### SETUP #########################################
args = parser.parse_arguments()
start_time = datetime.now()
args.save_dir = join("test", args.save_dir, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
commons.setup_logging(args.save_dir, console="info")
commons.make_deterministic(args.seed)
args.features_dim = 8448
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")

######################################### MODEL #########################################
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = network_FoL.FoLNet()
model = model.to(args.device)
if args.resume is not None:
    logging.info(f"Resuming model from {args.resume}")
    model = util.resume_model(args, model)
model = torch.nn.DataParallel(model)

# ######################################### TEST on TEST SET #########################################
for dataset_name in args.dataset_names:
    test_ds = datasets.BaseDataset(args, args.eval_datasets_folder, dataset_name, "test")
    logging.info(f"Test set: {test_ds}")
    recalls, recalls_str, recalls_rerank, recalls_str_rerank = test_FoL.test(args, test_ds, model, args.test_method)
    if isinstance(recalls_str, dict):
        for query_set_name in recalls_str:
            logging.info(f"Recalls on {query_set_name}: {recalls_str[query_set_name]}")
            logging.info(f"Reranking recalls on {query_set_name}: {recalls_str_rerank[query_set_name]}")
    else:
        logging.info(f"Recalls on test set {test_ds}: {recalls_str}")
        logging.info(f"Reranking recalls on test set {test_ds}: {recalls_str_rerank}")
    logging.info(f"Finished in {str(datetime.now() - start_time)[:-7]}")
