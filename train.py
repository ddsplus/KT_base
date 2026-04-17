import os
import argparse
import json
import pickle

import torch
import random
import numpy as np

from torch.utils.data import DataLoader, random_split
from torch.optim import SGD, Adam

from data_loaders.assist2009 import ASSIST2009
from data_loaders.assist2015 import ASSIST2015
from data_loaders.algebra2005 import Algebra2005
from data_loaders.statics2011 import Statics2011
from models.dkt import DKT
from models.dkt_plus import DKTPlus
from models.dkvmn import DKVMN
from models.sakt import SAKT
from models.gkt import PAM, MHA
from models.utils import collate_fn


def _create_subdata_pickles(dataset_dir, seq_len, num_users=200, num_q=100):
    os.makedirs(dataset_dir, exist_ok=True)

    q_list = np.arange(num_q)
    u_list = np.arange(num_users)

    q_seqs = []
    r_seqs = []

    for _ in range(num_users):
        L = random.randint(5, max(6, seq_len // 2))
        q_seq = np.random.randint(0, num_q, size=L)
        r_seq = np.random.binomial(1, 0.5, size=L)

        q_seqs.append(q_seq)
        r_seqs.append(r_seq)

    q2idx = {int(q): int(i) for i, q in enumerate(q_list)}
    u2idx = {int(u): int(i) for i, u in enumerate(u_list)}

    with open(os.path.join(dataset_dir, "q_seqs.pkl"), "wb") as f:
        pickle.dump(q_seqs, f)
    with open(os.path.join(dataset_dir, "r_seqs.pkl"), "wb") as f:
        pickle.dump(r_seqs, f)
    with open(os.path.join(dataset_dir, "q_list.pkl"), "wb") as f:
        pickle.dump(q_list, f)
    with open(os.path.join(dataset_dir, "u_list.pkl"), "wb") as f:
        pickle.dump(u_list, f)
    with open(os.path.join(dataset_dir, "q2idx.pkl"), "wb") as f:
        pickle.dump(q2idx, f)
    with open(os.path.join(dataset_dir, "u2idx.pkl"), "wb") as f:
        pickle.dump(u2idx, f)


def main(model_name, dataset_name, subdata_dir=None):
    if not os.path.isdir("ckpts"):
        os.mkdir("ckpts")

    ckpt_path = os.path.join("ckpts", model_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    ckpt_path = os.path.join(ckpt_path, dataset_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    with open("config.json") as f:
        config = json.load(f)
        model_config = config[model_name]
        train_config = config["train_config"]

    batch_size = train_config["batch_size"]
    num_epochs = train_config["num_epochs"]
    train_ratio = train_config["train_ratio"]
    learning_rate = train_config["learning_rate"]
    optimizer = train_config["optimizer"]  # can be [sgd, adam]
    seq_len = train_config["seq_len"]

    # prefer explicit subdata directory if provided, otherwise try ./subdata/<dataset_name>
    dataset_dir_arg = None
    if subdata_dir:
        # allow passing either a root that contains dataset dirs or a dataset-specific path
        candidate = os.path.join(subdata_dir, dataset_name)
        if os.path.isdir(candidate):
            dataset_dir_arg = candidate
        elif os.path.isdir(subdata_dir):
            dataset_dir_arg = subdata_dir
    else:
        candidate = os.path.join("subdata", dataset_name)
        if os.path.isdir(candidate):
            dataset_dir_arg = candidate

    # instantiate dataset, with fallback to creating minimal subdata pickles when files missing
    try:
        if dataset_name == "ASSIST2009":
            dataset = ASSIST2009(seq_len, dataset_dir=dataset_dir_arg) if dataset_dir_arg else ASSIST2009(seq_len)
        elif dataset_name == "ASSIST2015":
            dataset = ASSIST2015(seq_len, dataset_dir=dataset_dir_arg) if dataset_dir_arg else ASSIST2015(seq_len)
        elif dataset_name == "Algebra2005":
            dataset = Algebra2005(seq_len, dataset_dir=dataset_dir_arg) if dataset_dir_arg else Algebra2005(seq_len)
        elif dataset_name == "Statics2011":
            dataset = Statics2011(seq_len, datset_dir=dataset_dir_arg) if dataset_dir_arg else Statics2011(seq_len)
    except FileNotFoundError:
        # create tiny synthetic subdata on disk and retry (split/indexing will be done on CPU)
        fallback_dir = os.path.join("subdata", dataset_name)
        print(f"Dataset files not found. Creating fallback subdata at {fallback_dir}")
        _create_subdata_pickles(fallback_dir, seq_len)

        if dataset_name == "ASSIST2009":
            dataset = ASSIST2009(seq_len, dataset_dir=fallback_dir)
        elif dataset_name == "ASSIST2015":
            dataset = ASSIST2015(seq_len, dataset_dir=fallback_dir)
        elif dataset_name == "Algebra2005":
            dataset = Algebra2005(seq_len, dataset_dir=fallback_dir)
        elif dataset_name == "Statics2011":
            dataset = Statics2011(seq_len, datset_dir=fallback_dir)

    # use CPU for data split operations, train on cuda:0 if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(ckpt_path, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=4)
    with open(os.path.join(ckpt_path, "train_config.json"), "w") as f:
        json.dump(train_config, f, indent=4)

    if model_name == "dkt":
        model = DKT(dataset.num_q, **model_config).to(device)
    elif model_name == "dkt+":
        model = DKTPlus(dataset.num_q, **model_config).to(device)
    elif model_name == "dkvmn":
        model = DKVMN(dataset.num_q, **model_config).to(device)
    elif model_name == "sakt":
        model = SAKT(dataset.num_q, **model_config).to(device)
    elif model_name == "gkt":
        if model_config["method"] == "PAM":
            model = PAM(dataset.num_q, **model_config).to(device)
        elif model_config["method"] == "MHA":
            model = MHA(dataset.num_q, **model_config).to(device)
    else:
        print("The wrong model name was used...")
        return

    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size

    # ensure splitting uses CPU generator
    gen = torch.Generator(device="cpu")
    gen.manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=gen)

    if os.path.exists(os.path.join(dataset.dataset_dir, "train_indices.pkl")):
        with open(
            os.path.join(dataset.dataset_dir, "train_indices.pkl"), "rb"
        ) as f:
            train_dataset.indices = pickle.load(f)
        with open(
            os.path.join(dataset.dataset_dir, "test_indices.pkl"), "rb"
        ) as f:
            test_dataset.indices = pickle.load(f)
    else:
        with open(
            os.path.join(dataset.dataset_dir, "train_indices.pkl"), "wb"
        ) as f:
            pickle.dump(train_dataset.indices, f)
        with open(
            os.path.join(dataset.dataset_dir, "test_indices.pkl"), "wb"
        ) as f:
            pickle.dump(test_dataset.indices, f)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=test_size, shuffle=True,
        collate_fn=collate_fn
    )

    if optimizer == "sgd":
        opt = SGD(model.parameters(), learning_rate, momentum=0.9)
    elif optimizer == "adam":
        opt = Adam(model.parameters(), learning_rate)

    aucs, loss_means = \
        model.train_model(
            train_loader, test_loader, num_epochs, opt, ckpt_path
        )

    with open(os.path.join(ckpt_path, "aucs.pkl"), "wb") as f:
        pickle.dump(aucs, f)
    with open(os.path.join(ckpt_path, "loss_means.pkl"), "wb") as f:
        pickle.dump(loss_means, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="dkt",
        help="The name of the model to train. \
            The possible models are in [dkt, dkt+, dkvmn, sakt, gkt]. \
            The default model is dkt."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ASSIST2009",
        help="The name of the dataset to use in training. \
            The possible datasets are in \
            [ASSIST2009, ASSIST2015, Algebra2005, Statics2011]. \
            The default dataset is ASSIST2009."
    )
    parser.add_argument(
        "--subdata_dir",
        type=str,
        default=None,
        help="Optional path to a subdata directory or a root that contains per-dataset subfolders."
    )
    args = parser.parse_args()

    main(args.model_name, args.dataset_name, subdata_dir=args.subdata_dir)
