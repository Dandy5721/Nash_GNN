import os
import numpy as np
from typing import List
from torch_sparse import SparseTensor
from tqdm import tqdm as core_tqdm
import torch
import random
import codecs

import math
import random
import torch
from torch_geometric.utils import to_undirected, k_hop_subgraph
import torch_geometric.transforms as T


def prune_graph(adj_test, target_idx, k):
    adj_test = adj_test.cpu()
    u, v, _ = adj_test.coo()
    _, edge_index, __, ___ = k_hop_subgraph(target_idx,k,torch.stack((u,v),dim=0))
    graph_size = torch.Size((adj_test.size(0),adj_test.size(1)))
    new_adj_test = SparseTensor(row=edge_index[1], col=edge_index[0], value=None, sparse_sizes=graph_size,is_sorted=True).to_symmetric()
    return new_adj_test

def target_select(model,adj,features,labels,target_idx,num):
    # num highest margin
    # num lowest margin
    # 2num random
    # with torch.no_grad():
    #     pred = model(features,adj)[target_idx]
    #     pred_y = pred.argmax(-1)
    # correct_idx = labels[target_idx].view(-1)==pred_y.view(-1)
    # assert len(correct_idx) >= 4*num
    # pred_max = pred.max(-1)[0]
    # second_y =  pred
    # second_y[torch.arange(pred_y.size(0)),pred_y] = -1e9
    # margin = (pred_max-second_y.max(-1)[0])[correct_idx]
    # margin_max = margin.argsort()
    # random_ids = torch.randperm(len(margin)-2*num)[:2*num]
    # selected_ids = torch.cat((margin_max[:num],margin_max[-num:],margin_max[num:-num][random_ids]),dim=0)


    # sanity check
    with torch.no_grad():
        pred = model(features,adj)[target_idx]
        pred_y = pred.argmax(-1)
    pred_sort, _ = pred.sort(-1,descending=True)
    correct_idx = labels[target_idx].view(-1)==pred_y.view(-1)
    print(f"Correctly classified nodes: {correct_idx.sum()}")
    new_margin = pred_sort[correct_idx,0]-pred_sort[correct_idx,1]
    new_margin_max = new_margin.argsort()
    random_ids = torch.randperm(len(new_margin)-2*num)[:2*num]
    # (min, max, random, random)
    selected_ids = torch.cat((new_margin_max[:num],new_margin_max[-num:],new_margin_max[num:-num][random_ids]),dim=0)

    # assert (new_margin_max[:num]!=margin_max[:num]).sum()==0, print((new_margin_max[:num]!=margin_max[:num]).sum()) 
    # assert (new_margin_max[-num:]!=margin_max[-num:]).sum()==0, print((new_margin_max[:num]!=margin_max[:num]).sum()) 

    return target_idx[selected_ids]

def feat_normalize(features, norm=None, lim_min=-1.0, lim_max=1.0):
    r"""
    Description
    -----------
    Feature normalization function.

    Parameters
    ----------
    features : torch.FloatTensor
        Features in form of ``N * D`` torch float tensor.
    norm : str, optional
        Type of normalization. Choose from ``["linearize", "arctan", "tanh", "standarize"]``.
        Default: ``None``.
    lim_min : float
        Minimum limit of feature value. Default: ``-1.0``.
    lim_max : float
        Minimum limit of feature value. Default: ``1.0``.

    Returns
    -------
    features : torch.FloatTensor
        Normalized features in form of ``N * D`` torch float tensor.

    """
    if norm == "linearize":
        k = (lim_max - lim_min) / (features.max() - features.min())
        features = lim_min + k * (features - features.min())
    elif norm == "arctan":
        features = (features - features.mean()) / features.std()
        features = 2 * np.arctan(features) / np.pi
    elif norm == "tanh":
        features = (features - features.mean()) / features.std()
        features = np.tanh(features)
    elif norm == "standardize":
        features = (features - features.mean()) / features.std()
    else:
        features = features

    return features

def train_test_split_edges(data, use_mask=False, val_ratio=0.05, test_ratio=0.1):
    r"""Splits the edges of a :obj:`torch_geometric.data.Data` object
    into positive and negative train/val/test edges, and adds attributes of
    `train_pos_edge_index`, `train_neg_adj_mask`, `val_pos_edge_index`,
    `val_neg_edge_index`, `test_pos_edge_index`, and `test_neg_edge_index`
    to :attr:`data`.

    Args:
        data (Data): The data object.
        train_mask (bool, optional): if it's True, we will sample edges 
            accoding to the pre-defined split. (default: :`False`)
        val_ratio (float, optional): The ratio of positive validation
            edges. (default: :obj:`0.05`)
        test_ratio (float, optional): The ratio of positive test
            edges. (default: :obj:`0.1`)

    :rtype: :class:`torch_geometric.data.Data`
    """

    assert 'batch' not in data  # No batch-mode.
    
    if use_mask:
        # only use edges from trainset
        new_data = T.ToSparseTensor()(data)
        adj_train = new_data.adj_t[data.train_mask][:,data.train_mask]
        tval_mask = torch.logical_or(data.train_mask,data.val_mask)
        adj_val = new_data.adj_t[tval_mask][:,tval_mask]
        row, col = adj_val.coo()[:2]
        num_nodes = sum(tval_mask).item()
        print(f"# of edges for training: {len(row)}")
    else:
        num_nodes = data.num_nodes
        row, col = data.edge_index
    data.edge_index = None

    # Return upper triangular portion.
    mask = row < col
    row, col = row[mask], col[mask]

    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))

    # Positive edges.
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]

    r, c = row[:n_v], col[:n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)
    r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)

    r, c = row[n_v + n_t:], col[n_v + n_t:]
    data.train_pos_edge_index = torch.stack([r, c], dim=0)
    data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)

    # Negative edges.
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    neg_adj_mask[row, col] = 0

    neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
    perm = random.sample(range(neg_row.size(0)), min(n_v + n_t,
                                                     neg_row.size(0)))
    perm = torch.tensor(perm)
    perm = perm.to(torch.long)
    neg_row, neg_col = neg_row[perm], neg_col[perm]

    neg_adj_mask[neg_row, neg_col] = 0
    data.train_neg_adj_mask = neg_adj_mask

    row, col = neg_row[:n_v], neg_col[:n_v]
    data.val_neg_edge_index = torch.stack([row, col], dim=0)

    row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
    data.test_neg_edge_index = torch.stack([row, col], dim=0)

    return data


def get_index_induc(index_a, index_b):
    r"""

    Description
    -----------
    Get index under the inductive training setting.

    Parameters
    ----------
    index_a : tuple
        Tuple of index.
    index_b : tuple
        Tuple of index.

    Returns
    -------
    index_a_new : tuple
        Tuple of mapped index.
    index_b_new : tuple
        Tuple of mapped index.

    """

    i_a, i_b = 0, 0
    l_a, l_b = len(index_a), len(index_b)
    i_new = 0
    index_a_new, index_b_new = [], []
    while i_new < l_a + l_b:
        if i_a == l_a:
            while i_b < l_b:
                i_b += 1
                index_b_new.append(i_new)
                i_new += 1
            continue
        elif i_b == l_b:
            while i_a < l_a:
                i_a += 1
                index_a_new.append(i_new)
                i_new += 1
            continue
        if index_a[i_a] < index_b[i_b]:
            i_a += 1
            index_a_new.append(i_new)
            i_new += 1
        else:
            i_b += 1
            index_b_new.append(i_new)
            i_new += 1

    return index_a_new, index_b_new

def inductive_split(adj, split_idx):
    """
    inductive split adjs for PyG graphs
    will automatically use relative ids for splitted graphs
    """
    # adj =adj.to('cpu')
    adj_train = adj[split_idx["train"]][:,split_idx["train"]]
    train_mask = torch.zeros(adj.size(0)).bool()
    train_mask[split_idx["train"]] = 1
    val_mask = torch.zeros(adj.size(0)).bool()
    val_mask[split_idx["valid"]] = 1
    train_val_mask = torch.logical_or(train_mask, val_mask)
    adj_val = adj[train_val_mask][:,train_val_mask]
    adj_test = adj
    return adj_train, adj_val, adj_test


def set_rand_seed(rand_seed):
    rand_seed = rand_seed if rand_seed >= 0 else torch.initial_seed() % 4294967295  # 2^32-1
    random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)
    np.random.seed(rand_seed)


def extra_misg_ids(args, model, data, train_idx):
    """
    sample misclassified training samples
    and save to args.misg_path
    """
    y_true = data.y
    assert len(args.misg_path)>0
    with torch.no_grad():
        model.eval()
        out = model(data.x, data.adj_t)[train_idx]
        y_pred = out.argmax(dim=-1).to(y_true.device)
        model.train() 
    misg_ids = torch.nonzero(y_pred!=y_true[train_idx].view(-1),as_tuple=True)[0]
    misg_ids = train_idx[misg_ids].cpu()
    assert len(np.intersect1d(misg_ids,train_idx.cpu()))==len(misg_ids)
    misclass_data = {"ids":misg_ids,"preds":y_pred,"labels":y_true[train_idx]}
    misg_path = os.path.join(args.misg_path,"_".join([args.dataset,args.model]))
    print(f"Saving misclassified data to {misg_path+'.pt'}")
    print(f"Saving the trained GNN to {misg_path+'.model'}")
    torch.save(misclass_data,misg_path+'.pt')
    torch.save(model.state_dict(),misg_path+'.model')


def makedirs(path: str, isfile: bool = False):
    """
    Creates a directory given a path to either a directory or file.

    If a directory is provided, creates that directory. If a file is provided (i.e. isfiled == True),
    creates the parent directory for that file.

    :param path: Path to a directory or file.
    :param isfile: Whether the provided path is a directory or file.
    """
    if isfile:
        path = os.path.dirname(path)
    if path != '':
        os.makedirs(path, exist_ok=True)


def load_np_embedding(path: str):
    embedding = np.load(path)

    return embedding

def save_np_embedding(path: str, embedding: np.ndarray):
    path_dir = os.sep.join(path.split(os.sep)[:-1])
    if not os.path.exists(path_dir):
        os.mkdir(path_dir)
    np.save(path,embedding)

def save_features(path: str, features: List[np.ndarray]):
    """
    Saves features to a compressed .npz file with array name "features".

    :param path: Path to a .npz file where the features will be saved.
    :param features: A list of 1D numpy arrays containing the features for molecules.
    """
    np.savez_compressed(path, features=features)


def load_features(path: str) -> np.ndarray:
    """
    Loads features saved in a variety of formats.

    Supported formats:
    - .npz compressed (assumes features are saved with name "features")

    All formats assume that the SMILES strings loaded elsewhere in the code are in the same
    order as the features loaded here.

    :param path: Path to a file containing features.
    :return: A 2D numpy array of size (num_molecules, features_size) containing the features.
    """
    extension = os.path.splitext(path)[1]

    if extension == '.npz':
        features = np.load(path)['features']
    else:
        raise ValueError(f'Features path extension {extension} not supported.')

    return features


class tqdm(core_tqdm):

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("ascii", True)
        super(tqdm, self).__init__(*args, **kwargs)


def load_prebuilt_word_embedding(embedding_path, embedding_dim):
    """
    Read prebuilt word embeddings from a file
    :param embedding_path: string, file path of the word embeddings
    :param embedding_dim: int, dimensionality of the word embeddings
    :return: a dictionary mapping each word to its corresponding word embeddings
    """
    word_embedding_map = dict()

    if embedding_path is not None and len(embedding_path) > 0:
        for line in codecs.open(embedding_path, mode="r", encoding="utf-8"):
            line = line.strip()
            if not line or len(line.split())<=2:
                continue
            else:
                word_embedding = line.split()
                # print(word_embedding)
                assert len(word_embedding) == 1 + embedding_dim, print(len(word_embedding))
                word = word_embedding[0]
                embedding = [float(val) for val in word_embedding[1:]]
                if word in word_embedding_map.keys():
                    continue
                else:
                    word_embedding_map[word] = embedding
    # print(len(word_embedding_map.keys()),sorted(word_embedding_map.keys()))
    sorted_prebuilt_words = np.zeros((len(word_embedding_map.keys()),embedding_dim))
    for i in range(len(word_embedding_map.keys())):
        sorted_prebuilt_words[i] = word_embedding_map[str(i)]
    return sorted_prebuilt_words

import pickle as pkl
import sys

import networkx as nx
import scipy.sparse as sp

# geom-gcn
def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, labels, train_mask, val_mask, test_mask


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized

def get_train_val_test_split(random_state,
                             data,
                             train_examples_per_class=None, val_examples_per_class=None,
                             test_examples_per_class=None,
                             train_size=None, val_size=None, test_size=None):
    num_nodes = data.y.shape[0]
    labels = data.y

    random_state = np.random.RandomState(random_state)
    labels = torch.tensor(labels)
    labels = torch.nn.functional.one_hot(labels)
    num_samples, num_classes = labels.shape
    # num_samples, num_classes = labels.shape
    # num_samples = len(labels)
    # num_classes=int(labels.max() + 1)
    remaining_indices = list(range(num_samples))

    if train_examples_per_class is not None:
        train_indices = sample_per_class(random_state, labels, train_examples_per_class)
    else:
        # select train examples with no respect to class distribution
        train_indices = random_state.choice(remaining_indices, train_size, replace=False)

    if val_examples_per_class is not None:
        val_indices = sample_per_class(random_state, labels, val_examples_per_class, forbidden_indices=train_indices)
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(remaining_indices, val_size, replace=False)

    forbidden_indices = np.concatenate((train_indices, val_indices))
    if test_examples_per_class is not None:
        test_indices = sample_per_class(random_state, labels, test_examples_per_class,
                                        forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)

    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert len(np.concatenate((train_indices, val_indices, test_indices))) == num_samples

    if train_examples_per_class is not None:
        train_labels = labels[train_indices, :]
        train_labels = train_labels.numpy()
        train_sum = np.sum(train_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(train_sum).size == 1

    if val_examples_per_class is not None:
        val_labels = labels[val_indices, :]
        val_labels = val_labels.numpy()
        val_sum = np.sum(val_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(val_sum).size == 1

    if test_examples_per_class is not None:
        test_labels = labels[test_indices, :]
        test_labels = test_labels.numpy()
        test_sum = np.sum(test_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(test_sum).size == 1

    def get_mask(idx):
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[idx] = 1
        return mask
    data.idx_train = torch.tensor(train_indices)
    data.idx_val = torch.tensor(val_indices)
    data.idx_test = torch.tensor(test_indices)

    data.train_mask = get_mask(train_indices)
    data.val_mask = get_mask(val_indices)
    data.test_mask = get_mask(test_indices)
    print("number of training samples: ", len(train_indices) )
    print("number of val samples: ", len(val_indices))
    print("number of test samples: ", len(test_indices))

    return data

def sample_per_class(random_state, labels, num_examples_per_class, forbidden_indices=None):
    # labels = torch.tensor(labels)
    # labels = torch.nn.functional.one_hot(labels)
    num_samples, num_classes = labels.shape
    # num_samples = len(labels)
    # num_classes = int(labels.max() + 1)
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index, class_index] > 0.0:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [random_state.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])



import os

import scipy
from scipy.stats import sem
import numpy as np
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from sklearn.preprocessing import normalize
from torch_geometric.nn.conv.gcn_conv import gcn_norm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

class MaxNFEException(Exception): pass


def rms_norm(tensor):
  return tensor.pow(2).mean().sqrt()


def make_norm(state):
  if isinstance(state, tuple):
    state = state[0]
  state_size = state.numel()

  def norm(aug_state):
    y = aug_state[1:1 + state_size]
    adj_y = aug_state[1 + state_size:1 + 2 * state_size]
    return max(rms_norm(y), rms_norm(adj_y))

  return norm


def print_model_params(model):
  total_num_params = 0
  print(model)
  for name, param in model.named_parameters():
    if param.requires_grad:
      print(name)
      print(param.data.shape)
      total_num_params += param.numel()
  print("Model has a total of {} params".format(total_num_params))


def adjust_learning_rate(optimizer, lr, epoch, burnin=50):
  if epoch <= burnin:
    for param_group in optimizer.param_groups:
      param_group["lr"] = lr * epoch / burnin


def gcn_norm_fill_val(edge_index, edge_weight=None, fill_value=0., num_nodes=None, dtype=None):
  num_nodes = maybe_num_nodes(edge_index, num_nodes)

  if edge_weight is None:
    edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                             device=edge_index.device)

  if not int(fill_value) == 0:
    edge_index, tmp_edge_weight = add_remaining_self_loops(
      edge_index, edge_weight, fill_value, num_nodes)
    assert tmp_edge_weight is not None
    edge_weight = tmp_edge_weight

  row, col = edge_index[0], edge_index[1]
  deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
  deg_inv_sqrt = deg.pow_(-0.5)
  deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
  return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


def coo2tensor(coo, device=None):
  indices = np.vstack((coo.row, coo.col))
  i = torch.LongTensor(indices)
  values = coo.data
  v = torch.FloatTensor(values)
  shape = coo.shape
  print('adjacency matrix generated with shape {}'.format(shape))
  # test
  return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(device)


def get_sym_adj(data, opt, improved=False):
  edge_index, edge_weight = gcn_norm(  # yapf: disable
    data.edge_index, data.edge_attr, data.num_nodes,
    improved, opt['self_loop_weight'] > 0, dtype=data.x.dtype)
  coo = to_scipy_sparse_matrix(edge_index, edge_weight)
  return coo2tensor(coo)


def get_rw_adj_old(data, opt):
  if opt['self_loop_weight'] > 0:
    edge_index, edge_weight = add_remaining_self_loops(data.edge_index, data.edge_attr,
                                                       fill_value=opt['self_loop_weight'])
  else:
    edge_index, edge_weight = data.edge_index, data.edge_attr
  coo = to_scipy_sparse_matrix(edge_index, edge_weight)
  normed_csc = normalize(coo, norm='l1', axis=0)
  return coo2tensor(normed_csc.tocoo())


def get_rw_adj(edge_index, edge_weight=None, norm_dim=1, fill_value=0., num_nodes=None, dtype=None):
  num_nodes = maybe_num_nodes(edge_index, num_nodes)

  if edge_weight is None:
    edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                             device=edge_index.device)

  if not fill_value == 0:
    edge_index, tmp_edge_weight = add_remaining_self_loops(
      edge_index, edge_weight, fill_value, num_nodes)
    assert tmp_edge_weight is not None
    edge_weight = tmp_edge_weight

  row, col = edge_index[0], edge_index[1]
  indices = row if norm_dim == 0 else col
  deg = scatter_add(edge_weight, indices, dim=0, dim_size=num_nodes)
  deg_inv_sqrt = deg.pow_(-1)
  edge_weight = deg_inv_sqrt[indices] * edge_weight if norm_dim == 0 else edge_weight * deg_inv_sqrt[indices]
  return edge_index, edge_weight


def mean_confidence_interval(data, confidence=0.95):
  """
  As number of samples will be < 10 use t-test for the mean confidence intervals
  :param data: NDarray of metric means
  :param confidence: The desired confidence interval
  :return: Float confidence interval
  """
  if len(data) < 2:
    return 0
  a = 1.0 * np.array(data)
  n = len(a)
  _, se = np.mean(a), scipy.stats.sem(a)
  h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
  return h


def sparse_dense_mul(s, d):
  i = s._indices()
  v = s._values()
  return torch.sparse.FloatTensor(i, v * d, s.size())


def get_sem(vec):
  """
  wrapper around the scipy standard error metric
  :param vec: List of metric means
  :return:
  """
  if len(vec) > 1:
    retval = sem(vec)
  else:
    retval = 0.
  return retval


def get_full_adjacency(num_nodes):
  # what is the format of the edge index?
  edge_index = torch.zeros((2, num_nodes ** 2),dtype=torch.long)
  for idx in range(num_nodes):
    edge_index[0][idx * num_nodes: (idx + 1) * num_nodes] = idx
    edge_index[1][idx * num_nodes: (idx + 1) * num_nodes] = torch.arange(0, num_nodes,dtype=torch.long)
  return edge_index



from typing import Optional
import torch
from torch import Tensor
from torch_scatter import scatter, segment_csr, gather_csr


# https://twitter.com/jon_barron/status/1387167648669048833?s=12
# @torch.jit.script
def squareplus(src: Tensor, index: Optional[Tensor], ptr: Optional[Tensor] = None,
               num_nodes: Optional[int] = None) -> Tensor:
  r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements for applying the softmax.
        ptr (LongTensor, optional): If given, computes the softmax based on
            sorted inputs in CSR representation. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """
  out = src - src.max()
  # out = out.exp()
  out = (out + torch.sqrt(out ** 2 + 4)) / 2

  if ptr is not None:
    out_sum = gather_csr(segment_csr(out, ptr, reduce='sum'), ptr)
  elif index is not None:
    N = maybe_num_nodes(index, num_nodes)
    out_sum = scatter(out, index, dim=0, dim_size=N, reduce='sum')[index]
  else:
    raise NotImplementedError

  return out / (out_sum + 1e-16)


# Counter of forward and backward passes.
class Meter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.val = None
    self.sum = 0
    self.cnt = 0

  def update(self, val):
    self.val = val
    self.sum += val
    self.cnt += 1

  def get_average(self):
    if self.cnt == 0:
      return 0
    return self.sum / self.cnt

  def get_value(self):
    return self.val


class DummyDataset(object):
  def __init__(self, data, num_classes):
    self.data = data
    self.num_classes = num_classes


class DummyData(object):
  def __init__(self, edge_index=None, edge_Attr=None, num_nodes=None):
    self.edge_index = edge_index
    self.edge_attr = edge_Attr
    self.num_nodes = num_nodes
