import torch, os
import numpy as np
from torch_geometric.data import Data#, collate
# from torch_geometric.data import InMemoryDataset
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
# from torch_geometric.utils import to_networkx, to_dense_adj
# import networkx as nx
from tqdm import tqdm 
# from multiprocessing import get_context
# import igraph as ig
from scipy.io import loadmat
import pandas as pd

DATAROOT={
    'adni': '**/data/ADNI_BOLD_SC',
    'abide': '**/All_Dataset', 
    'neurocon': '**/All_Dataset', 
    'taowu': '**/All_Dataset', 
    'ppmi': '**/All_Dataset',
    'matai': '**/All_Dataset',
    'oasis': '**/OASIS_BOLD_SC'
}
def dataloader_generator(dname, batch_size, num_workers, nfold, dataset=None, path_len=5, cutoff=4, total_fold=5, **kwargs):
    kf = KFold(n_splits=total_fold, shuffle=True, random_state=142857)
    if dataset is None:
        dataset = DiagnosisDataset(dataset=dname, root=DATAROOT[dname])
    train_data, data = list(kf.split(list(range(len(dataset)))))[nfold]
    print(f'Fold {nfold + 1}, Train {len(train_data)} subjects, Val {len(data)} subjects, len(train_data)={len(train_data)}, len(data)={len(data)}')
    train_dataset = torch.utils.data.Subset(dataset, train_data)
    valid_dataset = torch.utils.data.Subset(dataset, data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
    loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, loader, dataset


class DiagnosisDataset:
    
    def __init__(self, root='**/All_Dataset', dataset='abide', datatype='BOLD', seq_len=500) -> None:
        outlier_time_len = seq_len
        self.datasets = [
            'abide', 'neurocon', 'taowu', 'ppmi', 'adni', 'matai', 'oasis',
        ]
        self.dataset = dataset
        assert datatype in ['BOLD', 'FC']
        assert dataset in self.datasets
        self.datafile = [
            'AAL116_features_timeseries.mat',
            'AAL116_correlation_matrix.mat',
        ]
        pathlist_fn = f'datasets/{dataset}_pathlist.txt'
        if not os.path.exists(pathlist_fn):
            
            if dataset == 'adni':
                pathlist = []
                print(root)
                r = f'{root}/AAL_116/{datatype}'
                for fn in os.listdir(r):
                    if 'sub-' in fn: 
                        pathlist.append(f'{r}/{fn}')
                np.savetxt(pathlist_fn, pathlist, fmt='%s')
            elif dataset == 'oasis':
                pathlist = []
                print(root)
                r = f'{root}/AAL_116/{datatype}'
                for fn in os.listdir(r):
                    if 'OAS' in fn: 
                        pathlist.append(f'{r}/{fn}')
                np.savetxt(pathlist_fn, pathlist, fmt='%s')
            else:
                pathlist = []
                r = f'{root}/{dataset}/{dataset}'
                for fn in os.listdir(r):
                    if 'sub-' in fn: 
                        pathlist.append(f'{r}/{fn}')
                np.savetxt(pathlist_fn, pathlist, fmt='%s')
        pathlist = np.loadtxt(pathlist_fn, dtype=str)
        self.pathlist = pathlist
        self.data_list = []
        self.labels = []
        self.label_list = []
        self.time_len = []
        if dataset in ['adni']:
            sub_label = pd.read_csv(f'{root}/subject_info_250.csv')
            sub_label = {l[1]: l[-1] for l in np.array(sub_label)}
            labels = list(np.unique(list(sub_label.values())))
            label_remap = {'CN': 'CN', 'SMC': 'CN', 'EMCI': 'CN', 'LMCI': 'AD', 'AD': 'AD'}
            labels = [label_remap[l] for l in labels]
            sub_label = {k: label_remap[v] for k,v in sub_label.items()}
            sub_labelid = {'sub-'+k.replace('_',''): labels.index(v) for k,v in sub_label.items()}
            self.sub2labelid = sub_labelid
            
        for path in tqdm(pathlist, desc='Loading data'):
            fn = path.split('/')[-1]
            # data = load_data(path) ADNI
            if dataset not in ['adni', 'oasis']:
                fn = f'{path}/{fn}_{self.datafile[0 if datatype == "BOLD" else 1]}'
                data = load_data(fn)
            else:
                data = load_data(path)
            # data = load_data(fn)

            if data.shape[1] > outlier_time_len: continue
            self.time_len.append(seq_len)
            self.data_list.append(data)
            if dataset not in ['adni']:
                label = ''.join([i for i in fn if not i.isdigit()])
            else:
                label = self.sub2labelid[fn.split('_')[0]]

            if label not in self.labels: self.labels.append(label)
            self.label_list.append(self.labels.index(label))

    def __getitem__(self, index):
        x = torch.from_numpy(self.data_list[index])
        adj = torch.corrcoef(x)
        edge_index = torch.stack(torch.where(adj>0.5))
        # adj = torch.sparse_coo_tensor(indices=edge_index, values=adj[edge_index[0], edge_index[1]], size=adj.shape)
        x = torch.cat([x, torch.zeros(x.shape[0], max(self.time_len) - x.shape[1])], dim=1) #fill 0
        # max_time_len = max(self.time_len)
        # current_time_len = x.shape[1]
        # columns_to_fill = max_time_len - current_time_len
        # if columns_to_fill > 0:
        #     repeat_indices = torch.arange(current_time_len - 1, -1, -1)[:columns_to_fill]
        #     repeated_data = torch.index_select(x, dim=1, index=repeat_indices)
        #     x = torch.cat([x, repeated_data], dim=1) #fill repeat
        # max_time_len = max(self.time_len)
        # current_time_len = x.shape[1]
        # columns_to_fill = max_time_len - current_time_len

        # if columns_to_fill > 0:
        #     left_columns = columns_to_fill // 2
        #     right_columns = columns_to_fill - left_columns
        #     if left_columns > 0:
        #         left_data = x[:, 0].unsqueeze(1).repeat(1, left_columns)
        #         x = torch.cat([left_data, x], dim=1)
        
        #     if right_columns > 0:
        #         right_data = x[:, -1].unsqueeze(1).repeat(1, right_columns)
        #     x = torch.cat([x, right_data], dim=1) # left and right fill
        num_features = x.shape[1]
        data = {'x': x, 'edge_attr': adj[edge_index[0], edge_index[1]], 'y': self.label_list[index], 'num_features': num_features, 'edge_index': edge_index}
        data = Data.from_dict(data)
        return data
    
    def __len__(self):
        return len(self.data_list)
    
def load_data(fn, delimiter=' '):
    # N x T
    if fn.endswith('.mat'):
        return np.array(loadmat(fn)['data']).astype(np.float32).T 
    if fn.endswith('.txt'):
        # return np.loadtxt(fn, delimiter=delimiter).T #ADNI
        return np.loadtxt(fn, delimiter='\t').T #OASIS



if __name__=='__main__':
    tl, l, ds = dataloader_generator('adni', 10, 4, 0)
    print(len(tl), torch.tensor(ds.label_list).bincount(), ds.labels)
    for d in tl:
        print(d['x'].min(), d['x'].median(), d['x'].mean(), d['x'].max())
        print(d['x'].shape, d['adj'].shape, d['y'])