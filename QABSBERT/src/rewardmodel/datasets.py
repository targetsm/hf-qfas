from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer
import json
from rewardmodel import utils
import torch
from pprint import pprint
import os

class RawComparisonDataset(Dataset):
    def __init__(self, mode):
        
        self.mode = mode

        # Select which files to fetch
        if mode == 'train':
            batches = list(range(3,11)) + [15]
        elif mode == 'minitrain':
            batches = [9]
        elif mode == 'valid1':
            batches = list(range(6,22))
        elif mode == 'valid2':
            batches = list(range(11,22))
        elif mode == 'all':
            batches = list(range(3,22))
        else:
            assert False, 'Invalid dataset mode'

        # Get all json files
        self.file_paths = ['rewardmodel/open_ai_data/comparisons/batch%i.json' % i for i in batches]
        assert len(self.file_paths) > 0, "No raw comparison files found"

        # Store documents, summaries and preferences in 4 separate lists
        self.documents = []
        self.summaries1 = []
        self.summaries2 = []
        self.preferences = []

        for file_path in self.file_paths:
            with open(file_path) as f:
                for line in f:
                    comparison = json.loads(line)

                    if self.mode == 'minitrain' and comparison['split'] != 'train':
                        continue

                    if self.mode in ['train', 'valid1', 'valid2'] and comparison['split'] != self.mode:
                        continue
            
                    document = comparison['info']['post']
                    summary1 = comparison['summaries'][0]['text']
                    summary2 = comparison['summaries'][1]['text']
                    preference = comparison['choice']   # 0 or 1

                    self.documents.append(document)
                    self.summaries1.append(summary1)
                    self.summaries2.append(summary2)
                    self.preferences.append(preference)

    def __getitem__(self, idx):
        return self.documents[idx], self.summaries1[idx], self.summaries2[idx], self.preferences[idx]

    def __len__(self):
        return len(self.documents)


class EmbeddedComparisonDataset(Dataset):
    def __init__(self, mode):
        
        self.mode = mode

        # Get all embeddings files    
        self.file_paths = []
        
        for f in os.listdir('rewardmodel/open_ai_data/comparisons/smaller_embeddings'):
            self.file_paths.append('rewardmodel/open_ai_data/comparisons/smaller_embeddings/' + f)

        self.file_paths.sort()

        # Compared to RawComparisonDataset, we have modified the splits to increase the size of training data
        if mode == 'train':
            self.file_paths = self.file_paths[40:]
        
        elif mode == 'minitrain':
            self.file_paths = self.file_paths[40:45]
        
        elif mode == 'valid1':
            self.file_paths = self.file_paths[:20]

        elif mode == 'valid2':
            self.file_paths = self.file_paths[20:40]
        
        elif mode == 'all':
            pass

        else:
            assert False, 'Invalid dataset mode'

        assert len(self.file_paths) > 0, 'No embedded comparison files found'

        # Store document||summary embeddings and preferences in 3 separate tensors
        tensor_dict = torch.load(self.file_paths[0])

        self.doc_and_summ1 = tensor_dict['doc_and_summ1']   # shape = (num_document, embedding_size)
        self.doc_and_summ2 = tensor_dict['doc_and_summ2']   # shape = (num_document, embedding_size)
        self.preferences = tensor_dict['preferences']       # shape = (num_document,)

        for f in self.file_paths[1:]:
            tensor_dict = torch.load(f)
            self.doc_and_summ1 = torch.vstack((self.doc_and_summ1, tensor_dict['doc_and_summ1']))
            self.doc_and_summ2 = torch.vstack((self.doc_and_summ2, tensor_dict['doc_and_summ2']))
            self.preferences = torch.hstack((self.preferences, tensor_dict['preferences']))
    
    def __getitem__(self, idx):
        # return self.doc_and_summ1[idx,:], self.doc_and_summ2[idx,:], self.preferences[idx]
        # Unlike proposed in the paper, we dont provide the MLP reward mdoel with the concatenated document and summary vector as input
        # We use the component-wise squared difference between the two vectors instead to reduce input dimensionality
        emb_len = self.doc_and_summ1.shape[1]
        diff_doc_and_summ1 = (self.doc_and_summ1[idx, : emb_len // 2] - self.doc_and_summ1[idx, emb_len // 2 : ]) ** 2
        diff_doc_and_summ2 = (self.doc_and_summ2[idx, : emb_len // 2] - self.doc_and_summ2[idx, emb_len // 2 : ]) ** 2
        return diff_doc_and_summ1, diff_doc_and_summ2, self.preferences[idx] 

    def __len__(self):
        return self.doc_and_summ1.shape[0]

def fetch_dataloader(dataset, mode, args):

    print('Creating %s %s dataset...' % (dataset, mode))
    if dataset == 'raw':
        dataset = RawComparisonDataset(mode)
    elif dataset == 'embedded':
        dataset = EmbeddedComparisonDataset(mode)
    else:
        assert False, 'Invalid dataset selected'

    if mode == 'train':
        shuffle = True
    else:
        shuffle = False

    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers, 
        shuffle=shuffle,
        pin_memory=True,
        drop_last=args.drop_last
    )

    return loader