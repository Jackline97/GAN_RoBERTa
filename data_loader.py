from transformers import RobertaTokenizer
import torch
from torch.utils.data import Dataset, DataLoader

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    device = torch.device('cpu')


tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
class Generate_torch_dataset(Dataset):
    def __init__(self, data_df, max_length):
        self.MAX_SEQ_LEN = max_length
        self.data = data_df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if len(self.data.columns) == 3:
            moment = self.data.iloc[idx, 0]
            moment_unlabel = self.data.iloc[idx, 1]
            CLS_TOKEN = tokenizer.cls_token
            final_moment = CLS_TOKEN + ' ' + moment
            final_moment_unlabel = CLS_TOKEN + ' ' + moment_unlabel
            word_encode = tokenizer.encode(text=final_moment, max_length=self.MAX_SEQ_LEN, padding='max_length',
                                           truncation=True)
            word_encode_unlabel = tokenizer.encode(text=final_moment_unlabel, max_length=self.MAX_SEQ_LEN,
                                                   padding='max_length', truncation=True)
            agency_label = self.data.iloc[idx, 2]
            word_encode = torch.tensor(word_encode).to(device)
            word_unlabel_encode = torch.tensor(word_encode_unlabel).to(device)
            agency_label = torch.tensor(agency_label).to(device)
            sample = {'comment': word_encode, 'comment_unlabel': word_unlabel_encode, 'hate': agency_label}
        else:
            moment = self.data.iloc[idx, 0]
            CLS_TOKEN = tokenizer.cls_token
            final_moment = CLS_TOKEN + ' ' + moment
            word_encode = tokenizer.encode(text=final_moment, max_length=self.MAX_SEQ_LEN, padding='max_length',
                                           truncation=True)
            agency_label = self.data.iloc[idx, 1]
            word_encode = torch.tensor(word_encode).to(device)
            agency_label = torch.tensor(agency_label).to(device)
            sample = {'comment': word_encode, 'hate': agency_label}
        return sample