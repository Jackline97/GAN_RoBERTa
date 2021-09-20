import word_preprocessor
import argparse
import torch
import os
import data_loader
import Models
from torch.utils.data import DataLoader
import pandas as pd
import trainer
import Test_module


parser = argparse.ArgumentParser()
parser.add_argument('--unlabelled_data', type=str, default='Data/Unlabelled_data/', help='Path to unlabelled dataset')
parser.add_argument('--labelled_data', type=str, default='Data/Labelled_data/'  ,help='Path to labelled dataset')
parser.add_argument('--semi_mode', type=bool, default=True , help='Whether to use semi_supervised learning')
parser.add_argument('--MAX_SEQ_LEN', type=int, default=128  ,help='Max sentence length')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--Extra_linear_layer_number', type=int, default=1, help ='Extra linear layer number')
parser.add_argument('--drop_out_rate', type=float, default=0.2, help ='drop out rate')
parser.add_argument('--word_embedding_size', type=int, default=768, help ='word embedding size')
parser.add_argument('--hidden_state_size', type= int, default=64,help = 'Hidden state size')
parser.add_argument('--lr', type= float, default=1e-6, help = 'learning rate')
parser.add_argument('--epoch', type= int, default=1, help = 'epoch number')
parser.add_argument('--model', type= str, default='model/', help = 'model folder path')
parser.add_argument('--run_mode', type= str, default='train', help = 'run mode')
parser.add_argument('--model_type', type= str, default='roberta-large', help = 'RoBERTa model size: (roberta-large, roberta-base)')
args = parser.parse_args()


if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    device = torch.device('cpu')


num_labels = 3
tokenizer = data_loader.tokenizer
## Stage one: Data_preprocessing and torch_dataset transformation

def Init_all_data(args):
    labelled_path = args.labelled_data
    unlabelled_path = args.unlabelled_data
    if unlabelled_path is not None and args.semi_mode is True:
        unlabelled_all = word_preprocessor.Word_Preprocessing(pd.read_csv(os.path.join(unlabelled_path, 'all_unlabeled_data.csv')), target='text').process_all()
    for filename in os.listdir(labelled_path):

        filename = os.path.join(labelled_path, filename)
        if '2400' in filename:
            test_2400 = word_preprocessor.Word_Preprocessing(pd.read_csv(filename), target='comment').process_all()
        elif 'EAtrain' in filename:
            EA_train = word_preprocessor.Word_Preprocessing(pd.read_csv(filename), target='text').process_all()
        elif 'EAvalid' in filename:
            EA_valid = word_preprocessor.Word_Preprocessing(pd.read_csv(filename), target='text').process_all()
        elif 'EAtest' in filename:
            EA_test = word_preprocessor.Word_Preprocessing(pd.read_csv(filename), target='text').process_all()

    if args.semi_mode is True:
        if len(unlabelled_all) == len(EA_train):
            EA_train.insert(1, 'unlabel_moment',unlabelled_all['text'])
        elif len(unlabelled_all) < len(EA_train):
            mask_token = 100
            Extra_size = len(EA_train) - len(unlabelled_all)
            Mask_text = [mask_token for i in range(Extra_size)]
            Mask_label = [mask_token for i in range(Extra_size)]
            Mask_df = pd.DataFrame({'text': Mask_text, 'hate': Mask_label})
            unlabelled_all = unlabelled_all.append(Mask_df, ignore_index=True)
            EA_train.insert(1, 'unlabel_moment', unlabelled_all['text'])
        else:
            mask_token = 100
            Extra_size = abs(len(EA_train) - len(unlabelled_all))
            Mask_text = [mask_token for i in range(Extra_size)]
            Mask_label = [mask_token for i in range(Extra_size)]
            Mask_df = pd.DataFrame({'text': Mask_text, 'hate': Mask_label})
            unlabelled_all = EA_train.append(Mask_df, ignore_index=True)
            EA_train.insert(1, 'unlabel_moment', unlabelled_all['text'])

    train_hate = data_loader.Generate_torch_dataset(EA_train, args.MAX_SEQ_LEN)
    test_hate = data_loader.Generate_torch_dataset(EA_test, args.MAX_SEQ_LEN)
    val_hate = data_loader.Generate_torch_dataset(EA_valid, args.MAX_SEQ_LEN)
    test_2400 = data_loader.Generate_torch_dataset(test_2400, args.MAX_SEQ_LEN)

    train_hate_loader = DataLoader(train_hate, batch_size=args.batch_size)
    test_hate_loader =  DataLoader(test_hate, batch_size=args.batch_size)
    val_hate_loader = DataLoader(val_hate, batch_size=args.batch_size)
    test_2400_loader = DataLoader(test_2400, batch_size=args.batch_size)

    return train_hate_loader, test_hate_loader, val_hate_loader, test_2400_loader


## Stage two: init model
def init_models(args,num_labels):
    Roberta_model = Models.ROBERTA_base_part(num_layers=args.Extra_linear_layer_number,dropout_rate=args.drop_out_rate, model_type = args.model_type).to(device)
    Generator = Models.generator(args.word_embedding_size, args.hidden_state_size).to(device)
    Discriminator = Models.discriminator(args.hidden_state_size, num_labels).to(device)
    return Roberta_model, Generator, Discriminator



def init_train(args,train_hate,val_hate, Discriminator,Generator, Roberta_model, device, mode):

    if mode == 'test':
        Roberta_model, Generator, Discriminator = Models.load_models(Roberta_model, Generator, Discriminator, args.model)
        return Roberta_model, Generator, Discriminator
    elif mode == 'train':
        model_trainer = trainer.trainer(train_hate=train_hate, val_hate=val_hate, tokenizer=tokenizer, Discriminator=Discriminator,Generator=Generator,
                                        Roberta_model=Roberta_model,batch_size=args.batch_size,lr=args.lr,bert_embedding_size=args.word_embedding_size,device=device, num_epochs=args.epoch)
        G_losses, D_losses, Roberta_model, Generator, Discriminator = model_trainer.train()
        return G_losses, D_losses, Roberta_model, Generator, Discriminator

print('Start loading data')
train_hate_loader, test_hate_loader, val_hate_loader, test_2400_loader = Init_all_data(args)
Roberta_model, Generator, Discriminator = init_models(args,num_labels)


assert args.run_mode in ['train','test'], 'run mode doesnt exist'
print('Start training model')
if args.run_mode == 'train':
    G_losses, D_losses, Roberta_model, Generator, Discriminator = init_train(args,train_hate_loader,val_hate_loader,
                                                                             Discriminator,Generator, Roberta_model, device,args.run_mode)
    Models.save_models(Generator,Discriminator, Roberta_model, args.model)
elif args.run_mode == 'test':
    assert (os.path.exists(args.model + 'generator.pt') and os.path.exists(args.model +'generator.pt') and os.path.exists(args.model + 'roberta.pt')), \
        'Model doesnt exist, use train mode instead.'
    Roberta_model, Generator, Discriminator = init_train(args, train_hate_loader, val_hate_loader, Discriminator,
                                                         Generator, Roberta_model, device,args.run_mode)
print(Roberta_model)

# in-domain test
print('In-domain test')
Test_module.evaluate(test_hate_loader, tokenizer, Discriminator, Roberta_model)

# 2400 test
print('Out-domain test')
Test_module.evaluate(test_2400_loader, tokenizer, Discriminator, Roberta_model)