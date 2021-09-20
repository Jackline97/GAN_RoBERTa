import torch.nn.functional as F
import torch
import numpy as np
from tqdm import tqdm
class trainer():
    def __init__(self, train_hate, val_hate, tokenizer, Discriminator,Generator, Roberta_model,batch_size,lr,bert_embedding_size,device, num_epochs):
        self.tokenizer = tokenizer
        self.Roberta_model = Roberta_model
        self.Generator = Generator
        self.Discriminator = Discriminator
        self.batch_size = batch_size
        self.lr = lr
        self.bert_embedding_size = bert_embedding_size
        self.device = device
        self.epochs = num_epochs
        self.train_iter = train_hate
        self.val_hate = val_hate


    def train(self):
        PAD_INDEX = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        criterion = torch.nn.BCEWithLogitsLoss()
        random_noise = torch.randn(self.batch_size, self.bert_embedding_size).to(self.device)
        optimizerD = torch.optim.Adam(self.Discriminator.parameters(), lr=self.lr)
        optimizerG = torch.optim.Adam(self.Generator.parameters(), lr=self.lr)
        optimizerR = torch.optim.Adam(self.Roberta_model.parameters(), lr=self.lr)
        fake_label = 2
        G_losses = []
        D_losses = []
        self.Discriminator.train()
        self.Roberta_model.train()
        self.Generator.train()
        for epoch in range(self.epochs):
            count = 0
            temp_G = []
            temp_D = []
            temp_V = []
            # Iter on the label data
            for data in tqdm(self.train_iter):
                optimizerD.zero_grad()
                optimizerR.zero_grad()
                source = data['comment']
                unlabel_souce = data['comment_unlabel']
                target = data['hate']
                mask = (source != PAD_INDEX).type(torch.uint8)
                roberta_embedding = self.Roberta_model(input_ids=source,
                                                  attention_mask=mask)
                mask_unlabel = (unlabel_souce != PAD_INDEX).type(torch.uint8)
                roberta_embedding_unlabel = self.Roberta_model(input_ids=unlabel_souce,
                                                          attention_mask=mask_unlabel)
                target = F.one_hot(target.to(torch.int64), num_classes=3)

                # Update Discriminator and Roberta loss
                output_from_R_cls = self.Discriminator(roberta_embedding)
                target = target.float()
                loss_R_cls = criterion(output_from_R_cls, target)
                output_from_R_cls_unlabel = self.Discriminator(roberta_embedding_unlabel)
                softmax_layer = torch.nn.Softmax(dim=1)
                u_output = softmax_layer(output_from_R_cls_unlabel)
                final_label = torch.max(u_output, dim=1).indices
                error_label = (final_label == 2)
                error_indices = error_label.nonzero()


                if error_indices.shape[0] != 0:
                    final_tensor = torch.index_select(output_from_R_cls_unlabel, 0, error_indices.squeeze())
                    real_target = torch.tensor([[1, 1, 0] for i in range(final_tensor.shape[0])]).to(self.device)
                    real_target = real_target.float()
                    loss_unlabel = criterion(final_tensor, real_target)
                    loss_R = loss_R_cls + loss_unlabel
                else:
                    loss_R = loss_R_cls

                label_fake = torch.full((self.batch_size,), fake_label, dtype=torch.float, device=self.device)
                output_from_G = self.Generator(random_noise.detach())
                output_from_fake_real = self.Discriminator(output_from_G)
                label_fake = F.one_hot(label_fake.to(torch.int64), num_classes=3)
                label_fake = label_fake.float()
                loss_F = criterion(output_from_fake_real, label_fake)
                err_D = loss_R + loss_F
                err_D.backward()
                optimizerD.step()
                optimizerR.step()

                # Update self.Generator loss
                optimizerG.zero_grad()
                output_fake = self.Discriminator(output_from_G.detach())
                err_G = criterion(output_fake, target.detach())
                err_G.backward()
                optimizerG.step()
                G_losses.append(float(err_G))
                D_losses.append(float(err_D))
                temp_D.append(float(err_D))
                temp_G.append(float(err_G))
                count += 1
            print('start val process')
            self.Generator.eval()
            self.Discriminator.eval()
            self.Roberta_model.eval()
            with torch.no_grad():
                for data in tqdm(self.val_hate):
                    source = data['comment']
                    target = data['hate']
                    mask = (source != PAD_INDEX).type(torch.uint8)
                    roberta_embedding = self.Roberta_model(input_ids=source,
                                                           attention_mask=mask)
                    target = F.one_hot(target.to(torch.int64), num_classes=3)
                    output_from_R_cls = self.Discriminator(roberta_embedding)
                    target = target.float()
                    loss_R_cls_val = criterion(output_from_R_cls, target)
                    temp_V.append(float(loss_R_cls_val))
            print('epoch_no [{}/{}]:'.format(epoch, self.epochs), 'training_loss_G:', float(np.mean(temp_G)),
                  'training_loss_D:', float(np.mean(temp_D)), 'Val_loss_cls:', float(np.mean(temp_V)))

        return G_losses, D_losses, self.Roberta_model,self.Generator, self.Discriminator