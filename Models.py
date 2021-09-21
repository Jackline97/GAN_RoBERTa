import torch
from transformers import RobertaModel
# Generator

class generator(torch.nn.Module):
    def __init__(self,word_embedding_size,g_hidden_size, dropout_rate = 0.2):
        super(generator,self).__init__()
        self.l1 = torch.nn.Linear(word_embedding_size, g_hidden_size)
        self.a1 = torch.nn.LeakyReLU()
        self.d1 = torch.nn.Dropout(dropout_rate)
        self.layer_hidden = torch.nn.Linear(g_hidden_size, g_hidden_size)
        self.l2 = torch.nn.Linear(g_hidden_size, g_hidden_size)
        self.a2 = torch.nn.LeakyReLU()
        self.d2 = torch.nn.Dropout(dropout_rate)
        self.layer_hidden2 = torch.nn.Linear(g_hidden_size, g_hidden_size)

    def forward(self,random_embedding):
        x = self.l1(random_embedding)
        x = self.a1(x)
        x = self.d1(x)
        x = self.layer_hidden(x)
        x = self.l2(x)
        x = self.a2(x)
        x = self.d2(x)
        final_embedding = self.layer_hidden2(x)
        return final_embedding


# Discriminator
# [hate, not hate]
# [real, not real]
class discriminator(torch.nn.Module):
    def __init__(self,g_hidden_size,num_labels, dropout_rate = 0.2):
        super(discriminator,self).__init__()
        self.l1 = torch.nn.Linear(g_hidden_size, int(g_hidden_size/2))
        self.a1 = torch.nn.LeakyReLU()
        self.d1 = torch.nn.Dropout(dropout_rate)
        self.classifier = torch.nn.Linear(int(g_hidden_size/2), num_labels)
    def forward(self, z):
        x = self.l1(z)
        x = self.a1(x)
        x = self.d1(x)
        final_cls = self.classifier(x)
        return final_cls


class ROBERTA_base_part(torch.nn.Module):
    def __init__(self, num_layers = 1, dropout_rate=0.2, model_type = 'roberta-large'):
        super(ROBERTA_base_part, self).__init__()
        self.roberta = RobertaModel.from_pretrained(model_type)
        self.layers = torch.nn.ModuleList()
        self.d1 = torch.nn.Dropout(dropout_rate)
        self.model_type = model_type
        # This part need to change
        if model_type = 'roberta-large':
            self.l1 = torch.nn.Linear(1024, 64)
        else:
            self.l1 = torch.nn.Linear(768, 64)
        for i in range(num_layers):
            self.layers.append(torch.nn.LayerNorm(64))
            self.layers.append(torch.nn.Dropout(dropout_rate))
            self.layers.append(torch.nn.Linear(64, 64))


    def forward(self, input_ids, attention_mask):
        output_list = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooler_output = output_list[1]
        x = self.d1(pooler_output)

        x = self.l1(x)
        for layer in self.layers:
            x = layer(x)
        return x

def save_models(generator, discriminator, roberta, path):
    torch.save(generator, path= path + '/generator.pt')
    torch.save(discriminator, path = path + '/discriminator.pt')
    torch.save(roberta, path = path + '/roberta.pt')

def load_models(generator, discriminator, roberta, path):
    generator = generator.load(path + '/generator.pt')
    discriminator = discriminator.load(path + '/discriminator.pt')
    roberta = roberta.load(path + '/roberta.pt')
    return generator, discriminator, roberta

