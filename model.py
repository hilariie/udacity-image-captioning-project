import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
  
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        
        super(DecoderRNN,self).__init__()
        self.n_hidden=hidden_size
        self.embed_size=embed_size
        self.vocab_size= vocab_size
        
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        self.lin = nn.Linear(hidden_size, vocab_size)
        
        
        
    
    def forward(self, features, captions):
        captions = captions[:, :-1]
        x = self.embedding(captions)
        
        x = torch.cat((features.unsqueeze(1), x), 1)
        
        x, caption = self.lstm(x)
        x = self.lin(x)
        
        
        return x

    '''def sample(self, inputs, states=None, max_len=20):
        caption = []
        
        hidden = (torch.randn(1, 1, self.n_hidden).to(inputs.device),
        torch.randn(1, 1, self.n_hidden).to(inputs.device))
        
        for i in range(max_len):
            lstm_out, hidden = self.lstm(inputs, hidden) 
            outputs = self.lin(lstm_out)        
            outputs = outputs.squeeze(1)                 
            wordid  = outputs.argmax(dim=1)              
            caption.append(wordid.item())
            
            
            inputs = self.embedding(wordid.unsqueeze(0)) 
          
            return caption'''
        
    def sample(self, inputs, states=None, max_len=20):
       # states = (torch.randn(self.num_layers, 1, self.hidden_size).to(inputs.device), torch.randn(self.num_layers, 1,
       #                                                                                            self.hidden_size).to(inputs.device)
                  
        states = (torch.randn(1, 1, 256).to(inputs.device),
                 torch.randn(1, 1, 256).to(inputs.device))
        preds = []
        count = 0
        word_item = None
        
        while count < max_len and word_item != 1 :
            
            #Predict output
            
            output_lstm, states = self.lstm(inputs, states)
            
            output = self.lin(output_lstm.squeeze(1))
            #output= output.squeeze(1)
            #Get max value
            prob, word = torch.max(output, 1)
            
            #append word
            word_item = word.item()
            preds.append(word_item)
            
            #next input is current prediction
            #inputs = inputs.unsqueeze(1)
            inputs = self.embedding(word)
            inputs = inputs.unsqueeze(1)
            
            count+=1
        
        return preds