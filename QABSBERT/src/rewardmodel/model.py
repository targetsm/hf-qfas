from numpy.core.shape_base import vstack
import torch
import torch.nn as nn
from transformers import BertModel
from rewardmodel import utils
class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MultiLayerPerceptron, self).__init__()

        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size, 1)

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        torch.nn.init.xavier_normal_(self.fc3.weight)
        torch.nn.init.xavier_normal_(self.fc4.weight)

        #self.sigmoid = nn.Sigmoid()        
        
    def forward(self, x):
        hidden = self.fc1(x)
        hidden = self.dropout(hidden)
        relu = self.activation(hidden)
        hidden = self.fc2(relu)
        hidden = self.dropout(hidden)
        relu = self.activation(hidden)
        hidden = self.fc3(relu)
        hidden = self.dropout(hidden)
        relu = self.activation(hidden)
        output = self.fc4(relu)
        
        #output = self.sigmoid(output)  #predicting a reward score instead of a probability
        return output

class RewardModel(nn.Module):
    def __init__(self):
        super(RewardModel, self).__init__()

        # modules
        self.scorer = MultiLayerPerceptron(input_size=768, hidden_size=256)
        self.softmax = torch.nn.Softmax(dim=1)

        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

        # loss function
        self.loss = nn.BCELoss()

    def forward(self, emb_doc_and_sum1, emb_doc_and_sum2, target_prefs):
        """
            - emb_doc_and_sum1, emb_doc_and_sum2 : torch.tensor(batch_size, 768)
            - target_prefs: torch.tensor[batch_size]
        """

        self.scorer.train()

        # Compute for each of these two vectors the reward
        reward1 = self.scorer(emb_doc_and_sum1)
        reward2 = self.scorer(emb_doc_and_sum2)

        # Compute 'likelihood' that our scorer prefers summary1 over summary2
        probs = self.softmax(torch.cat((reward1,reward2), axis=1))
        
        # Predicted preference, 0 if summ1, 1 if summ2
        predicted_preferences = torch.argmax(probs, axis=1)

        loss = self.loss(probs[:,1], target_prefs.float())

        out = {
            'predicted_preferences': predicted_preferences,
            'loss': loss
        }

        return out
    
    def reward(self, documents, summaries):
    
        self.scorer.eval()

        with torch.no_grad():
            
            # Vectorize documents and summaries with BERT, embedded_* : torch.tensor, shape=(batch_size, 768)
            embedded_documents = torch.tensor(list(map(lambda t: utils.encode_text(self.bert_model, None, t), documents)))
            embedded_summaries = torch.tensor(list(map(lambda t: utils.encode_text(self.bert_model, None, t), summaries)))

            emb_doc_and_summ = (embedded_documents - embedded_summaries) ** 2

            return self.scorer(emb_doc_and_summ)

