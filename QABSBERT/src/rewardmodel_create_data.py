from rewardmodel.config import args
from rewardmodel.datasets import fetch_dataloader
import rewardmodel.utils as utils
from transformers import BertTokenizer, BertModel
import torch
from pprint import pprint


def write_embeddings_to_disk(file_path, loader):
    batch = 0
    
    for documents, summaries1, summaries2, preferences in loader:
        print('Batch %d' % batch)

        # Clean documents and summaries
        documents = map(utils.clean_text, documents)
        summaries1 = map(utils.clean_text, summaries1)
        summaries2 = map(utils.clean_text, summaries2)

        # Vectorize documents and summaries with BERT, embedded_* : torch.tensor, shape=(batch_size, 768)
        embedded_documents = torch.tensor(list(map(lambda t: utils.encode_text(bert_model, bert_tokenizer, t), documents)))
        embedded_summaries1 = torch.tensor(list(map(lambda t: utils.encode_text(bert_model, bert_tokenizer, t), summaries1)))
        embedded_summaries2 = torch.tensor(list(map(lambda t: utils.encode_text(bert_model, bert_tokenizer, t), summaries2)))

        # Concat each document vector with the corresponding summary1 vector and the summary2 vector, respectively, doc_and_summ_* : torch.tensor, shape=(batch_size, 1536)
        doc_and_summ1 = torch.hstack((embedded_documents, embedded_summaries1))
        doc_and_summ2 = torch.hstack((embedded_documents, embedded_summaries2))

        # Save each batch to a file
        data = {
            'doc_and_summ1' : doc_and_summ1,
            'doc_and_summ2' : doc_and_summ2,
            'preferences' : preferences
        }

        torch.save(data, file_path + '_' + str(batch) + '.pt')

        batch += 1


if __name__ == '__main__':
    from rewardmodel.config import args
    pprint(vars(args))

    """
        Precompute the embedding of every document and summary pair from the TLDR comparison set
    """

    train_loader = fetch_dataloader('raw', 'train', args)
    valid1_loader = fetch_dataloader('raw', 'valid1', args)
    valid2_loader = fetch_dataloader('raw', 'valid2', args)

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    
    write_embeddings_to_disk('rewardmodel/open_ai_data/comparisons/smaller_embeddings/emb_train', train_loader)
    write_embeddings_to_disk('rewardmodel/open_ai_data/comparisons/smaller_embeddings/emb_valid1', valid1_loader)
    write_embeddings_to_disk('rewardmodel/open_ai_data/comparisons/smaller_embeddings/emb_valid2', valid2_loader)
