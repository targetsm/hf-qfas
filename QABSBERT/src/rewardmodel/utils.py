from comet_ml import Experiment as CometExperiment, ExistingExperiment
import comet_ml
import torch
import numpy as np
import re
import os

CUDA_THINGS = torch.Tensor

class Experiment():
    def __init__(self, args):
        self.args = args
        
        # comet logger
        if self.args.new_exp:
            self.comet_exp = CometExperiment(api_key=args.api_key, project_name=args.proj_name)
        else:
            try:
                existing_experiment = comet_ml.API(api_key=args.api_key).get_experiment_by_id(args.exp_key)
            except Exception:
                existing_experiment = None
            if existing_experiment:
                self.comet_exp = ExistingExperiment(api_key=args.api_key, previous_experiment=args.exp_key)
            else:   
                self.comet_exp = CometExperiment(api_key=args.api_key, project_name=args.proj_name)
            
        self.key = self.comet_exp.get_key()
        self.comet_exp.log_parameters(args)
        self.comet_exp.log_code(file_name='config.py')
        self.comet_exp.log_code(file_name='datasets.py')
        self.comet_exp.log_code(file_name='model.py')
        self.comet_exp.log_code(file_name='test.py')
        self.comet_exp.log_code(file_name='train.py')
        self.comet_exp.log_code(file_name='trainer.py')
        self.comet_exp.log_code(file_name='utils.py')
        self.comet_exp.set_name(self.key)

    def get_key(self):
        return self.key

    def log_metric(self, key, val, step):
        # e.g., self.experiment.log_metric(key, val, step=step)
        # print(key, val, step)
        self.comet_exp.log_metric(key, val, step=step)

    def log_image(self, im, fname, step):
        # e.g., self.experiment.log_image(im, name=fname, step=step)
        # print(im, fname, step)
        self.comet_exp.log_image(im, name=fname, step=step)

# push things to a device
def things2dev(obj, dev):
    if isinstance(obj, CUDA_THINGS):
        return obj.to(dev)
    if isinstance(obj, list):
        return [things2dev(x, dev) for x in obj]
    if isinstance(obj, tuple):
        return tuple(things2dev(list(obj), dev))
    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = things2dev(v, dev)
    return obj

def mkdir_p(exp_path):
    os.makedirs(exp_path, exist_ok=True)

# The following two functions are copied from: https://github.com/yg211/summary-reward-no-reference
def clean_text(text):
    """
        Normalize text
        Remove & Replace unnessary characters
        Parameter argument:
        text: a string (e.g. '.... *** New York N.Y is a city...')
        Return:
        text: a string (New York N.Y is a city.)
    """
    text = re.sub(u'\u201e|\u201c', u'', text)
    text = re.sub(u"\u2022", u'. ', text)
    text = re.sub(u"([.?!]);", u"\\1", text)
    text = re.sub(u'``', u'``', text)
    text = re.sub(u"\.\.+", u" ", text)
    text = re.sub(u"\s+\.", u".", text)
    text = re.sub(u"\?\.", u"?", text)
    text = re.sub(u'[\n\s\t_]+', u' ', text)
    text = re.sub(u"[*]", u"", text)
    text = re.sub(u"\-+", u"-", text)
    text = re.sub(u'^ ', u'', text)
    text = re.sub(u'\u00E2', u'', text)
    text = re.sub(u'\u00E0', u'a', text)
    text = re.sub(u'\u00E9', u'e', text)
    text = re.sub(u'#', u'', text)
    text = re.sub(u'-LRB-', u'(', text)
    text = re.sub(u'-lrb-', u'(', text)
    text = re.sub(u'-RRB-', u')', text)
    text = re.sub(u'-rrb-', u')', text)

    return text

def encode_text(model, tokenizer, text, stride=128, gpu=True):
    if tokenizer:
        tokens = tokenizer.encode(text)
    else:
        tokens = text
        
    model.eval()

    with torch.no_grad():
        if len(tokens) <= 510:
            tokens = torch.tensor(tokens).unsqueeze(0)
            if gpu:
                tokens = tokens.to('cuda')
                model.to('cuda')
            vv = model(tokens)[0][0].data.cpu().numpy()
            vv = np.mean(vv,axis=0)
        else:
            end_pointer = stride
            batch = []
            real_length = []
            att_masks = []
            while True:
                start_pointer = end_pointer-510
                if start_pointer < 0: start_pointer = 0
                if start_pointer >= len(tokens): break
                if end_pointer <= len(tokens):
                    batch.append(tokens[start_pointer:end_pointer])
                    real_length.append(end_pointer-start_pointer)
                    att_masks.append([1]*real_length[-1])
                else:
                    batch.append(tokens[start_pointer:end_pointer])
                    real_length.append(len(tokens)-start_pointer)
                    att_masks.append([1] * real_length[-1])
                end_pointer += stride
                #print(len(batch[-1]))

            #padding
            longest = max(real_length)
            for ii in range(len(batch)):
                batch[ii] += [0] * (longest-real_length[ii])
                att_masks[ii] += [0] * (longest-real_length[ii])

            batch = torch.tensor(batch)
            att_masks = torch.tensor(att_masks)
            if gpu:
                batch = batch.to('cuda')
                att_masks = att_masks.to('cuda')
                model.to('cuda')

            last_layers = model(input_ids=batch,attention_mask=att_masks)[0].data.cpu().numpy()
            vectors = []
            for ii,bb in enumerate(last_layers):
                vectors.append(np.mean(bb[:real_length[ii]],axis=0))
            vv = np.mean(vectors,axis=0)

    return vv
