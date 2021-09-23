import comet_ml
from rewardmodel.model import RewardModel
from tqdm import tqdm
from rewardmodel import utils
from rewardmodel.utils import Experiment
import numpy as np
import os.path as op
import torch
from sklearn.metrics import balanced_accuracy_score


"""
The Trainer class abstracts aways lots of boilerplate code such as saving weights to disks, training epochs, evaluation epochs, visualization etc.
Here is the pseudo-code of how each method interacts

def train:
    for each epoch:
        train_epoch() # train one epoch
        if every k epochs:
            eval_epoch() # eval on validation set(s)
        save_model()

See self.train() and its subroutines for details
"""

class Trainer():
    def __init__(self, train_loader, val_loaders, args):
        super().__init__()
        self.args = args
        self.train_loader = train_loader
        self.val_loaders = val_loaders

        self.model = RewardModel()
        self.model.cuda()
        print(self.model)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0.001)

        self.current_epoch = 0
        self.global_step = 0

        if self.args.load_ckpt != '':
            self.load_ckpt(self.args.load_ckpt)

        # experiment key
        if self.args.load_ckpt == '':
            self.args.new_exp = True
            self.experiment = Experiment(args)
        else:
            self.args.new_exp = False
            self.args.exp_key = self.args.load_ckpt.split('/')[1]
            self.experiment = Experiment(args)

        print('Experiment Key: %s' % (self.experiment.get_key()))

        # folder containing info of this experiment
        self.exp_path = op.join('logs', self.experiment.get_key())
        self.save_path = op.join(self.exp_path, 'rewardmodel.pt')  # model dumping
        utils.mkdir_p(self.exp_path)


    def train_epoch(self):
        assert self.train_loader is not None
        model = self.model
        train_loader = self.train_loader
        optimizer = self.optimizer
        epoch = self.current_epoch

        model.train()
        print_every = 20
        running_loss = 0.0
        pbar = tqdm(enumerate(train_loader, 0), total=len(train_loader))

        # training loop for epoch
        for i, batch in pbar:
            # push things to CUDA
            emb_doc_and_sum1, emb_doc_and_sum2, target_prefs = utils.things2dev(batch, 'cuda')

            # standard pytorch boilerplate stuff
            optimizer.zero_grad()
            out = model(emb_doc_and_sum1, emb_doc_and_sum2, target_prefs)
            loss = out['loss']
            loss.backward()
            optimizer.step()

            # logging
            running_loss += loss
            if i % print_every == print_every - 1:
                avg_loss = running_loss / print_every
                pbar.set_description('Epoch=%d loss=%.5f' % (epoch + 1, avg_loss))
                if self.experiment is not None:
                    self.experiment.log_metric('loss', avg_loss, self.global_step)
                running_loss = 0.0
            self.global_step += 1

        self.current_epoch += 1

    def eval_epoch(self, val_loader_dict):
        # evaluate on a data loader
        assert isinstance(val_loader_dict, dict)
        model = self.model
        val_loader = val_loader_dict['loader']
        prefix = val_loader_dict['prefix']

        pbar = tqdm(enumerate(val_loader, 0), total=len(val_loader))

        loss_list = []
        acc_list = []

        model.eval()

        with torch.no_grad():
            for i, batch in pbar:
                emb_doc_and_sum1, emb_doc_and_sum2, target_prefs = utils.things2dev(batch, 'cuda')
                out = model(emb_doc_and_sum1, emb_doc_and_sum2, target_prefs)

                loss = out['loss']
                predicted_prefs = out['predicted_preferences']

                loss_list.append(loss.cpu())
                acc_list.append(balanced_accuracy_score(target_prefs.cpu(), predicted_prefs.cpu()))
    

        avg_loss = np.mean(loss_list)
        self.experiment.log_metric(prefix + '_loss', avg_loss, self.global_step)
        print(prefix + '_loss: ' + str(avg_loss))

        avg_acc = np.mean(acc_list)
        self.experiment.log_metric(prefix + '_acc', avg_acc, self.global_step)
        print(prefix + '_acc: ' + str(avg_acc))

        return avg_loss

    def load_ckpt(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)

        self.global_step = checkpoint['step']
        self.current_epoch = checkpoint['epoch']

        model_sd = checkpoint['model']
        opt_sd = checkpoint['optimizer']
        
        print(self.model.load_state_dict(model_sd))
        print(self.optimizer.load_state_dict(opt_sd))


    def save_model(self):
        model_sd = self.model.state_dict()
        model_sd = utils.things2dev(model_sd, 'cpu')
        opt_sd = self.optimizer.state_dict()

        torch.save({
                    'epoch': self.current_epoch,
                    'step': self.global_step,
                    'model': model_sd,
                    'optimizer': opt_sd
                    }, self.save_path)
                    
        print('Saved model to: %s' % (self.save_path))

    def train(self):
        for epoch in range(self.args.num_epoch):
            # train one epoch
            self.train_epoch()

            if self.current_epoch % self.args.eval_every_epoch == 0:
                # evaluate on a list of loaders
                for loader in self.val_loaders:
                    # metric performance on each loader
                    self.eval_epoch(loader)

                self.save_model()
        print('Finished Training')
