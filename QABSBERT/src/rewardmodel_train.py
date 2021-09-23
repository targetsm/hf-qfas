from rewardmodel import trainer as tr
from rewardmodel import datasets
from pprint import pprint

def run_exp(args):
    # training loader, size of training set depends on --trainsplit
    train_loader = datasets.fetch_dataloader('embedded', args.trainsplit, args)

    # evaluation loader to see training set performance, use minitrain 
    train_val_loader = {'loader': datasets.fetch_dataloader('embedded', 'minitrain', args),
                        'prefix': 'train'}

    # evaluation loader to see validation set performance
    val_val_loader = {'loader': datasets.fetch_dataloader('embedded', 'valid1', args),
                      'prefix': 'valid1'}

    trainer = tr.Trainer(train_loader, [train_val_loader, val_val_loader], args)

    trainer.train()


def run_full_exp(args):
    # Training with both training and validation sets
    loader = datasets.fetch_dataloader('embedded', 'all', args)

    trainer = tr.Trainer(loader, [], args)
    trainer.train()


if __name__ == '__main__':
    from rewardmodel.config import args
    pprint(vars(args))
    if args.trainsplit == 'all':
        run_full_exp(args)
    else:
        run_exp(args)
