import torch
from torch import nn
from torch import optim


class DummyModel(nn.Module):
    '''
    Dummy model that trains, saves, and loads a single layer Conv2d net.
    Purpose was from SE(2) model always deleting a buffer of continuous filters between calls of model.train() and model.eval().
    '''
    def __init__(self, send_init_to_cuda=False):
        super(DummyModel, self).__init__()
        self.net = nn.Sequential(
                nn.Conv2d(1, 1, kernel_size=(3,3), stride=(1,1), padding=1),
                nn.ReLU()
        )
        if send_init_to_cuda:
            ### with cuda
            self.learnedW = nn.Parameter(torch.rand(1)).cuda()
        else:
            ### no cuda
            self.learnedW = nn.Parameter(torch.rand(1))

    def forward(self, x):
        '''

        :param x: input into conv net
        :return: simple activation
        '''
        x = self.net(x.unsqueeze(0).unsqueeze(0))
        x = x * self.learnedW
        return x

    @staticmethod
    def save_checkpoint(state, filename):
        '''

        :param state: model.state_dict()
        :param filename: file to save as
        :return:
        '''
        # model.eval()
        torch.save(state, filename)

    @staticmethod
    def load_checkpoint(checkpoint_fpath, model, optimizer):
        '''

        :param checkpoint_fpath: path ot checkpoint dict file
        :param model: current model
        :param optimizer: current optimizer
        :return: loaded state dicts for both model and optimizer
        '''
        # model.eval()
        checkpoint = torch.load(checkpoint_fpath)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        return model, optimizer


if __name__ == "__main__":
    ## run on either gpu or cpu
    if torch.cuda.is_available():
        # torch.cuda.set_device(0)
        pretrain_model = DummyModel(send_init_to_cuda=True).to(device='cuda')
        input = torch.rand(50, 50).cuda()
    else:
        pretrain_model = DummyModel(send_init_to_cuda=False).to(device='cpu')
        input = torch.rand(50, 50)

    ## initialize optimizer
    optimizer_pretrain = optim.Adam(pretrain_model.parameters(), lr=1e-3)
    ## run model
    output = pretrain_model(input)

    ## checkpoint dictionary
    checkpoint_dict = {
        'state_dict': pretrain_model.state_dict(),
        'optimizer': optimizer_pretrain.state_dict(),
    }

    ## save checkpoint dictionary to file
    DummyModel().save_checkpoint(checkpoint_dict, 'train.th')

    ## load "pre-trained" dummy model
    path_pretrain = 'train.th'
    # pretrain_model.load_state_dict(torch.load(path_pretrain)['state_dict'])
    pretrain_model, _, = DummyModel().load_checkpoint(path_pretrain, pretrain_model, optimizer_pretrain)

    ## check model parameters by name
    for name, param in pretrain_model.named_parameters():
        print(name)
        print(param)
