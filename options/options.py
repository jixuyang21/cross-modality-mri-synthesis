import argparse

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        
        ##### options for training and test
        self.parser.add_argument('--model', type=str, required=True, help='which model to use')
        self.parser.add_argument('--dataset', type=str, default='brats', help="name of dataset used for training")
        self.parser.add_argument('--mod-in', type=str, default='t1', help="modality of input image")
        self.parser.add_argument('--mod-out', type=str, default='t2', help="modality of output image")
        self.parser.add_argument('--dataroot', type=str, default='/media/hd1/jixuyang/datasets', help="path of data")
        self.parser.add_argument('--gpu', type=int, default=0, help="which gpu to use")
        
        
        ##### options for training only
        # training configs
        self.parser.add_argument('--batch-size', type=int, default=10, help="size of batches")
        self.parser.add_argument('--num-epochs', type=int, default=100, help="total epochs to train")
        # for optimization
        self.parser.add_argument('--lr', type=float, default=0.001, help="learning rate of model to update")
        self.parser.add_argument('--beta1', type=float, default=0.9, help="hyperparameter 'beta1' of Adam optimizer")
        self.parser.add_argument('--beta2', type=float, default=0.999, help="hyperparameter 'beta2' of Adam optimizer")
        # for loss calculation
        self.parser.add_argument('--lambda-adv', type=float, default=1.0, help="weight of GAN loss")
        self.parser.add_argument('--lambda-id', type=float, default=10.0, help='weight of L1 loss')
        self.parser.add_argument('--lambda-vgg', type=float, default=1.0, help="weight of vgg loss")
        self.parser.add_argument('--lambda-cyc', type=float, default=10.0, help="weight of cycle loss")
        self.parser.add_argument('--num-residual-blocks', type=int, default=9, help="number of residual blocks in cyclegan generator")
        # for saving models
        self.parser.add_argument('--save-start', type=int, default=0, help="starting saving models after this epoch")
        self.parser.add_argument('--save-end', type=int, default=100, help="last epoch of saving models")
        self.parser.add_argument('--save-interval', type=int, default=100, help="epoch interval of saving models")
        # for recording training infomation
        self.parser.add_argument('--check-loss', type=bool, default=True, help="whether to record loss curve")

        
        ##### options for test only
        self.parser.add_argument('--model-epoch', type=int, default=100, help="model of which epoch to be loaded for test")
        
        
    def parse(self):
        opt = self.parser.parse_args()
        assert opt.model in ['unet', 'pix2pix', 'pgan', 'cgan', 'cyclegan'], "Invalid model name!"
        
        self.parser.add_argument('--results-path', type=str, default=f"/media/hd1/jixuyang/proj/{opt.model}/results/{opt.dataset}/{opt.mod_in}__{opt.mod_out}", help="path to which the generated images from validation set will be saved")
        self.parser.add_argument('--ckpts-path', type=str, default=f"/media/hd1/jixuyang/proj/{opt.model}/ckpts/{opt.dataset}/{opt.mod_in}__{opt.mod_out}", help="path to which the trained models will be saved")
        self.parser.add_argument('--logs-path', type=str, default=f'/media/hd1/jixuyang/proj/{opt.model}/logs/{opt.dataset}/{opt.mod_in}__{opt.mod_out}', help='path to which the loss of training and validation will be saved')
            
        return self.parser.parse_args()
