import argparse, torch, os

def INFO(inputs):
    print("[StyleGAN2]" + inputs)

def parameters(args_dict):
    INFO("========== Parameter ==========")
    for key in sorted(args_dict.keys()):
        INFO("{:>5} : {}".format(key, args_dict[key]))
    INFO("===============================")

class TrainOptions():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--path', type=str, default='portrait')
        parser.add_argument('--epoch', type=int, default=500)
        parser.add_argument('--fmap_base', type=int, default=8<<10)
        parser.add_argument('--resolution', type=int, default=2<<6)
        parser.add_argument('--mapping_layers', type=int, default=8)
        parser.add_argument('--batch_size', type=int, default=8)
        parser.add_argument('--type', type=str, default='style')
        parser.add_argument('--resume', type=str, default='train_result/models/latest.pth')
        parser.add_argument('--det', type=str, default='train_result')
        self.opts = parser.parse_args()

    def parse(self):
        self.opts.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.opts.type not in ['style', 'origin']:
            raise Exception("You should assign 'style' or 'origin'")

        if not os.path.exists(self.opts.det):
            os.mkdir(self.opts.det)
        if not os.path.exists(os.path.join(self.opts.det, 'images')):
            os.mkdir(os.path.join(self.opts.det, 'images'))
        if not os.path.exists(os.path.join(self.opts.det, 'models')):
            os.mkdir(os.path.join(self.opts.det, 'models'))

        parameters(vars(self.opts))
        return self.opts


class InferenceOptions():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--resume', type=str, default='train_result/model/latest.pth')
        parser.add_argument('--type', type=str, default='style')
        parser.add_argument('--num_face', type=int, default=32)
        parser.add_argument('--det', type=str, default='result.png')
        self.opts = parser.parse_args()

    def parse(self):
        self.opts.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        parameters(vars(self.opts))
        return self.opts
