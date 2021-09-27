from tensorboardX import SummaryWriter
import os
import torch


class Recoder:
    def __init__(self):
        self.metrics = {}

    def record(self, name, value):
        if name in self.metrics.keys():
            self.metrics[name].append(value)
        else:
            self.metrics[name] = [value]

    def summary(self):
        kvs = {}
        for key in self.metrics.keys():
            kvs[key] = sum(self.metrics[key]) / len(self.metrics[key])
            del self.metrics[key][:]
            self.metrics[key] = []
        return kvs


class Logger:
    def __init__(self, args):
        self.writer = SummaryWriter(args.model_dir)
        self.recoder = Recoder()
        self.model_dir = args.model_dir

    def tensor2img(self, tensor):
        # implement according to your data, for example call viz.py
        return tensor.cpu().numpy()

    def record_scalar(self, name, value):
        self.recoder.record(name, value)

    def save_curves(self, epoch):
        kvs = self.recoder.summary()
        for key in kvs.keys():
            self.writer.add_scalar(key, kvs[key], epoch)

    def save_imgs(self, names2imgs, epoch):
        for name in names2imgs.keys():
            self.writer.add_image(name, self.tensor2img(names2imgs[name]), epoch)

    def save_check_point(self, model, epoch, step=0):
        model_name = '{epoch:02d}_{step:06d}.pth'.format(epoch=epoch, step=step)
        path = os.path.join(self.model_dir, model_name)
        # don't save model, which depends on python path
        # save model state dict
        torch.save(model.state_dict(), path)
