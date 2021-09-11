import torch

from data.data_entry import select_eval_loader
from model.model_entry import select_model
from options import prepare_test_args
from utils.logger import Recoder
import numpy as np
import cv2
import os

from utils.viz import label2rgb


class Evaluator:
    def __init__(self):
        args = prepare_test_args()
        self.args = args
        self.model = select_model(args)
        self.model.load_state_dict(torch.load(args.load_model_path).state_dict())
        self.model.eval()
        self.val_loader = select_eval_loader(args)
        self.recoder = Recoder()

    def eval(self):
        for i, data in enumerate(self.val_loader):
            img, pred, label = self.step(data)
            metrics = self.compute_metrics(pred, label)

            for key in metrics.keys():
                self.recoder.record(key, metrics[key])
            if i % self.args.viz_freq:
                self.viz_per_batch(img, pred, label, i)

        metrics = self.recoder.summary()
        result_txt_path = os.path.join(self.args.result_dir, 'result.txt')

        # write metrics to result dir,
        # you can also use pandas or other methods for better stats
        with open(result_txt_path, 'w') as fd:
            fd.write(str(metrics))

    def compute_metrics(self, pred, gt):
        # you can call functions in metrics.py
        l1 = (pred - gt).abs().mean()
        metrics = {
            'l1': l1
        }
        return metrics

    def viz_per_batch(self, img, pred, gt, step):
        # call functions in viz.py
        # here is an example about segmentation
        img_np = img[0].cpu().numpy().transpose((1, 2, 0))
        pred_np = label2rgb(pred[0].cpu().numpy())
        gt_np = label2rgb(gt[0].cpu().numpy())
        viz = np.concatenate([img_np, pred_np, gt_np], axis=1)
        viz_path = os.path.join(self.args.result_dir, "%04d.jpg" % step)
        cv2.imwrite(viz_path, viz)
    
    def step(self, data):
        img, label = data
        # warp input
        img = Variable(img).cuda()
        label = Variable(label).cuda()

        # compute output
        pred = self.model(img)
        return img, label, pred


def eval_main():
    evaluator = Evaluator()
    evaluator.eval()


if __name__ == '__main__':
    eval_main()
