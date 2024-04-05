import json

class Evaluator:

    def __init__(self, pred_list, gt_list, question_id_list):
        self.pred_list = pred_list
        self.gt_list = gt_list
        self.question_id_list = question_id_list

    def evaluate(self, ds):
        if ds == 'mme':
            pass
        elif ds == 'seedbench':
            pass
        elif ds == 'mmbench':
            pass
    