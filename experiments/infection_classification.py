"""
Author: Ibrahim Almakky
Date: 29/03/2021
"""

from typing import Optional

import torch
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score

from utils import visualisation
from experiments.experiment import Experiment
from experiments.utils.metrics import per_class_acc
from modelling import ThreeDCNN, densenet
from modelling.resnet import resnet

MODELS = {"3DCNN": ThreeDCNN.cnn, "DenseNet": densenet.densenet, "ResNet": resnet}


class ExpInit(Experiment):
    """
    Infection severity classification experiment class.

    Args:
        Experiment (str): The experiment name. This must
        match the name of json parameters file placed in
        in the params directory.
    """

    def __init__(self, exper_name):
        super().__init__(exper_name)
        self.classifier = self.init_model(self.parameters.model_params)
        self.classifier.to(self.device)
        self.init_training(self.classifier.parameters())

        self.models["classifier"] = self.classifier

        # Additional Termination Conditions
        self.termination_checker.add_termination_cnd(
            "val_f1",
            self.parameters.get_parameter(["training", "termination"], "f1"),
            increasing=True,
        )

    def init_model(self, params):
        try:
            classifier = MODELS[params["class"]](
                params,
                num_classes=self.trainset.num_classes,
                sample_size=self.parameters.get_img_size()[1],
                sample_duration=self.parameters.get_img_size()[0],
            )
        except KeyError as k_error:
            raise ValueError(
                "Specified model is not supported for classification."
            ) from k_error
        return classifier

    def get_roc_metrics(
        self, pred_probs: torch.Tensor, labels: list, classes: Optional[list] = None
    ) -> dict:

        roc_metrics = {}
        pred_probs = pred_probs.detach().tolist()
        macro_roc_auc_ovo = roc_auc_score(
            labels, pred_probs, multi_class="ovo", average="macro"
        )
        weighted_roc_auc_ovo = roc_auc_score(
            labels, pred_probs, multi_class="ovo", average="weighted"
        )
        macro_roc_auc_ovr = roc_auc_score(
            labels, pred_probs, multi_class="ovr", average="macro"
        )
        weighted_roc_auc_ovr = roc_auc_score(
            labels, pred_probs, multi_class="ovr", average="weighted"
        )

        print(
            "One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
            "(weighted by prevalence)".format(macro_roc_auc_ovo, weighted_roc_auc_ovo)
        )
        print(
            "One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
            "(weighted by prevalence)".format(macro_roc_auc_ovr, weighted_roc_auc_ovr)
        )

        roc_metrics["macro_roc_auc_ovo"] = macro_roc_auc_ovo
        roc_metrics["macro_roc_auc_ovr"] = macro_roc_auc_ovr
        roc_metrics["weighted_roc_auc_ovo"] = weighted_roc_auc_ovo
        roc_metrics["weighted_roc_auc_ovr"] = weighted_roc_auc_ovr

        labels = np.array(labels)
        pred_probs = np.array(pred_probs)
        # Compute ROC curve and ROC area for each class
        fpr = {}
        tpr = {}
        roc_auc = {}
        for i in range(0, len(classes)):
            fpr[i], tpr[i], _ = roc_curve(labels, pred_probs[:, i].ravel(), pos_label=i)
            roc_auc[i] = auc(fpr[i], tpr[i])

        roc_metrics["class_specific_roc_auc"] = roc_auc
        roc_metrics["fpr"] = [x.tolist() for x in fpr.values()]
        roc_metrics["tpr"] = [x.tolist() for x in tpr.values()]

        return roc_metrics

    def validate(self):
        self.classifier.eval()
        preds = []
        preds_probs = None
        labels = []
        softmax = torch.nn.Softmax(dim=-1)
        with torch.no_grad():
            for data in self.val_loader:
                val_inp, val_label = data[0].to(self.device), data[1]
                labels += val_label.clone()
                # print(len(labels))
                output = softmax(self.classifier(val_inp))
                if preds_probs is not None:
                    preds_probs = torch.cat((preds_probs, output), dim=0)
                else:
                    preds_probs = output
                # print(preds_probs.shape)
                _, pred = torch.max(output.data, 1)
                preds += pred.cpu()

        roc_metrics = self.get_roc_metrics(
            preds_probs, labels=labels, classes=self.trainset.CLASSES
        )

        self.logger.log_cont_metric(
            "macro_roc_auc_ovo", roc_metrics["macro_roc_auc_ovo"], self.epoch_num
        )
        self.logger.writer.add_scalar(
            "AUC/OVO/Macro", roc_metrics["macro_roc_auc_ovo"], self.epoch_num
        )
        self.logger.log_cont_metric(
            "macro_roc_auc_ovr", roc_metrics["macro_roc_auc_ovr"], self.epoch_num
        )
        self.logger.writer.add_scalar(
            "AUC/OVR/Macro", roc_metrics["macro_roc_auc_ovr"], self.epoch_num
        )
        self.logger.log_cont_metric(
            "weighted_roc_auc_ovo", roc_metrics["weighted_roc_auc_ovo"], self.epoch_num
        )
        self.logger.writer.add_scalar(
            "AUC/OVO/Weighted", roc_metrics["weighted_roc_auc_ovo"], self.epoch_num
        )
        self.logger.log_cont_metric(
            "weighted_roc_auc_ovr", roc_metrics["weighted_roc_auc_ovr"], self.epoch_num
        )
        self.logger.writer.add_scalar(
            "AUC/OVR/Weighted", roc_metrics["weighted_roc_auc_ovr"], self.epoch_num
        )

        cnf = metrics.confusion_matrix(labels, preds)
        self.logger.log_confmatrix(cnf, self.epoch_num)
        self.logger.log_cont_metric(
            "per_class_accuracy",
            str(per_class_acc(cnf).tolist()),
            self.epoch_num,
        )
        report = metrics.classification_report(labels, preds, output_dict=True)
        self.logger.log_cont_metric(
            "classification_report",
            report,
            self.epoch_num,
        )
        val_acc = metrics.accuracy_score(labels, preds)
        val_f1_score = metrics.f1_score(labels, preds, average="macro")
        self.logger.log_cont_metric("validation_f1", val_f1_score, self.epoch_num)

        self.termination_checker.update_metric("val_f1", val_f1_score)

        printed_cnf = visualisation.print_cm(cnf, self.trainset.CLASSES)
        self.logger.writer.add_scalar("Accuracy/Validation", val_acc, self.epoch_num)
        self.logger.writer.add_text("Confusion/Validation", printed_cnf, self.epoch_num)
        return val_acc, cnf

    def run_batches(self):
        self.epoch_labels = []
        self.epoch_preds = []
        super().run_batches()
        train_acc = metrics.accuracy_score(self.epoch_labels, self.epoch_preds)
        print("Train Accuracy = %.3f" % train_acc)
        self.logger.log_cont_metric("train_acc", train_acc, self.epoch_num)

    def batch(self, data):
        # DEBUG
        # self.validate()
        #################

        self.classifier.train()
        self.classifier.zero_grad()

        self.epoch_labels += data[1]

        inp, labels = data[0].to(self.device), data[1].to(self.device)

        output = self.classifier.forward(inp)

        _, pred = torch.max(output, 1)
        self.epoch_preds += pred.detach().cpu()

        err = self.criterion(output, labels)
        err.backward()
        self.optim.step()

        # Output training stats
        if self.batch_num % 2 == 0:
            print(
                "[%d/%d][%d/%d]\tLoss: %.4f"
                % (
                    self.epoch_num,
                    self.epochs,
                    self.batch_num,
                    len(self.train_loader),
                    err.item(),
                )
            )

        self.batch_loss.append(err.item())
