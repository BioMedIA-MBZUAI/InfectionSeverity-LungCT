"""
Author: Ibrahim Almakky
Date: 21/04/2021

"""

from math import ceil
import os
import threading

import numpy as np
import torch
import sklearn.metrics as metrics
import monai

from experiments.experiment import Experiment
from experiments.utils import ct
from experiments.utils.metrics import per_class_acc
from utils import visualisation
from modelling import ThreeDCNN
from modelling import densenet
from modelling import resnet
from modelling.region_prop_pipe import RegionPropPipline
from modelling.region_prop import cuboid_classifier

def save_pred_masks(
    img_ids: list,
    cuboid_coords: list,
    cuboid_ids: list,
    path: str,
    img_dims: tuple,
    cube_dims: tuple,
    orig_img_dims=None,
    pix_dims=None,
):
    assert len(img_ids) == len(cuboid_ids)
    if not os.path.isdir(path):
        os.mkdir(path)

    for i, img_id in enumerate(img_ids):
        cuboids = [cuboid_coords[x] for x in cuboid_ids[i]]
        mask = ct.generate_3dmask(img_dims, cube_dims, cuboids)
        mask = np.transpose(mask, (2, 1, 0))

        if orig_img_dims:
            mask = ct.resize_volume(mask, orig_img_dims[i])

        ct.save_nifti(
            mask,
            os.path.join(
                path,
                str.replace(img_id, ".pt", ""),
            ),
            pixdim=pix_dims[i],
        )


def count_infection(
    img_ids: list,
    img_dims: tuple,
    cube_dims: tuple,
    cuboid_ids: list,
    cuboid_coords: list,
    dataset_path: str,
    orig_img_dims: list,
):
    """
    Calculate the volume level infection coverage
    for a set of selected cuboids.
    """
    nibable_reader = monai.data.NibabelReader()
    names = []
    ratios = []
    for i, img_id in enumerate(img_ids):

        mask_file = os.path.join(
            dataset_path,
            "MosMedData/masks/" + str.split(img_id, ".pt")[0] + "_mask.nii",
        )
        if os.path.exists(mask_file):
            ct_img = nibable_reader.read(mask_file)  # load segm mask for img i
            segmentation_mask = ct_img.get_fdata()
        else:
            continue

        cuboids = [cuboid_coords[x] for x in cuboid_ids[i]]
        pred_mask = ct.generate_3dmask(
            img_dims=img_dims,
            cube_dims=cube_dims,
            cubes=cuboids,
        )
        pred_mask = ct.resize_volume(pred_mask, orig_img_dims[i])

        pred_mask[pred_mask >= 0.00001] = 1
        pred_mask[pred_mask < 0.00001] = 0

        infec_total = np.sum(segmentation_mask)

        # print(infec_total)
        # print(cub_total)

        intersection = np.logical_and(segmentation_mask, pred_mask)
        cub_infec_total = np.sum(intersection)

        names.append(img_id)
        ratios.append(cub_infec_total / infec_total)
    avrg_coverage = np.average(ratios)
    return avrg_coverage, (names, ratios)


def count_cube_level_infection(
    img_ids: list,
    cuboid_coords: list,
    cuboid_ids: list,
    img_dims: tuple,
    cube_dims: tuple,
    labels: list,
    preds: list,
    orig_img_dims=None,
    dataset_path=str,
):
    """
    TODO: This function requires further testing
    to ensure that the case where all cuboids are
    selected from the volume then a full infection
    coverage is achieved.
    This is not happening with the current version.
    """
    assert len(img_ids) == len(cuboid_ids)

    img_infec_info = {}
    total_imgs_inf = 0
    total_imgs_cub_inf = 0
    for i, img_id in enumerate(img_ids):
        nibable_reader = monai.data.NibabelReader()
        mask_file = os.path.join(
            dataset_path,
            "MosMedData/masks/" + str.split(img_id, ".pt")[0] + "_mask.nii",
        )

        segmentation_mask = None
        if os.path.exists(mask_file):
            ct_img = nibable_reader.read(mask_file)  # load segm mask for img i
            segmentation_mask = ct_img.get_fdata()
        else:
            continue

        segmentation_mask = segmentation_mask.swapaxes(0, 2)

        cuboids_orig = []
        cuboids = [cuboid_coords[x] for x in cuboid_ids[i]]

        total_inf = np.sum(segmentation_mask)
        total_imgs_inf += total_inf
        total_cubs_inf = 0

        # z, w, h
        ratios = (
            (orig_img_dims[i][0] / img_dims[1]),
            (orig_img_dims[i][1] / img_dims[2]),
            (orig_img_dims[i][2] / img_dims[0]),
        )

        for cube in cuboids:
            cube_x_orig = int(ratios[0] * cube[1])
            cube_y_orig = int(ratios[1] * cube[2])
            cube_z_orig = int(ratios[2] * cube[0])

            cube_coord_orig = [
                cube_z_orig,
                cube_x_orig,
                cube_y_orig,
            ]

            orig_cuboid_data = segmentation_mask[
                cube_z_orig : ceil(cube_z_orig + (cube_dims[0] * ratios[2])) + 1,
                cube_x_orig : ceil(cube_x_orig + (cube_dims[1] * ratios[0])) + 1,
                cube_y_orig : ceil(cube_y_orig + (cube_dims[2] * ratios[1])) + 1,
            ]
            total_cub_vol = (
                orig_cuboid_data.shape[0]
                * orig_cuboid_data.shape[1]
                * orig_cuboid_data.shape[2]
            )  # total area pixeles
            total_cub_inf = np.sum(orig_cuboid_data)  # total infection pixeles
            total_cubs_inf += total_cub_inf
            cuboids_orig.append([cube_coord_orig, total_cub_inf, total_cub_vol])

        total_imgs_cub_inf += total_cubs_inf
        img_infec_info[img_id] = {
            "total": {
                "total_cubs_inf": total_cubs_inf,
                "total_inf": total_inf,
                "total_ratio": total_cubs_inf / total_inf,
                "label": labels[i].item(),
                "pred": preds[i].item(),
            },
            "breakdown": cuboids_orig,
        }

    return (total_imgs_cub_inf / total_imgs_inf), img_infec_info


class ExpInit(Experiment):
    """The experiment class for region proposals method.
    The method utlises a small model to nominate possible
    cuboids. Those cuboids are fed into a classification model
    that classifies each of the cuboids. The overall classification
    is then based on the majority classification.

    This experiment requires a "region_proposal" dict in the parameters
    file that includes the following:
    - "cuboid_size": 3-dimensional list containing the size of the
    cuboids to be nominated and classified.
    - "num_cuboids":
    - "factors": The factors used to calculate the cuboids ground
    truth. Those are as follows:
        - "correct": The confidence value that is to be assigned
        to correctly classified cuboids.
        - "wrong": The confidence value to be assigned to
        misclassified cuboids.
        - "increase_conf": The factor at which to increase confidence
        in the non-selected cuboids when an overall wrong classification
        is acheived.
        - "decrease_conf": The factor to decrease confidence in the non-selected
        cuboids when an overall correct classification is acheived.
    - "model": The architecture definition and hyperparameters for
    the proposal model. This also contains the training parameters
    for the proposal model.

    Args:
        Experiment (str): The name of the experiment to be carried
        out. This must match with the name of the parameters json
        file.
    """

    MODELS = {
        "3DCNN": ThreeDCNN.cnn,
        "DenseNet": densenet.densenet,
        "ResNet": resnet.resnet,
        "CuboidClassifier": cuboid_classifier,
    }

    def __init__(self, params_file):
        super().__init__(params_file)

        self.epoch_num = 0
        self.batch_num = 0

        # Get a training sample
        sample_data = next(iter(self.train_loader))
        sample_data = sample_data[0]

        # Get params for JSON
        region_exp_params = self.parameters.get_params_branch("region_proposal")
        self.cuboid_size = region_exp_params["cuboid_size"]
        self.num_cuboids = region_exp_params["num_cuboids"]
        self.cuboids_gt_factors = region_exp_params["factors"]

        self.termination_checker.add_termination_cnd(
            "val_f1",
            self.parameters.get_parameter(["training", "termination"], "f1"),
            increasing=True,
        )
        self.termination_checker.add_termination_cnd(
            "val_acc",
            self.parameters.get_parameter(["training", "termination"], "val_acc"),
            increasing=True,
        )

        self.scheduler = None  # For Experiment class

        agg_params = region_exp_params["aggregation"]

        self.region_prop_pipline = RegionPropPipline(
            img_size=sample_data.shape[2:5],
            cub_size=self.cuboid_size,
            num_cubs=self.num_cuboids,
            num_classes=self.trainset.num_classes,
            propal_gen_params=region_exp_params["model"],
            cuboid_classifier_params=self.parameters.model_params,
            volume_classifier_params=agg_params["params"],
            rlogger=self.logger,
            device=self.device,
            approach=agg_params["params"]["pos_approach"],
            temperature=self.cuboids_gt_factors["temperature"],
            prox_effect_epoch=agg_params["params"]["prox_effect_epoch"],
        ).to(self.device)

        self.models["region_prop_pipeline"] = self.region_prop_pipline

        # Proposal Generator #
        self.proposal_criterion = self.init_criterion(region_exp_params["criterion"])
        self.prop_optim, self.prop_sched = self.init_optim(
            self.region_prop_pipline.proposal_gen.parameters(),
            region_exp_params["lr"],
            region_exp_params["optimizer"],
            region_exp_params["scheduler"],
        )

        prop_training_params = self.parameters.get_train_params()

        self.cuboid_classifier_criterion = self.init_criterion(
            prop_training_params["criterion"]
        )
        (
            self.cuboid_classifier_optim,
            self.cuboid_classifier_scheduler,
        ) = self.init_optim(
            self.region_prop_pipline.cuboid_classifier.parameters(),
            prop_training_params["lr"],
            prop_training_params["optimizer"],
            prop_training_params["scheduler"],
        )

        self.prop_loss = []
        # Aggregation MLP
        self.img_classify = None

        self.img_classify_criterion = self.init_criterion(
            agg_params["params"]["criterion"]
        )

        self.img_classify_optim, self.img_classify_schduler = self.init_optim(
            [
                {"params": self.region_prop_pipline.volume_classifier.parameters()},
                {"params": self.region_prop_pipline.pos_enc},
            ],
            agg_params["params"]["lr"],
            agg_params["params"]["optimizer"],
            agg_params["params"]["scheduler"],
        )

        self.optims["cuboid_classifier_optim"] = self.cuboid_classifier_optim
        self.optims["prop_optim"] = self.prop_optim
        self.optims["img_classify_optim"] = self.img_classify_optim

        try:
            self.load_cache(
                path=self.parameters.get_parameter(["general"], "checkpoint")
            )
        except KeyError:
            pass

        self.val_cuboid_heatmap = torch.zeros(self.region_prop_pipline.total_cuboids)
        self.train_cuboid_heatmap = torch.zeros(self.region_prop_pipline.total_cuboids)
        self.train_class_cuboid_heatmap = torch.zeros(
            len(self.trainset.CLASSES), self.region_prop_pipline.total_cuboids
        )
        self.val_class_cuboid_heatmap = torch.zeros(
            len(self.trainset.CLASSES), self.region_prop_pipline.total_cuboids
        )

    def concat_output(self, output, batch_size, num_classes):
        concat_output = torch.Tensor(batch_size, num_classes).to(self.device)

        j = 0
        for i in range(0, output.shape[0], self.num_cuboids):
            concat_output[j][:] = output[i : i + self.num_cuboids].mean(0)
            j += 1

        return concat_output

    def calc_prop_gt(self, top_prop, class_out, labels, num_classes):
        """
        Calculate the proposal's ground truth based on the
        classiciation outcome:
        - The confidence is lowered in cuboids that do not lead
        to a correct classification.
        - The confidence is maximized for cuboids that produce
        correct classifications.
        - The confidence is narrowly increased in all other cuboids
        when the majority of the proposed cuboids produce wrong
        classification.

        Args:
            porp_output ([type]): [description]
            top_prop ([type]): [description]
            class_out ([type]): The classification result
            from the classifier.
            labels ([type]): [description]

        Returns:
            [torch.Tensor]: The proposals ground truth based on the classification
            of each of the proposals put forward.
        """
        batch_size = top_prop.shape[0]

        wrong_cub_conf = self.cuboids_gt_factors["wrong"]
        correct_cub_conf = self.cuboids_gt_factors["correct"]
        increase_conf_factor = self.cuboids_gt_factors["increase_conf"]
        decrease_conf_factor = self.cuboids_gt_factors["decrease_conf"]

        prop_gt = torch.ones(batch_size, len(self.cubiods)).to(self.device)

        concat_output = self.concat_output(class_out, batch_size, num_classes)
        _, cuboids_classes = class_out.max(dim=1)

        for sample_i in range(0, batch_size):
            _, pred_ind = torch.max(concat_output[sample_i], 0)
            prop_idx = sample_i * self.num_cuboids
            # Correctly classified sample
            if pred_ind == labels[sample_i]:
                # Minimize the confidence in the rest of the
                # cuboids
                prop_gt[sample_i] = prop_gt[sample_i] * decrease_conf_factor
                for prop in top_prop[sample_i]:
                    # Maximize the confidence in ONLY in the cuboids
                    # that got the right classification result
                    if cuboids_classes[prop_idx] == labels[sample_i]:
                        prop_gt[sample_i][prop] = correct_cub_conf
                    else:
                        prop_gt[sample_i][prop] = wrong_cub_conf
                    prop_idx += 1
            # Wrongly classified sample
            else:
                # Narrowly increase confidence in the rest of the
                # cuboids
                prop_gt[sample_i] = prop_gt[sample_i] * increase_conf_factor
                for prop in top_prop[sample_i]:
                    # Maximize the confidence in the cuboids
                    # that got the right classification result
                    if cuboids_classes[prop_idx] == labels[sample_i]:
                        prop_gt[sample_i][prop] = correct_cub_conf
                    else:
                        prop_gt[sample_i][prop] = wrong_cub_conf
                    prop_idx += 1

        return prop_gt

    def run_batches(self):

        self.epoch_labels = []
        self.epoch_preds = []

        self.prop_loss = []
        self.aggr_loss = []

        self.train_pred_cuboids = []
        self.img_ids = []
        self.orig_dims = []

        super().run_batches()

        train_acc = metrics.accuracy_score(self.epoch_labels, self.epoch_preds)
        print("Train Accuracy = %.3f" % train_acc)
        self.logger.log_cont_metric("train_acc", train_acc, self.epoch_num)

        avrg_prop_loss = np.mean(self.prop_loss)
        self.prop_sched.step(avrg_prop_loss)
        self.logger.writer.add_scalar("Loss/PropTrain", avrg_prop_loss, self.epoch_num)
        self.logger.log_cont_metric("proposal_loss", avrg_prop_loss, self.epoch_num)

        if len(self.aggr_loss) > 0:
            avrg_agg_loss = np.mean(self.aggr_loss)
            self.img_classify_schduler.step(avrg_agg_loss)
            self.logger.writer.add_scalar(
                "Loss/Aggregator", avrg_agg_loss, self.epoch_num
            )
            self.logger.log_cont_metric(
                "Aggregator_loss", avrg_agg_loss, self.epoch_num
            )

        if len(self.train_loss) > 0:
            self.cuboid_classifier_scheduler.step(self.train_loss[-1])

        inf_coverage_score, inf_coverage = count_infection(
            self.img_ids,
            self.trainset.img_size,
            self.cuboid_size,
            self.train_pred_cuboids,
            self.region_prop_pipline.cuboids_coords,
            dataset_path=self.parameters.get_parameter(["dataset"], "data_path"),
            orig_img_dims=self.orig_dims,
        )
        self.logger.log_infection_coverage(inf_coverage, epoch=self.epoch_num)
        self.logger.writer.add_scalar(
            "Loss/Inf_Cov_Score_Train", inf_coverage_score, self.epoch_num
        )

        avrg_train_cuboid_heatmap = (
            (self.train_cuboid_heatmap / torch.sum(self.train_cuboid_heatmap))
            .numpy()
            .tolist()
        )
        avrg_val_cuboid_heatmap = (
            (self.val_cuboid_heatmap / torch.sum(self.val_cuboid_heatmap))
            .numpy()
            .tolist()
        )
        self.logger.log_metric("CubeHeatmap/Train", avrg_train_cuboid_heatmap)
        self.logger.log_metric("CubeHeatmap/Validation", avrg_val_cuboid_heatmap)

        # Class specific cuboid heatmaps
        for class_ind in range(0, self.train_class_cuboid_heatmap.shape[0]):
            avrg_train = (
                (
                    self.train_class_cuboid_heatmap[class_ind]
                    / torch.sum(self.train_class_cuboid_heatmap[class_ind])
                )
                .numpy()
                .tolist()
            )
            avrg_val = (
                (
                    self.val_class_cuboid_heatmap[class_ind]
                    / torch.sum(self.val_class_cuboid_heatmap[class_ind])
                )
                .numpy()
                .tolist()
            )
            self.logger.log_metric("CubeHeatmap/Train/" + str(class_ind), avrg_train)
            self.logger.log_metric("CubeHeatMap/Validation/" + str(class_ind), avrg_val)

    def log_selected_cuboids(self, cub_inds, labels, split="train"):
        i = 0
        for cub_ind in cub_inds:
            try:
                if split == "train":
                    self.train_cuboid_heatmap[cub_ind] += 1
                    self.train_class_cuboid_heatmap[labels[i]][cub_ind] += 1
                elif split == "val":
                    self.val_cuboid_heatmap[cub_ind] += 1
                    self.val_class_cuboid_heatmap[labels[i]][cub_ind] += 1
                i += 1
            except IndexError:
                self.logger.log_warning(
                    "Heamap log error",
                    value=None,
                    cub_ind=cub_ind,
                    train_cub_heatmap_shape=self.train_cuboid_heatmap.shape,
                )

    def get_cuboids_classes(self, batch_size, labels):
        """Resize the (Bx1) labels vector to (BxC), where
        C is the number of cuboids each sample is being
        subdivided into.

        Args:
            batch_inp ([type]): [description]
            labels ([type]): [description]

        Returns:
            [type]: [description]
        """

        new_labels = torch.zeros(batch_size, dtype=torch.long).to(self.device)

        start, end = 0, self.num_cuboids
        for label in labels:
            new_labels[start:end] = label
            start, end = start + self.num_cuboids, end + self.num_cuboids

        return new_labels

    def epoch(self):
        self.logger.log_cont_metric(
            "prop_optim_lr",
            self.prop_optim.param_groups[0]["lr"],
            self.epoch_num,
        )
        super().epoch()

    def batch(self, data):
        # DEBUG ###
        # self.validate()
        ##################

        self.region_prop_pipline.train()

        # Zero gradients
        self.img_classify_optim.zero_grad()
        self.prop_optim.zero_grad()
        self.cuboid_classifier_optim.zero_grad()

        # Read the img info
        img_headers = data[2]
        for img_header in img_headers:
            self.img_ids += [img_header["file_name"]]
            # TODO - Make this better without hardcoded numbers
            self.orig_dims += [(512, 512, img_header["orig_size"][2])]

        self.epoch_labels += data[1]

        inp, labels = data[0].to(self.device), data[1].to(self.device)

        cuboid_labels = self.get_cuboids_classes(
            self.num_cuboids * inp.shape[0],
            labels.tolist(),
        )

        (
            agg_classification,
            cub_class,
            prop_output,
            top_prop,
        ) = self.region_prop_pipline.train_forward(inp)

        self.train_pred_cuboids += top_prop.tolist()

        prop_gt = self.region_prop_pipline.weak_sup_gt(
            prop_output,
            top_prop,
            torch.nn.functional.softmax(cub_class.detach(), dim=-1),
            labels,
            agg_classification,
            self.epoch_num,
            d_coef=self.cuboids_gt_factors["decrease_conf"],
            c_coef=self.cuboids_gt_factors["increase_conf"],
        )

        self.log_selected_cuboids(top_prop, cuboid_labels, split="train")

        class_err = self.img_classify_criterion(agg_classification, labels)
        class_err.backward()
        self.img_classify_optim.step()

        _, pred = torch.max(agg_classification, 1)
        self.epoch_preds += pred.cpu().detach()

        self.aggr_loss.append(class_err.item())

        # Output training stats
        if self.batch_num % 2 == 0:
            print(
                "[%d/%d][%d/%d]\tAggregate Loss: %.4f"
                % (
                    self.epoch_num,
                    self.epochs,
                    self.batch_num,
                    len(self.train_loader),
                    class_err.item(),
                )
            )

        cuboid_labels = torch.clip(cuboid_labels, min=0, max=1)

        cuboid_classification_error = self.cuboid_classifier_criterion(
            cub_class, cuboid_labels
        )
        cuboid_classification_error.backward()
        self.cuboid_classifier_optim.step()
        self.batch_loss.append(cuboid_classification_error.item())

        prop_output = torch.sigmoid(prop_output)
        prop_err = self.proposal_criterion(prop_output, prop_gt)
        prop_err.backward()
        self.prop_optim.step()
        self.prop_loss.append(prop_err.item())

        # Output training stats
        if self.batch_num % 2 == 0:
            print(
                "[%d/%d][%d/%d]\tLoss: %.4f, Proposal generator: %.4f"
                % (
                    self.epoch_num,
                    self.epochs,
                    self.batch_num,
                    len(self.train_loader),
                    cuboid_classification_error.item(),
                    prop_err.item(),
                )
            )

    def validate(self):
        self.region_prop_pipline.eval()

        preds = []
        labels = []
        cuboids = []
        img_ids = []
        orig_dims = []
        pix_dims = []

        img_dims = None

        with torch.no_grad():
            for data in self.val_loader:
                val_inp, val_label = data[0].to(self.device), data[1]
                labels += val_label

                # Read the img info
                img_headers = data[2]
                for img_header in img_headers:
                    img_ids += [img_header["file_name"]]
                    # TODO - Make this better without hardcoded numbers
                    orig_dims += [(512, 512, img_header["orig_size"][2])]
                    pix_dims += [img_header["pix_dim"]]

                img_dims = data[0].shape[2:5]

                (
                    agg_classification,
                    top_prop,
                ) = self.region_prop_pipline(val_inp)
                cuboids += top_prop.tolist()

                new_labels = self.get_cuboids_classes(
                    self.num_cuboids * val_inp.shape[0], val_label
                )

                self.log_selected_cuboids(top_prop.flatten(), new_labels, split="val")

                _, pred = torch.max(agg_classification, 1)
                preds += pred.cpu()

        cnf = metrics.confusion_matrix(labels, preds)
        print(cnf)
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

        self.logger.log_confmatrix(cnf, self.epoch_num)
        val_acc = metrics.accuracy_score(labels, preds)
        val_f1_score = metrics.f1_score(labels, preds, average="macro")

        self.termination_checker.update_metric("val_f1", val_f1_score)
        self.termination_checker.update_metric("val_acc", val_acc)

        printed_cnf = visualisation.print_cm(cnf, self.trainset.CLASSES)
        self.logger.writer.add_scalar("Validation/Accuracy", val_acc, self.epoch_num)
        self.logger.writer.add_scalar("Validation/F1", val_f1_score, self.epoch_num)
        self.logger.log_cont_metric("validation_f1", val_f1_score, self.epoch_num)
        self.logger.writer.add_text("Validation/Confusion", printed_cnf, self.epoch_num)

        inf_coverage_score, inf_coverage = count_infection(
            img_ids,
            img_dims,
            self.cuboid_size,
            cuboids,
            self.region_prop_pipline.cuboids_coords,
            dataset_path=self.parameters.get_parameter(["dataset"], "path"),
            orig_img_dims=orig_dims,
        )
        self.logger.log_infection_coverage(inf_coverage, epoch=self.epoch_num)
        self.logger.writer.add_scalar(
            "Validation/Inf_Cov_Score", inf_coverage_score, self.epoch_num
        )

        if val_acc > np.max(self.val_accs) and self.epoch_num > 0:
            save_preds_thread = threading.Thread(
                target=save_pred_masks,
                name="SavePreds",
                args=(
                    img_ids,
                    self.region_prop_pipline.cuboids_coords,
                    cuboids,
                    os.path.join(self.logger.get_log_dir(), "pred_masks"),
                    img_dims,
                    self.cuboid_size,
                    orig_dims,
                    pix_dims,
                ),
            )
            save_preds_thread.start()

        return val_acc, cnf
