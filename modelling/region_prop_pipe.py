import numpy as np
import torch
import torch.nn as nn

from experiments.utils import logger
from modelling.densenet import densenet
from modelling.resnet import resnet
from modelling import ThreeDCNN
from modelling.mlp import MLP
from modelling.region_prop import APPROACHES, cuboid_classifier
from modelling.ensemble.classifier import ClassifierEnsemble

MODELS = {
    "3DCNN": ThreeDCNN.cnn,
    "DenseNet": densenet,
    "ResNet": resnet,
    "CuboidClassifier": cuboid_classifier,
    "ensemble": ClassifierEnsemble,
    "MLP": MLP,
}


def mesh_3D(image_shape: torch.Size, mesh_size: tuple):
    # Original image coords
    assert len(image_shape) == 3

    img_z, img_x, img_y = image_shape[0], image_shape[1], image_shape[2]

    # Mesh Cuboid dims
    cub_z, cub_x, cub_y = mesh_size[0], mesh_size[1], mesh_size[2]

    x_num_cubs = int((img_x / cub_x))
    y_num_cubs = int((img_y / cub_y))
    z_num_cubs = int((img_z / cub_z))

    x_coords = np.linspace(0, img_x - cub_x, x_num_cubs).astype(int)
    y_coords = np.linspace(0, img_y - cub_y, y_num_cubs).astype(int)
    z_coords = np.linspace(0, img_z - cub_z, z_num_cubs).astype(int)

    # Get all combinations of coordinates
    cuboids = [(z, x, y) for x in x_coords for y in y_coords for z in z_coords]

    return cuboids


def proximity_mat(cuboids: list, cuboid_dims: tuple) -> torch.FloatTensor:
    # Prepare proximity matrix of num_cuboidsxnum_cuboids size
    cuboids = np.array(cuboids)
    num_cuboids = cuboids.shape[0]
    proximity = np.zeros(shape=(num_cuboids, num_cuboids))
    for i in range(num_cuboids):
        for j in range(num_cuboids):
            proximity[i, j] = (
                (np.abs(cuboids[i][0] - cuboids[j][0]) // cuboid_dims[0])
                + (np.abs(cuboids[i][1] - cuboids[j][1]) // cuboid_dims[1])
                + (np.abs(cuboids[i][2] - cuboids[j][2]) // cuboid_dims[2])
            )
    proximity = proximity / np.max(proximity)
    proximity[proximity == 0] = float(
        "Inf"
    )  # Remove the impact of the cuboid on itself
    proximity = 2 * (1 / (1 + np.exp(20 * proximity)))
    return torch.tensor(proximity, dtype=torch.float32)


def proximity_mult(probs: torch.Tensor, proximity: torch.Tensor):
    assert proximity.shape[0] == proximity.shape[1]
    num_cuboids = proximity.shape[0]
    batch_size = probs.shape[0]
    adds = torch.zeros((batch_size, num_cuboids), device=probs.device)
    for i in range(num_cuboids):
        # exp_prox = 1 / (torch.exp(proximity[i, :]))
        exp_prox = proximity[i, :]
        exp_prox = exp_prox.repeat(batch_size, 1)
        factors = exp_prox * probs
        factors = factors / torch.max(factors)  # Normalize the factors
        # This is the threshold afterwich the impact of the cuboid will be considered
        factors[factors == 0] = 0.85
        factors = -2 * ((1 / (1 + torch.exp(factors - 0.85))) - 0.5)
        factors[factors < 0] = 0  # Apply relu
        adds[:, i] = torch.mean(factors, dim=1)
    # adds = 1 / (prox_factor + torch.exp(3 - adds))
    # adds = adds / torch.mean(adds, dim=0)
    new_probs = probs + adds
    print(torch.mean(new_probs - probs))
    return new_probs


class RegionPropPipline(nn.Module):
    def __init__(
        self,
        img_size: torch.Size,
        cub_size: tuple,
        num_cubs: int,
        num_classes: int,
        propal_gen_params: dict,
        cuboid_classifier_params: dict,
        volume_classifier_params: dict,
        rlogger: logger.Logger,
        device: torch.device,
        approach="pos_enc_vec",
        temperature=None,
        prox_effect_epoch=-1,
    ):
        super().__init__()

        self.logger = rlogger

        self.cuboids_coords = mesh_3D(img_size, cub_size)
        self.total_cuboids = len(self.cuboids_coords)
        self.cuboid_size = cub_size
        self.num_cubs = num_cubs
        self.approach = approach
        self.temperature = temperature
        self.num_classes = num_classes

        self.prox_effect_epoch = prox_effect_epoch
        self.proximity = proximity_mat(
            cuboids=self.cuboids_coords, cuboid_dims=self.cuboid_size
        ).to(device)

        self.embed_dim = cuboid_classifier_params["fcn"][-1]

        self.APPROACHES = {
            "pos_enc_vec": self._forward_pos_enc_vec,
            "topk_cat": self._forward_concat_topk,
            "topk_and_conf": self._forward_concat_topk_and_conf,
            "forward_pos_enc_vec_binary_cub_classifier": self._forward_pos_enc_vec_binary_cub_classifier,
            "forward_pos_enc_vec_with_prop_feat": self._forward_pos_enc_vec_with_prop_feat,
        }

        self.proposal_gen = MODELS[propal_gen_params["class"]](
            params=propal_gen_params,
            num_classes=len(self.cuboids_coords),
            sample_size=img_size[1],
            sample_duration=img_size[0],
        )
        self.prop_sig = torch.nn.Sigmoid()

        self.cuboid_classifier = MODELS[cuboid_classifier_params["class"]](
            cuboid_classifier_params,
            num_classes=2,  # Binary classification of either infected or not
            sample_size=self.cuboid_size[1],
            sample_duration=self.cuboid_size[0],
        )

        self.pos_enc = nn.Parameter(
            torch.randn(
                1,
                self.total_cuboids,
                self.embed_dim,
            ),
            requires_grad=True,
        )

        if self.approach == "forward_pos_enc_vec_with_prop_feat":
            prop_features = propal_gen_params["fcn"][-1]
        else:
            prop_features = None
        self.volume_classifier = self.init_volume_classifier(
            params=volume_classifier_params,
            cub_features=self.embed_dim,
            prop_features=prop_features,
        )
        self.softmax = nn.Softmax(dim=-1)

    def init_volume_classifier(self, params, cub_features, prop_features=None) -> MLP:
        # Get model params
        try:
            layers = params["layers"]
            dropout = params["dropout"]
            hidden_activations = params["activations"]
            out_activation = params["out_activation"]
            batch_norm = params["batch_norm"]
            pos_approach = params["pos_approach"]
        except KeyError as k_error:
            self.logger.log_error("Missing img classifier parameter")
            raise KeyError("Missing img classifier parameter") from k_error

        # Add the size of the last layer based on the
        # number of classes
        if layers[0] == -1:
            emb_size = cub_features
            if pos_approach == "topk_cat":
                emb_size += 1
            elif pos_approach == "topk_and_conf":
                emb_size += 2
            layers[0] = emb_size * self.num_cubs
            if prop_features is not None:
                layers[0] += prop_features
        if layers[-1] == -1:
            layers[-1] = self.num_classes

        image_classifer = MLP(
            layers,
            dropout,
            batch_norm=batch_norm,
            activation=hidden_activations,
            final_activation=out_activation,
        )

        return image_classifer

    def extract_cuboid(self, img_tensor, bl_coords):
        return img_tensor[
            0,
            bl_coords[0] : bl_coords[0] + self.cuboid_size[0],
            bl_coords[1] : bl_coords[1] + self.cuboid_size[1],
            bl_coords[2] : bl_coords[2] + self.cuboid_size[2],
        ]

    def extract_batch_cuboids(self, img_batch, cuboids: list):

        device = img_batch.device

        batch_size = img_batch.shape[0]
        num_cuboids = cuboids.shape[1]

        new_batch = torch.zeros(
            batch_size * num_cuboids,
            1,
            self.cuboid_size[0],
            self.cuboid_size[1],
            self.cuboid_size[2],
        ).to(device)

        z = 0
        for i in range(0, batch_size):
            for c in range(0, num_cuboids):
                new_batch[z] = self.extract_cuboid(
                    img_batch[i], self.cuboids_coords[cuboids[i][c]]
                )
                z += 1

        return new_batch

    def _forward_concat_topk_and_conf(self, x):
        prop_cubs = self.proposal_gen(x)
        prop_cubs = self.prop_sig(prop_cubs)
        top_conf, top_prop = torch.topk(prop_cubs, self.num_cubs, -1)

        self.log_cuboid_probably_stats(prop_cubs)

        select_cubs = self.extract_batch_cuboids(x, top_prop)

        cub_class = self.cub_classifier(select_cubs)
        cub_features = self.cub_classifier.get_latent_features().detach()

        flat_topk = torch.flatten(top_prop)
        cub_features = torch.cat(
            (cub_features, flat_topk.view(flat_topk.shape[0], 1)), dim=1
        )

        flat_top_conf = torch.flatten(top_conf.detach())
        cub_features = torch.cat(
            (cub_features, flat_top_conf.view(flat_top_conf.shape[0], 1)), dim=1
        )

        cub_features = cub_features.view(
            -1,
            (self.embed_dim + 2) * self.num_cubs,
        )
        agg_classification = self.volume_classifier(cub_features)

        return agg_classification, cub_class, prop_cubs, top_prop

    def _forward_dynamic(self, x):
        prop_cubs = self.proposal_gen(x)

        prop_cubs = self.prop_sig(prop_cubs)

        self.log_cuboid_probably_stats(prop_cubs)

        _, top_prop = torch.topk(prop_cubs, self.num_cubs, -1)

        select_cubs = self.extract_batch_cuboids(x, top_prop)

        cub_class = self.cub_classifier(select_cubs)
        cub_features = self.cub_classifier.get_latent_features().detach()

        flat_topk = torch.flatten(top_prop)
        cub_features = torch.cat(
            (cub_features, flat_topk.view(flat_topk.shape[0], 1)), dim=1
        )

        cub_features = cub_features.view(
            -1,
            (self.embed_dim + 1) * self.num_cubs,
        )

        # cub_features = self.dropout(cub_features)
        agg_classification = self.volume_classifier(cub_features)

        return agg_classification, cub_class, prop_cubs, top_prop

    def _forward_concat_topk(self, x):
        prop_cubs = self.proposal_gen(x)
        prop_cubs = self.prop_sig(prop_cubs)
        _, top_prop = torch.topk(prop_cubs, self.num_cubs, -1)

        self.log_cuboid_probably_stats(prop_cubs)

        select_cubs = self.extract_batch_cuboids(x, top_prop)

        cub_class = self.cub_classifier(select_cubs)
        cub_features = self.cub_classifier.get_latent_features().detach()

        flat_topk = torch.flatten(top_prop)
        cub_features = torch.cat(
            (cub_features, flat_topk.view(flat_topk.shape[0], 1)), dim=1
        )

        cub_features = cub_features.view(
            -1,
            (self.embed_dim + 1) * self.num_cubs,
        )

        # cub_features = self.dropout(cub_features)
        agg_classification = self.volume_classifier(cub_features)

        return agg_classification, cub_class, prop_cubs, top_prop

    def _forward_pos_enc_vec(self, x):
        # print(x.shape)
        prop_cubs = self.proposal_gen(x)
        # general_features = self.proposal_gen.get_latent_features().detach()
        prop_cubs = self.prop_sig(prop_cubs)
        _, top_prop = torch.topk(prop_cubs, self.num_cubs, -1)

        self.log_cuboid_probably_stats(prop_cubs)

        select_cubs = self.extract_batch_cuboids(x, top_prop)

        cub_class = self.cub_classifier(select_cubs)
        cub_features = self.cub_classifier.get_latent_features().detach()

        # cub_features = cub_features.view(x.shape[0], self.num_cubs, -1)
        # print(cub_features.shape)
        pos_embed = self.pos_enc[:, torch.flatten(top_prop)]
        pos_embed = pos_embed.view(x.shape[0], -1)

        cub_features = cub_features.view(
            -1,
            self.embed_dim * self.num_cubs,
        )

        cub_features += pos_embed
        # print(cub_features.shape)
        # cub_features = torch.cat((cub_features, general_features), dim=1)
        # cub_features = self.dropout(cub_features)
        agg_classification = self.volume_classifier(cub_features)
        # print(agg_classification)

        return agg_classification, cub_class, prop_cubs, top_prop

    def _forward_pos_enc_vec_binary_cub_classifier(self, x):
        prop_cubs = self.proposal_gen(x)
        prop_cubs = self.prop_sig(prop_cubs)
        _, top_prop = torch.topk(prop_cubs, self.num_cubs, -1)

        self.log_cuboid_probably_stats(prop_cubs)

        select_cubs = self.extract_batch_cuboids(x, top_prop)

        cub_class = self.cuboid_classifier(select_cubs)
        # cub_features = torch.clip(cub_class, min=0, max=1)
        cub_features = self.cuboid_classifier.get_latent_features().detach()

        # cub_features = cub_features.view(x.shape[0], self.num_cubs, -1)
        # print(cub_features.shape)
        pos_embed = self.pos_enc[:, torch.flatten(top_prop)]
        pos_embed = pos_embed.view(x.shape[0], -1)

        cub_features = cub_features.view(
            -1,
            self.embed_dim * self.num_cubs,
        )

        cub_features += pos_embed
        # print(cub_features.shape)
        # cub_features = torch.cat((cub_features, general_features), dim=1)
        # cub_features = self.dropout(cub_features)
        agg_classification = self.volume_classifier(cub_features)
        # print(agg_classification)

        return agg_classification, cub_class, prop_cubs, top_prop

    def _forward_pos_enc_vec_with_prop_feat(self, x):
        prop_cubs = self.proposal_gen(x)
        general_features = self.proposal_gen.get_latent_features().detach()
        # prop_cubs = self.prop_sig(prop_cubs)
        _, top_prop = torch.topk(prop_cubs, self.num_cubs, -1)

        self.log_cuboid_probably_stats(prop_cubs)

        select_cubs = self.extract_batch_cuboids(x, top_prop)

        cub_class = self.cuboid_classifier(select_cubs)

        # Binarize cuboid classifications
        # cub_class = torch.clip(cub_class, min=0, max=1)
        # cub_class = torch.softmax(cub_class, dim=-1)
        cub_features = self.cuboid_classifier.get_latent_features().detach()

        pos_embed = self.pos_enc[:, torch.flatten(top_prop)]
        pos_embed = pos_embed.view(x.shape[0], -1)

        cub_features = cub_features.view(
            -1,
            self.embed_dim * self.num_cubs,
        )

        cub_features += pos_embed
        cub_features = torch.cat((cub_features, general_features), dim=1)
        # cub_features = self.dropout(cub_features)
        agg_classification = self.volume_classifier(cub_features)

        return agg_classification, cub_class, prop_cubs, top_prop

    def _forward(self, x):
        return self.APPROACHES[self.approach](x)

    def train_forward(self, x):
        return self._forward(x)

    def forward(self, x):
        output = self._forward(x)
        classification = self.softmax(output[0])
        return classification, output[3]

    def log_cuboid_probably_stats(self, prop_cubs):
        stds = torch.std(prop_cubs, dim=-1)
        self.logger.log_info("Standard Deviation", stds.tolist())
        mins, _ = torch.min(prop_cubs, dim=-1)
        self.logger.log_info("Min Confidence", mins.tolist())
        maxs, _ = torch.max(prop_cubs, dim=-1)
        self.logger.log_info("Max Confidence", maxs.tolist())

    def combine_weak_gt(
        self,
        cub_classes,
        cub_labels,
        vol_classification,
        prop_gt,
        selected_or_not,
        epoch,
        prox_effect_epoch=-1,
        d_coef=0.1,
        c_coef=0.9,
    ):

        if self.temperature is None:
            current_temp = 0
        else:
            current_temp = self.temperature * (epoch + 1)

        cub_correct = cub_classes == cub_labels
        vol_correct = vol_classification == cub_labels
        both_correct = torch.logical_and(
            vol_correct,
            cub_correct,
        )
        both_wrong = torch.logical_and(
            torch.logical_not(vol_correct),
            torch.logical_not(cub_correct),
        )
        vol_only_correct = torch.logical_and(
            vol_correct,
            torch.logical_not(cub_correct),
        )
        cub_only_correct = torch.logical_and(
            cub_correct,
            torch.logical_not(vol_correct),
        )

        prop_gt = torch.softmax(prop_gt, dim=-1)

        if prox_effect_epoch > -1:
            if epoch > prox_effect_epoch:
                prop_gt = proximity_mult(prop_gt, self.proximity)

        # This will not apply to any cuboids not selected
        # as they will have the class=-1
        prop_gt[torch.logical_and(both_correct, (selected_or_not))] = 1
        prop_gt[torch.logical_and(both_wrong, selected_or_not)] = 0

        # Decrease the confidence in cuboids that were not selected
        # AND where the selected cuboids lead to the correct classification
        # overall
        # decrease_coef = (1 + 1 / ((1 / d_coef) + np.exp(1 + current_temp))) - 1
        # decrease_coef = 1/(1+np.exp(-(d_coef-1)*(1-current_temp)))
        decrease_coef = d_coef - ((1 / (1 + np.exp(-current_temp))) - 0.5)
        # decrease_coef = d_coef * (1 / (1 + current_temp * current_temp))
        # decrease_coef = d_coef * (1 - current_temp)
        prop_gt[torch.logical_and(vol_only_correct, selected_or_not)] *= decrease_coef

        # Increase the confidence in cuboids that we not selected
        # AND where the selected cuboids lead to an wrong classification
        # overall
        # increase_coef = c_coef * (1 + current_temp)
        # increase_coef = 1 + 1/(1+np.exp(-(c_coef-1)*(1-current_temp)))
        # increase_coef = 1 + 1 / ((1 / c_coef) + np.exp(1 + current_temp))
        # increase_coef = 1+(c_coef * (1 / (1 - current_temp*current_temp)))
        # increase_coef = c_coef * (1 + current_temp)
        # increase_coef = 1 + 1 / ((1 / c_coef) + np.exp(1 - current_temp))
        increase_coef = c_coef + ((1 / (1 + np.exp(-current_temp))) - 0.5)

        prop_gt[
            torch.logical_and(both_wrong, torch.logical_not(selected_or_not))
        ] *= increase_coef
        prop_gt[cub_only_correct] *= increase_coef

        prop_gt = torch.clip(prop_gt, min=0, max=1)
        # prop_gt = torch.softmax(prop_gt, dim=-1)

        return prop_gt

    def weak_sup_gt(
        self,
        prop_out,
        topk,
        cub_class,
        labels,
        vol_classification,
        epoch,
        d_coef=0.9,
        c_coef=1.1,
        binary=True,
    ):

        batch_size = prop_out.shape[0]
        total_cubs = prop_out.shape[1]

        prop_gt = prop_out.clone().detach()

        # -1 class is assigned for cuboids that are not selected.
        cub_classes = torch.full((batch_size, total_cubs), fill_value=-1)

        selected_or_not = torch.zeros(size=(batch_size, total_cubs))
        selected_or_not[
            torch.LongTensor([[x] for x in range(0, batch_size)]),
            topk,
        ] = 1

        _, cub_class = torch.max(cub_class, dim=-1)
        cub_class = cub_class.view(batch_size, -1)

        cub_classes[torch.LongTensor([[x] for x in range(0, batch_size)]), topk] = (
            cub_class.detach().cpu().clone()
        )

        labels = labels.view(labels.shape[0], 1)
        cub_labels = labels.expand(labels.shape[0], total_cubs).detach().cpu()

        vol_classification = self.softmax(vol_classification).detach().cpu()
        _, vol_classification = torch.max(vol_classification, dim=-1)
        vol_classification = vol_classification.expand(size=(1, -1))
        vol_classification = vol_classification.transpose(0, 1)
        vol_classification = vol_classification.expand(size=(batch_size, total_cubs))

        if binary:
            cub_classes[cub_classes > 0] = 1
            cub_labels[cub_labels > 0] = 1
            vol_classification[vol_classification > 0] = 1

        return self.combine_weak_gt(
            cub_classes,
            cub_labels,
            vol_classification,
            prop_gt,
            selected_or_not,
            epoch,
            self.prox_effect_epoch,
            d_coef,
            c_coef,
        )
