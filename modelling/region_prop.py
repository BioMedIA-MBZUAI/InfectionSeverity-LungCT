import numpy as np
import torch
import torch.nn as nn

from experiments.utils import logger
from modelling.densenet import densenet
from modelling.resnet import resnet
from modelling.ThreeDCNN import ThreeDCNN
from modelling.mlp import MLP

MODELS = {"DenseNet": densenet, "ResNet": resnet, "MLP": MLP}
APPROACHES = ["pos_enc_vec", "topk_cat"]


class CuboidClassifier(nn.Module):
    def __init__(self, cube_size: tuple, channels=1, num_classes=5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(channels, 32, kernel_size=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        )
        inp_sample = torch.ones(
            [
                1,
                channels,
            ]
            + cube_size
        )

        # Use the input sampler to caclulate the
        # the output size from the cnn layers
        sample_out = self.conv(inp_sample)
        conv_out_shape = sample_out.shape

        out_size = 1
        for i in range(2, len(conv_out_shape)):
            out_size = out_size * conv_out_shape[i]

        self.fc = nn.Sequential(
            nn.Linear((64 * out_size), 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.class_layer = nn.Sequential(
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        self.latent_features = out.clone()
        return self.class_layer(out)

    def get_latent_features(self):
        return self.latent_features


def cuboid_classifier(params, **kwargs):
    return CuboidClassifier(
        cube_size=[
            kwargs["sample_duration"],
            kwargs["sample_size"],
            kwargs["sample_size"],
        ],
        channels=kwargs["channels"],
        num_classes=kwargs["classes"],
    )


def mesh(image_size: tuple, mesh_size: tuple):
    # Original image coords
    img_z, img_x, img_y = image_size[0], image_size[1], image_size[2]

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


def proximity_mat(cuboids: list, cuboid_dims: tuple, device) -> torch.FloatTensor:
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
    proximity = 2*(1/(1+np.exp(20*proximity)))
    return torch.tensor(proximity, dtype=torch.float32, device=device)


def proximity_mult(probs: torch.Tensor, proximity: torch.Tensor, epoch):
    assert proximity.shape[0] == proximity.shape[1]
    num_cuboids = proximity.shape[0]
    batch_size = probs.shape[0]
    adds = torch.zeros((batch_size, num_cuboids), device=probs.device)
    for i in range(num_cuboids):
        # exp_prox = 1 / (torch.exp(proximity[i, :]))
        exp_prox = proximity[i, :]
        exp_prox = exp_prox.repeat(batch_size, 1)
        factors = exp_prox * probs
        factors = factors/torch.max(factors) # Normalize the factors
        # This is the threshold afterwich the impact of the cuboid will be considered
        factors[factors == 0] = 0.85 
        factors = -2*((1/(1+torch.exp(factors-0.85)))-0.5)
        factors[factors<0] = 0 # Apply relu
        adds[:, i] = torch.mean(factors, dim=1)
    # adds = 1 / (prox_factor + torch.exp(3 - adds))
    # adds = adds / torch.mean(adds, dim=0)
    new_probs = probs + adds
    print(torch.mean(new_probs - probs))
    return new_probs


class RegionPropPipline(nn.Module):
    def __init__(
        self,
        img_size: tuple,
        cub_size: tuple,
        num_cubs: int,
        total_cubodis: int,
        prop_model: nn.Module,
        cub_classifier: nn.Module,
        embed_dim: int,
        aggreg_model: nn.Module,
        logger: logger.Logger,
        device: torch.device,
        approach="pos_enc_vec",
        temperature=None,
        prox_effect_epoch=-1,
    ):
        super().__init__()
        self.num_cubs = num_cubs
        self.cub_size = cub_size
        self.embed_dim = embed_dim
        self.approach = approach
        self.logger = logger
        self.device = device
        self.total_cubs = total_cubodis
        self.prox_effect_epoch = prox_effect_epoch

        self.cuboids = mesh(img_size, cub_size)
        self.proximity = proximity_mat(
            cuboids=self.cuboids, cuboid_dims=self.cub_size, device=device
        )
        self.cub_prop = prop_model
        self.prop_sig = torch.nn.Sigmoid().to(device)

        self.cub_classifier = cub_classifier
        self.pos_enc = nn.Parameter(
            torch.randn(
                1,
                total_cubodis,
                embed_dim,
            ),
            requires_grad=True,
        ).to(device)
        self.softmax = nn.Softmax(dim=-1)
        self.final_agg = aggreg_model

        self.temprature = temperature

    def extract_batch_cuboids(self, img_batch, cuboids: list):

        device = img_batch.device

        batch_size = img_batch.shape[0]
        num_cuboids = cuboids.shape[1]

        new_batch = torch.zeros(
            batch_size * num_cuboids,
            1,
            self.cub_size[0],
            self.cub_size[1],
            self.cub_size[2],
        ).to(device)

        z = 0
        for i in range(0, batch_size):
            for c in range(0, num_cuboids):
                new_batch[z] = self.extract_cuboid(
                    img_batch[i], self.cuboids[cuboids[i][c]]
                )
                z += 1

        return new_batch

    def extract_cuboid(self, img_tensor, bl_coords):
        return img_tensor[
            0,
            bl_coords[0] : bl_coords[0] + self.cub_size[0],
            bl_coords[1] : bl_coords[1] + self.cub_size[1],
            bl_coords[2] : bl_coords[2] + self.cub_size[2],
        ]

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

        if self.temprature is None:
            current_temp = 0
        else:
            current_temp = self.temprature * epoch

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

        if prox_effect_epoch > -1:
            if epoch > prox_effect_epoch:
                prop_gt = proximity_mult(prop_gt, self.proximity, epoch)

        # This will not apply to any cuboids not selected
        # as they will have the class=-1
        prop_gt[torch.logical_and(both_correct, (selected_or_not))] = 1
        prop_gt[torch.logical_and(both_wrong, selected_or_not)] = 0

        # Decrease the confidence in cuboids that were not selected
        # AND where the selected cuboids lead to the correct classification
        # overall
        # decrease_coef = (1 + 1 / ((1 / d_coef) + np.exp(1 + current_temp))) - 1
        decrease_coef = d_coef * (1 / (1 + current_temp * current_temp))
        # decrease_coef = (1 + 1 / ((1 / d_coef) + np.exp(1 + current_temp))) - 1
        prop_gt[torch.logical_and(vol_only_correct, selected_or_not)] *= decrease_coef

        # Increase the confidence in cuboids that we not selected
        # AND where the selected cuboids lead to an wrong classification
        # overall
        # increase_coef = c_coef * (1 + current_temp)
        # increase_coef = 1 + 1/(1+np.exp(-(c_coef-1)*(1-current_temp)))
        increase_coef = 1 + 1 / ((1 / c_coef) + np.exp(1 + current_temp))
        # increase_coef = 1+(c_coef * (1 / (1 - current_temp*current_temp)))
        # increase_coef = 1 + 1 / ((1 / c_coef) + np.exp(1 - current_temp))
        # increase_coef = 1+(c_coef * np.power(current_temp,3))
        prop_gt[
            torch.logical_and(both_wrong, torch.logical_not(selected_or_not))
        ] *= increase_coef
        prop_gt[cub_only_correct] *= increase_coef

        prop_gt = torch.clip(prop_gt, min=0, max=1)

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
            prox_effect_epoch=self.prox_effect_epoch,
            d_coef=d_coef,
            c_coef=c_coef,
        )

    def _forward_concat_topk_and_conf(self, x):
        prop_cubs = self.cub_prop(x)
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
        agg_classification = self.final_agg(cub_features)

        return agg_classification, cub_class, prop_cubs, top_prop

    def _forward_dynamic(self, x):
        prop_cubs = self.cub_prop(x)

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
        agg_classification = self.final_agg(cub_features)

        return agg_classification, cub_class, prop_cubs, top_prop

    def _forward_concat_topk(self, x):
        prop_cubs = self.cub_prop(x)
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
        agg_classification = self.final_agg(cub_features)

        return agg_classification, cub_class, prop_cubs, top_prop

    def _forward_pos_enc_vec(self, x):
        # print(x.shape)
        prop_cubs = self.cub_prop(x)
        # general_features = self.cub_prop.get_latent_features().detach()
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
        agg_classification = self.final_agg(cub_features)
        # print(agg_classification)

        return agg_classification, cub_class, prop_cubs, top_prop

    def _forward_pos_enc_vec_binary_cub_classifier(self, x):
        prop_cubs = self.cub_prop(x)
        prop_cubs = self.prop_sig(prop_cubs)
        _, top_prop = torch.topk(prop_cubs, self.num_cubs, -1)

        self.log_cuboid_probably_stats(prop_cubs)

        select_cubs = self.extract_batch_cuboids(x, top_prop)

        cub_class = self.cub_classifier(select_cubs)
        cub_class = torch.clip(cub_class, min=0, max=1)
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
        agg_classification = self.final_agg(cub_features)
        # print(agg_classification)

        return agg_classification, cub_class, prop_cubs, top_prop

    def _forward_pos_enc_vec_with_prop_feat(self, x):
        prop_cubs = self.cub_prop(x)
        general_features = self.cub_prop.get_latent_features().detach()
        prop_cubs = self.prop_sig(prop_cubs)
        _, top_prop = torch.topk(prop_cubs, self.num_cubs, -1)

        self.log_cuboid_probably_stats(prop_cubs)

        select_cubs = self.extract_batch_cuboids(x, top_prop)

        cub_class = self.cub_classifier(select_cubs)

        # Binarize cuboid classifications
        cub_class = torch.softmax(cub_class, dim=-1)
        # cub_class = torch.clip(cub_class, min=0, max=1)
        cub_features = self.cub_classifier.get_latent_features().detach()

        pos_embed = self.pos_enc[:, torch.flatten(top_prop)]
        pos_embed = pos_embed.view(x.shape[0], -1)

        cub_features = cub_features.view(
            -1,
            self.embed_dim * self.num_cubs,
        )

        cub_features += pos_embed
        cub_features = torch.cat((cub_features, general_features), dim=1)
        # cub_features = self.dropout(cub_features)
        agg_classification = self.final_agg(cub_features)

        return agg_classification, cub_class, prop_cubs, top_prop

    def log_cuboid_probably_stats(self, prop_cubs):
        stds = torch.std(prop_cubs, dim=-1)
        self.logger.log_info("Standard Deviation", stds.tolist())
        mins, _ = torch.min(prop_cubs, dim=-1)
        self.logger.log_info("Min Confidence", mins.tolist())
        maxs, _ = torch.max(prop_cubs, dim=-1)
        self.logger.log_info("Max Confidence", maxs.tolist())

    def _forward(self, x):
        if self.approach == "pos_enc_vec":
            return self._forward_pos_enc_vec(x)
        elif self.approach == "topk_cat":
            return self._forward_concat_topk(x)
        elif self.approach == "topk_and_conf":
            return self._forward_concat_topk_and_conf(x)
        elif self.approach == "forward_pos_enc_vec_binary_cub_classifier":
            return self._forward_pos_enc_vec_binary_cub_classifier(x)
        elif self.approach == "forward_pos_enc_vec_with_prop_feat":
            return self._forward_pos_enc_vec_with_prop_feat(x)

    def train_forward(self, x):
        return self._forward(x)

    def forward(self, x):
        output = self._forward(x)
        classification = self.softmax(output[0])
        return classification, output[3]


# class RegionPropTrainer:
#     def __init__(
#         self,
#         pipeline: RegionPropPipline,
#         classifier_criterion,
#         classifier_optim,
#         prop_criterion,
#         prop_optim,
#         agg_criterion,
#         agg_optim,
#     ) -> None:
#         self.pipeline = pipeline
#         self.classifier_criterion = classifier_criterion
#         self.classifier_optim = classifier_optim
#         self.prop_criterion = prop_criterion
#         self.prop_optim = prop_optim
#         self.agg_criterion = agg_criterion
#         self.agg_optim = agg_optim

#     def calc_prop_gt(self, top_prop, class_out, labels, num_classes):
#         """
#         Calculate the proposal's ground truth based on the
#         classiciation outcome:
#         - The confidence is lowered in cuboids that do not lead
#         to a correct classification.
#         - The confidence is maximized for cuboids that produce
#         correct classifications.
#         - The confidence is narrowly increased in all other cuboids
#         when the majority of the proposed cuboids produce wrong
#         classification.

#         Args:
#             porp_output ([type]): [description]
#             top_prop ([type]): [description]
#             class_out ([type]): The classification result
#             from the classifier.
#             labels ([type]): [description]

#         Returns:
#             [torch.Tensor]: The proposals ground truth based on the classification
#             of each of the proposals put forward.
#         """
#         batch_size = top_prop.shape[0]

#         wrong_cub_conf = self.cuboids_gt_factors["wrong"]
#         correct_cub_conf = self.cuboids_gt_factors["correct"]
#         increase_conf_factor = self.cuboids_gt_factors["increase_conf"]
#         decrease_conf_factor = self.cuboids_gt_factors["decrease_conf"]

#         prop_gt = torch.ones(batch_size, len(self.cubiods)).to(self.device)

#         concat_output = self.concat_output(class_out, batch_size, num_classes)
#         _, cuboids_classes = class_out.max(dim=1)

#         for sample_i in range(0, batch_size):
#             _, pred_ind = torch.max(concat_output[sample_i], 0)
#             prop_idx = sample_i * self.num_cuboids
#             # Correctly classified sample
#             if pred_ind == labels[sample_i]:
#                 # Minimize the confidence in the rest of the
#                 # cuboids
#                 prop_gt[sample_i] = prop_gt[sample_i] * decrease_conf_factor
#                 for prop in top_prop[sample_i]:
#                     # Maximize the confidence in ONLY in the cuboids
#                     # that got the right classification result
#                     if cuboids_classes[prop_idx] == labels[sample_i]:
#                         prop_gt[sample_i][prop] = correct_cub_conf
#                     else:
#                         prop_gt[sample_i][prop] = wrong_cub_conf
#                     prop_idx += 1
#             # Wrongly classified sample
#             else:
#                 # Narrowly increase confidence in the rest of the
#                 # cuboids
#                 prop_gt[sample_i] = prop_gt[sample_i] * increase_conf_factor
#                 for prop in top_prop[sample_i]:
#                     # Maximize the confidence in the cuboids
#                     # that got the right classification result
#                     if cuboids_classes[prop_idx] == labels[sample_i]:
#                         prop_gt[sample_i][prop] = correct_cub_conf
#                     else:
#                         prop_gt[sample_i][prop] = wrong_cub_conf
#                     prop_idx += 1

#         return prop_gt

#     def backward(
#         self,
#     ):
#         agg_loss = self.agg_criterion()

#     def forward(self, input: torch.Tensor):
#         return self.pipeline(input)

#     def __call__(self, input: torch.Tensor, *args: Any, **kwds: Any) -> Any:
#         self.forward(input)


# def region_porp_pipeline():
#     region_prop = MODELS["DenseNet"]()
#     return RegionPropPipline
