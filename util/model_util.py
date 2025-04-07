import copy

import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score
from torch.nn.functional import cross_entropy, interpolate, l1_loss
from torchvision.models import densenet161, resnet50, vgg16
from tqdm import tqdm


class Enhancer(nn.Module):
    def __init__(self, num_layer=5, num_iter=5, checkpoint_path=None):
        super(Enhancer, self).__init__()
        self.num_layer = num_layer
        self.num_iter = num_iter

        # build conv layers dynamically
        in_chnl = 2
        base_chnl = 16
        layers = []
        for i in range(num_layer):
            # out_chnl = (i + 1) * base_chnl
            out_chnl = int(base_chnl * 2**i)

            layers.append(nn.Conv2d(in_chnl, out_chnl, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))

            if i < num_layer - 1:
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))

            in_chnl = copy.copy(out_chnl)

        self.feat = nn.Sequential(*layers)

        # coefficient prediction network
        self.pred = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_chnl, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, self.num_iter),
            nn.Tanh(),
        )

        # load pre-trained weights
        if checkpoint_path is not None:
            print("Use pretrained weights from %s." % checkpoint_path)
            self.load_pretrained_weights(checkpoint_path)
        else:
            print("Use random weight initialization.")

    def forward(self, x):
        # resize the input image to 256x256
        x_small = t.cat(
            [
                interpolate(x, size=(256, 256), mode="bilinear", align_corners=False),
                interpolate(1 - x, size=(256, 256), mode="bilinear", align_corners=False),
            ],
            dim=1,
        )

        # run model
        pred_feat = self.feat(x_small)
        pred_alpha = self.pred(pred_feat)

        # apply enhancement to the original image
        out_img = x.clone()
        for i in range(self.num_iter):
            out_img = out_img + pred_alpha[:, i].view(x.shape[0], 1, 1, 1) * out_img * (1 - out_img)

        return out_img

    def load_pretrained_weights(self, checkpoint_path):
        pretrained_dict = t.load(checkpoint_path, weights_only=True)
        model_dict = self.state_dict()

        for k in pretrained_dict.keys():
            if k in model_dict:
                if pretrained_dict[k].size() == model_dict[k].size():
                    model_dict[k] = pretrained_dict[k]
                else:
                    print(f"Skip {k}, required {model_dict[k].shape}, loaded {pretrained_dict[k].shape}.")

        self.load_state_dict(model_dict)


class Classifier(nn.Module):
    def __init__(self, backbone="resnet", num_class=2, dropout=0.5, checkpoint_path=None, freeze_encoder=False):
        super(Classifier, self).__init__()

        # model backbone
        if backbone == "resnet":
            self.model = resnet50(weights=None)
            num_features = self.model.fc.in_features
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(num_features, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(1024, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(512, num_class),
            )
            self.model.fc = self.classifier

        elif backbone == "densenet":
            self.model = densenet161(weights=None)
            num_features = self.model.classifier.in_features
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(num_features, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(1024, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(512, num_class),
            )
            self.model.classifier = self.classifier

        else:
            raise NotImplementedError("Backbone type: ", backbone, " not implemented")

        # load pre-trained weights
        if checkpoint_path is not None:
            print("Use pretrained weights from %s." % checkpoint_path)
            self.load_pretrained_weights(checkpoint_path)
        else:
            print("Use random weight initialization.")

        if freeze_encoder:
            print("Freeze encoder layers...")
            for param in self.model.parameters():
                param.requires_grad = False

            # unfreeze classifier layers
            for param in self.classifier.parameters():
                param.requires_grad = True

    def forward(self, x):
        rgb_img = x.repeat(1, 3, 1, 1)
        out = self.model(rgb_img)

        return out

    def load_pretrained_weights(self, checkpoint_path):
        pretrained_dict = t.load(checkpoint_path, weights_only=True)
        model_dict = self.state_dict()

        for k in pretrained_dict.keys():
            if k in model_dict:
                if pretrained_dict[k].size() == model_dict[k].size():
                    model_dict[k] = pretrained_dict[k]
                else:
                    print(f"Skip {k}, required {model_dict[k].shape}, loaded {pretrained_dict[k].shape}.")

        self.load_state_dict(model_dict)


class VGG16(nn.Module):
    def __init__(self, num_class=165, dropout=0.5, checkpoint_path=None):
        super(VGG16, self).__init__()
        # load VGG16 backbone
        self.vgg = vgg16(weights=None)

        # modify the classifier
        num_features = self.vgg.classifier[0].in_features
        self.vgg.classifier = nn.Sequential(
            nn.Linear(num_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_class),
        )

        # initialize weights
        if checkpoint_path is None:
            self._initialize_weights()
        else:
            self.load_pretrained_weights(checkpoint_path)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def load_pretrained_weights(self, checkpoint_path):
        pretrained_dict = t.load(checkpoint_path, weights_only=True)
        model_dict = self.state_dict()

        for k in pretrained_dict.keys():
            if k in model_dict:
                if pretrained_dict[k].size() == model_dict[k].size():
                    model_dict[k] = pretrained_dict[k]
                else:
                    print(f"Skip {k}, required {model_dict[k].shape}, loaded {pretrained_dict[k].shape}.")

        self.load_state_dict(model_dict)

    def forward(self, x):
        pred = self.vgg(x)

        return pred


class VGGLoss(nn.Module):
    def __init__(self, checkpoint_path, device="cpu", target_layers="conv2_2"):
        super(VGGLoss, self).__init__()
        vgg = VGG16(checkpoint_path=checkpoint_path).vgg.features.to(device).eval()

        # map layer names to indices (before ReLU)
        self.layer_indices = {
            "conv1_1": 0,
            "conv1_2": 2,
            "conv2_1": 5,
            "conv2_2": 7,
            "conv3_1": 10,
            "conv3_2": 12,
            "conv3_3": 14,
            "conv4_1": 17,
            "conv4_2": 19,
            "conv4_3": 21,
            "conv5_1": 24,
            "conv5_2": 26,
            "conv5_3": 28,
        }

        # ensure target_layers is always a tuple
        if isinstance(target_layers, str):
            target_layers = (target_layers,)

        # validate and sort the target layers
        self.target_layers = sorted(target_layers, key=lambda x: self.layer_indices[x])
        if not all(layer in self.layer_indices for layer in self.target_layers):
            raise ValueError(f"Invalid layer(s) in {target_layers}. Valid layers: {list(self.layer_indices.keys())}.")

        # extract the required parts of the VGG model
        self.vgg = vgg[: self.layer_indices[self.target_layers[-1]] + 1]
        self.target_indices = [self.layer_indices[layer] for layer in self.target_layers]

        # reduce features to 1x1
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # freeze VGG weights
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        if pred.size(1) == 1:
            pred = pred.repeat(1, 3, 1, 1)
        if target.size(1) == 1:
            target = target.repeat(1, 3, 1, 1)

        loss = 0
        for i, layer in enumerate(self.vgg):
            pred = layer(pred)
            target = layer(target)

            if i in self.target_indices:
                # apply adaptive pooling to reduce features to 1x1
                pred_pool = self.adaptive_pool(pred)
                target_pool = self.adaptive_pool(target)

                # compute loss for the current layer
                loss += l1_loss(pred_pool, target_pool)

        return loss


def trainClassifier(dataloader, classifier, optim, device="cpu", progbar=True):
    iterator = tqdm(dataloader, total=len(dataloader), desc="Train") if progbar else dataloader

    classifier.train()

    total_loss = 0
    for img_full, _, _, patho in iterator:
        img_full, patho = img_full.to(device), patho.to(device)

        # zero gradients
        optim.zero_grad()

        # forward pass
        pred_logit = classifier(img_full)

        # compute classification loss over the batch
        batch_loss = cross_entropy(pred_logit, patho.long())

        # back-propagate and update weights
        batch_loss.backward()
        optim.step()

        # accumulate loss
        total_loss += batch_loss.item()

    # average loss over the dataset
    total_loss /= len(dataloader)

    return total_loss


def valClassifier(dataloader, classifier, device="cpu", progbar=True):
    iterator = tqdm(dataloader, total=len(dataloader), desc="Val") if progbar else dataloader

    classifier.eval()

    gt_list = []
    label_list = []
    total_loss = 0.0
    with t.no_grad():
        for img_full, _, _, patho in iterator:
            img_full = img_full.to(device)
            patho = patho.to(device)

            # forward pass
            pred_logit = classifier(img_full)

            # convert logits to probabilities
            pred_prob = F.softmax(pred_logit, dim=1)

            # predicted labels
            pred_label = t.argmax(pred_prob, dim=1)

            # compute classification loss over the batch
            batch_loss = cross_entropy(pred_logit, patho.long())
            total_loss += batch_loss.item()

            # collect results
            gt_list.extend(patho.cpu().numpy())
            label_list.extend(pred_label.cpu().numpy())

    # average loss over the dataset
    avg_loss = total_loss / len(dataloader)

    # convert to numpy arrays
    gt_array = np.array(gt_list)
    label_array = np.array(label_list)

    # compute worst group performance
    val_worst = 1.0
    val_confmat = confusion_matrix(gt_array, label_array)
    for i in range(val_confmat.shape[0]):
        row_sum = val_confmat[i, :].sum()
        perc = val_confmat[i, i] / row_sum
        if perc < val_worst:
            val_worst = perc

    return avg_loss, val_worst


def testClassifier(dataloader, classifier, task="binary", device="cpu", progbar=True):
    assert task in ["binary", "multi-class"], "task must be 'binary' or 'multi-class'."
    iterator = tqdm(dataloader, total=len(dataloader), desc="Test") if progbar else dataloader

    classifier.eval()

    gt_list = []
    prob_list = []
    label_list = []
    with t.no_grad():
        for img_full, _, _, patho in iterator:
            img_full = img_full.to(device)
            patho = patho.to(device)

            # forward pass
            pred_logit = classifier(img_full)

            # convert logits to probabilities
            pred_prob = F.softmax(pred_logit, dim=1)

            # predicted labels
            pred_label = t.argmax(pred_prob, dim=1)

            # collect results
            gt_list.extend(patho.cpu().numpy())
            prob_list.extend(pred_prob.cpu().numpy())
            label_list.extend(pred_label.cpu().numpy())

    # convert to numpy arrays
    gt_array = np.array(gt_list)
    prob_array = np.array(prob_list)
    label_array = np.array(label_list)

    if task == "multi-class":
        # identify unique classes in ground-truth
        unique_class = np.unique(gt_array)

        if len(unique_class) < 2:
            test_auc = float("nan")
        elif len(unique_class) == 2:
            # standard binary AUC
            test_auc = roc_auc_score(gt_array, prob_array[:, 1])
        else:
            # multi-class AUC
            test_auc = roc_auc_score(gt_array, prob_array, multi_class="ovr")

        # compute confusion matrix
        test_confmat = confusion_matrix(gt_array, label_array)

        return test_auc, test_confmat

    else:
        # compute precision
        test_prec = precision_score(gt_array, label_array)

        # compute recall
        test_recall = recall_score(gt_array, label_array)

        return test_prec, test_recall


def trainEnhancerVGG(
    dataloader_list, enhancer, classifier, vgg_loss, optim, loss_lambda=0.5, device="cpu", progbar=True
):
    total_batches = min(len(dataloader_list[0]), len(dataloader_list[1]))
    iterator = (
        tqdm(zip(dataloader_list[0], dataloader_list[1]), total=total_batches, desc="Train")
        if progbar
        else zip(dataloader_list[0], dataloader_list[1])
    )

    enhancer.train()
    classifier.eval()

    total_loss_vgg = 0
    total_loss_ce = 0
    for (img_full_src, _, _, patho_src), (img_full_tgt, _, _, patho_tgt) in iterator:
        img_full_src = img_full_src.to(device)
        patho_src = patho_src.to(device)
        img_full_tgt = img_full_tgt.to(device)
        patho_tgt = patho_tgt.to(device)

        # skip if the sample sizes don't match
        if img_full_src.shape[0] != img_full_tgt.shape[0]:
            continue

        # zero the gradients
        optim.zero_grad()

        # forward pass
        pred_data_src = enhancer(img_full_src)
        pred_data_tgt = enhancer(img_full_tgt)
        pred_logit_src = classifier(pred_data_src)
        pred_logit_tgt = classifier(pred_data_tgt)

        # compute loss
        loss_vgg = vgg_loss(pred_data_src, img_full_tgt) + vgg_loss(pred_data_tgt, img_full_tgt)
        loss_ce = cross_entropy(pred_logit_src, patho_src.long()) + cross_entropy(pred_logit_tgt, patho_tgt.long())

        ratio = (loss_ce.item() * 1000) // (loss_vgg.item() * 1000)
        loss = loss_lambda * ratio * loss_vgg + (1 - loss_lambda) * loss_ce

        # back-propagate and update weights
        loss.backward()
        optim.step()

        # accumulate the loss
        total_loss_vgg += loss_vgg.item()
        total_loss_ce += loss_ce.item()

    # compute average loss for the epoch
    total_loss_vgg /= total_batches
    total_loss_ce /= total_batches

    return total_loss_vgg, total_loss_ce


def valEnhancer(dataloader, enhancer, classifier, device="cpu", progbar=True):
    iterator = tqdm(dataloader, total=len(dataloader), desc="Val") if progbar else dataloader

    enhancer.eval()
    classifier.eval()

    prob_list = []
    label_list = []
    gt_list = []
    machine_list = []
    total_loss = 0.0
    total_sample = 0

    with t.no_grad():
        for img_full, _, machine, patho in iterator:
            img_full = img_full.to(device)
            machine = machine.to(device)
            patho = patho.to(device)

            # forward pass
            pred_data = enhancer(img_full)
            pred_logit = classifier(pred_data)
            pred_prob = F.softmax(pred_logit, dim=1)
            pred_label = t.argmax(pred_prob, dim=1)

            # compute classification loss
            batch_loss = cross_entropy(pred_logit, patho.long(), reduction="sum")
            total_loss += batch_loss.item()
            total_sample += patho.size(0)

            # save results to lists
            gt_list.extend(patho.cpu().numpy())
            prob_list.extend(pred_prob.cpu().numpy())
            label_list.extend(pred_label.cpu().numpy())
            machine_list.extend(machine.cpu().numpy())

    avg_loss = total_loss / total_sample if total_sample > 0 else 0.0

    # convert lists to numpy arrays
    gt_array = np.array(gt_list)
    prob_array = np.array(prob_list)
    label_array = np.array(label_list)
    machine_array = np.array(machine_list)

    # iterate over each machine group
    worst_perf = 1.0
    n_classes = prob_array.shape[-1]
    unique_machines = np.unique(machine_array)
    for m in unique_machines:
        idx = np.where(machine_array == m)[0]
        if len(idx) == 0:
            continue
        gt_m = gt_array[idx]
        pred_m = label_array[idx]
        confmat_m = confusion_matrix(gt_m, pred_m, labels=list(range(n_classes)))

        # compute performance percentage
        for i in range(n_classes):
            row_sum = confmat_m[i, :].sum()
            if row_sum == 0:
                continue
            perc = confmat_m[i, i] / row_sum  #
            if perc < worst_perf:
                worst_perf = perc

    return avg_loss, worst_perf


def testEnhancer(dataloader, enhancer, classifier, task="binary", device="cpu", progbar=True):
    assert task in ["binary", "multi-class"], "task must be 'binary' or 'multi-class'."
    iterator = tqdm(dataloader, total=len(dataloader), desc="Test") if progbar else dataloader

    enhancer.eval()
    classifier.eval()

    prob_list = []
    label_list = []
    gt_list = []
    with t.no_grad():
        for img_full, _, _, patho in iterator:
            img_full = img_full.to(device)
            patho = patho.to(device)

            # forward pass
            pred_data = enhancer(img_full)
            pred_logit = classifier(pred_data)
            pred_prob = F.softmax(pred_logit, dim=1)
            pred_label = t.argmax(pred_prob, dim=1)

            # save to lists
            gt_list.extend(patho.cpu().numpy())
            prob_list.extend(pred_prob.cpu().numpy())
            label_list.extend(pred_label.cpu().numpy())

    # convert to numpy arrays
    gt_array = np.array(gt_list)
    prob_array = np.array(prob_list)
    label_array = np.array(label_list)

    if task == "multi-class":
        # compute AUC
        unique_class = np.unique(gt_array)
        if len(unique_class) < 2:
            test_auc = float("nan")
        elif len(unique_class) == 2:
            test_auc = roc_auc_score(gt_array, prob_array[:, 1])
        else:
            test_auc = roc_auc_score(gt_array, prob_array, multi_class="ovr")

        # compute confusion matrix
        test_confmat = confusion_matrix(gt_array, label_array, labels=[i for i in range(prob_array.shape[-1])])
        row_sums = test_confmat.sum(axis=1, keepdims=True)
        safe_row_sums = np.where(row_sums == 0, 1, row_sums)
        test_confmat_pcnt = test_confmat / safe_row_sums

        return test_auc, test_confmat_pcnt

    else:
        precision = precision_score(gt_array, label_array, zero_division=0)
        recall = recall_score(gt_array, label_array, zero_division=0)

        return precision, recall


def trainVGG(dataloader, model, optim, device="cpu", progbar=True):
    if progbar:
        iterator = tqdm(dataloader, total=len(dataloader), desc="Train")
    else:
        iterator = dataloader

    model.train()

    # iterate over the dataset
    train_loss = 0.0
    for i_idx, (data, label) in enumerate(iterator):
        data, label = data.to(device), label.to(device)

        # back-propagate
        optim.zero_grad()
        pred = model(data)
        loss = cross_entropy(pred, label)
        loss.backward()
        optim.step()

        # accumulate loss
        train_loss += loss.item()

    # average loss for this epoch
    train_loss /= len(dataloader)

    return train_loss


def valVGG(dataloader, model, top_n=1, device="cpu", progbar=True):
    if progbar:
        iterator = tqdm(dataloader, total=len(dataloader), desc="Validate")
    else:
        iterator = dataloader

    model.eval()

    # iterate over the dataset
    val_loss = 0.0
    topn_correct = 0
    total_sample = 0
    with t.no_grad():
        for data, label in iterator:
            data, label = data.to(device), label.to(device)

            # inference
            pred = model(data)
            loss = cross_entropy(pred, label)
            val_loss += loss.item()

            # get top-n predictions
            _, pred_topn = pred.topk(top_n, 1, largest=True, sorted=True)
            correct = pred_topn.eq(label.view(-1, 1).expand_as(pred_topn))
            topn_correct += correct.sum().item()
            total_sample += data.size(0)

    # average loss for the validation set
    val_loss /= len(dataloader)

    # compute top-n accuracy
    topn_acc = topn_correct / total_sample

    return val_loss, topn_acc
