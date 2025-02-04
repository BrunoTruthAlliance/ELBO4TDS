from torch import nn
import torch
import torch.nn.functional as F
from itertools import product
from torch.nn import BCEWithLogitsLoss
# from torch.nn import CrossEntropyLoss
# from util.utils import binary_label_smoothing_with_p, list_to_slate, binary_label_smoothing
# from sklearn.metrics import log_loss
DEFAULT_EPS = 1e-10
PADDED_Y_VALUE = -1
PADDED_INDEX_VALUE = -1


def get_loss(loss, true, pred, group=None, user_embedding=None, item_embedding=None):

    if isinstance(loss, str):
        if loss.lower() == "bce_with_logit_loss":
            print('bce_with_logit_loss')
            return nn.BCEWithLogitsLoss()(pred, true)
        elif loss.lower() == "celoss":
            return nn.CrossEntropyLoss(reduction='none')(pred, true)
    else:
        return


# def sigmoid_m(x, m):
#     return m / (m + torch.exp(-x))


# def pointwise(true, pred, S1, m):
#     pred = sigmoid_m(S1 * pred, m)
#     return nn.BCELoss()(pred.squeeze(), true.squeeze())


# def listwise(true, pred, total_len, S2, padded_value_indicator=PADDED_Y_VALUE):
#     mask = true == padded_value_indicator
#     pred = pred * S2
#     pred[mask] = float('-inf')
#     true[mask] = float(0)
#
#     pred = F.softmax(pred, dim=1) + DEFAULT_EPS
#     pred = torch.log(pred)
#
#     return (-torch.sum(true * pred))/total_len


# def our_loss(true, pred, group, S1, S2, m, weight):
#     true_l, pred_l = list_to_slate(true, pred, group)
#     total_len = sum(group)
#
#     loss_1 = pointwise(true, pred, S1, m)
#
#     loss_2 = listwise(true_l, pred_l, total_len, S2)
#
#     return weight*loss_1 + (1-weight)*loss_2


def contrast_loss(y, user_embedding, item_embedding, groups):
    tau = 0.0001
    norm = torch.norm(user_embedding)*torch.norm(item_embedding)*tau
    sim = torch.exp(torch.sum(user_embedding * item_embedding, 1, True) / norm)
    result = 0
    sum_group = 0
    for group in groups:
        g_sim = sim[sum_group:sum_group+group]
        Q = g_sim[y[sum_group:sum_group+group] == 1]
        N = torch.sum(g_sim[y[sum_group:sum_group+group] == 0])
        if Q.shape[0] > 0 and N != 0:
            result += -torch.mean(torch.log(Q/N))
        sum_group += group

    return result/len(groups)


def setrank(true, pred, user_embedding, item_embedding):
    pos_indices = true.squeeze() == 1
    neg_indices = true.squeeze() == 0

    pos_pred = pred[pos_indices]
    neg_pred = pred[neg_indices]

    if pos_pred.size(0) == 0 or neg_pred.size(0) == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    neg_sum = torch.sum(neg_pred)
    loss = torch.sum(-torch.log(pos_pred / (pos_pred + neg_sum)))

    return loss


def batch_setrank(true, pred, groups, user_embedding, item_embedding):
    sum_group = 0
    total_loss = 0
    for group in groups:
        true_group = true[sum_group:sum_group + group]
        pred_group = pred[sum_group:sum_group + group]
        sum_group += group
        total_loss += setrank(true_group, pred_group,
                              user_embedding, item_embedding)

    return total_loss / len(groups)


def jrc(true, pred, group):
    group_len = len(group)
    alpha = 0.5
    label = true.squeeze().long()
    logits = pred.squeeze()
    batch = true.shape[0]
    mask = torch.repeat_interleave(torch.arange(
        group_len), torch.tensor(group)).to(true.device)
    mask = mask.unsqueeze(-1).expand(batch, group_len)
    mask_m = torch.arange(group_len).repeat(batch, 1).to(true.device)
    mask = (mask == mask_m).int()
    ce_loss = F.cross_entropy(logits, label)

    logits = logits.unsqueeze(1).expand(batch, group_len, 2)
    y = label.unsqueeze(1).expand(batch, group_len)
    y_neg, y_pos = 1 - y, y
    y_neg = y_neg * mask
    y_pos = y_pos * mask
    logits = logits + (1 - mask.unsqueeze(2)) * -1e9

    l_neg, l_pos = logits[:, :, 0], logits[:, :, 1]

    loss_pos = -torch.sum(y_pos * F.log_softmax(l_pos, dim=0), dim=0)
    loss_neg = -torch.sum(y_neg * F.log_softmax(l_neg, dim=0), dim=0)
    ge_loss = torch.mean((loss_pos + loss_neg) / torch.sum(mask, dim=0))

    loss = alpha * ce_loss + (1 - alpha) * ge_loss
    return loss


def set2setrank(true, pred, w):
    pos_indices = true.squeeze() == 1
    neg_indices = true.squeeze() == 0

    pos_pred = pred[pos_indices]
    neg_pred = pred[neg_indices]

    if pos_pred.size(0) == 0 or neg_pred.size(0) == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    pos_pred_exp = pos_pred.unsqueeze(0)  # (1, num_pos)
    neg_pred_exp = neg_pred.unsqueeze(1)  # (num_neg, 1)
    log_sigmoid_diff = -torch.log(torch.sigmoid(pos_pred_exp - neg_pred_exp))
    loss_2 = torch.mean(log_sigmoid_diff, dim=1)
    fneg = torch.min(loss_2)
    loss_2 = torch.sum(loss_2)

    pos_pred_exp = pos_pred.unsqueeze(0)  # (1, num_pos)
    pos_pred_diff = torch.abs(pos_pred.unsqueeze(
        1) - pos_pred_exp)  # (num_pos, num_pos)
    log_sigmoid_diff = torch.where(pos_pred_diff > 0.5, torch.full_like(
        pos_pred_diff, 0.5, device=pos_pred_diff.device), pos_pred_diff)
    # log_sigmoid_diff = -torch.log(torch.sigmoid(pos_pred_diff))
    # fpos = torch.sum(log_sigmoid_diff, dim=1)
    fpos = torch.mean(log_sigmoid_diff)

    loss_3 = -torch.log(torch.sigmoid(fpos - fneg))
    return loss_2 + w * loss_3


def batch_set2setrank(true, pred, groups, w=1):
    sum_group = 0
    loss = 0
    for group in groups:
        true_group = true[sum_group:sum_group + group]
        pred_group = pred[sum_group:sum_group + group]
        sum_group += group
        loss += set2setrank(true_group, pred_group, w)

    return loss / len(groups)


def listNet(y_pred, y_true, eps=DEFAULT_EPS, padded_value_indicator=PADDED_Y_VALUE):
    """
    ListNet loss introduced in "Learning to Rank: From Pairwise Approach to Listwise Approach".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    mask = y_true == padded_value_indicator
    y_pred[mask] = float('-inf')
    y_true[mask] = float('-inf')

    preds_smax = F.softmax(y_pred, dim=1)
    true_smax = F.softmax(y_true, dim=1)

    preds_smax = preds_smax + eps
    preds_log = torch.log(preds_smax)

    return torch.mean(-torch.sum(true_smax * preds_log, dim=1))


def rankNet_weightByGTDiff(y_pred, y_true, padded_value_indicator=PADDED_Y_VALUE):
    """
    Wrapper for RankNet employing weighing by the differences of ground truth values.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    return rankNet(y_pred, y_true, padded_value_indicator, weight_by_diff=True)


def rankNet_weightByGTDiff_pow(y_pred, y_true, padded_value_indicator=PADDED_Y_VALUE):
    """
    Wrapper for RankNet employing weighing by the squared differences of ground truth values.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    return rankNet(y_pred, y_true, padded_value_indicator, weight_by_diff=False, weight_by_diff_powed=True)


def rankNet(y_pred, y_true, padded_value_indicator=PADDED_Y_VALUE, weight_by_diff=False, weight_by_diff_powed=False):
    """
    RankNet loss introduced in "Learning to Rank using Gradient Descent".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param weight_by_diff: flag indicating whether to weight the score differences by ground truth differences.
    :param weight_by_diff_powed: flag indicating whether to weight the score differences by the squared ground truth differences.
    :return: loss value, a torch.Tensor
    """
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    mask = y_true == padded_value_indicator
    y_pred[mask] = float('-inf')
    y_true[mask] = float('-inf')

    # here we generate every pair of indices from the range of document length in the batch
    document_pairs_candidates = list(product(range(y_true.shape[1]), repeat=2))

    pairs_true = y_true[:, document_pairs_candidates]
    selected_pred = y_pred[:, document_pairs_candidates]

    # here we calculate the relative true relevance of every candidate pair
    true_diffs = pairs_true[:, :, 0] - pairs_true[:, :, 1]
    pred_diffs = selected_pred[:, :, 0] - selected_pred[:, :, 1]

    # here we filter just the pairs that are 'positive' and did not involve a padded instance
    # we can do that since in the candidate pairs we had symetric pairs so we can stick with
    # positive ones for a simpler loss function formulation
    the_mask = (true_diffs > 0) & (~torch.isinf(true_diffs))
    pred_diffs = pred_diffs[the_mask]

    weight = None
    if weight_by_diff:
        abs_diff = torch.abs(true_diffs)
        weight = abs_diff[the_mask]
    elif weight_by_diff_powed:
        true_pow_diffs = torch.pow(
            pairs_true[:, :, 0], 2) - torch.pow(pairs_true[:, :, 1], 2)
        abs_diff = torch.abs(true_pow_diffs)
        weight = abs_diff[the_mask]

    # here we 'binarize' true relevancy diffs since for a pairwise loss we just need to know
    # whether one document is better than the other and not about the actual difference in
    # their relevancy levels
    true_diffs = (true_diffs > 0).type(torch.float32)
    true_diffs = true_diffs[the_mask]

    return BCEWithLogitsLoss(weight=weight)(pred_diffs, true_diffs)


def lambdaLoss(y_pred, y_true, eps=DEFAULT_EPS, padded_value_indicator=PADDED_Y_VALUE, weighing_scheme=None, k=None, sigma=1., mu=10.,
               reduction="mean", reduction_log="binary"):
    """
    LambdaLoss framework for LTR losses implementations, introduced in "The LambdaLoss Framework for Ranking Metric Optimization".
    Contains implementations of different weighing schemes corresponding to e.g. LambdaRank or RankNet.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :param weighing_scheme: a string corresponding to a name of one of the weighing schemes
    :param k: rank at which the loss is truncated
    :param sigma: score difference weight used in the sigmoid function
    :param mu: optional weight used in NDCGLoss2++ weighing scheme
    :param reduction: losses reduction method, could be either a sum or a mean
    :param reduction_log: logarithm variant used prior to masking and loss reduction, either binary or natural
    :return: loss value, a torch.Tensor
    """

    device = y_pred.device
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    padded_mask = y_true == padded_value_indicator
    y_pred[padded_mask] = float("-inf")
    y_true[padded_mask] = float("-inf")

    # Here we sort the true and predicted relevancy scores.
    y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
    y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

    # After sorting, we can mask out the pairs of indices (i, j) containing index of a padded element.
    true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
    true_diffs = true_sorted_by_preds[:, :,
                                      None] - true_sorted_by_preds[:, None, :]
    padded_pairs_mask = torch.isfinite(true_diffs)

    if weighing_scheme != "ndcgLoss1_scheme":
        padded_pairs_mask = padded_pairs_mask & (true_diffs > 0)

    ndcg_at_k_mask = torch.zeros(
        (y_pred.shape[1], y_pred.shape[1]), dtype=torch.bool, device=device)
    ndcg_at_k_mask[:k, :k] = 1

    # Here we clamp the -infs to get correct gains and ideal DCGs (maxDCGs)
    true_sorted_by_preds.clamp_(min=0.)
    y_true_sorted.clamp_(min=0.)

    # Here we find the gains, discounts and ideal DCGs per slate.
    pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
    D = torch.log2(1. + pos_idxs.float())[None, :]
    maxDCGs = torch.sum(((torch.pow(2, y_true_sorted) - 1) / D)
                        [:, :k], dim=-1).clamp(min=eps)
    G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

    # Here we apply appropriate weighing scheme - ndcgLoss1, ndcgLoss2, ndcgLoss2++ or no weights (=1.0)
    if weighing_scheme is None:
        weights = 1.
    else:
        weights = globals()[weighing_scheme](
            G, D, mu, true_sorted_by_preds)  # type: ignore

    # We are clamping the array entries to maintain correct backprop (log(0) and division by 0)
    scores_diffs = (
        y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]).clamp(min=-1e8, max=1e8)
    scores_diffs.masked_fill(torch.isnan(scores_diffs), 0.)
    weighted_probas = (torch.sigmoid(
        sigma * scores_diffs).clamp(min=eps) ** weights).clamp(min=eps)
    if reduction_log == "natural":
        losses = torch.log(weighted_probas)
    elif reduction_log == "binary":
        losses = torch.log2(weighted_probas)
    else:
        raise ValueError(
            "Reduction logarithm base can be either natural or binary")

    if reduction == "sum":
        loss = -torch.sum(losses[padded_pairs_mask & ndcg_at_k_mask])
    elif reduction == "mean":
        loss = -torch.mean(losses[padded_pairs_mask & ndcg_at_k_mask])
    else:
        raise ValueError("Reduction method can be either sum or mean")

    return loss


def ndcgLoss1_scheme(G, D, *args):
    return (G / D)[:, :, None]


def ndcgLoss2_scheme(G, D, *args):
    pos_idxs = torch.arange(1, G.shape[1] + 1, device=G.device)
    delta_idxs = torch.abs(pos_idxs[:, None] - pos_idxs[None, :])
    deltas = torch.abs(torch.pow(torch.abs(
        D[0, delta_idxs - 1]), -1.) - torch.pow(torch.abs(D[0, delta_idxs]), -1.))
    deltas.diagonal().zero_()

    return deltas[None, :, :] * torch.abs(G[:, :, None] - G[:, None, :])


def lambdaRank_scheme(G, D, *args):
    return torch.abs(torch.pow(D[:, :, None], -1.) - torch.pow(D[:, None, :], -1.)) * torch.abs(G[:, :, None] - G[:, None, :])


def ndcgLoss2PP_scheme(G, D, *args):
    return args[0] * ndcgLoss2_scheme(G, D) + lambdaRank_scheme(G, D)


def rankNet_scheme(G, D, *args):
    return 1.


def rankNetWeightedByGTDiff_scheme(G, D, *args):
    return torch.abs(args[1][:, :, None] - args[1][:, None, :])


def rankNetWeightedByGTDiffPowed_scheme(G, D, *args):
    return torch.abs(torch.pow(args[1][:, :, None], 2) - torch.pow(args[1][:, None, :], 2))
