import torch


def maskedfusionin(ir_feat, vis_feat, ir_mask, vis_mask):
    assert (vis_feat.size()[:2] == ir_feat.size()[:2])
    size = ir_feat.size()
    vis_mean, vis_std = calc_mean_fusion_std(vis_feat, mask=vis_mask)
    ir_mean, ir_std = calc_mean_fusion_std(ir_feat, mask=ir_mask)
    normalized_feat = (ir_feat - ir_mean.expand(size)) / ir_std.expand(size)
    vis_normalized_feat = normalized_feat * vis_std.expand(size) + vis_mean.expand(size)
    return ir_feat * (1 - ir_mask) + vis_normalized_feat * ir_mask


def masked_adain(ir_feat, vis_feat, ir_mask, vis_mask):
    assert (vis_feat.size()[:2] == ir_feat.size()[:2])
    size = ir_feat.size()
    vis_mean, vis_std = calc_mean_std(vis_feat, mask=vis_mask)
    ir_mean, ir_std = calc_mean_std(ir_feat, mask=ir_mask)
    normalized_feat = (ir_feat - ir_mean.expand(size)) / ir_std.expand(size)
    vis_normalized_feat = normalized_feat * vis_std.expand(size) + vis_mean.expand(size)
    return ir_feat * (1 - ir_mask) + vis_normalized_feat * ir_mask


def maskedadain(ir_feat, vis_feat, ir_mask, vis_mask):
    assert (vis_feat.size()[:2] == ir_feat.size()[:2])
    size = ir_feat.size()
    vis_mean, vis_std = calc_mean_std(vis_feat, mask=vis_mask)
    ir_mean, ir_std = calc_mean_std(ir_feat, mask=ir_mask)
    normalized_feat = (ir_feat - ir_mean.expand(size)) / ir_std.expand(size)
    vis_normalized_feat = normalized_feat * vis_std.expand(size) + vis_mean.expand(size)
    return ir_feat * (1 - ir_mask) + vis_normalized_feat * ir_mask


def fusion_in(ir_feat, vis_feat):
    assert (vis_feat.size()[:2] == ir_feat.size()[:2])
    size = ir_feat.size()
    ir_mean, ir_std = calc_mean_fusion_std(ir_feat)
    vis_mean, vis_std = calc_mean_fusion_std(vis_feat)
    normalized_feat = (ir_feat - ir_mean.expand(size)) / ir_std.expand(size)
    return normalized_feat * vis_std.expand(size) + vis_mean.expand(size)


def fusiondetails_in(ir_feat, vis_feat, vis_weight):
    assert (vis_feat.size()[:2] == ir_feat.size()[:2])
    size = ir_feat.size()
    ir_mean, ir_std = calc_mean_std(ir_feat)
    vis_mean, vis_std = calc_mean_std(vis_feat)
    normalized_feat = (ir_feat - ir_mean.expand(size)) / ir_std.expand(size)

    vis_weight_mean = vis_weight.mean(dim=0).transpose(-1, -2)  # (4096, 4096)
    # fused_img = torch.matmul(ir_feat.view(bs, -1), vis_weight_mean)
    ir_feat_reshaped = ir_feat.view(size[0], -1)  # (bs, 4096)
    fused_img = torch.matmul(vis_weight_mean, ir_feat_reshaped.t()).t()
    fused_mean, fused_std = calc_mean_std(fused_img.view(size))
    # fused_img_square = torch.matmul(vis_weight_mean, (ir_feat_reshaped ** 2).t()).t()
    # print("fused img square ", fused_img_square.shape)
    # fused_std = (fused_img_square.view(size) - (fused_mean * fused_mean).expand(size)) ** 0.5

    return normalized_feat * fused_std.expand(size) + fused_mean.expand(size)
    # return normalized_feat * vis_std.expand(size) + vis_mean.expand(size)


def calc_mean_fusion_std(feat, eps=1e-5, mask=None):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    if len(size) == 2:
        print("warning： size equal to 2！")
        return calc_mean_fusion_std_2d(feat, eps, mask)
    assert (len(size) == 3)
    C,H,W = size
    if mask is not None:
        flat_mask = mask.view(-1)
        feat_var = feat.view(C, -1)[:, flat_mask == 1]
        feat_var = feat_var.var(dim=0, unbiased=False) + eps
        masked_feat_var = torch.ones(H * W, device=feat.device)
        if torch.sum(flat_mask) > 0:
            masked_feat_var[flat_mask == 1] = feat_var
        feat_std = masked_feat_var.sqrt().view(1, H, W)

        feat_mean_mask = feat.view(C, -1)[:, flat_mask == 1]
        feat_mean_mask = feat_mean_mask.mean(dim=0)
        masked_feat_means = torch.zeros(H * W, device=feat.device)
        if torch.sum(flat_mask) > 0:
            masked_feat_means[flat_mask == 1] = feat_mean_mask
        feat_mean = masked_feat_means.view(1, H, W)
    else:
        feat_var = feat.var(dim=0) + eps
        feat_std = feat_var.sqrt().view(1, H, W)
        feat_mean = feat.mean(dim=0).view( 1, H, W)
    return feat_mean, feat_std


def calc_mean_fusion_std_2d(feat, eps=1e-5, mask=None):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 2)
    C, D = size
    if mask is not None:
        feat_var = feat.view(C, -1)[:, mask.view(-1) == 1]
        feat_var = feat_var.var(dim=0) + eps
        feat_std = feat_var.sqrt().view(1, D)
        feat_mean = feat.view(C, -1)[:, mask.view(-1) == 1]
        feat_mean = feat_mean.mean(dim=0).view(1, D)
    else:
        feat_var = feat.view(C, -1).var(dim=0) + eps
        feat_std = feat_var.sqrt().view(1, D)
        feat_mean = feat.view(C, -1).mean(dim=0).view(1, D)

    return feat_mean, feat_std


def adain(ir_feat, vis_feat):
    assert (vis_feat.size()[:2] == ir_feat.size()[:2])
    size = ir_feat.size()
    ir_mean, ir_std = calc_mean_std(ir_feat)
    vis_mean, vis_std = calc_mean_std(vis_feat)
    normalized_feat = (ir_feat - ir_mean.expand(size)) / ir_std.expand(size)
    return normalized_feat * vis_std.expand(size) + vis_mean.expand(size)


def calc_mean_std(feat, eps=1e-5, mask=None):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    if len(size) == 2:
        return calc_mean_std_2d(feat, eps, mask)

    assert (len(size) == 3)
    C = size[0]
    if mask is not None:
        feat_var = feat.view(C, -1)[:, mask.view(-1) == 1].var(dim=1) + eps
        feat_std = feat_var.sqrt().view(C, 1, 1)
        feat_mean = feat.view(C, -1)[:, mask.view(-1) == 1].mean(dim=1).view(C, 1, 1)
    else:
        feat_var = feat.view(C, -1).var(dim=1) + eps
        feat_std = feat_var.sqrt().view(C, 1, 1)
        feat_mean = feat.view(C, -1).mean(dim=1).view(C, 1, 1)

    return feat_mean, feat_std