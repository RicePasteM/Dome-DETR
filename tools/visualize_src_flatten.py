import matplotlib.pyplot as plt

def visualize_src_flatten(src_flatten, spatial_shapes, savename="feature", is_flatten=True):
    print("visualizing src_flatten")
    if is_flatten:
        bs, _, c = src_flatten.shape
    else:
        bs = src_flatten.shape[0]
        c = src_flatten.shape[-1]
    visual_features = []
    start_idx = 0
    if is_flatten:
        for i, (h, w) in enumerate(spatial_shapes):
            end_idx = start_idx + h * w
            feat = src_flatten[:, start_idx:end_idx, :]  # [bs, h*w, c]
            feat = feat.view(bs, h, w, c)
            visual_features.append(feat)
            start_idx = end_idx
    else:
        visual_features = src_flatten

    for lvl, feat in enumerate(visual_features):
        aggregated = feat.sum(dim=-1)  # Shape: [bs, h, w]
        aggregated_normalized = (aggregated - aggregated.min()) / (aggregated.max() - aggregated.min())
        plt.figure()
        plt.title(f"Level {lvl} - Channel Sum")
        plt.imshow(aggregated_normalized.squeeze(0).cpu().detach().numpy(), cmap='plasma')
        plt.colorbar()
        plt.savefig(f"./visualize/{savename}_level_{lvl}_channel_sum.png")
        plt.close()