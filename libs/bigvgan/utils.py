# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.
# Vendored for WESPER from NVIDIA/BigVGAN at commit 7d2b454 (see UPSTREAM_COMMIT).
# Changed: relative imports; only the two helpers bigvgan.py uses are kept (the rest needed
#   matplotlib, scipy and training code).


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)
