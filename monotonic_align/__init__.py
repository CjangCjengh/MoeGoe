from numpy import zeros, int32, float32
from torch import from_numpy
from monotonic_align.monotonic_align.core import maximum_path_c

def maximum_path(neg_cent, mask):
  """ Cython optimized version.
  neg_cent: [b, t_t, t_s]
  mask: [b, t_t, t_s]
  """
  device = neg_cent.device
  dtype = neg_cent.dtype
  neg_cent = neg_cent.data.cpu().numpy().astype(float32)
  path = zeros(neg_cent.shape, dtype=int32)

  t_t_max = mask.sum(1)[:, 0].data.cpu().numpy().astype(int32)
  t_s_max = mask.sum(2)[:, 0].data.cpu().numpy().astype(int32)
  maximum_path_c(path, neg_cent, t_t_max, t_s_max)
  return from_numpy(path).to(device=device, dtype=dtype)
