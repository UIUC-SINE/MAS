import numpy as np
import pkg_resources

strands = np.load(pkg_resources.resource_filename('mas', 'data/strands.npy'))
# old_strands = np.load(pkg_resources.resource_filename('mas', 'data/old_strands.npy'))
strands_ext = np.load(pkg_resources.resource_filename('mas', 'data/strands_ext.npy'))
strand_highres = np.load(pkg_resources.resource_filename('mas', 'data/strand_highres.npy'))
