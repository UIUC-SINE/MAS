import numpy as np
import pkg_resources

strand = np.load(pkg_resources.resource_filename('mas', 'data/strand.npy'))
strands = np.load(pkg_resources.resource_filename('mas', 'data/strands.npy'))
