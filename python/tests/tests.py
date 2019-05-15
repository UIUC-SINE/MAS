import unittest
import numpy as np
from mas.forward_model import (
    upsample, downsample, size_equalizer, get_measurements, add_noise
)
from mas.psf_generator import PhotonSieve, PSFs

# subclass unittest.TestCase to add assertEqualNp
class TestCase(unittest.TestCase):
    def assertEqualNp(self, a, b):
        self.assertTrue(np.all(np.isclose(a, b)))

class BlockFunctionTests(TestCase):
    def setUp(self):
        self.x = np.random.random((3, 4, 5, 5))
        self.y = np.random.random((4, 3, 5, 5))

    def test_block_mul(self):
        from mas.block import block_mul

        # matrix x matrix
        result = block_mul(self.x, self.y)
        self.assertEqual(result.shape, (3, 3, 5, 5))

        # matrix x col vec
        result = block_mul(self.x, self.y[:, 0, :, :])
        self.assertEqual(result.shape, (3, 5, 5))

class ForwardModelTests(TestCase):

    def setUp(self):
        self.simple_array = np.array(((1, 2), (3, 4)))

    def test_size_equalizer(self):
        # scale up
        bigger = size_equalizer(self.simple_array, (4, 4))
        self.assertEqualNp(
            bigger,
            np.array(
                ((0, 0, 0, 0),
                 (0, 1, 2, 0),
                 (0, 3, 4, 0),
                 (0, 0, 0, 0))
            )
        )
        # scale down
        self.assertEqualNp(size_equalizer(bigger, (2, 2)), self.simple_array)

        self.assertEqual(
            size_equalizer(np.random.random((5, 5)), (4, 4)).shape,
            (4, 4)
        )

        # test vectorization
        x = np.random.random((3, 3))
        y = np.repeat(x[np.newaxis, :, :], 4, axis=0)
        self.assertEqual(
            size_equalizer(y, (2, 2)).shape,
            (4, 2, 2)
        )

    # upsample and downsample image
    def test_upsample_downsample(self):
        upsampled = upsample(self.simple_array, factor=2)
        self.assertEqualNp(
            upsampled,
            np.array(
                ((1, 1, 2, 2),
                 (1, 1, 2, 2),
                 (3, 3, 4, 4),
                 (3, 3, 4, 4))
            )
        )

        self.assertEqualNp(
            downsample(upsampled, factor=2),
            self.simple_array
        )

class DeconvolutionTests(TestCase):
    def setUp(self):
        self.sources = np.ones((2, 4, 4))
        self.ps = PhotonSieve()
        wavelengths = np.array([33.4e-9, 33.5e-9])
        self.psfs = PSFs(
            self.ps,
            source_wavelengths=wavelengths,
            measurement_wavelengths=wavelengths
        )
        measured = get_measurements(
            sources=self.sources,
            psfs=self.psfs,
            real=True
        )
        self.measured_noisy = add_noise(measured, max_count=10, model='poisson')

    def test_tikhonov(self):
        from mas.deconvolution import tikhonov

        recon = tikhonov(
            sources=self.sources,
            measurements=np.fft.fftshift(self.measured_noisy, axes=(1, 2)),
            psfs=self.psfs,
            tikhonov_lam=5e-2,
            tikhonov_order=1
        )

    def test_admm(self):
        from mas.deconvolution import admm

        recon = admm(
            sources=sources,
            measurements=np.fft.fftshift(measured_noisy, axes=(2,3)),
            psfs=psfs,
            regularizer=regularizer,
            recon_init_method='tikhonov',
            iternum=30,
            nu=14e-2,
            lam=5e-5,
            tikhonov_lam=1e-1,
            tikhonov_order=1,
            patch_shape=(6,6,1),
            transform=dctmtx((6,6,8)),
            learning=True,
            window_size=(30,30),
            group_size=70
            # model=model
        )



class StrandGeneratorTests(TestCase):
    def test_strand(self):
        from mas.strand_generator import strand

        generated = strand(0, 0, thickness=1, intensity=1, image_width=4)
        self.assertEqualNp(
            generated,
            np.array(
                ((1, 0, 0, 0),
                 (1, 0, 0, 0),
                 (1, 0, 0, 0),
                 (1, 0, 0, 0))
            )
        )


if __name__ == '__main__':
    unittest.main()


