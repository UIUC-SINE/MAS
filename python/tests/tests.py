import unittest
import numpy as np
from mas.forward_model import (upsample, downsample, size_equalizer)

# subclass unittest.TestCase to add assertEqualNp
class TestCase(unittest.TestCase):
    def assertEqualNp(self, a, b):
        self.assertTrue(np.all(np.isclose(a, b)))

class BlockFunctionTests(TestCase):
    pass

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


