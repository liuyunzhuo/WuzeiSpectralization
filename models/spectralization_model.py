from .pix2pix_model import Pix2PixModel
import torch
import numpy as np


class SpectralizationModel(Pix2PixModel):
    """This is a subclass of Pix2PixModel for blur image spectralization (blur image -> spectral images cube).

    The model training requires '-dataset_model spectralization' dataset.
    It trains a pix2pix model, mapping from 1 channel to multi channels.
    By default, the spectralization dataset will automatically set '--input_nc ' and '--output_nc '.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        By default, we use 'colorization' dataset for this model.
        See the original pix2pix paper (https://arxiv.org/pdf/1611.07004.pdf) and colorization results (Figure 9 in the paper)
        """
        Pix2PixModel.modify_commandline_options(parser, is_train)
        parser.set_defaults(dataset_mode='spectralization', netG='unet_512')
        return parser

    def __init__(self, opt):
        """Initialize the class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        For visualization, we set 'visual_names' as 'real_A' (input real image),
        'real_B_rgb' (ground truth RGB image), and 'fake_B_rgb' (predicted RGB image)
        We convert the Lab image 'real_B' (inherited from Pix2pixModel) to a RGB image 'real_B_rgb'.
        we convert the Lab image 'fake_B' (inherited from Pix2pixModel) to a RGB image 'fake_B_rgb'.
        """
        # reuse the pix2pix model
        Pix2PixModel.__init__(self, opt)
        # specify the images to be visualized.
        self.visual_names = ['real_A', 'fake_B_2', 'real_B_2', 'fake_B_15', 'real_B_15', 'fake_B_31', 'real_B_31']

    def strench_hw_image(self, image):
        """Strench 1-channel gray image"""
        image = torch.unsqueeze(image, 0)
        image = image.data.cpu().float().numpy()
        image_min = np.min(image)
        image_max = np.max(image)
        max_min = image_max - image_min
        stretched_img = (image - image_min) / max_min
        stretched_img = np.transpose(stretched_img.astype(np.float32), (1, 2, 0)) * 255
        stretched_img = np.repeat(stretched_img, 3, axis=2)
        return stretched_img

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        self.real_B_2 = self.strench_hw_image(self.real_B[0, 1, :, :])
        self.fake_B_2 = self.strench_hw_image(self.fake_B[0, 1, :, :])
        self.real_B_15 = self.strench_hw_image(self.real_B[0, 14, :, :])
        self.fake_B_15 = self.strench_hw_image(self.fake_B[0, 14, :, :])
        self.real_B_31 = self.strench_hw_image(self.real_B[0, 30, :, :])
        self.fake_B_31 = self.strench_hw_image(self.fake_B[0, 30, :, :])

    def get_spectral_radiance_dict(self):
        """Export spectral_cube_dict for mat file """
        cube = self.fake_B[0, :, :, :]
        # adapt MATLAB CxHxW -》HxWxC
        cube = cube.permute(1, 2, 0)
        cube = cube.cpu().detach().numpy()
        bands = np.asarray(self.opt.out_channel)
        return {'cube': cube, 'bands': bands}
