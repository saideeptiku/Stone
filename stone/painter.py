"""
Modules for inpainting RSSI images

calling test functions
InpaintIOBuilder.test_build_input()
InpaintIOBuilder.test_build_output()

generator for training inpainter
Every Sequence must implement the __getitem__ and the __len__ methods. 
If you want to modify your dataset between epochs you may implement on_epoch_end. 
The method __getitem__ should return a complete batch.

"""

import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.utils import Sequence


class InPaintGenerator(Sequence):
    """
    Generator for training the in-painting encoder
    """

    def __init__(self, x, batch_size=32, p_turn_off=(0.0, 0.9),
                 missing_only=True, lazy_off=True):
        """
        args:
        ----
        x: input images

        kwargs:
        -------
        batch_size: bs of generator
        p_turn_off: probability of pixel being turned off
        missing_only: if True, output is copied from output 
                      with the addition of output
        """

        self.x = x
        self.batch_size = batch_size
        self.p_turn_off = p_turn_off
        self.missing_only = missing_only
        self.lazy_off = lazy_off

    def test_mask_plot(self, num_rows=2):
        ins, outs = self.__getitem__(0)

        fig, axes = plt.subplots(num_rows, 4, figsize=(10, 3 * num_rows))

        if len(axes.shape) == 1:
            axes = axes.reshape((-1, 4))

        for i in range(num_rows):

            axes[i, 0].imshow(ins[i, :, :, 1] + outs[i, :, :, 0],
                              vmin=0, vmax=1, cmap='binary')
            axes[i, 0].set_title("ORIGINAL")

            axes[i, 1].imshow(ins[i, :, :, 0], vmin=0, vmax=1, cmap='binary')
            axes[i, 1].set_title("MASK")

            axes[i, 2].imshow(ins[i, :, :, 1], vmin=0, vmax=1, cmap='binary')
            axes[i, 2].set_title("MASK IMG")

            axes[i, 3].imshow(outs[i, :, :, 0], vmin=0, vmax=1, cmap='binary')
            axes[i, 3].set_title("MASKED VALUES")

        plt.show()

    @staticmethod
    def build_mask_off(x, p=(0, 1), lazy=True):
        """
        Build a mask of x depicting visible pixels
        Then turn of pixels in mask based on uniform prob dist of p

        args & kwargs:
        --------------
        x: input image (K, K, 1)
        p: probability distribution of turn off
        lazy: if true, only turn off pixels that have value

        returns:
        --------
        returns a mask (K, K, 1)
        """
        # built mask using AP augmentation strategy
        if not lazy:
            x_mask = x != 0

            num_off = int(np.count_nonzero(x_mask) * np.random.uniform(*p))
            r, c, _ = np.where(x != 0)
            rc_on = np.vstack((r, c)).T
            np.random.shuffle(rc_on)
            rc_off = rc_on[:num_off, :]
            x_mask[rc_off[:, 0], rc_off[:, 1]] = False

        # lazy method
        else:
            p_off = np.random.uniform(*p)
            x_mask = np.random.choice([True, False],
                                      size=x.shape,
                                      p=[p_off, 1-p_off])

        return x_mask.astype(int)

    @staticmethod
    def mask_img(x, p=(0, 1), lazy=True):
        """
        args:
        -----
        x: input image with shape (N, M, 1)
        p: probability bounds over which pixels turned off

        returns:
        --------
        x_mask: mask on ox with value 1 where pixels are turned off
        x_off: image with pixels turned off where x_mask is 1
        x_on: images with pixels where x_mask is 1
        """
        x_mask = InPaintGenerator.build_mask_off(x, p, lazy=lazy)

        x_off = np.ma.masked_array(x, x_mask.astype(bool)).filled(0)

        x_on = np.ma.masked_array(x, ~x_mask.astype(bool)).filled(0)

        return x_mask, x_off, x_on

    def make_batch(self):
        """
        Given the training images generate
        a two channel image with 1st channel showing mask position
        and second channel with original image with masked values
        a one channel image with masked values

        args:
        -----
        x:  numpy array of single channel images (NUM_IMAGES, N, M, 1)
        bs: batch size as int
        p:  lower and upper bound probability of AP being turned off

        returns:
        --------
        maked_images: Array with shape (bs, N, M, 2), 
                      where channel 0 is mask and 1 is masked image
        mask_values:  Array with shape (bs, N, M, 1)
                      has values masked in channel 1 of "masked_images"
        """
        # shape of image as the input
        _, img_h, img_w, _ = self.x.shape

        # matrices to store output batch size in
        masked_images = np.zeros((self.batch_size, img_h, img_w, 2))
        mask_values = np.zeros((self.batch_size, img_h, img_w, 1))

        # shuffle x and keep a batch size x (pick indices)
        inds = np.random.choice(np.arange(self.x.shape[0]),
                                size=self.batch_size)

        # for each image in batch of images call mask_img
        for i, img in enumerate(self.x[inds, :, :, :]):
            x_mask, x_off, x_on = InPaintGenerator.mask_img(img,
                                                            self.p_turn_off, 
                                                            lazy=self.lazy_off)

            # stack x_mask and x_off on axis 0
            stacked_x = np.stack((x_mask, x_off), axis=2)
            stacked_x = stacked_x.reshape(stacked_x.shape[:-1])

            masked_images[i, :, :, :] = stacked_x

            # mask_values is original image or only masked pixels
            if self.missing_only:  # default true
                mask_values[i, :, :, :] = x_on
            else:
                mask_values[i, :, :, :] = img

        # convert matrices to float
        return masked_images.astype(np.float), mask_values.astype(np.float)

    # functions required by this class called by fit

    def on_epoch_end(self):
        pass

    def __len__(self):
        # print("__len__ called", np.ceil(len(self.x) / self.batch_size))
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, index):
        return self.make_batch()


class Painter():
    """
    Class mainly used after deployment of in-painting model
    requirements:
    -------------
    import numpy as np
    from matplotlib import pyplot as plt 
    """

    def __init__(self, train_waps, painter_model, copy_input=True, img_shape=None):
        """
        train_waps: An iterable of train waps as strings
        copy_input: If True, painter places inpainted pixels within input image
                    Otherwise, (False) keep output as is
        img_shape: (H, W), use for make image function
        """
        # will be used when building output
        self.copy_input = copy_input
        self.img_shape = img_shape

        # store train_waps in image format
        # enables quick numpy manip later
        train_waps = np.array(train_waps).reshape((1, -1))
        self.train_waps_img = Painter.make_images(train_waps,
                                                  force_shape=img_shape)

        # the current mask based on last fit function
        self.mask = None

        # save the painter model
        # build a wrapper on the predict function
        self.model = painter_model

    def fit(self, missing_waps):
        """
        build mask based on missing_waps
        so that we don't have to recompute mask on build functions
        missing_waps: WiFi AP names that are missing in x samples 
        """
        # make sure missing waps is in image format
        if len(np.array(missing_waps).shape) != 4:
            missing_waps = np.array(missing_waps).reshape((1, -1))
            missing_waps = Painter.make_images(np.array(missing_waps),
                                               force_shape=self.img_shape)

        # mask is of shape (1, N, N, 1)
        self.mask = np.isin(self.train_waps_img, missing_waps).astype(int)

    def build_input(self, x):
        """
        Build input for inpainting

        args:
        -----
        x: array of inputs in range [0, 1]; shape is (N, M) or (N, K, K, 1)
        """
        if self.mask is None:
            print("WARNING: fit() was not called. Did nothing")
            return None

        # make sure x is in correct shape
        if len(x.shape) != 4:
            x = Painter.make_images(np.array(x),
                                    force_shape=self.img_shape)

        # NOTE: This mask does not need to be applied on X
        # As X here is already expecetd to be missing values
        # stack mask at 0 and x itself on 1
        # return reshaped as (-1, K, K, 2)
        mask = np.repeat(self.mask, x.shape[0], axis=0)
        # x = np.ma.masked_array(x, mask).filled(0)
        x = np.stack((mask, x), axis=3)
        return x.reshape(x.shape[:-1])

    def build_output(self, x, y):
        """
        x: input image with missing waps; preferred shape (-1, K, K, 1)
        y: model output with predicted values; preferred shape (-1, K, K, 1)
        """

        if self.mask is None:
            print("WARNING: fit() was not called. Did nothing")
            return None

        # make sure x is in correct shape
        if len(x.shape) != 4:
            x = Painter.make_images(np.array(x), force_shape=self.img_shape)

        # make sure x is in correct shape
        if len(y.shape) != 4:
            y = Painter.make_images(np.array(y), force_shape=self.img_shape)

        # take values from Y using mask and apply to X
        # X then can be sent to another model to make some prediction
        mask = np.repeat(self.mask, x.shape[0], axis=0)
        mask_y = np.ma.masked_array(y, np.logical_not(mask)).filled(0)
        mask_x = np.ma.masked_array(x, mask).filled(0)

        return mask_x + mask_y

    def predict(self, x):
        """
        merge the output and input of the model

        args:
        -----
        x: input images; single channel (-1, K, K, 1)

        """

        # build input
        masked_inputs = self.build_input(x)
        # pass to model
        paints = self.model.predict(masked_inputs.astype(float))

        if self.copy_input:  # default
            outs = self.build_output(x, paints)
        else:
            outs = paints

        # return output
        return outs

    @staticmethod
    def make_images(vectors, fmt='tf', force_shape=None):
        """convert into N, H, W, C, with C=1 or BW img
        assumes H == W by default, 
        :param vectors: 2d array
        :param fmt: 'tf' (N, H, W, 1) or 'tf' (N, 1, H, W)
        :param force_shape: force shape (H, W), C is always 1
        :returns: returns images in 'th' or 'tf' format
        :rtype: numpy array
        """
        closest_sqaure = np.square(np.ceil(np.sqrt(vectors.shape[1])))

        if force_shape is None:
            req_pad = int(closest_sqaure - vectors.shape[1])
            img_size = int(np.sqrt(closest_sqaure))
            h, w = img_size, img_size
        elif force_shape is not None and len(force_shape) == 2:
            h, w = force_shape
            req_pad = int((h * w) - vectors.shape[1])
        else:
            print("WARNING: invalid input for force_shape. auto computing...")
            req_pad = int(closest_sqaure - vectors.shape[1])
            img_size = int(np.sqrt(closest_sqaure))
            h, w = img_size, img_size

        # pad if required
        if req_pad != 0:
            # do padding here
            vectors = np.hstack((
                vectors,
                np.zeros((vectors.shape[0], req_pad))
            ))

        if fmt == 'tf':
            return vectors.reshape((-1, h, w, 1))
        elif fmt == 'th':
            return vectors.reshape((-1, 1, h, w))
        else:
            print("ERROR: Invalid fmt argument")
            raise Exception

    ###################################
    # SELF TEST METHODS #
    ###################################
    @staticmethod
    def test_build_input():

        # build a vector of fake macs
        def print_img(x):
            print(x.reshape(x.shape[:-1]))

        train_waps = ["WAP_"+str(x) for x in range(9)]

        self = Painter(train_waps)

        missing_waps = ["WAP_3", "WAP_4", "WAP_5"]
        self.fit(missing_waps)

        print("shapes")
        print(self.train_waps_img.shape,
              self.mask.shape)

        print("WiFi APs")
        print_img(self.train_waps_img)
        print("Mask")
        print_img(self.mask)

        # make sample input images
        x = np.random.random((2, 3, 3, 1))
        x[:, 1, :, :] = 0

        print("Input Images")
        print(x.shape)
        print_img(x[0, :, :, 0].reshape((1, 3, 3, 1)))
        print_img(x[1, :, :, 0].reshape((1, 3, 3, 1)))

        masked_x = self.build_input(x)

        print(masked_x.shape)
        print("MASKED STACK 0")
        print_img(masked_x[0, :, :, 0].reshape((1, 3, 3, 1)))
        print_img(masked_x[0, :, :, 1].reshape((1, 3, 3, 1)))

        print("MASKED STACK 1")
        print_img(masked_x[1, :, :, 0].reshape((1, 3, 3, 1)))
        print_img(masked_x[1, :, :, 1].reshape((1, 3, 3, 1)))

    @staticmethod
    def test_build_output():

        # build a vector of fake macs
        def print_img(x):
            print(x.reshape(x.shape[:-1]))

        # prepare the input to model
        mask = np.array([[0, 0, 0, ],
                         [1, 1, 1, ],
                         [0, 0, 0, ]
                         ])

        img_base = np.random.random((3, 3))
        img_masked = np.ma.masked_array(img_base, mask.astype(bool)).filled(0)
        input_x = np.vstack((mask, img_masked)).reshape((1, 3, 3, 2))

        # output to model
        out_y = np.ma.masked_array(img_base,
                                   ~mask.astype(bool)).filled(0).reshape((1, 3, 3, 1))

        # given both things
        # apply output function
        train_waps = ["WAP_"+str(x) for x in range(9)]
        self = Painter(train_waps)

        missing_waps = ["WAP_3", "WAP_4", "WAP_5"]
        self.fit(missing_waps)

        img_masked = img_masked.reshape((1, 3, 3, 1))
        print(img_masked.shape, out_y.shape)

        print("masked img")
        print_img(img_masked)
        print("model output")
        print_img(out_y)

        painted = self.build_output(img_masked, out_y)

        print("build_output")
        print_img(painted)
