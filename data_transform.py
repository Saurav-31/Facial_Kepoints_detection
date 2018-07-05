from skimage.transform import rotate, resize
import numpy as np
import torch


class Rotate2d(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        image = rotate(image, self.degree)
        keypoints = self.rot_pts(keypoints, self.degree * np.pi / 180)

        return {'image':image, 'keypoints': keypoints}

    @staticmethod
    def rot_pts(pts, theta):
        flip_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        l = np.array([[i[0] - 50, -i[1] + 50] for i in pts.reshape(14, 2)])
        p = [np.matmul(flip_matrix, l[i]) for i in range(14)]
        q = [[i[0] + 50, -i[1] + 50] for i in p]
        q = np.array(q).reshape(-1)
        return q


class flip2d(object):
    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        image, keypoints = self.flip_img(image, keypoints)
        return {'image': image, 'keypoints': keypoints}

    @staticmethod
    def flip_img(img, points):
        new_img = np.flip(img, 1)
        # new_img = rotate(img, -180)
        return new_img.copy(), np.array([[100 - i[0], i[1]] for i in points.reshape(-1, 2)]).reshape(-1)


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = resize(image, (new_h, new_w))

        # h and w are swapped for keypoints because for images,
        # x and y axes are axis 1 and 0 respectively
        keypoints = keypoints * [new_w / w, new_h / h]

        return {'image': img, 'keypoints': keypoints}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        keypoints = keypoints - [left, top]

        return {'image': image, 'keypoints': keypoints}


class ToTensor(object):

    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        # print((image > 0).sum())
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'keypoints': torch.from_numpy(keypoints)}

