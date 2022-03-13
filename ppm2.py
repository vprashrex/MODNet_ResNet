import cv2
import numpy as np
from torch.utils.data import DataLoader
from scipy.ndimage import grey_dilation, grey_erosion
import random
import os
import numpy
from torchvision.transforms import functional as F


class PPM100:
    def __init__(self,
                 root, matte_suffix='.jpg',
                 image_suffix='.jpg',
                 transform=False):
        super(PPM100, self).__init__()
        self.root = root
        self.matte_suffix = matte_suffix
        self.image_suffix = image_suffix
        self.transform = transform
        self.image_ids = []
        self.base_size = 768
        self.crop_size = 512
        self.retain_rato = (0.5, 0.5)
        self.mean = [0.5,0.5,0.5]
        self.std = [0.5,0.5,0.5]
        self.__read_image_ids()

    def __read_image_ids(self):
        image_dir = os.path.join(self.root, 'image')

        for image_name in os.listdir(image_dir):
            self.image_ids.append(image_name.split('.')[0])

    def resize_short_length(self, image, matte):
        h, w = image.shape[:2]
        if h < w:
            new_h = self.base_size
            new_w = int(self.base_size * w/h + 0.5)
        else:
            new_w = self.base_size
            new_h = int(self.base_size * h/w + 0.5)
        image = cv2.resize(image, (new_w, new_h), cv2.INTER_CUBIC)
        matte = cv2.resize(matte, (new_w, new_h), cv2.INTER_CUBIC)
        return image, matte

    def rand_crop(self, image, matte):
        h, w = image.shape[:2]
        x = random.randint(0, w - self.crop_size)
        y = random.randint(0, h - self.crop_size)
        image = image[y:y+self.crop_size, x:x+self.crop_size]
        matte = matte[y:y+self.crop_size, x:x+self.crop_size]
        return image, matte

    def normalize(self, image):
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        mean = np.array(mean)[np.newaxis, np.newaxis, :]
        std = np.array(std)[np.newaxis, np.newaxis, :]
        image = F.normalize(image, mean, std)
        return image

    def RandomJitter(self, image):
        image = image/255.0
        img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hue = np.random.randint(-40, 40)
        img[:, :, 0] = np.remainder(img[:, :, 0].astype(np.float32) + hue, 360)
        sat_bar = img[:, :, 1].mean()
        sat_jitter = np.random.rand()*(1.1 - sat_bar)/5 - (1.1 - sat_bar)/10
        sat = img[:, :, 1]
        sat = np.abs(sat+sat_jitter)
        sat[sat > 1] = 2 - sat[sat > 1]
        img[:, :, 1] = sat
        val_bar = img[:, :, 2].mean()
        val_jitter = np.random.rand()*(1.1 - val_bar)/5-(1.1 - val_bar)/10
        val = img[:, :, 2]
        val = np.abs(val + val_jitter)
        val[val > 1] = 2 - val[val > 1]
        img[:, :, 2] = val
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        img = img*255.0
        return img

    def foreground_jitter(self, img, alpha):
        bg = img
        bg = bg.astype(np.float32)
        img = self.RandomJitter(img)
        alpha = np.dstack([alpha]*3)
        #alpha = alpha.astype(np.float32)
        #alpha = alpha/255.0
        res = alpha * img + (1-alpha) * bg
        return res

    def background_jitter(self, img, alpha):
        fg = img
        fg = fg.astype(np.float32)
        bg = self.RandomJitter(img)
        alpha = np.dstack([alpha]*3)
        #alpha = alpha.astype(np.float32)
        #alpha = alpha/255.0
        res = alpha * img + (1-alpha) * bg
        return res

    def fg_grayscale(self, img, alpha=None):
        bg = img.astype(np.float32)
        alpha = np.dstack([alpha]*3)
        #alpha = alpha.astype(np.float32)
        #alpha = alpha/255.0
        im = F.to_tensor(img)
        im = F.rgb_to_grayscale(im, num_output_channels=3).contiguous()
        im = numpy.array(im)
        im = np.transpose(im, (1, 2, 0))
        res = alpha * im + (1-alpha) * bg
        return res

    def bg_grayscale(self, img, alpha):
        fg = img.astype(np.float32)
        alpha = np.dstack([alpha]*3)
        #alpha = alpha.astype(np.float32)
        #alpha = alpha/255.0
        bg = F.to_tensor(img)
        bg = F.rgb_to_grayscale(bg, num_output_channels=3).contiguous()
        bg = numpy.array(bg)
        bg = np.transpose(bg, (1, 2, 0))
        res = alpha * fg + (1-alpha) * bg
        return res

    def RandomCenterCrop(self, im, matte):
        r_w = self.retain_rato[0]
        r_h = self.retain_rato[1]
        h, w = im.shape[:2]
        if r_w == 1. and r_h == 1.:
            if matte is None:
                return im
            else:
                return im, matte
        else:
            rand_w = np.random.randint(w * (1-r_w))
            rand_h = np.random.randint(h * (1-r_h))
            offset_w = 0 if rand_w == 0 else np.random.randint(rand_w)
            offset_h = 0 if rand_h == 0 else np.random.randint(rand_h)
            p0, p1, p2, p3 = offset_h, h + offset_h - \
                rand_h, offset_w, w + offset_w - rand_w
            im = im[p0:p1, p2:p3, :]
            if matte is not None:
                matte = matte[p0:p1, p2:p3]

        return im, matte

    def gen_trimap(self, matte):
        h, w = matte.shape
        side = int((h+w)/2 * 0.05)
        #side = 512
        im_size = side
        trimap = (matte >= 0.9).astype('float32')
        not_bg = (matte > 0).astype('float32')
        d_size = im_size // 256 * random.randint(10, 20)
        e_size = im_size // 256 * random.randint(10, 20)

        trimap[np.where((grey_dilation(not_bg, size=(d_size, d_size))
                         - grey_erosion(trimap, size=(e_size, e_size))) != 0)] = 0.5

        return trimap

    def matte_transform(self, matte):
        matte = matte.astype(np.float32)
        matte = matte/255.0
        return matte

    def RandomHorizontalFlip(self, image, matte):
        image = cv2.flip(image, 1)
        matte = cv2.flip(matte, 1)
        return image, matte
    
    def input_transform(self,image):
        
        image = image/255.0
        image -= self.mean
        image /= self.std
        return image

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_name = os.path.join(
            self.root, 'image', image_id + self.image_suffix
        )
        matte_name = os.path.join(
            self.root, 'matte', image_id + self.matte_suffix
        )

        image = cv2.imread(image_name, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        matte = cv2.imread(matte_name, cv2.IMREAD_GRAYSCALE)

        assert image is not None, 'can not read image form ' + image_name
        assert matte is not None, 'can not read matte from ' + matte_name

        if random.random() < 0.5:
            image, matte = self.resize_short_length(image, matte)
            if random.random() < 0.1:
                image, matte = self.RandomCenterCrop(image, matte)
                image = cv2.resize(image, (512, 512), cv2.INTER_CUBIC)
                matte = cv2.resize(matte, (512, 512), cv2.INTER_CUBIC)
            else:
                image, matte = self.rand_crop(image, matte)
        else:
            image = cv2.resize(image, (512, 512), cv2.INTER_CUBIC)
            matte = cv2.resize(matte, (512, 512), cv2.INTER_CUBIC)

        if random.random() < 0.15:
            image, matte = self.RandomHorizontalFlip(image, matte)
        
        if self.transform:
            image = image.astype(np.float32)
            matte = self.matte_transform(matte)
            trimap = self.gen_trimap(matte)
            
            if random.random() < 0.4:
                if random.random() < 0.6:
                    image = self.foreground_jitter(image, matte)
                else:
                    image = self.background_jitter(image, matte)

                if random.random() < 0.3:
                    image = self.fg_grayscale(image, matte)
                else:
                    image = self.bg_grayscale(image, matte)
            
            image = self.input_transform(image)
            image = np.transpose(image, (2, 0, 1))
            matte = np.expand_dims(matte,axis=0)
            trimap = np.expand_dims(trimap,axis=0)
        
            return image,trimap,matte
        else:
            return image,matte
        
    def __len__(self):
        return len(self.image_ids)


if __name__ == '__main__':
    ds = PPM100('PhotoMatte300', transform=True)
    loader = DataLoader(ds, 1)
    for i, (x, y, z) in enumerate(loader):
        x = x*0.5+0.5
        img = x.cpu().detach().numpy()[0]
        #matte = y.cpu().detach().numpy()[0]
        #matte = np.transpose(matte,(1,2,0))
        img = np.transpose(img, (1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = np.uint8(img*255.0)
        #matte = np.uint8(matte*255.0)
        cv2.imwrite('result/{}.png'.format(i), img)
