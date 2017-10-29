import numpy as np
from scipy.ndimage.interpolation import rotate,shift
from keras.datasets import mnist

def rotate_img(img,ang):
    rotimg = rotate(img,ang,reshape=False,order=0)
    return rotimg

def random_shift(img,dcdr=2):
    shifts = np.random.randint(low=-dcdr,high=dcdr,size=2)
    shiftimg = shift(img,shifts,order=0)
    return shiftimg

def get_rfunc(eye, ang, dim_per_view=3, views=12):
    i = np.int(ang/(360/views))
    r_func = np.roll(eye,i,axis=1)
    return r_func

class MNIST:
    def __init__(self):
        """
        MNIST data generator
        """
        (x_train, self.y_train), (x_test, self.y_test) = mnist.load_data()
        self.x_train = x_train.astype('float32') / 255.
        self.x_test = x_test.astype('float32') / 255.
        self.original_dim = np.prod(self.x_train.shape[-2:])

    def generator(self,batch_size=32,do_shift=False,do_rotate=False,shuffle=True,**kwargs):
        outshp = (batch_size,self.original_dim)
        samples = self.x_train.shape[0]
        nbatch = samples%batch_size
        if do_rotate:
            angles0 = kwargs.get("angles",np.arange(360))
            angles = np.random.choice(angles0,size=batch_size)
        if do_shift:
            dcdr = kwargs.get("dcdr",2)
        while True:
            for b in range(0,samples,batch_size):
                xx_train = self.x_train[b:np.min([b+batch_size,samples])]
                xx_transformed = xx_train.copy()
                for i in range(batch_size):
                    if do_rotate:
                        xx_transformed[i] = rotate_img(xx_transformed[i],angles[i])
                    if do_shift:
                        xx_transformed[i] = random_shift(xx_transformed[i],dcdr)
                # xx_train = xx_train.reshape(outshp)
                # xx_transformed = xx_transformed.reshape(outshp)
                if shuffle:
                    ids_shuffle = np.random.choice(range(batch_size),size=batch_size,replace=False)
                    xx_train,xx_transformed = xx_train[ids_shuffle],xx_transformed[ids_shuffle]
                yield xx_train,xx_transformed
