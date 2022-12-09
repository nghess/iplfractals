import os
import cv2
import time
import numpy as np
from numpy import inf
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
from skimage.transform import resize


class Generate:
    # Generate 2d or 3d fractal
    def __init__(self, beta=4, seed=117, size=256, dimension=2, preview=False, save=False, method="ifft"):
        # Set Seed
        np.random.seed(seed)
        # Set Size
        size = size+1
        # Set properties
        self.beta = beta
        self.seed = seed
        self.size = size
        self.dimension = dimension

        #Alert related
        assert self.dimension == 2 or self.dimension == 3, "Dimension must be either 2 or 3"
        np.seterr(divide='ignore')


        if dimension == 2 and method == "ifft":
            # Build power spectrum
            f = [x/size for x in range(0, int(size/2)+1)] + [x/size for x in range(-int(size/2), 0)]
            u = np.reshape(f, (size, 1))
            v = np.reshape(f, (1, size))
            powerspectrum = (u**2 + v**2)**(-beta/2)
            powerspectrum[powerspectrum == inf] = powerspectrum[0,1]
            # Noise and ifft
            phases = np.random.normal(0, 255, size=[size, size])
            pattern = np.fft.ifftn(powerspectrum**0.5 * (np.cos(2*np.pi*phases)+1j*np.sin(2*np.pi*phases)))
            # Normalize result
            pattern = np.real(pattern)
            self.pattern = (pattern-np.amin(pattern))/np.amax(pattern-np.amin(pattern))

        if dimension == 3 and method == "ifft":
            # Build power spectrum
            f = np.around([x/size for x in range(0, int(size/2)+1)] + [x/size for x in range(-int(size/2), 0)], 4)
            u = np.reshape(f, (size, 1))
            v = np.reshape(f, (1, size))
            w = np.reshape(f, (size, 1, 1))
            powerspectrum = (u**2 + v**2 + w**2)**(-beta/2)
            powerspectrum[powerspectrum == inf] = powerspectrum[0,1,0]
            # Noise and ifft
            phases = np.random.normal(0, 255, size=[size, size, size])
            pattern = np.fft.ifftn(powerspectrum**0.5 * (np.cos(2*np.pi*phases)+1j*np.sin(2*np.pi*phases)))
            # Normalize result
            pattern = np.real(pattern)
            self.pattern = (pattern-np.amin(pattern))/np.amax(pattern-np.amin(pattern))

    def previewAnim(self, reps=3, mode='gs'):
        if reps == 1:
            reps = 2
        for i in range(reps-1):
            for k in range(self.size):
                cv2.imshow('Fractal Preview', self.pattern[k, :, :])
                cv2.waitKey(16)

    def preview2d(self, index=-1, size=256):
        # 2d grayscale and BW previews
        if self.dimension == 2:
            preview = cv2.resize(self.pattern, [size, size], interpolation=cv2.INTER_AREA)
            prev_bw = (preview > .5)
            previews = [preview, prev_bw]
            for i in range(2):
                plt.subplot(1, 2, i+1), plt.imshow(previews[i], 'Greys')
                plt.xticks([]), plt.yticks([])
            plt.show()
        # 2d slices of 3d fractals for preview
        if self.dimension == 3:
            if index != -1:
                assert 0 < index <= 100, "Index must be between 1-100"
                frame = int((index/100)*self.size)
            else:
                frame = -1
            preview = cv2.resize(self.pattern[frame, :, :], [size, size], interpolation=cv2.INTER_AREA)
            prev_bw = (preview > .5)
            previews = [preview, prev_bw]
            for i in range(2):
                plt.subplot(1, 2, i+1), plt.imshow(previews[i], 'Greys')
                plt.xticks([]), plt.yticks([])
            plt.show()

    def preview3d(self):
        # Check if 3 dimensional, and resize to 64x64x64
        assert self.pattern.ndim == 3, "Fractal must be 3 dimensional"
        prev3d = resize(self.pattern, (64, 64, 64))

        # Create vectors for 3d plot
        z, x, y = prev3d.nonzero()
        color = prev3d.flatten()
        color = color[:]

        #Display 3d Fractal
        fig = plt.figure()
        plt.rcParams["figure.figsize"] = 5, 5
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c=color, alpha=1, cmap="Greys")
        plt.show()

    def boxcount(self, threshold=.5, frame=False):
        # 2d box count function
        if self.pattern.ndim == 2 or frame:
            def count(img, k):
                box = np.add.reduceat(
                    np.add.reduceat(fractal, np.arange(0, fractal.shape[0], k), axis=0),
                    np.arange(0, fractal.shape[1], k), axis=1)
                return len(np.where((box > 0) & (box < k*k))[0])

        # 3d box count function
        elif self.pattern.ndim == 3:
            def count(img, k):
                reducer = np.add.reduceat(np.add.reduceat(fractal, np.arange(0, fractal.shape[0], k), axis=0),
                                          np.arange(0, fractal.shape[1], k), axis=1)
                box = np.add.reduceat(reducer, np.arange(0, fractal.shape[2], k), axis=2)
                return len(np.where((box > 0) & (box < k*k*k))[0])

        # Threshold and box count
        fractal = (self.pattern < threshold)
        p = min(fractal.shape)
        n = 2**np.floor(np.log(p)/np.log(2))
        n = int(np.log(n)/np.log(2))
        sizes = 2**np.arange(n-1, 0, -1)
        counts = []
        for size in sizes:
            counts.append(count(fractal, size))
        m, b = np.polyfit(np.log(sizes), np.log(counts), 1)
        return -m

    def avgBoxcount(self):
        # Check if fractal is 3d
        assert self.pattern.ndim == 3, "Average box count is for 3d fractals only."

        def abc(fractal):
            def count(fractal, k):
                box = np.add.reduceat(
                    np.add.reduceat(fractal, np.arange(0, fractal.shape[0], k), axis=0),
                    np.arange(0, fractal.shape[1], k), axis=1)
                return len(np.where((box > 0) & (box < k*k))[0])

            # Threshold and box count
            fractal = (self.pattern < .5)
            p = min(fractal.shape)
            n = 2**np.floor(np.log(p)/np.log(2))
            n = int(np.log(n)/np.log(2))
            sizes = 2**np.arange(n-1, 0, -1)
            counts = []
            for size in sizes:
                counts.append(count(fractal, size))
            m, b = np.polyfit(np.log(sizes), np.log(counts), 1)
            return -m

        boxcounts = []

        for i in range(0, len(self.pattern), 3):
            frame = self.pattern[i, :, :]
            slope2d = abc(frame)
            boxcounts.append(slope2d)
        return np.mean(boxcounts)


    def write(self, location="E:/fractals"):

        # Check if root directory exists
        assert os.path.exists(location), "Root directory doesn't exist."

        # Save 2d fractal
        if self.dimension == 2:
            #folder = f"{location}/{self.seed}/{self.beta}/"
            #os.mkdir(folder)
            cv2.imwrite(f"{self.beta}_{self.seed}.png", self.pattern*255)
        if self.dimension == 3:
            folder = f"{location}/{self.seed}/{self.beta}/"
            if not os.path.exists(folder):
                os.makedirs(folder)
            for i in range(self.size):
                cv2.imwrite(f"{folder}{self.beta}_{self.seed}_{i:03d}.png", self.pattern[i, :, :]*255)