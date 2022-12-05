import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import time


class Generate:
    # Generate 2d or 3d fractal
    def __init__(self, beta=4, seed=117, size=256, dimension=2, preview="False", save="False", method="ifft"):
        self.beta = beta
        self.seed = seed
        self.size = size
        self.dimension = dimension
        assert self.dimension == 2 or self.dimension == 3, "Dimension must be either 2 or 3"

        # Set Seed
        np.random.seed(seed)

        if dimension == 2 and method == "ifft":
            # Build power spectrum
            f = [x/size for x in range(1, int(size/2)+1)] + [x/size for x in range(-int(size/2), 0)]
            u = np.reshape(f, (size, 1))
            v = np.reshape(f, (1, size))
            pattern = (u**2 + v**2)**(-beta/2)
            # Noise and ifft
            phases = np.random.normal(0, 255, size=[size, size])
            pattern = np.fft.ifftn(pattern**0.5 * (np.cos(2*np.pi*phases)+1j*np.sin(2*np.pi*phases)))
            # Normalize result
            pattern = np.real(pattern)
            self.pattern = (pattern-np.amin(pattern))/np.amax(pattern-np.amin(pattern))

        if dimension == 3 and method == "ifft":
            # Build power spectrum
            f = [x/size for x in range(1, int(size/2)+1)] + [x/size for x in range(-int(size/2), 0)]
            u = np.reshape(f, (size, 1))
            v = np.reshape(f, (1, size))
            w = np.reshape(f, (size, 1, 1))
            pattern = (u**2 + v**2 + w**2)**(-beta/2)
            # Noise and ifft
            phases = np.random.normal(0, 255, size=[size, size, size])
            pattern = np.fft.ifftn(pattern**0.5 * (np.cos(2*np.pi*phases)+1j*np.sin(2*np.pi*phases)))
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

    def preview2d(self, size=256):
        # 2d grayscale and BW previews
        if self.dimension == 2:
            preview = cv2.resize(self.pattern, [size, size], interpolation=cv2.INTER_AREA)
            prev_bw = (preview > .5)
            previews = [preview, prev_bw]
            for i in range(2):
                plt.subplot(1, 2, i+1), plt.imshow(previews[i], 'gray')
                plt.xticks([]), plt.yticks([])
            plt.show()
        # 2d slices of 3d fractals for preview
        if self.dimension == 3:
            preview = cv2.resize(self.pattern[-1, :, :], [size, size], interpolation=cv2.INTER_AREA)
            prev_bw = (preview > .5)
            previews = [preview, prev_bw]
            for i in range(2):
                plt.subplot(1, 2, i+1), plt.imshow(previews[i], 'gray')
                plt.xticks([]), plt.yticks([])
            plt.show()

    def boxcount(self, threshold=.5):
        if self.dimension == 2:
            def count(img, k):
                box = np.add.reduceat(
                    np.add.reduceat(fractal, np.arange(0, fractal.shape[0], k), axis=0),
                    np.arange(0, fractal.shape[1], k), axis=1)
                return len(np.where((box > 0) & (box < k*k))[0])
            fractal = (self.pattern < threshold)
            p = min(fractal.shape)
            n = 2**np.floor(np.log(p)/np.log(2))
            n = int(np.log(n)/np.log(2))
            sizes = 2**np.arange(n-1, 0, -1)
            counts = []
            for size in sizes:
                counts.append(count(fractal, size))
            m, b = np.polyfit(np.log(sizes), np.log(counts), 1)
            print(f"D = {round(-m,3)}")
            #return -m, b, sizes, counts

        if self.dimension == 3:
            def count(img, k):
                reducer = np.add.reduceat(np.add.reduceat(fractal, np.arange(0, fractal.shape[0], k), axis=0),
                                          np.arange(0, fractal.shape[1], k), axis=1)
                box = np.add.reduceat(reducer, np.arange(0, fractal.shape[2], k), axis=2)
                return len(np.where((box > 0) & (box < k*k*k))[0])
            fractal = (self.pattern < threshold)
            p = min(fractal.shape)
            n = 2**np.floor(np.log(p)/np.log(2))
            n = int(np.log(n)/np.log(2))
            sizes = 2**np.arange(n-1, 0, -1)
            counts = []
            for size in sizes:
                counts.append(count(fractal, size))
            m, b = np.polyfit(np.log(sizes), np.log(counts), 1)
            print(f"D = {round(-m,3)}")
            #return -m, b, sizes, counts

    def write(self, location="E:/fractals"):
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


"""
#Testing:
begin = time.time()
print(finish - begin)
finish = time.time()
"""
test = Generate(beta=4, seed=117, size=256, dimension=3)
#test.preview2d()
test.previewAnim(3)
#test.boxcount2d()
test.boxcount()
"""
test.write()
"""

