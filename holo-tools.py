import random
import subprocess
import sys
import pyfftw
import numpy

from PIL import Image


def float2bitmap(x: numpy.ndarray):
    mx = x.max()
    return numpy.array(x/mx * 255, numpy.uint8)


def bitmap2float(x: numpy.ndarray):
    mx = x.max()
    return x/mx


def gs_algo(target):
    d = bitmap2float(target) + 1j*0
    # d = bitmap2float(target) + 1j*random.random()
    a = pyfftw.interfaces.numpy_fft.ifft2(d)

    # TODO switch matrix quadrants

    for i in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        print(f'pass {i}')
        b = 1 * numpy.exp(1j * a.imag)
        c = pyfftw.interfaces.numpy_fft.fft2(b)
        d = target.real * numpy.exp(1j * c.imag)
        a = pyfftw.interfaces.numpy_fft.ifft2(d)
    return a.imag


def main(argv):
    mode = argv[1]
    in_file = argv[2]

    in_img = Image.open(in_file)
    in_arr = numpy.array(in_img, dtype=numpy.double)

    phase = gs_algo(in_arr)

    out_img = Image.fromarray(float2bitmap(pyfftw.interfaces.numpy_fft.fft2(phase).imag))   # retrieve image
    # out_img = Image.fromarray(float2bitmap(phase))
    out_img.show()


if __name__ == '__main__':
    main(sys.argv)
