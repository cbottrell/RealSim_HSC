#!/bin/env python3
'''
> head coords.txt
# ra         dec             outfile(optional)
4.62630      16.44671        a.png
4.63742      16.44747        b.png
4.62537      16.44345        c.png
4.65146      16.44694        d.png
4.64854      16.44695        e.png
4.65057      16.44706        f.png
4.62701      16.44535        g.png
4.65151      16.44745        h.png
4.63186      16.44911        i.png
4.63830      16.44271        j.png
> python colorPostage.py --user YOUR_ACCOUNT --outDir pngs coords.txt  

'''

import argparse
import tarfile
import subprocess
import tempfile
import getpass
import os, os.path
import contextlib
import logging ; logging.basicConfig(level=logging.INFO)
try:
    import pyfits # type: ignore
except:
    import astropy.io.fits as pyfits # type: ignore
import numpy
import PIL.Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outDir', '-o', required=True)
    parser.add_argument('--user', '-u', required=True)
    parser.add_argument('--filters', '-f', nargs=3, default=['HSC-I', 'HSC-R', 'HSC-G'])
    parser.add_argument('--fov', default='30asec')
    parser.add_argument('--rerun', default='any',
                        choices='la2016 la2014'.split())
    parser.add_argument('--color', choices='hsc sdss'.split(), default='hsc')
    parser.add_argument('input', type=argparse.FileType('r'))
    args = parser.parse_args()

    password = os.environ.get('HSC_PASSWORD') or getpass.getpass('Password? ')
    checkPassword(args.user, password)

    coords, outs = loadCoords(args.input)

    mkdir_p(args.outDir)

    batchSize = 300
    for batchI, batchCoords in enumerate(batch(coords, batchSize)):
        with requestFileFor(batchCoords, args.filters, args.fov, args.rerun) as requestFile:
            tarMembers = queryTar(args.user, password, requestFile)
            for i, rgb in rgbBundle(tarMembers):
                j = batchI * batchSize + i
                outFile = os.path.join(args.outDir, outs[j])
                logging.info('-> {}'.format(outFile))
                makeColorPng(rgb, outFile, args.color)


TOP_PAGE = 'https://hscla.mtk.nao.ac.jp/das_cutout/la2016/'
API = 'https://hscla.mtk.nao.ac.jp/das_cutout/la2016/cgi-bin/cutout'


def loadCoords(input):
    import re
    comment = re.compile(r'\s*(?:$|#)')
    num = 1
    coords = []
    outs = []
    for line in input:
        if comment.match(line):
            continue
        cols = line.split()
        if len(cols) == 2:
            ra, dec = cols
            out = '{}.png'.format(num)
        else:
            ra, dec, out = cols
        ra = float(ra)
        dec = float(dec)
        num += 1
        coords.append([ra, dec])
        outs.append(out)
    return coords, outs


@contextlib.contextmanager
def requestFileFor(coords, filters, fov, rerun):
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write('#? filter ra dec sw sh rerun\n'.encode('utf-8'))
        for coord in coords:
            for filterName in filters:
                tmp.write('{} {} {} {} {} {}\n'.format(filterName, coord[0], coord[1], fov, fov, rerun).encode())
        tmp.flush()
        yield tmp.name


def batch(arr, n):
    i = 0
    while i < len(arr):
        yield arr[i : i + n]
        i += n


def rgbBundle(files):
    mktmp = tempfile.NamedTemporaryFile
    with mktmp() as r, mktmp() as g, mktmp() as b:
        lastObjNum = 0
        rgb = {}
        for info, fileObj in files:
            resNum = int(os.path.basename(info.name).split('-')[0])
            objNum = (resNum - 2) // 3
            if lastObjNum != objNum:
                yield lastObjNum, rgb
                rgb.clear()
                lastObjNum = objNum
            ch = 'gbr'[resNum % 3]
            dst = locals()[ch]
            copyFileObj(fileObj, dst)
            rgb[ch] = dst.name
        if len(rgb) > 0:
            yield objNum, rgb



def copyFileObj(src, dst):
    dst.seek(0)
    dst.write(src.read())
    dst.truncate()


def checkPassword(user, password):
    with tempfile.NamedTemporaryFile() as netrc:
        netrc.write('machine hscla.mtk.nao.ac.jp login {} password {}'.format(user, password).encode('ascii'))
        netrc.flush()
        httpCode = subprocess.check_output(
            ['curl', '--netrc-file', netrc.name, '-o', os.devnull, '-w', '%{http_code}', '-s', TOP_PAGE]).strip()
        if httpCode != b'200':
            raise RuntimeError('Account or Password is not correct')


def queryTar(user, password, requestFile):
    with tempfile.NamedTemporaryFile() as netrc:
        netrc.write('machine hscla.mtk.nao.ac.jp login {} password {}'.format(user, password).encode('ascii'))
        netrc.flush()
        
        pipe = subprocess.Popen([
            'curl', '--netrc-file', netrc.name,
            '--form', 'list=@{}'.format(requestFile),
            '--silent',
            API,
        ], stdout=subprocess.PIPE)

        with tarfile.open(fileobj=pipe.stdout, mode='r|*') as tar:
            while True:
                try:
                    info = tar.next()
                except tarfile.StreamError as e:
                    if tar.offset == 0:
                        break
                    else:
                        raise
                if info is None: break
                logging.info('extracting {}...'.format(info.name))
                f = tar.extractfile(info)
                yield info, f
                f.close()

        pipe.wait()


def makeColorPng(rgb, out, color):
    if len(rgb) == 0:
        return

    with pyfits.open(list(rgb.values())[0]) as hdul:
        template = hdul[1].data

    layers = [numpy.zeros_like(template) for i in range(3)]
    for i, ch in enumerate('rgb'):
        if ch in rgb:
            with pyfits.open(rgb[ch]) as hdul:
                x = scale(hdul[1].data, hdul[0].header['FLUXMAG0'])
                layers[i] = x

    if color == 'hsc':
        layers = hscColor(layers)
    elif color == 'sdss':
        layers = sdssColor(layers)
    else:
        assert False

    layers = numpy.array(layers)
    layers[layers < 0] = 0
    layers[layers > 1] = 1
    layers = layers.transpose((1, 2, 0))[::-1, :, :]

    layers = numpy.array(255 * layers, dtype=numpy.uint8)
    img = PIL.Image.fromarray(layers)
    img.save(out)


def scale(x, fluxMag0):
    mag0 = 19
    scale = 10 ** (0.4 * mag0) / fluxMag0
    x *= scale
    return x


def hscColor(rgb):
    u_min = -0.05
    u_max = 2. / 3.
    u_a = numpy.exp(10.)
    for i, x in enumerate(rgb):
        x = numpy.arcsinh(u_a*x) / numpy.arcsinh(u_a)
        x = (x - u_min) / (u_max - u_min)
        rgb[i] = x
    return rgb


def sdssColor(rgb):
    u_a = numpy.exp(10.)
    u_b = 0.05
    r, g, b = rgb
    I = (r + g + b) / 3.
    for i, x in enumerate(rgb):
        x = numpy.arcsinh(u_a * I) / numpy.arcsinh(u_a) / I * x
        x += u_b
        rgb[i] = x
    return rgb


def mkdir_p(d):
    try:
        os.makedirs(d)
    except:
        pass


if __name__ == '__main__':
    main()
