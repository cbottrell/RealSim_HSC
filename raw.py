import math
import lsst.daf.persistence as dafPersist
import lsst.afw.geom as afwGeom
import lsst.afw.coord as afwCoord
import lsst.afw.image as afwImage
import lsst.geom as lsstGeom

def findTractPatch(skymap, coord):
    tractInfo, patchInfo = skymap.findClosestTractPatchList([coord])[0]
    wcs = tractInfo.getWcs()
    #x, y = wcs.skyToPixel(coord.toIcrs())
    x, y = wcs.skyToPixel(coord)
    for pi in patchInfo:
        if pi.getInnerBBox().contains(lsstGeom.Point2I(int(math.floor(x+0.5)), int(math.floor(y+0.5)))):
            break

    return tractInfo.getId(), '%d,%d' % (pi.getIndex()), wcs

def getCorners(skymap, tract, ipx, ipy):
    pxmin, pymin = skymap[tract].getPatchInfo((ipx,ipy)).getOuterBBox().getBegin()
    pxmax, pymax = skymap[tract].getPatchInfo((ipx,ipy)).getOuterBBox().getEnd()

    return pxmin, pymin, pxmax, pymax

def getBBox(xmin, ymin, xmax, ymax, pxmin, pymin, pxmax, pymax):
    return lsstGeom.Box2I(lsstGeom.Point2I(max(xmin,pxmin),max(ymin,pymin)),
                         lsstGeom.Point2I(min(xmax,pxmax-1),min(ymax,pymax-1)))

def cutoutImage(butler, skymap, specObjID, ra, dec, bands, width=60.,outdir='.'):
    # width is in arcsec

    #coord = afwCoord.Coord(afwGeom.Point2D(ra, dec))
    coord = lsstGeom.SpherePoint(ra, dec, lsstGeom.degrees)
    tract, patch, wcs = findTractPatch(skymap, coord)
    print(tract, patch)
    ipx, ipy = map(int, patch.split(','))

    #x, y = wcs.skyToPixel(coord.toIcrs())
    x, y = wcs.skyToPixel(coord)
    ix = int(math.floor(x+0.5))
    iy = int(math.floor(y+0.5))
    width_in_pixel = int(width / wcs.getPixelScale().asArcseconds() / 2) * 2 + 1

    pxmin, pymin, pxmax, pymax = getCorners(skymap, tract, ipx, ipy)

    bbox = lsstGeom.Box2I(lsstGeom.Point2I(ix-width_in_pixel//2,iy-width_in_pixel//2),
                         lsstGeom.Point2I(ix+width_in_pixel//2,iy+width_in_pixel//2))
    xmin, ymin = bbox.getBegin()
    xmax, ymax = bbox.getEnd()
    xmax -= 1
    ymax -= 1

    patches = list()
    bboxes = list()

    patches.append((ipx,ipy))
    bboxes.append(getBBox(xmin, ymin, xmax, ymax, pxmin, pymin, pxmax, pymax))

    if ix-width_in_pixel//2 < pxmin:
        patches.append((ipx-1,ipy))
        _pxmin, _pymin, _pxmax, _pymax = getCorners(skymap, tract, ipx-1, ipy)
        bboxes.append(getBBox(xmin, ymin, xmax, ymax, _pxmin, _pymin, _pxmax, _pymax))
        if iy-width_in_pixel//2 < pymin:
            patches.append((ipx,ipy-1))
            _pxmin, _pymin, _pxmax, _pymax = getCorners(skymap, tract, ipx, ipy-1)
            bboxes.append(getBBox(xmin, ymin, xmax, ymax, _pxmin, _pymin, _pxmax, _pymax))

            patches.append((ipx-1,ipy-1))
            _pxmin, _pymin, _pxmax, _pymax = getCorners(skymap, tract, ipx-1, ipy-1)
            bboxes.append(getBBox(xmin, ymin, xmax, ymax, _pxmin, _pymin, _pxmax, _pymax))
        elif iy+width_in_pixel//2 > pymax:
            patches.append((ipx,ipy+1))
            _pxmin, _pymin, _pxmax, _pymax = getCorners(skymap, tract, ipx, ipy+1)
            bboxes.append(getBBox(xmin, ymin, xmax, ymax, _pxmin, _pymin, _pxmax, _pymax))

            patches.append((ipx-1,ipy+1))
            _pxmin, _pymin, _pxmax, _pymax = getCorners(skymap, tract, ipx-1, ipy+1)
            bboxes.append(getBBox(xmin, ymin, xmax, ymax, _pxmin, _pymin, _pxmax, _pymax))
        else:
            pass
    elif ix+width_in_pixel//2 > pxmax:
        patches.append((ipx+1,ipy))
        _pxmin, _pymin, _pxmax, _pymax = getCorners(skymap, tract, ipx+1, ipy)
        bboxes.append(getBBox(xmin, ymin, xmax, ymax, _pxmin, _pymin, _pxmax, _pymax))
        if iy-width_in_pixel//2 < pymin:
            patches.append((ipx,ipy-1))
            _pxmin, _pymin, _pxmax, _pymax = getCorners(skymap, tract, ipx, ipy-1)
            bboxes.append(getBBox(xmin, ymin, xmax, ymax, _pxmin, _pymin, _pxmax, _pymax))

            patches.append((ipx+1,ipy-1))
            _pxmin, _pymin, _pxmax, _pymax = getCorners(skymap, tract, ipx+1, ipy-1)
            bboxes.append(getBBox(xmin, ymin, xmax, ymax, _pxmin, _pymin, _pxmax, _pymax))
        elif iy+width_in_pixel//2 > pymax:
            patches.append((ipx,ipy+1))
            _pxmin, _pymin, _pxmax, _pymax = getCorners(skymap, tract, ipx, ipy+1)
            bboxes.append(getBBox(xmin, ymin, xmax, ymax, _pxmin, _pymin, _pxmax, _pymax))

            patches.append((ipx+1,ipy+1))
            _pxmin, _pymin, _pxmax, _pymax = getCorners(skymap, tract, ipx+1, ipy+1)
            bboxes.append(getBBox(xmin, ymin, xmax, ymax, _pxmin, _pymin, _pxmax, _pymax))
        else:
            pass
    else:
        if iy-width_in_pixel//2 < pymin:
            patches.append((ipx,ipy-1))
            _pxmin, _pymin, _pxmax, _pymax = getCorners(skymap, tract, ipx, ipy-1)
            bboxes.append(getBBox(xmin, ymin, xmax, ymax, _pxmin, _pymin, _pxmax, _pymax))
        elif iy+width_in_pixel//2 > pymax:
            patches.append((ipx,ipy+1))
            _pxmin, _pymin, _pxmax, _pymax = getCorners(skymap, tract, ipx, ipy+1)
            bboxes.append(getBBox(xmin, ymin, xmax, ymax, _pxmin, _pymin, _pxmax, _pymax))
        else:
            pass

    for band in bands:

        try:
            fname = butler.get('deepCoadd_calexp_filename',
                               {'tract': tract, 'patch': patch, 'filter': band})[0]
            exposure = afwImage.ExposureF(fname, origin=afwImage.PARENT)
            psfExp = exposure.getPsf().computeImage(lsstGeom.Point2D(ix, iy))
            psf_fname = '%s/%s_%s_psf.fits' % (outdir, specObjID, band)
            psfExp.writeFits(psf_fname)

            x0, y0 = exposure.getXY0()

            newExp = afwImage.ExposureF(width_in_pixel, width_in_pixel)
            wcs = exposure.getWcs()

            for _patch, _bbox in zip(patches, bboxes):
                fname = butler.get('deepCoadd_calexp_filename',
                                   {'tract': tract, 'patch': '%d,%d' % (_patch), 'filter': band})[0]
                exposure = afwImage.ExposureF(fname, bbox=_bbox, origin=afwImage.PARENT)
                sub = afwImage.MaskedImageF(newExp.getMaskedImage(), lsstGeom.Box2I(lsstGeom.Point2I(_bbox.getMinX()-xmin, _bbox.getMinY()-ymin), _bbox.getDimensions()), afwImage.PARENT)
                #sub <<= exposure.getMaskedImage()
                sub.assign(exposure.getMaskedImage())

            image_fname = '%s/%s_%s.fits' % (outdir, specObjID, band)
            newExp.setXY0(lsstGeom.Point2I(xmin,ymin))
            newExp.setWcs(wcs)
            newExp.writeFits(image_fname)

        except Exception as e:
            print(e.args)

if __name__ == '__main__':

    #butler = dafPersist.Butler('/gpfs02/HSC_DR/hsc_ssp/dr2/s18a/data/s18a_wide')
    butler = dafPersist.Butler('/gpfs02/HSC_DR/hsc_ssp/dr3/s20a/data/s20a_wide')
    skymap = butler.get('deepCoadd_skyMap')

    """
    src = butler.get('deepCoadd_meas', {'tract': 9813, 'patch': '4,4', 'filter': 'HSC-I'})
    print(src[0].getCoord())
    print(type(src[0].getCoord()))
    import sys
    sys.exit(1)
    """

    bands = ['HSC-G', 'HSC-R', 'HSC-I', 'HSC-Z', 'HSC-Y']
    '''
    f = open('HSCsdsshostless.dat')
    for line in f:
        if line[0] == '#':
            continue
        items = line.split()
        specObjID = items[0]
        ra  = float(items[1])
        dec = float(items[2])
        print(specObjID, ra, dec)
    '''

    specObjID='hello_world'
    ra = 16.6027
    dec = 0.549747
    cutoutImage(butler, skymap, specObjID, ra, dec, bands, 60,outdir='~/Projects/HSC/Cutouts')

#    f.close()

