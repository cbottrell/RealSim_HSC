import os,sys,time,tempfile
import illustris_python as il
from astropy.io import fits
from astropy.cosmology import Planck15 as cosmo
from astropy.convolution import convolve
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from scipy.interpolate import RectBivariateSpline
from hsc_utils import hsc_sql,hsc_image,hsc_psf
from rebin import rebin
import numpy as np
from glob import glob

rsdir = os.path.dirname(__file__)

def get_subhalos(basePath,snap,cosmology,mstar_lower=9,mstar_upper=np.inf):
    '''
    mstar_lower and mstar_upper have log10(Mstar/Msun) units and are physical (i.e. Msun, not Msun/h).
    '''
    little_h = cosmology.H0.value/100.
    
    ptNumStars = il.snapshot.partTypeNum('stars') 
    fields = ['SubhaloMassType','SubhaloFlag']
    subs = il.groupcat.loadSubhalos(basePath,snap,fields=fields)

    mstar = subs['SubhaloMassType'][:,ptNumStars]
    flags = subs['SubhaloFlag']
    subs = np.arange(subs['count'],dtype=int)
    
    # convert to units used by TNG (1e10 Msun/h)
    mstar_lower = 10**(mstar_lower)/1e10*little_h
    mstar_upper = 10**(mstar_upper)/1e10*little_h
    subs = subs[(flags!=0)*(mstar>=mstar_lower)*(mstar<=mstar_upper)]
    return subs,mstar[subs]

def hsc_calibrate_msb(filename,pixelscale=0.168,extension=1):
    '''Calibrate HSC cutout image to magnitude surface brightness [AB mag/arcsec2].'''
    data = fits.getdata(filename,extension)
    hdr = fits.getheader(filename,0)
    # f0 = hdr['FLUXMAG0']
    f0 = 10**(0.4*27)
    data = -2.5*np.log10(data/f0/pixelscale**2) # mag/arscec2
    data[np.isnan(data)]=99
    return data   

def hsc_calibrate_nanomaggies(filename,pixelscale=0.168,extension=1):
    '''Calibrate HSC cutout image to linear AB nanomaggies.'''
    data = fits.getdata(filename,extension)
    hdr = fits.getheader(filename,0)
    # f0 = hdr['FLUXMAG0']
    f0 = 10**(0.4*27)
    data = data/f0*1e9 # nanomaggies
    return data   

def hsc_calibrate_variance_nanomaggies(filename,pixelscale=0.168,extension=3):
    '''Calibrate HSC variance image to squared AB nanomaggies.'''
    variance = fits.getdata(filename,extension)
    hdr = fits.getheader(filename,0)
    # f0 = hdr['FLUXMAG0']
    f0 = 10**(0.4*27)
    variance = variance*(1e9/f0)**2 # nanomaggies
    return variance   

def coord_mask(filename,
               keys=['MP_BAD', 'MP_BRIGHT_OBJECT', 'MP_CLIPPED', 'MP_CR',
                     'MP_CROSSTALK', 'MP_DETECTED', 'MP_DETECTED_NEGATIVE',
                     'MP_EDGE', 'MP_INEXACT_PSF', 'MP_INTRP', 
                     'MP_NOT_DEBLENDED', 'MP_NO_DATA', 'MP_REJECTED', 
                     'MP_SAT', 'MP_SENSOR_EDGE', 'MP_SUSPECT', 
                     'MP_UNMASKEDNAN']):
    '''
    Coordinate mask for insertions. Assumes that the mask is the 2nd hdu object. Use all flags by default (i.e. only pixels with no flags are used for insertions). Bright object flags are very conservative and so are omitted. One should recall, however,that these flags will only necessarily apply to the pixel at the center of the insertion.
    '''
    mask_hdr = fits.getheader(filename,2)
    mask_dat = fits.getdata(filename,2).astype(np.uint32)
    mask_out = np.ones_like(mask_dat)
    
    for key in keys:
        mask_out[(mask_dat & 2**mask_hdr[key]).astype(bool)]=0
    return mask_out.astype(np.uint8)

def get_insertion_coords(mask,npixels_hsc):
    '''
    Use mask image to find injection coordinates.
    A border restriction is also used such that such that either
    the centroid is always placed inside a 10% buffer from the edges
    of the cutout. If this is smaller than the half-width of the 
    insertion image, the half-wdith is used instead.
    '''
    nrows,ncols = mask.shape
    hw = int(np.ceil(npixels_hsc/2))
    # pixels within 10% of image size from either side
    border_mask = np.zeros(mask.shape,dtype=np.uint8)
    row_min,row_max = max(hw,int(0.1*nrows)),min(nrows-hw,int(0.9*nrows))
    col_min,col_max = max(hw,int(0.1*ncols)),min(ncols-hw,int(0.9*ncols))
    border_mask[row_min:row_max,col_min:col_max]=1
    # map of possible injection sites
    mask *= border_mask
    # index of injection site for map
    index = np.random.choice(int(np.sum(mask)))
    # coordinates of injection site 
    return np.argwhere(mask)[index]

def psf_jitter(ra,dec,jitter_arcsec=3.):
    '''Introduce jitter in PSF reconstruction position to avoid perfect PSF reconstruction with respect to those used to create synthetic images.'''
    jitter_deg = jitter_arcsec / 3600.
    ra += np.random.randn()*jitter_deg
    dec+= np.random.randn()*jitter_deg
    return ra,dec

def realsim(il_path,img_path,out_path,src_cat,sim_tag,snap,sub,cam,
            dr='pdr3',rerun='pdr3_wide',filters='GRIZY',
            hsc_arcsec_per_pixel=0.168,photoz_tol=0.01,
            task_idx=0,seed=None,verbose=False):
    '''
    il_path is the path to the base illustris data (i.e. {sim_tag}/output)
    img_path is the path to the photometric images
    out_path is the path to the output HSC mocks
    sim_tag is the simulation tag (e.g. 'TNG50-1')
    snap is the simulations snapshot number
    sub is the subhalo id
    src_cat is the insertion statistics table (data).
    src_cat should be a formatted array, table or dataframe with columns:
        ['object_id', 'ra', 'dec', 'photoz_best']
    filters must be a capitalized string of chars for each filter.
    hsc_arscec_per_pixel is the pixel scale of the HSC-SSP ccds.
    photoz_tol is the match tolerance for the reference sample photoz matching
    '''
    seed = np.random.seed(seed)
    
    db_id = f'{sim_tag}_{snap:03}_{sub}_{cam}'
    
    filename = f'{img_path}/{snap:03}/shalo_{snap:03}-{sub}_{cam}_photo.fits'
    if not os.access(filename,0):
        print(f'File: {filename} not found. Quitting...')
        return
    
    outname = f'{out_path}/{snap:03}/shalo_{snap:03}-{sub}_{cam}_HSC_{filters}.fits'
    if os.access(outname,0):
        print(f'Output file: {outname} already exists. Quitting...')
        return
    
    if not os.access(f'{out_path}/{snap:03}',0):
        os.system(f'mkdir -p {out_path}/{snap:03}')
    
    # first get shared information
    filt0 = f'SUBARU_HSC.{filters[0]}'
    hdr = fits.getheader(filename,filt0)
    redshift = hdr['Redshift']
    npixels = hdr['NAXIS1']
    kpc_per_pixel = hdr['CDELT1']
    kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(z=redshift).value/60.
    arcsec_per_pixel = kpc_per_pixel/kpc_per_arcsec
    fov_kpc = kpc_per_pixel*npixels
    fov_arcsec = fov_kpc/kpc_per_arcsec
    npixels_hsc = int(np.floor(fov_arcsec/hsc_arcsec_per_pixel))
    pc_per_pixel_hsc = fov_kpc/npixels_hsc*1000.
    dlum = cosmo.luminosity_distance(z=redshift).value
    
    # random photoz matched sample from source catalogue
    zmatch_idxs = abs(src_cat['photoz_best']-redshift)<=photoz_tol
    sub_cat = src_cat.loc[zmatch_idxs]
    zmatch = sub_cat.sample(1,replace=False)
    object_id = zmatch.object_id.values[0]
    ra = zmatch.ra.values[0]
    dec = zmatch.dec.values[0]
    tract = zmatch.tract.values[0]
    patch = f'{zmatch.patch.values[0]:03}'
    # convert integer patch to hsc format
    patch = f'{patch[0]},{patch[-1]}'
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        if verbose: 
            start = time.time()
            print(f'Getting full-patch cutouts from data archive server...')
            
        for filt in filters:
            while True:
                if os.system(f'wget -O {db_id}-Cutout-HSC-{filt}.fits --user {os.environ["HSC_SSP_CAS_USERNAME"]} --password {os.environ["HSC_SSP_CAS_PASSWORD"]} https://hsc-release.mtk.nao.ac.jp/archive/filetree/{rerun}/deepCoadd-results/HSC-{filt}/{tract}/{patch}/calexp-HSC-{filt}-{tract}-{patch}.fits')==0:
                    break
                else:
                    time.sleep(120)
                    continue

        if verbose: 
            print(f'Finished in {time.time()-start} seconds.')

        cutout_names = [f"{db_id}-Cutout-HSC-{filt}.fits" for filt in filters]

        # # use i-band bit-mask for coordinate insertion, if available
        if 'I' in filters:
            cutout_name_mask = f"{db_id}-Cutout-HSC-I.fits"
        else:
            cutout_name_mask = cutout_names[0]
            
        mask_keys=['MP_BAD', 'MP_BRIGHT_OBJECT',
                   'MP_CLIPPED', 'MP_CR', 'MP_CROSSTALK',
                   'MP_DETECTED', 'MP_DETECTED_NEGATIVE', 'MP_EDGE',
                   'MP_INEXACT_PSF', 'MP_INTRP', 'MP_NO_DATA', 'MP_REJECTED',
                   'MP_SAT', 'MP_SENSOR_EDGE', 'MP_SUSPECT', 'MP_UNMASKEDNAN']
        
        mask = coord_mask(cutout_name_mask,keys=mask_keys)
        row,col = get_insertion_coords(mask,npixels_hsc)
        row_min = int(row-npixels_hsc/2)
        row_max = row_min+npixels_hsc
        col_min = int(col-npixels_hsc/2)
        col_max = col_min+npixels_hsc
        # get ra,dec of coords for headers and PSF
        wcs = WCS(fits.getheader(cutout_name_mask,1))
        ra,dec = wcs.all_pix2world(col,row,1,ra_dec_order=True)

        if verbose: 
            start = time.time()
            print(f'Getting PSF reconstructions from HSC-SSP data archive server...')
            
        # get psf images for target ra,dec location
        while True:
            try:
                os.system(f'python {rsdir}/hsc_dat/{dr}/downloadPsf/downloadPsf.py --rerun={rerun} --ra={ra} --dec={dec} --centered=false --name="{db_id}-PSF-{{filter}}" --user={os.environ["HSC_SSP_CAS_USERNAME"]}')
                break
            except:
                time.sleep(10)
                pass
        psf_names = [f'{db_id}-PSF-HSC-{filt}.fits' for filt in filters]
            
        # get psfs with 3 arcsec offsets for morphological analyses
        jra,jdec = psf_jitter(ra,dec)
        while True:
            try:
                os.system(f'python {rsdir}/hsc_dat/{dr}/downloadPsf/downloadPsf.py --rerun={rerun} --ra={ra} --dec={dec} --centered=false --name="{db_id}-JitterPSF-{{filter}}" --user={os.environ["HSC_SSP_CAS_USERNAME"]}')
                break
            except:
                time.sleep(10)
                pass
        jpsf_names = [f'{db_id}-JitterPSF-HSC-{filt}.fits' for filt in filters]
        
        if verbose: print(f'Finished in {time.time()-start} seconds.')

        hdu_list = fits.HDUList()
        for i,(filt,psf_name,jpsf_name,cutout_name) in enumerate(zip(filters, psf_names, jpsf_names, cutout_names)):
            if verbose: 
                start = time.time()
                print(f'Processing {filt}-band HDU...')
            # image data and header information
            data_img = fits.getdata(filename,f'SUBARU_HSC.{filt}')
            hdr = fits.getheader(filename,f'SUBARU_HSC.{filt}')

            # convert data to nanomaggies from AB surface brightness
            data_img =  10**(0.4*(22.5-data_img))*arcsec_per_pixel**2 
            apmag = 22.5-2.5*np.log10(np.nansum(data_img)) # for header
            # rebin to hsc ccd scale
            if npixels < npixels_hsc:
                interp = RectBivariateSpline(
                    np.linspace(-1,1,npixels),
                    np.linspace(-1,1,npixels),
                    data_img,kx=1,ky=1)
                data_img = interp(
                    np.linspace(-1,1,npixels_hsc),
                    np.linspace(-1,1,npixels_hsc))*(npixels/npixels_hsc)**2
            else:
                data_img = rebin(data_img,(npixels_hsc,npixels_hsc))

            # convolve with PSF
            psf = fits.getdata(psf_name)
            data_img = convolve(data_img,psf)

            # generate science, mask, and variance cutouts
            data_img += hsc_calibrate_nanomaggies(cutout_name)[row_min:row_max,col_min:col_max]
            data_msk = fits.getdata(cutout_name,2)[row_min:row_max,col_min:col_max]
            data_var = hsc_calibrate_variance_nanomaggies(cutout_name)
            data_var = data_var[row_min:row_max,col_min:col_max]
            cutout_hdr = fits.getheader(cutout_name,0)

            hdu = fits.ImageHDU(data_img,name=f'SUBARU_HSC.{filt}')
            hdr['BUNIT'] = ('AB nanomaggies, m_AB = 22.5-2.5log10(f)','Image units')
            hdr['CRPIX1'] = (int(npixels_hsc/2), 'X-axis coordinate system reference pixel')
            hdr['CRVAL1'] = (0., 'Coordinate system value at X-axis reference pix')
            hdr['CDELT1'] = (pc_per_pixel_hsc, 'Coordinate increment along X-axis')
            hdr['CTYPE1'] = ('pc', 'Physical units of the X-axis increment')
            hdr['CRPIX2'] = (int(npixels_hsc/2), 'Y-axis coordinate system reference pixel')
            hdr['CRVAL2'] = (0., 'Coordinate system value at Y-axis reference pix')
            hdr['CDELT2'] = (pc_per_pixel_hsc, 'Coordinate increment along Y-axis')
            hdr['CTYPE2'] = ('pc', 'Physical units of the Y-axis increment')
            hdr['COSMO'] = (cosmo.name,'Cosmology')
            hdr['FOVKPC'] = (fov_kpc,'Field of view (kpc)')
            hdr['FOVARC'] = (fov_arcsec,'Field of view (arcsec)')
            hdr['KPCARC'] = (kpc_per_arcsec,'Angular scale (kpc/arcsec)')
            hdr['DLUM'] = (dlum,'Luminosity distance (Mpc)')
            hdr['APMAG'] = (apmag,f'Apparent {filt}-band AB magnitude')
            hdr['DR'] = (dr,'HSC-SSP Data Release')
            hdr['RERUN'] = (rerun,'HSC-SSP Rerun')
            hdr['RA'] = (float(ra),
                         'HSC-SSP insertion right ascension (centroid)')
            hdr['DEC'] = (float(dec),
                          'HSC-SSP insertion declination (centroid)')
            hdr['BGMEAN'] = (cutout_hdr['BGMEAN'],
                             'HSC-SSP background mean [original]')
            hdr['BGVAR'] = (cutout_hdr['BGVAR'],
                            'HSC-SSP background variance [original]')
            hdr['VARSC'] = (cutout_hdr['variance_scale'],'HSC-SSP variance scale [original]')
            hdu.header = hdr
            hdu_list.append(hdu)
            hdu = fits.ImageHDU(data_msk,name=f'SUBARU_HSC.{filt} MASK')
            hdu_list.append(hdu)
            hdu = fits.ImageHDU(data_var,name=f'SUBARU_HSC.{filt} VARIANCE')
            hdu_list.append(hdu)
            hdu_jpsf = fits.open(jpsf_name)[0]
            hdu_jpsf.header['JRA']=(float(jra),'Jittered PSF RA')
            hdu_jpsf.header['JDEC']=(float(jdec),'Jittered PSF DEC')
            hdu_jpsf.name = f'SUBARU_HSC.{filt} PSF'
            hdu_list.append(hdu_jpsf)
            if verbose: 
                print(f'Finished in {time.time()-start} seconds.')
        
        if verbose: print(f'Writing results to {outname} ...')
        hdu_list.writeto(outname,overwrite=True)
        if verbose: print(f'Done.')

def main():
    
    sim_tag = os.getenv('SIM')
    njobs = int(os.getenv('JOB_ARRAY_NJOBS'))
    job_idx = int(os.getenv('JOB_ARRAY_INDEX'))
    
    snap_min,snap_max = 72,91
    snaps = np.arange(snap_min,snap_max)[::-1]
    cams = ['v0','v1','v2','v3']
    dr = 'pdr3'
    rerun = 'pdr3_wide'
    
    if 'pdr' in dr:
        # export public dr keys for data server access
        os.environ['HSC_SSP_CAS_PASSWORD'] = os.environ['SSP_PDR_PWD']
        os.environ['HSC_SSP_CAS_USERNAME'] = os.environ['SSP_PDR_USR']
    else:
        # export internal dr keys for data server access
        os.environ['HSC_SSP_CAS_PASSWORD'] = os.environ['SSP_IDR_PWD']
        os.environ['HSC_SSP_CAS_USERNAME'] = os.environ['SSP_IDR_USR']
        
    project_path = f'/vera/ptmp/gc/bconn/SKIRT/IllustrisTNG/{sim_tag}/HSCSSP'
    # base path where TNG data is stored
    il_path = f'/virgotng/universe/IllustrisTNG/{sim_tag}/output'
    # photometry path
    img_path = f'{project_path}/Idealized'
    # output path
    out_path = f'{project_path}/{rerun}'
    
    csv_name = 'Catalogues/HSC-dr3-s20a_wide-photoz_mizuki-CLEAN.csv'
    csv_name = f'{rsdir}/{csv_name}'
    src_cat = hsc_sql.load_sql_df(csv_name)
    print('Loaded source catalogue.')
    
    for snap in snaps:
        subs,mstar = get_subhalos(il_path,snap,cosmology=cosmo,
                                  mstar_lower=10)
        for sub in subs[job_idx::njobs]:
            for cam in cams:
                print(f'Running HSC RealSim for SIMTAG:{sim_tag}, SNAP:{snap}, SUB:{sub}, CAM:{cam}...')
                start = time.time()
                try:
                    realsim(il_path,img_path,out_path,src_cat,
                            sim_tag,snap,sub,cam,verbose=True,
                            dr=dr,rerun=rerun)
                except:
                    print(f'Failed for SIMTAG:{sim_tag}, SNAP:{snap}, SUB:{sub}, CAM:{cam}.')

                print(f'Fulltime: {time.time()-start} seconds.\n')
    
if __name__=='__main__':
    
    main()
