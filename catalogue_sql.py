import hsc_sql
import pandas as pd

def main():

    dr='dr3'
    rerun = 's20a_wide'
    csv_name = f'Catalogues/HSC-{dr}-{rerun}-photoz_mizuki.csv'
    clean = True
    
    sql_cmd = [
        f'SELECT f.object_id, f.ra, f.dec, f.tract, f.patch,',
        f'z.photoz_best, z.photoz_risk_best, z.reduced_chisq,',
        f'z.prob_gal, z.prob_star, z.prob_qso',
        f'FROM {rerun}.forced as f',
        f'LEFT JOIN {rerun}.photoz_mizuki as z',
        f'ON f.object_id = z.object_id',
        f'WHERE z.photoz_median>0.01 and z.photoz_median<2',
              ]
    sql_cmd = ' '.join(sql_cmd)
    if clean:
        sql_cmd += ' '+' '.join([f'AND {cnd}' for cnd in hsc_sql.hsc_coadd_cnds(rerun)])
    sql_cmd += f' ORDER BY f.object_id;'
    
    df = hsc_sql.load_sql_df(csv_name,dr=dr,sql_cmd=sql_cmd)
    
if __name__=='__main__':
    main()
