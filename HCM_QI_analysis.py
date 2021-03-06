# Quntitative DHI analysis
# written by Y.Fan May/2018
# Last update on Aug 6/2018
'''
This is a module to analyze the seismic DHI features. There are five main features measured in this module: up down ratio, 
AVO, contact sharpness, contact comformability to structure and pachiness of DHI. These features defines main characters of 
the hydrocarbon related DHI in Brunei. For a new basin or field, depending on the rock and fluid properties, different 
features of DHI may be more suitable to differenciate hydrocarbons. The selection of DHI features can be adviced by the 
corresponding QI team. Besides the five main features, we also measure fluid contacts (CTD) and potential hydrocarbon volume.
v1: add regridding for depth map and full amplitude as sometimes they are not exactly on the same grid. 
v2: add QC plots 
v3: add subsands to calculate volume. e.g. add MBA1, MBA2, MBA3,MBA4 to the main sand MBA
'''

import time
import pandas as pd
import numpy as np
import math
from scipy.interpolate import griddata
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab
import os

#-----------------------------------------------------------------------------
# 
#-----------------------------------------------------------------------------
def DHI_analysis_v3(f_dep,f_amp,f_amp_far,f_amp_near,polygon_file,fajar_property_database,well_top,output_dir):
    '''
    Parameters
    ----------
    f_dep: file name of the depth map in ascii format. e.g. exported ascii map from NDI.
    f_amp: file name of the full amplitude in ascii format. Same as above. Files has three columns x,y,z(or amp)
    f_amp_far: file name of the far amplitude in ascii format.
    f_amp_near: file name of the near amplitude in ascii format.
    polygon_file: file of the polygon file in ascii format that is exported from NDI. The polygon file can have multiple 
              polygons from the same corresponding map  
    fajar_property_database: file of the FAJAR property database
    well_top: well top marker of the corresponding map. This marker is used to extract property from FAJAR database. 
              e.g. 'CM1', 'K10.10'
    output_dir: directory to store the QC plots
    '''
    # define output directory and color map to use
    # choose a colormap from NDI
    path = 'C:/Apps/Ddrive_BSP_Projects/NDI_cmap/'
    cmap_file = path+'Jason_AI_Map_254.cmap'
    outpath = output_dir+'QI_analysis_QC_plots/'
    #outpath = 'output/QI_analysis_QC_plots'  #define output folder for QC plots
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    cmap,cmap_r=import_ndi_cm(cmap_file)
    #################################################################################
    ''''## step1: load data'''
    T0 = time.time()  # start clock to calculate program running time
    print('######################################################################')
    print('Step1: loading depth and amplitude maps ...')
#    print('loading depth and amplitude maps ...')
    xd,yd,dep = np.loadtxt(f_dep, skiprows=1, unpack=True) # load depth file
#    print('depth map loaded. loading amplitude full ...')
    xa,ya,ampft = np.loadtxt(f_amp, skiprows=1, unpack=True) # load amplitude file
    amp = abs(ampft)        # when amp extracted from tough, they are negative 
#    print('amplitude full loaded. loading amplitude far ...')
    xf,yf,ampf = np.loadtxt(f_amp_far, skiprows=1, unpack=True) # load amplitude file
#    print('amplitude far loaded. loading amplitude near ...')
    xn,yn,ampn = np.loadtxt(f_amp_near, skiprows=1, unpack=True) # load amplitude file
#    print('all maps loaded. loading FAJAR database...')
    fajar = pd.read_csv(fajar_property_database, header=0)
    top=fajar.Pick_Name==well_top[0]  # pick the welltop to work on 
    #tops = (fajar.Pick_Name==well_top[0])|(fajar.Pick_Name==well_top[1])|(fajar.Pick_Name==well_top[2])
    fajar1 = fajar[top].reset_index(drop=True)
    #fajar1_mulSand = fajar[tops].reset_index(drop=True)
    w_x = fajar1['Easting']; w_y = fajar1['Northing'];w_name=fajar1['Wellbore_Name']
    sandlist=list(fajar.Pick_Name)
#    print('FAJAR database loaded.')
    dt1 = time.time()-T0; 
#    print('total data loading time',round(dt1),'s.')
    
    ## load and format polygons
    t0=time.time()
    poly_all=[]
    df_blk = pd.read_table(polygon_file,delim_whitespace=True,header=None) # read polygon files
    df_blk.columns = ['x', 'y','z','f1','f2','f3','poly_name']
    plist = df_blk.poly_name.unique()  # polygon name list

    for pdx in range(0,len(plist)):
        df_blk0 = df_blk[df_blk.poly_name==plist[pdx]]
        df_blk0.index = pd.RangeIndex(len(df_blk0.index))
        poly_blk = make_poly1(df_blk0)
        poly_all.append(poly_blk)
    dt2 = time.time()-t0; 
    #################################################################################
    '''## Step2: Dividing the maps into each block polygons'''
    print('######################################################################')
    print('Step2: Dividing the maps into each block polygons...')
    amp_blk_all = [];dep_blk_all = [];x_blk_all = [];y_blk_all=[];
    ampf_blk_all = [];xf_blk_all=[];yf_blk_all=[];
    ampn_blk_all = [];xn_blk_all=[];yn_blk_all=[];

    ## cut the amplitude with blk polygon
    time0=time.time()
    for pdx in range(0,len(poly_all)):
        poly = poly_all[pdx]
        amp_blk,xa_blk,ya_blk=cut_map_poly1(amp,xa,ya,poly)
        dep_blk,xd_blk,yd_blk=cut_map_poly1(dep,xd,yd,poly)
        ## regrid the depth and full amplitude map because the input maps may not be exactly on the same grid
        ## This is critical for the CTD stacking
        points_a = np.asarray(list(map(lambda x,y:[x,y],xa_blk,ya_blk)))
        points_d = np.asarray(list(map(lambda x,y:[x,y],xd_blk,yd_blk)))
        sampling = 15 # spatial sampling in xy for new grid 
        nx = int((max(xa_blk)-min(xa_blk))/sampling)
        ny = int((max(ya_blk)-min(ya_blk))/sampling)
        grid_x, grid_y = np.mgrid[min(xa_blk):max(xa_blk):nx*1j, min(ya_blk):max(ya_blk):ny*1j] # new grid
        grid_a = griddata(points_a, np.asarray(amp_blk), (grid_x, grid_y), method='cubic')  # interp amp to new grid
        grid_d = griddata(points_d, np.asarray(dep_blk), (grid_x, grid_y), method='cubic')  # interp dep to new grid
        amp_blk,x_blk,y_blk = cut_map_poly1_grid(grid_a,grid_x, grid_y,poly)   # cut the regrided amp to AOI
        dep_blk,x_blk,y_blk = cut_map_poly1_grid(grid_d,grid_x, grid_y,poly)   # cut the regrided dep to AOI
        amp_blk_all.append(amp_blk);dep_blk_all.append(dep_blk);
        x_blk_all.append(x_blk);y_blk_all.append(y_blk);
    time1 = time.time()
    dt31 = round(time1-time0,1)
    print('full stack map dividing finished. it takes',dt31,'s.')

    # far
    for pdx in range(0,len(poly_all)):
        poly = poly_all[pdx]
        ampf_blk,xf_blk,yf_blk=cut_map_poly1(ampf,xf,yf,poly)
        ampf_blk_all.append(ampf_blk);
        xf_blk_all.append(xf_blk);yf_blk_all.append(yf_blk);
    time2 = time.time()
    dt32 = round(time2-time1,1)
    print('far stack map dividing finished. it takes',dt32,'s.')

    # near
    for pdx in range(0,len(poly_all)):
        poly = poly_all[pdx]
        ampn_blk,xn_blk,yn_blk=cut_map_poly1(ampn,xn,yn,poly)
        ampn_blk_all.append(ampn_blk);
        xn_blk_all.append(xn_blk);yn_blk_all.append(yn_blk);
    time3 = time.time()
    dt33 = round(time3-time2,1)
    print('near stack map dividing finished. it takes',dt33,'s.')    

    #################################################################################
    '''## Step3: DHI analysis in each block.'''
    print('######################################################################')
    print('Step3:  DHI analysis in each block...')
    dataout = []; 
    ## loop through all the polygons
    for pdx in range(0,len(plist)):
        t0 = time.time() # loop starting time
        print(plist[pdx])
        """
        choose the block polygon
        """
        poly_blk = poly_all[pdx]
        amp_blk = np.asarray(amp_blk_all[pdx]);dep_blk = np.asarray(dep_blk_all[pdx]);
        x_blk = np.asarray(x_blk_all[pdx]);y_blk=np.asarray(y_blk_all[pdx]);
        ampf_blk = np.asarray(ampf_blk_all[pdx]);
        xf_blk = np.asarray(xf_blk_all[pdx]);yf_blk=np.asarray(yf_blk_all[pdx]);    
        ampn_blk = np.asarray(ampn_blk_all[pdx]);
        xn_blk = np.asarray(xn_blk_all[pdx]);yn_blk=np.asarray(yn_blk_all[pdx]);
        """
        CTD stacking in polygon
        """
        # convert grid data dep and amp to 1d array
        dep_blk_grid = dep_blk.copy();x_blk_grid=x_blk.copy();y_blk_grid=y_blk.copy();
        dep_blk1 = dep_blk.flatten(); amp_blk1 = amp_blk.flatten()
        x_blk1 = x_blk.flatten(); y_blk1 = y_blk.flatten()
        np.warnings.filterwarnings('ignore') #turn off the warning for comparing nan value
        dep_blk = dep_blk1[(abs(amp_blk1)>0)&(dep_blk1>0)];amp_blk = amp_blk1[(abs(amp_blk1)>0)&(dep_blk1>0)]  # to remove nan values
        x_blk = x_blk1[(abs(amp_blk1)>0)&(dep_blk1>0)];y_blk = y_blk1[(abs(amp_blk1)>0)&(dep_blk1>0)]

        win_half=5; #half window size in meter for moving average
        # define a regular grid to do moving average
        dep_reg_blk = np.linspace(np.min(dep_blk), np.max(dep_blk), num=np.int(np.max(dep_blk)-np.min(dep_blk))) # basically every meter.
        amp_reg_blk = running_mean(amp_blk,dep_blk,dep_reg_blk,win_half)
        
        ## calculate the derivative of the CTD: the slope of the amplitude change
        dx = dep_reg_blk[1]-dep_reg_blk[0]
        dydx = np.gradient(amp_reg_blk, dx)
        dydx_avg = running_mean(dydx,dep_reg_blk,dep_reg_blk,win_half)
        
        # searching for contact. this is the part that can be optimized in the future 
        # the contact picking is the most critical part of the program as it affects all the other measured parameters.
        # what are the criteria that determins a good contact?
        win_c = 100;  # the half window to search for contact
        d_max1 = dep_reg_blk[amp_reg_blk==np.nanmax(amp_reg_blk[0:int(0.8*len(amp_reg_blk))])]  # depth of the brightest amplitude
        d_max = d_max1[-1]  # where there are more than 1 d_max
        d_c = dep_reg_blk[dydx_avg==np.nanmin(dydx_avg[(dep_reg_blk>max(d_max-win_c/8,min(dep_reg_blk)))&(dep_reg_blk<min(d_max+win_c,max(dep_reg_blk)))])]
        #d_c = [d_c[0]]
        """
        updip downdip
        """
        ## create up down polygon 
        # when contact is too close to top or bottom; below is a temporary fix
        cgap = 5
        if (abs(d_c-np.min(dep_blk))<cgap)|(abs(d_c-np.max(dep_blk))<cgap):
            poly_up = poly_blk
            poly_down = poly_blk
            buffer=0
            amp_ud = amp_blk; dep_ud = dep_blk; 
            amp_dd = amp_blk; dep_dd = dep_blk;
        else:
            poly_up,poly_down,buffer = cut_poly(x_blk,y_blk,dep_blk,poly_blk,d_c)
            amp_ud = amp_blk[dep_blk<d_c]; dep_ud = dep_blk[dep_blk<d_c]; 
            amp_dd = amp_blk[dep_blk>d_c+buffer]; dep_dd = dep_blk[dep_blk>d_c+buffer];
        # add a 'best fit' normal distribution line to the histogram
        mu_ud = np.mean(amp_ud); sd_ud=np.std(amp_ud);
        mu_dd = np.mean(amp_dd); sd_dd=np.std(amp_dd);
        ud_ratio = mu_ud/mu_dd

        """
        contact sharpness
        """
        idx_win = 10; # the half window around contact to measure sharpness 
        idx_c= np.where(dep_reg_blk==d_c)
        # sharpness1 is the percentage change in amplitude around contact +-idx_win. 
        sharpness1 = (amp_reg_blk[max(idx_c[0]-idx_win,0)]-amp_reg_blk[min(idx_c[0]+idx_win,len(amp_reg_blk)-1)])/amp_reg_blk[idx_c[0]]
        sharpness = sharpness1[0]

        """
        contact comformability
        """
        # can be optimized or redefined
        win_c = 5  # half windown 5m around contact
        amp_c = amp_blk[(dep_blk<d_c+win_c)&(dep_blk>d_c-win_c)]  # find amplitude dc within the window
        mu_amp_c = np.mean(amp_c); sd_amp_c=np.std(amp_c);
        # normalize standard deviation by the amplitude itself
        sd_amp_c_norm = round(sd_amp_c/mu_amp_c,3);

        """
        AVO: far near ratio
        """
        ## Updip
        ampf_ud = [];xf_ud=[];yf_ud=[];
        ampn_ud = [];xn_ud=[];yn_ud=[];
        ampf_dd = [];xf_dd=[];yf_dd=[];
        ampn_dd = [];xn_dd=[];yn_dd=[];

        # far stack   
        ampf_ud,xf_ud,yf_ud = cut_map_poly1(ampf_blk,xf_blk,yf_blk,poly_up)
        ampf_dd,xf_dd,yf_dd = cut_map_poly1(ampf_blk,xf_blk,yf_blk,poly_down)

        # near stack
        ampn_ud,xn_ud,yn_ud = cut_map_poly1(ampn_blk,xn_blk,yn_blk,poly_up)
        ampn_dd,xn_dd,yn_dd = cut_map_poly1(ampn_blk,xn_blk,yn_blk,poly_down)

        # up down ratio (far and near) and AVO 
        udf_ratio = np.mean(ampf_ud)/np.mean(ampf_dd);
        udn_ratio = np.mean(ampn_ud)/np.mean(ampn_dd);
        fn_ratio_up = np.mean(ampf_ud)/np.mean(ampn_ud);
        fn_ratio_down = np.mean(ampf_dd)/np.mean(ampn_dd);

        """
        calculate the size of updip area
        """
        area_up = round(PolygonArea(poly_up)/10**6,3)  # in km square
        area_blk = round(PolygonArea(poly_blk)/10**6,3)  # in km square
        # dip of structure
        win_c1 = 3  # half windown around contact
        x_blk_c = x_blk[(dep_blk<d_c+win_c1)&(dep_blk>d_c-win_c1)]  # find area around contact
        y_blk_c = y_blk[(dep_blk<d_c+win_c1)&(dep_blk>d_c-win_c1)]
        dep1= np.sort(dep_blk[dep_blk<d_c])
        d_mid = dep1[int(len(dep1)/2)]   # mid point from contact to the crest

        x0 = x_blk[dep_blk==d_mid];  # find the point at the mid way from contact to the crest  
        y0 = y_blk[dep_blk==d_mid]; 
        x0 = x0;y0=y0;
        #x0 = x0[0];y0=y0[0];
        dist0 = np.sqrt((x0-x_blk_c)**2+(y0-y_blk_c)**2)   # lateral distance between the mid point to contact
        theta = np.arctan( (d_c-d_mid)/np.min(dist0) )  # the dip angle
        theta1 = round(57.2958*theta[0],1)
        # here one may define more points to calculate the dip variation of the structure
        # or define a depth area function to estimate volume more accurately
        area_up1 = round(area_up/np.cos(theta[0]),2)  # correct area size in 3D

        """
        Estimate the volume using the information from FAJAR database 
        """
        d_net,por2,sh2,Call,color_top,well_used,wx_blk,wy_blk, = prop_from_fajar(well_top,fajar,poly_blk,poly_up)
        HC_volume = area_up1*d_net*por2*sh2*10**6   # in cubic meters
        
        """
        ## Step4: Return output dataframe 
        """
        data = []
        data = [{'a': plist[pdx],'a1': well_top, 'b': round(ud_ratio,2), 'c': round(fn_ratio_up,2), 'd':round(sharpness,2),'e':round(sd_ud/mu_ud,2),'f':sd_amp_c_norm,'g':round(d_c[0])-10,'h':area_up1,'h1':d_net,'h2':por2,'h3':sh2,'i':HC_volume,'j':Call,'k':well_used}]
        dataout = dataout+data
        df = pd.DataFrame(dataout)
        df.columns=['block_name','sand','up_down_ratio','farOVnear_updip','contactSharp_normalized','amp_up_Coherence_normalized','comformability','contact_CTD','area_updip','thickness','POR2','SH2','HC_Volume_estimate','Call','e.g. well in block']                          

        '''
        QC plots
        '''    
        #QC: plot the data to see if the polygons and CTD analysis make sense. 
        bf= 500  #AOI outside of polygon area
        xmin = np.min(x_blk)-bf;xmax = np.max(x_blk)+bf
        ymin = np.min(y_blk)-bf;ymax = np.max(y_blk)+bf
        sqcut = (xa<xmax)&(xa>xmin)&(ya<ymax)&(ya>ymin)
        xa_blk0 = xa[sqcut];ya_blk0=ya[sqcut];amp_blk0 = amp[sqcut];
        amin = np.min(amp_blk0); amp_blk0_sort = np.sort(amp_blk0);
        pc = 99.5/100 # percentage of data for color scale
        amax = amp_blk0_sort[int(len(amp_blk0)*pc)]#np.max(amp_blk0_sort);
        
        fig1=plt.figure(figsize=(16,16/1.41))   #A4 ratio 1.41
        # amplitude Full
        plt.subplot(221)
        cont_space = 20 # contour spacing in meter
        cont_num = int((max(dep_reg_blk)-min(dep_reg_blk))/cont_space)
#        plt.scatter(xa_blk0, ya_blk0, s=50, c=amp_blk0, marker=".",cmap = cmap_r);plt.colorbar()
        plt.scatter(xa_blk0, ya_blk0, s=50, c=amp_blk0, marker=".",cmap = cmap_r,vmin=amin,vmax=amax);plt.colorbar()        
        S1 = plt.contour(x_blk_grid, y_blk_grid,dep_blk_grid,cont_num,colors='w')
        plt.clabel(S1,inline=1,inline_spacing=0,fontsize=10,fmt='%1.0f',colors='b')
        #plt.scatter(x_blk, y_blk, s=50, c=amp_blk, marker=".",cmap = cmap_r);plt.colorbar()
        plt.axis('equal')
        plt.xlim((xmin,xmax));plt.ylim((ymin,ymax))
        plt.plot(np.asarray(poly_blk)[:,0],np.asarray(poly_blk)[:,1], linestyle='--', color='w', linewidth=4)
        plt.plot(np.asarray(poly_up)[:,0],np.asarray(poly_up)[:,1], linestyle='--', color='r', linewidth=2)
        plt.plot(np.asarray(poly_down)[:,0],np.asarray(poly_down)[:,1], linestyle='--', color='k', linewidth=2)
        plt.scatter(np.asarray(poly_down)[:,0],np.asarray(poly_down)[:,1], s=10, marker=".",color='b')
        #print(color_top)
        plt.scatter(wx_blk,wy_blk,c=color_top,s=50,marker='o',cmap='bwr',vmin=0, vmax=2)
        plt.title('Amplitude',color='r',bbox=dict(facecolor='orange',ec='orange', alpha=0.5),fontdict=dict(fontsize=16),loc='left')
        plt.xlabel('X (m)');plt.ylabel('Y (m)');

        # CTD
        plt.subplot(243)
        plt.scatter(amp_blk, dep_blk, s=40, c=amp_blk, marker=".",vmin=amin,vmax=amax,cmap = cmap_r)
        plt.plot([np.min(amp_blk),np.max(amp_blk)],[d_c, d_c],'r--')
        plt.plot([np.min(amp_blk),np.max(amp_blk)],[d_c-idx_win, d_c-idx_win],'g--')
        plt.plot([np.min(amp_blk),np.max(amp_blk)],[d_c+idx_win, d_c+idx_win],'g--')
        plt.plot(amp_reg_blk, dep_reg_blk, 'k-',linewidth=5)
        #plt.ylim(np.min(dep_blk),np.max(dep_blk)-500)
        plt.gca().invert_yaxis()
        plt.ylabel('Depth (m)',fontsize=16)
        plt.title('Amp vs Depth and CTD',color='r',bbox=dict(facecolor='orange',ec='orange', alpha=0.5),fontdict=dict(fontsize=16),loc='left')
        #plt.xlim(np.min(x31),np.max(x31))
        
        plt.subplot(244)
        plt.plot(dydx_avg,dep_reg_blk,'k-',linewidth=3)
        plt.plot([np.min(dydx_avg),np.max(dydx_avg)],[d_c, d_c],'r-')
        plt.plot([np.min(dydx_avg),np.max(dydx_avg)],[d_c-idx_win, d_c-idx_win],'g--')
        plt.plot([np.min(dydx_avg),np.max(dydx_avg)],[d_c+idx_win, d_c+idx_win],'g--')
        plt.gca().invert_yaxis()
        plt.title('Gradient of CTD',color='r',bbox=dict(facecolor='orange',ec='orange', alpha=0.5),fontdict=dict(fontsize=16),loc='left')

        # compare updip and downdip
        plt.subplot(223)
        mu_ud = np.mean(amp_ud); sd_ud=np.std(amp_ud);
        fit_ud = mlab.normpdf(sorted(amp_ud), mu_ud, sd_ud)
        mu_dd = np.mean(amp_dd); sd_dd=np.std(amp_dd);
        fit_dd = mlab.normpdf(sorted(amp_dd), mu_dd, sd_dd)
        #plt.figure(figsize=(16,6))
        plt.plot(sorted(amp_ud), fit_ud, 'r--',linewidth=3)
        plt.plot(sorted(amp_dd), fit_dd, 'b--',linewidth=3)
        plt.legend(['Updip','Downdip'],loc='upper right',fontsize=16)
        plt.hist(amp_ud, bins=100,normed=1, facecolor='red', alpha=0.5)
        plt.hist(amp_dd, bins=100,normed=1, facecolor='blue', alpha=0.5)
        plt.title('Updip-Downdip ratio',color='r',bbox=dict(facecolor='orange',ec='orange', alpha=0.5),fontdict=dict(fontsize=16),loc='left')

        ## compare far and near
        plt.subplot(224)
        mu_udf = np.mean(ampf_ud); sd_udf=np.std(ampf_ud);
        fit_udf = mlab.normpdf(sorted(ampf_ud), mu_udf, sd_udf)
        mu_udn = np.mean(ampn_ud); sd_udn=np.std(ampn_ud);
        fit_udn = mlab.normpdf(sorted(ampn_ud), mu_udn, sd_udn)
        #plt.figure(figsize=(16,6))
        plt.plot(sorted(ampf_ud), fit_udf, 'r--',linewidth=3)
        plt.plot(sorted(ampn_ud), fit_udn, 'b--',linewidth=3)
        plt.legend(['Far updip','Near updip'],loc='upper right',fontsize=16)
        plt.hist(ampf_ud, bins=100,normed=1, facecolor='red', alpha=0.5)
        plt.hist(ampn_ud, bins=100,normed=1, facecolor='blue', alpha=0.5)
        plt.title('AVO',color='r',bbox=dict(facecolor='orange',ec='orange', alpha=0.5),fontdict=dict(fontsize=16),loc='left')
        #plt.show()

        figname = outpath+'/'+plist[pdx]+"_QC.png"   #save the figure
        fig1.savefig(figname, bbox_inches='tight')
        plt.close()
        dt = round(time.time()-t0,3)
        print(plist[pdx],':',len(amp_blk), 'data points; running time',dt,'s')
    #################################################################################
    # amplitude plot of the whole area
    fig1=plt.figure(figsize=(16,16/1.41))   #A4 ratio 1.41
    plt.scatter(xa, ya, s=50, c=amp, marker=".",cmap = cmap_r,vmin=0,vmax=np.mean(amp)+4*np.std(amp));plt.colorbar()        
    plt.axis('equal')
    for pdx in range(0,len(plist)):
        poly_blk = poly_all[pdx]
        plt.plot(np.asarray(poly_blk)[:,0],np.asarray(poly_blk)[:,1], linestyle='--', color='w', linewidth=2)
    plt.scatter(w_x,w_y,s=50,marker='o',color='k')
    plt.title('Amplitude',color='r',bbox=dict(facecolor='orange',ec='orange', alpha=0.5),fontdict=dict(fontsize=16),loc='left')
    plt.xlabel('X (m)');plt.ylabel('Y (m)');
    figname = outpath+'/'+well_top+'_amp_summary.png'   #save the figure
    fig1.savefig(figname, bbox_inches='tight')
    plt.close()
    #################################################################################
    dt = time.time()-T0; 
    print('total running time',round(dt),'s.')
    print('######################################################################')
    return df 
#-----------------------------------------------------------------------------
# 
#-----------------------------------------------------------------------------
def prop_from_fajar(zone,fajar,poly_blk,poly_up):
    sandlist=list(fajar.Pick_Name.unique())
    top=fajar.Pick_Name==zone  # pick the welltop to work on

    # select the well tops in the updip polygon
    sh2_thrd = 0.2; # saturation threshhold to defind HC
    sh2_max = 0.7   # max saturation     
    #name =  plist[pdx]
    # find all subsands, assuming they have numbers after sand name 
    matching = [s for s in sandlist if (s[0:len(zone)]==zone)]
    matching2 = [s for s in matching if len(s)>len(zone)]
    matching22 = [s for s in matching2 if s[len(zone)].isdigit()]
    sands = [zone]+matching22    # main sand and subsand list 

    # adding subsands to the main sand
    pat = '|'.join([r'\b{}\b'.format(x) for x in sands])
    fajar['subsands'] = fajar['Pick_Name'].str.contains(pat).astype(int)
    fajar1 = fajar[fajar['subsands']==1]

    fajar10= pd.DataFrame()
    welllist=fajar1.Wellbore_Name.unique()
    for idx in range(len(welllist)):
        well = welllist[idx]
        fajar0 = fajar1[(fajar1.Wellbore_Name==well)]
        fajar00 = fajar0[fajar0.Pick_Name==zone]
        fajar00.NET_THICKNESS_SUM_TVDSS_DIFF=np.nansum(fajar0.NET_THICKNESS_SUM_TVDSS_DIFF)  # sum of the thickness
        fajar0_por = fajar0[(fajar0.POR2mean>0)&(fajar0.POR2mean<0.4)&(fajar0.NET_THICKNESS_SUM_TVDSS_DIFF>0)]
        if len(fajar0_por)>0:   #thickness weighted average of porosity
            fajar00.POR2mean=np.average(list(fajar0_por.POR2mean),weights=list(fajar0_por.NET_THICKNESS_SUM_TVDSS_DIFF))
        fajar10 = fajar10.append(fajar00)

    fajar10 = fajar10.reset_index(drop=True)
    w_x = fajar10['Easting']; w_y = fajar10['Northing'];w_name=fajar10['Wellbore_Name']
    wx_blk=[];wy_blk=[];color_top=[];wx_ud=[];wy_ud=[];w_name_blk=[];w_name_ud=[];

    Call = 'TBD'   # default fluid is unknown
    for idx in range(0,len(w_x)):
        if inside_polygon(w_x[idx], w_y[idx], poly_blk):
            wx_blk.append(w_x[idx]);wy_blk.append(w_y[idx]);w_name_blk.append(w_name[idx])
            if inside_polygon(w_x[idx], w_y[idx], poly_up):
                wx_ud.append(w_x[idx]);wy_ud.append(w_y[idx]);w_name_ud.append(w_name[idx])

    p = np.asarray(poly_blk)
    rad = 5000   # pick the wells with the radius of 5km from the center of the block
    cent=((np.max(p[:,0])+np.min(p[:,0]))/2,(np.max(p[:,1])+np.min(p[:,1]))/2)   # center of the polygon
    dist = np.sqrt((cent[0]-w_x)**2+(cent[1]-w_y)**2) 
    fajar10['dist'] = dist
    fajar12 = fajar10.sort_values(by=['dist'])
    fajar121 = fajar12[fajar12.dist<rad]
    if len(fajar121)<1:            # if no well found within the radius
        fajar121 = fajar12[0:2]    # pick nearest 2 wells 
    fajar22=fajar121.reset_index(drop=True)
    sh2 = np.nanmean(fajar22[(fajar22.SH2mean>0.35)&(fajar22.SH2mean<0.9)].SH2mean)
                          
    if len(wx_ud)>0:    # if there are wells updip
        well_used = w_name_ud;
        # select the dataframe for the wells in the updip polygon
        fajar2=pd.DataFrame(columns=fajar10.columns)
        for idx in range(0,len(wx_blk)):
            #when there are more than one top at the same location, such as ST, pick the first one
            fajar2=fajar2.append(fajar10[(w_y==wy_blk[idx])&(w_x==wx_blk[idx])][0:1])             
        # define properties
        fajar2=fajar2.reset_index(drop=True)
        por2 = np.nanmean(fajar2[(fajar2.POR2mean>0)&(fajar2.POR2mean<0.4)].POR2mean)
        d_net = np.nanmean(fajar2[fajar2.NET_THICKNESS_SUM_TVDSS_DIFF>0].NET_THICKNESS_SUM_TVDSS_DIFF)
        color_top=fajar2.SH2mean.copy()
        color_top[color_top>=sh2_thrd]=2        # define HC color
        color_top[color_top<sh2_thrd]=0         # Brine color
        color_top[np.isnan(color_top)]=1        # unknow saturation
        if any(fajar2[fajar2.SH2mean>0].SH2mean>=sh2_thrd):
            Call = 'HC'
        else:#all(fajar2[fajar2.SH2mean>0].SH2mean<sh2_thrd):
            Call = 'Brine'
    else:
        well_used = list(fajar22['Wellbore_Name'])
        por2 = np.nanmean(fajar22[(fajar22.POR2mean>0)&(fajar22.POR2mean<0.35)].POR2mean)
        d_net = np.nanmean(fajar22[fajar22.NET_THICKNESS_SUM_TVDSS_DIFF>0].NET_THICKNESS_SUM_TVDSS_DIFF) 
    return (d_net,por2,sh2,Call,color_top,well_used,wx_blk,wy_blk,)
        
                          
## define a function to check if a point is inside of a polygon
def inside_polygon(x, y, points):
    """
    Return True if a coordinate (x, y) is inside a polygon defined by
    a list of verticies [(x1, y1), (x2, x2), ... , (xN, yN)].

    Reference: http://www.ariel.com.au/a/python-point-int-poly.html
    """
    n = len(points)
    inside = False
    p1x, p1y = points[0]
    for i in range(1, n + 1):
        p2x, p2y = points[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside
#-----------------------------------------------------------------------------
# 
#-----------------------------------------------------------------------------
# define a moving average function
def running_mean(amp,dep,dep_reg,win_half):
    amp_reg = [];
    for idx in range(0,len(dep_reg)):
        amp_reg1 = np.mean(amp[(dep<dep_reg[idx]+win_half)&(dep>dep_reg[idx]-win_half)])
        amp_reg.append(amp_reg1)
    amp_reg = np.asarray(amp_reg)
    return amp_reg
#-----------------------------------------------------------------------------
# 
#-----------------------------------------------------------------------------
# define a function to read exported ndi polygon files and change the format for inside_polygon function
def make_poly(f_poly_blk):
    df_blk = pd.read_table(f_poly_blk,delim_whitespace=True, header=None)
    df_blk.columns = ['x', 'y','z','f1','f2','f3','poly_name']
    ## rearrange the polygon data format
    poly_blk = [];
    for idx in range(0,len(df_blk)):
        point = (df_blk.x[idx],df_blk.y[idx])
        poly_blk.append(point)
    return poly_blk

# if the polygon is already red in a dateframe
def make_poly1(df_blk):
    df_blk.columns = ['x', 'y','z','f1','f2','f3','poly_name']
    ## rearrange the polygon data format
    poly_blk = [];
    for idx in range(0,len(df_blk)):
        point = (df_blk.x[idx],df_blk.y[idx])
        poly_blk.append(point)
    return poly_blk
#-----------------------------------------------------------------------------
# 
#-----------------------------------------------------------------------------
# define a function to find the depth of a polygon
def poly_dep(poly_blk,x_blk,y_blk,dep_blk):
    d = np.zeros(len(poly_blk))
    for idx in range(0,len(poly_blk)):
        dist = np.sqrt( ((poly_blk[idx][0]-x_blk)**2)+((poly_blk[idx][1]-y_blk)**2) )
        d[idx] = dep_blk[dist==np.min(dist)]
    return d
#-----------------------------------------------------------------------------
# functions to cut maps into blocks
#-----------------------------------------------------------------------------
# cut only the amplitude map; e.g. there is no depth map for far and near
def cut_map_poly1(amp0,xa0,ya0,poly):
    amp_cut = [];x_cut=[];y_cut=[]
    sqcut = (xa0<np.max(np.asarray(poly)[:,0]))&(xa0>np.min(np.asarray(poly)[:,0]))&(ya0<np.max(np.asarray(poly)[:,1]))&(ya0>np.min(np.asarray(poly)[:,1]))
    amp = amp0[sqcut];xa = xa0[sqcut];ya = ya0[sqcut];
    for idx in range(0,len(amp)):
        if inside_polygon(xa[idx], ya[idx], poly):
            amp_cut.append(amp[idx]);
            x_cut.append(xa[idx]);y_cut.append(ya[idx]); 
    return (amp_cut,x_cut,y_cut)
# cut a grided map
def cut_map_poly1_grid(grid_z,grid_x, grid_y,poly):
    amp_cut = [];x_cut=[];y_cut=[]
    xmin = np.min(np.asarray(poly)[:,0])
    xmax=np.max(np.asarray(poly)[:,0]);
    ymin = np.min(np.asarray(poly)[:,1])
    ymax=np.max(np.asarray(poly)[:,1]);
    
    grid_x1 = grid_x[(grid_y[:,0]>ymin)&(grid_y[:,0]<ymax),:]
    grid_y1 = grid_y[(grid_y[:,0]>ymin)&(grid_y[:,0]<ymax),:]
    grid_z1 = grid_z[(grid_y[:,0]>ymin)&(grid_y[:,0]<ymax),:]
    
    grid_x2 = grid_x1[:,(grid_x1[0,:]>xmin)&(grid_x1[0,:]<xmax)]
    grid_y2 = grid_y1[:,(grid_x1[0,:]>xmin)&(grid_x1[0,:]<xmax)]
    grid_z2 = grid_z1[:,(grid_x1[0,:]>xmin)&(grid_x1[0,:]<xmax)]
    
    for idx in range(0,len(grid_x2[0,:])):
        for jdx in range(0,len(grid_x2[:,0])):
            if inside_polygon(grid_x2[jdx,idx], grid_y2[jdx,idx], poly):
                pass
            else:
                grid_z2[jdx,idx]='nan'
    return (grid_z2,grid_x2,grid_y2)

#-----------------------------------------------------------------------------
# 
#-----------------------------------------------------------------------------
# define a function to cut the polygon based on depth
# x,y,dep defines the structure map
# poly_blk are the points defines the polygon
# d_c is the estimated contact 
def cut_poly(x_blk,y_blk,dep_blk,poly_blk,d_c):
    win_c = 2 # half depth window around d_c to search for contact
    win_gap = 20 # depth window between up down polygon boundry
    dc_up = d_c  # make the gap to the updip half size
    dc_down = d_c+win_gap 
    poly_blk1 = np.asarray(poly_blk)
    d = poly_dep(poly_blk,x_blk,y_blk,dep_blk)  # find the depth of the polygon points
    xc_up = x_blk[(dep_blk<dc_up+win_c)&(dep_blk>dc_up-win_c)]  # find points along dc_up within the window 
    yc_up = y_blk[(dep_blk<dc_up+win_c)&(dep_blk>dc_up-win_c)]    
    xc_down = x_blk[(dep_blk<dc_down+win_c)&(dep_blk>dc_down-win_c)]  # find points along dc_down within the window 
    yc_down = y_blk[(dep_blk<dc_down+win_c)&(dep_blk>dc_down-win_c)]
 
    if any(item == 0 for item in [len(xc_up),len(xc_down)]): #if the gap between d_c to the end of the block is less than win_gap
        win_gap=0
        dc_up = d_c-win_gap
        dc_down = d_c+win_gap
        xc_up = x_blk[(dep_blk<dc_up+win_c)&(dep_blk>dc_up-win_c)]  # find points along dc_up within the window 
        yc_up = y_blk[(dep_blk<dc_up+win_c)&(dep_blk>dc_up-win_c)]    
        xc_down = x_blk[(dep_blk<dc_down+win_c)&(dep_blk>dc_down-win_c)]  # find points along dc_down within the window 
        yc_down = y_blk[(dep_blk<dc_down+win_c)&(dep_blk>dc_down-win_c)]
    # updip polygon
    poly_c_up = [];
    for idx in range(0,len(xc_up)):
        point = [xc_up[idx],yc_up[idx]]
        poly_c_up.append(point)
    poly_up = poly_blk1[d<dc_up]
    polynew = poly_up.tolist()+poly_c_up
    poly_up = sort_poly_point(polynew)   # sort the poly points by polar angle
    # downdip polygon    
    poly_c_down = [];

    for idx in range(0,len(xc_down)):
        point = [xc_down[idx],yc_down[idx]]
        poly_c_down.append(point)
    poly_down = poly_blk1[d>dc_down]
    polynew = poly_down.tolist()+poly_c_down
    poly_down = sort_poly_point(polynew) # sort the poly points by polar angle
    
    return (poly_up,poly_down,win_gap)
#-----------------------------------------------------------------------------
# 
#-----------------------------------------------------------------------------
# function to fort the poly points by polar angle; QC the output. If doesn't look right, try different definition of centroid. 
def sort_poly_point(poly_point):
    # compute centroid
    #cent=(sum([p[0] for p in poly_point])/len(poly_point),sum([p[1] for p in poly_point])/len(poly_point))
    p = np.asarray(poly_point)
    cent=((np.max(p[:,0])+np.min(p[:,0]))/2,(np.max(p[:,1])+np.min(p[:,1]))/2)
    # sort by polar angle
    poly_point.sort(key=lambda p: math.atan2(p[1]-cent[1],p[0]-cent[0]))
    poly_point = poly_point+[poly_point[0]]
    return poly_point
#-----------------------------------------------------------------------------
# 
#-----------------------------------------------------------------------------
#calculate polygon area
# examples
#corners = [(2.0, 1.0), (4.0, 5.0), (7.0, 8.0)]
def PolygonArea(corners):
    n = len(corners) # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area
#-----------------------------------------------------------------------------
# 
#-----------------------------------------------------------------------------
# import a ndi color map to python 
def import_ndi_cm(cmap_file):
    df_cm = pd.read_table(cmap_file,delim_whitespace=True,header=None,skiprows=1)
    color=[]
    for idx in range(0,len(df_cm)):
        color.append((df_cm.loc[idx,0],df_cm.loc[idx,1],df_cm.loc[idx,2]))
    
    import matplotlib.colors as clr
    """    
    fdx1 = cmap_file.rfind('/')
    fdx2 = cmap_file.rfind('.')
    fcmap = cmap_file[fdx1+1:fdx2]
    fcmap_r = cmap_file[fdx1+1:fdx2]+'_r' 
    """
    cmap = clr.LinearSegmentedColormap.from_list('cm',color/np.max(color),len(color))
    cmap_r = clr.LinearSegmentedColormap.from_list('cm_r',color[::-1]/np.max(color),len(color))
    return (cmap,cmap_r)

