params_flag = {
             'flag_parallel' : True,
             'flag_restart'  : False,

             'flag_georef_mode' : 'SimpleHomography', #'WithTerrain' (not working) or 'SimpleHomography'

              #lwir
              'flag_lwir_processRawData' : True,
              'flag_lwir_plot_warp' :      True,
              'flag_lwir_georef' :      True,   
              'flag_lwir_plot_georef' : True,
              #vis
              'flag_vis_processRawData' : False,
              'flag_vis_plot_warp' : False,
              'flag_vis_georef' : False,
              'flag_vis_plot_georef' : False,
              #mir
              'flag_mir_processRawData': False,

               }

params_rawData = {'root':           '/home/paugam/disc/EstrellaData/2008_Ngarkat/',
                  'root_data':      '/home/paugam/disc/EstrellaData/2008_Ngarkat/Data/',
                  #'root_postproc' : '/home/paugam/2008_Ngarkat/Postproc/',
                  'root_postproc' : '/home/paugam/disc/EstrellaData/2008_Ngarkat/Postproc/',
                  'plotname': 'ngarkat',
                  'fire_date'      : '2008-03-05',
                  
                  #lwir
                  'root_data_DirLwir': '/FLIR570/MAT/',
                  'startTime_lwir' :  0,
                  'endTime_lwir'   :  320,
                  'period_lwir'    :  -1. , # seconds. set a negative value to use all frame

                  #vis
                  'root_data_DirVis': None,
                  'startTimeVis' : None,
                  'endTimeVis'   : None,

                  'dir_time_calibration' :  None, 
                  'time_calibration_vis' :  None,
                  'time_calibration_mir' :  None,
                  'time_calibration_lwir':  None,
                 }


params_gps = {  'dir_gps'      : 'Instruments_Location/',
                #
                'cf_format'    : 'kml_latlong', 
                'loc_cf_file'  : 'Ngarkat_gcps.kml',
                'cf_feature_name'   : 'Ngarkat_gcps', 
                'cornerFireName' : 'cornerFireNames.txt',
                #
                'loc_camera_file' : None,
                #
                'contour_file'               : 'Ngarkat_plotContour_tuned.kml', 
                'contour_feature_name'   : 'plotmask', 
                'ctr_format'                 : 'kml_latlong', 
                'contour_file_shrinkPlotMask': None, # (x2 grid_resolution) 
             }

params_grid={
            'grid_resolution' : 1,
            'grid_size' : 550,
            }

params_lwir_camera={
            #########
            #driver.py
            ########
            'flag_costFunction' : 'EP08',
            'camera_name' : 'agema570_certec',
            'cameraLens' : 24,
            'cameraDimension' : [240, 320],
            #
            'dir_input'      : 'LWIR33/',
            'dir_img_input'  : 'raw_data/npy/',
            'dir_img_input_*': 'ngarkat_*.npy',
            # use in cornerFirePicker in driver.py to control fire cluster selection
            'temperature_threshold_cornerFirePicker': -999, 
            # param to convert theraml signal to uint8 img in optris.py
            'clahe_clipLimit' : 2,
            # for the segmentation of the helico skid
            'mask_helico_seg_mean_threshold' : 26,
            'mask_helico_seg_gradmag_threshold' : 2,
            #
            'time_start_processing' : 0,
            'shrink_factor' : 1,                 # to decrease the resolution of the input image.
            'track_mode' : 'track_background',
            #
            'warp_on_prev_first': False,          # to feature alignement is only run first on prev image. If result ok, skip match with all img tail. This is used to save time. 
            #
            'ref00Update_threshold' : 0.9,       # in manuscript: rho_ECC^Align = ref00Update_threshold * corr_ref
                                                 # when three image are below ref00Update_threshold * corr_ref -> trigger reference update
            'forceRef00Updateat' : [],           # to force reference update on specific images to help aglo.
            #
            'time_tail_ref00' : 20.0,      # in the manuscript: t_tail 
            'incr_ifile_default' : 1,      # if you want to skip frame. at 1Hz frame rate not recommended, keep to 1.
            'energy_good_1' : 0.65,         # in mansucript: rho_ECC^Ref. if rho_ECC > energy_good_1, img is kept in stack of potential new Ref img 
            'energy_good_2' : 0.6,         # in mansucript: act for rho_ECC^Ref as well. if rho_ECC remains > energy_good_2 for the last nbre_consecutive_frame_above_35 img, 
                                           #                img is kept in stack of potential new Ref img
            'nbre_consecutive_frame_above_35': 4,# need to be set to 4. I think there is some hard set in the code that are not consistant with changing this value.
            #
            'kernel_ring':451,                    # because of distrotion is not corrected, only extended area around the plot is used for the alignement. not all image. 
                                                 # this is why the 51. this kernel is applied to the grid. the resulting mask cut out part of img that are far from the plot center.
                                                 # assumming that the camera is pointing at the plot center.
            # feature-based alignement: 
            'of_blockSize': 11,
            'of_qualityLevel':0.1,
            'of_feature_kernel_ring':True,
            # area-based alignement
            'kernel_warp' : 5,                   # in manuscript: k^mask.  see mask_EP08 in tools.py
            'kernel_plot' : 0,                   # in manuscript: k^plot. see mask_EP08 in tools.py
            'lowT_param' : [350, 5],             # control mask used in EP08 [temp threshold, kernel]. see mask_EP08 in tools.py
            'final_opti_threshold' : 0.,         # in manuscript: flag^ECC. >1 it switch-off ECC call, if ==0 it call it for every frame.
                                                 #                          0< <1 it calls ECC only if the relative diff of rho_corr to rho_corr_Ref is > to given value
            #
            'frame2skip':[1081],
            
            #########
            #refined_lwir.py
            #########
            'ringPlot_refined': 250,
            'reso_refined':2,
            'diskSize_refined': 30, # it was 30
            'diskSize_ssim_refined': 19,
            'igeostart_refined':0,
            'time_ff_behind_refined':65,
            'temp_threshFire_refined':630,
            'ssim2D_thres_refined': .57, 
            'ssim2dF_filter_refined': True,
            'time_aheadOf_refined':0,
            'mask_burnNobun_type_refined': 'thr',  # methode used to compute the mask burnNoBurn that control part of the plot that was already burnt before the start of the monitoring
            'mask_burnNobun_type_val_refined': '20.0',
            
            #########
            #ssim_prev4.py
            #########
            'final_selection_force2keep': [265,504,506,851,1095],
            'final_selection_4bin': [263,264,505,850,1082,1083,1084,1085,1086,1087,1088,1092,1093,1094,434,],

            'temperature_threshold_cornerFirePicker': 0,
            }

params_vis_camera={
                'inverseColor' : False,
                'warp_on_prev_first': False, 
                'shrink_factor' : 4,
                'dir_img_input_*' : '*.JPG',
                'track_mode' : 'track_background',
                'ref00Update_threshold' : 0.8,
                'flag_costFunction' : 'EP08',
                'kernel_plot' : 3,
                'cameraDimension' : [3840, 2880],
                'camera_name' : 'goproKCL',
                'cameraLens' : 'na',
                'final_opti_threshold' : -0.0,
                'time_tail_ref00' : 60.0,
                'energy_good_1' : 0.7,
                'energy_good_2' : 0.65,
                'time_start_processing' : 0,
                'clahe_clipLimit' : 2,
                'filenames_no_helico_mask' : [2885, 2917],
                'dir_input' : 'VIS301c/',
                'dir_img_input' : 'Raw/',
                'forceRef00Updateat' : [],
                'kernel_warp' : 7,
                'incr_ifile_default' : 1,
                'nbre_consecutive_frame_above_35': 4,
                'of_blockSize': 11,
                'of_qualityLevel':0.3,
                'of_feature_kernel_ring':True,
                'kernel_ring':51,
                }

params_mir_camera={
                'nbrePix_height' : 240,
                'cameraLens' : 40,
                'cp_lwir_mir' : [[[115, 166], [261, 181], [305, 187], [291, 135], [133, 98], [178, 252]], [[25, 119], [212, 129], [266.5, 138.5], [243, 73], [43, 34], [110, 225]]],
                'mir_name_ref' : 'calibaration_vis_IR/26nextMorning/test_set/NEW000159.MAT', # 26burn1/NEW000228.MAT',
                'lwir_name_ref' : 'calibaration_vis_IR/26nextMorning/test_set/Record_2014-08-27_06-36-02.csv', #26burn1/Record_2014-08-26_10-53-44.csv',
                'time_start' : 250,
                'camera_name' : 'agema550_40',
                'nbrePix_width' : 320,
                'plot_mir_lwir_match' : True,
                'dir_img_input' : 'raw/',
                'sensor_width' : 10,
                'dir_input' : 'MIR301c/',
                'mir_fire_BT': 500, 
                'mir_ff_close_pix':3,
#'file_arrivalTime_lwir':'ROS_Sensitivity_LWIR301c/output_dx=01.0_dt=000.0_lwir_deepLv5_fsrbf/arrivalTime_interp_and_clean.npy',
                'file_arrivalTime_lwir':'LWIR301c/Georef_refined_SH/front_LN/skukuza6_arrivalTime_LN.npy',
                'filteron':'yes#-999#1.e6', # always on
                }

params_georef={
                'dir_dem_input' : 'DEM/',
                'dem_file' : 'ngarkat_ngarkat_dem.npy',
                'run_opti' : True,
                # lwir
                'trange' : [22, 150],
                'ssim_win_lwir' : 0,
                '#frames_history_tail_lwir' : 12, # in the manuscript: n_tail
                # visible
                'ssim_win_visible' : 21,
                '#frames_history_tail_visible' : 16,
                # to activate corner fire tracking
                'look4cf' : False,
                'cornerFire_Temp_threshold' : -99,
                # for CNN segmentation
                'cnn_meanTemp':305,
                'cnn_version':'v5',
                #
                'angle_prettyPlot':2.5,
                }

params_veg={
                'D' : [400.0, 400, 400],
                'fuelDepth' : [-999, 0.5, 0.1],
                'ratio_fuel_density' : [0, 0.8333333333333334, 0.16666666666666666],
                'gray_scale' : [(0, 65), (65, 110), (110, 255)],
                'fuelWidth' : ['na', 'na', 'na'],
                'fuelLenght' : ['na', 'na', 'na'],
                'sv' : [-999, 5000.0, 5000.0],
                'fuelLoad' : [-999, -999, -999],
                'Temperature_veg' : [30, 30, 29],
                'nbre_species' : 3,
                'moisture' : [0.01, 0.01, 0.01],
                'total_fuel_load' : 0.2654,
                'bulkD' : ['na', 'na', 'na'],
                }

params_sat = {
    'name': 'TET-1', # corrected time to fit english summer time of the helico data
    'time': '2014-08-26_11:07:34',
    'frp_toa' : 11.96, 
    'atm_transmittance': [ 0.67, 0.75],
    'va':  20.2,
    'az': 92.1,
    'locPlot1D': (78,12),
    }

params_navashni={
    'fuelLoad': 2654., #kg/ha
    'moisture': 22.8,
    'temperatue': 30, #C
    'humidity': 60,   #%
    'windSpeed': 2.63 #m/s
}
