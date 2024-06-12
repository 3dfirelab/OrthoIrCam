params_flag = {
             'flag_parallel' : True,
             'flag_restart'  : False,

             'flag_georef_mode' : 'SimpleHomography', #'WithTerrain' (not working) or 'SimpleHomography'

              #lwir
              'flag_lwir_processRawData' : False,
              'flag_lwir_plot_warp' :      False,
              'flag_lwir_georef' :      True,   
              'flag_lwir_plot_georef' : False,
              #vis
              'flag_vis_processRawData' : True,
              'flag_vis_plot_warp' : False,
              'flag_vis_georef' : True,
              'flag_vis_plot_georef' : True,
              #mir
              'flag_mir_processRawData': False,

               }

params_rawData = {'root':           '/mnt/dataEstrella/2024_Bcn_RadiationSeverity/',
                  'root_data':      '/mnt/dataEstrella/2024_Bcn_RadiationSeverity/Straw1/',
                  'root_postproc' : '/mnt/dataEstrella/2024_Bcn_RadiationSeverity/Straw1/Postproc/',
                  'plotname': 'straw1',
                  'fire_date'      : '2024-02-26',
                  
                  #lwir
                  'root_data_DirLwir': '/Optris/',
                  'startTime_lwir' :  0,
                  'endTime_lwir'   :  700,
                  'period_lwir'    :  -1. , # seconds. set a negative value to use all frame
                  'inputFormat_lwir' : 'netcdf',
                  'inputFile_lwir' : 'optris640_16060082_20240226_132604.nc',
                  'outputFilename_lwir': 'temp',
                  'applyNoiseFilter_lwir': False, 
                  'startMaskTimeOptris' : -999,
                  'endMaskTimeOptris'   : -999,

                  #vis
                  #'root_data_DirVis': '/Visible_Agus/burning/',
                  'root_data_DirVis': '/Visible_Agus/pre-burning/',
                  'startTimeVis' : -3000,
                  'endTimeVis'   : 300,

                  'dir_time_calibration' :  'Straw1/CalibrationTime_vis_IR/', 
                  'time_calibration_vis' :  'P1380488.JPG',
                  'time_calibration_mir' :  None,
                  'time_calibration_lwir':  'straw1_georef_000008_SH.npy',
                 }


params_gps = {  'dir_gps'      : 'Instruments_Location/',
                #
                'cf_format'    : 'textFile_xyz', 
                'loc_cf_file'       : 'fuelBedLocation.txt',
                'cornerFireName' : 'cornerFireNames.txt',
                #
                'loc_camera_file' : None,
                #
                'contour_file'               : 'fuelBedLocation.txt', 
                'contour_feature_name'   : None, 
                'ctr_format'                 : 'textFile_xyz', 
                'contour_file_shrinkPlotMask': None, # (x2 grid_resolution) 
             }

params_grid={
            'grid_resolution' : .005,
            'grid_size' : 2,
            }

params_lwir_camera={
            #########
            #driver.py
            ########
            'flag_costFunction' : 'EP08',
            'camera_name' : 'optrisPI640_10mm',
            'cameraLens' : 60,
            'cameraDimension' : [480, 640],
            'load_radiance' : False,
            #
            'dir_input'      : 'LWIR/',
            'dir_img_input'  : 'raw_data/npy/',
            'dir_img_input_*': 'temp_id*.npy',
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
            'warp_using_fixCamera': True,          # for small scale experiment with fix camera, homography from first frame is used
            #
            'ref00Update_threshold' : 0.8,       # in manuscript: rho_ECC^Align = ref00Update_threshold * corr_ref
                                                 # when three image are below ref00Update_threshold * corr_ref -> trigger reference update
            'forceRef00Updateat' : [],           # to force reference update on specific images to help aglo.
            #
            'time_tail_ref00' : 20.0,      # in the manuscript: t_tail 
            'incr_ifile_default' : 1,      # if you want to skip frame. at 1Hz frame rate not recommended, keep to 1.
            'energy_good_1' : 0.7,         # in mansucript: rho_ECC^Ref. if rho_ECC > energy_good_1, img is kept in stack of potential new Ref img 
            'energy_good_2' : 0.55,         # in mansucript: act for rho_ECC^Ref as well. if rho_ECC remains > energy_good_2 for the last nbre_consecutive_frame_above_35 img, 
                                           #                img is kept in stack of potential new Ref img
            'nbre_consecutive_frame_above_35': 4,# need to be set to 4. I think there is some hard set in the code that are not consistant with changing this value.
            #
            'kernel_ring':451,                    # because of distrotion is not corrected, only extended area around the plot is used for the alignement. not all image. 
                                                 # this is why the 51. this kernel is applied to the grid. the resulting mask cut out part of img that are far from the plot center.
                                                 # assumming that the camera is pointing at the plot center.
            # feature-based alignement: 
            'of_blockSize': 7,
            'of_qualityLevel':0.15,
            'of_feature_kernel_ring':True,
            # area-based alignement
            'kernel_warp' : 5,                   # in manuscript: k^mask.  see mask_EP08 in tools.py
            'kernel_plot' : 0,                   # in manuscript: k^plot. see mask_EP08 in tools.py
            'lowT_param' : [310, 21],             # control mask used in EP08 [temp threshold, kernel]. see mask_EP08 in tools.py
            'final_opti_threshold' : 0.,         # in manuscript: flag^ECC. >1 it switch-off ECC call, if ==0 it call it for every frame.
                                                 #                          0< <1 it calls ECC only if the relative diff of rho_corr to rho_corr_Ref is > to given value
            'diskSize_driver_px' : 25,                   # in manuscript: k^mask.  see mask_EP08 in tools.py
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
            'mask_burnNobun_type_val_refined': '12.0',
            
            #########
            #ssim_prev4.py
            #########
            'final_selection_force2keep': [265,504,506,851,1095],
            'final_selection_4bin': [263,264,505,850,1082,1083,1084,1085,1086,1087,1088,1092,1093,1094,434,],

            }

params_vis_camera={
                'inverseColor' : False,
                'warp_on_prev_first': False, 
                'warp_using_fixCamera': True,          # for small scale experiment with fix camera, homography from first frame is used
                'shrink_factor' : 1,
                'dir_img_input_*' : '*.JPG',
                'track_mode' : 'track_background',
                'ref00Update_threshold' : 0.8,
                'flag_costFunction' : 'EP08',
                'kernel_plot' : 3,
                'cameraDimension' : [3840, 2880],
                'camera_name' : 'lumixAgus_14mm',
                'cameraLens' : 'na',
                'final_opti_threshold' : -0.0,
                'time_tail_ref00' : 60.0,
                'energy_good_1' : 0.7,
                'energy_good_2' : 0.65,
                'time_start_processing' : -3000,
                'clahe_clipLimit' : 2,
                'filenames_no_helico_mask' : None,  
                'dir_input' : 'VIS-pre/',
                'dir_img_input' : 'Raw/',
                'forceRef00Updateat' : [],
                'kernel_warp' : 7,
                'incr_ifile_default' : 1,
                'nbre_consecutive_frame_above_35': 4,
                'of_blockSize': 11,
                'of_qualityLevel':0.3,
                'of_feature_kernel_ring':True,
                'kernel_ring':51,
                # use in cornerFirePicker in driver.py to control fire cluster selection
                'temperature_threshold_cornerFirePicker': 20,    }

params_mir_camera={
                }

params_georef={
                'dir_dem_input' : 'DEM/',
                #'dem_file' : 'zeros_1cm.npy',
                'dem_file' : 'zeros_05cm.npy',
                #'dem_file' : 'zeros_01cm.npy',
                'run_opti' : False,
                # lwir
                'trange' : [22, 30],
                'ssim_win_lwir' : 0,
                '#frames_history_tail_lwir' : 4, # in the manuscript: n_tail
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
                }

params_sat = {
    }

params_navashni={
}
