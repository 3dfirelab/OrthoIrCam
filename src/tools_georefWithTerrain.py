from __future__ import print_function
from __future__ import division
from builtins import zip
from builtins import range
from past.utils import old_div
import os,sys,glob
import numpy as np 
import matplotlib.pyplot as plt
import cv2
from scipy import spatial
import matplotlib.tri as mtri
import transformation 
from shapely.geometry import Polygon, Point
import itertools
import pdb 
import datetime
import multiprocessing
import matplotlib as mpl 



########################################
def to_planar_coord(pts_xy):
    xx = pts_xy[1] - pts_xy[0]
    xx = old_div(xx,np.linalg.norm(xx))
    zz = np.cross(xx,pts_xy[2] - pts_xy[0])
    zz = old_div(zz,np.linalg.norm(zz))
    yy = np.cross(zz,xx)
    return pts_xy, np.array( [ [np.dot(xx,pts_xy[i]-pts_xy[0]), np.dot(yy,pts_xy[i]-pts_xy[0])] for i in range(pts_xy.shape[0])]), (xx,yy,zz,pts_xy[0])


########################################
def to_3d_coord(pts2d,xx,yy,zz,x0):
    return np.array([x0 + pts2d[i,0]*xx + pts2d[i,1]*yy   for i in range(pts2d.shape[0])])


########################################
def star_triangle_img_pixel_intersection(param):
    return triangle_img_pixel_intersection(*param)

def triangle_img_pixel_intersection( i_tri, pts_xy ,rvec, tvec, K, D, ni, nj):

    out = []
    out_pt_xyz = []

    pts_xy3D, pts_xy2D, tri_coord = to_planar_coord(np.array(pts_xy))

    pts_ij_, _ = cv2.projectPoints(np.array(pts_xy), rvec, tvec, K, D)
    pts_ij = np.fliplr(pts_ij_[:,0,:])

    ii_img_polygon_l = max([0, int(np.round(pts_ij[:,0].min())) -1])
    ii_img_polygon_u = min([ni,int(np.round(pts_ij[:,0].max())) +1])
    jj_img_polygon_l = max([0, int(np.round(pts_ij[:,1].min())) -1])
    jj_img_polygon_u = min([nj,int(np.round(pts_ij[:,1].max())) +1])

    if ((ii_img_polygon_l>=ni) & (ii_img_polygon_u>=ni)) |\
       ((ii_img_polygon_l<=0   ) & (ii_img_polygon_u<=0   )) |\
       ((jj_img_polygon_l>=nj) & (jj_img_polygon_u>=nj)) |\
       ((jj_img_polygon_l<=0   ) & (jj_img_polygon_u<=0   )) : 
        return out, out_pt_xyz

    tri_H, mask = cv2.findHomography(pts_ij, pts_xy2D, cv2.RANSAC,5.0)


    tri_polygon = Polygon(pts_ij[:3])#.convex_hull
    tri_area       = tri_polygon.area
    if tri_area < 5.e-3 : #for point on the edge where distortion is big due to the bad distortion model
                          #for now we neglect those points. To be removed when the distortion is better estimated ##MERDE
        return out, out_pt_xyz

    tri_area_m2    = .5*np.linalg.norm(np.cross( (pts_xy2D[1]-pts_xy2D[0]), (pts_xy2D[0]-pts_xy2D[2]) ))

    idxi = np.arange(ii_img_polygon_l,ii_img_polygon_u) 
    idxj = np.arange(jj_img_polygon_l,jj_img_polygon_u) 
    for iii,jjj in itertools.product(idxi, idxj):
        
        pts = [ [ iii  , jjj   ], \
                [ iii+1, jjj   ], \
                [ iii+1, jjj+1 ], \
                [ iii  , jjj+1 ], \
              ]
        img_polygons_ = Polygon(pts)

        intersection = tri_polygon.intersection(img_polygons_)
        if intersection.area!=0:
            intersect_ij = np.array(intersection.exterior.coords.xy,dtype=np.float32).T.reshape(-1,1,2)
            intersect_xy = cv2.perspectiveTransform(intersect_ij,tri_H).reshape(-1,2)
            intersect_xyz = to_3d_coord(intersect_xy,*tri_coord)
            tri_img_intersection_area_m2 = Polygon(intersect_xy).area
         
            if old_div((tri_img_intersection_area_m2 - tri_area_m2),tri_area_m2) > 1.e-3 :
                
                plt.clf()
                ax = plt.subplot(121)
                ax.scatter(intersect_xy[:,0],intersect_xy[:,1],c='m',s=100)
                ax.scatter(pts_xy2D[:3,0],pts_xy2D[:3,1],s=10)
                ax = plt.subplot(122)
                #imgi,imgj = img_polygons[iii,jjj].exterior.coords.xy
                imgi,imgj = img_polygons_.exterior.coords.xy
                ax.scatter(imgi,imgj,c='r',s=100)
                ax.scatter(intersect_ij[:,0,0],intersect_ij[:,0,1],c='m',s=50)
                ax.scatter(pts_ij[:3,0],pts_ij[:3,1],s=10)
                plt.show()
                pdb.set_trace()
                out.append([None])
            else:
                out.append([iii, jjj, i_tri, tri_img_intersection_area_m2])
           

            for pt in pts: 
                if intersection.intersects(Point(pt)) & (pt[0]>0) & (pt[1]>0) & (pt[0]<ni) & (pt[1]<nj)  : 
                    pt_xy = cv2.perspectiveTransform(np.array(pts[0],dtype=np.float32).reshape(-1,1,2),tri_H).reshape(-1,2)
                    pt_xyz = to_3d_coord(pt_xy,*tri_coord)
                    out_pt_xyz.append([pt[0],pt[1],pt_xyz])


    return out, out_pt_xyz

#################################################
def set_triangle(burnplot, dir_data, flag_restart):

    nx,ny = burnplot.shape

    if (not(flag_restart)) | (not(os.path.isfile(dir_data+'tri_angles.npy'))):
        print('terrain triangulation')
        idx = np.where(burnplot.mask>=0) #all points
        points = np.dstack((burnplot.grid_e[idx],burnplot.grid_n[idx],burnplot.terrain[idx])).reshape(-1,3)

        points_shift = points.mean(axis=0) # centered data for qhull in delaunay triangulation
        points -= points_shift

        tess = spatial.Delaunay(points[:,:2])
        points += points_shift 
        tri = mtri.Triangulation(points[:,0],points[:,1],triangles=tess.vertices)

        np.save( dir_data+'/tri_angles', [ tri, points] )

    else:
        tri, points = np.load(dir_data+'/tri_angles.npy')
   
    try: 
        print('map triangles') 
        triangles = []
        triangles_grid = []
        triangles_area = []
        for simplex in tri.triangles:
            pts_xy = list(zip( points[simplex,0], points[simplex,1], points[simplex,2] ))
            extra_points = [np.array(pts_xy[0]) + (np.array(pts_xy[1])-np.array(pts_xy[0])) + (np.array(pts_xy[2])-np.array(pts_xy[0])) ] 
            triangles.append( pts_xy +  [tuple(extra_point_) for extra_point_ in extra_points] )
            
            idx_grid = ([np.unravel_index(simplex,(nx,ny))[0].min()],[np.unravel_index(simplex,(nx,ny))[1].min()])
            #this works because the triangle generated by spatial.Delaunay above are run on the cartesian grid.
            #so each cartesian grid is split in two triangle as shown below. The formulation to get idx_grid is only correct in this
            #situation.
            # +---+
            # |\  |
            # | \ |
            # |  \|
            # +---+

            if len(idx_grid[0])!=1:
                print('issue in finding grid idx')
                pdb.set_trace()

            triangles_grid.append(idx_grid)
        
            pts_xy = np.array(triangles[-1][:-1])
            grid_area_m2    = .5*np.linalg.norm(np.cross( (pts_xy[1] -pts_xy[0]), (pts_xy[0] -pts_xy[2]) ))
            triangles_area.append(grid_area_m2)
    except: 
        print('if it crashed here, probably tri_angles.npy is no more matching your grid, check in') 
        print(dir_data)
        print('and restart. (deleting tri_angles.npy could help)')
        sys.exit()

    np.save( dir_data + 'triangle_coord', np.array( triangles     ))
    np.save( dir_data + 'triangle_grid' , np.array( triangles_grid))
    np.save( dir_data + 'triangle_aera',  np.array( triangles_area ))

    return tri, points, triangles, triangles_grid, triangles_area



