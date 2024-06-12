from builtins import range
from builtins import object
import numpy as np
import os
from scipy import ndimage
import matplotlib.pyplot as plt 
import pdb
from matplotlib.widgets import Button
from time import sleep


#####################################################
def add_askyesorno(fig,question):

    fig.text(0.05,0.65,question)

    callback = Index()
    axyes     = plt.axes([0.05, 0.535, 0.1, 0.07])
    axno      = plt.axes([0.05, 0.465, 0.1, 0.07])
    axReset   = plt.axes([0.05, 0.395, 0.1, 0.07])
    axBackgrd = plt.axes([0.05, 0.325, 0.1, 0.07])
  
    byes = Button(axyes, 'yes')
    byes.on_clicked(callback.yes)

    bno = Button(axno, 'no')
    bno.on_clicked(callback.no)
    
    breset = Button(axReset, 'reset')
    breset.on_clicked(callback.reset)
    
    bbackgrd = Button(axBackgrd, 'try background')
    bbackgrd.on_clicked(callback.backgrd)
    
    return callback, byes, bno,breset, bbackgrd


######################################################
class Index(object):
    ind = ''
    def yes(self, event):
        self.ind = 'yes'
        if self.ind == 'yes' :
            plt.close()

    def no(self, event):
        self.ind = 'no'
        if self.ind == 'no' :
            plt.close()
    
    def reset(self, event):
        self.ind = 'reset'
        if self.ind == 'reset' :
            plt.close()
    
    def backgrd(self, event):
        self.ind = 'backgrd'
        if self.ind == 'backgrd' :
            plt.close()


class cornerFirePicker(object):
    def __init__(self, line, temp, fireName, rawid, winsize=10, suffix='gcp_',outdir='./', temp_threshold=80 , flag=None):
        self.line = line
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())  
        self.temp = temp
        self.temp_threshold = temp_threshold
        self.rawid = rawid
        self.fireName = fireName
        self.nbrePt = 0
        self.outdir = outdir
        self.suffix = suffix
        self.winsize = winsize

        answerok = 'mm'
        
        self.coords = np.array([])
        question = "Is cF selction ok?"
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)
        
        if flag is None:
            callback,byes,bno,breset,bbackgrd = add_askyesorno(self.line.figure, question)
        else:
            plt.title('close figure when the 4 corner fires are set, \n if you cannot click on one, the threshold temp in the config is probably to high')
        plt.show(block=True)
        if flag is None:
            answerok = callback.ind
        else:
            if self.coords.shape[0]==4:
                answerok = 'yes'
        
        if answerok == 'yes':
            self.cfok = True
            if flag is None:
                np.save(self.outdir+self.suffix+self.fireName+"_{:d}".format(self.rawid),
                        self.coords )
            if flag == '4ref00':
                print (self.coords)
                with open(self.outdir+self.suffix+self.fireName, 'w') as f:
                    for i,j in zip(self.coords[:,0],self.coords[:,1]): 
                        f.write('{:.0f}  {:.0f}\n'.format(j,i))

        else: 
            self.cfok = False
        self.answer = answerok


    def __call__(self, event):
        #print 'click', event
        if event.inaxes!=self.line.axes:
            self.line.figure.canvas.mpl_disconnect(self.cid)
            return
        xs_ = event.xdata
        ys_ = event.ydata
       
        #define roi around selected point
        nx,ny = self.temp.shape[:2]
        x,y,w,h = (xs_,ys_,self.winsize,self.winsize)
        xmin = int(round(max([x-w,0]),0))
        xmax = int(round(min([x+w+1,nx-1]),0))
        ymin = int(round(max([y-h,0]),0))
        ymax = int(round(min([y+h+1,ny-1]),0))
        roi = self.temp[xmin:xmax,ymin:ymax] 
        if len(roi.flatten()) == 0: 
            pdb.set_trace()
       
        if self.temp_threshold > 0: 

            #get cluster above T T_threshold
            idx = np.where(roi > self.temp_threshold)
            mask = np.zeros(roi.shape)
            mask[idx] = 1
            s = [[0,1,0], \
                 [1,1,1], \
                 [0,1,0]] # for diagonal
            labeled, num_cluster = ndimage.label(mask, structure=s )
            if num_cluster == 0:
                return 
            temp_cluster = []
            for i_cluster in range(num_cluster):
                idx = np.where(labeled==i_cluster+1)
                temp_cluster.append(roi[idx].max())
            i_cluster = np.array(temp_cluster).argmax()
            idx_cluster = np.where(roi == roi[np.where(labeled==i_cluster+1)].max())

            xs = idx_cluster[0][0] + xmin
            ys = idx_cluster[1][0] + ymin
       
        else:
            xs, ys = xs_,ys_
        
        self.xs.append(xs)
        self.ys.append(ys)
        self.line.set_data(self.xs, self.ys)
        self.line.figure.canvas.draw()
        self.nbrePt += 1
        
        print (self.nbrePt)
        if self.nbrePt ==4: 
            self.line.figure.canvas.mpl_disconnect(self.cid)
            self.coords = np.array( [self.ys[1:], self.xs[1:]], dtype=float).T

