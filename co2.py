import time

import MDSplus
import numpy
import numpy as np


class gadata:
    def __init__(
        self,
        signal,
        shot,
        tree=None,
        connection=None,
        nomds=False
        ):
        
        self.signal = signal
        self.shot = shot
        self.zdata = -1
        self.xdata  = -1
        self.ydata 	= -1
        self.zunits = ''
        self.xunits = ''
        self.yunits = ''
        self.rank = -1
        self.connection = connection

        t0 =  time.time()
        found = 0
        
        if self.connection is None:
            print('No connection!!!')
            self.connection = MDSplus.Connection('atlas.gat.com')

        if nomds == False:
            try:     
                if tree != None:
                    tag 	= self.signal
                    fstree 	= tree 
                    found   = 1
                else:
                    tag 		= self.connection.get('findsig("'+self.signal+'",_fstree)').value
                    fstree    	= self.connection.get('_fstree').value 
                self.connection.openTree(fstree,shot)
                self.zdata  	= self.connection.get('_s = '+tag).data()
                self.zunits 	= self.connection.get('units_of(_s)').data()  
                self.rank   	= numpy.rank(self.zdata)	
                self.xdata     	= self.connection.get('dim_of(_s)').data()
                self.xunits 	= self.connection.get('units_of(dim_of(_s))').data()
                if self.xunits == '' or self.xunits == ' ': 
                    self.xunits     = self.connection.get('units(dim_of(_s))').data()
                if self.rank > 1:
                    self.ydata 	= self.connection.get('dim_of(_s,1)').data()
                    self.yunits 	= self.connection.get('units_of(dim_of(_s,1))').data()
                    if self.yunits == '' or self.yunits == ' ':
                        self.yunits     = self.connection.get('units(dim_of(_s,1))').data()
    
                found = 1	

                # MDSplus seems to return 2-D arrays transposed.  Change them back.
                if numpy.rank(self.zdata) == 2: self.zdata = numpy.transpose(self.zdata)
                if numpy.rank(self.ydata) == 2: self.ydata = numpy.transpose(self.ydata)
                if numpy.rank(self.xdata) == 2: self.xdata = numpy.transpose(self.xdata)

            except Exception as e:
                #print '   Signal not in MDSplus: %s' % (signal,) 
                pass
        

        # Retrieve data from PTDATA
                if found == 0:
                        self.zdata = self.connection.get('_s = ptdata2("'+signal+'",'+str(shot)+')')
                        if len(self.zdata) != 1:
                                self.xdata = self.connection.get('dim_of(_s)')
                                self.rank = 1
                                found = 1

        # Retrieve data from Pseudo-pointname 
        if found == 0:
            self.zdata = self.connection.get('_s = pseudo("'+signal+'",'+str(shot)+')')
            if len(self.zdata) != 1:
                self.xdata = self.connection.get('dim_of(_s)')
                self.rank = 1
                found = 1

        if found == 0: 
            #print "   No such signal: %s" % (signal,)
            return

        print('   GADATA Retrieval Time : ',time.time() - t0)
        
atlconn = MDSplus.Connection('atlas.gat.com')
shot = 132708
signal = "denr0_uf_1"

mydata = gadata(signal, shot, connection=atlconn)
tmin = 0
tmax = 3000
itime = (tmin <= mydata.xdata) & (mydata.xdata <= tmax)

print(mydata.xdata)