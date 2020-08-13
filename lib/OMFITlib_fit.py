#-*-Python-*-
# Created by logannc at 20 Mar 2017  15:42

#from numpy import linspace, atleast_1d
import numpy as np
from scipy import interpolate
import scipy.signal

class interp1e(interpolate.interp1d):
    """
    Shortcut for scipy.interpolate.interp1d with fill_value='extrapolate' and bounds_error=False as defaults
    """
    __doc__ += interpolate.interp1d.__doc__.replace('\n    ----------\n', ':\n').replace('\n    -------\n', ':\n').replace('\n    --------\n', ':\n')

    def __init__(self, x, y, *args, **kw):
        kw.setdefault('fill_value', 'extrapolate')
        kw.setdefault('bounds_error', False)
        interpolate.interp1d.__init__(self, x, y, *args, **kw)

def smooth(x, window_len=11, window='hanning', axis=0):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the beginning and end part of the output signal.

    input:
        :param x: the input signal
        :param window_len: the dimension of the smoothing window; should be an odd integer; is ignored if `window` is an array
        :param window: the window function to use, see scipy.signal.get_window documentation for list of available windows
                'flat' or 'boxcar' will produce a moving average smoothing
                Can also be an array, in which case it is used as the window function and `window_len` is ignored
        :param axis: The axis that is smoothed

    output:
        the smoothed signal

    :Examples:

    >> t=np.linspace(-2,2,100)
    >> x=np.sin(t)+randn(len(t))*0.1
    >> y=smooth(x,11)

    see also:

    scipy.signal.get_window, np.convolve, scipy.signal.lfilter
    """
    def smooth1d(x, win):
        window_len = win.size
        s = np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
        y = np.convolve(win/win.sum(), s, mode='valid')
        start = int(np.ceil(window_len/2.)-1)
        stop = start+len(x)
        return y[start:stop]

    window_len = int(window_len)
    if window_len < 3:
        return x
    if window == 'flat': # backwards compatibility
        window = 'boxcar'
    if isinstance(window, str) or isinstance(window, tuple):
        if window_len%2 == 0: # Use odd-length window to avoid shifting data
            window_len += 1
        win = scipy.signal.get_window(window, window_len, fftbins=False)
    else:
        win = np.asarray(window)
        if len(win.shape) != 1:
            raise ValueError('window must be 1-D')
    if x.shape[axis] < win.size:
        raise ValueError("Input vector needs to be bigger than window length "
                         "of {}".format(win.size))

    return np.apply_along_axis(smooth1d, axis, x, win)

class UncertainRBF(object):
    r"""
    A class for radial basis function fitting of n-dimensional uncertain scattered data

    Parameters:

    :param \*args: arrays `x, y, z, ..., d, e` where
                 `x, y, z, ...` are the coordinates of the nodes
                 `d` is the array of values at the nodes, and
                 `e` is the standard deviation error of the values at the nodes

    :param centers: None the RBFs are centered on the input data points (can be very expensive for large number of nodes points)
                    -N: N nodes randomly distributed in the domain
                    N: N*N nodes uniformly distributed in the domain
                    np.array(N,X): user-defined array with X coordinates of the N nodes

    :param epsilon: float Adjustable constant for gaussian - defaults to approximate average distance between nodes

    :param function: 'multiquadric': np.sqrt((r / self.epsilon) ** 2 + 1)   #<--- default
                     'inverse': 1.0 / np.sqrt((r / self.epsilon) ** 2 + 1)
                     'gaussian': np.exp(-(r**2 / self.epsilon))
                     'linear': r
                     'cubic': r ** 3
                     'quintic': r ** 5
                     'thin_plate': r ** 2 * np.log(r)

    :param norm: default "distance" is the euclidean norm (2-norm)

    :Examples:

    >>> x=np.linspace(0,1,21)
    >>> y=np.linspace(0,1,21)
    >>> e=x*0+.5
    >>> e[abs(x-0.5)<0.1]=8
    >>> y[abs(x-0.5)<0.1]=10
    >>>
    >>> x1 = np.linspace(0,1,100)
    >>> y1 = UncertainRBF(x, y, e, centers=None, epsilon=1)(x1)
    >>> y0 = UncertainRBF(x, y, e*0+1, centers=None, epsilon=1)(x1)
    >>>
    >>> plt.subplot(2,2,1)
    >>> errorbar(x,y,e,ls='',marker='.',label='raw 1D data')
    >>> uband(x1,y1,label='1D RBF w/ uncertainty')
    >>> plt.plot(x1,nominal_values(y0),label='1D RBF w/o uncertainty')
    >>> plt.title('1D')
    >>> legend(loc=0)
    >>>
    >>> x = np.random.rand(1000)*4.0-2.0
    >>> y = np.random.rand(1000)*4.0-2.0
    >>> e = np.random.rand(1000)
    >>> z = x*np.exp(-x**2-y**2)
    >>> ti = np.linspace(-2.0, 2.0, 100)
    >>> XI, YI = np.meshgrid(ti, ti)
    >>>
    >>> rbf = UncertainRBF(x, y, z+e, abs(e), centers=5, epsilon=1)
    >>> ZI = nominal_values(rbf(XI, YI))
    >>>
    >>> rbf = UncertainRBF(x, y, z+e, abs(e)*0+1, centers=5, epsilon=1)
    >>> ZC = nominal_values(rbf(XI, YI))
    >>>
    >>> plt.subplot(2,2,3)
    >>> plt.scatter(x, y, c=z, s=100, edgecolor='none')
    >>> plt.xlim(-2, 2)
    >>> plt.ylim(-2, 2)
    >>> plt.colorbar()
    >>> plt.title('raw 2D data (w/o noise)')
    >>>
    >>> plt.subplot(2,2,2)
    >>> plt.pcolor(XI, YI, ZI)
    >>> plt.xlim(-2, 2)
    >>> plt.ylim(-2, 2)
    >>> plt.colorbar()
    >>> plt.title('2D RBF w/ uncertainty')
    >>>
    >>> plt.subplot(2,2,4)
    >>> plt.pcolor(XI, YI, ZC)
    >>> plt.xlim(-2, 2)
    >>> plt.ylim(-2, 2)
    >>> plt.colorbar()
    >>> plt.title('2D RBF w/o uncertainty')
    """

    def __init__(self, x, d, e, centers=None, function='multiquadric', epsilon=None, norm=None):
        #import pdb; pdb.set_trace()
        if not np.all([xx.shape == d.shape for xx in x]) and (e.shape == d.shape):
            raise ValueError("Array lengths must be equal")

        # npts by ndim
        self.xi = np.asarray([np.asarray(a, dtype=np.float_).flatten() for a in x]).T
        self.di = np.atleast_2d(np.asarray(d).flatten()).T
        self.ei = np.atleast_2d(np.asarray(e).flatten()).T

        self.indim = self.xi.shape[1]
        self.outdim = self.di.shape[1]

        if centers is None:
            self.centers=self.xi
        elif np.iterable(centers):
            self.centers=centers
        elif centers>0:
            self.centers=np.array([x.flatten() for x in meshgrid(*[np.linspace(min(self.xi[:,k]),max(self.xi[:,k]),centers) for k in  range(self.indim)])]).T
        else:
            self.centers=np.array([np.random.uniform(min(self.xi[i]), max(self.xi[i]), self.indim) for i in range(abs(centers))])
        self.numCenters = self.centers.shape[0]
        self.centers = self.centers.T # ndim by numcenters

        # default "distance" is the euclidean norm (2-norm)
        if norm is None:
            self.norm = lambda ctrs, pts: np.linalg.norm(ctrs[:,np.newaxis,...] - pts[...,np.newaxis], axis=0)
        else:
            self.norm = norm

        self.epsilon = epsilon
        if self.epsilon is None:
            # default epsilon is the "the average distance between nodes" based on a bounding hypercube
            dim = self.xi.shape[0]
            ximax = np.max(self.xi, axis=1)
            ximin = np.min(self.xi, axis=1)
            edges = ximax-ximin
            edges = edges[np.nonzero(edges)]
            if not len(edges):
                self.epsilon = (np.max(self.xi)-np.min(self.xi))/self.xi.size
            else:
                self.epsilon = 1./np.power(np.prod(edges)/self.indim, 1.0/edges.size)

        # function options
        function_options = {'multiquadric': lambda r: np.sqrt((r / self.epsilon) ** 2 + 1),
                            'inverse': lambda r: 1.0 / np.sqrt((r / self.epsilon) ** 2 + 1),
                            'gaussian': lambda r: np.exp(-(r**2 / self.epsilon)),
                            'linear': lambda r: r,
                            'cubic': lambda r: r ** 3,
                            'quintic': lambda r: r ** 5,
                            'thin_plate': lambda r: r ** 2 * np.log(r)}
        if isinstance(function, str):
            self.fun = function_options[function]  # demands it be one of the defined keys
        else:
            self.fun = function

        G = self._calcAct(self.xi.T)
        import pdb; pdb.set_trace()
        D = np.linalg.pinv(np.matrix(G/self.ei[np.newaxis,:]))
        self.Wd = np.dot(D, self.di/self.ei)
        self.We = np.dot(D, self.ei*0+1.)

    def _calcAct(self, X):
        """
        Calculate the radial basis function of "radius" between points X and the centers.

        :param X: np.ndarray. Shape (ndims, npts)

        :return: np.ndarray. Shape (npts, numcenters)
        """
        return self.fun(self.norm(self.centers, X))

    def __call__(self, *args):
        r"""
        evaluate interpolation at coordinates

        :param \*args: arrays `x, y, z, ...` the coordinates of the nodes

        :return: uncertain array, of the same shape of coordinates
        """
        args = [np.asarray(x) for x in args]
        if not np.all([x.shape == y.shape for x in args for y in args]):
            raise ValueError("Array lengths must be equal")
        shp = args[0].shape
        X = np.asarray([a.flatten() for a in args], dtype=np.float_) # ndim, npts
        G = self._calcAct(X)

        return uarray(np.array(np.dot(G, self.Wd)),np.array(abs(np.dot(G, self.We)))).reshape(shp)

def fit_rbf(x0, t0, y0, e0,
            x_out, t_out,
            xbias_power=1,
            xt_correlation_ratio=10,
            centers_per_slice=10,
            interp_and_smooth=True,
            function='gaussian',
            debug=False,
            symmetry=0,
            **kwargs):
    """
    Function wrapper for an uncertainties RBF that can bias the x axis based on the smoothed y derivative
    and/or a simple power scaling.

    This defines the key word arguments that will be accepted from the GUI.

    :param x0: ndarray. Radial locations of input profile data.
    :param t0: ndarray. Times of input profile data.
    :param z0: ndarray. Values of input profile data.
    :param e0: ndarray. Errors of input profile data.
    :param x_out: ndarray. Output radial grid.
    :param t_out: ndarray. Output time grid.
    :param xbias_power: float. Fit in 0.5 * (x + x ** (1 + bias) space to allow fine structure at large x.
    :param xt_correlation_ratio: Fit in normalized (x, t) domain of 1-by-ratio.
    :param centers_per_slice: integer. Number of basis functions per unique t.
    :param interp_and_smooth: bool. Modify x based on smoothed y derivative before fitting.
    :param function: string. Basis function type passed to UncertaintiesRBF.
    :param debug: bool. Plot intermediate steps.
    :return: ndarray. The fit values on a regular grid t_out by x_out.

    """

    # basic variables
    x_out = np.atleast_1d(x_out)
    #
    #import pdb; pdb.set_trace()
    t0 = np.tile(t0,(x0.shape[0],1))
    
    x0 = x0.T
    t0 = t0.T
    y0 = y0.T
    e0 = e0.T
    #import pdb; pdb.set_trace()
    t_out = np.atleast_1d(t_out)
    missing_times = []

    if symmetry:
        index=(x0>=0)
        x0=x0[index]
        t0=t0[index]
        y0=y0[index]
        e0=e0[index]

    # simple bias of structure to core/ edge
    if not interp_and_smooth:  # bias all times similarly
        x_in_mod = 0.5 * (x0 + x0 ** (1 + xbias_power))
        x_out_mod = 0.5 * (x_out + x_out ** (1 + xbias_power))
        xmin, xmax = np.min(x_out_mod), np.max(x_out_mod)
    # bias the x-axis of each time by its derivative, fit a ~line in x
    else:
        x_in_mod = x0 * 1.0
        x_out_mod = x_out * 1.0
        x_out_correction = np.ones((t_out.shape[0], x_out.shape[0]))
        smooth_window = int(5 * 30. / centers_per_slice) * 2 + 1
        grid = np.linspace(np.nanmin(x0), np.nanmax(x0), 200)
        for itime, time in enumerate(t_out):
            i = np.where(t0 == time)[0]
            if len(i)==0:
                # Keep track of time indices that are missing data, and assign `nan`
                missing_times.append(itime)
                continue
            #import pdb; pdb.set_trace()
            x0_, y0_ = np.ndarray.flatten(x0[i]), np.ndarray.flatten(y0[i])
            j = np.argsort(x0_)
            # stretch x where |dy/dx| is large
            y_grid = interp1e(x0_[j], y0_[j])(grid)
            xsign = np.sign(grid)
            xsign[xsign == 0] = 1
            dys = np.cumsum(smooth(xsign * np.abs(np.gradient(y_grid)), smooth_window))
            # normalize
            dys = 1 + (dys - min(dys)) / (max(dys) - min(dys))
            dys /= interp1e(grid, dys)(1)
            # extra edge bias
            dys = dys ** (1. + np.sqrt(xbias_power))
            x_in_mod[i] *= interp1e(grid, dys)(x0[i])
            x_out_correction[itime] = interp1e(grid, dys)(x_out)
        xmin, xmax = np.min(x_out_correction * x_out), np.max(x_out_correction * x_out)

    # fit inside a 1 by 1 box stretched by the time correlation factor
    xnorm = 1 / (xmax - xmin)
    tnorm = xt_correlation_ratio / np.ptp(t0)
    if np.ptp(t0) == 0:
        tnorm = 1.0  # if only one time
    t0 *= tnorm
    x_in_mod *= xnorm
    x_out_mod *= xnorm

    if symmetry:
        x_in_mod=hstack((-x_in_mod[::-1],x_in_mod))
        t0=hstack((t0[::-1],t0))
        y0=hstack((symmetry*y0[::-1],y0))
        e0=hstack((e0[::-1],e0))
        centers_per_slice*=2

    # automatic distribution of basis function centers evenly in x at every time
#    import pdb; pdb.set_trace()
    centers = np.linspace(np.amin(x_in_mod), np.amax(x_in_mod), centers_per_slice)
    eps = centers[1] - centers[0]
    centers = np.array([x.flatten() for x in np.meshgrid(np.unique(t0)[::], centers)]).T

    # make the rbf and interpolate
    rbf = UncertainRBF((t0, x_in_mod), y0, e0, epsilon=eps, centers=centers, function=function)
    xgrid, tgrid = np.meshgrid(x_out_mod, tnorm * t_out)
    if interp_and_smooth:
        xgrid = xnorm * x_out_correction * x_out
    y_out = rbf(tgrid, xgrid)

    # Assing nan to missing times
    for itime in missing_times:
        y_out[itime,:] = np.nan

    # debugging check to see what actually got fit
    if debug:
        f, ax = subplots()
        sc = scatter(x_in_mod, y0, c=t0)
        sc = scatter(xgrid, nominal_values(y_out), c=tgrid, marker='x')
        cb = f.colorbar(sc)
        ax.set_xlabel('Fit-space radial axis')
        ax.set_ylabel('Fit-space values')
        cb.set_label('Fit-space time')

    return y_out


def fit_biasedrbf(x0, t0, y0, e0, x_out, t_out, xbias_position=0.98, xbias_width=0.05, xbias_power=1, xt_correlation_ratio=10,
            centers_per_slice=10, function='gaussian', debug=False):
    """
    Function wrapper for an uncertainties RBF that fits on a biased x axis given by
    x + power*atan((x - position)/width). This enables finer structure near the bias position.

    This defines the key word arguments that will be accepted from the GUI.

    :param x0: ndarray. Radial locations of input profile data.
    :param t0: ndarray. Times of input profile data.
    :param z0: ndarray. Values of input profile data.
    :param e0: ndarray. Errors of input profile data.
    :param x_out: ndarray. Output radial grid.
    :param t_out: ndarray. Output time grid.
    :param xbias_position: float. Fit done on x + power*atan((x - position)/width)
    :param xbias_width: float. Fit done on x + power*atan((x - position)/width)
    :param xbias_power: float. Fit done on x + power*atan((x - position)/width)
    :param xt_correlation_ratio: (x,t) space is normalized to a 1-by-ratio box.
    :param centers_per_slice: Number of basis functions per unique t value.
    :param function: string. Radial basis function passed to UncertainRBF.
    :param debug: bool. Plot intermediate steps.
    :return: ndarray. The values of the fit on (x_out, t_out) grid.
    """

    # x bias allowing localized structure
    xnorm = 1 / np.ptp(x_out)

    def bias_function(x, xnorm=xnorm):
        positive_bias = xbias_power * np.arctan((x - xbias_position) / xbias_width) * 2 / np.pi
        return x * xnorm + positive_bias

    x_out_mod = bias_function(x_out)
    x_in_mod = bias_function(x0)

    # fit inside a 1 by 1 box stretched by the time correlation factor
    xnorm = 1 / np.ptp(x_out_mod)
    tnorm = xt_correlation_ratio / np.ptp(t0)
    if np.ptp(t0) == 0:
        tnorm = 1.0  # if only one time
    t0 *= tnorm
    x_in_mod *= xnorm
    x_out_mod *= xnorm

    # automatic distribution of basis function centers evenly in x at every time
    centers = np.linspace(np.min(x_out_mod), np.max(x_out_mod), centers_per_slice)
    eps = centers[1] - centers[0]
    centers = array([x.flatten() for x in meshgrid(np.unique(t0)[::], centers)]).T

    # make the rbf and interpolate
    rbf = UncertainRBF((t0, x_in_mod), y0, e0, epsilon=eps, centers=centers, function=function)
    xgrid, tgrid = meshgrid(x_out_mod, tnorm * t_out)
    y_out = rbf(tgrid, xgrid)

    # debugging check to see what actually got fit
    if debug:
        f, ax = subplots()
        sc = scatter(x_in_mod, y0, c=t0)
        sc = scatter(xgrid, nominal_values(yy), c=tgrid, marker='x')
        cb = f.colorbar(sc)
        ax.set_xlabel('Fit-space radial axis')
        ax.set_ylabel('Fit-space values')
        cb.set_label('Fit-space time')

    return y_out


def GAPSpline(x_fit, x, y, ye, set_autoknot=True, num_knots=3, knot_locs=None,
             set_onAxis_value=False, onAxis_value='free', set_Boundary=False, boundary_value='free',
             set_edgeSlope=False, edgeSlope_value='free', set_fitEdge='fit', set_allowneg=True, fill_value=np.nan):
    """
    Function wrapper that runs a remote IDL season GAprofiles spline fit and returns a modified interpolation
    object with methods to get the basic spline info.

    :param x_fit: ndarray. Grid on which to return the spline fit.
    :param x: ndarray. Radial locations of input data in rho.
    :param y: ndarray. Input profile data values.
    :param ye: ndarray. Error of input values.
    :param set_autoknot: bool. Use the GAprofiles auto-knot feature to adjust knot locations.
    :param num_knots: integer. The number of knot locations to use.
    :param knot_locs: ndarray. Specific knot locations to use.
    :param set_onAxis_value: bool. Require the on axis value to be onAxis_value.
    :param onAxis_value: float. Value the axis is constrained to if set_onAxis_value is true.
    :param set_Boundary: bool. Require to boundary value to be boundary_value.
    :param boundary_value: float. Value the boundary is required to be is set_Boundary is true.
    :param set_edgeSlope: bool. Require the edge slope to be edgeSlope_value.
    :param edgeSlope_value: float. Restrict the edge slope to this value if set_edgeSlope is true.
    :param set_fitEdge: bool. Include data beyond rho = 1 in the fit.
    :param set_allowneg: bool. Include negative data in the fit.
    :param fill_value: bool. Value used when interpolating the GAprofiles spline beyond the x_out domain.
    :return: Class that wraps the GAprofiles spline fitter in a 1D interpolation object.

    """
    if set_autoknot:
        print('   Using autoknotting')
        knot_locs = None
    else:
        print('   Set knot locations')
        num_knots = len(knot_locs)

    if set_onAxis_value:
        if onAxis_value == 'T_e':
            # Constrain on axis value to match electron temperature fit at the axis.
            # This only makes sense when fitting an ion profile.
            print('   Using time-dependent fit constraint')
            print('   Constraining onAxis value of current fit using result from fitting T_e.')
            print('   If you are not fitting another temperature, then this DOES NOT make sense. Rethink.')
            try:
                T_e_onAxis = root['OUTPUTS']['FIT']['T_e'][:, 0]
            except Exception:
                print('   Time dependent boundary constraint could not be resolved. Freeing axis value.')
                onAxis_value = 'free'
            else:
                # we will have to extract the nominal values of the data later, but for now, we should keep the set
                # complete so we can use its time base to make sure we have the right slice.
                onAxis_value = T_e_onAxis
    else:
        onAxis_value = 'free'

    if set_fitEdge:
        fitEdge = 'fit'
        boundary_value = 'free'
        edgeSlope_value = 'free'
    else:
        fitEdge = 'free'
        if set_Boundary:
            boundary_value = boundary_value
        else:
            boundary_value = 'free'
        if set_edgeSlope:
            edgeSlope_value = edgeSlope_value
        else:
            edgeSlope_value = 'free'

    # normalize values
    ymax = np.max(abs(y))
    y /= ymax
    ye /= ymax
    if is_numeric(edgeSlope_value):
        edgeSlope_value /=ymax
    if is_numeric(onAxis_value):
        onAxis_value /=ymax
    if is_numeric(boundary_value):
        boundary_value /= ymax

    # Support venus iris saturn. Fallback on venus
    serverPicker=root['SETTINGS']['REMOTE_SETUP']['serverPicker']
    if not is_server(serverPicker,['venus','iris','saturn']):
        serverPicker='venus'

    # Write the data to text file
    scratch['spl_mod_input'] = OMFITascii('spl_mod_input.dat')
    with open(scratch['spl_mod_input'].filename,'w') as f:
        for i in range(len(x)):
            f.write('{} {} {}\n'.format(x[i],y[i],ye[i]))

    # Define IDL text
    runName = 'idl_input.pro'
    idl_cmd = []
    idl_cmd.append("""!PATH=''\n""")
    if is_server(serverPicker,'venus'):
        idl_cmd.append("""!PATH = !PATH + ':' + expand_path("+/usr/local/rsi/idl/lib")\n""")
        idl_cmd.append("""!PATH = !PATH + ':' + expand_path("+/usr/local/rsi/idl/examples")\n""")
        idl_cmd.append("""!PATH = !PATH + ':' + expand_path("+/link/idl")\n""")
        idl_cmd.append("ospath = GETENV('OSPATH')\n")
        idl_cmd.append("""!PATH = !PATH + ':' + expand_path("/f/mdsplus/"+ospath+"/idl")\n""")
    elif is_server(serverPicker,['iris','saturn']):
        idl_cmd.append("""!PATH = !PATH + ':' + expand_path("+/fusion/usc/src/4dlib/FITTING")\n""")
        idl_cmd.append("""!PATH = !PATH + ':' + expand_path("+/fusion/usc/link/idl")\n""")
    idl_cmd.append('data = FLTARR(3,{})\n'.format(len(x)))
    idl_cmd.append("OPENR,lun,'spl_mod_input.dat',/GET_LUN\n")
    idl_cmd.append('READF,lun,data\n')
    idl_cmd.append('x = data[0,*]\n')
    idl_cmd.append('y = data[1,*]\n')
    idl_cmd.append('ye = data[2,*]\n')
    idl_cmd.append('knotloc=FLTARR(20)\n')
    if (set_autoknot == False):
        for ii in range(0,num_knots):
            cmdstr = 'knotloc[%d] = %f\n' % (ii, knot_locs[ii])
            idl_cmd.append(cmdstr)
    idl_cmd.append("model_interp = 'spline'\n")

    cmdstr = "model = {"
    if set_autoknot:
        cmdstr += "AUTOKNOT:'yes', "
    else:
        cmdstr += "AUTOKNOT:'no', "
    cmdstr += "NUMKNOT:%d, " % num_knots
    cmdstr += "KNOTLOC:knotloc, TYPE:'point', AXIS:%r, BOUNDARY:%r, " % (onAxis_value, boundary_value)
    cmdstr += "EDGESLOPE:%r, EDGE:%r, X_AXIS:'rho', " % (edgeSlope_value, fitEdge)
    cmdstr += "NORMALIZE:'no', FTOL:0.01, MAXIT:2000}\n"
    idl_cmd.append(cmdstr)
    if set_allowneg:
        idl_cmd.append('f = SPL_MOD(x,y,ye,MODEL=model,/NEG)\n')
    else:
        idl_cmd.append('f = SPL_MOD(x,y,ye,MODEL=model)\n')

    idl_cmd.append("SAVE,f,FILENAME='spl_mod_f.sav'\n")
    idl_cmd.append('exit\n')

    # Define executable variables for remote execution
    runScript    = OMFITascii(runName,fromString=''.join(idl_cmd))
    server       = SERVER[serverPicker]['server']
    tunnel       = SERVER[serverPicker]['tunnel']
    remotedir    = OMFITworkDir(root,SERVER[serverPicker]['server'])+'/spl_mod/'
    workdir      = root['SETTINGS']['SETUP']['workDir']+'/spl_mod/'

    # Execute the fit
    OMFITx.executable(root,
                      inputs=[runScript,scratch['spl_mod_input']],
                      outputs=['spl_mod_f.sav'],
                      executable=SERVER[serverPicker]['idl']+' '+runName+';ls *',
                      workdir=workdir,
                      server=server,
                      tunnel=tunnel,
                      remotedir=remotedir)

    # Save result to the OMFIT tree
    scratch['spl_mod_f'] = OMFITidlSav('./spl_mod_f.sav')

    x_spl = np.array(scratch['spl_mod_f']['f']['XFIT'])
    y_spl = np.array(scratch['spl_mod_f']['f']['YFIT']) * ymax
    # todo: switch the commented for class
    #interp1d.__init__(self, x_spl, y_spl, bounds_error=False, fill_value=nan, assume_sorted=True)
    self = interp1d(x_spl, y_spl, bounds_error=False, fill_value=fill_value, assume_sorted=True)

    self._redchi = scratch['spl_mod_f']['f']['REDCHISQ']
    self._knot_locs = np.array(scratch['spl_mod_f']['f']['KNOTLOC'])
    self._knot_vals = np.array(scratch['spl_mod_f']['f']['KNOTVAL'])


    # todo: unindent these 1 level for class
    def __call__(self, x):
        """
        Interpolate GAprofiles spline output to arbitrary locations.

        :param x: Locations of desired values.
        :return: uarray. Interpolated y values at x.
        """
        y = interp1d.__call__(self,x)
        return unumpy.uarray(y, y*0)

    def get_knots(self):
        """
        :return: knot locations used in GAprofiles spline.
        """
        return self._knot_locs * 1.0

    def get_knot_values(self):
        """
        :return: knot values used in GAprofiles spline.
        """
        return self._knot_vals * 1.0

    # hack because no classes allowed
    self.__cal__ = types.MethodType(__call__, self)
    self.get_knots = types.MethodType(get_knots, self)
    self.get_knot_values = types.MethodType(get_knot_values, self)
    return self


def ytransform(z, transform=1, reverse=False):
    """
    Modify the measured data prior to fitting, and then the fit prior to saving.

    :param z: original y data or previously transformed data
    :param transform: int/float/str. Choose from log, symlog, hyperbola, or any power transform.
    :param reverse: reverse the transformation (from z to y)
    :return: ndarray.

    """
    try:
        if transform == 'log':
            if reverse:
                z1 = 10 ** z
            else:
                z1 = unumpy.log10(z)
        elif transform == 'symlog':
            if reverse:
                z1 = unumpy.arcsinh(z)
            else:
                z1 = unumpy.sinh(z)
        elif transform == 'hyperbola':
            if reverse:
                z1 = (z + unumpy.sqrt(z ** 2 + 4)) / 2.
            else:
                z1 = z - 1 / z
        else:
            if reverse:
                z1 = z ** (1. / transform)
            else:
                z1 = z ** transform
    except ValueError:
        direction, source = 'forward', 'data'
        if reverse:
            direction, source = 'reverse', 'fit'
        raise OMFITexception('Failed to {:} {:} transform {:}. Consider changing the transform.'.format(direction, transform, source))
    return z1


def add_long_names(fit_dataset, overwrite=False):
    """
    Adds some standard long_name conventions to a fit dataset.
    The long names are generally informative and are used by this modules plotting scripts.

    :param fit_dataset: Dataset. The fits.
    :return: Dataset. Same dataset with some additional attirbutes assigned to DataArrays

    """
    # collection of common "measurements" and their long names
    keymap = {'n_e': 'Electron Density',
              'T_e': 'Electron Temperature',
              'n_12C6': 'Carbon Density',
              'T_12C6': 'Carbon Temperature',
              'omega_tor_12C6': 'Carbon Toroidal Rotation',
              'V_pol_12C6': 'Carbon Poloidal Velocity',
              'P_rad': 'Radiated Power',
              'dn_e_drho': r'$\frac{d n_e}{d \rho}$',
              'dn_12C6_drho': r'$\frac{d n_12C6}{d \rho}$',
              'dT_e_drho': r'$\frac{d T_e}{d \rho}$',
              'dT_12C6_drho': r'$\frac{d T_12C6}{d \rho}$',
              'domega_tor_12C6_drho': r'$\frac{d \omega_{\phi,12C6}}{d \rho}$',
              'dV_pol_12C6_drho': r'$\frac{d V_{\theta,12C6}}{d \rho}$',
              'dP_rad_drho': r'$\frac{d P_{rad}}{d \rho}$',
              }
    # time derivative follow rho derivative convention
    for key, val in list(keymap.items()):
        if key.endswith('_drho'):
            keymap[key.replace('_drho', '_dt')] = val.replace(r'd \rho', 'd t')

    # make a copy of the original dataset if not doing this in place
    ds = copy.deepcopy(fit_dataset)

    # add the long names to everything we have in our common collection
    for key in ds.data_vars:
        if key in keymap:
            if 'long_name' not in ds[key].attrs or overwrite:
                ds[key].attrs['long_name'] = keymap[key]
        elif 'long_name' not in ds[key].attrs:
            printd(" > Do not have a long name for " + key)

    return ds
