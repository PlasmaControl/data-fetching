# file processed by 2to3
from __future__ import print_function, absolute_import

import sys

if 'omfit_tree' in sys.modules:
    print('Loading fit utility functions...')

try:
    # framework is running
    from .startup_choice import *
except (ValueError, SystemError):  # catch error in Python2.x
    # class is imported by itself
    from startup_choice import *
except ImportError as _excp:  # catch error in Python3.x
    # class is imported by itself
    if 'attempted relative import with no known parent package' in str(_excp):
        from startup_choice import *
except ModuleNotFoundError:  # catch error in Python3.x
    # class is imported by itself
    from startup_choice import *

from classes.utils_base import *
from classes.utils_base import _available_to_user_math
from classes.utils_math import *
import numpy as np


# -----------
# profiles fitting
# -----------
@_available_to_user_math
def autoknot(x, y, x0, evaluate=False, minDist=None, s=3, w=None, allKnots=False, userFunc=None, *args, **kwargs):
    """
    This function returns the optimal location of the inner-knots for a nth degree spline interpolation of y=f(x)

    :param x: input x array

    :param y: input y array

    :param x0: initial knots distribution (list) or number of knots (integer)

    :param s: order of the spline

    :param w: input weights array

    :param allKnots: returns all knots or only the central ones exluding the extremes

    :param userFunc: autoknot with user defined function with signature `y0=userFunc(x,y)(x0)`

    :param minDist: a number between >0 and infinity (though usually <1), which sets the minimum distance between knots.
                    If small knots will be allowed to be close to one another, if large knots will be equispaced.
                    Use `None` to automatically determine this parameter based on: `0.5/len(knots)`
                    If minDist is a string than it will be evaluated (the knots locations in the string can be accessed as `knots`).

    :return: x0 optimal location of inner knots to spline interpolate y=f(x)
            f1=interpolate.LSQUnivariateSpline(x,y,x0,k=s,w=w)
    """

    def _func_arbitrary(xd,s):
        x0=numpy.cumsum([0]+xd.tolist())
        x0=x0/max(x0)
        if x0[1]<ux[1] or x0[-2]>ux[-2]:
           x0=(x0[1:-1]-x0[1])/(x0[-2]-x0[1])*(ux[-2]-ux[1])+ux[1]
           x0=[ux[0]]+x0.tolist()+[ux[-1]]
        x0=numpy.array(x0)
        x0=x0/max(x0)

        y0=userFunc(x,y)(x0)
        def cost(y0,func,w):
            y1=func(x0,y0)(x)
            return numpy.sqrt(numpy.mean((abs(y - y1) * w) ** 2))
        bounds=numpy.reshape([min(y),max(y)]*len(y0),(-1,2))
        y0=optimize.fmin_tnc(cost, y0, args=(userFunc,w), bounds=bounds, approx_grad=True, pgtol=1E-1)[0]
        y1=userFunc(x0,y0)(x)
        return y1,y0,x0

    def _func(xd,s):
        x0=numpy.cumsum([0]+xd.tolist())
        x0=x0/max(x0)
        if x0[1]<ux[1] or x0[-2]>ux[-2]:
           x0=(x0[1:-1]-x0[1])/(x0[-2]-x0[1])*(ux[-2]-ux[1])+ux[1]
           x0=[ux[0]]+x0.tolist()+[ux[-1]]
        x0=numpy.array(x0)
        x0=x0/max(x0)

        try:
            f1=interpolate.LSQUnivariateSpline(x,y,x0[1:-1],k=s,w=w)
        except ValueError as _excp:
            raise OMFITexception(_excp)
        y0=f1(x0)
        y1=f1(x)
        return y1,y0,x0

    if userFunc is not None:
        func=_func_arbitrary
    else:
        func=_func

    def cost(xd,s,w):
        try:
            y1,y0,x0=_func(xd,s)
        except ValueError:
            return numpy.inf
        c = numpy.sqrt(numpy.mean((abs(y - y1) * w) ** 2))
        if numpy.isnan(c):
            c=max(abs(y))
        return c

    if w is None: w=x*0+1

    xm=min(x)
    xM=max(x)
    ux=(numpy.unique(x)-xm)/(xM-xm)

    if is_int(x0):
        #initial guess
        if allKnots:
            n=x0-2
        else:
            n=x0
        x0=[]
        xt=numpy.linspace(numpy.unique(x)[1],numpy.unique(x)[-2], int(n*(3+numpy.log(n))) ).tolist()
        for kkk in range(n):
            x0.append(numpy.nan)
            started=True
            for xt_chosen in xt:
                x0[kkk]=xt_chosen
                try:
                    y1=interpolate.LSQUnivariateSpline(x,y,sorted(x0),k=s,w=w)(x)
                except ValueError as _excp:
                    raise OMFITexception(_excp)
                c = numpy.sqrt(numpy.mean((abs(y - y1) * w) ** 2))
                if started or c<mincost:
                    mincost=c
                    minxost_x=xt_chosen
                    started=False
            xt.pop(xt.index(minxost_x))
            x0[kkk]=minxost_x
        x0=[xm,xM]+x0
        x0=numpy.array(sorted(x0))

    x0=numpy.atleast_1d(x0).tolist()
    if min(x0)!=xm:
        x0=[xm]+x0
    if max(x0)!=xM:
        x0=x0+[xM]
    x0=numpy.atleast_1d(x0)

    x=(x-xm)/(xM-xm)
    x0=(x0-xm)/(xM-xm)
    x[-1]=x0[-1]=1
    x[0]=x0[0]=0

    x0=sorted(x0)
    xd=numpy.diff(x0)
    n=len(xd)

    #set minimum knots distance
    knots=x0
    if minDist is None:
        minDist=0.01*len(knots)
    elif isinstance(minDist,basestring):
        minDist=eval(minDist)

    bounds=numpy.reshape([minDist,1]*n,(-1,2))

    if evaluate:
        y1_,y0_,x0=func(xd,s)
        if allKnots:
            return y1_,y0_
        else:
            return y1_,y0_[1:-1]

    xd=optimize.fmin_l_bfgs_b(cost, xd, args=(s,w), bounds=bounds, approx_grad=True)[0]

    y1,y0,x0=func(xd,s)
    x0=x0*(xM-xm)+xm

    if allKnots:
        return x0,y0
    else:
        return x0[1:-1]

class knotted_fit_base(object):
    '''
    The base class for the types of fits that have free knots and locations
    '''
    kw_vars = ['min_slope', 'monotonic', 'min_dist', 'first_knot', 'knots', 'fixed_knots', 'fit_SOL', 'outliers']
    kw_defaults = [None, False, 0, None, 3, False, False, 3]

    def __init__(self, x, y, yerr):
        '''
        Does basic checking for x,y,yerr then stores them in
        self.x, self.y, self.yerr
        such that x is monotonically increasing
        '''
        valid_indices = ~numpy.isnan(x) & ~numpy.isnan(y) & ~numpy.isnan(yerr) & (yerr > 0)
        if not numpy.any(valid_indices):
            raise OMFITexception('No valid data passed to '+self.__class__.__name__)
        sort_index = numpy.argsort(x[valid_indices])
        self.x_orig = self.x = x[valid_indices][sort_index]
        self.y_orig = self.y = y[valid_indices][sort_index]
        self.e_orig = self.e = yerr[valid_indices][sort_index]
        self.all_one_sign = not (min(self.y) < 0 and max(self.y) > 0)

    def fit_knot_range(self, knots_range, **kw):
        r"""
        Try a range of number of knots
        Stop the loop if the current number of knots did no better than the previous best

        :param knots_range: A tuple, ``(min_num_knots, max_num_knots)`` passed to ``range``
        :param \**kw: The keywords passed to ``fit_single``
        """
        redchi = 1e4
        tag = None
        for i in range(*knots_range):
            kw['knots'] = i
            self.fit_single(**kw)
            best_tag, best_redchi = self.get_best_fit()
            if best_redchi == redchi:
                return
            else:
                redchi = best_redchi
                tag = best_tag

    def get_tag(self, **kw):
        r"""
        Get the tag for the settings given by ``**kw``

        :param \**kw: The Fit Keywords documented in the ``__init__`` method
        """
        tag = []
        for default, var in zip(self.kw_defaults, self.kw_vars):
            val = kw.get(var, self.kw_orig[var])
            if numpy.array(val != self.kw_orig[var]).any():
                tag.append('%s=%s' % (var, val))
        if tag:
            return ';'.join(tag)
        return 'default'

    def fit_single(self, **keyw):
        r"""
        Perform a single fit for the given ``**keyw``

        :param \**keyw: The Fit Keywords documented in the ``__init__`` method

        :return: The ``lmfit.MinimizerResult`` instance for the fit

        The fit is also stored in ``self.fits[self.get_tag(**keyw)]``
        """
        kw = copy.deepcopy(self.kw_orig)
        kw.update(keyw)
        outliers = kw['outliers']

        # For variable knots, get a guess on the knot values from the fixed_knot solution
        if not kw['fixed_knots']:
            kw['fixed_knots'] = True
            kw['outliers'] = 0
            self.fit_single(**kw)
            kw['fixed_knots'] = False

        # Determine whether to use the scrape off layer points
        ind_valid = self.x_orig >= 0
        if not kw['fit_SOL']:
            ind_valid = ind_valid & (self.x_orig <= 1)

        # Always start with original data
        self.restore_orig_data()

        # Do the first fit with all data
        if outliers and outliers > 0:
            kw['outliers'] = 0
            params = self.build_params(**kw)
            tag_no_outliers = self.get_tag(**kw)
            self.fit_single(**kw)
            kw['outliers'] = outliers
            self.restore_orig_data()
            diff = self.residual(self.fits[tag_no_outliers].params)
            ind_valid = ind_valid & (diff <= outliers) | (self.x_orig == 0)

        # Only fit the valid data (stored in self.x, self.y, self.e)
        self.x = self.x_orig[ind_valid]
        self.y = self.y_orig[ind_valid]
        self.e = self.e_orig[ind_valid]
        params = self.build_params(**kw)
        tag = self.get_tag(**kw)
        if tag in self.fits:
            return self.fits[tag]
        # Store the lmfit.MinimizerResult in self.fits, along with data, outliers, and kw
        # out = lmfit.minimize(self.residual,params,method='nelder')
        self.fits[tag] = lmfit.minimize(self.residual, params)
        self.fits[tag].x = self.x
        self.fits[tag].y = self.y
        self.fits[tag].e = self.e
        self.fits[tag].xo = self.x_orig[~ind_valid]
        self.fits[tag].yo = self.y_orig[~ind_valid]
        self.fits[tag].eo = self.e_orig[~ind_valid]
        self.fits[tag].kw = kw
        self.restore_orig_data()
        return self.fits[tag]

    def restore_orig_data(self):
        ''' Restore the original data'''
        self.x = self.x_orig
        self.y = self.y_orig
        self.e = self.e_orig

    def get_zk(self, params):
        '''
        Return the array of knot values from the ``lmfit.Parameters`` object
        '''
        return numpy.array([params['zk%d' % i].value for i in range(params['nk'].value)])

    def get_xk(self, params):
        '''
        Return the array of knot locations from the ``lmfit.Parameters`` object
        '''
        return numpy.array([params['xk%d' % i].value for i in range(params['nk'].value)])

    def valid_xk(self, params):
        '''
        :param params: An ``lmfit.Paramaters`` object to check

        :return: True if numpy.all(min_dist< xk_i-xk_(i-1)))
        '''
        dxk = numpy.diff(self.get_xk(params))
        return (dxk > min([params['dxk%d' % i].min for i in range(1, params['nk'].value)])).all()

    def residual(self, params):
        '''
        Return the array of numpy.sqrt(((ymodel-y)/yerr)**2) given the ``lmfit.Parameters`` instance

        Developers note: For the default data set tested, choosing the sqrt of the
        square did better than the signed residual
        '''
        ymodel = self.model(params, self.x)
        return numpy.sqrt((((ymodel - self.y) / self.e)) ** 2)

    def get_param_unc(self, lmfit_out):
        """
        Get parameters with correlated uncertainties

        :param lmfit_out: ``lmfit.MinimizerResult`` instance

        :return: tuple of xk, zk, ybc, xbc
        """
        result = {}
        for k in lmfit_out.params:
            result[k] = lmfit_out.params[k].value
        if lmfit_out.errorbars:
            corr_params = uncertainties.correlated_values([lmfit_out.params[k].value for k in lmfit_out.var_names], lmfit_out.covar)  # ,labels=lmfit_out.var_map)
            for vi, var in enumerate(lmfit_out.var_names):
                result[var] = corr_params[vi]
        nk = result['nk']
        xk = numpy.array([result['xk%d' % i] for i in range(nk)])
        zk = numpy.array([result['zk%d' % i] for i in range(nk)])
        ybc = result['ybc']
        xbc = result['xbc']
        return xk, zk, ybc, xbc

    def plot(self, **kw):
        r"""
        Plot all fits calculated so far, each in its own tab of a ``FigureNotebook``,
        where the tab is labeled by the shortened tag of the tag of the fit

        :param \**kw: Dictionary passed to self.plot_individual_fit

        :return: The ``FigureNotebook`` instance created
        """

        from omfit_plot import FigureNotebook
        fn = FigureNotebook(nfig=len(self.fits), labels=list(map(self.short_tag, sorted(self.fits.keys()))))
        for k, v in list(self.fits.items()):
            fg = fn.add_figure(label=self.short_tag(k))
            ax = fg.add_subplot(111)
            self.plot_individual_fit(k, ax=ax, **kw)
        fn.draw()
        return fn

    def short_tag(self, tag):
        '''Return a shortened version of the tag'''
        return tag.replace('knots', 'k').replace('fixed', 'f').replace('monotonic', 'm').replace('fit_', '')

    def plot_individual_fit(self, tag, ax=None, x=np.linspace(0, 1.1, 1001)):
        '''
        Plot a single fit, characterized by ``tag``

        :param tag: The tag of the fit that is to be plotted, must be in self.fits.keys()

        :param ax: The axes to plot into (one is created if ``None``)

        :param x: The x values to use for plotting the fitted curve
        '''
        from omfit_plot import uband
        if tag not in self.fits:
            printe('%s is not a valid tag' % tag)
            printe('Use one of:')
            printe('\n'.join(list(self.fits.keys())))
            return
        k = tag
        v = self.fits[k]
        if ax is None:
            import matplotlib.pyplot as plt
            ax = plt.gca()
        ax.set_title(r'%s: Reduced $\chi^2$=%g' % (self.short_tag(k), v.redchi))
        # Plot data
        ax.errorbar(v.x, v.y, v.e, linestyle='', label='Data')
        max_x = max(v.x) * 1.01
        # Plot outliers
        if len(v.xo):
            ax.errorbar(v.xo, v.yo, v.eo, color='m', linestyle='', label='Outlier')
            max_x = max([max(v.x), max(v.xo)]) * 1.01
        ax.set_xlim(left=0, right=max_x)
        # Plot fit
        uband(x, self.model(v.params, x, lmfit_out=v), zorder=100, label='Fit', ax=ax)

    def has_large_xerrorbars(self, lmfit_out):
        '''
        :param lmfit_out: A ``lmfit.MinimizerResult`` object
        :return: True if the errorbars of the knot locations are larger than the distance between knots
        '''
        v = lmfit_out
        if v.errorbars:
            xk = self.get_param_unc(v)[0]
            dxk = numpy.diff(nominal_values(xk))
            if (std_devs(xk[1:-1]) > dxk[:-1]).any():
                return True
            if (std_devs(xk[1:-1]) > dxk[1:]).any():
                return True
            return False
        return None

    def has_large_errorbars(self, lmfit_out, verbose=False):
        '''
        :param lmfit_out: A ``lmfit.MinimizerResult`` object
        :return: True if any of the following are False:

        1) the errorbars of the knot locations are smaller than the distance between the knots
        2) the errorbars in the fit at the data locations is not larger than the range in data
        3) the errorbars in the fit at the data locations is not larger than the fit value at that location, if the data are all of one sign
        '''
        v = lmfit_out
        if v.errorbars:
            if self.has_large_xerrorbars(v):
                if verbose:
                    printi('Fit has large xerrorbars')
                return True
            ymodel = self.model(v.params, v.x, v)
            if self.all_one_sign and (std_devs(ymodel) > abs(nominal_values(ymodel))).any():
                if verbose:
                    printi('Fit has errorbars larger than the fit itself:')
                return True
            if (std_devs(ymodel) > (abs(v.y).max() - abs(v.y).min())).any():
                if verbose:
                    printi('Fit has errorbars larger than the range in data')
                return True
            return False
        return None

    def get_best_fit(self, verbose=False, allow_no_errorbar=None):
        '''
        Figure out which is the best fit so far

        The best fit is characterized as being the fit with the lowest reduced
        chi^2 that is valid.  The definition of valid is

        1) the knots are in order
        2) the knots are at least min_dist apart
        3) the errorbars on the fit parameters were able to be determined
        4) the errorbars of the knot locations are smaller than the distance between the knots
        5) the errorbars in the fit at the data locations is not larger than the range in data
        6) the errorbars in the fit at the data locations is not larger than the fit value at that location

        :param verbose: If ``True``, print the tag and reduced chi2 of all fits
        :param allow_no_errorbar: If ``True``, if there is no valid fit found with
            errorbars, return the best fit without errorbars

        :return: A tuple of (best_tag, best_reduced_chi2)
        '''
        if allow_no_errorbar is None:
            allow_no_errorbar = self.allow_no_errorbar
        best_tag = None
        best_redchi = 1e4
        if verbose:
            printi('Reduced chi^2; fit type')
        for k, v in list(self.fits.items()):
            if verbose:
                printi(v.redchi, k)
            if not v.errorbars:
                if verbose:
                    printi('%s rejected because no errorbars' % k)
                continue
            if (v.redchi > best_redchi):
                if verbose:
                    printi('%s rejected because redchi too large' % k)
                continue
            xk = self.get_param_unc(v)[0]
            dxk = numpy.diff(nominal_values(xk))
            if v.kw['min_dist'] < 0 and not (dxk >= abs(v.kw['min_dist'])).all():
                if verbose:
                    printi('%s rejected because knots too close together' % k)
                continue
            if self.has_large_errorbars(v, verbose=verbose):
                if verbose:
                    printi('%s rejected because errorbars too large' % k)
                continue
            best_tag = k
            best_redchi = v.redchi
        if best_tag is None and allow_no_errorbar:
            if verbose:
                printi('Looking for best fit without errorbars')
            for k, v in list(self.fits.items()):
                if verbose:
                    printi(v.redchi, v.errorbars, k)
                if not v.errorbars and (v.redchi < best_redchi):
                    best_tag = k
                    best_redchi = v.redchi

        return best_tag, best_redchi

    def plot_best_fit(self, **kw):
        '''A convenience function for plotting the best fit'''
        best_tag, best_redchi = self.get_best_fit()
        if best_tag:
            self.plot_individual_fit(best_tag, **kw)
        else:
            printi('No valid fit:')
            self.get_best_fit(verbose=True)

    def __call__(self, x):
        '''
        Evaluate the best fit at ``x``
        '''
        best_tag = self.get_best_fit()[0]
        if best_tag:
            return self.model(self.fits[best_tag].params, x, self.fits[best_tag])
        return numpy.array([numpy.nan] * len(x))

class fitSL(knotted_fit_base):
    '''
    Fit a profile of data using integrated scale lengths, ideally obtaining
    uncertainties in the fitting parameters.

    Due to the nature of integrating scale lengths, this fitter is only good
    for data > 0 or data < 0.

    :Examples:

    >>> pkl = OMFITpickle(OMFITsrc+'/../samples/data_pickled.pkl')
    >>> x = pkl['x'][0,:]
    >>> y = pkl['y'][0,:]
    >>> yerr = pkl['e'][0,:]
    >>> fit = fitSL(x,y,yerr,fixed_knots=True,knots=-7,plot_best=True)

    Along the way of obtaining the fit with the desired parameters, other
    intermediate fits may be obtained.  These are stored in the ``fits``
    attribute (a ``dict``), the value of whose keys provide an indication of how
    the fit was obtained, relative to the starting fit.  For instance, to provide
    a variable knot fit, a fixed knot (equally spaced) fit is performed first.
    Also an initial fit is necessary to know if there are any outliers, and then
    the outliers can be detected.  The ``get_best_fit`` method is useful for
    determing which of all of the fits is the best, meaning the valid fit with
    the lowest reduced chi^2.  Here valid means

    1. the knots are in order
    2. the knots are at least min_dist apart
    3. the errorbars on the fit parameters were able to be determined
    4. the errorbars of the knot locations are smaller than the distance between
       the knots

    Note that 1) and 2) should be satisfied by using lmfit Parameter constraints,
    but it doesn't hurt to double check :-)

    Developer note: If the fitter is always failing to find the errorbars due to
    tolerance problems, there are some tolerance keywords that can be passed to
    ``lmfit.minimize``: ``xtol``, ``ftol``, ``gtol``
    that could be exposed.

    '''

    def __init__(self, x, y, yerr, knots=3, min_dist=0, first_knot=None,
                 fixed_knots=False, fit_SOL=False, monotonic=False, min_slope=None,
                 outliers=3, plot_best=False, allow_no_errorbar=False):
        '''
        Initialize the fitSL object, including calculating the first fit(s)

        :param x: The x values of the data

        :param y: The values of the data

        :param yerr: The errors of the data

        Fit Keywords:

        :param knots:
            * Positive integer: Use this number of knots as default (>=3)
            * Negative integer: Invoke the ``fit_knot_range`` method for the range (3,abs(knots))
            * list-like: Use this list as the starting point for the knot locations

        :param min_dist: The minimum distance between knot locations
            * min_dist > 0 (faster) enforced by construction
            * min_dist < 0 (much slower) enforced with lmfit

        :param first_knot: The first knot can be constrained to be above first_knot.
            The default is above ``min(x)+min_dist``  (The zeroth knot is at 0.)

        :param fixed_knots: If ``True``, do not allow the knot locations to change

        :param fit_SOL: If ``True``, include data points with x>1

        :param monotonic: If ``True``, only allow positive scale lengths

        :param min_slope: Constrain the scale lengths to be above ``min_slope``

        :param outliers: Do an initial fit, then throw out any points a factor
            of ``outliers`` standard deviations away from the fit

        Convenience Keywords:

        :param plot_best: Plot the best fit

        :param allow_no_errorbar: If ``True``, ``get_best_fit`` will return the
            best fit without errorbars if no valid fit with errorbars exists
        '''
        knotted_fit_base.__init__(self, x, y, yerr)
        y = self.y_orig
        if (numpy.array(y) < 0).any() and (numpy.array(y) > 0).any():
            raise OMFITexception('The scale length fitter is not appropriate for mixed sign data (y < 0 and y > 0)')
        self.sign = 1
        if (numpy.array(y) < 0).any():
            self.sign = -1
        self.y_max = max(abs(self.y_orig))
        orig_knots = knots
        knots_range = None
        if isinstance(knots, int) and knots < 0:
            knots = abs(knots)
            knots_range = (3, knots + 1)
        self.allow_no_errorbar = allow_no_errorbar
        self.kw_orig = {}
        for var in self.kw_vars:
            self.kw_orig[var] = eval(var)
        self.fits = {}
        if knots_range:
            self.fit_knot_range(knots_range, **self.kw_orig)
        else:
            self.fit_single(**self.kw_orig)
        if plot_best:
            self.plot_best_fit()

    def build_params(self, **keyw):
        r"""
        Build the ``lmfit.Parameters`` object needed for the fitting

        :param \**keyw: The Fit Keywords documented in the ``__init__`` method

        :return: The ``lmfit.Parameters`` translation of the settings given by ``**keyw``
        """
        kw = copy.deepcopy(self.kw_orig)
        kw.update(keyw)
        tag = self.get_tag(**kw)

        # Put the Fit Keywords in the current namespace
        min_slope = kw['min_slope']
        monotonic = kw['monotonic']
        min_dist = kw['min_dist']
        first_knot = kw['first_knot']
        knots = kw['knots']
        fixed_knots = kw['fixed_knots']
        fit_SOL = kw['fit_SOL']
        outliers = kw['outliers']

        # Determine the maximum x to be fit
        max_x = max([max(self.x), 1])
        if not fit_SOL:
            max_x = 1

        # Determine whether knots is the number of knots or the knot locations
        if isinstance(knots, int):
            if knots < 3:
                raise ValueError('The value of knots must be >= 3')
            nk = knots
            if first_knot is not None:
                knots = numpy.array([0] + list(numpy.linspace(first_knot, max_x, nk - 1)))
            else:
                knots = numpy.linspace(0, max_x, nk)
        elif numpy.iterable(knots):
            nk = len(knots)
            if (knots[0] != 0 or knots[-1] != max_x) and not fixed_knots:
                print('Resetting knots[0],knots[-1] to 0,%g' % max_x)
                knots[-1] = max_x
                knots[0] = 0
        else:
            raise ValueError('knots must be an integer >= 3 or an iterable')

        # Check initial knots
        if min_dist < 0 and (numpy.diff(knots) < abs(min_dist)).any():
            raise ValueError('Initial knots violated min_dist: %s' % knots)

        # Convert monotonic to min_slope
        if monotonic:
            if min_slope:
                min_slope = max([min_slope, 0])
            else:
                min_slope = 0

        # add lmfit parameters
        params = lmfit.Parameters()

        # number of knots
        params.add('nk', value=len(knots), vary=False)
        nk = params['nk'].value
        if min_dist >= 0:
            # knot locations
            for i in range(nk):
                params.add('xk%d' % i, value=knots[i], min=abs(min_dist) * i,
                           max=knots[-1] - (nk - i - 1) * abs(min_dist), vary=not fixed_knots)
            params['xk0'].vary = False
            params['xk%d' % (nk - 1)].vary = False
        else:
            # knot difference constraints
            params.add('xk0', value=knots[0], vary=False)
            for i in range(1, nk):
                params.add('dxk%d' % i, value=knots[i] - knots[i - 1], min=abs(min_dist), max=(knots[-1] - knots[0] - abs(min_dist) * (nk - 2)), vary=not fixed_knots)
            denom = '(' + '+'.join(['dxk%d' % k for k in range(1, nk)]) + ')'
            num = '(%s-xk0)' % (knots[-1])
            for i in range(1, nk - 1):
                params.add('xk%d' % i, value=knots[i],
                           expr='xk0+(%s)/%s*%s' % ('+'.join(['dxk%d' % k for k in range(1, i + 1)]), denom, num))
            params.add('xk%d' % (nk - 1), value=knots[-1], vary=False)

        # guess of value of z
        guess_z = [(max(abs(self.y)) - min(abs(self.y))) / (max(self.x) - min(self.x)) / (numpy.mean(abs(self.y)))] * len(knots)
        guess_ybc = abs(self.y[0]) / self.y_max

        # If the fixed knot solution exists, use it as a starting guess
        if not fixed_knots:
            kw['fixed_knots'] = True
            fixed_tag = self.get_tag(**kw)
            kw['fixed_knots'] = False
            if fixed_tag in self.fits:
                guess_z = self.get_zk(self.fits[fixed_tag].params)
                guess_ybc = self.fits[fixed_tag].params['ybc'].value

        # zeroth knot value
        if knots[0] == 0:
            params.add('zk0', value=0, vary=False)
        else:
            params.add('zk0', value=guess_z[0], min=min_slope)

        # knot values
        for i in range(1, nk):
            params.add('zk%d' % i, value=guess_z[i], min=min_slope)

        # knot boundary condition value
        params.add('ybc', value=guess_ybc, min=max([0, min(abs(self.y)) / 2. / self.y_max]), max=max(abs(self.y)) * 2. / self.y_max)

        # knot boundary condition location
        params.add('xbc', value=0, vary=False)

        # constraint on first knot location
        if first_knot:
            if 'dxk1' in params:
                params['dxk1'].min = max([first_knot, abs(min_dist)])
            params['xk1'].min = first_knot
        else:
            if 'dxk1' in params:
                params['dxk1'].min = min(self.x) + abs(min_dist)
            params['xk1'].min = min(self.x) + abs(min_dist)

        # keeping last knot value positive
        if min_slope:
            params['zk%d' % (nk - 1)].min = max([0, min_slope])
        else:
            params['zk%d' % (nk - 1)].min = 0

        return params

    def model(self, params, x, lmfit_out=None):
        '''
        Return the model integrated scale length curve at x

        :param params: The `lmfit.Parameters` object

        :param x: evaluate model at x

        :param lmfit_out: ``lmfit.MinimizerResult`` instance to use for getting uncertainties in the curve
        '''
        if not lmfit_out:
            xk = self.get_xk(params)
            zk = self.get_zk(params)
            xbc = params['xbc'].value
            ybc = params['ybc'].value
            return self.sign * self.y_max * integz(xk, zk, xbc, ybc, x)
        else:
            xk, zk, ybc, xbc = self.get_param_unc(lmfit_out)

            def wrappable_integz(xkzk, x):
                xbc = xkzk[0, -1]
                xk = xkzk[0, :-1]
                zk = xkzk[1, :-1]
                ybc = xkzk[1, -1]
                return integz(xk, zk, xbc, ybc, x)

            wrapped_integz = uncertainties.unumpy.core.wrap_array_func(wrappable_integz)
            return self.sign * self.y_max * wrapped_integz(numpy.array([xk.tolist() + [xbc], zk.tolist() + [ybc]]), x)

    def plot(self, showZ=True, x=np.linspace(0, 1.1, 111)):
        '''
        Plot all fits calculated so far, each in its own tab of a ``FigureNotebook``,
        where the tab is labeled by the shortened tag of the tag of the fit

        :param showZ: Overplot the values of the inverse scale lengths in red

        :param x: The x values to use for plotting the fitted curve

        :return: The ``FigureNotebook`` instance created
        '''
        return knotted_fit_base.plot(self, showZ=showZ, x=x)

    def plot_individual_fit(self, tag, ax=None, showZ=True, x=np.linspace(0, 1.1, 1001)):
        '''
        Plot a single fit, characterized by ``tag``

        :param tag: The tag of the fit that is to be plotted, must be in self.fits.keys()

        :param ax: The axes to plot into (one is created if ``None``)

        :param showZ: Overplot the values of the inverse scale lengths in red

        :param x: The x values to use for plotting the fitted curve
        '''
        from omfit_plot import uerrorbar

        knotted_fit_base.plot_individual_fit(self, tag, x=x, ax=ax)
        k = tag
        v = self.fits[k]
        if ax is None:
            import matplotlib.pyplot as plt
            ax = plt.gca()
        # Plot scale lengths
        if showZ:
            ax.spines['right'].set_color('red')
            ax2 = ax.twinx()
            xk, zk, ybc, xbc = self.get_param_unc(v)
            uerrorbar(xk, zk, color='red', label='Scale length', ax=ax2)
            ax2.set_ylim([min(nominal_values(zk) - std_devs(zk)), max(nominal_values(zk) + std_devs(zk))])
            ax2.tick_params(axis='y', colors='red')
            ax2.set_xlim([0, 1.1])
            ax2.axvline(1.0, color='k')
            ax2.set_ylabel(r'$\partial\log()/\partial\rho$', color='red')

class fitSpline(knotted_fit_base):
    '''
    Fit a spline to some data; return the fit with uncertainties
    '''

    def __init__(self, x, y, yerr, knots=3, min_dist=0, first_knot=None,
                 fixed_knots=False, fit_SOL=False, monotonic=False, min_slope=None,
                 outliers=3, plot_best=False, allow_no_errorbar=False):
        knotted_fit_base.__init__(self, x, y, yerr)

        self.y_max = self.y[np.argmax(abs(self.y))]

        orig_knots = knots
        knots_range = None
        if isinstance(knots, int) and knots < 0:
            knots = abs(knots)
            knots_range = (3, knots + 1)
        self.allow_no_errorbar = allow_no_errorbar
        self.kw_orig = {}
        for var in self.kw_vars:
            self.kw_orig[var] = eval(var)
        self.fits = {}
        if knots_range:
            self.fit_knot_range(knots_range, **self.kw_orig)
        else:
            self.fit_single(**self.kw_orig)
        if plot_best:
            self.plot_best_fit()

    def build_params(self, **keyw):
        r"""
        Build the ``lmfit.Parameters`` object needed for the fitting

        :param \**keyw: The Fit Keywords documented in the ``__init__`` method

        :return: The ``lmfit.Parameters`` translation of the settings given by ``**keyw``
        """
        kw = copy.deepcopy(self.kw_orig)
        kw.update(keyw)
        tag = self.get_tag(**kw)

        # Put the Fit Keywords in the current namespace
        min_slope = kw['min_slope']
        monotonic = kw['monotonic']
        min_dist = kw['min_dist']
        first_knot = kw['first_knot']
        knots = kw['knots']
        fixed_knots = kw['fixed_knots']
        fit_SOL = kw['fit_SOL']
        outliers = kw['outliers']

        # Determine the maximum x to be fit
        max_x = max([max(self.x), 1])
        if not fit_SOL:
            max_x = 1

        # Determine whether knots is the number of knots or the knot locations
        if isinstance(knots, int):
            if knots < 3:
                raise ValueError('The value of knots must be >= 3')
            nk = knots
            if first_knot is not None:
                knots = numpy.array([0] + list(numpy.linspace(first_knot, max_x, nk - 1)))
            else:
                knots = numpy.linspace(0, max_x, nk)
        elif numpy.iterable(knots):
            nk = len(knots)
            if (knots[0] != 0 or knots[-1] != max_x) and not fixed_knots:
                print('Resetting knots[0],knots[-1] to 0,%g' % max_x)
                knots[-1] = max_x
                knots[0] = 0
        else:
            raise ValueError('knots must be an integer >= 3 or an iterable')

        # Check initial knots
        if min_dist < 0 and (numpy.diff(knots) < abs(min_dist)).any():
            raise ValueError('Initial knots violated min_dist: %s' % knots)

        # Convert monotonic to min_slope
        if monotonic:
            raise NotImplementedError('Not yet available')

        # add lmfit parameters
        params = lmfit.Parameters()

        # number of knots
        params.add('nk', value=len(knots), vary=False)
        nk = params['nk'].value
        if False:  # min_dist>=0:
            # knot locations
            for i in range(nk):
                params.add('xk%d' % i, value=knots[i], min=abs(min_dist) * i,
                           max=knots[-1] - (nk - i - 1) * abs(min_dist), vary=not fixed_knots)
            params['xk0'].vary = False
            params['xk%d' % (nk - 1)].vary = False
        if True:
            # knot difference constraints
            params.add('xk0', value=knots[0], vary=False)
            for i in range(1, nk):
                params.add('dxk%d' % i, value=knots[i] - knots[i - 1], min=abs(min_dist), max=(knots[-1] - knots[0] - abs(min_dist) * (nk - 2)), vary=not fixed_knots)
            denom = '(' + '+'.join(['dxk%d' % k for k in range(1, nk)]) + ')'
            num = '(%s-xk0)' % (knots[-1])
            for i in range(1, nk - 1):
                params.add('xk%d' % i, value=None,  # knots[i],
                           expr='xk0+(%s)/%s*%s' % ('+'.join(['dxk%d' % k for k in range(1, i + 1)]), denom, num))
            params.add('xk%d' % (nk - 1), value=knots[-1], vary=False)

        # guess of value of knot values
        guess_z = []
        for k in knots:
            guess_z.append(self.y[closestIndex(self.x, k)] / self.y_max)
        guess_z = numpy.array(guess_z)

        # If the fixed knot solution exists, use it as a starting guess
        if not fixed_knots:
            kw['fixed_knots'] = True
            fixed_tag = self.get_tag(**kw)
            kw['fixed_knots'] = False
            if fixed_tag in self.fits:
                guess_z = self.get_zk(self.fits[fixed_tag].params)
                guess_ybc = self.fits[fixed_tag].params['ybc'].value

        # knot values
        for i in range(0, nk):
            params.add('zk%d' % i, value=guess_z[i])

        # knot boundary condition value (not active currently)
        params.add('ybc', value=0, vary=False)

        # knot boundary condition location
        params.add('xbc', value=0, vary=False)

        # constraint on first knot location
        if first_knot:
            if 'dxk1' in params:
                params['dxk1'].min = max([first_knot, abs(min_dist)])
            # params['xk1'].min = first_knot
        else:
            if 'dxk1' in params:
                params['dxk1'].min = min(self.x) + abs(min_dist)
            # params['xk1'].min = min(self.x) + abs(min_dist)

        return params

    def model(self, params, x, lmfit_out=None):
        '''
        Return the model spline curve at x

        :param params: The `lmfit.Parameters` object

        :param x: evaluate model at x

        :param lmfit_out: ``lmfit.MinimizerResult`` instance to use for getting uncertainties in the curve
        '''
        if not lmfit_out:
            xk = self.get_xk(params)
            zk = self.get_zk(params)
            xbc = params['xbc'].value
            ybc = params['ybc'].value
            try:
                return self.y_max * CubicSpline(xk, zk, bc_type=((1, 0), (2, 0)))(x)
            except ValueError:
                raise
        else:
            xk, zk, ybc, xbc = self.get_param_unc(lmfit_out)

            def wrappable_cubic_spline(xkzk, x):
                xk, zk = xkzk
                try:
                    return CubicSpline(xk, zk, bc_type=((1, 0), (2, 0)))(x)
                except ValueError:
                    raise

            wrapped_cubic_spline = uncertainties.unumpy.core.wrap_array_func(wrappable_cubic_spline)

            return self.y_max * wrapped_cubic_spline(numpy.array((xk, zk)), x)

def xy_outliers(x, y, cutoff=1.2, return_valid=False):
    '''
    This function returns the index of the outlier x,y data
    useful to run before doing a fit of experimental data
    to remove outliers. This function works assuming that
    the first and the last samples of the x/y data set are
    valid data points (i.e. not outliers).

    :param x: x data (e.g. rho)

    :param y: y data (e.g. ne)

    :param cutoff: sensitivity of the cutoff (smaller numbers -> more sensitive [min=1])

    :param return_valid: if False returns the index of the outliers, if True returns the index of the valid data

    :return: index of outliers or valid data depending on `return_valid` switch
    '''

    index=numpy.argsort(y)
    x=x[index]
    y=y[index]

    index=numpy.argsort(x)
    x=x[index]
    y=y[index]

    x=(x-numpy.min(x))/(numpy.max(x)-numpy.min(x))
    y=(y-numpy.min(y))/(numpy.max(y)-numpy.min(y))

    def step(k1=0,path=[],dpath=[]):
        if len(x)-1 in path and 0 in path:
            return
        dist=numpy.sqrt((x[k1]-x)**2+(y[k1]-y)**2)
        dist[path]=1E100
        i=numpy.argmin(dist)
        path.append(i)
        dpath.append(dist[i])
        step(i,path,dpath)
        return

    val=numpy.zeros(numpy.size(x))
    tension=0.0
    for k in range(len(x)):
        path=[]
        dpath=[]
        step(k1=k,path=path,dpath=dpath)
        clustering=(1-numpy.array(dpath)/numpy.sqrt(2.))
        val[path]+= tension + clustering

    outliers=numpy.where(val<(numpy.mean(val)/numpy.max([1,cutoff])))[0].tolist()
    if 0 in outliers:
        outliers.remove(0)
    if len(x)-1 in outliers:
        outliers.remove(len(x)-1)

    if return_valid:
        return [kk for kk in range(len(x)) if kk not in outliers]
    else:
        return outliers

# GPR fitting class using gptools package
class fitGP(object):
    """
    Inputs:
    --------
    x: array or array of arrays
        Independent variable data points.
    y: array or array of arrays
        Dependent variable data points.
    e: array or array of arrays
        Uncertainty in the dependent variable data points.
    noise_ub: float, optional
        Upper bound on a multiplicative factor that will be optimized to infer the most probable systematic
        underestimation of uncertainties. Note that this factor is applied over the entire data profile,
        although diagnostic uncertainties are expected to be heteroschedastic.  Default is 2 (giving
        significant freedom to the optimizer).
    random_starts: int, optional
        Number of random starts for the optimization of the hyperparameters of the GP. Each random
        starts begins sampling the posterior distribution in a different way. The optimization that
        gives the largest posterior probability is chosen. It is recommended to increase this value
        if the fit results difficult. If the regression fails, it might be necessary to vary the
        constraints given in the _fit method of the class GPfit2 below, which has been kept rather
        general for common usage. Default is 20.
    zero_value_outside: bool, optional
        Set to True if the profile to be evaluated is expected to go to zero beyond the LCFS, e.g. for
        electron temperature and density; note that this option does NOT force the value to be 0 at the LCFS,
        but only attempts to constrain the fit to stabilize to 0 well beyond rho=1. Profiles like those of
        omega_tor_12C6 and T_12C6 are experimentally observed not to go to 0 at the LCFS, so this option
        should be set to False for these. Default is True.
    ntanh: integer, optional
        Set to 2 if an internal transport barrier is expected. Default is 1 (no ITB expected).
        This parameter has NOT been tested recently.
    verbose: bool, optional
        If set to True, outputs messages from non-linear kernel optimization. Default is False.

    Returns:
    ----------
    (object) fit: call at points at which the profile is to be evaluated, e.g. if locations are stored in
        an array ``xo'', call fo = fit(xo). For an example, see 7_fit.py in OMFITprofiles.
    """

    def __init__(self, xx, yy, ey, noise_ub=2.0, random_starts=20, zero_value_outside=True, ntanh=1, verbose=False):
        self.xx = np.atleast_2d(xx)
        self.yy = np.atleast_2d(yy)
        self.ey = np.atleast_2d(ey)
        self.ntanh = ntanh
        self.noise_ub = noise_ub
        self.random_starts = random_starts
        self.initial_params = [2.0, 0.5, 0.05, 0.1, 0.5] # for random_starts!= 0, the initial state of the hyperparameters is not actually used.
        self.verbose = verbose
        self.zero_value_outside = zero_value_outside

        self.gp = []
        if not self.xx.size:
            return

        for k in range(self.xx.shape[0]):
            if verbose:
                printi('fitting profile '+str(k+1)+' of '+str(self.xx.shape[0]))
            i = ~np.isnan(self.yy[k,:])&~np.isnan(self.xx[k,:])&~np.isnan(self.ey[k,:])
            self.gp.append(self._fit(self.xx[k,i],self.yy[k,i],self.ey[k,i]))

    def _fit(self, xx, yy, ey):
        import gptools
        import copy
        norm = np.mean(abs(yy))
        yy = yy/norm
        ey = ey/norm

        for kk in range(self.ntanh):

            hprior = (
                # Set a uniform prior for sigmaf
                gptools.UniformJointPrior([(0,10),])*
                # Set Gamma distribution('alternative form') for the other 4 priors of the Gibbs 1D Tanh kernel
                gptools.GammaJointPriorAlt([1.0,0.5,0.0,1.0],[0.3,0.25,0.1,0.05])
                )

            k = gptools.GibbsKernel1dTanh(
                #= ====== =======================================================================
                #0 sigmaf Amplitude of the covariance function
                #1 l1     Small-X saturation value of the length scale.
                #2 l2     Large-X saturation value of the length scale.
                #3 lw     Length scale of the transition between the two length scales.
                #4 x0     Location of the center of the transition between the two length scales.
                #= ====== =======================================================================
                initial_params=self.initial_params,
                fixed_params=[False]*5,
                hyperprior=hprior,
                )

            if kk==0:
                nk = gptools.DiagonalNoiseKernel(1, n=0, initial_noise=np.mean(ey)*self.noise_ub,
                        fixed_noise=False, noise_bound=(min(ey), max(ey)*self.noise_ub))#, enforce_bounds=True)
                #printd "noise_ub= [", min(ey), ",",max(ey)*self.noise_ub,"]"
                ke=k
            else:  #the following is from Orso's initial implementation. Not tested on ITBs!
                nk = gptools.DiagonalNoiseKernel(1, n=0, initial_noise=gp.noise_k.params[0], fixed_noise=False)
                k1 = gptools.GibbsKernel1dTanh(
                    initial_params=copy.deepcopy(gp.k.params[-5:]),
                    fixed_params=[False]*5)
                ke+=k1

            # Create and populate GP:
            gp = gptools.GaussianProcess(ke, noise_k=nk)
            gp.add_data(xx, yy, err_y=ey)
            gp.add_data(0, 0, n=1, err_y=0.0) #zero derivative on axis

            #================= Add constraints ====================
            # Impose constraints on values in the SOL
            if self.zero_value_outside:
                    gp.add_data(max([1.1,max(xx)])+0.1, 0, n=0, err_y=0) #zero beyond edge
                    gp.add_data(max([1.1,max(xx)])+0.2, 0, n=0, err_y=0) #zero beyond edge

            # Impose constraints on derivatives in the SOL
            #grad=gradient(yy,xx) # rough estimate of gradients -- this seems broken...
            grad=np.gradient(yy)/np.gradient(xx)  # alternative rough estimte of gradients

            gp.add_data(max([1.1,max(xx)]),0,n=1, err_y=abs(max(grad)*max(ey/yy))) # added uncertainty in derivative
            #printd "Added {:.0f}% of max(gradient) in max(grad) on GPR derivative constraints outside of the LCFS".format(max(ey/yy)*100)
            gp.add_data(max([1.1,max(xx)])+0.1, 0, n=1) #zero derivative far beyond at edge

            for kk1 in range(1,3):
                if self.zero_value_outside:
                    gp.add_data(max([1.1,max(xx)])+0.1*kk1, 0, n=0, err_y=np.mean(ey)) #zero at edge
                gp.add_data(max([1.1,max(xx)])+0.1*kk1, 0, n=1) #zero derivative beyond the edge

            # In shots where data is missing at the edge, attempt forcing outer stabilization
            if max(xx)<0.8:
                print("Missing data close to the edge. Fit at rho>0.8 might be rather wild.")
                if self.zero_value_outside:
                    if max(ey/yy)<0.1:
                        gp.add_data(1.0, 0, n=0, err_y=max(ey)*2)
                    else:
                        gp.add_data(1.0, 0, n=0, err_y=max(ey))
                # pad SOL with zero-derivative constraints
                for i in numpy.arange(5):
                    gp.add_data(1.05+0.02*i,0,n=1) #exact derivative=0

            #============ Optimization of hyperparameters ===========
            print('Number of random starts: ', self.random_starts)
            if kk==0:
                # Optimize hyperparameters:
                gp.optimize_hyperparameters(
                    method='SLSQP',
                    verbose=self.verbose,
                    num_proc=None,    #if 0, optimization with 1 processor in series; if None, use all available processors
                    random_starts=self.random_starts,
                    opt_kwargs={ 'bounds': (ke+nk).free_param_bounds,})

            else:
                # Optimize hyperparameters:
                gp.optimize_hyperparameters(
                    method='SLSQP',
                    verbose=self.verbose,
                    num_proc=None,
                    random_starts=self.random_starts,
                    opt_kwargs={ 'bounds': ke.free_param_bounds,},)

        gp.norm=norm
        self.inferred_params=copy.deepcopy(gp.k.params)
        self.final_noise=copy.deepcopy(gp.noise_k.params)
        print('------> self.inferred_params: ', self.inferred_params)
        print('-------> self.final_noise: ', self.final_noise)
        print('-------> numpy.mean(ey) =', np.mean(ey))
        print('-------> self.final_noise/ numpy.mean(ey) =', self.final_noise/np.mean(ey))
        return gp

    def __call__(self, Xstar, n=0, use_MCMC=False, profile=None):
        """
        Evaluate the fit at specific locations.
        Inputs:
        ----------
        Xstar: array
            Independent variable values at which we wish to evaluate the fit.
        n: int, optional
            Order of derivative to evaluate. Default is 0 (data profile)
        use_MCMC: bool, optional
            Set whether MCMC sampling and a fully-Bayesian estimate for a fitting
            should be used. This is recommended for accurate computations of gradients
            and uncertainties.
        Profile: int, optional
            Profile to evaluate if more than one has been computed and include in the gp
            object. To call the nth profile, set profile=n. If None, it will return an
            array of arrays.
        Outputs:
        ----------
        Value and error of the fit evaluated at the Xstar points
        """

        if profile is None:
            profile=list(range(len(self.gp)))
            print("len(profile) = ", len(profile))
        M=[]
        D=[]
        run_on_engaging=False
        for k in numpy.atleast_1d(profile):
            if n>1:
                print('Trying to evaluate higher derivatives than 1. Warning: *NOT* TESTED!')
            else:
                print('Proceeding with the evaluation of {:}-derivative'.format(n))

            predict_data={'Xstar':Xstar,'n': n, 'gp': self.gp[k]}
            if use_MCMC:
                print('*************** Using MCMC for predictions ********************')
                if run_on_engaging: # set up to run on engaging. This is only preliminary!
                    tmp_python_script = '''
                        def predict_MCMC(Xstar, n, gp):
                            """
                            Helper function to call gptool's predict method with MCMC
                            """
                            out=gp.predict_MCMC(Xstar,n=n,full_output=True, noise=True, return_std=True, full_MCMC=True)
                            return out'''

                    out=OMFITx.remote_python(module_root=None,
                                         python_script=tmp_python_script,
                                         target_function=predict_MCMC,
                                         namespace=predict_data,
                                         remotedir=OMFITtmpDir,
                                         workdir=OMFITtmpDir,
                                         server=OMFIT['MainSettings']['SERVER']['engaging']['server'],
                                         tunnel=OMFIT['MainSettings']['SERVER']['engaging']['tunnel'])
                else:
                    out=OMFITx.remote_python(root,
                                         python_script=tmp_python_script,
                                         target_function=predict_MCMC,
                                         namespace=predict_data)
            else:
                out=self.gp[k].predict(Xstar,n=n,full_output=True, noise=True)

            m=out['mean'] #covd=out['cov'] has size len(Xstar) x len(Xstar)
            std= out['std'] # equivalent to numpy.squeeze(numpy.sqrt(diagonal(covd)))

            # Multiply the outputs by the norm, since data were divided by this before fitting
            m=m*self.gp[k].norm
            d=std*self.gp[k].norm
            M.append(m)
            D.append(d)
        M = numpy.squeeze(M)
        D = numpy.squeeze(D)
        return unumpy.uarray(M, D)

    def plot(self, profile=None, ngauss=1):
        """
        Function to conveniently plot the input data and the result of the fit.
        Inputs:
        -----------
        Profile: int, optional
            Profile to evaluate if more than one has been computed and included in the gp
            object. To call the nth profile, set profile=n. If None, it will return an
            array of arrays.
        ngauss: int, optional
            Number of shaded standard deviations
        Outputs:
        -----------
        None
        """
        from matplotlib import pyplot

        pyplot.figure()
        if profile is None:
            profile=list(range(len(self.gp)))

        Xstar=numpy.linspace(0,numpy.nanmin([1.2,numpy.nanmax(self.xx+0.1)]),1000)
        for k in profile:
            ua = self(Xstar,0,k)
            m, d = nominal_values(ua), std_devs(ua)
            pyplot.errorbar(self.xx[k,:],self.yy[k,:],self.ey[k,:],color='b',linestyle='')
            pyplot.plot(Xstar,m,linewidth=2,color='g')
            for kk in range(1,ngauss+1):
                pyplot.fill_between(Xstar, m-d*kk, m+d*kk, alpha=0.25, facecolor='g')
            pyplot.axvline(0,color='k')
            pyplot.axhline(0,color='k')

class fitCH(object):
    """
    Fitting of kinetic profiles by Chebyshev polynomials
    Adapted from MATLAB function by A. Marinoni <marinoni@fusion.gat.com>

    :param x: radial coordinate

    :param y: data

    :param yerr: data uncertainties

    :param m: Polynomial degree
    """
    def __init__(self,x, y, yerr, m=18):
        #Beginning of the actual routine
        m0 = [] #polynomial coefficients to be discarded (can be empty, so keep all coefficients)
        n = len(x)

        ##trick/cheat: the fit is better data are mirrored as we avoid cusp and salient points on
        #axis spuriously generated by chebycheff nodes packed at the end of the
        #interval
        #x = [-x(end:-1:1);x]
        #y = [y(end:-1:1);y]
        #yerr = [yerr(end:-1:1);yerr]
        x=hstack((flipud(-x),x))
        y=hstack((flipud(y),y))
        yerr=hstack((flipud(yerr),yerr))

        #errorbar(x,y,yerr)
        #axvline(0,ls='--',color='k')

        ## Generate the z variable as a mapping of input x data range into the interval [-1,1]
        z = ((x-min(x))-(max(x)-x))/(max(x)-min(x));
        jacob = 2/(max(x)-min(x))

        ##Defining data variables like in manuscript
        b = y/yerr

        ##Building the Vandermonde matrix
        A_d = numpy.zeros((len(z),m+1))
        A_d[:,1] = numpy.ones((1,len(z)))
        if m > 1:
           A_d[:,2] = z
        if m > 2:
          for k in range(3,m+1):
             A_d[:,k] = 2*z*A_d[:,k-1] - A_d[:,k-2]  ## recurrence relation
        A_d = np.dot(np.diag(1/yerr),A_d)
        #A_d = A_d(:,~ismember([1:m+1],m0))
        a = numpy.linalg.lstsq(A_d,b)[0]

        ##Computing unnormalized chi2 and quantities that might be of interest
        yfit_data = np.dot(np.dot(np.diag(yerr),A_d),a) #Fit on the data radial points
        res = y-yfit_data;         #residual
        db = res/yerr
        chisq = numpy.sum(db ** 2)
        deg3dom = max(0,len(y)-(m+2-len(m0)))
        normres = norm(res)
        C = np.linalg.pinv(np.dot(A_d.T,A_d))

        ##De-mirroring
        y = y[n+1:]
        x = x[n+1:]
        yerr = yerr[n+1:]
        z = z[n+1:]

        ##Computing uncertainties on the coefficients (this is not necessary for
        ##the fit and can be commented out)
        da = np.dot(np.dot(C,A_d.T),db)

        self.jacob=jacob
        self.C=C
        self.a=a
        self.x=x
        self.y=y
        self.yerr=yerr
        self.m=m

    def __call__(self,rho):
        """
        Calculate fit and uncertainties

        :param rho: rho grid of the fit

        :return:  value and error of the fit evaluated at the rho points
        """
        jacob=self.jacob
        C=self.C
        a=self.a
        m=self.m

        ##Fitted profiles on new cordinate and remapping it with the same jacobian
        zz = np.dot(rho,jacob)

        ##Computing the A matrix on the new radial cordinate and its gradient
        A = numpy.zeros((len(zz),m+1))
        W = numpy.zeros((len(zz),m+1))
        A[:,1] = numpy.ones((1,len(zz)))
        if m > 1:
           A[:,2] = zz
           W[:,2] = numpy.ones(len(zz))*jacob
        if m > 2:
          for k in range(3,m+1):
             A[:,k] = 2*zz*A[:,k-1] - A[:,k-2]  ## recurrence relation
             W[:,k] = gradient(A[:,k],rho) ##Computing along rho instead of along zz and then multiply by jacobian
        #A = A(:,~ismember([1:m+1],m0))
        #W = W(:,~ismember([1:m+1],m0))
        yfit = np.dot(A,a)
        yfit_g = np.dot(W,a)

        ##Scale length
        yfit_sl = -yfit_g/yfit

        #Computing covariance matrices between quantities computed at points (x1,x2)
        cov_gy = np.dot(np.dot(A,C),W.T)
        var_yy = np.dot(np.dot(A,C),A.T)
        var_gg = np.dot(np.dot(W,C),W.T)

        ##Computing covariance vectors at same points (x,x), i.e. taking the diagonal of the matrix
        sig2p = np.diag(var_yy)
        sig2g = np.diag(var_gg)
        cov = np.diag(cov_gy)

        ##Sigmas
        dyfit = numpy.sqrt(sig2p)
        dyfit_g = numpy.sqrt(sig2g)

        ##Formulas to estimate uncertainties on scale length...do not implement as
        ##is still generally wrong inside rho=0.1-0.2. The simplified equation
        ##seems to be better?!
        dump1 = 6*cov**2/yfit**4-4*yfit_g*cov/yfit**3-24*yfit_g*sig2p*cov/yfit_g**5
        dump2 = sig2g+yfit_g**2
        dump3 = sig2p/yfit**4+8*sig2p**2/yfit**6+(1/yfit+sig2p/yfit**3+3*sig2p**2/yfit**5)**2
        dump4 = (-cov/yfit**2-3*cov*sig2p/yfit**4+yfit_g/yfit+yfit_g/yfit**3*sig2p+3*yfit_g*sig2p**2/yfit**5)**2
        sig2sl = dump1+dump2*dump3-dump4 #Eq. 17
        sig2sl_sim = (yfit_g/yfit)**2*(sig2g/yfit_g**2+sig2p/yfit**2) #Eq 18
        dyfit_sl_an = numpy.sqrt(sig2sl) #Analythical uncertainty
        dyfit_sl_an_sim = numpy.sqrt(sig2sl_sim) #Simplified analythical uncertainty

        self.rho=rho
        self.yfit=yfit
        self.dyfit=dyfit

        return yfit,dyfit

    def plot(self):
        """
        Plotting of the raw data and fit with uncertainties

        :return: None
        """
        from matplotlib import pyplot
        pyplot.errorbar(self.x,self.y,self.yerr,color='b',linestyle='',label="Raw data")
        pyplot.plot(self.rho,self.yfit,'g',label="fit")
        pyplot.fill_between(self.rho, self.yfit-self.dyfit, self.yfit+self.dyfit, alpha=0.25, facecolor='b')
        pyplot.legend(loc=0)

class fitLG(object):
    """
    This class provides linear fitting of experimental profiles, with gaussian bluring for smoothing.

    This procedure was inspired by discussions with David Eldon about the `Weighted Average of Interpolations to a Common base` (WAIC) technique that he describes in his thesis.
    However the implementation here is quite a bit different, in that instead of using a weighted average the median profile is taken, which allows for robust rejection of outliers.
    In this implementation the profiles smoothing is obtained by radially perturbing the measurements based on the farthest distance to their neighboring point.

    :param x: x values of the experimental data

    :param y: experimental data

    :param e: uncertainties of the experimental data

    :param d: data time identifier

    :param ng: number of gaussian instances

    :param sm: smoothing

    :param nmax: take no more than nmax data points
    """
    def __init__(self,x,y,e,d,ng=100,sm=1,nmax=None):
        if nmax:
            i=pylab.randint(0,len(x),min([nmax,len(x)]))
            x=x[i]
            y=y[i]
            e=e[i]
            d=d[i]
        i=numpy.argsort(x)
        self.x=x[i]
        self.y=y[i]
        self.e=e[i]
        self.d=d[i]

        self.ux=numpy.unique(x)
        self.dx=numpy.max([hstack((0,abs(numpy.diff(self.ux)))),hstack((0,abs(numpy.diff(self.ux))))],0)

        self.ng=ng
        self.sm=sm
        self._doPlot=False

    def __call__(self,x0):
        x=self.x
        y=self.y
        e=self.e
        d=self.d
        ng=self.ng

        X=[]
        Y=[]
        E=[]
        for k in numpy.unique(d):
            i=numpy.where(d==k)[0]
            if not len(i) or not len(x[i]):
                continue
            X.append(x[i])
            Y.append(y[i])
            E.append(e[i])
            if self._doPlot:
                pyplot.ioff()
            if ng>0:
                dx=interp1e(self.ux,self.dx)(x[i])
                R=numpy.reshape(randn(ng*i.size),(ng,i.size))
                X.extend(x[i][np.newaxis,:]+R*dx[np.newaxis,:]*self.sm)
                R=numpy.reshape(randn(ng*i.size),(ng,i.size))
                Y.extend(y[i][np.newaxis,:]+R*e[i][np.newaxis,:])
                E.extend(numpy.tile(e[i], (ng, 1)))
                if self._doPlot:
                    for k in range(1,ng+1):
                        pyplot.errorbar(X[-k],Y[-k],E[-k],ls='',color='g',alpha=1./ng)
            if self._doPlot:
                pyplot.ion()

        Y0 = numpy.zeros((len(X), len(x0)))
        Y0[:] = numpy.nan
        E0 = numpy.zeros((len(X), len(x0)))
        E0[:] = numpy.nan
        for k in range(len(X)):
            inside = numpy.where((x0 >= min(X[k])) & (x0 <= max(X[k])))[0]
            if len(inside)>1:
                Y0[k,inside]=interp1e(X[k],Y[k])(x0[inside])
                E0[k,inside]=interp1e(X[k],E[k])(x0[inside])
        y0 = numpy.nanmedian(Y0, 0)
        e0 = numpy.nanmedian(E0, 0)
        #plt.plot(x0[i0],y0[i0],'ob')

        ok = numpy.where(numpy.isnan(y0) == 0)[0]
        #handle the core
        i00=numpy.where(x0==0)[0]
        i01=numpy.argmin(x0[ok])
        y0[i00]=y0[ok][i01]
        #handle the edge
        i00=numpy.where(x0==1)[0]
        i01=numpy.argmin(1-x0[ok])
        y0[i00]=y0[ok][i01]
        #interpolate between
        ok=numpy.where(numpy.isnan(y0)==0)[0]
        no=numpy.where(numpy.isnan(y0)==1)[0]
        y0[no]=interp1e(x0[ok],y0[ok])(x0[no])
        e0[no]=interp1e(x0[ok],e0[ok])(x0[no])
        return y0,e0

    def plot(self,variations=True):
        x=self.x
        y=self.y
        e=self.e
        x0=numpy.linspace(0,max(x),201)
        try:
            self._doPlot=variations
            y0,e0=self(x0)
        finally:
            self._doPlot=False
        pyplot.errorbar(x,y,e,color='r',ls='')
        pyplot.errorbar(x0,y0,e0,color='k')

@_available_to_user_math
def mtanh(c, x, y=None, e=1., a2_plus_a3=None):
    """
    Modified tanh function

    >>> if len(c)==6:
    >>>     y=a0*(a1+tanh((a2-x)/a3))+a4*(a5-x)*(x<a5)
    >>> if len(c)==5:
    >>>     y=a0*(a1+tanh((a2-x)/a3))+a4*(a2-a3-x)*(x<a2-a3)

    :param c: array of coefficients [a0,a1,a2,a3,a4,(a5)]

    :param x: x data to fit

    :param y: y data to fit

    :param e: y error of the data to fit

    :return: cost, or evaluates y if y==None
    """
    if (len(c) == 6 and a2_plus_a3 is None) or (len(c) == 5 and a2_plus_a3 is not None):
        if a2_plus_a3 is not None:
            a0, a1, a2, a4, a5 = c
            a3 = a2_plus_a3 - a2
        else:
            a0, a1, a2, a3, a4, a5 = c
        yt = a0 * (a1 + numpy.tanh((a2 - x) / a3))
        ytm = yt + a4 * (a5 - x) * (x < a5)
        if y is not None:
            cost = numpy.sqrt(numpy.mean(((y - ytm) / e) ** 2))
            cost *= (1 + abs(a2 - a5) / a3 / 10.)
            return cost
        else:
            return ytm
    elif (len(c) == 5 and a2_plus_a3 is None) or (len(c) == 4 and a2_plus_a3 is not None):
        if a2_plus_a3 is not None:
            a0, a1, a2, a4 = c
            a3 = a2_plus_a3 - a2
        else:
            a0, a1, a2, a3, a4 = c
        yt = a0 * (a1 + numpy.tanh((a2 - x) / a3))
        aa4 = a2 + a4
        y4 = a0 * (a1 + numpy.tanh((a2 - aa4) / a3))
        d = -a0 / numpy.cosh((a2 - aa4) / a3) ** 2 / a3
        ytm = yt * (x > aa4) + (y4 + d * (x - aa4)) * (x <= aa4)
        if y is not None:
            cost = numpy.sqrt(numpy.mean(((y - ytm) / e) ** 2))
            return cost
        else:
            return ytm

def mtanh_gauss_model(x, bheight, bsol, bpos, bwidth, bslope, aheight, awidth, aexp):

    '''
    Modified hyperbolic tangent function for fitting pedestal with gaussian function for the fitting of the core.
    Stefanikova, E., et al., RewSciInst, 87 (11), Nov 2016
    This function is design to fit H-mode density and temeprature profiles as a function of psi_n.
    '''
    # To be sure psi > 0
    x = abs(x)

    mtanh = lambda x, bslope: (1 + bslope*x)*(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    fped  = lambda x, bheight, bsol, bpos, bwidth, bslope: (bheight - bsol)/2. * (mtanh((bpos-x)/(2*bwidth),bslope)+1)+bsol
    ffull = lambda x, bheight, bsol, bpos, bwidth, bslope, aheight, awidth, aexp: fped(x, bheight, bsol, bwidth, bslope, bpos) + (aheight - fped(x, bheight, bsol, bwidth, bslope, bpos))*np.exp(-(x/awidth)**aexp)

    mtanh_un = lambda x, bslope: (1 + bslope*x)*(unumpy.exp(x)-unumpy.exp(-x))/(unumpy.exp(x)+unumpy.exp(-x))
    fped_un  = lambda x, bheight, bsol, bpos, bwidth, bslope: (bheight - bsol)/2. * (mtanh_un((bpos-x)/(2*bwidth),bslope)+1)+bsol
    ffull_un = lambda x, bheight, bsol, bpos, bwidth, bslope, aheight, awidth, aexp: fped_un(x, bheight, bsol, bwidth, bslope, bpos) + (aheight - fped_un(x, bheight, bsol, bwidth, bslope, bpos))*unumpy.exp(-(x/awidth)**aexp)

    if numpy.any(is_uncertain([bheight, bsol, bpos, bwidth, bslope, aheight, awidth, aexp])):
        return ffull_un(x, bheight, bsol, bpos, bwidth, bslope, aheight, awidth, aexp)
    return ffull(x, bheight, bsol, bpos, bwidth, bslope, aheight, awidth, aexp)

def tanh_model(x, a1, a2, a3, c):
    if is_uncertain([a1, a2, a3, c]):
        return a1 * unumpy.tanh((a2 - x) / a3) + c
    return a1 * tanh((a2 - x) / a3) + c

def tanh_poly_model(x, a1, a2, a3, c, p2, p3, p4):
    if is_uncertain([a1, a2, a3, c, p2, p3, p4]):
        return tanh_model(x, a1, a2, a3, c) + a1 * (-1 * unumpy.tanh(a2 / a3) ** 2 + 1) / a3 * x + p2 * x ** 2 + p3 * x ** 3 + p4 * x ** 4
    return tanh_model(x, a1, a2, a3, c) + a1 * (-1 * tanh(a2 / a3) ** 2 + 1) / a3 * x + p2 * x ** 2 + p3 * x ** 3 + p4 * x ** 4

class fit_base(object):
    def __init__(self,x,y,yerr):
        valid_ind = ~numpy.isnan(x) & ~numpy.isnan(y) & ~numpy.isnan(yerr) & (yerr>0)
        self.x = x[valid_ind]
        self.ymax = max(abs(y[valid_ind]))
        self.y = y[valid_ind]/self.ymax
        self.yerr = yerr[valid_ind]/self.ymax

    def get_param_unc(self,lmfit_out):
        result = {}
        for k in lmfit_out.params:
            result[k] = lmfit_out.params[k].value
        if lmfit_out.errorbars:
            corr_params = uncertainties.correlated_values([lmfit_out.params[k].value for k in lmfit_out.var_names],lmfit_out.covar)
            for vi,var in enumerate(lmfit_out.var_names):
                result[var] = corr_params[vi]
        return result

    @property
    def uparams(self):
        return self.get_param_unc(self.best_fit)

    @property
    def params(self):
        tmp = self.get_param_unc(self.best_fit)
        for item in tmp:
            tmp[item] = nominal_values(tmp[item])
        return tmp

    def __call__(self,x):
        if not hasattr(self,'best_fit'):
            return x*numpy.nan
        try:
            return self.best_fit.eval(x=x,**self.get_param_unc(self.best_fit))*self.ymax
        except Exception:
            return self.best_fit.eval(x=x)*self.ymax

    def valid_fit(self,model_out):
        y = model_out.eval(x=self.x,**self.get_param_unc(model_out))
        return numpy.all(abs(nominal_values(y)) > std_devs(y))

class fit_mtanh_gauss(fit_base):
    def __init__(self, x, y, yerr, **kw):
        fit_base.__init__(self, x, y, yerr)
        x = self.x
        y = self.y
        yerr = self.yerr
        kw.setdefault('verbose', False)
        kw.setdefault('bheight', max(y) / 5.0)
        kw.setdefault('bsol', max(y) / 60.0)
        kw.setdefault('bpos', 1.)
        kw.setdefault('bwidth', 1.02)
        kw.setdefault('bslope', max(y) / 500.0)
        kw.setdefault('aheight', max(y))
        kw.setdefault('awidth', 0.5)
        kw.setdefault('aexp', 3.)
        self.best_fit = lmfit.Model(mtanh_gauss_model).fit(y, x=x, weights=1 / yerr, **kw)

class fit_tanh(fit_base):
    def __init__(self, x, y, yerr, **kw):
        fit_base.__init__(self, x, y, yerr)
        y = self.y
        yerr = self.yerr
        kw.setdefault('verbose', False)
        kw.setdefault('a3', 0.05)
        kw.setdefault('a2', 0.95)
        kw.setdefault('c', y[np.argmin(abs(y))])
        kw.setdefault('a1', y[np.argmax(abs(y))] - y[np.argmin(abs(y))])
        self.best_fit = lmfit.Model(tanh_model).fit(y, x=x, weights=1 / yerr, **kw)

class fit_tanh_poly(fit_base):
    def __init__(self,x,y,yerr, **kw):
        fit_base.__init__(self, x, y, yerr)
        y = self.y
        yerr = self.yerr
        kw.setdefault('a3', 0.05)
        kw.setdefault('a2', 0.95)
        kw.setdefault('c', min(abs(y)))
        kw.setdefault('a1', numpy.mean(y))
        kw.setdefault('p2', .5 * numpy.mean(y))
        kw.setdefault('p3', 1 * numpy.mean(y))
        kw.setdefault('p4', 1 * numpy.mean(y))
        kw.setdefault('verbose', False)
        self.best_fit4 = lmfit.Model(tanh_poly_model).fit(y, x=x, weights=1 / yerr, **kw)
        params = copy.deepcopy(self.best_fit4.params)
        params['p4'].vary = False
        params['p4'].value = 0
        self.best_fit3 = lmfit.Model(tanh_poly_model).fit(y, x=x, weights=1 / yerr, params=params)
        params = copy.deepcopy(self.best_fit3.params)
        params['p3'].vary = False
        params['p3'].value = 0
        self.best_fit2 = lmfit.Model(tanh_poly_model).fit(y, x=x, weights=1 / yerr, params=params)
        best_redchi = 1e30
        for p in range(2, 5):
            bf = getattr(self, 'best_fit%d' % p)
            if bf.redchi < best_redchi:
                if self.valid_fit(bf):
                    self.best_fit = bf
                    best_redchi = bf.redchi

def tanh_through_2_points(x0, y0, x=None):
    '''
    Find tanh passing through two points x0=[x_0,x_1] and y0=[y_0,y_1]

    y=a1*tanh((a2-x)/a3)+c

    :param x0: iterable of two values (x coords)

    :param y0: iterable of two values (y coords)

    :param x: array of coordinates where to evaluate tanh. If None function will return fit parameters

    :return: if `x` is None then return tanh coefficients (a1,a2,a3,c), otherwise returns y(x) points
    '''
    y=numpy.array(y0)-y0[1]
    a1=y[0]/tanh(1)
    a2=x0[1]
    a3=x0[1]-x0[0]
    c=y0[1]
    if x is None:
        return a1,a2,a3,c
    else:
        return a1*tanh((a2-numpy.array(x))/a3)+c

def toq_profiles(psin, width, core, ped, edge, expin, expout):
    '''
    This is a direct fortran->python translation from the TOQ source

    :param psin: psi grid

    :param width: width of tanh

    :param core: core value

    :param ped: pedestal value

    :param edge: separatrix value

    :param expin: inner exponent (EPED1: 1.1 for n_e, and 1.2 for T_e)

    :param expout: outner exponent (EPED1: 1.1 for n_e, and 1.4 for T_e)

    :return: profile
    '''

    # psi_N at tanh symmetry point
    xphalf = 1. - width

    # pconst=?
    pconst = 1. - tanh((1. - xphalf) / width)

    # normalization so that n_ped=ped
    a_n = 2. * (ped - edge) / (1. + tanh(1.) - pconst)

    # psi_N at pedestal
    xped = xphalf - width

    # core density
    coretanh = 0.5 * a_n * (1. - tanh(-xphalf / width) - pconst) + edge

    data = numpy.zeros(psin.shape)

    # for all flux surfaces
    for i in range(len(psin)):
        # psi at flux surface
        xpsi = psin[i]

        # density at flux surface
        data[i] = _data = 0.5 * a_n * (1. - tanh((xpsi - xphalf) / width) - pconst) + edge

        # xtoped is proportional to psi, but normed to equal 1 at top of ped
        xtoped = xpsi / xped

        # if core is set, then add polynomial piece to get desired core value (add only inside pedestal)
        if (core > 0. and xpsi < xped):
            # density at flux surface
            data[i] = _data = _data + (core - coretanh) * (1. - xtoped ** expin) ** expout

    return data

@_available_to_user_math
def mtanh_polyexp(x, params):
    """
    given lmfit Parameters object, generate mtanh_poly
    """

    nc = params['pow_c'].value+1  #always a fixed value
    core = np.zeros(nc)
    keys = list(params.keys())
    for key in keys:
        if 'core' in key:
            core[int(key.split('core')[1])] = params[key].value
    core_poly = np.zeros(len(x)) + core[0]
    for i in range(1, len(core)):
        core_poly += core[i] * x**(i)

    xsym = params['xsym'].value
    blend_width = params['blend_width'].value
    edge_width = params['edge_width'].value
    offset = params['offset'].value
    A = params['A'].value

    edge_func = offset + A*np.exp(-(x - xsym)/edge_width)

    z = (xsym - x)/blend_width

    y_fit = (core_poly*np.exp(z) + edge_func*np.exp(-z))/(np.exp(z) + np.exp(-z))

    return y_fit

def mtanh_polyexp_res(params, x_data, y_data, y_err):
    y_model = mtanh_polyexp(x_data, params)
    res = (y_data - y_model)/y_err
    return res

@_available_to_user_math
class GASpline(object):
    """
    Python based replacement for GAprofiles IDL spline routine "spl_mod"
    * Code accepts irregularly spaced (x,y,e) data and returns fit on regularly spaced grid
    * Numerical spline procedure based on Numerical Recipes Sec. 3.3 equations 3.3.2, 3.3.4, 3.3.5, 3.3.7
    * Auto-knotting uses LMFIT minimization with chosen scheme
    * Boundary conditions enforced with matrix elements

    The logic of this implementation is as follows:
    * The defualt is to auto-knot, which uses least-squares minimization to choose the knot locations.
    * If auto-knotting, then there are options of the guess of the knot locations or option to bias the knots. Else manual knots are used and LMFIT is not called
    * If the knot guess is present, then it is used else there are two options. Else we use the knot guess.
    * If the knot bias is None or >-1 then the knot guess is uniformly distributed using linspace. Else we use linspace with a knot bias.

    For the edge data, the logic is as follows:
    * We can have auto/manual knots, free/fixed boundary value, and fit/ignore edge data.
    * When we fit edge data, then that edge data places a constraint on boundary value.

    When monte-carlo is used, the return value is an unumpy uncertainties array that contains the mean and standard-deviation of the monte-carlo trials.

    >>> x = numpy.linspace(0,1,21)
    >>> y = np.sin(2*np.pi*x)
    >>> e = np.repeat(0.1,len(x))
    >>> xo = numpy.linspace(0,1,101)
    >>> fit_obj = GASpline(x,y,e,xo,numknot=4,doPlot=True,monte=20)
    >>> uerrorbar(x,uarray(y,e))
    >>> plt.plot(xo,nominal_values(fit_obj(xo)))
    >>> uband(xo,fit_obj(xo))
    """

    def __init__(self, xdata, ydata, ydataerr, xufit,
                 y0=None, y0p=0.0, y1=None, y1p=None, sol=True, solc=False,
                 numknot=3, autoknot=True, knotloc=None, knotbias=0,
                 sepval_min=None, sepval_max=None,
                 scrapewidth_min=None, scrapewidth_max=None,
                 method='leastsq', maxiter=2000, monte=0,
                 verbose = False, doPlot = False):

        # On initialization create the parameters and set up all options for the fit.
        # First there is the data
        # All data
        self.xdata = xdata
        self.ydata = ydata
        self.ydataerr = ydataerr
        self.ydatawgt = 1./self.ydataerr

        # Core data
        edgemask = (xdata <= 1.0)
        self.xdatacore = xdata[edgemask]
        self.ydatacore = ydata[edgemask]
        self.ydatacoreerr = ydataerr[edgemask]
        self.ydatacorewgt = 1./self.ydatacoreerr
        self.edgemask = edgemask

        # Edge data
        coremask = (xdata > 1.0)
        self.xdataedge = xdata[coremask]
        self.ydataedge = ydata[coremask]
        self.ydataedgeerr = ydataerr[coremask]
        self.ydataedgewgt = 1./self.ydataedgeerr
        self.coremask = coremask

        # Then each keyword is stored
        self.numknot = numknot
        self.bcs = {'y0':y0, 'y0p':y0p, 'y1':y1, 'y1p':y1p}
        # If there are edge points, then we can fit them, else disable.
        if np.sum(coremask) > 0:
            self.sol = sol
        else:
            self.sol = False
        self.solc = solc
        self.sepval_min = sepval_min
        self.sepval_max = sepval_max
        self.scrapewidth = 0.0
        self.scrapewidth_min = scrapewidth_min
        self.scrapewidth_max = scrapewidth_max
        self.autoknot = autoknot
        self.knotbias = knotbias
        self.method = method
        self.maxiter = maxiter
        self.monte = monte
        self.verbose = verbose

        ####
        # Now the output
        ####
        # knot locations for basic fit, or average of monte-carlo trials
        self.knotloc = knotloc
        # The knot locations of the monte-carlo trials
        self.knotlocs = None
        # The knot locations for each iteration
        self.iknotloc = None
        # The knot values, or the mean of the knot values from monte-carlo
        self.knotval = None
        # The knot values of the monte-carlo trials
        self.knotvals = None
        # The data used in the fits
        if self.sol:
            self.xdatafit = self.xdata
            self.ydatafit = self.ydata
            self.ydatafiterr = self.ydataerr
            self.ydatafitwgt = self.ydatawgt
        else:
            self.xdatafit = self.xdatacore
            self.ydatafit = self.ydatacore
            self.ydatafiterr = self.ydatacoreerr
            self.ydatafitwgt = self.ydatacorewgt
        # The data used in the monte-carlo trials
        self.ydatafits = None
        # Fit on data axis or mean of monte-carlo trials
        self.yfit = None
        # The fits from the monte-carlo trials on data axis
        self.yfits = None
        # The uncertainty in the fit from the stddev of the monte-carlo trials
        self.yfiterr = np.zeros(len(self.xdata))
        # Uniform / user x axis
        self.xufit = xufit
        # Fit on user axis
        self.yufit = None
        self.yufits = None
        self.yufiterr = np.zeros(len(self.xdata))
        # First derivative or mean of monte-carlo trials on user axis
        self.ypufit = None
        self.ypufits = None
        self.ypufiterr = np.zeros(len(self.xdata))
        # Fit quantities consistent with LMFIT
        # Number of observations
        self.ndata = None
        # Number of free parameters
        self.nvarys = None
        # Degrees of freedom = number of observations - number of free parameters
        self.nfree = None
        self.rchi2 = None
        self.irchi2 = None
        self.rchi2s = None

        # Do the fit
        self.do_fit()

        if monte is not None and monte > 0:
            knotlocs = np.zeros((self.numknot,monte))
            knotvals = np.zeros((self.numknot,monte))
            ydatafits = np.zeros((len(self.xdatafit),monte))
            yfits = np.zeros((len(self.xdatafit),monte))
            yufits = np.zeros((len(self.xufit),monte))
            ypufits = np.zeros((len(self.xufit),monte))
            rchi2s= np.zeros(monte)
            for im in range(monte):
                if self.sol:
                    self.ydatafit = self.ydata + np.random.normal(0.0, 1.0, len(self.xdata))*self.ydataerr
                else:
                    self.ydatafit = self.ydatacore + np.random.normal(0.0, 1.0, len(self.xdatacore))*self.ydatacoreerr

                self.knotloc = knotloc
                self.do_fit()
                knotlocs[:,im] = self.knotloc
                knotvals[:,im] = self.knotval
                ydatafits[:,im] = self.ydatafit
                yfits[:,im] = self.yfit
                yufits[:,im] = self.yufit
                ypufits[:,im] = self.ypufit
                rchi2s[im] = self.rchi2

            # Replace the data
            if self.sol:
                self.ydatafit = self.ydata
            else:
                self.ydatafit = self.ydatacore
            self.ydatafits = ydatafits
            # Replace the knot locations for the nominal unperturbed data
            self.knotlocs = knotlocs
            self.knotloc = np.mean(knotlocs,axis=1)
            self.knotval = np.mean(knotvals,axis=1)
            self.rchi2 = np.mean(rchi2s)
            # Store the fit as the mean of the monte-carlo trials
            self.yfits = yfits
            self.yfit = np.mean(yfits,axis=1)
            self.yufits = yufits
            self.yufit = np.mean(yufits,axis=1)
            self.ypufits = ypufits
            self.ypufit = np.mean(ypufits,axis=1)
            # Store the uncertianty as the standard deviation of the monte-carlo trials
            self.yfiterr = np.std(yfits,axis=1)
            self.yufiterr = np.std(yufits,axis=1)
            self.ypufiterr = np.std(ypufits,axis=1)

        # Do the plot
        if doPlot:
            self.plot()

    def design_matrix(self, xcore, knotloc, bcs):
        ''' Design Matrix for cubic spline interpolation
        Numerical Recipes Sec. 3.3
        :param xcore: rho values for data on [0,1]
        :param knotloc: knot locations on [0,1]
        :param bcs: Dictionary of boundary conditions
        :return: design matrix for cubic interpolating spline
        '''

        nx = len(xcore)
        numknot = len(knotloc)

        a = np.zeros((nx,numknot))
        b = np.zeros((nx,numknot))
        gee = np.zeros((numknot,numknot))
        h = np.zeros((numknot,numknot))
        geeinvh = np.zeros((numknot,numknot))
        rowa = np.zeros(numknot)
        rowb = np.zeros(numknot)
        c = np.zeros(numknot)
        fx = np.zeros(numknot)

        for i in range(len(xcore)):
            rowa[:] = 0.0
            rowb[:] = 0.0

            bt = numpy.where((knotloc >= xcore[i]) & (knotloc>0.0))[0]
            j = bt[0]
            # Eq. 3.3.2
            rowa[j-1] = (knotloc[j] - xcore[i])/(knotloc[j] - knotloc[j-1])
            rowa[j] = 1.0 - rowa[j-1]

            # Eq. 3.3.4
            rowb[j-1] = (rowa[j-1]**3 - rowa[j-1]) * (knotloc[j] - knotloc[j-1])**2/6.0
            rowb[j] = (rowa[j]**3 - rowa[j]) * (knotloc[j] - knotloc[j-1])**2/6.0

            a[i,:] = rowa
            b[i,:] = rowb

        # Apply B.C. for y(0)
        if bcs['y0'] is not None:
            fx[0] = bcs['y0']

        # Apply B.C. for y'(0)
        if bcs['y0p'] is not None:
            gee[0,0] = (knotloc[1]-knotloc[0])/3.0
            gee[0,1] = gee[0,0]/2.0
            h[0,0] = -1.0/(knotloc[1]-knotloc[0])
            h[0,1] = -1.0*h[0,0]
            c[0] = -1.0*bcs['y0p']
        else:
            gee[0,0] = 1.0

        # Apply B.C. for y(1)
        if bcs['y1'] is not None:
            fx[numknot-1] = bcs['y1']

        # Apply B.C. for y'(1)
        if bcs['y1p'] is not None:
            gee[numknot-1,numknot-1] = -1.0*(knotloc[numknot-1]-knotloc[numknot-2])/3.0
            gee[numknot-1,numknot-2] = gee[numknot-1,numknot-1]/2.0
            h[numknot-1,numknot-1] = 1.0/(knotloc[numknot-1] - knotloc[numknot-2])
            h[numknot-1,numknot-2] = -1.0*h[numknot-1,numknot-1]
            c[numknot-1] = -1.0*bcs['y1p']
        else:
            gee[numknot-1,numknot-1] = 1.0

        # Internal knots
        # Eq. 3.3.7
        for i in range(1,numknot-1,1):
            gee[i,i-1] = (knotloc[i] - knotloc[i-1])/6.0
            gee[i,i] = (knotloc[i+1] - knotloc[i-1])/3.0
            gee[i,i+1] = (knotloc[i+1] - knotloc[i])/6.0
            h[i,i-1] = 1.0 / (knotloc[i] - knotloc[i-1])
            h[i,i] = -1.0 * (1.0/(knotloc[i+1] - knotloc[i]) + 1.0/(knotloc[i] - knotloc[i-1]))
            h[i,i+1] = 1.0 / (knotloc[i+1] - knotloc[i])

        # LU Decomposition
        lu, piv = scipy.linalg.lu_factor(gee,overwrite_a=True)

        # Solve system
        for i in range(numknot):
            geeinvh[:,i] = scipy.linalg.lu_solve((lu,piv),h[:,i])

        # c is non-zero for derivative constraints
        geeinvc = scipy.linalg.lu_solve((lu,piv),c)

        d = a + np.dot(b,geeinvh)

        ap = np.zeros((nx,numknot))
        bp = np.zeros((nx,numknot))
        rowap = np.zeros(numknot)
        rowbp = np.zeros(numknot)

        for i in range(len(xcore)):
            rowap[:] = 0.0
            rowbp[:] = 0.0

            bt = numpy.where((knotloc >= xcore[i]) & (knotloc>0.0))[0]
            j = bt[0]

            # Eq. 3.3.5
            rowap[j-1] = -1.0/(knotloc[j] - knotloc[j-1])
            rowap[j] = -1.0*rowap[j-1]

            rowbp[j-1] = (knotloc[j] - knotloc[j-1])*(1.0 - 3.0*a[i,j-1]**2)/6.0
            rowbp[j] = (knotloc[j] - knotloc[j-1])*(3.0*a[i,j]**2 - 1.0)/6.0

            ap[i,:] = rowap
            bp[i,:] = rowbp

        dp = ap + np.dot(bp,geeinvh)
        dpp = np.dot(a,geeinvh)

        return d, dp, dpp, geeinvc, fx, b

    def get_knotvals(self, xcore, ycore, wgt, d, geeinvc, fx, b, knotloc, bcs):
        '''
        Get the spline y-values at the knot locations that best fit the data
        :param xdata: x values of measured data
        :param ydata: values of measured data
        :param wgt: weight of measured data
        :param knotloc: location of spline knots [0, ..., 1]
        :param bcs: dictionary of boundary conditions
        :param d, geeinvc, fx, b: Return values from design_matrix
        :return: values of the cubic interpolating spline at knot locations that best match the data
        '''

        nx = len(xcore)
        numknots = len(knotloc)

        dwgt = np.zeros((nx,numknots))
        sub = np.dot(b,geeinvc)
        for i in range(nx):
            dwgt[i,:] = d[i,:] * wgt[i]

        # Here is where normalization would happen
        dnorm = np.ones(nx)
        datatofit = dnorm*ycore

        ydatawgt = (datatofit - sub - np.dot(d,fx))*wgt

        if bcs['y0'] is not None:
            mini = 1
        else:
            mini = 0

        if bcs['y1'] is not None:
            maxi = len(knotloc)-1
        else:
            maxi = len(knotloc)

        decom = np.dot(dwgt[:,mini:maxi].T,dwgt[:,mini:maxi])

        # Make A = U s V'
        u, s, v = np.linalg.svd(decom)

        # Solve Ax = B using
        # C = u' B
        # S = solve(diag(s),C)
        B = np.dot(dwgt[:,mini:maxi].T, ydatawgt)
        C = np.dot(u.T, B)
        S = np.linalg.solve(np.diag(s),C)
        knotval = np.dot(v.T,S)
        if mini == 1:
            knotval = np.insert(knotval,0,bcs['y0'])
        if maxi == len(knotloc)-1:
            knotval = np.append(knotval,bcs['y1'])

        return knotval

    def get_spl(self, x, y, w, bcs, k):
        '''
        :param xdata: rho values
        :param ydata: data values
        :param wgt: data weight (1/uncertainty)
        :param knotloc: location of knots
        :param bcs: boundary conditions
        :return: spline fit on rho values
        '''

        # Get the design matrix
        d, dp, dpp, geeinvc, fx, b = self.design_matrix(x, k, bcs)

        knotval = self.get_knotvals(x, y, w, d, geeinvc, fx, b, k, bcs)

        yfit = np.dot(d,knotval) + np.dot(b,geeinvc)
        yfitp = np.dot(dp,knotval)
        yfitpp = np.dot(dpp,knotval)

        return knotval, yfit, yfitp, yfitpp

    def do_fit(self):

        def params_to_array(params):
            '''
            Convert params dictionary to array of knot locations
            :param params: lmfit parameters dictionary
            :return: knot locations numpy.ndarray
            '''
            keys = list(params.keys())
            if 'sepval' in keys:
                keys.remove('sepval')
            if 'scrapewidth' in keys:
                keys.remove('scrapewidth')

            knotloc = np.zeros(len(keys))
            for i,k in enumerate(keys):
                knotloc[i] = params[k].value

            sor = np.argsort(knotloc)
            knotloc = knotloc[sor]

            return knotloc

        def callback(params, iter, resid, *args, **kw):
            r"""
            :param params: parameter dictionary
            :param iter: iteration number
            :param resid: residual
            :param \*args: extra arguments
            :param \**kw: extra keywords
            :return:
            """
            p = np.zeros(len(params))
            for i,key in enumerate(params.keys()):
                p[i] = params[key].value

        def linearfit(params):
            '''
            Function to minimize for auto-knotting and edge fitting
            :param params: lmfit parameters dictionary
            :return: residual for least-squares minimization
            '''

            # Turn lmfit params into array
            knotloc = params_to_array(params)

            # Store the knot locations for each iteration of the solver.
            if self.iknotloc is None:
                self.iknotloc = np.atleast_2d(knotloc)
            else:
                self.iknotloc = np.append(self.iknotloc,knotloc[np.newaxis,:],axis=0)

            if self.sol:
                # The core part
                x = self.xdatafit[self.edgemask]
                y = self.ydatafit[self.edgemask]
                w = self.ydatafitwgt[self.edgemask]

                # Set the boundary conditions to the least-squares current iteration
                self.bcs['y1'] = params['sepval'].value

                # Do the core fit with the specified knots and boundary
                knotval, yfit, yfitp, yfitpp = self.get_spl(x,y,w,self.bcs,knotloc)

                # Must make sure that we evaluate the core spline at x=1.0 for the exponential decay
                # to be accurately captured starting at the boundary in the least-squares
                if x[-1] != 1.0:
                    xx = np.append(x,1.0)
                    d, dp, dpp, geeinvc, fx, b = self.design_matrix(xx,knotloc,self.bcs)
                    tmp = np.dot(d,knotval) + np.dot(b,geeinvc)
                else:
                    tmp = y

                # Compute the edge part
                scrapewidth = params['scrapewidth'].value
                yfitedge = tmp[-1]*np.exp((1.0-self.xdataedge)/scrapewidth)

                # Set the edge derivative to the least-squares current iteration
                if self.solc:
                    self.bcs['y1p'] = -(tmp[-1]/scrapewidth)

                # Do the fit again with constrained edge value and derivative
                knotval, yfit, yfitp, yfitpp = self.get_spl(x,y,w,self.bcs,knotloc)

                # Form residual
                resid = (y - yfit)*w
                resid = np.append(resid,(self.ydataedge - yfitedge)*self.ydataedgewgt)

                # Number of observations
                ndata = len(x) + len(self.xdataedge)

                # Number of core variables in the number of internal knots
                nvarys = len(knotloc) - 2
                # Add the two edge values as a free parameters
                nvarys += 2

            else:
                x = self.xdatafit
                y = self.ydatafit
                w = self.ydatafitwgt

                bcs = self.bcs
                knotval, yfit, yfitp, yfitpp = self.get_spl(x,y,w,bcs,knotloc)

                # Form residual
                resid = (y - yfit)*w

                # Number of observations
                ndata = len(x)

                # Number of variables is the number of knots minus the two knots at the boundaries
                nvarys = len(knotloc) - 2

            # Degrees of freedom is the number of data points less the number of variables
            nfree = ndata - nvarys

            # Store reduced chi2 as a function of iteration
            if self.irchi2 is None:
                self.irchi2 = np.atleast_1d(np.sum(resid**2)/nfree)
            else:
                self.irchi2 = np.append(self.irchi2, np.sum(resid**2)/nfree)
            return resid

        # If there is no knot bias then it is uniform or user, else we bias it to the edge
        if self.knotbias == 0:
            # If there are no knot locations then they are uniformly spaced
            if self.knotloc is None:
                self.knotloc = numpy.linspace(0,1,self.numknot)
        else:
            knottmp = 1.0/numpy.linspace(1,self.numknot,self.numknot)**self.knotbias
            self.knotloc = np.abs((knottmp-knottmp[0])/(knottmp[-1]-knottmp[0]))

        # Create parameters for lmfit
        params = lmfit.Parameters()
        for i in range(len(self.knotloc)):
            params.add('k{}'.format(i),value=self.knotloc[i], min=0.01, max=0.99)
            if not self.autoknot:
                params['k{}'.format(i)].vary = False
        # rho locations [0, 1] always fixed
        params['k0'].vary = False
        params['k0'].min = 0.0
        params['k0'].max=1.0
        params['k0'].value = 0.0
        params['k{}'.format(self.numknot-1)].vary = False
        params['k{}'.format(self.numknot-1)].min = 0.0
        params['k{}'.format(self.numknot-1)].max = 1.0
        params['k{}'.format(self.numknot-1)].value = 1.0
        if self.sol:
            params.add('scrapewidth')
            params['scrapewidth'].vary = True
            params['scrapewidth'].value = 0.1
            if self.scrapewidth_min is not None:
                params['scrapewidth'].min = self.scrapewidth_min
            if self.scrapewidth_max is not None:
                params['scrapewidth'].max = self.scrapewidth_max

            params.add('sepval')
            params['sepval'].vary = True
            params['sepval'].value = 0.1*np.mean(self.ydatacore)
            if self.sepval_min is not None:
                params['sepval'].min = self.sepval_min
            if self.sepval_max is not None:
                params['sepval'].max = self.sepval_max

        ####
        # Perform the fit
        ####
        fitter = lmfit.Minimizer(linearfit,params,iter_cb=callback)
        if self.method == 'leastsq':
            out = fitter.minimize(method='leastsq',maxfev=self.maxiter)
        else:
            fit_kws = {'options':{'maxiter':self.maxiter}}
            # Note that in general the lmfit MinimizerResult object cannot be stored
            out = fitter.minimize(method=self.method,**fit_kws)

        ####
        # Process and gather the fit outputs
        ####
        if self.verbose:
            print(lmfit.fit_report(out))

        # Store number of variables
        self.nvarys = out.nvarys

        # Store the number of degrees of freedom (ndata - nvarys)
        self.nfree = out.nfree

        # Store the knot locations and the scrape off layer width
        self.knotloc = params_to_array(out.params)
        if self.sol:
            self.scrapewidth = out.params['scrapewidth'].value

        ####
        # Get the spline fit on the data coordinates and rchi2
        ####
        if self.sol:
            x = self.xdatafit[self.edgemask]
            y = self.ydatafit[self.edgemask]
            w = self.ydatafitwgt[self.edgemask]
        else:
            x = self.xdatafit
            y = self.ydatafit
            w = self.ydatafitwgt
        knotval, yfit, yfitp, yfitpp = self.get_spl(x, y, w, self.bcs, self.knotloc)
        self.knotval = knotval
        if self.sol:
            if x[-1] != 1.0:
                x = np.append(x,1.0)
                d, dp, dpp, geeinvc, fx, b = self.design_matrix(x,self.knotloc,self.bcs)
                tmp = np.dot(d,knotval) + np.dot(b,geeinvc)
            self.yfit = np.append(yfit,tmp[-1]*np.exp((1.0-self.xdataedge)/self.scrapewidth))
        else:
            self.yfit = yfit
        self.resid = (self.ydatafit - self.yfit)*self.ydatafitwgt
        self.rchi2 = np.sum(self.resid**2) / self.nfree

        ####
        # Get the spline fit on the user axis coordinates
        ####
        edgemask = (self.xufit<=1.0)
        coremask = (self.xufit>1.0)
        d, dp, dpp, geeinvc, fx, b = self.design_matrix(self.xufit[edgemask],self.knotloc,self.bcs)
        yufit = np.dot(d,self.knotval) + np.dot(b,geeinvc)
        ypufit = np.dot(dp,self.knotval)
        if self.sol:
            self.yufit = np.append(yufit,yufit[-1]*np.exp((1.0-self.xufit[coremask])/self.scrapewidth))
            self.ypufit = np.append(ypufit,-1.0*yufit[-1]*np.exp((1.0-self.xufit[coremask])/self.scrapewidth)/self.scrapewidth)
        else:
            self.yufit = np.append(yufit,np.repeat(numpy.nan, np.sum(coremask)))
            self.ypufit = np.append(ypufit,np.repeat(numpy.nan, np.sum(coremask)))

    def plot(self):
        # Plot the data and fit
        from omfit_plot import FigureNotebook, uband, uerrorbar
        fn = FigureNotebook('GASpline')
        fig, ax = fn.subplots(nrows=2,sharex=True,label='Data and Fit')
        uerrorbar(self.xdata,uarray(self.ydata,self.ydataerr),ax=ax[0],markersize=3.0,label='All data')
        ax[0].plot(self.xdatafit, nominal_values(self.yfit),marker='o',markersize=3.0,label='Fit on Data')
        ax[0].plot(self.xufit, nominal_values(self.yufit),label='Fit on User')
        ax[0].plot(self.knotloc,self.knotval,marker='o',label='Knotloc,val')
        ax[0].legend()
        ax[1].plot(self.xdatafit,self.resid,marker='o',label='Weighted Residual')
        ax[1].axhline(0.0,color='k',ls='dashed')
        ax[1].legend()
        fig, ax = fn.subplots(nrows=2,sharex=True,label='Convergence')
        ax[0].plot(self.iknotloc,marker='o', mec='none')
        ax[0].set_ylabel('Knot Locations')
        ax[1].plot(self.irchi2,marker='o', mec='none')
        ax[1].set_ylabel('Reduced chi2')
        ax[1].set_xlabel('Iteration')
        ax[1].set_yscale('log')
        if self.monte is not None and self.monte > 0:
            fig, ax = fn.subplots(nrows=2,sharex=True,label='Monte-Carlo Fits')
            for i in range(self.monte):
                uerrorbar(self.xdatafit, uarray(self.ydatafits[:, i], self.ydatafiterr), ax=ax[0], markersize=3.0,
                          alpha=0.2, markeredgecolor='none')
                ax[0].plot(self.xdatafit, self.yfits[:,i],color='black')
                ax[0].plot(self.knotlocs[:,i], np.zeros(self.numknot), marker='o', ls='None', color='black')
                ax[1].plot(self.xufit, self.ypufits[:,i], color='black')
            fig, ax = fn.subplots(nrows=2,sharex=True,label='Monte-Carlo Results')
            uerrorbar(self.xdata, uarray(self.ydata, self.ydataerr), ax=ax[0], markersize=3.0, label='Data',
                      markeredgecolor='none')
            for i in range(self.numknot):
                ax[0].axvline(self.knotloc[i], color='k', ls='dashed', label='_' * (i==0) + 'Knot Location')
                ax[1].axvline(self.knotloc[i], color='k', ls='dashed')
            for i in range(self.monte):
                ax[0].plot(self.xufit, self.yufits[:,i],color='black',alpha=0.2,ls='dashed')
            uband(self.xufit, uarray(self.yufit, self.yufiterr),ax=ax[0], label='Spline fit')
            ax[0].legend(loc='best', frameon=False)
            for i in range(self.monte):
                ax[1].plot(self.xufit, self.ypufits[:,i],color='black',alpha=0.2,ls='dashed')
            ax[1].plot([0], [0], lw=0)  # dummy to sync colors
            uband(self.xufit, uarray(self.ypufit, self.ypufiterr), ax=ax[1], label='Spline derivative')
            # This line below shows why the numerical uncertainties derivative is incorrect.
            # uerrorbar(self.xufit, deriv(self.xufit, uarray(self.yufit, self.yufiterr)),ax=ax[1], color='black',alpha=0.2)
            ax[1].legend(loc='best', frameon=False)

    def __call__(self, x=None, n=0):
        """
        Returns y values of the spline fit.

        :param x: numpy.ndarray. Must be subset of the fit x. Default is all fit x.
        :param n: int. Order of the derivative returned
        :return: uarray. nth derivative of fit at x
        """
        if x is None:
            x = self.xufit * 1.0
        if not set(x) <= set(self.xufit):
            raise ValueError('Requested x must be in fit x')
        if n == 0:
            if self.monte is not None and self.monte > 0:
                return uarray(self.yufit, self.yufiterr)
            else:
                return uarray(self.yufit, self.yufit * 0)
        elif n == 1:
            if self.monte is not None and self.monte > 0:
                return uarray(self.ypufit, self.ypufiterr)
            else:
                return uarray(self.ypufit, self.ypufit * 0)
        else:
            raise ValueError('Only returns values and first order derivatives')


@_available_to_user_math
class MtanhPolyExpFit(object):
    """
    Generalized fitter derived from B. Grierson tools
    fits core with pow_core polynomial C(x), and
    edge with offset exponential of form
    E(x) = offset + A*np.exp(-(x - xsym)/edge_width)
    blends functions together about x=x_sym with tanh-like behavior
    y_fit = (C(x)*np.exp(z) + E(x)*np.exp(-z))/(np.exp(z) + np.exp(-z))
    where z = (xsym - x)/blend_width

    :param method: minimization method to use

    :param verbose: turns on details of set flags

    :param onAxis_gradzero: turn on to force C'(0) = 0 (effectively y'(0) for xsym/blend_width >> 1)

    :param onAxis_value: set to force y(0) = onAxis_value

    :param fitEdge: set = False to require E(x) = offset, A=edge_width=0

    :param edge_value: set to force y(x=1) = edge_value

    :param maxiter: controls maximum # of iterations if 'leastsq' used.  Set = -1 to use defaults.

    :param blend_width_min: minimum value for the core edge blending

    :param edge_width_min: minimum value for the edge

    :param sym_guess: guess for the x location of the pedestal symmetry point

    :param sym_min: constraint for minimum x location for symmetry point

    :param sym_max: constraint for maximum x location for symmetry point

    :Methods:

    :__call__(x): Evaluate mtanh_polyexp at x, propagating correlated uncertainties in the fit parameters.


    """

    def __init__(self, x_data, y_data, y_err, pow_core=2, method='leastsq', verbose=False, onAxis_gradzero=False,
                 onAxis_value=None, fitEdge=True, edge_value=None, maxiter=None, blend_width_min=1.e-3, edge_width_min=1.e-3,
                 edge_width_max=None, sym_guess=0.975, sym_min=0.9, sym_max=1.0):

        # Create input parameters list
        params = lmfit.Parameters()

        # Core poly initial guess = max(y_data)*(1 - x^2)
        # Note onAxis_value will overwrite core[0]
        params.add('pow_c', value=pow_core, vary=False)
        core = np.zeros(pow_core+1)
        core[:] = 1.
        if pow_core >= 2:
            core[2] = -1.
        maxy = y_data[np.argmax(abs(y_data))]  # Keeps sign to handle negative profiles e.g V_tor
        miny = y_data[np.argmin(abs(y_data))]
        core *= maxy
        for i in range(len(core)):
            params.add('core{}'.format(i),value=core[i])

        # "blending" and edge parameters
        params.add('xsym', value=sym_guess, min=sym_min, max=sym_max)
        params.add('blend_width', value=max(0.05, 1.5*blend_width_min), min=blend_width_min)
        if edge_width_max is None:
            params.add('edge_width', value=max(0.05, 1.5*edge_width_min), min=edge_width_min)
        else:
            params.add('edge_width', value=0.5*(edge_width_max+edge_width_min), min=edge_width_min, max=edge_width_max)
        params.add('offset', value=miny)
        params.add('A', value=(maxy+miny)/2.)

        # Add some dependant parameters for use in constraints
        params.add('em2z0', expr='exp(-2*xsym/blend_width)')
        params.add('E0', expr='offset + A*exp(xsym/edge_width)')
        params.add('E0p', expr='-A*exp(xsym/edge_width)/edge_width')
        params.add('y0', expr='(core0 + E0*em2z0)/(1. + em2z0)')

        params.add('em2z1', expr='exp(-2*(xsym-1)/blend_width)')
        params.add('E1', expr='offset + A*exp((xsym-1)/edge_width)')
        core_sum = '+'.join(['core{}'.format(i) for i in range(pow_core+1)])
        params.add('y1', expr='('+core_sum+' + E1*em2z1)/(1. + em2z1)')

        # Set up constraints
        if onAxis_value is not None:
            if verbose:
                print('constrain on-axis value = %f'%onAxis_value)
            params['y0'].set(value=onAxis_value, vary=False)
            params['core0'].set(expr='(1. + em2z0)*y0 - E0*em2z0')
        else:
            if verbose:
                print('on-axis value unconstrained')

        if onAxis_gradzero and ('core1' in params):
            if verbose:
                print('constrain on-axis gradient = 0')
            params['core1'].set(expr='(-E0p -(2*E0)/blend_width + (2*y0)/blend_width)*em2z0')
        else:
            if verbose:
                print('on-axis gradient unconstrained')

        if edge_value is not None:
            if verbose:
                print('constrain edge value = %f'%edge_value)
            params['y1'].set(value=edge_value, vary=False)
            core_sub = '-'.join(['core{}'.format(i) for i in range(pow_core)])
            params['core{}'.format(pow_core)].set(expr='(1. + em2z1)*y1 - E1*em2z1 - '+core_sub)
        else:
            if verbose:
                print('edge value unconstrained')

        if (not fitEdge) or (max(x_data) <= 1.):
            # Definitely needs better tuning for initial guesses
            # turns out better to keep A and edge width even if only doing x <= 1
            if verbose:
                print('only fitting x <= 1')  # ; setting A=0,edge_width=blend_width'
            # params['A'].set(value=0, vary=False)
            # params['edge_width'].set(expr='blend_width')  # since A=0, value doesn't matter
            if maxy >= 0.0:
                params['A'].set(min=0.)
                params['offset'].set(min=0.)
            else:
                params['A'].set(max=0.)
                params['offset'].set(max=0.)

            idx = numpy.where(x_data <= 1.)
            x_core = x_data[idx]
            y_core = y_data[idx]
            err_core = y_err[idx]

            fitter = lmfit.Minimizer(mtanh_polyexp_res, params, fcn_args=(x_core, y_core, err_core))
        else:
            fitter = lmfit.Minimizer(mtanh_polyexp_res, params, fcn_args=(x_data, y_data, y_err))

        if (method == 'leastsq') and (maxiter != -1):
            if verbose:
                print('Using maximum of %r iterations'%maxiter)
            fitout = fitter.minimize(method=method, params=params, maxfev=maxiter)
        else:
            fitout = fitter.minimize(method=method, params=params)

        # force errors to zero for now. todo: proper errorbars.
        fitout.errorbars = False

        # store the MinimizerResult and make its properties directly accessible in this class
        self._MinimizerResult = fitout
        self.__dict__.update(fitout.__dict__)

    def __call__(self, x):
        """
        Generate mtanh_poly including correlated uncertainties from the lmfit
        MinimizerResult formed at initialization.

        :param x: numpy.ndarray. Output grid.
        :return y: UncertaintiesArray. Fit values at x.

        """
        lmfit_out = self._MinimizerResult
        # error check- if no error bars, just return function w/o error bars
        if (lmfit_out.errorbars == False):
            y_fit = mtanh_polyexp(x, lmfit_out.params)
            return y_fit

        # take from OMFIT fit_base object
        uparams = {}
        for k in lmfit_out.params:
            uparams[k] = lmfit_out.params[k].value
        if lmfit_out.errorbars:
            corr_params = uncertainties.correlated_values([lmfit_out.params[k].value for k in lmfit_out.var_names],
                                                          lmfit_out.covar)
            for vi, var in enumerate(lmfit_out.var_names):
                uparams[var] = corr_params[vi]

        nc = uparams['pow_c'] + 1  # always a fixed value
        core = np.zeros(nc)

        # write like this to ensure starts as complex array in case on-axis value set
        from uncertainties import ufloat
        core_poly = np.zeros(len(x)) + ufloat(0., 0)
        core_poly += uparams['core0']
        for ii in range(1, nc):
            core_poly += uparams['core%s' % ii] * (x ** ii)

        xsym = uparams['xsym']
        blend_width = uparams['blend_width']
        edge_width = uparams['edge_width']
        offset = uparams['offset']
        A = uparams['A']
        edge_func = offset + A * unumpy.exp((xsym - x) / edge_width)

        z = (xsym - x) / blend_width

        y_fit = (core_poly * unumpy.exp(z) + edge_func * unumpy.exp(-z)) / (unumpy.exp(z) + unumpy.exp(-z))

        return y_fit


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
                    numpy.array(N,X): user-defined array with X coordinates of the N nodes

    :param epsilon: float Adjustable constant for gaussian - defaults to approximate average distance between nodes

    :param function: 'multiquadric': numpy.sqrt((r / self.epsilon) ** 2 + 1)   #<--- default
                     'inverse': 1.0 / numpy.sqrt((r / self.epsilon) ** 2 + 1)
                     'gaussian': np.exp(-(r**2 / self.epsilon))
                     'linear': r
                     'cubic': r ** 3
                     'quintic': r ** 5
                     'thin_plate': r ** 2 * numpy.log(r)

    :param norm: default "distance" is the euclidean norm (2-norm)

    :Examples:

    >>> x=numpy.linspace(0,1,21)
    >>> y=numpy.linspace(0,1,21)
    >>> e=x*0+.5
    >>> e[abs(x-0.5)<0.1]=8
    >>> y[abs(x-0.5)<0.1]=10
    >>>
    >>> x1 = numpy.linspace(0,1,100)
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
        if not numpy.all([xx.shape == d.shape for xx in x]) and (e.shape == d.shape):
            raise ValueError("Array lengths must be equal")

        # npts by ndim
        self.xi = np.asarray([np.asarray(a, dtype=np.float_).flatten() for a in x]).T
        self.di = np.atleast_2d(np.asarray(d).flatten()).T
        self.ei = np.atleast_2d(np.asarray(e).flatten()).T

        self.indim = self.xi.shape[1]
        self.outdim = self.di.shape[1]

        if centers is None:
            self.centers=self.xi
        elif numpy.iterable(centers):
            self.centers=centers
        elif centers>0:
            self.centers=numpy.array([x.flatten() for x in meshgrid(*[numpy.linspace(min(self.xi[:,k]),max(self.xi[:,k]),centers) for k in  range(self.indim)])]).T
        else:
            self.centers=numpy.array([numpy.random.uniform(min(self.xi[i]), max(self.xi[i]), self.indim) for i in range(abs(centers))])
        self.numCenters = self.centers.shape[0]
        self.centers = self.centers.T # ndim by numcenters

        # default "distance" is the euclidean norm (2-norm)
        if norm is None:
            self.norm = lambda ctrs, pts: np.linalg.norm(ctrs[:,np.newaxis,...] - pts[...,numpy.newaxis], axis=0)
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
        function_options = {'multiquadric': lambda r: numpy.sqrt((r / self.epsilon) ** 2 + 1),
                            'inverse': lambda r: 1.0 / numpy.sqrt((r / self.epsilon) ** 2 + 1),
                            'gaussian': lambda r: np.exp(-(r**2 / self.epsilon)),
                            'linear': lambda r: r,
                            'cubic': lambda r: r ** 3,
                            'quintic': lambda r: r ** 5,
                            'thin_plate': lambda r: r ** 2 * numpy.log(r)}
        if isinstance(function, str):
            self.fun = function_options[function]  # demands it be one of the defined keys
        else:
            self.fun = function

        G = self._calcAct(self.xi.T)
        D = np.linalg.pinv(np.matrix(G/self.ei[np.newaxis,:]))
        self.Wd = np.dot(D, self.di/self.ei)
        self.We = np.dot(D, self.ei*0+1.)

    def _calcAct(self, X):
        """
        Calculate the radial basis function of "radius" between points X and the centers.

        :param X: numpy.ndarray. Shape (ndims, npts)

        :return: numpy.ndarray. Shape (npts, numcenters)
        """
        return self.fun(self.norm(self.centers, X))

    def __call__(self, *args):
        r"""
        evaluate interpolation at coordinates

        :param \*args: arrays `x, y, z, ...` the coordinates of the nodes

        :return: uncertain array, of the same shape of coordinates
        """
        args = [np.asarray(x) for x in args]
        if not numpy.all([x.shape == y.shape for x in args for y in args]):
            raise ValueError("Array lengths must be equal")
        shp = args[0].shape
        X = np.asarray([a.flatten() for a in args], dtype=np.float_) # ndim, npts
        G = self._calcAct(X)

        return uarray(numpy.array(np.dot(G, self.Wd)),numpy.array(abs(np.dot(G, self.We)))).reshape(shp)

# ------------
# More fitting
# ------------
@_available_to_user_math
def bimodal_gaussian(xx, center0, center1, sigma0, sigma1, amp0, amp1, delta0=None, delta1=None, debug=False):
    """
    Calculates bimodal gaussian function. There are two gaussians added together.

    :param xx: Independent variable

    :param center0: Center of first gaussian before log transform

    :param center1: Center of second gaussian before log transform

    :param sigma0: Sigma of first gaussian before log transform

    :param sigma1: Sigma of second gaussian before log transform

    :param amp0: Amplitude of first gaussian

    :param amp1: Amplitude of second gaussian

    :param delta0: The fitter uses this variable to help it set up limits internally. The deltas are not actually used in the model.

    :param delta1: Not used in the model; here to help the fitter.

    :param debug: T/F: print debugging stuff (keep this off during fit, but maybe on for testing)

    :return: Model function y(x)
    """

    # Calculate arguments for the exponentials
    arg0 = ((xx - center0) / sigma0) ** 2 / 2.0
    arg1 = ((xx - center1) / sigma1) ** 2 / 2.0

    # Floating underflow protection: enforce limits
    pad = max([abs(amp0 * 2), abs(amp1 * 2)])  # Padding gives a margin below overflow limit to allow for amplitude
    minpad = 1e3
    if pad < minpad:
        pad = minpad
    pad *= 1e4  # Pad a little more
    too_big = sys.float_info.max / pad  # Take the operating systems floating max and reduce it by pad
    big = numpy.log(too_big)  # We will use np.exp() on arg, so take numpy.log(too_big) to compare to arg before np.exp()

    def apply_limits(c):
        c = numpy.atleast_1d(c)
        c[c > big] = big
        c[c < -big] = -big
        return c

    arg0 = apply_limits(arg0)
    arg1 = apply_limits(arg1)

    if debug:
        printd('  Calculating model...')
        printd('  arg0 : min = {:}, mean = {:}, max = {:}'.format(arg0.min(), numpy.mean(arg0), arg0.max()))
        printd('  arg1 : min = {:}, mean = {:}, max = {:}'.format(arg1.min(), numpy.mean(arg1), arg1.max()))
        printd('  amp0 = {:}, amp1 = {:}'.format(amp0, amp1))
    model_y = np.exp(-arg0) * amp0 + np.exp(-arg1) * amp1
    if debug:
        printd('  Calculated model: min = {:}, mean = {:}, max = {:}'.format(
            model_y.min(), numpy.mean(model_y), model_y.max())
        )

    return model_y

@_available_to_user_math
class BimodalGaussianFit(object):
    """
    The fit model is a sum of two Gaussians. If a middle value is provided, limits will be imposed to try to keep the
    Gaussians separated.
    """

    default_guess = {}

    def __init__(self, x=None, pdf=None, guess=default_guess, middle=None, spammy=False, limits=None):
        """
        Initialize variables and call functions

        :param x: Independent variable
        :param pdf: Probability distribution function
        :param guess: Guess for parameters. Use {} or leave as default for auto guess.
        :param middle: X value of a dividing line that is known to be between two separate Gaussians.
            Sets up additional limits on centers. No effect if None.
        :param spammy: Many debugging print statements
        :param limits: None for default limits or dictionary with PARAM_LIM keys where PARAM is center0, center1, etc.
            and LIM is min or max
        """
        from lmfit import Model

        self.x = x
        self.pdf = pdf
        self.middle = middle
        self.spammy = spammy
        self.limits = limits

        self.guess = guess
        self.make_guess()

        self.model = Model(bimodal_gaussian)
        self.result = self.bimodal_gaussian_fit()

    def make_guess(self):
        """
        Given a dictionary with some guesses (can be incomplete or even empty), fills in any missing values with
        defaults, makes consistent deltas, and then defines a parameter set.

        Produces a parameter instance suitable for input to lmfit .fit() method and stores it as self.guess.
        """

        from lmfit import Parameters

        # Make default guesses
        if self.middle is not None \
                and numpy.any(numpy.atleast_1d(self.x <= self.middle)) \
                and numpy.any(numpy.atleast_1d(self.x >= self.middle)):
            printd('middle was defined: {:}'.format(self.middle))
            amp0g = pdf[self.x <= self.middle].max()
            amp1g = pdf[self.x >= self.middle].max()
            cen0g = self.x[self.pdf[self.x <= self.middle].argmax()]
            cen1g = self.x[self.pdf[self.x >= self.middle].argmax() + numpy.where(self.x >= self.middle)[0][0]]
            sig1g = cen1g/8.
            if sig1g < 0.5:
                sig1g = 0.5
            if sig1g > 20:
                sig1g = 20
            c0max = self.middle*0.9
            c1min = self.middle*1.1
            d1min = self.middle
        else:
            amp0g = self.pdf.max()
            amp1g = self.pdf.max()
            cen0g = self.x[self.pdf.argmax()]
            cen1g = 15.
            sig1g = 1.5
            c0max = self.x.max()
            c1min = self.x.min()
            d1min = 2
        c0min = 0.1
        c1max = self.x.max()*2
        sig0g = cen0g/6.
        if sig0g < 0.2:
            sig0g = 0.2
        if sig0g > 10:
            sig0g = 10
        # Define limits
        min_amp = self.pdf.max()/25.
        max_amp = self.pdf.max()*4.

        if self.limits is None:
            self.limits = {}
        c0min = self.limits.get('center0_min', c0min)
        c0max = self.limits.get('center0_max', c0max)
        c1min = self.limits.get('center1_min', c1min)
        c1max = self.limits.get('center1_max', c1max)
        a0min = self.limits.get('amp0_min', min_amp)
        a0max = self.limits.get('amp0_max', max_amp)
        a1min = self.limits.get('amp1_min', min_amp)
        a1max = self.limits.get('amp1_max', max_amp)
        d0min = self.limits.get('delta0_min', 0)
        d0max = self.limits.get('delta1_min', c0max)
        d1min = self.limits.get('delta1_min', d1min)
        d1max = self.limits.get('delta1_min', c1max)

        printd('initial guess for center1, cen1g = {:}'.format(cen1g))

        # Fill in dictionary of guesses
        self.guess.setdefault('center0', cen0g)
        self.guess.setdefault('center1', cen1g)
        self.guess.setdefault('amp0', amp0g)
        self.guess.setdefault('amp1', amp1g)
        if 'delta0' not in list(self.guess.keys()):
            self.guess.setdefault('sigma0', sig0g)
        else:
            self.guess.setdefault('sigma0', self.guess['center0']-self.guess['delta0'])
        # Update delta0, which might change delta0 if inconsistent delta0 & sigma0 were supplied.
        printd('guess = {:}'.format(self.guess))
        self.guess['delta0'] = self.guess['center0'] - self.guess['sigma0']
        if 'delta1' not in list(self.guess.keys()):
            self.guess.setdefault('sigma1', sig1g)
        else:
            self.guess.setdefault('sigma1', self.guess['center1']-self.guess['delta1'])
        # Update delta0, which might change delta0 if inconsistent delta0 & sigma0 were supplied.
        self.guess['delta1'] = self.guess['center1'] - self.guess['sigma1']

        pars = Parameters()
        pars.add('amp0', value=self.guess['amp0'], vary=True, min=a0min, max=a0max)
        pars.add('amp1', value=self.guess['amp1'], vary=True, min=a1min, max=a1max)
        pars.add('center0', value=self.guess['center0'], vary=True, min=c0min, max=c0max)
        pars.add('center1', value=self.guess['center1'], vary=True, min=c1min)
        pars.add('delta0', value=self.guess['delta0'], vary=True, min=d0min, max=d0max)
        pars.add('delta1', value=self.guess['delta1'], vary=True, min=d1min, max=d1max)
        pars.add('sigma0', expr='center0-delta0')
        pars.add('sigma1', expr='center1-delta1')
        pars.add('debug', value=self.spammy, vary=False)

        self.guess = pars

    def bimodal_gaussian_fit(self):
        """
        Fits a probability distribution function (like a histogram output, maybe) with a two gaussian function
        :return: Minimizer result
        """

        # Do the fit
        printd('fitting...')
        result = self.model.fit(
            self.pdf,
            xx=self.x,  # ms
            params=self.guess,
            method='nelder')

        return result
