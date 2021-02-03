import numpy as np
from numba import njit

# lpf = linearized position fit
#
# A.P. Colijn - Feb. 2021
lpf_iter_max = 100


@njit
def lpf_lnlike(xhit, nhit, xf, nuv, area_sensor):
    r0 = nuv * area_sensor * xhit[0][2] / 4 / np.pi
    logl = 0

    # print('next')
    for ih in range(len(nhit)):
        nexp = lpf_nexp(xhit[ih], xf, r0)
        # print(xhit[ih], nexp, xf, r0)

        logl = logl + nexp - nhit[ih] * np.log(nexp)

    return logl


@njit
def lpf_lnlike_r(xhit, nhit, xf, r0):
    logl = 0

    # print('next')
    for ih in range(len(nhit)):
        nexp = lpf_nexp(xhit[ih], xf, r0)
        # print(xhit[ih], nexp, xf, r0)

        logl = logl + nexp - nhit[ih] * np.log(nexp)

    return logl


@njit
def lpf_initialize(xhit, nhit):
    """
    Initialize the lpf_fitter. Estimate the initial position and uv photons

    :param xhit:
    :param nhit:
    :return:
    """
    nmax = -1
    nn = 0
    xfit = np.zeros(3)
    for ihit in range(len(nhit)):
        nh = nhit[ihit]
        if nh > -1:
            if nh > nmax:
                # initialize fit to position under light detector with highest signal
                nmax = nh
                xfit[0] = xhit[ihit][0]
                xfit[1] = xhit[ihit][1]

        if nh > 1:
            nn = nn + 1

    if nn < 3:
        print("lpf_initialize:: WARNING not sensors hit. n=", nn)

    r0 = nmax * xhit[0][2] ** 3
    xfit[2] = 0

    # r0 = 1000000 * area_sensor * xhit[0][2] / 4 / np.pi

    logl_min = 1e12
    xtemp = np.zeros(3)
    for xxx in np.arange(xfit[0] - 50, xfit[0] + 50, 5.):
        for yyy in np.arange(xfit[1] - 50, xfit[1] + 50, 5.):
            # print('xxx',xxx,'yyy',yyy)
            xtemp[0] = xxx
            xtemp[1] = yyy
            logl = lpf_lnlike_r(xhit, nhit, xtemp, r0)

            if logl < logl_min:
                logl_min = logl
                xfit[0] = xxx
                xfit[1] = yyy

    return xfit, r0


@njit
def lpf_minimize(xhit, nhit, area_sensor):
    """
    Linearized -log(L) fit for S2 position

    :param xhit: positions of the sensors (list with 3D arrays with length n-sensor)
    :param nhit: number of p.e. on each sensor (length n-sensor)
    :param area_sensor: area of teh sensor
    :return: xf: fitted position
    :return: xiter: intermediate fit results
             nuv: number of emitted photons
    """
    # initialize
    #
    xfit, r0 = lpf_initialize(xhit, nhit)

    # arrays to store teh fit resuults for each iteration
    xiter = np.zeros((lpf_iter_max + 1, 4))
    xiter[0][0] = xfit[0]
    xiter[0][1] = xfit[1]
    xiter[0][2] = r0
    xiter[0][3] = lpf_lnlike_r(xhit, nhit, xfit, r0)

    # iterate & minimize
    #
    for lpf_iter in range(lpf_iter_max):
        # initialize error matrix and vector
        #
        g = np.zeros(3)
        m = np.zeros((3, 3))

        # calculate the sums
        #
        for isensor in range(len(nhit)):
            for i in range(3):
                g[i] = g[i] + lpf_f(i, xhit[isensor], nhit[isensor], xfit, r0)
                for j in range(3):
                    m[i][j] = m[i][j] + lpf_deriv_f(i, j, xhit[isensor], nhit[isensor], xfit, r0)

        # invert the matrix
        #
        minv = np.linalg.inv(m)
        # multiply with vector to get corrections to the current fit parameters
        #
        # result = minv.dot(g)
        result = np.dot(minv, g)

        # if abs(result[1])>1000:
        #    print('WARNING:: result = ',result)

        # update fit result
        xfit[0] = xfit[0] - result[0]
        xfit[1] = xfit[1] - result[1]
        r0 = r0 - result[2]

        xiter[lpf_iter + 1][0] = xfit[0]
        xiter[lpf_iter + 1][1] = xfit[1]
        xiter[lpf_iter + 1][2] = r0
        xiter[lpf_iter + 1][3] = lpf_lnlike_r(xhit, nhit, xfit, r0)

        if (abs(result[0]) < 0.1) and abs(result[1] < 0.1):  # if position no longer changes -> terminate loop
            break

    # calculate the number of uv photons
    #
    nuv = 4 * np.pi * r0 / area_sensor / xhit[0][2]
    return xfit[0], xfit[1], nuv, xiter


@njit
def lpf_dist(x0, x1):
    d2 = (x0[0] - x1[0]) ** 2 + (x0[1] - x1[1]) ** 2 + (x0[2] - x1[2]) ** 2
    return np.sqrt(d2)


@njit
def lpf_nexp(xi, xf, r0):
    #
    # calculate the expected number of p.e. for a lightsensor
    #
    delta = lpf_dist(xi, xf)
    nexp = r0 / delta ** 3

    return nexp


@njit
def lpf_f(i, xi, ni, xf, r0):
    """
    Calculate the minimizer functions
    :param i:
    0=F0 (x)
    1=F1 (y)
    2=F2 (r0)
    :param xi: sensor position
    :param ni: hits for the sensor
    :param xf: assumed fit position
    :param r0: assumed number of UV photons (re-normalized)
    :return f: function value
    """

    f = 0
    if i < 2:
        f = -3 * (xf[i] - xi[i]) * (lpf_nexp(xi, xf, r0) - ni) / lpf_dist(xi, xf) ** 2
    elif i == 2:
        f = (lpf_nexp(xi, xf, r0) - ni) / r0

    return f


@njit
def lpf_deriv_f(i, j, xi, ni, xf, r0):
    """
    Derivatives of the minimizer functions

    :param i:
    0=F0
    1=F1
    2=F2
    :param j:
    0=x
    1=y
    z=r0
    :param xi: hit position
    :param ni: number of hits
    :param xf: fit position
    :param r0: number of photons
    :return:
    """

    d = lpf_dist(xi, xf)
    n0 = lpf_nexp(xi, xf, r0)

    deriv = 0
    if i == 0:
        dx = xf[0] - xi[0]
        if j == 0:  # dF0/dx
            deriv = -3 * (n0 - ni) / d ** 2
            deriv = deriv - 3 * dx * (n0 - ni) * lpf_deriv_dist_min2(0, xi, xf)
            deriv = deriv - 3 * dx * lpf_deriv_n(0, xi, xf, r0) / d ** 2
        elif j == 1:  # dF0/dy
            deriv = - 3 * dx * (n0 - ni) * lpf_deriv_dist_min2(1, xi, xf)
            deriv = deriv - 3 * dx * lpf_deriv_n(1, xi, xf, r0) / d ** 2
        elif j == 2:  # dF0/dr0
            deriv = -3 * dx * lpf_deriv_n(2, xi, xf, r0) / d ** 2
    elif i == 1:
        dy = xf[1] - xi[1]
        if j == 0:  # dF1/dx
            deriv = - 3 * dy * (n0 - ni) * lpf_deriv_dist_min2(0, xi, xf)
            deriv = deriv - 3 * dy * lpf_deriv_n(0, xi, xf, r0) / d ** 2
        elif j == 1:  # dF1/dy
            deriv = -3 * (n0 - ni) / d ** 2
            deriv = deriv - 3 * dy * (n0 - ni) * lpf_deriv_dist_min2(1, xi, xf)
            deriv = deriv - 3 * dy * lpf_deriv_n(1, xi, xf, r0) / d ** 2
        elif j == 2:  # dF1/dr0
            deriv = -3 * dy * lpf_deriv_n(2, xi, xf, r0) / d ** 2
    elif i == 2:
        if j == 0:
            deriv = lpf_deriv_n(0, xi, xf, r0) / r0
        elif j == 1:
            deriv = lpf_deriv_n(1, xi, xf, r0) / r0
        elif j == 2:
            deriv = lpf_deriv_n(2, xi, xf, r0) / r0 - (n0 - ni) / r0 ** 2

    return deriv


@njit
def lpf_deriv_n(i, xi, xf, r0):
    """
    Derivative of n wrt to fit parameters

    :param i:
    0=x
    1=y
    2=r0
    :param xi: hit position
    :param xf: fit position
    :param r0: number of photons
    :return: dn/di
    """

    if i < 2:
        deriv = -3 * lpf_nexp(xi, xf, r0) * (xf[i] - xi[i]) / lpf_dist(xi, xf) ** 2
    elif i == 2:
        deriv = lpf_nexp(xi, xf, r0) / r0
    else:
        deriv = 0.

    return deriv


@njit
def lpf_deriv_dist(i, xi, xf):
    """
    Derivative of distance wrt fit parameters

    :param i: 0=x
              1=y
              2=r0
    :param xi: hit position
    :param xf: fit position
    :return: dDist/di
    """

    if i < 2:
        deriv = (xf[i] - xi[i]) / lpf_dist(xi, xf)
    else:
        deriv = 0.0

    return deriv


@njit
def lpf_deriv_dist_min2(i, xi, xf):
    """
    Derivative of 1/dist**2

     :param i: 0=x
               1=y
               2=r0
    :param xi: hit position
    :param xf: fit position
    :return: d(1/Dist**2)/di
    """

    deriv = 0.0
    if i < 2:
        d = lpf_dist(xi, xf)
        deriv = -(2 / d ** 3) * lpf_deriv_dist(i, xi, xf)

    return deriv
# --------------------------------------------------------------------------------------- #
