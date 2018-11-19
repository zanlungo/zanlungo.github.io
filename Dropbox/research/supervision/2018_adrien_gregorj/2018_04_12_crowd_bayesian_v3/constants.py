from math import pi
import numpy as np

D_BIN_SIZE = 25 # previously 50
D_MIN_TOLERABLE = 0
D_MAX_TOLERABLE = 2000# previously 2500

D_MIN_PLOT = 0
D_MAX_PLOT = 2000

VEL_BIN_SIZE = 3/120 # previously 3/60 
VEL_MIN_TOLERABLE = 0 # m/s # prevoously -3
VEL_MAX_TOLERABLE = 2 # m/s # previously 3

VEL_MIN_PLOT = 0
VEL_MAX_PLOT = 2.5

VELDIFF_BIN_SIZE = 0.5 / 60 
VELDIFF_MIN_TOLERABLE = 0 # m/s
VELDIFF_MAX_TOLERABLE = 0.5 # m/s

VELDIFF_MIN_PLOT = 0
VELDIFF_MAX_PLOT = 0.5

VDDOT_BIN_SIZE = 2 * pi / 60 
VDDOT_MIN_TOLERABLE = -pi # radians
VDDOT_MAX_TOLERABLE = pi# radians

VDDOT_MIN_PLOT = -pi
VDDOT_MAX_PLOT = pi

VVDOT_BIN_SIZE = 2 * pi / 60 
VVDOT_MIN_TOLERABLE = -pi # radians
VVDOT_MAX_TOLERABLE = pi # radians

VVDOT_MIN_PLOT = -pi
VVDOT_MAX_PLOT = pi

HEIGHT_BIN_SIZE = 7.5 # cm 
HEIGHT_MIN_TOLERABLE = 500 # mm
HEIGHT_MAX_TOLERABLE = 2100 # mm

HEIGHT_MIN_PLOT = 1250
HEIGHT_MAX_PLOT = 2000

# set the range of heightdiff to an integer multiple of HEIGHTDIFF_BIN_SIZE to 
# avoid problems in plotting pdf 
HEIGHTDIFF_BIN_SIZE = 7.5 # cm
HEIGHTDIFF_MIN_TOLERABLE = 0 # mm
HEIGHTDIFF_MAX_TOLERABLE = 1125 # mm # previously 2100

HEIGHTDIFF_MIN_PLOT = 0
HEIGHTDIFF_MAX_PLOT = 600

VELOCITY_THRESHOLD = 0.5 # m/s
DISTANCE_THRESHOLD = 2000 # mm

X_POSITION_THRESHOLD = (-10000, 50000)
Y_POSITION_THRESHOLD = (-25000, 10000)

HISTOG_PARAM_TABLE = {
    'd': (D_MIN_TOLERABLE, D_MAX_TOLERABLE, D_BIN_SIZE),
    'v_g': (VEL_MIN_TOLERABLE, VEL_MAX_TOLERABLE, VEL_BIN_SIZE),
    'v_diff': (VELDIFF_MIN_TOLERABLE, VELDIFF_MAX_TOLERABLE, VELDIFF_BIN_SIZE),
    'vv_dot': (VVDOT_MIN_TOLERABLE, VVDOT_MAX_TOLERABLE, VVDOT_BIN_SIZE),
    'vd_dot': (VDDOT_MIN_TOLERABLE, VDDOT_MAX_TOLERABLE, VDDOT_BIN_SIZE),
    'h_avg': (HEIGHT_MIN_TOLERABLE, HEIGHT_MAX_TOLERABLE, HEIGHT_BIN_SIZE),
    'h_diff': (HEIGHTDIFF_MIN_TOLERABLE, HEIGHTDIFF_MAX_TOLERABLE, HEIGHTDIFF_BIN_SIZE),
    'h_short': (HEIGHT_MIN_TOLERABLE, HEIGHT_MAX_TOLERABLE, HEIGHT_BIN_SIZE),
    'h_tall': (HEIGHT_MIN_TOLERABLE, HEIGHT_MAX_TOLERABLE, HEIGHT_BIN_SIZE)
}

PLOT_PARAM_TABLE = {
    'd': (D_MIN_PLOT, D_MAX_PLOT),
    'v_g': (VEL_MIN_PLOT, VEL_MAX_PLOT),
    'v_diff': (VELDIFF_MIN_PLOT, VELDIFF_MAX_PLOT),
    'vv_dot': (VVDOT_MIN_PLOT, VVDOT_MAX_PLOT),
    'vd_dot': (VDDOT_MIN_PLOT, VDDOT_MAX_PLOT),
    'h_avg': (HEIGHT_MIN_PLOT, HEIGHT_MAX_PLOT),
    'h_diff': (HEIGHTDIFF_MIN_PLOT, HEIGHTDIFF_MAX_PLOT),
    'h_short': (HEIGHT_MIN_PLOT, HEIGHT_MAX_PLOT),
    'h_tall': (HEIGHT_MIN_PLOT, HEIGHT_MAX_PLOT)
}

PARAM_NAME_TABLE = {
    'v_g': r'$v_g$',
    'v_diff': r'$\omega$',
    'd': r'$\delta$',
    'h_diff': r'$\Delta_{\eta}$'
}

PARAM_UNIT_TABLE = {
    'v_g': 'm/sec',
    'v_diff': 'm/sec',
    'd': 'mm',
    'h_diff': 'mm'
}

TRADUCTION_TABLE = {
    'koibito': 'M',
    'doryo': 'C',
    'yujin': 'Fr',
    'kazoku': 'Fa',
    'kazokuc': 'Fa+K',
    'kazokua': 'Fa-K'
}