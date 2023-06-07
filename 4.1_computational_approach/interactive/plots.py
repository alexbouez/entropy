''' Present an interactive function explorer with slider widgets.
Use the ``bokeh serve`` command to run the example by executing:
    bokeh serve plots.py
at your command prompt. Then navigate to the URL
    http://localhost:5006/gauss
in your browser.
'''
import math
from operator import truediv
import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, CustomJS, Dropdown, Spinner, Div
from bokeh.plotting import figure

# Settings
n = 16
N = 2**n
mask = 0xFFFF


#############
# Functions #
############# 

def normalize(y):
    sf = sum(y)
    return y.copy()/sf if sf > 0 else y.copy()

def flatten(y, t):
    m = max(y)
    return t*y.copy()/m if m > 0 else y.copy()

def entropy(p):
    ent = 0.0
    for i in p:
        if i > 0:
            ent -= i * math.log(i, 2)
    return ent if ent else 0

# Probability distributions

def make_gauss(sigma, mu, N):
    s = -1.0 / (2 * sigma**2)
    def f(x):
        return min(max(math.exp(s * (x - mu)**2), 0), 1)
    
    x = np.array([i    for i in range(0, N)])
    y = np.array([f(i) for i in range(0, N)])
    return x, y

def make_step(l, a, N):
    def f(x):
        return 1 if (x >= a) & (x <= a + l) else 0
    
    x = np.array([i    for i in range(0, N)])
    y = np.array([f(i) for i in range(0, N)])
    return x, y

def make_pyramid(w, m, N):
    s = 2/w
    b1, b2 = 1 - 2*m/w, 1 + 2*m/w
    def f(x):
        if (x >= m-w/2) & (x <= m):
            return min(max(s * x + b1, 0), 1)
        elif (x > m) & (x <= m+w/2):
            return min(max(-s * x + b2, 0), 1)
        return 0

    x = np.array([i    for i in range(0, N)])
    y = np.array([f(i) for i in range(0, N)])
    return x, y

# Hadamard transform

def fwht(y):
    a, h = y.copy(), 1
    while h < len(a):
        # perform FWHT
        for i in range(0, len(a), h * 2):
            for j in range(i, i + h):
                x = a[j]
                y = a[j + h]
                a[j] = x + y
                a[j + h] = x - y
        # normalize and increment
        a = a/2
        h *= 2
    return a 

def sorted_fwht(y):
    return np.sort(fwht(y))

def vect_mult(x, y):
    assert(len(x)==len(y))
    res = np.zeros(len(y))
    for i in range(len(x)):
        res[i] = min(max(x[i] * y[i], -1), 1)
    return res

def xor_convolve(x, yh):
    # second argument is already hadamard
    xh = fwht(normalize(x))
    return fwht(vect_mult(xh,yh))

# Rotation functions

def circular_right_shift16(x, a):
    return (x >> a) | (x << (n-a))  & 0xFFFF
    
def circular_left_shift16(x, a):
    return (x << a) & 0xFFFF | (x >> (n-a))

def apply_rot(y, a):
    res = np.zeros(len(y))
    for i in range(0, len(y)):
        j = circular_right_shift16(i,a)
        res[j] += y[i]
    return flatten(res, 1)

def apply_mult_rot(y, a, b, c):
    global N
    res = np.zeros(len(y))
    for i in range(0, len(y)):
        if (a,b,c)==(16,16,16):
            j = i  
        else:
            j = 0
            for r in [a,b,c]:
                if r != 16:
                    j ^= circular_right_shift16(i,r)
        j = j % N
        res[j] += y[i]
    return flatten(res, 1)

# Multi rounds

def multi_fwht(y, a, b, c, rounds, last_rot):
    assert(rounds>=0)
    yh = fwht(normalize(y))
    res = np.zeros(len(y))
    res[0] = 1
    for r in range(rounds):
        res = xor_convolve(res, yh)
        if (r < rounds - 1) or last_rot:
            res = apply_mult_rot(res, a, b, c)
    return flatten(res, 1)


####################
# INTERACTIVE PLOT #
####################

cwidth = 300

# Initialization
transform = fwht
multi_rounds = multi_fwht
make_function = make_gauss
last_rot = True 

x, y = make_function(4048, 16384, N)
source1 = ColumnDataSource(data=dict(x=x, y=y))
source2 = ColumnDataSource(data=dict(x=x, y=y))

z = flatten(transform(normalize(y)),1)
source3 = ColumnDataSource(data=dict(x=x, z=z))

# Gaussian plot
plot1 = figure(aspect_ratio=4/3, sizing_mode='scale_both',
              title="Probability distribution of input P(X)",
              tools="crosshair,pan,reset,save,wheel_zoom",
              x_range=[0, N], y_range=[-0.1, 1.1])
plot1.line('x', 'y', source=source1, line_width=1, line_alpha=0.6)

# Input plot
plot2 = figure(aspect_ratio=4/3, sizing_mode='scale_both',
              title="Probability distribution of output P(Y)",
              tools="crosshair,pan,reset,save,wheel_zoom",
              x_range=[0, N], y_range=[-0.1, 1.1])
plot2.line('x', 'y', source=source2, line_width=1, line_alpha=0.6)

# Hadamard plot
plot3 = figure(sizing_mode='stretch_both', 
              title="Hadamard transform", #y_axis_type="log",
              tools="crosshair,pan,reset,save,wheel_zoom", toolbar_location="above",
              x_range=[0, N], y_range=[-1.1, 1.1])
plot3.line('x', 'z', source=source3, line_width=1, line_alpha=0.6)


# Set up widgets
menu = [("Gaussian function", "gauss"), ("Step function", "step"), ("Pyramid function", "pyramid")]
dropdown1 = Dropdown(label="Gaussian function", menu=menu)
dropdown1.js_on_event("menu_item_click", CustomJS(code="console.log('dropdown: ' + this.item, this.toString())"))

menu = [("Hadamard transform", "hadamard"), ("Fourier transform", "fourier"), ("Sorted Hadamard transform", "sorted")]
dropdown2 = Dropdown(label="Hadamard transform", menu=menu)
dropdown2.js_on_event("menu_item_click", CustomJS(code="console.log('dropdown: ' + this.item, this.toString())"))

menu = [("Include final rotation", "last_rot"), ("Exclude final rotation", "no_last_rot")]
dropdown3 = Dropdown(label="Include final rotation", menu=menu)
dropdown3.js_on_event("menu_item_click", CustomJS(code="console.log('dropdown: ' + this.item, this.toString())"))

sigma = Slider(title="Standard deviation (sigma)", value=4048, start=1, end=16000, step=1)
mu = Slider(title="Mean (mu)", value=16384, start=1, end=65536, step=1)

rotation1 = Slider(title="Cyclic shift (a)", value=0, start=0, end=n, step=1)
rotation2 = Slider(title="Cyclic shift (b)", value=0, start=0, end=n, step=1)
rotation3 = Slider(title="Cyclic shift (c)", value=0, start=0, end=n, step=1)

rounds = Slider(title="Rounds", value=1, start=1, end=10, step=1)

dumdiv1 = Div(text='', width=10)
dumdiv2 = Div(text='', width=10)

sigma2 = Spinner(title=sigma.title, value=sigma.value)
mu2 = Spinner(title=mu.title, value=mu.value)
entropy1 = Spinner(title="Entropy of P(X)", value=round(entropy(normalize(y)), 5))
entropy2 = Spinner(title="Entropy of P(Y)", value=round(entropy(normalize(y)), 5))

# Set up callbacks
def set_sliders(function):
    if function == "step":
        sigma.title, sigma.end, mu.title = "Length (l)", 65536, "Start (a)"
    elif function == "pyramid":
        sigma.title, sigma.end, mu.title = "Width (w)", 65536, "Mean (m)"
    else: 
        sigma.title, sigma.end, mu.title = "Standard deviation (sigma)", 16000, "Mean (mu)"
    sigma2.title, mu2.title = sigma.title, mu.title

def handler1(event):
    global make_function
    if event.item == "step":
        make_function = make_step
        dropdown1.label = "Step function"
    elif event.item == "pyramid":
        make_function = make_pyramid
        dropdown1.label = "Pyramid function"
    else: 
        make_function = make_gauss
        dropdown1.label = "Gaussian function"
    set_sliders(event.item)
    update_data("value", 0, 0)
dropdown1.on_click(handler1)

def handler2(event):
    global multi_rounds, transform
    if event.item == "fourier":
        transform = fwht
        multi_rounds = multi_fwht
        dropdown2.label = "Fourier transform"
    elif event.item == "sorted":
        transform = sorted_fwht
        multi_rounds = multi_fwht
        dropdown2.label = "Sorted Hadamard transform"
    else: 
        transform = fwht
        multi_rounds = multi_fwht
        dropdown2.label = "Hadamard transform"
    set_sliders(event.item)
    update_data("value", 0, 0)
dropdown2.on_click(handler2)

def handler3(event):
    global last_rot
    if event.item == "last_rot":
        dropdown3.label = "Include final rotation"
        last_rot = True
    else: 
        dropdown3.label = "Exclude final rotation"
        last_rot = False
    update_data("value", 0, 0)
dropdown3.on_click(handler3)

# Set up callbacks
def update_data(attrname, old, new):
    global make_function, multi_rounds, transform, last_rot
    # Get the current slider values
    s = min(max(sigma.value, 1), N)
    m = min(max(mu.value, 0), N)

    a = min(max(rotation1.value, 0), n)
    b = min(max(rotation2.value, 0), n)
    c = min(max(rotation3.value, 0), n)
    r = min(max(rounds.value, 1), 10)

    # Generate the new curve
    x, y = make_function(s, m, N)
    y = np.nan_to_num(y)
    source1.data = dict(x=x, y=y)
    
    y2 = np.nan_to_num( multi_rounds(y, a, b, c, r, last_rot) )
    source2.data = dict(x=x, y=y2)
    
    z = np.nan_to_num( flatten(transform(normalize(y2)), 1) )
    source3.data = dict(x=x, z=z)

    # Update values
    entropy1.value = round(entropy(normalize(y)), 5)
    entropy2.value = round(entropy(normalize(y2)), 5)
    sigma2.value = sigma.value

for w in [sigma, mu, rotation1, rotation2, rotation3, rounds]:
    w.on_change('value', update_data)

def update_sliders(attrname, old, new):
    s = min(max(sigma2.value, 1), N)
    m = min(max(mu2.value, 1), N)

    # Update values
    sigma.value = s
    mu.value = m

for w in [sigma2, mu2]:
    w.on_change('value', update_sliders)

##########
# LAYOUT #
##########

inputs1 = column(dropdown1, sigma, mu, rounds, plot1, entropy1, dumdiv1, sigma2, 
    width = cwidth)
inputs2 = column(dropdown3, rotation1, rotation2, rotation3, plot2, entropy2, dumdiv2, mu2,
    width = cwidth)
inputs3 = column(plot3, sizing_mode="stretch_both")
curdoc().add_root(row(inputs1, inputs2, inputs3, sizing_mode="stretch_height"))

curdoc().title = "Interactive plots"