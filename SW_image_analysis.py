import skimage as ski
from matplotlib import pyplot as plt
from skimage import measure
import numpy as np

def plot_cont(impath,xrange,yrange,rgb,mask_level=100, con_level=0.5, minsize=2500):
    """Given path to image, plot all contours found using scikit-image.
    Returns list of contours with size above 2500 (default).

    Params:
        impath: (str) path to image
        mask_level: (0<int<255) value level of mask to plot
        xrange: (2 element list) x-axis range to crop
        yrange: (2 element list) y-axis range to crop
        con_level: (float) contour level
        rgb: (int) sets which array to use -- 0 for r, 1 for g, 2 for b
        minsize: (int) min contour size
    """

    # Read image
    fullimage = ski.io.imread(impath)


    # # Print image stats
    # print(f'Size: {fullimage.size} \nMin: {fullimage.min()} \nMax: {fullimage.max()}')

    # Crop and choose color image
    image = fullimage[yrange[0]:yrange[1],xrange[0]:xrange[1],rgb]

    # Apply mask
    mask=image<mask_level

    # Find contours
    contours = measure.find_contours(mask, con_level)

    # Refine contours
    big_con=[]
    for con in contours:
        if con.size >minsize:
            big_con.append(con)
    fig, ax = plt.subplots()
    ax.imshow(mask, interpolation='nearest', cmap='gray')
    i=0
    for con in big_con:
        ax.plot(con[:,1], con[:,0], linewidth=1, label=f"{i}")
        i+=1
    ax.legend(loc='lower left',fontsize='xx-small')
    plt.show()

    return {'contours': big_con, 'mask': mask}


def find_scale_factor(l_cont,r_cont,y_coord,image,plot=False):
    """Calculate scale factor from image of pendulum given set of points comprising
    the left and right sides of the pendulum. Returns a pixels to mm
    conversion factor.

    Params:
    l_cont: (2xn array) set of points comprising left side of pendulum, with x points in first column
    r_cont: (2xn array) set of points comprising right side of pendulum, with x points in first column
    y_coord: (int) y-coordinate at which to compute distance from left contour
    image: (opt.)(array) image to display
    plot: (opt.) (bool) whether to plot the result
    """

    # Find linear fit of both contours
    m1, b1 = np.polyfit(l_cont[0], l_cont[1], 1)
    m2, b2 = np.polyfit(r_cont[0], r_cont[1], 1)

    # Find slope of perp line
    m3=-1/m2

    # Find x of ycoord from left contour
    x_coord=(y_coord-b1)/m1

    # Find y-int of perp line using slope and left contour point
    b3=y_coord-m3*x_coord

    # Find where perp line intersects with both lines
    xl=(b1-b3)/(m3-m1)
    yl=m1*(b1-b3)/(m3-m1)+b1
    xr=(b2-b3)/(m3-m2)
    yr=m2*(b2-b3)/(m3-m2)+b2

    # Calculate distance between these points
    dist = np.sqrt(((xl-xr)**2 + (yl-yr)**2))
    scale_fact = 3.16/dist

    # Plotting
    if plot:
        # Create dummy x
        dumx = np.linspace(np.min(l_cont[0])-10,np.max(r_cont[0])+10, 50)

        fig, ax = plt.subplots()
        ax.imshow(image, interpolation='nearest', cmap='gray')
        ax.plot(l_cont[0], l_cont[1], linewidth=1)
        ax.plot(r_cont[0], r_cont[1], linewidth=1)
        ax.plot(l_cont[0], l_cont[0] * m1 + b1)
        ax.plot(r_cont[0], r_cont[0] * m2 + b2)

        # Plot points of intersection
        ax.plot(xl, yl, 'bo')
        ax.plot(xr, yr, 'yo')

        # perp line
        ax.plot(dumx, m3 * dumx + b3)
        plt.show()

    print(f'\nScale factor: {scale_fact:3f} mm/pixel')
    return scale_fact


def calc_dist(pendulum, attract,y_coord,scale_factor,image,plot=False):
    """Calculate distance between pendulum and attractor.

    Params:
    pendulum: (2xn array) set of points comprising right side of pendulum, with x points in first column
    attract: (2xn array) set of points comprising left side of attractor, with x points in first column
    y_coord: (int) y-coordinate at which to compute distance from left contour
    scale_factor: (float) pixel to mm conversion factor
    image: (opt.)(array) image to display
    plot: (opt.) (bool) whether to plot the result
    """
    # Find linear fit of both contours
    m1, b1 = np.polyfit(pendulum[0], pendulum[1], 1)
    m2, b2 = np.polyfit(attract[0], attract[1], 1)

    # Find slope of perp line
    m3 = -1 / m2

    # Find x of ycoord from left contour
    x_coord = (y_coord - b1) / m1

    # Find y-int of perp line using slope and left contour point
    b3 = y_coord - m3 * x_coord

    # Find where perp line intersects with both lines
    xl = (b1 - b3) / (m3 - m1)
    yl = m1 * (b1 - b3) / (m3 - m1) + b1
    xr = (b2 - b3) / (m3 - m2)
    yr = m2 * (b2 - b3) / (m3 - m2) + b2

    # Calculate distance between these points
    dist = np.sqrt(((xl - xr) ** 2 + (yl - yr) ** 2))*scale_factor
    if plot:
        # Create dummy x
        dumx = np.linspace(np.min(pendulum[0]) - 10, np.max(attract[0]) + 10, 50)

        fig, ax = plt.subplots()
        ax.imshow(image, interpolation='nearest', cmap='gray')
        ax.plot(pendulum[0], pendulum[1], linewidth=1)
        ax.plot(attract[0], attract[1], linewidth=1)
        ax.plot(pendulum[0], pendulum[0] * m1 + b1)
        ax.plot(attract[0], attract[0] * m2 + b2)

        # Plot points of intersection
        ax.plot(xl, yl, 'bo')
        ax.plot(xr, yr, 'yo')

        # perp line
        ax.plot(dumx, m3 * dumx + b3)
        plt.show()

    print(f'\nDistance between pendulum and attractor: {dist*10**(3):3f} {"\u03BC"}m')

def con_img(con, mask, method):
    """
    Takes contour and nxn image/mask and returns a nxn array which contains the contour.
    Acts a preparation step for taking the cross correlation.

    Parameters:
        con: (mx2 array-like) contour
        mask: (nxl array-like) image which you want the final dimensions of
        method: (str) either "row" or "col" to indicate method of computation
    """
    # Create empty array for image
    fin_image=np.zeros(np.shape(mask))

    # Separate into columns
    x_col=con[:,1]
    y_col=con[:,0]

    # Take transpose so I can iterate over each set of coords
    coords=np.asarray([x_col,y_col]).T

    # Set each contour pixel to one on full image
    # # Note that this removes some precision since the original
    # # make_contours function is to 0.5 precision and this turns that
    # # into an integer
    for x,y in coords:
        fin_image[int(y)][int(x)]=1

    # For iterating over each row
    if method=="row":
        # Scan each row for 1
        for row in fin_image:
            if 1 in row:
                # Locate indices of where 1 is
                idx=np.where(row==1)[0]

                # Using first and last index, set row to white
                row[idx[0]:idx[-1]]=1

        return fin_image

    # For iterating over each column
    if method=="col":

        # Take the transpose to treat each column as a row
        fin_image=fin_image.T
        for row in fin_image:
            if 1 in row:
                # Locate indices of where 1 is
                idx=np.where(row==1)[0]

                # Using first and last index, set row to white
                row[idx[0]:idx[-1]]=1

        # Take the transpose again upon return to restore to original orientation
        return fin_image.T

def qwik_plot(mask,att,pend,att_range,pend_range):
    fig,ax=plt.subplots()
    ax.imshow(mask, interpolation='nearest', cmap='gray')
    ax.plot(att[:,1][att_range[0]:att_range[1]], att[:,0][att_range[0]:att_range[1]], linewidth=1)
    ax.plot(pend[:,1][pend_range[0]:pend_range[1]], pend[:,0][pend_range[0]:pend_range[1]], linewidth=1)
    plt.show()


def qwikr_plot(mask, att, pend, att_range, pend_range):
    fig, ax = plt.subplots()
    ax.imshow(mask, interpolation='nearest', cmap='gray')
    ax.plot(att[0][att_range[0]:att_range[1]], att[ 1][att_range[0]:att_range[1]], linewidth=1)
    ax.plot(pend[0][pend_range[0]:pend_range[1]], pend[ 1][pend_range[0]:pend_range[1]], linewidth=1)
    plt.show()

def crosscorr(im1,im2,step_size, direction,max=10,plot=True):
    # Direction: 0 is y, 1 is x
    if direction==0:
        dir_label="y"
    elif direction==1:
        dir_label="x"


    # For graphing
    shifts = range(1,max+1)
    all_pos_ccs=[]
    all_neg_ccs=[]

    # Value to use for normalization
    norm=np.tensordot(im1,im1,axes=2)

    # Baseline cross correlation
    base=np.tensordot(im1,im2,axes=2)/norm
    print(f'Baseline CC: {base:.5f}')

    # Add base
    all_pos_ccs.append(base)

    # Calculate positive CC shifts
    for i in shifts:

        # Shift im1
        shifted_pos = np.roll(im1,i*step_size,direction)

        # Calcuate CC between shifted im1 and unchanged im2
        shifted_pos_cc=np.tensordot(shifted_pos,im2, axes=2)/norm
        all_pos_ccs.append(shifted_pos_cc)

     # Repeat for negative shifts
    for i in shifts:

        # Shift im1
        shifted_neg = np.roll(im1,-i*step_size,direction)

        # Calcuate CC between shifted im1 and unchanged im2
        shifted_neg_cc=np.tensordot(shifted_neg,im2, axes=2)/norm
        all_neg_ccs.append(shifted_neg_cc)


    # Graphing
    all_ccs=np.concatenate((np.asarray(list(reversed(all_neg_ccs))),np.asarray(all_pos_ccs)))
    all_shifts=range(-10, 11, 1)

    # check length
    len(all_ccs) == len(all_shifts)

    if plot:
        fig, ax = plt.subplots()
        plt.plot(all_shifts, all_ccs, 'o')
        plt.xlabel(f"Shift in {dir_label} (pixels)")
        plt.ylabel("Cross-correlation")
        plt.title(f"Cross-correlation in {dir_label} direction")
        plt.show()

    return (all_shifts, all_ccs)


def arr_exp(small_im,exp_factor, con=False):
    """Expands small_arr by exp_factor while still preserving image.
    Parameters:
        small_im: (nxm array-like) original image
        exp_factor: (int or float) factor by which to expand both dimensions
    """

    # Create empty array for expanded image
    xdim, ydim = np.shape(small_im)
    big_arr = np.zeros((xdim * exp_factor, ydim * exp_factor))

    # Iterate over small_arr dimensions
    for n, row in enumerate(small_im):
        for m, i in enumerate(row):
            nn, mm = n * exp_factor, m * exp_factor
            big_arr[nn, mm] = i
            big_arr[nn + 1, mm] = i
            big_arr[nn, mm + 1] = i
            big_arr[nn + 1, mm + 1] = i

    if con:
        re_con = np.asarray([x * exp_factor for x in con])
        return big_arr, re_con
    else:
        return big_arr

# Write least squared gaussian fit function
def gauss_fit(x_data, y_data, plot=True):

    # Calculate mean and stddev of x data for fit
    mean = np.mean(x_data)
    std = np.std(x_data)
    xmusig= np.asarray([-1*(x-mean)**2/std**2 for x in x_data])

    # Create empty matrix for transpose of x
    xT=[[0], [0]]
    xT[0]=xmusig
    xT[1]=np.ones(np.shape(x_data))

    # Get x by taking the transpose
    x=np.transpose(xT)
    xTx=np.linalg.inv(np.matmul(xT,x))

    # Check shape
    print(np.shape(xTx))     # S.B. 2x2
    xTy=np.matmul(xT, np.log(y_data))

    # Coeffs (S.B. 2x1)
    coef=np.matmul(xTx, xTy)
    a=coef[0]
    b=coef[1]
    sol=np.exp(a+b*x_data)

    # Calculating residuals
    eps=y_data-sol

    if plot:
        # Plot raw data
        fig,ax=plt.subplots()
        plt.plot(x_data, y_data, 'o', label="Raw data")

        # Plot fit
        plt.plot(x_data, sol, '-', label="Gaussian fit")

        # Gaussian test
        #plt.plot(x_data, np.exp(np.asarray([x*-1 for x in x_data])))
        ax.set_xlabel("shift")
        ax.set_ylabel("cross correlation")
        ax.legend()

    print(f"Coefficients: a={a}, b={b}")
    return coef

