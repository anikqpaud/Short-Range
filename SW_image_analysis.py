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

    # Print image stats
    print(f'Size: {fullimage.size} \nMin: {fullimage.min()} \nMax: {fullimage.max()}')

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
        print(con.size)
        i+=1
    ax.legend(loc='lower left',fontsize='xx-small')
    plt.show()

    return big_con


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