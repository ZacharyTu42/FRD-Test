from logging import raiseExceptions
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define model function to be used to fit to the data:
def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))


def lorentzian(x, *p):
    a, x0 , gam = p
    return a * gam**2 / ( gam**2 + ( x - x0 )**2)

def asym_gauss (x, *p):
    """ Asymmetric Gaussian function with variable continuum level .
    Input :
    x - x- values ( radii )
    *p - parameters [ Amplitude , mean value , standard deviation ,
    constant left , constant right ]
    Output :
    f - y- values , fitted values of assymmetric Gaussian """
    A, mu , sigma , Kl , Kr = p
    f = np.zeros(len(x)) # initializing the array with the result
    cut = np.argmax(x > mu) # find x corresponding to the mean value
    f[0:cut]=(A-Kl)*np.exp(-(x[0:cut]-mu)**2/(2.*sigma**2)) + Kl
    f[cut:] = (A-Kr)*np.exp(-(x[cut:]-mu)**2/(2.*sigma**2)) + Kr

    return f

def frd(image_data1, image_data2, deltad=5, pixel_size=2.4e-3):
    ___, ___, coeff1, half_width1, ____ = image_data1
    ___, ___, coeff2, half_width2, ____ = image_data2
    
    rad1 = coeff1[1] * pixel_size
    rad2 = coeff2[1] * pixel_size
    deltaw = (half_width2-half_width1) * pixel_size
    theta_out = np.arctan((rad2-rad1)/deltad)
    theta_exp = np.arctan((deltaw*np.cos(theta_out)**2)/deltad)

    return theta_exp / theta_out

class annulus:

    def __init__(self, image_path, min=15, max=180):
        self.image_path = image_path
        self.min = min
        self.max = max
        self.image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)

    def circlefinder(self):
        image = self.image
        # Binarizes Image
        #img = cv2.convertScaleAbs(image, alpha=9, beta=0)
        img = cv2.threshold(image, self.min, self.max, cv2.THRESH_BINARY)[1]
    
        # Detects the contours on the binary image using cv2.CHAIN_APPROX_NONE and extracts the outer and inner contours of annulus
        cnts, hier = cv2.findContours(image=img.copy(), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        cnts = sorted(cnts, key=len)
        outer = cnts[-1] # outer diameter contour
        inner = cnts[-2] # inner diameter contour

        # Initialize variables
        cx = list()
        cy = list()
        radii = list()
        # Finds center of mass and inner and outer circle radii
        for i in [inner, outer]:
            M = cv2.moments(i)

            cx.append(int(M['m10']/M['m00']))
            cy.append(int(M['m01']/M['m00']))
    
            area = cv2.contourArea(i)
            rad = int((area/np.pi)**(1/2))
            radii.append(rad)

        cx = sum(cx)/2
        cy = sum(cy)/2
        # Function outputs center of mass coordinates and radii
        return cx, cy

    def radial_profile(self, center):
        """ Bins pixel values radially around a given point
        Input:
        data - array-like, A grayscale image
        center - array-like, The x and y coordinates to bin radially around
        Output:
        radialprofile - A list of the binned intensity values
        """
        data = self.image
        y, x = np.indices((data.shape))
        r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        r = r.astype(np.int64)

        tbin = np.bincount(r.ravel(), data.ravel())
        nr = np.bincount(r.ravel())
        radialprofile = tbin / nr
        return radialprofile

    def ring_fit(self):
        """ Takes initial conditions from circlefinder() function for a given image
        and finds the Full Width Half Maximum (FWHM), inner and outer radius,
        and plots an asymmetric gaussian fit onto data
        Input:
        imgdata - Output of circlefinder() function
        Output:
        data - array-like, Passes the output of radial_profile() function
        coeff - array-like, List of coefficients from the curve_fit() function for an asymetric gaussian (A, mu, sigma, Kl, Kr)
        half_width - float, Half of the FWHM of the asymmetric gaussian
        """
        cx, cy = self.circlefinder() # Unpack imgdata from circlefinder()
        size = np.shape(self.image) # Find the size of image
    
        data = self.radial_profile((cx, cy)) # data from radial_profile (intensity values radial from COM)
    
        datalength = len(data) # Find length of data
        xdata = np.linspace(1, datalength, datalength) # Generate generic x values for curve_fit()
    
        # Starting conditions for asymmetric gaussian function
        p0 = [max(data), np.argmax(data), 50., 100., 100.] # A, mu, sigma, Kl, Kr

        coeff, ____ = curve_fit(asym_gauss, xdata, data, p0=p0) # Find correct coefficients for asym_gauss() fit
        coeff = np.abs(coeff) # Take the absolute value of coefficients (find better way to fix this issue?)


        fwhm = 2*np.sqrt(2*np.log(2))*coeff[2] # FWHM calculation
        half_width = fwhm/2 # Half of FWHM

        return xdata, data, coeff, half_width, [cx, cy]

    def plot(self):
        xdata, data, coeff, half_width, com = self.ring_fit()
        mean = coeff[1] # Mean value of peak in pixels
        cx, cy = com
        fitdata = asym_gauss(xdata, *coeff) # Generate y values for plotting fit

        # Finding the inner and outer edges of annulus
        inner_rad = mean-half_width
        outer_rad = mean+half_width

        fig, ax = plt.subplots(1, 2, figsize=(15,6), tight_layout=True)

        midcircle = plt.Circle((cx, cy), mean, fill=False, color='red', linestyle='dashed')
        innercircle = plt.Circle((cx, cy), inner_rad, fill=False, color='green')
        outercircle = plt.Circle((cx, cy), outer_rad, fill=False, color='green')
        ax[0].add_artist(midcircle)
        ax[0].add_artist(innercircle)
        ax[0].add_artist(outercircle)

        ax[0].imshow(self.image, cmap='gray')

        ax[1].plot(data, label='Original Data', color='green')
        ax[1].plot(fitdata, label='Fitted Data', linestyle='--', color='red')
    
        ax[1].legend()
        plt.show()

def fit_check(ans):
    while ans.lower() != 'y':
        if ans.lower() != 'y' or 'n':
            print("Unknown Command, please enter Y or N.")
            continue
        else:
            minimum = input('Please enter a new minimum theshold value: \n')
            image1 = annulus(imagepath1, min=minimum)
            ans = input('Does the fit look good? (Y/N): \n')
    
    return image1


imagepath1 = input('Path to first image: \n')
imagepath2 = input('Path to second image: \n')
try:
    image1 = annulus(imagepath1)
except AttributeError:
    print('File not found with path {}, please enter a valid path.'.format(imagepath1))
    exit()

image1.plot()
ans = input('Does the fit look good? (Y/N): \n')
while ans.lower() != 'y':
    if ans.lower() != 'y' and ans.lower() != 'n':
        print("Unknown Command, please enter Y or N.")
        continue
    else:
        minimum = input('Please enter a new minimum theshold value: \n')
        image1 = annulus(imagepath1, min=float(minimum))
        image1.plot()
        ans = input('Does the fit look good? (Y/N): \n')

try:
    image2 = annulus(imagepath2)
except AttributeError:
    print('File not found with path {}, please enter a valid path.'.format(imagepath2))
    exit()
image2.plot()

ans = input('Does the fit look good? (Y/N): \n')
while ans.lower() != 'y':
    if ans.lower() != 'y' and ans.lower() != 'n':
        print("Unknown Command, please enter Y or N.")
        continue
    else:
        minimum = input('Please enter a new minimum theshold value: \n')
        image2 = annulus(imagepath1, min=float(minimum))
        image2.plot()
        ans = input('Does the fit look good? (Y/N): \n')

deltad = input('What is the distance between the two measurements? (Default is 5mm): \n')

while type(deltad) != float or type(deltad) !=None:
    try:
        if len(deltad) == 0:
            deltad = 5
            break
        else:
            deltad=float(deltad)
            break
    except ValueError:
        print('Input was not a number')
        pass
    
    deltad = input('What is the distance between the two measurements? (Default is 5mm): \n')

ratio = frd(image1.ring_fit(), image2.ring_fit(), deltad=deltad)
print('The FRD for this image pair is: {:.4}'.format(ratio))