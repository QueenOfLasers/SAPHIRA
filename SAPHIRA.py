from DaniMath import *
from os import chdir, path, listdir, remove, mkdir
from time import time
from sys import stdout
import matplotlib.pyplot as plt
import warnings
import csv
import random
from numpy import fft
from numpy import ma
from scipy import signal, stats
from numpy import histogram

pi = math.pi

e = math.e

#mks
h = 6.626068e-34
c = 2.9998e8
kB = 1.3806503e-23

##defaultdir = "c:/Users/Dani/SAPHIRA"
defaultdir = "d:/SAPHIRA/"
##defaultdir = "d:/SAPHIRA/IRTF 29-30 Apr 2014/HD 88986 64x64"

class Star:
    def __init__(self, y, x):

        self.y = y #Location in y.
        self.x = x #Location in x.
        self.ys = [y]
        self.xs = [x]
        self.radius = 2 #Measured radius of star.
        self.pixels = 1 #Number of pixels found in star, for verification.
    def getPixels(self):
        return self.pixels
    def params(self):
        return self.y, self.x, self.radius, self.pixels
    def getY(self):
        return self.y
    def getX(self):
        return self.x
    def getRadius(self):
        return self.radius
    def getList(self):
        return self.ys, self.xs
    def setRadius(self,r):
        self.radius = r
        return
    def newPixel(self, y, x):
        #If it's within the current radius, count it.
        if sqrt((self.y - y)**2 + (self.x - x)**2) < self.radius:
            self.pixels += 1
            self.ys.append(y)
            self.xs.append(x)
            return True
        #If it's borderline, expand the radius and count it.
        elif sqrt((self.y - y)**2 + (self.x - x)**2) < self.radius + 1:
            self.pixels += 1
            self.radius += 1
            self.ys.append(y)
            self.xs.append(x)
            return True
        else:
            return False

def centroid(data):
    #Finds the centroid of a 2D array.
    #First, compute background values.
    #bkg = mean(mean(data))
    Mx = 0
    My = 0
    total = 0
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            #Remove background values on the fly, replace with zero if negative.
            #Mx += (data[y][x] - bkg if data[y][x] - bkg > 0 else 0) * x
            #My += (data[y][x] - bkg if data[y][x] - bkg > 0 else 0) * y
            #Sum them up. Int to prevent overflow, we don't need 8 decimal places for every pixel.
            try:
                Mx += int(data[y,x] * x)
                My += int(data[y,x] * y)
                total += (data[y][x])
            except ValueError:
                print "NaN detected in centroid(), skipping."
    return (float(My)/total, float(Mx)/total)  

def maximum(data):
    #Finds the maximum entry of a 2D array or list via basic iteration.
    #Returns the x and y coordinates of the entry, not the value.
    maxy = 0
    maxx = 0
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            if data[maxy,maxx] < data[y,x]:
                #Check for outlier pixels.
                try:
                    if not (data[y,x] > 2*data[y+1,x] or data[y,x] > 3*data[y,x+1]):
                        maxy = y
                        maxx = x
                except:
                    if not (data[y,x] > 2*data[y-1,x] or data[y,x] > 3*data[y,x-1]):
                        maxy = y
                        maxx = x
    return maxy, maxx

def circtrim(data, radius, cy, cx):
    #Trims a circle at y, x in data to produce an 'aperture' array.
    #First, trim down to a box circumscribing our target aperture.
    out = data[cy-radius:cy+radius+1,cx-radius:cx+radius+1]
    for y in range(len(out)):
        for x in range(len(out[0])):
            #If inner corner of pixel is outside radius, throw away value.
            if sqrt((abs(x - radius) - 0.5)**2 + (abs(y - radius) - 0.5)**2) > radius:
                out[y][x] = 0
    return out

def annulus(data, inner, outer, cy, cx):
    #Similar to circtrim, but removes an inner radius to produce an annulus image.
    #Useful for background measurements.
    outercircle = circtrim(data, outer, cy, cx)
    innercircle = circtrim(data, inner, cy, cx)
    outercircle[int(outer) - int(inner):int(outer) + int(inner) + 1,int(outer) - int(inner):int(outer) + int(inner) + 1] -= innercircle
    return outercircle

def annulussubtract(data, circRadius, inner, outer, cy, cx):
    #Measures a mean background value from the surrounding annulus,
    #then subtracts that (corrected for number of pixels) from the central circle.
    ann = annulus(data, inner, outer, cy, cx)
    annlist = []
    #print ann.flatten()
    for i in ann.flatten():
        if i <> 0:
            annlist.append(i)
    #print annlist
    circ = circtrim(data, circRadius, cy, cx)
    #circ -= sum(sum(ann)) / float(nann)
    #circ -= ann.sum()/nann
    #print circ.mean()
    #print median(annlist)
    circ -= numpy.median(annlist)
    return circ

def removenegatives(data):
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            if data[y,x] < 0.:
                data[y,x] = 0.
    return data

def openfits(filename):
    #Simply wraps the fits reading stuff.
    hdulist = pyfits.open(filename,ignore_missing_end=True)
    #If it's a single frame, return a 2D array of it.
    if len(hdulist) > 1:
        data = numpy.zeros([len(hdulist),hdulist[0].data.shape[0],hdulist[0].data.shape[1]])
        for i in range(len(hdulist)):
            data[i,:,:] = hdulist[i].data
    else:
        if len(hdulist[0].data) == 1:
            data = hdulist[0].data[0]
        #If it's multiple frames, then return a 3D array of all of them.
        else:
            data = hdulist[0].data
    hdulist.close()
    return data

def savefits(filename, data):
    #Wraps the fits writing stuff.
    hdu = pyfits.PrimaryHDU(data)
    hdulist = pyfits.HDUList([hdu])
    hdulist.writeto(filename)
    hdulist.close()
    return

def cropcube(data,ystart,ystop,xstart,xstop):
    i = 0
    out = numpy.zeros((len(data),ystop-ystart,xstop-xstart))
    while i < len(data):
        y = 0
        temp = numpy.zeros((ystop-ystart,xstop-xstart))
        while y < ystop-ystart:
            temp[y] = data[i][y+ystart][xstart:xstop]
            y += 1
        out[i] = temp
        i += 1
    print "Datacube cropped from "+str(data.shape[1])+"x"+str(data.shape[2])+\
        " to "+str(out.shape[1])+"x"+str(out.shape[2])+"."
    return out

def sdarray(data):
    #Returns averaged std dev of pixels in a datacube.
    stddevlist = []
    x = 0
    y = 0
    while x < data.shape[1]:
        while y < data.shape[2]:
            stddevlist.append(stddev(data[:,x,y]))
            y += 1
        x += 1
    return mean(stddevlist)

def removeSpikes(data, threshhold = 50, debug = False, delete = False, byFrame = False):
    start = data.shape[0]
    count = 0
    #print "Removing spikes",
    n = 1
    while n < data.shape[0] - 1:
        y = 0
        if byFrame:
            lower = numpy.mean(data[n,:,:]) - numpy.mean(data[n-1,:,:])
            upper = numpy.mean(data[n,:,:]) - numpy.mean(data[n+1,:,:])
            if abs(upper) > threshhold \
               and abs(lower) > threshhold \
               and upper / abs(upper) == lower / abs(lower):
                if delete:
                    data = numpy.delete(data,n,0)
                #We've deleted that frame, so start over.
                    y = 0
                    x = 0
                    #If we're at the end, go ahead and return what we have.
                    count += 1
                    if n == data.shape[0] - 1:
                        return data
                else:
                    data[n,:,:] = (data[n - 1,:,:] + data[n + 1, :, :]) / 2.
                    count += 1
        else:
            while y < data.shape[1]:
                x = 0
                while x < data.shape[2]:
                    lower = data[n,y,x] - data[n-1,y,x]
                    upper = data[n,y,x] - data[n+1,y,x]
                    if abs(upper) > threshhold \
                       and abs(lower) > threshhold \
                       and upper / abs(upper) == lower / abs(lower): #Check for significance. and abs(lower - upper) < threshhold
##                      print upper,lower
                        if delete:
                            data = numpy.delete(data,n,0)
                        #We've deleted that frame, so start over.
                            y = 0
                            x = 0
                            #If we're at the end, go ahead and return what we have.
                            count += 1
                            if n == data.shape[0] - 1:
                                return data
                        else:
                            data[n,y,x] = (data[n - 1,y,x] + data[n + 1, y, x]) / 2.
                            count += 1
                    else:
                        x += 1
                y += 1
        n += 1
    #    if n % 100 == 0:
    #        print ".",
    #print " "
    if debug:
        print "Spike removal complete: " + str(count),
        if byFrame:
            print "frames of " + str(start),
        else:
            print "pixels",
        if delete:
            print "deleted."
        else:
            print "interpolated."
    return data

def measureDark(data):
    d = removeSpikes(data)
    #Cutoff if it saturates.
    if [0] in d:
        d = d[:numpy.where(d == 0)[0][0]]
    beta, alpha = linearfit(range(len(d.ravel())), d.ravel())
    #print "Dark current is " + str(-beta) + " ADU per frame."
    #print "With frame time of " + str(frameTime * 1000) + \
    #      "ms, dark current is " + str(-beta / frameTime) + " ADU/s."
    return beta

def cropSaturation(data, value):
    try:
        return data[:(next(x[0] for x in enumerate(data) if x[1] < value))]
    except StopIteration:
        return data


def decaytime(data):
    x = 10
    start = data[x]
    while start - data[x] < 20000:
        x += 1
    return x

def hist(data):
    n, bins, patches = plt.hist(data.ravel(),50)
    #plt.axis([min(data),max(data)])
    plt.grid(True)
    plt.show()

def timeseq(data):
    plt.plot(range(len(data)),data.ravel())
    plt.show()

def timeseq2(data1,data2):
    plt.plot(range(len(data1)),data1.ravel(),range(len(data2)),data2.ravel())
    plt.show()

def timeseq3(data1,data2,data3):
    plt.plot(range(len(data1)),data1.ravel(),range(len(data2)),data2.ravel()\
             ,range(len(data3)),data3.ravel())
    plt.show()

def timeseq4(data1,data2,data3,data4):
    plt.plot(range(len(data1)),data1.ravel(),range(len(data2)),data2.ravel()\
             ,range(len(data3)),data3.ravel(),range(len(data4)),data4.ravel())
    plt.show()

def timeseq6(data1,data2,data3,data4,data5,data6):
    plt.plot(range(len(data1)),data1.ravel(),range(len(data2)),data2.ravel()\
             ,range(len(data3)),data3.ravel(),range(len(data4)),data4.ravel()\
             ,range(len(data5)),data5.ravel(),range(len(data6)),data6.ravel())
    plt.show()

def spike(data):
    sd = stddev(data)
    m = mean(data)
    spikes = []
    for i in range(len(data))[1:-1]:
        if abs(data[i] - data[i+1]) > 3 * sd and abs(data[i] - data[i-1]) > 3 * sd:
            spikes.append(abs(data[i] - m))
    return mean(spikes)

def measureshutter(data, ycent, xcent, ycenb, xcenb, r = 10):
    #Divides two images from each other to produce the
    #differential polarization image (+Q - -Q).
    working = numpy.copy(data)
    #The extra math here keeps us from considering overscan rows.
    top = 0
    bottom = 0
    for y in range(ycent-r,ycent+r+1):
        top += sum(working[y,xcent-r:xcent+r])
    for y in range(ycenb-r,ycenb+r+1):
        bottom += sum(working[y,xcenb-r:xcenb+r])
##    final = cropcube(working,rows - 1,working.shape[0] - rows,0,working.shape[1]-1)
    return top/bottom

def combine(data):
    #Combines all of the images in a cube into one mean image, with no jitter adjustment.
    out = numpy.zeros(data[0].shape)
    print "Combining images from datacube",
    for i in range(len(data)):
        for y in range(len(data[0])):
            for x in range(len(data[0,0])):
                out[y,x] += (data[i,y,x] / float(len(data)))
        print ".",
    print "combined."
    return out

def stack(data):
    #Adjusts all images into a cube so they have common centroid.
    #Make sure to use dark adjusted images or the dark current will wash out the
    #relevant centroid data.
    out = numpy.zeros(data.shape)
    temp = numpy.zeros(data.shape) #Place holder between dimensions.
    print "Adjusting images for jitter",
    target = centroid(data[0])
    for i in range(len(data)):
        center = centroid(data[i])
        delx = int(round(target[1] - center[1]))
        dely = int(round(target[0] - center[0]))
        #Make y adjustment.
        if dely == 0:
            temp[i,:] = data[i,:]
        elif dely > 0:
            temp[i,dely:] = data[i,:-dely]
        elif dely < 0:
            temp[i,:dely] = data[i,-dely:]
        #Make x adjustment, row-by-row.
        for y in range(len(data[i])):
            if delx == 0:
                out[i,y,:] = temp[i,y,:]
            elif delx > 0:
                out[i,y,delx:] = temp[i,y,:-delx]
            elif delx < 0:
                out[i,y,:delx] = temp[i,y,-delx:]
        print ".",
    print "jitter adjustment complete."
    return out

def fitscombine(filename1, filename2):
    data1 = openfits(filename1)
    data2 = openfits(filename2)
    savename = ""
    i = 0
    while filename1[i] == filename2[i]:
            savename += filename1[i]
            i += 1
    savefits(savename + "Combined1.fits", (data1 - data2) / 2.)

def measurebottombias(filename1, filename2, rad = 50, shift = 100):
    data1 = openfits(filename1)
    data1 -= data1[0,0]
    y1,x1 = maximum(data1)
    print "Starting image search at "+str(y1)+", "+str(x1)+"."
    for i in range(100):
        if data1[y1-shift/2-rad:y1+shift/2+rad+1,x1-rad:x1+rad+1].sum() < \
           data1[y1-shift/2-rad+1:y1+shift/2+rad+1+1,x1-rad:x1+rad+1].sum():
            y1 += 1
        elif data1[y1-shift/2-rad:y1+shift/2+rad+1,x1-rad:x1+rad+1].sum() < \
           data1[y1-shift/2-rad-1:y1+shift/2+rad+1-1,x1-rad:x1+rad+1].sum():
            y1 -= 1
        if data1[y1-shift/2-rad:y1+shift/2+rad+1,x1-rad:x1+rad+1].sum() < \
           data1[y1-shift/2-rad:y1+shift/2+rad+1,x1-rad+1:x1+rad+1+1].sum():
            x1 += 1
        elif data1[y1-shift/2-rad:y1+shift/2+rad+1,x1-rad:x1+rad+1].sum() < \
           data1[y1-shift/2-rad:y1+shift/2+rad+1,x1-rad-1:x1+rad+1-1].sum():
            x1 -= 1
    print "Image center found at "+str(y1)+", "+str(x1)+"."
    data2 = openfits(filename2)
    data2 -= data2[0,0]
    y2,x2 = maximum(data2)
    print "Starting image search at "+str(y2)+", "+str(x2)+"."
    for i in range(100):
        if data2[y2-shift/2-rad:y2+shift/2+rad+1,x2-rad:x2+rad+1].sum() < \
           data2[y2-shift/2-rad+1:y2+shift/2+rad+1+1,x2-rad:x2+rad+1].sum():
            y2 += 1
        elif data2[y2-shift/2-rad:y2+shift/2+rad+1,x2-rad:x2+rad+1].sum() < \
           data2[y2-shift/2-rad-1:y2+shift/2+rad+1-1,x2-rad:x2+rad+1].sum():
            y2 -= 1
        if data2[y2-shift/2-rad:y2+shift/2+rad+1,x2-rad:x2+rad+1].sum() < \
           data2[y2-shift/2-rad:y2+shift/2+rad+1,x2-rad+1:x2+rad+1+1].sum():
            x2 += 1
        elif data2[y2-shift/2-rad:y2+shift/2+rad+1,x2-rad:x2+rad+1].sum() < \
           data2[y2-shift/2-rad:y2+shift/2+rad+1,x2-rad-1:x2+rad+1-1].sum():
            x2 -= 1
    print "Image center found at "+str(y2)+", "+str(x2)+"."
    #Subtract bottom of one from top of other.
    biasfactor = 0.1
    order = 0.1
    while order > 0.00001:
        if data1[y1-shift:y1+1,:].sum() * (biasfactor+order) - data2[y2:y2+shift+1,:].sum() < 0:
            biasfactor += order
        if data1[y1-shift:y1+1,:].sum() * (biasfactor+order) - data2[y2:y2+shift+1,:].sum() > 0:
            order /= 10.
    print biasfactor
    #And vice versa.
    biasfactor = 0.1
    order = 0.1
    while order > 0.00001:
        if data2[y2-shift:y2+1,:].sum() * (biasfactor+order) - data1[y1:y1+shift+1,:].sum() < 0:
            biasfactor += order
        if data2[y2-shift:y2+1,:].sum() * (biasfactor+order) - data1[y1:y1+shift+1,:].sum() > 0:
            order /= 10.
    print biasfactor

def fitsmerge(filename1, filename2, outfilename, xshift, yshift, low1, high1, low2, high2):
    #Merges two potentially dissimilar fits. Parameters should be determined manually.
    #Filenames are the two files to merge.
    #x/yshift (ints) are the coords of a reference object in 2 relative to 1.
    #low/high1/2 are the low and high boundaries of the desired window for the
    #respective images.
    data1 = openfits(filename1)
    data2 = openfits(filename2)
    #Adjust pixel values to match, based on range given.
    data2 = (data2 - (high2 + low2) / 2.) * (high1 - low1) / (high2 - low2) + (high1 + low1) / 2.
    #Shift based on measured reference object by cropping images.
    if yshift < 0:
        #Reference object in 2 is lower than in 1.
        data2 = data2[:yshift,:]
        data1 = data1[-yshift:,:]
    elif yshift > 0:
        #Reference object in 2 is higher than in 1.
        data2 = data2[yshift:,:]
        data1 = data1[:-yshift,:]
    if xshift < 0:
        #Reference object in 2 is further left than in 1.
        data2 = data2[:,:xshift]
        data1 = data1[:,-xshift:]
    elif xshift > 0:
        #Reference object in 2 is further right than in 1.
        data2 = data2[:,xshift:]
        data1 = data1[:,:-xshift]
    #Mean the two together.
    out = (data1 + data2) / 2.
    #Save the result.
    savefits(outfilename, out)

def colormerge(redfile,greenfile,bluefile,outfile,xshift,yshift,low,high):
    from pygame import Surface, image
    red = openfits(redfile)
    green = openfits(greenfile)
    blue = openfits(bluefile)
    #Shift based on measured reference object by cropping images.
    if yshift < 0:
        #Reference object in 2 is lower than in 1.
        blue = blue[:yshift,:]
        red = red[-yshift:,:]
    elif yshift > 0:
        #Reference object in 2 is higher than in 1.
        blue = blue[yshift:,:]
        red = red[:-yshift,:]
    if xshift < 0:
        #Reference object in 2 is further left than in 1.
        blue = blue[:,:xshift]
        red = red[:,-xshift:]
    elif xshift > 0:
        #Reference object in 2 is further right than in 1.
        blue = blue[:,xshift:]
        red = red[:,:-xshift]
    #Make a pygame Surface image.
    pic = Surface((green.shape[1],green.shape[0]))
    #Turn pixel values into color values.
    red = (red - (high + low) / 2) * (256 / (high - low)) + (128)
    green = (green - (high + low) / 2) * (256 / (high - low)) + (128)
    blue = (blue - (high + low) / 2) * (256 / (high - low)) + (128)
    for y in range(green.shape[0]):
        for x in range(green.shape[1]):
            pic.set_at((x,y),(int(red[y,x]),int(green[y,x]),int(blue[y,x]),255))
        #print y,
    image.save(pic,outfile)

def stripemean(data):
    means = []
    for i in range(16):
        left = (i * 256 - 128)
        right = (i * 256 + 126)
        means.append(mean(mean(data[:, (left if left > 0 else 0):(right if  right < 4095 else 4095)])))
    for i in range(len(means)/2):
        print means[i*2 + 1] - means[i*2]

def fresnel(i,n2 = 1.474, n1 = 1.0):
    Rs = ((n1 * cosd(i) - n2 * sqrt(1 - (n1 / n2 * sind(i))**2)) / \
         (n1 * cosd(i) + n2 * sqrt(1 - (n1 / n2 * sind(i))**2)))**2
    Rp = ((n1 * sqrt(1 - (n1 / n2 * sind(i))**2) - n2 * cosd(i)) / \
          (n1 * sqrt(1 - (n1 / n2 * sind(i))**2) + n2 * cosd(i)))**2
    print "Rs:", Rs
    print "Rp:", Rp

def measureGain():
    flatfiles = ["./2013aug24/collected/DomeFlats_Kp_on.fits",\
                 "./2013aug24/collected/DomeFlats_Kp_on_2.fits",\
                 "./2013aug24/collected/DomeFlats_J_on.fits"]
    darkfiles = ["./2013aug24/collected/DomeFlats_Kp_off.fits",\
                 "./2013aug24/collected/DomeFlats_Kp_off_2.fits",\
                 "./2013aug24/collected/DomeFlats_J_off.fits"]
    for flatfile,darkfile in map(None,flatfiles,darkfiles):
        print flatfile
        flat = openfits(flatfile)
        flat = cropcube(flat,512,1024,0,1024)
        dark = openfits(darkfile)
        dark = cropcube(dark,512,1024,0,1024)
        dark = combine(dark)
        #flat -= dark
        #print "Dark corrected."
        del dark
        u = numpy.mean(flat)
        print "Mean:", u
        s = numpy.std(flat)
        del flat
        print "Standard Deviation:", s
        print "Gain:", u / s**2

def collect():
    #Collect all images of same target into multi-frame .fits file for simpler processing.
    target = ""
    directories = ["2013aug24","2013aug25"]
    data = numpy.empty([0,0,0])
    for d in directories:
        chdir(d)
        if not path.isdir("collected"):
            mkdir("collected")
        #Get a list of objects before we do anything.
        files = listdir(defaultdir + "/" + d)
        for f in files:
            if ".fits" in f:
                #Suppress the warnings since pyfits complains about the nirc2 header formats
                #for some reason, but works anyway.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category = UserWarning)
                    hdulist = pyfits.open(f,ignore_missing_end=True)
                    print f, hdulist[0].header['OBJECT'], hdulist[0].header['FWINAME']
                    if hdulist[0].header['OBJECT'] == "DomeFlats":
                        current = hdulist[0].header['OBJECT'] + "_" + hdulist[0].header['FWINAME'] + "_" + hdulist[0].header['FLSPECTR']
                    else:
                        current = hdulist[0].header['OBJECT'] + "_" + hdulist[0].header['FWINAME']
                    #If it's a new target, save what we have and start a new one.
                    if current != target:
                        #If there's existing data, save it.
                        if data.shape != (0L,0L,0L):
                            #If there's an existing file of this target, save to that one.
                            if path.isfile("./" + "collected/" + target + ".fits"):
                                #olddata = openfits("../" + target + ".fits")
                                #combined = numpy.append(olddata, data)
                                #remove("../" + target + ".fits")
                                #savefits("../" + target + ".fits",combined)
                                savefits("./"+ "collected/" + target + "_2.fits",data)
                            #Otherwise, write a new one.
                            else:
                                print "Saving file ", d + "/collected/" + target + ".fits"
                                print data.shape
                                savefits("./" + "collected/" + target + ".fits",data)
                        #Read in new name.
                        target = current
                        #Start over with data.
                        data = numpy.empty(numpy.append(numpy.array([0]), hdulist[0].data.shape))
                    #Add data to current set.
                    for h in hdulist:
                        data = numpy.append(data, [h.data], axis = 0)
        #Save the last one.
        savefits("./" + "collected/" + target + ".fits",data)
        chdir(defaultdir)

def quickCDS():
    #To CDS the SAPHIRA images for some reason.
    target = "../131209_160959.fits"
    hdulist = pyfits.open(target)
    print hdulist[0].data.shape
    data1 = hdulist[0].data[0,:,:]
    data2 = hdulist[0].data[1,:,:]
    savefits("../131209_160959_CDS.fits",data2 - data1)

def subtractFirstImage(filename):
    data = openfits(filename)
    print data.shape
    i = 2
    while i < data.shape[0]:
        data[i,:,:] -= data[1,:,:]
        i += 1
    savefits(filename[:-5] + "sub.fits",data)

def cdsCube(data, rolling = False, delta = 1):
    if delta > 1 and not rolling:
        print "Warning: delta only does something for rolling CDS."
    i = 0
    if rolling:
        out = numpy.empty((data.shape[0] - delta,data.shape[1],data.shape[2]))
        for i in range(out.shape[0]):
            out[i,:,:] = data[i + delta,:,:] - data[i,:,:]
    else:
        out = numpy.empty((data.shape[0]/2,data.shape[1],data.shape[2]))
        for i in range(out.shape[0]):
            out[i,:,:] = data[i*2 + 1,:,:] - data[i*2,:,:]
    return out

##def cdsCube(data):
##    i = 0
##    out = numpy.empty((data.shape[0]/2,data.shape[1],data.shape[2]))
##    while i < out.shape[0]:
##        out[i,:,:] = data[i*2 + 1,:,:] - data[i*2,:,:]
##        i += 1
##    return out

def cdsFits(filename):
    cdsdata = cdsCube(openfits(filename))
    savefits(filename[:-5] + "cds.fits",cdsdata)

def flip(filename):
    hdulist = pyfits.open(filename)
    data = hdulist[0].data[1,::-1,:]
    savefits("../131209_160959_flipped.fits",data)

def gaussianerror(data, cy, cx, a, C):
    err = 0
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            r = sqrt((cy - y)**2 + (cx - x)**2)
            err += int(abs(data[y,x] - int(C * e**(r**2 / (2 * a**2)))))
            #print err
    return err

def subtractnth(data, n=4):
    i = 0
    a = numpy.empty((data.shape[0]-n,data.shape[1],data.shape[2]))
    while i < a.shape[0]:
        a[i,:,:] = data[i+n,:,:] - data[n-1,:,:]
        i += 1
    return a
    
def avalancheGain():
    capac = None
    
##    lightson = ["140207_171953f.fits","140207_113018f.fits","140207_113342f.fits","140207_113513f.fits","140207_113702f.fits"\
##                "140207_114105f.fits","140207_114314f.fits",\
##                "140207_114449f.fits","140207_114816f.fits","140207_114953f.fits","140207_115233f.fits","140207_115401f.fits","140207_115522f.fits",\
##                "140207_115545f.fits","140207_115802f.fits","140207_115824f.fits"]
##    lightsoff = ["140207_171910f.fits","140207_113102f.fits","140207_113409f.fits","140207_113541f.fits","140207_113755f.fits","140207_114129f.fits","140207_114344f.fits",\
##                 "140207_114520f.fits","140207_114845f.fits","140207_115016f.fits","140207_115302f.fits","140207_115426f.fits","140207_115612f.fits",\
##                 "140207_115634f.fits","140207_115853f.fits","140207_115918f.fits"]
##    bias = [0.25,0.25,0.50,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,10.5,11.5,11.5]
##    lightson = ["140307_125123f.fits","140307_125259f.fits","140307_125443f.fits","140307_125643f.fits","140307_125936f.fits","140307_130145f.fits",\
##                "140307_130449f.fits","140307_130715f.fits","140307_130923f.fits","140307_131347f.fits","140307_131722f.fits","140307_131936f.fits"]
##    lightsoff = ["140307_125035f.fits","140307_125230f.fits","140307_125417f.fits","140307_125557f.fits","140307_125846f.fits","140307_130108f.fits",\
##                "140307_130417f.fits","140307_130624f.fits","140307_130855f.fits","140307_131320f.fits","140307_131652f.fits","140307_131911f.fits"]
##    lightson = ["140311_150232f.fits","140311_150418f.fits","140311_150601f.fits","140311_150733f.fits","140311_150857f.fits","140311_151023f.fits",\
##                "140311_151158f.fits","140312_151402f.fits","140312_151550f.fits","140312_151722f.fits","140312_151836f.fits","140312_151956f.fits"]
##    lightsoff =["140311_150152f.fits","140311_150333f.fits","140311_150510f.fits","140311_150657f.fits","140311_150819f.fits","140311_150953f.fits",\
##                "140311_151127f.fits","140312_151332f.fits","140312_151520f.fits","140312_151638f.fits","140312_151808f.fits","140312_151928f.fits"]
##    bias = [0.50,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5]
##    timing = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
##    cutoff = 5000
    #M02815-12
##    chdir("D:/SAPHIRA") #"140408_170748f.fits" "140408_170727f.fits"
##    lightson = ["140331_102124f.fits","140331_102551f.fits","140331_102754f.fits","140331_102918f.fits","140331_103034f.fits","140331_103156f.fits","140331_103323f.fits",\
##                "140331_103508f.fits","140331_103620f.fits","140331_103748f.fits","140331_103908f.fits","140331_104025f.fits","140408_162302f.fits","140408_164514f.fits"]
##    lightsoff =["140331_101950f.fits","140331_102222f.fits","140331_102726f.fits","140331_102846f.fits","140331_103007f.fits","140331_103126f.fits","140331_103244f.fits",\
##                "140331_103419f.fits","140331_103552f.fits","140331_103714f.fits","140331_103839f.fits","140331_103955f.fits","140408_162247f.fits","140408_164458f.fits"]
##    bias = [0.50,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5]
##    frametime = 0.0070
##    timing = numpy.array([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.01,0.01])
##    timing += frametime
##    cutoff = 32000
##    unity = 2
##    title = "Optical gain for SAPHIRA device M02815-12 (VDD = 4V) 14 Apr 2014"
    #M02775-10
##    lightsoff = ["140411_143535f.fits","140411_143728f.fits","140411_143919f.fits","140411_144225f.fits","140411_144357f.fits",\
##                 "140411_144611f.fits","140411_144823f.fits","140411_144959f.fits","140411_145222f.fits","140411_145359f.fits","140411_133805f.fits",\
##                 "140411_133957f.fits","140411_134311f.fits","140411_134432f.fits"]
##    lightson =  ["140411_143632f.fits","140411_143802f.fits","140411_143955f.fits","140411_144305f.fits","140411_144428f.fits",\
##                 "140411_144712f.fits","140411_144851f.fits","140411_145039f.fits","140411_145304f.fits","140411_145442f.fits","140411_133844f.fits",\
##                 "140411_134227f.fits","140411_134332f.fits","140411_134453f.fits"]
##    bias =   [0.5,1.5,2.5,4.5,5.5,6.5,7.5,8.5, 9.5, 10.5,11.5,12.5,13.5,14.5]
##    timing = [0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.02,0.02,0.02,0.02,0.02,0.02,0.02]
##    cutoff = 32000
##    unity = 2
##    title = "Optical gain for SAPHIRA device M02775-10 (VDD = 4V) 11 Apr 2014"
    #,"140414_091937f.fits"
    #,"140414_092015f.fits"
    #,14.5
    #,0.01
##    lightsoff = ["140413_112839f.fits","140413_112937f.fits","140413_113243f.fits","140413_113425f.fits","140413_113632f.fits","140413_113839f.fits",\
##                 "140413_114010f.fits","140413_114243f.fits","140413_114419f.fits","140414_091324f.fits","140414_091444f.fits","140414_091600f.fits",\
##                 "140414_091725f.fits","140414_091829f.fits","140211_105239f.fits","140218_102122f.fits"]
##    lightson =  ["140413_112859f.fits","140413_113113f.fits","140413_113317f.fits","140413_113501f.fits","140413_113702f.fits","140413_113925f.fits",\
##                 "140413_114058f.fits","140413_114304f.fits","140413_114441f.fits","140414_091356f.fits","140414_091517f.fits","140414_091632f.fits",\
##                 "140414_091747f.fits","140414_091855f.fits","140211_105303f.fits","140218_102600f.fits"]
##    bias =   [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5,11.5,12.5,13.5,10.0,10]
##    timing = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.013,0.013]
##    cutoff = 35000
##    unity = 2
##    title = "Optical gain for SAPHIRA device M02775-10 (VDD = 4V) 15 Apr 2014"
##    lightsoff = ["140413_112839f.fits","140413_112937f.fits","140413_113243f.fits","140413_113425f.fits","140413_113632f.fits","140413_113839f.fits",\
##                 "140413_114010f.fits","140413_114243f.fits","140413_114419f.fits","140414_091324f.fits","140414_091444f.fits","140414_091600f.fits",\
##                 "140414_091725f.fits","140414_091829f.fits"]
##    lightson =  ["140413_112859f.fits","140413_113113f.fits","140413_113317f.fits","140413_113501f.fits","140413_113702f.fits","140413_113925f.fits",\
##                 "140413_114058f.fits","140413_114304f.fits","140413_114441f.fits","140414_091356f.fits","140414_091517f.fits","140414_091632f.fits",\
##                 "140414_091747f.fits","140414_091855f.fits"]
##    bias =   [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5,11.5,12.5,13.5]
##    timing = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
##    cutoff = 35000
##    unity = 2
##    title = "Mean avalanche gain"
##    lightsoff = ["140414_104306f.fits","140414_104438f.fits","140414_104736f.fits","140414_104848f.fits"]
##    lightson =  ["140414_104339f.fits","140414_104457f.fits","140414_104756f.fits","140414_104910f.fits"]
##    bias = [2.5, 5.5, 9.5, 13.5]
##    timing = [0.01, 0.01, 0.01, 0.01]
##    cutoff = 6000
##    unity = 0
##    title = "Optical gain for SAPHIRA device M02775-10 (VDD = 6V) 14 Apr 2014"
##    lightsoff = ["140601_110223.fits","140601_110405.fits","140601_110524.fits","140601_110646.fits","140601_110750.fits","140601_110913.fits",\
##                 "140601_111018.fits","140601_111142.fits","140601_111224.fits","140601_111306.fits","140601_111343.fits","140601_111426.fits",\
##                 "140601_111509.fits","140601_111623.fits","140601_111704.fits","140601_111818.fits"]
##    lightson =  ["140601_110251.fits","140601_110434.fits","140601_110551.fits","140601_110710.fits","140601_110816.fits","140601_110939.fits",\
##                 "140601_111044.fits","140601_111157.fits","140601_111241.fits","140601_111320.fits","140601_111356.fits","140601_111437.fits",\
##                 "140601_111543.fits","140601_111641.fits","140601_111715.fits","140601_111842.fits"]
##    bias =   [ 1.5, 2.0, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5,10.5,11.5,12.5,13.5,14.5, 1.5]
##    timing = [0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.11]
##    cutoff = 20000
##    unity = 2
##    title = "Optical gain for SAPHIRA device M04055-06 (VDD = 4V) 1 Jun 2014"
##    lightsoff = ["140602_094907.fits","140602_095027.fits"]
##    lightson =  ["140602_094944.fits","140602_095916.fits"]
##    bias =   [ 2.5, 14.5]
##    timing = [0.02,0.02]
##    cutoff = 21000
##    unity = 0
##    title = "Quickie optical gain for SAPHIRA device M04055-06 (VDD = 4V) 2 Jun 2014"
##    lightsoff = ["140602_111231.fits","140602_111342.fits","140602_111441.fits","140602_111527.fits","140602_111619.fits","140602_111655.fits",\
##                 "140602_111808.fits","140602_111850.fits","140602_111921.fits","140602_111958.fits","140602_112040.fits","140602_112116.fits",\
##                 "140602_112200.fits","140602_112240.fits","140602_112326.fits"]
##    lightson =  ["140602_111305.fits","140602_111402.fits","140602_111456.fits","140602_111545.fits","140602_111630.fits","140602_111711.fits",\
##                 "140602_111823.fits","140602_111901.fits","140602_111934.fits","140602_112015.fits","140602_112051.fits","140602_112127.fits",\
##                 "140602_112215.fits","140602_112253.fits","140602_112337.fits"]
##    bias =   [ 1.5, 2.0, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5,10.5,11.5,12.5,13.5,14.5]
##    timing = [   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1]
##    cutoff = 36000
##    unity = 2
##    title = "Optical gain for SAPHIRA device M04055-06 (VDD = 4V) (T = 60K) 2 Jun 2014"
    
##    lightsoff = ["141216_101240.fits","141216_101351.fits","141216_101533.fits","141216_101715.fits","141216_101824.fits",\
##                 "141216_101918.fits","141216_102030.fits","141216_102224.fits","141216_102333.fits","141216_102427.fits",\
##                 "141216_102549.fits","141216_102804.fits","141216_103008.fits","141216_103108.fits","141216_103209.fits"]
##    lightson =  ["141216_101308.fits","141216_101428.fits","141216_101551.fits","141216_101735.fits","141216_101837.fits",\
##                 "141216_101939.fits","141216_102116.fits","141216_102251.fits","141216_102350.fits","141216_102441.fits",\
##                 "141216_102605.fits","141216_102853.fits","141216_103025.fits","141216_103123.fits","141216_103228.fits"]
##    bias =   [0.5,  1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5,10.5,11.5,12.5,13.5]
##    timing = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
##    cutoff = 35000
##    unity = 4
##    title = "Optical gain for SAPHIRA device M02775-10 (VDD = 3.5V) 30 Dec 2014"

##    lightsoff = ["150223_124200.fits","150223_124448.fits","150223_124818.fits","150223_125248.fits","150223_125618.fits",\
##                 "150223_130047.fits","150223_130424.fits","150223_130852.fits","150223_131248.fits","150223_131627.fits",\
##                 "150223_132030.fits"]
##    lightson =  ["150223_124213.fits","150223_124504.fits","150223_124830.fits","150223_125259.fits","150223_125631.fits",\
##                 "150223_130107.fits","150223_130444.fits","150223_130901.fits","150223_131302.fits","150223_131638.fits",\
##                 "150223_132043.fits"]
##    bias =   [ 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5,10.5,11.5,12.5,13.5]
##    timing = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
##    cutoff = 25000
##    unity = 1
##    title = "Optical gain for SAPHIRA device M02815-12 (VDD = 3.5V) 23 Feb 2015"


##    lightsoff = ["150330_101851.fits","150330_102005.fits","150330_102056.fits","150330_102149.fits",\
##                 "150330_102240.fits","150330_102328.fits","150330_102421.fits","150330_102518.fits",\
##                 "150330_102603.fits","150330_102747.fits","150330_102843.fits","150330_102949.fits"]
##    lightson =  ["150330_101858.fits","150330_102017.fits","150330_102111.fits","150330_102159.fits",\
##                 "150330_102251.fits","150330_102340.fits","150330_102432.fits","150330_102527.fits",\
##                 "150330_102617.fits","150330_102759.fits","150330_102852.fits","150330_103001.fits"]
##    bias =   [ 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5,10.5,11.5,12.5]
##    timing = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
##    capac =  [41.3,37.3,34.8,33.2,32.1,31.4,31.0,30.7,30.5,30.5,30.5,30.5]
##    cutoff = 15000
##    unity = 1
##    start = 2
##    title = "Optical gain for SAPHIRA device M02775-35 (VDD = 3.5V) 30 Mar 2015"
##    window = [[64, -64],[96,-96]]


##    lightsoff = ["150406_111215.fits","150406_111304.fits","150406_111407.fits","150406_111502.fits",\
##                 "150406_111627.fits","150406_111718.fits","150406_111802.fits","150406_111846.fits",\
##                 "150406_111932.fits","150406_112015.fits","150406_112059.fits","150406_112142.fits",\
##                 "150406_112221.fits"]
##    lightson =  ["150406_111229.fits","150406_111316.fits","150406_111415.fits","150406_111513.fits",\
##                 "150406_111640.fits","150406_111726.fits","150406_111814.fits","150406_111902.fits",\
##                 "150406_111940.fits","150406_112025.fits","150406_112108.fits","150406_112151.fits",\
##                 "150406_112235.fits"]
##    bias =   [ 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5,10.5,11.5,12.5,13.5]
##    timing = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
##    cutoff = 35000
##    unity = 1
##    start = 5
##    title = "Optical gain for SAPHIRA device M04935-17 (VDD = 3.5V) 6 Apr 2015"


##    lightsoff = ["150421_095248.fits","150421_095357.fits","150421_095445.fits","150421_095537.fits",\
##                 "150421_095635.fits","150421_095716.fits","150421_095801.fits","150421_095844.fits",\
##                 "150421_095921.fits","150421_100000.fits","150421_100044.fits","150421_100130.fits",\
##                 "150421_100208.fits"]
##    lightson =  ["150421_095304.fits","150421_095410.fits","150421_095501.fits","150421_095550.fits",\
##                 "150421_095646.fits","150421_095725.fits","150421_095812.fits","150421_095854.fits",\
##                 "150421_095931.fits","150421_100013.fits","150421_100058.fits","150421_100139.fits",\
##                 "150421_100217.fits"]
##    bias =   [ 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5,10.5,11.5,12.5,13.5]
##    timing = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
##    cutoff = 35000
##    unity = 1
##    start = 5
##    title = "Optical gain for SAPHIRA device M04935-17 mask removed (VDD = 3.5V) 21 Apr 2015"

    lightsoff = ["150629_153156.fits","150629_153249.fits","150629_151305.fits","150629_151348.fits",\
                 "150629_151422.fits","150629_151502.fits","150629_151548.fits","150629_151637.fits",\
                 "150629_151742.fits","150629_151829.fits","150629_151917.fits","150629_145916.fits","150629_150008.fits"]
    lightson =  ["150629_153217.fits","150629_153259.fits","150629_151316.fits","150629_151357.fits",\
                 "150629_151433.fits","150629_151512.fits","150629_151600.fits","150629_151648.fits",\
                 "150629_151754.fits","150629_151840.fits","150629_151928.fits","150629_145935.fits","150629_150020.fits"]
    bias =   [ 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5,10.5,11.5,12.5,13.5]
    timing = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
    capac =  [41.3,37.3,34.8,33.2,32.1,31.4,31.0,30.7,30.5,30.5,30.5,30.5,30.5]
    cutoff = 35000
    unity = 1
    start = 2
    title = "Optical gain for SAPHIRA device M02775-35 (VDD = 3.5V) 29 Jun 2015"
    window = [[64, -64],[96,-96]]





##    lightsoff = ["150928_141041.fits","150928_141154.fits","150928_141242.fits","150928_141336.fits",\
##                 "150928_141424.fits","150928_141519.fits","150928_141609.fits","150928_141713.fits",\
##                 "150928_141808.fits","150928_141900.fits"]
##    lightson =  ["150928_141102.fits","150928_141210.fits","150928_141253.fits","150928_141351.fits",\
##                 "150928_141438.fits","150928_141530.fits","150928_141625.fits","150928_141727.fits",\
##                 "150928_141822.fits","150928_141915.fits"]
##    bias =   [ 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5,10.5,11.5]
##    timing = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
##    cutoff = 35000
##    unity = 0
##    start = 2
##    title = "Optical gain for SAPHIRA device M06495-19 (VDD = 3.5V) 28 Sep 2015"


##    lightsoff = ["151006_093810.fits","151006_093915.fits","151006_094019.fits","151006_094126.fits",
##                 "151006_094223.fits","151006_094316.fits","151006_094402.fits","151006_094507.fits",
##                 "151006_094554.fits","151006_094646.fits","151006_094740.fits"]
##    lightson =  ["151006_093843.fits","151006_093938.fits","151006_094040.fits","151006_094138.fits",
##                 "151006_094238.fits","151006_094328.fits","151006_094418.fits","151006_094519.fits",
##                 "151006_094609.fits","151006_094703.fits","151006_094754.fits"]
##    bias =   [ 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5,10.5,11.5]
##    timing = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
##    cutoff = 35000
##    unity = 1
##    start = 2
##    title = "Optical gain for SAPHIRA device M06495-27 (T = 50K, VDD = 3.5V) 6 Oct 2015"


##    lightsoff = ["151006_111537.fits","151006_111659.fits","151006_111746.fits","151006_111836.fits",
##                 "151006_111932.fits","151006_112018.fits","151006_112107.fits","151006_112157.fits",
##                 "151006_112248.fits","151006_112332.fits","151006_112423.fits"]
##    lightson =  ["151006_111608.fits","151006_111709.fits","151006_111758.fits","151006_111855.fits",
##                 "151006_111945.fits","151006_112029.fits","151006_112119.fits","151006_112208.fits",
##                 "151006_112259.fits","151006_112345.fits","151006_112433.fits"]
##    bias =   [ 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5,10.5,11.5]
##    timing = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
##    cutoff = 35000
##    unity = 1
##    start = 1
##    title = "Optical gain for SAPHIRA device M06495-27 (T = 40K, VDD = 3.5V) 6 Oct 2015"


##    lightsoff = ["151006_135310.fits","151006_135413.fits","151006_135518.fits","151006_135617.fits",
##                 "151006_135659.fits","151006_135744.fits","151006_135832.fits","151006_135921.fits",
##                 "151006_140110.fits","151006_140207.fits","151006_140253.fits","151006_140343.fits"]
##    lightson =  ["151006_135341.fits","151006_135432.fits","151006_135532.fits","151006_135629.fits",
##                 "151006_135710.fits","151006_135757.fits","151006_135844.fits","151006_135936.fits",
##                 "151006_140123.fits","151006_140219.fits","151006_140307.fits","151006_140359.fits"]
##    bias =   [ 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5,10.5,11.5,12.5]
##    timing = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
##    cutoff = 35000
##    unity = 1
##    start = 1
##    title = "Optical gain for SAPHIRA device M06495-27 (T = 45K, VDD = 3.5V) 6 Oct 2015"


##    lightsoff = ["151006_170703.fits","151006_170752.fits","151006_170839.fits","151006_170935.fits",
##                 "151006_171031.fits","151006_171131.fits","151006_171210.fits","151006_171301.fits",
##                 "151006_171340.fits","151006_171421.fits","151006_171500.fits","151006_171537.fits",
##                 "151006_171616.fits","151006_171701.fits"]
##    lightson =  ["151006_170713.fits","151006_170804.fits","151006_170855.fits","151006_170947.fits",
##                 "151006_171052.fits","151006_171142.fits","151006_171225.fits","151006_171310.fits",
##                 "151006_171351.fits","151006_171432.fits","151006_171509.fits","151006_171547.fits",
##                 "151006_171628.fits","151006_171711.fits"]
##    bias =   [ 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5,10.5,11.5,12.5,13.5,14.5]
##    timing = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
##    cutoff = 35000
##    unity = 1
##    start = 1
##    title = "Optical gain for SAPHIRA device M06495-27 (T = 55K, VDD = 3.5V) 6 Oct 2015"
##    window = [[0, -1], [0, -1]]


##    lightsoff = ["151019_105108.fits","151019_105533.fits","151019_105638.fits","151019_110004.fits",
##                 "151019_110052.fits","151019_110144.fits","151019_110245.fits","151019_110354.fits",
##                 "151019_110440.fits","151019_110525.fits","151019_110610.fits","151019_110649.fits",
##                 "151019_110739.fits","151019_112739.fits"]
##    lightson =  ["151019_105418.fits","151019_105545.fits","151019_105651.fits","151019_110018.fits",
##                 "151019_110105.fits","151019_110200.fits","151019_110300.fits","151019_110406.fits",
##                 "151019_110452.fits","151019_110538.fits","151019_110620.fits","151019_110706.fits",
##                 "151019_110750.fits","151019_112751.fits"]
##    bias =   [ 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5,10.5,11.5,12.5,13.5,14.5]
##    timing = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
##    cutoff = 35000
##    unity = 1
##    start = 1
##    title = "Optical gain for SAPHIRA device M06665-12 (T = 60K, VDD = 3.5V) 19 Oct 2015"
####    window = [[0, -1], [0, -1]]
##    window = [[64, -64],[96,-96]]


##    lightsoff = ["151026_144148.fits","151026_144307.fits","151026_144407.fits","151026_144501.fits",
##                 "151026_144557.fits","151026_144910.fits","151026_144955.fits","151026_145045.fits",
##                 "151026_145138.fits","151026_145242.fits","151026_145325.fits","151026_145427.fits",
##                 "151026_145516.fits","151026_145558.fits"]
##    lightson =  ["151026_144208.fits","151026_144330.fits","151026_144418.fits","151026_144514.fits",
##                 "151026_144610.fits","151026_144926.fits","151026_145010.fits","151026_145058.fits",
##                 "151026_145149.fits","151026_145255.fits","151026_145337.fits","151026_145439.fits",
##                 "151026_145527.fits","151026_145613.fits"]
##    bias =   [ 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5,10.5,11.5,12.5,13.5,14.5]
##    timing = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
##    cutoff = 35000
##    unity = 1
##    start = 1
##    title = "Optical gain for SAPHIRA device M06715-27 (T = 60K, VDD = 3.5V) 26 Oct 2015"
####    window = [[0, -1], [0, -1]]
##    window = [[64, -64],[96,-96]]


##    lightsoff = ["151103_151229.fits","151103_151338.fits","151103_151456.fits","151103_151553.fits",
##                 "151103_151706.fits","151103_151804.fits","151103_151906.fits","151103_151947.fits",
##                 "151103_152030.fits","151103_152114.fits","151103_152202.fits","151103_152242.fits",
##                 "151103_152331.fits","151103_152424.fits"]
##    lightson =  ["151103_151242.fits","151103_151353.fits","151103_151508.fits","151103_151620.fits",
##                 "151103_151732.fits","151103_151818.fits","151103_151918.fits","151103_152000.fits",
##                 "151103_152042.fits","151103_152125.fits","151103_152213.fits","151103_152254.fits",
##                 "151103_152348.fits","151103_152433.fits"]
##    bias =   [ 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5,10.5,11.5,12.5,13.5,14.5]
##    timing = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
##    cutoff = 35000
##    unity = 1
##    start = 1
##    title = "Optical gain for SAPHIRA device M06665-03 (T = 60K, VDD = 3.5V) 3 Nov 2015"
##    window = [[64, -64],[96,-96]]


##    lightsoff = ["151108_111031.fits","151108_111141.fits","151108_111234.fits","151108_111330.fits",
##                 "151108_111415.fits","151108_111527.fits","151108_111608.fits","151108_111645.fits",
##                 "151108_111731.fits","151108_111824.fits","151108_111908.fits","151108_112007.fits",
##                 "151108_112049.fits","151108_112137.fits"]
##    lightson =  ["151108_111056.fits","151108_111159.fits","151108_111249.fits","151108_111346.fits",
##                 "151108_111427.fits","151108_111539.fits","151108_111619.fits","151108_111655.fits",
##                 "151108_111740.fits","151108_111834.fits","151108_111918.fits","151108_112017.fits",
##                 "151108_112101.fits","151108_112148.fits"]
##    bias =   [ 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5,10.5,11.5,12.5,13.5,14.5]
##    timing = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
##    cutoff = 35000
##    unity = 1
##    start = 1
##    title = "Optical gain for SAPHIRA device M06715-29 (T = 60K, VDD = 3.5V) 8 Nov 2015"
##    window = [[64, -64],[96,-96]]


##    lightsoff = ["151130_145727.fits","151130_145954.fits","151130_150154.fits","151130_150344.fits",
##                 "151130_150531.fits","151130_150721.fits","151130_150909.fits","151130_151055.fits",
##                 "151130_151253.fits","151130_151443.fits","151130_151649.fits","151130_151830.fits",
##                 "151130_152025.fits","151130_152208.fits"]
##    lightson =  ["151130_145736.fits","151130_150006.fits","151130_150207.fits","151130_150353.fits",
##                 "151130_150541.fits","151130_150730.fits","151130_150919.fits","151130_151107.fits",
##                 "151130_151304.fits","151130_151454.fits","151130_151658.fits","151130_151843.fits",
##                 "151130_152036.fits","151130_152217.fits"]
##    bias =   [ 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5,10.5,11.5,12.5,13.5,14.5]
##    timing = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
##    cutoff = 35000
##    unity = 1
##    start = 1
##    title = "Optical gain for SAPHIRA device M06715-34 (T = 60K, VDD = 3.5V) 30 Nov 2015"
##    window = [[64, -64],[96,-96]]


##    lightsoff = ["160407_COM2Voff-16-0.fits","160407_COM1Voff-18-0.fits","160407_COM0Voff-20-0.fits",
##                 "160407_COM-1Voff-22-0.fits","160407_COM-2Voff-34-0.fits","160407_COM-3Voff-36-0.fits",
##                 "160407_COM-4Voff-38-0.fits","160407_COM-5Voff-40-0.fits","160407_COM-6Voff-42-0.fits",
##                 "160407_COM-7Voff-44-0.fits","160407_COM-8Voff-46-0.fits","160407_COM-9Voff-48-0.fits",
##                 "160407_COM-10Voff-50-0.fits","160407_COM-11Voff-55-0.fits","160407_COM-12Voff-56-0.fits",
##                 "160407_COM-13Voff-58-0.fits","160407_COM-14Voff-60-0.fits","160407_COM-15Voff-62-0.fits"]
##    lightson =  ["160407_COM2Von-17-0.fits","160407_COM1Von-19-0.fits","160407_COM0Von-21-0.fits",
##                 "160407_COM-1Von-23-0.fits","160407_COM-2Von-35-0.fits","160407_COM-3Von-37-0.fits",
##                 "160407_COM-4Von-39-0.fits","160407_COM-5Von-41-0.fits","160407_COM-6Von-43-0.fits",
##                 "160407_COM-7Von-45-0.fits","160407_COM-8Von-47-0.fits","160407_COM-9Von-49-0.fits",
##                 "160407_COM-10Von-51-0.fits","160407_COM-11Von-54-0.fits","160407_COM-12Von-57-0.fits",
##                 "160407_COM-13Von-59-0.fits","160407_COM-14Von-61-0.fits","160407_COM-15Von-64-0.fits"]
##    bias =   [ 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5]
##    timing = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
##    capac =  [37.3,34.8,33.2,32.1,31.4,31.0,30.7,30.5, 30.5, 30.5, 30.5, 30.5, 30.5, 30.5, 30.5, 30.5, 30.5, 30.5]
##    cutoff = -15000
##    unity = 0
##    start = 5
##    title = "Optical gain for SAPHIRA device M06715-27 (T = 60K, VDD = 5.0V) w/ PB-32 7 Apr 2016"
####    title = "Avalanche Gain for Mark 14 SAPHIRA"
##    window = [[0, -1],[64,-1]]


##    lightsoff = ["160515_2Voff-36-0.fits","160515_1Voff-38-0.fits","160515_0Voff-40-0.fits",
##                 "160515_-1Voff-42-0.fits","160515_-2Voff-44-0.fits","160515_-3Voff-47-0.fits",
##                 "160515_-4Voff-49-0.fits","160515_-5Voff-53-0.fits","160515_-6Voff-54-0.fits",
##                 "160515_-7Voff-56-0.fits"]
##    lightson =  ["160515_2V0.8-37-0.fits","160515_1V0.8-39-0.fits","160515_0V0.8-41-0.fits",
##                 "160515_-1V0.8-43-0.fits","160515_-2V0.8-46-0.fits","160515_-3V0.8-48-0.fits",
##                 "160515_-4V0.8-50-0.fits","160515_-5V0.8-52-0.fits","160515_-6V0.8-55-0.fits",
##                 "160515_-7V0.8-57-0.fits"]
##    bias =   [ 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5]
##    timing = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
##    cutoff = -50000
##    unity = 0
##    start = 1
##    title = "Optical gain for SAPHIRA device M06665-?? (T = 60K, VDD = 5.0V) w/ PB-32 15 Jun 2016"
##    window = [[64, -64],[96,-96]]

##    lightsoff = ["160621_2VLEDoff-4-0.fits","160621_0VLEDoff-8-0.fits",
##                 "160621_-1VLEDoff-10-0.fits","160621_-2VLEDoff-12-0.fits","160621_-3VLEDoff-14-0.fits",
##                 "160621_-4VLEDoff-16-0.fits","160621_-5VLEDoff-18-0.fits","160621_-6VLEDoff-20-0.fits",
##                 "160621_-7VLEDoff-22-0.fits","160621_-8VLEDoff-24-0.fits","160621_-10VLEDoff-28-0.fits",
##                 "160621_-11VLEDoff-30-0.fits","160621_-12VLEDoff-32-0.fits","160621_-13VLEDoff-34-0.fits",
##                 "160621_-14VLEDoff-36-0.fits","160621_-15VLEDoff-38-0.fits"]
##    lightson =  ["160621_2VLED0.8V-5-0.fits","160621_0VLED0.8V-9-0.fits",
##                 "160621_-1VLED0.8V-11-0.fits","160621_-2VLED0.8V-13-0.fits","160621_-3VLED0.8V-15-0.fits",
##                 "160621_-4VLED0.8V-17-0.fits","160621_-5VLED0.8V-19-0.fits","160621_-6VLED0.8V-21-0.fits",
##                 "160621_-7VLED0.8V-23-0.fits","160621_-8VLED0.8V-25-0.fits","160621_-10VLED0.8V-29-0.fits",
##                 "160621_-11VLED0.8V-31-0.fits","160621_-12VLED0.8V-33-0.fits","160621_-13VLED0.8V-35-0.fits",
##                 "160621_-14VLED0.8V-37-0.fits","160621_-15VLED0.8V-39-0.fits"]
##    bias =   [ 2.5, 4.5, 5.5, 6.5, 7.5, 8.5,10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5]
##    timing = [0.01,0.01,0.01,0.01,0.01,0.01,0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
##    cutoff = -10000
##    unity = 0
##    start = 5
##    title = "Optical gain for SAPHIRA device M06665-25 (T = 60K, VDD = 5.0V) w/ PB-32 21 Jun 2016"
##    window = [[0, -1],[64,-1]]

##    lightsoff = ["160719_COM2VLEDoff-11-0.fits",
##                 "160719_COM1VLEDoff-13-0.fits",
##                 "160719_COM0VLEDoff-15-0.fits",
##                 "160719_COM-1VLEDoff-17-0.fits",
##                 "160719_COM-2VLEDoff-20-0.fits",
##                 "160719_COM-3VLEDoff-22-0.fits",
##                 "160719_COM-4VLEDoff-24-0.fits",
##                 "160719_COM-5VLEDoff-26-0.fits",
##                 "160719_COM-6VLEDoff-28-0.fits",
##                 "160719_COM-7VLEDoff-30-0.fits",
##                 "160719_COM-8VLEDoff-32-0.fits",
##                 "160719_COM-9VLEDoff-34-0.fits",
##                 "160719_COM-10VLEDoff-36-0.fits",
##                 "160719_COM-11VLEDoff-38-0.fits",
##                 "160719_COM-12VLEDoff-40-0.fits",
##                 "160719_COM-13VLEDoff-42-0.fits",
##                 "160719_COM-14VLEDoff-44-0.fits",
##                 "160719_COM-15VLEDoff-46-0.fits"]
##    lightson =  ["160719_COM2VLED6mA-12-0.fits",
##                 "160719_COM1VLED6mA-14-0.fits",
##                 "160719_COM0VLED6mA-16-0.fits",
##                 "160719_COM-1VLED6mA-19-0.fits",
##                 "160719_COM-2VLED6mA-21-0.fits",
##                 "160719_COM-3VLED6mA-23-0.fits",
##                 "160719_COM-4VLED6mA-25-0.fits",
##                 "160719_COM-5VLED6mA-27-0.fits",
##                 "160719_COM-6VLED6mA-29-0.fits",
##                 "160719_COM-7VLED6mA-31-0.fits",
##                 "160719_COM-8VLED6mA-33-0.fits",
##                 "160719_COM-9VLED6mA-35-0.fits",
##                 "160719_COM-10VLED6mA-37-0.fits",
##                 "160719_COM-11VLED6mA-39-0.fits",
##                 "160719_COM-12VLED6mA-41-0.fits",
##                 "160719_COM-13VLED6mA-43-0.fits",
##                 "160719_COM-14VLED6mA-45-0.fits",
##                 "160719_COM-15VLED6mA-47-0.fits"]
##    bias =   [ 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5]
##    timing = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
##    cutoff = -20000
##    unity = 0
##    start = 5
##    title = "Optical gain for SAPHIRA device M06665-25 (T = 60K, VDD = 5.0V) w/ PB-32 19 Jul 2016"
##    window = [[0, -1],[0,-1]]

##    lightsoff = ["160719_161419.fits",
##                 "160719_162057.fits",
##                 "160719_162216.fits",
##                 "160719_162300.fits",
##                 "160719_162356.fits",
##                 "160719_162458.fits",
##                 "160719_162600.fits",
##                 "160719_162700.fits",
##                 "160719_162756.fits",
##                 "160719_162855.fits",
##                 "160719_162956.fits",
##                 "160719_163057.fits",
##                 "160719_163156.fits",
##                 "160719_163253.fits"]
##    lightson =  ["160719_161435.fits",
##                 "160719_162110.fits",
##                 "160719_162200.fits",
##                 "160719_162314.fits",
##                 "160719_162413.fits",
##                 "160719_162512.fits",
##                 "160719_162613.fits",
##                 "160719_162713.fits",
##                 "160719_162811.fits",
##                 "160719_162911.fits",
##                 "160719_163012.fits",
##                 "160719_163110.fits",
##                 "160719_163210.fits",
##                 "160719_163309.fits"]
##    bias =   [ 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5]
##    timing = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
##    capac  = [37.5, 35., 33., 32.,31.5, 31.,30.5, 30.,  30.,  30.,  30.,  30.,  30.,  30.]
##    cutoff = -20000
##    unity = 0
##    start = 250
##    title = "Optical gain @ 3.1um for SAPHIRA device M06665-25 (T = 60K, VDD = 5.0V) w/ ARC 19 Jul 2016"
##    window = [[0, -1],[0,-1]]

##    lightsoff = ["160719_172316.fits",
##                 "160719_172504.fits",
##                 "160719_172602.fits",
##                 "160719_172648.fits",
##                 "160719_172736.fits",
##                 "160719_172826.fits",
##                 "160719_172916.fits",
##                 "160719_173005.fits",
##                 "160719_173050.fits",
##                 "160719_173137.fits",
##                 "160719_173231.fits",
##                 "160719_173319.fits",
##                 "160719_173420.fits",
##                 "160719_173507.fits"]
##    lightson =  ["160719_172406.fits",
##                 "160719_172519.fits",
##                 "160719_172613.fits",
##                 "160719_172701.fits",
##                 "160719_172749.fits",
##                 "160719_172840.fits",
##                 "160719_172928.fits",
##                 "160719_173018.fits",
##                 "160719_173103.fits",
##                 "160719_173150.fits",
##                 "160719_173243.fits",
##                 "160719_173332.fits",
##                 "160719_173432.fits",
##                 "160719_173522.fits"]
##    bias =   [ 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5]
##    timing = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
##    capac  = [37.5, 35., 33., 32.,31.5, 31.,30.5, 30.,  30.,  30.,  30.,  30.,  30.,  30.]
##    cutoff = -20000
##    unity = 0
##    start = 200
##    title = "Optical gain @ 1.7um for SAPHIRA device M06665-25 (T = 60K, VDD = 5.0V) w/ ARC 19 Jul 2016"
##    window = [[0, -1],[0,-1]]

##    lightsoff = ["161103_152223.fits",
##                 "161103_152322.fits",
##                 "161103_152406.fits",
##                 "161103_152446.fits",
##                 "161103_152528.fits",
##                 "161103_152621.fits",
##                 "161103_152703.fits",
##                 "161103_152746.fits",
##                 "161103_152834.fits",
##                 "161103_152926.fits",
##                 "161103_153008.fits",
##                 "161103_153049.fits",
##                 "161103_153133.fits"]
##    lightson =  ["161103_152245.fits",
##                 "161103_152335.fits",
##                 "161103_152418.fits",
##                 "161103_152458.fits",
##                 "161103_152541.fits",
##                 "161103_152636.fits",
##                 "161103_152713.fits",
##                 "161103_152803.fits",
##                 "161103_152851.fits",
##                 "161103_152937.fits",
##                 "161103_153018.fits",
##                 "161103_153102.fits",
##                 "161103_153142.fits"]
##    bias =   [ 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5]
##    timing = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
##    capac  = [37.3,34.8,33.2,32.1,31.4, 31.,30.7,30.5, 30.5, 30.5, 30.5, 30.5, 30.5, 30.5]
##    cutoff = -5000
##    unity = 0
##    start = 5
##    title = "Optical gain @ 1.7um for SAPHIRA device M06665-23 (T = 60K, VDD = 3.5V) w/ ARC 3 Nov 2016"
##    window = [[0, -1],[0,-1]]

##    lightsoff = ["170313_115023.fits",
##                 "170313_115127.fits",
##                 "170313_115234.fits",
##                 "170313_115330.fits",
##                 "170313_115416.fits",
##                 "170313_115507.fits",
##                 "170313_115549.fits",
##                 "170313_115634.fits",
##                 "170313_115723.fits",
##                 "170313_115802.fits",
##                 "170313_115852.fits",
##                 "170313_115928.fits",
##                 "170313_120005.fits",
##                 "170313_155328.fits"]
##    lightson =  ["170313_115054.fits",
##                 "170313_115201.fits",
##                 "170313_115257.fits",
##                 "170313_115342.fits",
##                 "170313_115429.fits",
##                 "170313_115520.fits",
##                 "170313_115605.fits",
##                 "170313_115646.fits",
##                 "170313_115737.fits",
##                 "170313_115817.fits",
##                 "170313_115902.fits",
##                 "170313_115937.fits",
##                 "170313_120017.fits",
##                 "170313_155342.fits"]
##    bias =   [ 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5]
##    timing = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
##    capac =  [41.3,37.3,34.8,33.2,32.1,31.4,31.0,30.7,30.5, 30.5, 30.5, 30.5, 30.5, 30.5]
##    cutoff = -9000
##    unity = 1
##    start = 1
##    title = "Optical gain for SAPHIRA device M09225-27 (T = 60K, VDD = 4.0V) 13 Mar 2017"
##    window = [[64, -64],[96,-96]]

##    lightsoff = ["170324_100357.fits",
##                 "170324_100436.fits",
##                 "170324_100530.fits",
##                 "170324_100605.fits",
##                 "170324_100644.fits",
##                 "170324_100742.fits",
##                 "170324_100822.fits",
##                 "170324_100903.fits",
##                 "170324_100938.fits",
##                 "170324_101018.fits",
##                 "170324_101055.fits",
##                 "170324_101131.fits",
##                 "170324_101218.fits",
##                 "170324_101303.fits",
##                 "170324_101342.fits",
##                 "170324_101427.fits",
##                 "170324_101510.fits"]
##    lightson =  ["170324_100408.fits",
##                 "170324_100448.fits",
##                 "170324_100541.fits",
##                 "170324_100616.fits",
##                 "170324_100653.fits",
##                 "170324_100751.fits",
##                 "170324_100831.fits",
##                 "170324_100912.fits",
##                 "170324_100948.fits",
##                 "170324_101030.fits",
##                 "170324_101105.fits",
##                 "170324_101145.fits",
##                 "170324_101228.fits",
##                 "170324_101314.fits",
##                 "170324_101353.fits",
##                 "170324_101436.fits",
##                 "170324_101527.fits"]
##    bias =   [ 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5]
##    timing = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
##    capac =  [46.5,41.3,39.1,37.3,35.8,34.8,33.2,32.1,31.4,31.0,30.7,30.5, 30.5, 30.5, 30.5, 30.5, 30.5]
##    cutoff = -9000
##    unity = 0
##    start = 1
##    title = "Optical gain for SAPHIRA device M09225-11 (T = 60K, VDD = 4.0V) 24 Mar 2017"
##    window = [[64, -64],[96,-96]]

##    lightsoff = ["170410_100055.fits",
##                 "170410_100131.fits",
##                 "170410_100209.fits",
##                 "170410_100244.fits",
##                 "170410_100320.fits",
##                 "170410_100359.fits",
##                 "170410_100440.fits",
##                 "170410_100514.fits",
##                 "170410_100549.fits",
##                 "170410_100626.fits",
##                 "170410_100659.fits",
##                 "170410_100736.fits",
##                 "170410_100813.fits",
##                 "170410_100846.fits",
##                 "170410_100923.fits",
##                 "170410_100955.fits",
##                 "170410_101034.fits"]
##    lightson =  ["170410_100105.fits",
##                 "170410_100139.fits",
##                 "170410_100219.fits",
##                 "170410_100253.fits",
##                 "170410_100330.fits",
##                 "170410_100409.fits",
##                 "170410_100450.fits",
##                 "170410_100523.fits",
##                 "170410_100602.fits",
##                 "170410_100635.fits",
##                 "170410_100712.fits",
##                 "170410_100745.fits",
##                 "170410_100821.fits",
##                 "170410_100857.fits",
##                 "170410_100931.fits",
##                 "170410_101011.fits",
##                 "170410_101043.fits"]
##    bias =   [ 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5]
##    timing = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
##    capac =  [46.5,41.3,39.1,37.3,35.8,34.8,33.2,32.1,31.4,31.0,30.7,30.5, 30.5, 30.5, 30.5, 30.5, 30.5]
##    cutoff = -15000
##    unity = 0
##    start = 1
##    title = "Optical gain for SAPHIRA device M09215-10 (T = 60K, VDD = 4.0V) 10 Apr 2017"
##    window = [[64, -64],[96,-96]]

##    lightsoff = ["170413_104900.fits",
##                 "170413_104950.fits",
##                 "170413_105027.fits",
##                 "170413_105101.fits",
##                 "170413_105138.fits",
##                 "170413_105213.fits",
##                 "170413_105250.fits",
##                 "170413_105331.fits",
##                 "170413_105407.fits",
##                 "170413_105445.fits",
##                 "170413_105524.fits",
##                 "170413_105601.fits",
##                 "170413_105634.fits"]
##    lightson =  ["170413_104918.fits",
##                 "170413_104959.fits",
##                 "170413_105036.fits",
##                 "170413_105111.fits",
##                 "170413_105147.fits",
##                 "170413_105223.fits",
##                 "170413_105259.fits",
##                 "170413_105343.fits",
##                 "170413_105416.fits",
##                 "170413_105456.fits",
##                 "170413_105533.fits",
##                 "170413_105610.fits",
##                 "170413_105643.fits"]
##    bias =   [ 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5]
##    timing = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
##    capac =  [37.3,34.8,33.2,32.1,31.4,31.0,30.7,30.5, 30.5, 30.5, 30.5, 30.5, 30.5]
##    cutoff = -15000
##    unity = 0
##    start = 1
##    title = "Optical gain for SAPHIRA device M09215-10 (T = 60K, VDD = 4.0V) 13 Apr 2017"
##    window = [[64, -64],[96,-96]]

##    lightsoff = ["170417_114947.fits",
##                 "170417_115051.fits",
##                 "170417_115132.fits",
##                 "170417_115209.fits",
##                 "170417_115305.fits",
##                 "170417_115343.fits",
##                 "170417_115418.fits",
##                 "170417_115454.fits",
##                 "170417_115532.fits",
##                 "170417_115605.fits",
##                 "170417_115639.fits",
##                 "170417_115714.fits",
##                 "170417_115801.fits",
##                 "170417_115833.fits",
##                 "170417_115926.fits",
##                 "170417_120006.fits",
##                 "170417_120038.fits"]
##    lightson =  ["170417_115015.fits",
##                 "170417_115110.fits",
##                 "170417_115142.fits",
##                 "170417_115218.fits",
##                 "170417_115313.fits",
##                 "170417_115353.fits",
##                 "170417_115430.fits",
##                 "170417_115504.fits",
##                 "170417_115542.fits",
##                 "170417_115614.fits",
##                 "170417_115651.fits",
##                 "170417_115723.fits",
##                 "170417_115810.fits",
##                 "170417_115842.fits",
##                 "170417_115934.fits",
##                 "170417_120015.fits",
##                 "170417_120047.fits"]
##    bias =   [ 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5]
##    timing = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
##    capac =  [46.5,41.3,39.1,37.3,35.8,34.8,33.2,32.1,31.4,31.0,30.7,30.5, 30.5, 30.5, 30.5, 30.5, 30.5]
##    cutoff = -15000
##    unity = 0
##    start = 1
##    title = "Optical gain for SAPHIRA device M09215-18\nT = 60K, VDD = 4.2V, PRV = 4.0V 17 Apr 2017"
##    window = [[64, -64],[96,-96]]
    
    gain = []
    biases = []

    
##    window = [[175,195],[220,235]]
##    window = [[175,195],[120,135]]
##    window = [[75,95],[120,135]]
##    window = [[32,-32],[32,-32]]
##    window = [[64, -64],[96,-96]]
    i = 0
##    print len(lightson),len(lightsoff),len(bias)
    for i in range(len(lightson)):
##        print i
        print lightson[i], lightsoff[i], bias[i]
        on = openfits(lightson[i])
        off = openfits(lightsoff[i])
        print on.shape, off.shape
        onsub = subtractnth(on,n=start)
        offsub = subtractnth(off,n=start)
        n = 1
        print numpy.mean(on[-n,window[0][0]:window[0][1],window[1][0]:window[1][1]]), cutoff
        print on[-n,window[0][0],window[1][0]]
        if cutoff > 0:
            while abs(numpy.mean(on[-n,window[0][0]:window[0][1],window[1][0]:window[1][1]])) < abs(cutoff):
                n += 1
        else:
            while abs(numpy.mean(onsub[-n,window[0][0]:window[0][1],window[1][0]:window[1][1]])) > abs(cutoff):
                n += 1
        if n >= on.shape[0] - 1:
            print "Image saturates immediately."
        else:
            print "Using frame #",on.shape[0]-n
            diff = offsub[-n,window[0][0]:window[0][1],window[1][0]:window[1][1]] - onsub[-n,window[0][0]:window[0][1],window[1][0]:window[1][1]]
            m = numpy.median(diff)
            toPlot = []
##            for j in range(len(offsub)): #Temp
##                toPlot.append(numpy.median(off[j,window[0][0]:window[0][1],window[1][0]:window[1][1]]) - numpy.median(on[j,window[0][0]:window[0][1],window[1][0]:window[1][1]])) #Temp
##            plt.plot(range(len(toPlot)),toPlot) #Temp
            print "Difference mean:", m
            if capac:
                m *= capac[i] / capac[0]
            t = (onsub.shape[0] - n) * timing[i]
            gain.append(m / t)
            print on.shape[0] - n
            print onsub.shape[0] - n
            print "Measured gain:", (m/t) / gain[0]
            biases.append(bias[i])
            plt.subplot(5,4,i)
            plt.plot(numpy.median(numpy.median(offsub, axis = 1), axis = 1))
            plt.plot(numpy.median(numpy.median(onsub, axis = 1), axis = 1))
            plt.title("Bias Voltage: " + str(bias[i]) + "V")
            plt.legend(["off","on"], loc = 3)
    plt.show()
            
##    plt.show() #Temp
    gain /= gain[unity]
    fig, ax = plt.subplots()
    fig.set_facecolor('white')
##    ax.semilogy(biases,gain,'bo-') #blue
    ax.semilogy(biases,gain,'ko-') #black
##    plt.ylim(0.7,1000)
    plt.ylim(0.7,200)
    plt.grid(True)
    plt.grid(b=True, which='minor',axis='y')
    #Label points.
    for b, g in zip(biases,gain):
        if float(g) > 10:
            gTxt = round(g, 1)
        else:
            gTxt = round(g, 2)
        ax.annotate('{}'.format(gTxt),xy=(b,g),xytext = (-5,5), ha = 'right', textcoords = 'offset points')
    if capac:
        title += "\nCapacitance-Corrected"
    plt.xlabel("Bias (V)")
    plt.ylabel("Gain (normalized)")
    plt.title(title)
    plt.show()

def voltageGain():
##    files = ["140313_100047f.fits","140313_100240f.fits","140313_100430f.fits","140313_100725f.fits","140313_101034f.fits","140313_101315f.fits"]
    #0V common
##    files = ["140313_105257f.fits","140313_105428f.fits","140313_105515f.fits","140313_105710f.fits","140313_105800f.fits"]
    #-8V common
##    files = ["140313_105916f.fits","140313_110000f.fits","140313_110058f.fits","140313_110137f.fits","140313_110219f.fits"]
    #video board only
##    files = ["140314_100204f.fits","140314_100312f.fits","140314_100348f.fits","140314_100402f.fits","140314_100420f.fits","140314_100441f.fits"]
    #-11V common M02775-10
##    files = ["140414_145420f.fits","140414_145446f.fits","140414_145510f.fits","140414_145537f.fits","140414_145559f.fits"]
##    files = ["140414_163040f.fits","140414_163237f.fits","140414_163420f.fits","140414_163610f.fits","140414_163740f.fits"]
    #-11V common M04055-06
##    files = ["140602_091605.fits","140602_091636.fits","140602_091702.fits","140602_091725.fits","140602_091746.fits"]
##    files = ["140602_093723.fits","140602_093759.fits","140602_093852.fits","140602_093920.fits","140602_093942.fits"]
##    files = ["140602_133254.fits","140602_133323.fits","140602_133343.fits","140602_133419.fits","140602_133442.fits"]
    #1V common VDD = 3.5V
##    files = ["141216_103417.fits","141216_103503.fits","141216_103534.fits","141216_103610.fits","141216_103636.fits"]
    #M06665-03 VDD = 3.5V
##    files = ["151104_131937.fits","151104_132038.fits","151104_132107.fits","151104_132131.fits","151104_132155.fits"]
    #M06715-34 VDD = 4.5V
##    PRV = [3.5,3.4,3.3,3.2,3.1,3.0]
##    PRV = [3.5,3.45,3.4,3.35,3.3]
##    PRV = [4.0, 3.95, 3.9, 3.85, 3.8]
##    PRV = [.3530,.3093,.2583,.2047,.1514,.1059]
##    window = [[175,195],[220,235]]
##    window = [[10,255],[130,134]]
##    window = [[0,64],[0,64]]

##    files = ["160415_PRV_4.51V-11-0.fits",
##             "160415_PRV_4.45V-12-0.fits",
##             "160415_PRV_4.39V-13-0.fits",
##             "160415_PRV_4.33V-14-0.fits",
##             "160415_PRV_4.26V-15-0.fits"]
##    PRV = [4.51, 4.45, 4.39, 4.33, 4.26]

    #M06665-25 VDD = 3.5V
##    files = ["161012_164943.fits",
##             "161012_165111.fits",
##             "161012_165146.fits",
##             "161012_165217.fits",
##             "161012_165245.fits"]
##    PRV = [3.5, 3.45, 3.4, 3.35, 3.3]

    #M06665-25 VDD = 4.0V
##    files = ["161013_141139.fits",
##             "161013_141305.fits",
##             "161013_141340.fits",
##             "161013_141409.fits",
##             "161013_141442.fits",
##             "161013_141512.fits",
##             "161013_141547.fits"]
##    PRV = [3.6, 3.55, 3.5, 3.45, 3.4, 3.35, 3.3]


    #M06665-23 VDD = 4.0V
##    files = ["161016_143509.fits",
##             "161016_143539.fits",
##             "161016_143605.fits",
##             "161016_143627.fits",
##             "161016_143726.fits",
##             "161016_143754.fits",
##             "161016_143821.fits"]
##    PRV = [3.6, 3.55, 3.5, 3.45, 3.4, 3.35, 3.3]
##    PRV = [3.5831, 3.5407, 3.4982, 3.4486, 3.3991, 3.3493, 3.2998]

    #M06665-23 VDD = 4.0V
##    files = ["161103_101027.fits",
##             "161103_101109.fits",
##             "161103_101208.fits",
##             "161103_101233.fits",
##             "161103_101329.fits",
##             "161103_101416.fits",
##             "161103_101455.fits"]
##    PRV = [3.5831, 3.5407, 3.4982, 3.4486, 3.3991, 3.3493, 3.2998]

    #Leach only
##    files = ["161103_140919.fits",
##             "161103_140950.fits",
##             "161103_140905.fits",
##             "161103_141004.fits",
##             "161103_140743.fits"]
##    PRV = [0.200, 0.250, 0.300, 0.350, 0.400]

##    #M06665-23 VDD = 4.0V
##    files = ["161107_153612.fits",
##             "161107_153653.fits",
##             "161107_153733.fits",
##             "161107_153835.fits",
##             "161107_153924.fits",
##             "161107_154029.fits",
##             "161107_154131.fits"]
##    PRV = [3.5831, 3.5407, 3.4982, 3.4486, 3.3991, 3.3493, 3.2998]

    #M09225-27 VDD = 5.0V
##    files = ["170313_161919.fits",
##             "170313_161952.fits",
##             "170313_162018.fits",
##             "170313_162042.fits",
##             "170313_162103.fits"]
##    files = ["170313_170816.fits",
##             "170313_170907.fits",
##             "170313_170948.fits",
##             "170313_171010.fits",
##             "170313_171053.fits"]
##    PRV = [4.5, 4.45, 4.4, 4.35, 4.3]
##    files = ["170314_091829.fits",
##             "170314_092006.fits",
##             "170314_092058.fits",
##             "170314_092127.fits",
##             "170314_092605.fits",
##             "170314_092450.fits",
##             "170314_092655.fits",
##             "170314_094512.fits",
##             "170314_092655.fits"]
##    PRV = [4.5856, 4.5434, 4.5012, 4.4615, 4.4216, 4.3816, 4.3416, 4.3416, 4.3016]
##    files = ["170321_152211.fits",
##             "170321_152404.fits",
##             "170321_152437.fits",
##             "170321_152458.fits",
##             "170321_152523.fits"]
##    PRV = [4.5855, 4.5434, 4.5011, 4.4614, 4.4215]

    #M09225-27 VDD = 5.0V
##    files = ["170313_162404.fits",
##             "170313_162426.fits",
##             "170313_162448.fits",
##             "170313_162510.fits",
##             "170313_162531.fits"]
##    PRV = [4.5, 4.45, 4.4, 4.35, 4.3]

    #M09225-11 VDD = 4.0V
##    files = ["170324_110224.fits",#COMMON = 1V
##             "170324_110333.fits",
##             "170324_110405.fits",
##             "170324_110433.fits",
##             "170324_110456.fits",
##             "170324_110524.fits",
##             "170324_110545.fits"]
##    files = ["170324_135322.fits",#COMMON = 2.5V
##             "170324_135346.fits",
##             "170324_135410.fits",
##             "170324_135430.fits",
##             "170324_135451.fits",
##             "170324_135517.fits",
##             "170324_135539.fits"]
##    PRV = [4.5856, 4.5434, 4.5012, 4.4615, 4.4216, 4.3816, 4.3416]

##    files = ["170406_165915.fits",
##             "170406_165941.fits",
##             "170406_170009.fits",
##             "170406_170030.fits",
##             "170406_170052.fits"] #VDD = 3.5V
##    files = ["170406_171101.fits",
##             "170406_171123.fits",
##             "170406_171144.fits",
##             "170406_171204.fits",
##             "170406_171227.fits",
##             "170406_171249.fits",
##             "170406_171306.fits"]
##    PRV = [3.5831, 3.5407, 3.4982, 3.4486, 3.3991, 3.3493, 3.2998]

    #M09215-10 VDD = 4.0V
##    files = ["170408_161941.fits",
##             "170408_162008.fits",
##             "170408_162027.fits",
##             "170408_162048.fits",
##             "170408_162107.fits",
##             "170408_162128.fits",
##             "170408_162149.fits"]#100K
##    files = ["170409_160301.fits",
##             "170409_160339.fits",
##             "170409_160403.fits",
##             "170409_160423.fits",
##             "170409_160444.fits",
##             "170409_160504.fits",
##             "170409_160526.fits"]#80K
    files = ["170410_093424.fits",
             "170410_093451.fits",
             "170410_093525.fits",
             "170410_093548.fits",
             "170410_093611.fits",
             "170410_093632.fits",
             "170410_093653.fits"]
    PRV = [3.5831, 3.5407, 3.4982, 3.4486, 3.3991, 3.3493, 3.2998]

        
    window = [[64,128],[96,196]]
##    window = [[0,-1],[0,8]] #Injected onto video card #1.
    means = []
    for f in files:
        d = openfits(f)[10:,window[0][0]:window[0][1],window[1][0]:window[1][1]]
        means.append(numpy.median(d))
        medians = []
        for i in range(d.shape[0]):
            medians.append(numpy.median(d[i,:,:]))
##        plt.plot(medians)
##        plt.show()
##    diffs = []
##    for i in range(len(means)-1):
##        diffs.append(means[i] - means[i + 1])
    plt.plot(PRV,means,'o-',label = "Data")
    plt.ylabel("median (ADU)")
    plt.xlabel("PRV (V)")
    m,b = linearfit(PRV,means)
    print 1e6/m, "uV/ADU"
    Vmin = numpy.min(PRV)
    Vmax = numpy.max(PRV)
    plt.plot([Vmin,Vmax],[Vmin * m + b,Vmax * m + b],'k--',label = str(round(1e6/m,2)) + "$\mu$V/ADU")
    plt.legend(loc = 2)
    plt.title("Voltage Gain")
    plt.show()
##    diff = mean(diffs)
##    print diffs
####    print 100000 / diff, "uV/ADU"
##    print 50000 / diff, "uV/ADU"

def voltageGain2():
    data = numpy.array([[["150128_141016.fits","150128_141020.fits","150128_141023.fits","150128_141026.fits","150128_141029.fits"],\
          ["150128_141113.fits","150128_141119.fits","150128_141122.fits","150128_141125.fits","150128_141128.fits"]],\
         [["150128_141319.fits","150128_141324.fits","150128_141328.fits","150128_141332.fits","150128_141335.fits"],\
          ["150128_141414.fits","150128_141419.fits","150128_141422.fits","150128_141425.fits","150128_141428.fits"]],\
         [["150128_142042.fits","150128_142108.fits","150128_142113.fits","150128_142117.fits","150128_142121.fits"],\
          ["150128_142153.fits","150128_142158.fits","150128_142202.fits","150128_142206.fits","150128_142209.fits"]],\
         [["150128_112311.fits","150128_112315.fits","150128_112319.fits","150128_112323.fits","150128_112327.fits"],\
          ["150128_112354.fits","150128_112357.fits","150128_112401.fits","150128_112404.fits","150128_112407.fits"]]])
    PRV = [4.5, 4, 3.5, 3.5]
    VDD = [5, 4.5, 4, 3.5]
    for j, d in enumerate(data):
        diff = 0
        for i in range(d.shape[1]):
            high = openfits(d[0,i])
            low = openfits(d[1,i])
            highpix = numpy.zeros([high.shape[1],high.shape[2]])
            lowpix = numpy.zeros([low.shape[1],low.shape[2]])
            for y in range(high.shape[1]):
                for x in range(high.shape[2]):
                    highpix = numpy.median(high[3:,y,x])
                    lowpix = numpy.median(low[3:,y,x])
            diffpix = highpix - lowpix
            diff += numpy.mean(diffpix)
        diff /= d.shape[1]
        print "VDD:", VDD[j], "PRV:", PRV[j], "Voltage Gain:", 50000 / diff, "uV/ADU"

def darkCurrent(filename, timing = 1):
##    window = [[175,195],[235,245]]
##    window = [[155, 215],[205,275]]
##    window = [[70,110],[128,144]]
##    window = [[32,-32],[32,-32]]
##    window = [[0,48],[32,-32]] #bottom strip
##    window = [[64,-64],[96, -96]] #mask hole
    window = [[-28, 256],[32, -32]] #mask covered
##    window = [[0, 32],[0, 32]] #specular glow
##    cutoff = 25000
    cutoff = 20000
##    cutoff = 0
    data = openfits(filename)[:,window[0][0]:window[0][1],window[1][0]:window[1][1]]
    stop = 0
    for i in range(data.shape[0]):
##        if mean(mean(data[i,:,:])) > cutoff:
        if numpy.median(data[i,:,:]) > cutoff:
            stop = i
##    d = (data[5,:,:] - data[stop,:,:]) * 2.89 / timing
    print "Using",stop,"frames"
##    cts = mean(mean(d))
    cts = (numpy.median(data[5,:,:]) - numpy.median(data[stop,:,:])) * 2.89 / timing
##    cts = numpy.median(d)
##    noise = stddev(d.flatten())
##    noise = numpy.std(d, ddof = 1)
    print cts / (stop - 5) #, "+/-", noise / (stop - 5), "e- / sec"

def chargeGain(nfiles = 10, maskper = True, maskmean = False, maskmid = 7000, maskrange = 100):
    #nfiles: umber of files to run, starting with the first.
    #maskper: Mask per each file, or just the first and apply to all?
    #"140317_094858f.fits"
##    filenames = ["140317_095123f.fits","140317_095212f.fits","140317_095331f.fits",\
##                 "140317_095400f.fits","140317_095429f.fits","140317_101616f.fits","140317_101654f.fits","140317_101721f.fits"]
    filenames = ["141208_154132.fits","141208_154144.fits","141208_154149.fits",\
                 "141208_154154.fits","141208_154157.fits","141208_154204.fits",\
                 "141208_154210.fits","141208_154213.fits","141208_154216.fits",\
                 "141208_154219.fits"]
    if nfiles > len(filenames):
        nfiles = len(filenames)
    data = openfits(filenames[0])
    ylist = []
    xlist = []
    signals = numpy.zeros((len(filenames),data.shape[0] - 11))
    variances = numpy.zeros((len(filenames),data.shape[0] - 11))
    if maskper == True and maskmean == True:
        print "Cannot mask per frame and use mean mask simultaneously."
        maskper = False
    for n,f in enumerate(filenames[:nfiles]):
        data = openfits(f)
        if maskper or n == 0:
            #If we want masks for each frame, generate them here.
            #Otherwise, just do it for the first one.
            ylist = []
            xlist = []
            #If we want to generate the mask from a mean image, replace the data
            #with all the images meaned together.
            if maskmean:
                data = numpy.zeros(data.shape)
                for g in filenames:
                    data += openfits(g)
                data /= len(filenames)
            for y in range(data.shape[1]):
                for x in range(data.shape[2]):
                    v = data[10,y,x] - data[-1,y,x]
                    if v > maskmid - maskrange and v < maskmid + maskrange:
                        ylist.append(y)
                        xlist.append(x)
            print len(ylist), "pixels passed through mask."
            #Put the first frame data back if we substituted in the meaned data.
            if maskmean:
                data = openfits(f)
        for i in range(11,data.shape[0]):
            d = []
            for y,x in map(None, ylist, xlist):
                d.append(data[10,y,x] - data[i,y,x])
            signals[n,i - 11] = mean(d)
            variances[n,i - 11] = stddev(d)**2
        print "Ramp",n,"processed."
        plt.plot(range(11,data.shape[0]), data[11:,ylist[50],xlist[50]] - data[10,ylist[50],xlist[50]])
    plt.legend()
    plt.show()
    sigmean = []
    varmean = []
    for i in range(11,data.shape[0]):
        #Mean the results from the available datasets.
        sigmean.append(mean(signals[:,i - 11]))
        varmean.append(mean(variances[:,i - 11]))
    m, b = linearfit(sigmean[:-4], varmean[:-4])
    vfit = []
    for s in sigmean:
        vfit.append(s * m + b)
    if b < 0:
        print "Problem: Y-intercept below 0."
        b = 0
    plt.plot(sigmean,varmean,'r+')
    plt.plot(sigmean,vfit,'k')
    plt.title("RN = " + str(int(100 * sqrt(b/m)) / 100.) + r'$e^-$' + ",  gain = " + str(int(100 * 1/m) / 100.) + r'$e^-$' + "/ADU" +\
              "     maskinterval = (" + str(maskmid - maskrange) + "," + str(maskmid + maskrange) + ")")
##    plt.title("Charge gain")
    print "RN = ", str(sqrt(b/m)), "gain = ", str(1/m)
    plt.xlabel("Signal (ADU)")
    plt.ylabel("Variance (ADU**2)")
    plt.show()

def avg(data, avg = 4, rolling = False):
    d = numpy.array(data)
    if rolling:
        averaged = numpy.zeros((d.shape[0] - avg))
        if avg > 1:
            for i in range(averaged.shape[0]):
                for j in range(avg):
                    averaged[i] += d[i + j]
                averaged[i] /= avg
        else:
            return d
    else:
        averaged = numpy.zeros((d.shape[0] / avg))
        if avg > 1:
            for i in range(averaged.shape[0]):
                for j in range(avg):
                    averaged[i] += d[i * avg + j]
                averaged[i] /= avg
        else:
            return d
    return averaged

def avgCube(data, avg = 4, rolling = False):
    d = numpy.array(data)
    if rolling:
        averaged = numpy.zeros((d.shape[0] - avg,d.shape[1],d.shape[2]))
        if avg > 1:
            for i in range(averaged.shape[0]):
                for j in range(avg):
                    averaged[i,:,:] += d[i + j,:,:]
                averaged[i,:,:] /= avg
        else:
            averaged = d
    else:
        averaged = numpy.zeros((d.shape[0] / avg,d.shape[1],d.shape[2]))
        if avg > 1:
##            print "Warning: Old averaging technique in use."
            for i in range(averaged.shape[0]):
                for j in range(avg):
                    averaged[i,:,:] += d[i * avg + j,:,:]
                averaged[i,:,:] /= avg
        else:
            averaged = d
    return averaged    

def avgFits(filename, avg = 4):
    averaged = avgCube(openfits(filename),avg)
    savefits(filename[:-5] + "avg" + str(avg) + ".fits", averaged)

def medianPixels(data):
    median = numpy.zeros((data.shape[0]))
    for i in range(data.shape[0]):
        d = numpy.array(data[i,:,:])
        e = numpy.sort(d,axis = None)
        median[i] = e[len(e)/2]
    return median
    
def LEDonofframps(x = 3):
    length = 20000
##    title = "M02885-12, VDD = 4V, COMMON = -8V  "
##    filenames = ["140408_114000f.fits","140408_114007f.fits","140408_114010f.fits","140408_114013f.fits","140408_114016f.fits",\
##                 "140408_114019f.fits","140408_114021f.fits","140408_114024f.fits","140408_114027f.fits","140408_114030f.fits"]
##    filenames2 =["140408_114046f.fits","140408_114050f.fits","140408_114053f.fits","140408_114057f.fits","140408_114100f.fits",\
##                 "140408_114103f.fits","140408_114107f.fits","140408_114109f.fits","140408_114112f.fits","140408_114115f.fits"]
##    title = "M02885-12, VDD = 4V, COMMON = -9V  "
##    filenames = ["140408_161829f.fits","140408_161849f.fits","140408_161853f.fits","140408_161859f.fits","140408_161905f.fits",\
##                 "140408_161913f.fits","140408_161919f.fits","140408_161928f.fits","140408_161933f.fits","140408_161938f.fits"]
##    filenames2 =["140408_161956f.fits","140408_161959f.fits","140408_162002f.fits","140408_162005f.fits","140408_162008f.fits",\
##                 "140408_162011f.fits","140408_162014f.fits","140408_162017f.fits","140408_162020f.fits","140408_162023f.fits"]
##    title = "M02885-12, VDD = 4V, COMMON = -10V  "
##    filenames = ["140408_164740f.fits","140408_164746f.fits","140408_164749f.fits","140408_164752f.fits","140408_164755f.fits",\
##                 "140408_164800f.fits","140408_164805f.fits","140408_164809f.fits","140408_164812f.fits","140408_164814f.fits"]
##    filenames2 =["140408_164825f.fits","140408_164829f.fits","140408_164833f.fits","140408_164836f.fits","140408_164839f.fits",\
##                 "140408_164843f.fits","140408_164847f.fits","140408_164850f.fits","140408_164856f.fits","140408_164901f.fits"]
    title = "M02885-12, VDD = 4V, COMMON = -11V  "
    filenames = ["140408_170409f.fits","140408_170434f.fits","140408_170437f.fits","140408_170440f.fits","140408_170443f.fits",\
                 "140408_170447f.fits","140408_170450f.fits","140408_170454f.fits","140408_170457f.fits","140408_170500f.fits"]
    filenames2 =["140408_170512f.fits","140408_170541f.fits","140408_170544f.fits","140408_170547f.fits","140408_170551f.fits",\
                 "140408_170554f.fits","140408_170557f.fits","140408_170600f.fits","140408_170603f.fits","140408_170606f.fits"]
    for i in range(len(filenames)):
        data = openfits(filenames[i])
        plt.plot(range(data.shape[0]),data[:,0,x], label = filenames[i][7:13])
    plt.title(title + " pixel " + str(x) + ", LED off")
    plt.legend()
    plt.show()
    for i in range(len(filenames2)):
        data = openfits(filenames2[i])
        plt.plot(range(data.shape[0]),data[:,0,x], label = filenames2[i][7:13])
    plt.title(title + " pixel " + str(x) + ", LED on")
    plt.legend()
    plt.show()


def countPhotons(x = 0, avg = 2, histrange = (0,30), binspacing = 0.5, rampplot = False, screenforpeaks = False, mediansubtraction = False,\
                 fullstatistics = False, perpixelstatistics = False, ratioplot = False, meanplot = False, onoffplot = False,\
                 writecsv = False, csvfile = "default.csv"):
    nbins = (histrange[1] - histrange[0]) / binspacing
##    filenames = ["140218_130855f.fits","140218_130905f.fits","140218_130911f.fits","140218_130917f.fits","140218_130924f.fits",\
##                 "140218_130935f.fits","140218_130941f.fits","140218_130948f.fits","140218_130955f.fits","140218_131002f.fits"]
##    filenames2 = ["140218_131023f.fits","140218_131028f.fits","140218_131035f.fits","140218_131042f.fits","140218_131049f.fits",\
##                 "140218_131056f.fits","140218_131102f.fits","140218_131107f.fits","140218_131114f.fits","140218_131119f.fits"]
##    filenames = ["140220_152022f.fits","140220_152028f.fits","140220_152035f.fits","140220_152042f.fits",\
##                 "140220_152047f.fits","140220_152054f.fits","140220_152101f.fits","140220_152108f.fits","140220_152114f.fits"]
##    filenames2 = ["140220_152154f.fits","140220_152159f.fits","140220_152205f.fits","140220_152211f.fits",\
##                 "140220_152216f.fits","140220_152223f.fits","140220_152230f.fits","140220_152236f.fits","140220_152242f.fits"]
##    filenames = ["140224_152725f.fits","140224_152737f.fits","140224_152742f.fits","140224_152746f.fits","140224_152750f.fits",\
##                 "140224_152758f.fits","140224_152803f.fits","140224_152811f.fits","140224_152815f.fits","140224_152820f.fits"]
##    filenames2 = ["140224_152844f.fits","140224_152851f.fits","140224_152857f.fits","140224_152902f.fits","140224_152906f.fits",\
##                 "140224_152911f.fits","140224_152915f.fits","140224_152920f.fits","140224_152925f.fits","140224_152930f.fits"]
##    filenames = ["140224_164447f.fits","140224_164454f.fits","140224_164504f.fits","140224_164511f.fits","140224_164514f.fits",\
##                 "140224_164518f.fits","140224_164525f.fits","140224_164530f.fits","140224_164535f.fits","140224_164540f.fits"]
##    filenames2 = ["140224_164551f.fits","140224_164603f.fits","140224_164608f.fits","140224_164613f.fits","140224_164619f.fits",\
##                 "140224_164624f.fits","140224_164630f.fits","140224_164634f.fits","140224_164638f.fits","140224_164643f.fits"]
##    filenames = ["140226_151834f.fits","140226_151841f.fits","140226_151846f.fits","140226_151850f.fits","140226_151854f.fits",\
##                 "140226_151859f.fits","140226_151904f.fits","140226_151908f.fits","140226_151912f.fits","140226_151915f.fits"]
##    filenames = ["140227_161030f.fits","140227_161038f.fits","140227_161042f.fits","140227_161046f.fits","140227_161046f.fits",\
##                 "140227_161052f.fits","140227_161056f.fits","140227_161102f.fits","140227_161105f.fits","140227_161109f.fits"]
    #0.95V
##    filenames2 = ["140226_151933f.fits","140226_151937f.fits","140226_151941f.fits","140226_151944f.fits","140226_151947f.fits",\
##                 "140226_151951f.fits","140226_151954f.fits","140226_151959f.fits","140226_152003f.fits","140226_152008f.fits"]
    #0.75V
##    filenames2 = ["140227_153949f.fits","140227_153958f.fits","140227_154005f.fits","140227_154011f.fits","140227_154016f.fits",\
##                 "140227_154025f.fits","140227_154032f.fits","140227_154039f.fits","140227_154043f.fits","140227_154049f.fits"]
    #VDD = 4V COMMON = -5V
##    filenames2 = ["140318_140330f.fits","140318_140411f.fits","140318_140416f.fits","140318_140424f.fits","140318_140430f.fits",\
##                 "140318_140437f.fits","140318_140444f.fits","140318_140451f.fits","140318_140456f.fits","140318_140502f.fits"]
##    filenames = ["140318_140521f.fits","140318_140526f.fits","140318_140532f.fits","140318_140537f.fits","140318_140542f.fits",\
##                 "140318_140549f.fits","140318_140554f.fits","140318_140600f.fits","140318_140605f.fits","140318_140609f.fits"]
    #VDD = 4V COMMON = -6V
##    filenames2 = ["140318_140724f.fits","140318_140732f.fits","140318_140737f.fits","140318_140742f.fits","140318_140747f.fits",\
##                 "140318_140753f.fits","140318_140758f.fits","140318_140803f.fits","140318_140810f.fits","140318_140815f.fits"]
##    filenames = ["140318_140836f.fits","140318_140842f.fits","140318_140847f.fits","140318_140852f.fits","140318_140857f.fits",\
##                  "140318_140902f.fits","140318_140907f.fits","140318_140912f.fits","140318_140917f.fits","140318_140922f.fits"]
    #VDD = 4V COMMON = -8V "140318_141030f.fits",   "140318_141134f.fits",
##    filenames = ["140318_141037f.fits","140318_141042f.fits","140318_141047f.fits","140318_141051f.fits",\
##                 "140318_141055f.fits","140318_141100f.fits","140318_141105f.fits","140318_141110f.fits","140318_141114f.fits"]
##    filenames2 = ["140318_141138f.fits","140318_141143f.fits","140318_141147f.fits","140318_141151f.fits",\
##                 "140318_141155f.fits","140318_141201f.fits","140318_141205f.fits","140318_141209f.fits","140318_141214f.fits"]
    #New clocking, VDD = 4V, COMMON = -8V, new detector M02885-12.
##    filenames = ["140403_145008f.fits","140403_145015f.fits","140403_145018f.fits","140403_145022f.fits","140403_145024f.fits",\
##                 "140403_145027f.fits","140403_145030f.fits","140403_145034f.fits","140403_145040f.fits","140403_145043f.fits"]
##    filenames2 = ["140403_145118f.fits","140403_145123f.fits","140403_145127f.fits","140403_145130f.fits","140403_145133f.fits",\
##                 "140403_145136f.fits","140403_145139f.fits","140403_145142f.fits","140403_145145f.fits","140403_145148f.fits"]
##    filenames = ["140403_173723f.fits","140403_173733f.fits","140403_173738f.fits","140403_173742f.fits","140403_173746f.fits",\
##                 "140403_173750f.fits","140403_173753f.fits","140403_173757f.fits","140403_173802f.fits","140403_173805f.fits"]
##    filenames2 = ["140403_173836f.fits","140403_173841f.fits","140403_173844f.fits","140403_173849f.fits","140403_173852f.fits",\
##                  "140403_173856f.fits","140403_173900f.fits","140403_173903f.fits","140403_173907f.fits","140403_173910f.fits"]
##    filenames = ["140404_144236f.fits","140404_144245f.fits","140404_144251f.fits","140404_144255f.fits","140404_144258f.fits",\
##                 "140404_144302f.fits","140404_144309f.fits","140404_144313f.fits","140404_144317f.fits","140404_144321f.fits"]
##    filenames2 = ["140404_144349f.fits","140404_144352f.fits","140404_144356f.fits","140404_144359f.fits","140404_144402f.fits",\
##                  "140404_144406f.fits","140404_144409f.fits","140404_144412f.fits","140404_144415f.fits","140404_144419f.fits"]
##    filenames = ["140404_155232f.fits","140404_155237f.fits","140404_155241f.fits","140404_155244f.fits","140404_155248f.fits",\
##                 "140404_155251f.fits","140404_155254f.fits","140404_155258f.fits","140404_155303f.fits","140404_155306f.fits"]
##    filenames = ["140407_092239f.fits","140407_092247f.fits","140407_092252f.fits","140407_092301f.fits","140407_092304f.fits",\
##                 "140407_092307f.fits","140407_092311f.fits","140407_092314f.fits","140407_092318f.fits","140407_092321f.fits"]
##    filenames2 = ["140404_155326f.fits","140404_155330f.fits","140404_155334f.fits","140404_155337f.fits","140404_155341f.fits",\
##                  "140404_155344f.fits","140404_155349f.fits","140404_155352f.fits","140404_155356f.fits","140404_155400f.fits"]
    #Fixed 1/4 ramp glitching.
##    title = "M02885-12, VDD = 4V, COMMON = -8V  "
##    filenames = ["140408_114000f.fits","140408_114007f.fits","140408_114010f.fits","140408_114013f.fits","140408_114016f.fits",\
##                 "140408_114019f.fits","140408_114021f.fits","140408_114024f.fits","140408_114027f.fits","140408_114030f.fits"]
##    filenames2 =["140408_114046f.fits","140408_114050f.fits","140408_114053f.fits","140408_114057f.fits","140408_114100f.fits",\
##                 "140408_114103f.fits","140408_114107f.fits","140408_114109f.fits","140408_114112f.fits","140408_114115f.fits"]
##    blacklist = []
##    title = "M02885-12, VDD = 4V, COMMON = -9V  "
##    filenames = ["140408_161829f.fits","140408_161849f.fits","140408_161853f.fits","140408_161859f.fits","140408_161905f.fits",\
##                 "140408_161913f.fits","140408_161919f.fits","140408_161928f.fits","140408_161933f.fits","140408_161938f.fits"]
##    filenames2 =["140408_161956f.fits","140408_161959f.fits","140408_162002f.fits","140408_162005f.fits","140408_162008f.fits",\
##                 "140408_162011f.fits","140408_162014f.fits","140408_162017f.fits","140408_162020f.fits","140408_162023f.fits"]
##    blacklist = [14]
##    title = "M02885-12, VDD = 4V, COMMON = -10V  "
##    filenames = ["140408_164740f.fits","140408_164746f.fits","140408_164749f.fits","140408_164752f.fits","140408_164755f.fits",\
##                 "140408_164800f.fits","140408_164805f.fits","140408_164809f.fits","140408_164812f.fits","140408_164814f.fits"]
##    filenames2 =["140408_164825f.fits","140408_164829f.fits","140408_164833f.fits","140408_164836f.fits","140408_164839f.fits",\
##                 "140408_164843f.fits","140408_164847f.fits","140408_164850f.fits","140408_164856f.fits","140408_164901f.fits"]
##    blacklist = [0, 11, 13, 14, 15]
    title = "M02885-12, VDD = 4V, COMMON = -11V  "
    filenames = ["140408_170409f.fits","140408_170434f.fits","140408_170437f.fits","140408_170440f.fits","140408_170443f.fits",\
                 "140408_170447f.fits","140408_170450f.fits","140408_170454f.fits","140408_170457f.fits","140408_170500f.fits"]
    filenames2 =["140408_170512f.fits","140408_170541f.fits","140408_170544f.fits","140408_170547f.fits","140408_170551f.fits",\
                 "140408_170554f.fits","140408_170557f.fits","140408_170600f.fits","140408_170603f.fits","140408_170606f.fits"]
    blacklist = [0, 4, 11, 12, 13, 14, 15]
    
##    title = "M02885-12, VDD = 4V, LED off, "
##    #-8V
##    filenames = ["140408_114000f.fits","140408_114007f.fits","140408_114010f.fits","140408_114013f.fits","140408_114016f.fits",\
##                 "140408_114019f.fits","140408_114021f.fits","140408_114024f.fits","140408_114027f.fits","140408_114030f.fits"]
##    #-9V
##    filenames2 = ["140408_161829f.fits","140408_161849f.fits","140408_161853f.fits","140408_161859f.fits","140408_161905f.fits",\
##                 "140408_161913f.fits","140408_161919f.fits","140408_161928f.fits","140408_161933f.fits","140408_161938f.fits"]
##    #-10V
##    filenames3 = ["140408_164740f.fits","140408_164746f.fits","140408_164749f.fits","140408_164752f.fits","140408_164755f.fits",\
##                 "140408_164800f.fits","140408_164805f.fits","140408_164809f.fits","140408_164812f.fits","140408_164814f.fits"]
##    #-11V
##    filenames4 = ["140408_170409f.fits","140408_170434f.fits","140408_170437f.fits","140408_170440f.fits","140408_170443f.fits",\
##                 "140408_170447f.fits","140408_170450f.fits","140408_170454f.fits","140408_170457f.fits","140408_170500f.fits"]
##    blacklist = [0, 4, 11, 12, 13, 14, 15]
##    labels = ["-8V","-9V","-10V","-11V"]
    
##    title = "M02885-12, VDD = 4V, LED on, "
##    #-8V
##    filenames = ["140408_114046f.fits","140408_114050f.fits","140408_114053f.fits","140408_114057f.fits","140408_114100f.fits",\
##                 "140408_114103f.fits","140408_114107f.fits","140408_114109f.fits","140408_114112f.fits","140408_114115f.fits"]
##    #-9V
##    filenames2 =["140408_161956f.fits","140408_161959f.fits","140408_162002f.fits","140408_162005f.fits","140408_162008f.fits",\
##                 "140408_162011f.fits","140408_162014f.fits","140408_162017f.fits","140408_162020f.fits","140408_162023f.fits"]
##    #-10V
##    filenames3 = ["140408_164825f.fits","140408_164829f.fits","140408_164833f.fits","140408_164836f.fits","140408_164839f.fits",\
##                 "140408_164843f.fits","140408_164847f.fits","140408_164850f.fits","140408_164856f.fits","140408_164901f.fits"]
##    #-11V
##    filenames4 = ["140408_170512f.fits","140408_170541f.fits","140408_170544f.fits","140408_170547f.fits","140408_170551f.fits",\
##                 "140408_170554f.fits","140408_170557f.fits","140408_170600f.fits","140408_170603f.fits","140408_170606f.fits"]
##    blacklist = [0, 4, 11, 12, 13, 14, 15]
##    labels = ["-8V","-9V","-10V","-11V"]
##    lightsoff = "140407_165102f.fits"
##    lightson = "140407_165134f.fits"
    cutoff = 5000 #0 for full ramp, otherwise is number of frames in ramp
    a = openfits(filenames[0])
    pix = numpy.zeros(((a.shape[0])*len(filenames) + 10,16))
    pix2 = numpy.zeros(((a.shape[0])*len(filenames2) + 10,16))
    if 'filenames3' in locals():
        pix3 = numpy.zeros(((a.shape[0])*len(filenames3) + 10,16))
    if 'filenames4' in locals():
        pix4 = numpy.zeros(((a.shape[0])*len(filenames4) + 10,16))
    if cutoff > 0:
        pix = numpy.zeros(((cutoff - 10)*len(filenames) + 10,16))
        pix2 = numpy.zeros(((cutoff - 10)*len(filenames2) + 10,16))
        if 'filenames3' in locals():
            pix3 = numpy.zeros(((cutoff - 10)*len(filenames3) + 10,16))
        if 'filenames4' in locals():
            pix4 = numpy.zeros(((cutoff - 10)*len(filenames4) + 10,16))
##    pix = openfits(lightsoff)[:20000,0,0:16]
##    pix2 = openfits(lightson)[:20000,0,0:16]
    #Blacklisted pixels.
##    blacklist = [3,10]
##    blacklist = [6,14]
##    blacklist = [0,15]
##    blacklist = []
    #Sew the frames together.
    for i,f in enumerate(filenames):
        d = openfits(f)
        ramplength = d.shape[0] - 10
        if cutoff > 0:
            ramplength = cutoff - 10 
        for n in range(16):
            if n not in blacklist:
##                print d.shape[0]
                pix[i*ramplength:(i+1)*ramplength,n] = d[5:ramplength + 5,0,n]
    for i,f in enumerate(filenames2):
        d = openfits(f)
        ramplength = d.shape[0] - 10
        if cutoff > 0:
            ramplength = cutoff - 10 
        for n in range(16):
            if n not in blacklist:
                pix2[i*ramplength:(i+1)*ramplength,n] = d[5:ramplength + 5,0,n]
    if 'filenames3' in locals():
        for i,f in enumerate(filenames3):
            d = openfits(f)
            ramplength = d.shape[0] - 10
            if cutoff > 0:
                ramplength = cutoff - 10 
            for n in range(16):
                if n not in blacklist:
                    pix3[i*ramplength:(i+1)*ramplength,n] = d[5:ramplength + 5,0,n]
    if 'filenames4' in locals():
        for i,f in enumerate(filenames4):
            d = openfits(f)
            ramplength = d.shape[0] - 10
            if cutoff > 0:
                ramplength = cutoff - 10 
            for n in range(16):
                if n not in blacklist:
                    pix4[i*ramplength:(i+1)*ramplength,n] = d[5:ramplength + 5,0,n]
    averaged = numpy.zeros((pix.shape[0] - avg,16))
    averaged2 = numpy.zeros((pix2.shape[0] - avg,16))
    if 'pix3' in locals():
        averaged3 = numpy.zeros((pix3.shape[0] - avg,16))
    if 'pix4' in locals():
        averaged4 = numpy.zeros((pix4.shape[0] - avg,16))
    i = 0
##    print pix.shape[0]
    if avg > 1:
        while i + 1 < averaged.shape[0]:
##            averaged[i,:] = mean(pix[avg * i:avg * (i + 1),:])
            averaged[i,:] = mean(pix[i:i + avg,:])
            i += 1
##            averaged[i,:] -= mean(averaged[i,:])
            #print averaged[i]
    else:
        averaged = pix
    i = 0
    if avg > 1:
        while i + 1 < averaged2.shape[0]:
            averaged2[i,:] = mean(pix2[i:i + avg,:])
            i += 1
    else:
        averaged2 = pix2
    if 'pix3' in locals():
        i = 0
        if avg > 1:
            while i + 1 < averaged3.shape[0]:
                averaged3[i,:] = mean(pix3[i:i + avg,:])
                i += 1
        else:
            averaged3 = pix3
    if 'pix4' in locals():
        i = 0
        if avg > 1:
            while i + 1 < averaged4.shape[0]:
                averaged4[i,:] = mean(pix4[i:i + avg,:])
                i += 1
        else:
            averaged4 = pix4
    i = 0
##    cds = numpy.zeros((averaged.shape[0]/2,averaged.shape[1]))
##    cds2 = numpy.zeros((averaged2.shape[0]/2,averaged2.shape[1]))
    cds = numpy.zeros((averaged.shape[0],averaged.shape[1]))
    cds2 = numpy.zeros((averaged2.shape[0],averaged2.shape[1]))
    if 'pix3' in locals():
        cds3 = numpy.zeros((averaged3.shape[0],averaged3.shape[1]))
    if 'pix4' in locals():
        cds4 = numpy.zeros((averaged4.shape[0],averaged4.shape[1]))
    while i + avg * 2 < cds.shape[0]:
##            cds[i,:] = (averaged[i * 2,:] - averaged[i * 2 + 2,:])
##            cds2[i,:] = (averaged2[i * 2,:] - averaged2[i * 2 + 2,:])
##            cds[i + 1,:] = (averaged[i * 2 + 1,:] - averaged[i * 2 + 3,:])
##            cds2[i + 1,:] = (averaged2[i * 2 + 1,:] - averaged2[i * 2 + 3,:])
        for j in range(avg):
            cds[i + j,:] = (averaged[(i + j),:] - averaged[(i + j) + avg,:])
            cds2[i + j,:] = (averaged2[(i + j),:] - averaged2[(i + j) + avg,:])
            if 'pix3' in locals():
                cds3[i + j,:] = (averaged3[(i + j),:] - averaged3[(i + j) + avg,:])
            if 'pix4' in locals():
                cds4[i + j,:] = (averaged4[(i + j),:] - averaged4[(i + j) + avg,:])
        i += avg
##        cds[i] = (averaged[i * 2 ,:] - averaged[i * 2 + 1,:])
##        cds2[i] = (averaged2[i * 2 ,:] - averaged2[i * 2 + 1,:])
##        i += 1
##    j = 0
##    while j < averaged.shape[1]:
##        m,b = linearfit(range(5,averaged.shape[0]),averaged[5:,j])
##        i = 0
##        while i < averaged.shape[0]:
##            #averaged[i,j] -= m * i + b
##            i += 1
##        j += 1
    for i in range(averaged.shape[1]):
        averaged[:,i] -= mean(averaged[:,i])
    if mediansubtraction:
        for c in cds[:]:
            #print c.shape
            d = numpy.array(c)
            d.sort()
            c -= d[d.shape[0]/2]
        for c in cds2[:]:
            d = numpy.array(c)
            d.sort()
            c -= d[d.shape[0]/2]
        if 'pix3' in locals():
            for c in cds3[:]:
                d = numpy.array(c)
                d.sort()
                c -= d[d.shape[0]/2]
        if 'pix4' in locals():
            for c in cds4[:]:
                d = numpy.array(c)
                d.sort()
                c -= d[d.shape[0]/2]
    if onoffplot:
        plt.plot(range(cds.shape[0]/10 - 30),cds[:cds.shape[0]/10 - 30,3] + 100, label = "LED off")
        plt.plot(range(cds2.shape[0]/10 - 30),cds2[:cds2.shape[0]/10 - 30,3], label = "LED on")
        plt.xlabel("Frame #")
        plt.ylabel("counts")
        plt.legend()
        plt.title(title + "LED off/on, pixel 3, AVG" + str(avg))
        plt.show()
    if rampplot:
##    plt.plot(range(cds.shape[0]-11),cds[10:-1,x])
##        plt.plot(range(cds.shape[0]-11),cds[10:-1,x],range(cds.shape[0]-11),cds[10:-1,x+1],range(cds.shape[0]-11),cds[10:-1,x+2])
##        plt.plot(range(cds.shape[0]-11),cds[10:-1,x],range(cds.shape[0]-11),cds[10:-1,x+1],range(cds2.shape[0]-11),cds2[10:-1,x] + 20,range(cds2.shape[0]-11),cds2[10:-1,x+1] + 20, range(cds.shape[0]-11),cds[10:-1,x] - cds[10:-1,x+1] + 200)
##        plt.plot(range(cds.shape[0]-11),cds[10:-1,x],range(cds.shape[0]-11),cds[10:-1,x+1],range(cds.shape[0]-11),cds[10:-1,x+2])
        plt.plot(range(averaged.shape[0]-11),averaged[10:-1,x],label = "0")
        plt.plot(range(averaged.shape[0]-11),averaged[10:-1,x+1],label = "1")
        plt.plot(range(averaged.shape[0]-11),averaged[10:-1,x+2],label = "2")
        plt.plot(range(averaged.shape[0]-11),averaged[10:-1,x+3],label = "3")
        plt.plot(range(averaged.shape[0]-11),averaged[10:-1,x+4],label = "4")
        plt.plot(range(averaged.shape[0]-11),averaged[10:-1,x+5],label = "5")
        plt.plot(range(averaged.shape[0]-11),averaged[10:-1,x+6],label = "6")
        plt.plot(range(averaged.shape[0]-11),averaged[10:-1,x+7],label = "7")
        plt.plot(range(averaged.shape[0]-11),averaged[10:-1,x+8],label = "8")
        plt.plot(range(averaged.shape[0]-11),averaged[10:-1,x+9],label = "9")
        plt.plot(range(averaged.shape[0]-11),averaged[10:-1,x+10],label = "10")
        plt.plot(range(averaged.shape[0]-11),averaged[10:-1,x+11],label = "11")
        plt.plot(range(averaged.shape[0]-11),averaged[10:-1,x+12],label = "12")
        plt.plot(range(averaged.shape[0]-11),averaged[10:-1,x+13],label = "13")
        plt.plot(range(averaged.shape[0]-11),averaged[10:-1,x+14],label = "14")
        plt.plot(range(averaged.shape[0]-11),averaged[10:-1,x+15],label = "15")
        plt.title(title + "LED off, " + "AVG" + str(avg))
        plt.xlabel("Frame #")
        plt.ylabel("counts")
        plt.legend()
        plt.show()
        plt.plot(range(averaged2.shape[0]-11),averaged2[10:-1,x],label = "0")
        plt.plot(range(averaged2.shape[0]-11),averaged2[10:-1,x+1],label = "1")
        plt.plot(range(averaged2.shape[0]-11),averaged2[10:-1,x+2],label = "2")
        plt.plot(range(averaged2.shape[0]-11),averaged2[10:-1,x+3],label = "3")
        plt.plot(range(averaged2.shape[0]-11),averaged2[10:-1,x+4],label = "4")
        plt.plot(range(averaged2.shape[0]-11),averaged2[10:-1,x+5],label = "5")
        plt.plot(range(averaged2.shape[0]-11),averaged2[10:-1,x+6],label = "6")
        plt.plot(range(averaged2.shape[0]-11),averaged2[10:-1,x+7],label = "7")
        plt.plot(range(averaged2.shape[0]-11),averaged2[10:-1,x+8],label = "8")
        plt.plot(range(averaged2.shape[0]-11),averaged2[10:-1,x+9],label = "9")
        plt.plot(range(averaged2.shape[0]-11),averaged2[10:-1,x+10],label = "10")
        plt.plot(range(averaged2.shape[0]-11),averaged2[10:-1,x+11],label = "11")
        plt.plot(range(averaged2.shape[0]-11),averaged2[10:-1,x+12],label = "12")
        plt.plot(range(averaged2.shape[0]-11),averaged2[10:-1,x+13],label = "13")
        plt.plot(range(averaged2.shape[0]-11),averaged2[10:-1,x+14],label = "14")
        plt.plot(range(averaged2.shape[0]-11),averaged2[10:-1,x+15],label = "15")
        plt.title(title + "LED on, " + "AVG" + str(avg))
        plt.xlabel("Frame #")
        plt.ylabel("counts")
        plt.legend()
        plt.show()
    if meanplot:
        m = numpy.zeros(averaged.shape[0])
        for i in range(averaged.shape[0]):
            m[i] = mean(averaged[i,:])
        plt.plot(range(m.shape[0]),m)
        plt.title(title + "LED off mean plot, AVG" + str(avg))
        plt.show()
    if screenforpeaks:
        i = 0
##        cds = cds2
##        cds2 = numpy.array(-cds)
        while i < cds.shape[1]:
            blank = []
            blank2 = []
            blank3 = []
            blank4 = []
            for j, c in enumerate(cds[:,i]):
                if j > 0 and j < cds.shape[0] - 1:
                    if c <= cds[j - 1,i] or c <= cds[j + 1,i]:
                        blank.append(j)
            for j, c in enumerate(cds2[:,i]):
                if j > 0 and j < cds2.shape[0] - 1:
                    if c <= cds2[j - 1,i] or c <= cds2[j + 1,i]:
                        blank2.append(j)
            for b in blank:
                cds[b, i] = -5000
            for b in blank2:
                cds2[b, i] = -5000
            if 'pix3' in locals():
                for j, c in enumerate(cds3[:,i]):
                    if j > 0 and j < cds3.shape[0] - 1:
                        if c <= cds3[j - 1,i] or c <= cds3[j + 1,i]:
                            blank3.append(j)
                for b in blank3:
                    cds3[b, i] = -5000
            if 'pix4' in locals():
                for j, c in enumerate(cds4[:,i]):
                    if j > 0 and j < cds4.shape[0] - 1:
                        if c <= cds4[j - 1,i] or c <= cds4[j + 1,i]:
                            blank4.append(j)
                for b in blank4:
                    cds4[b, i] = -5000
            i += 1
##    print cds2[5,:]
##    nbins = 60
##    histrange = (0,30)
##    cds2 -= mean(cds2[5:-(1000/avg),:].flatten())
##    cdshi = 0
##    cds2hi = 0
##    for x in cds[5:-(1000/avg),:].flatten():
##        if x > 10 and x < 15:
##            cdshi += 1
##    for x in cds2[5:-(1000/avg),:].flatten():
##        if x > 18.5:
##            cds2hi += 1
##    print "lights off between 10 and 15:", cdshi
##    print "lights on > 19:", cds2hi
##    plt.hist([cds[5:-(1000/avg),:].flatten(), cds2[5:-(1000/avg),:].flatten
##              ()],bins=nbins,range = histrange)
    if 'pix4' in locals():
        plt.hist([cds[5:-5,:].flatten(),cds2[5:-5,:].flatten(),cds3[5:-5,:].flatten(),cds4[5:-5,:].flatten()],bins=nbins,range = histrange,label = labels)
        plt.legend()
    elif 'pix3' in locals():
        plt.hist([cds[5:-5,:].flatten(),cds2[5:-5,:].flatten(),cds3[5:-5,:].flatten()],bins=nbins,range = histrange)
    else:
        plt.hist([cds[5:-(1000/avg),:].flatten(),cds2[5:-(1000/avg),:].flatten()],bins=nbins,range = histrange)
    if writecsv:
        with open(csvfile, 'wb') as csvfile:
            filewriter = csv.writer(csvfile, delimiter = ',', quotechar = '|', quoting = csv.QUOTE_MINIMAL)
            hist1 = numpy.histogram([cds[5:-5,:].flatten()], bins = nbins, range = histrange)
            filewriter.writerow(hist1[1])
            filewriter.writerow(hist1[0])
            hist2 = numpy.histogram([cds2[5:-5,:].flatten()], bins = nbins, range = histrange)
            filewriter.writerow(hist2[0])
            if 'pix3' in locals():
                hist3 = numpy.histogram([cds3[5:-5,:].flatten()], bins = nbins, range = histrange)
                filewriter.writerow(hist3[0])
            if 'pix4' in locals():
                hist4 = numpy.histogram([cds4[5:-5,:].flatten()], bins = nbins, range = histrange)
                filewriter.writerow(hist4[0])
    plt.xlim(histrange)
    plt.title(title + "avg = "+str(avg)+", bins = "+str(nbins)+", median subtraction "+str(mediansubtraction)+", screened "+str(screenforpeaks))
    plt.show()
    #Let's subtract one from the other.
    hist1 = numpy.histogram([cds[5:-(1000/avg),:].flatten()], bins = nbins, range = histrange)
    hist2 = numpy.histogram([cds2[5:-(1000/avg),:].flatten()], bins = nbins, range = histrange)
    hist1mean = 0
    if fullstatistics:
        print "LED off"
        print "NCOUNT\tADUs\tPRODUCT\tCUMUL\tCUMUL/16"
        cumulative = 0
        for h,b in map(None,hist1[0],hist1[1][:-1]):
            if b >= 5:
                hist1mean += float(b) * float(h)
                stdout.write(str(h)+"\t"+str(b)+";\t"+str(int(h*b))+"\t")
                cumulative += int(h * b)
                if (b + 0.5) % 2.5 < 0.1:
                    print str(cumulative) + "\t" + str(cumulative/32)
                    cumulative = 0
                stdout.write("\n")
        print "Total:",hist1mean,"/16 =",hist1mean/16
        print "\n"
        print "LED on"
        print "NCOUNT\tADUs\tPRODUCT\tCUMUL\tCUMUL/16"
        hist2mean = 0
        cumulative = 0
        for h,b in map(None,hist2[0],hist2[1][:-1]):
            if b >= 5:
                hist2mean += b * h
                stdout.write(str(h)+"\t"+str(b)+";\t"+str(int(h*b))+"\t")
                cumulative += int(h * b)
                if (b + 0.5) % 2.5 < 0.1:
                    print str(cumulative) + "\t" + str(cumulative/16)
                    cumulative = 0
                stdout.write("\n")
        print "Total:",hist2mean,"/16 =",hist2mean/16
    if perpixelstatistics:
        for n in range(cds.shape[1]):
            print "Pixel",n
            histoff = numpy.histogram([cds[5:-(1000/avg),n].flatten()], bins = nbins, range = histrange)
            histon = numpy.histogram([cds2[5:-(1000/avg),n].flatten()], bins = nbins, range = histrange)
            if n == 6:
                interval = averaged.shape[0] / 9
                for i in range(9):
                    plt.plot(range(interval),averaged[i * interval:((i + 1) * interval),x+n])
                plt.show()
                for i in range(9):
                    plt.plot(range(interval),averaged2[i * interval:((i + 1) * interval),x+n])
                plt.show()
            cutoffs = [7.5, 8.5, 9.5, 10.5, 11.5, 12.5]
            runningtotalon = 0
            runningtotaloff = 0
            for h,b in map(None,histon[0],histon[1][:-1]):
                runningtotalon += b * h
            for h,b in map(None,histoff[0],histoff[1][:-1]):
                runningtotaloff += b * h
            print "Total: off",runningtotaloff,"on",runningtotalon
            for c in cutoffs:
                cutofftotalon = 0
                cutofftotaloff = 0
                for h,b in map(None,histon[0],histon[1][:-1]):
                    if b >= c:
                        cutofftotalon += b * h
                for h,b in map(None,histoff[0],histoff[1][:-1]):
                    if b >= c:
                        cutofftotaloff += b * h
                print "Cutoff",c,": off",cutofftotaloff,"on",cutofftotalon,"on/off",cutofftotalon/cutofftotaloff
    if ratioplot:
        #print hist1[0] / hist2[0]
        ax = plt.subplot(111)
        ax.bar(hist2[1][:-1], hist2[0] / hist1[0].astype(float), width = 0.5)
    ##    ax.bar(hist2[1][:-1], hist2[0] - hist1[0], width = 0.5)
        plt.title(title + " ratio plot" + "AVG" + str(avg))
        plt.show()

def fitPC():
    histrange = (-250,250)
    binspacing = 1
    y = 0
    nbins = histrange[1] - histrange[0] / binspacing
    #Take the off and on histograms, fit the off data to the low end of the on curve with a constant coefficient,
    #and subtract the fit data to get a gain profile.
    title = "M02885-12, VDD = 4V, COMMON = -11V  "
    #Lights off.
    filenames = ["140408_170409f.fits","140408_170434f.fits","140408_170437f.fits","140408_170440f.fits","140408_170443f.fits",\
                 "140408_170447f.fits","140408_170450f.fits","140408_170454f.fits","140408_170457f.fits","140408_170500f.fits"]
    #Lights 0.8V.
    filenames2 =["140408_170512f.fits","140408_170541f.fits","140408_170544f.fits","140408_170547f.fits","140408_170551f.fits",\
                 "140408_170554f.fits","140408_170557f.fits","140408_170600f.fits","140408_170603f.fits","140408_170606f.fits"]
    blacklist = [0, 4, 11, 12, 13, 14, 15]
    cutoff = 5000
    #Sew together the datasets
    LEDoff = openfits(filenames[0])[:cutoff,:,:]
    for f in filenames[1:]:
        LEDoff = numpy.append(LEDoff, openfits(f)[:cutoff,:,:], axis = 0)
    LEDon = openfits(filenames2[0])[:cutoff,:,:]
    for f in filenames2[1:]:
        LEDon = numpy.append(LEDon, openfits(f)[:cutoff,:,:], axis = 0)
    #Run the CDS stuff.
    CDSoff = numpy.zeros((LEDoff.shape[0]/2,LEDoff.shape[1],LEDoff.shape[2]))
    CDSoff[0,:,:] = LEDoff[0,:,:] - LEDoff[1,:,:]
    for i in range(2,LEDoff.shape[0],2):
        CDSoff[i/2,:,:] = LEDoff[i,:,:] - LEDoff[i+1,:,:]
    CDSon = numpy.zeros((LEDon.shape[0]/2,LEDon.shape[1],LEDon.shape[2]))
    CDSon[0,:,:] = LEDon[0,:,:] - LEDon[1,:,:]
    for i in range(2,LEDon.shape[0],2):
        CDSon[i/2,:,:] = LEDon[i,:,:] - LEDon[i+1,:,:]
    #Generate histograms.
    histoff = numpy.histogram([CDSoff[:,:,:].flatten()], bins = nbins, range = histrange)
    histon = numpy.histogram([CDSon[:,:,:].flatten()], bins = nbins, range = histrange)
    #Fit the RN/dark section of the off histogram to the on one to get a coefficient.
    coef = 0.8
    delta = 0.1
    fitindices = []
    for x in range(-10,-5):
        fitindices.append(numpy.where(histoff[1] == x)[0][0])
    while delta > 0.00001:
        rsquared = 0
        rsquaredup = 0
        rsquareddown = 0
        for i in fitindices:
            rsquared += (histon[1][i] - (histoff[1][i] * coef)) ** 2
            rsquaredup += (histon[1][i] - (histoff[1][i] * (coef + delta))) ** 2
            rsquareddown += (histon[1][i] - (histoff[1][i] * (coef - delta))) ** 2
        if rsquaredup < rsquared:
            coef += delta
        elif rsquareddown < rsquared:
            coef -= delta
        else:
            delta /= 10.
    print coef
    ax = plt.subplot(111)
    ax.bar(histoff[1][:-1],histoff[0], 0.5, color = 'blue')
    ax.bar(histon[1][:-1] + 0.5,histon[0], 0.5, color = 'red')
##    ax.bar(histoff[1][:-1],histon[0] - coef * histoff[0])
    plt.show()
    

def checkRamp(filename, x = 0):
    data = openfits(filename)[:,:,:]
    plt.plot(range(data.shape[0]-11),data[10:-1,x],label = "1")
    plt.plot(range(data.shape[0]-11),data[10:-1,x+1],label = "2")
##    plt.plot(range(data.shape[0]-11),data[10:-1,x+2],label = "3")
##    plt.plot(range(data.shape[0]-11),data[10:-1,x+3],label = "4")
##    plt.plot(range(data.shape[0]-11),data[10:-1,x+4],label = "5")
##    plt.plot(range(data.shape[0]-11),data[10:-1,x+5],label = "6")
##    plt.plot(range(data.shape[0]-11),data[10:-1,x+6],label = "7")
##    plt.plot(range(data.shape[0]-11),data[10:-1,x+7],label = "8")
##    plt.plot(range(data.shape[0]-11),data[10:-1,x+8],label = "9")
##    plt.plot(range(data.shape[0]-11),data[10:-1,x+9],label = "10")
##    plt.plot(range(data.shape[0]-11),data[10:-1,x+10],label = "11")
##    plt.plot(range(data.shape[0]-11),data[10:-1,x+11],label = "12")
##    plt.plot(range(data.shape[0]-11),data[10:-1,x+12],label = "13")
##    plt.plot(range(data.shape[0]-11),data[10:-1,x+13],label = "14")
##    plt.plot(range(data.shape[0]-11),data[10:-1,x+14],label = "15")
##    plt.plot(range(data.shape[0]-11),data[10:-1,x+15],label = "16")
    plt.title("LED off")
    plt.xlabel("Frame #")
    plt.ylabel("counts")
    plt.show()
##    m = numpy.zeros(data.shape[0])
##    med = numpy.zeros(data.shape[0])
##    for i in range(data.shape[0]):
##        m[i] = data[i,:,:].mean()
##        med[i] = numpy.sort(data[i,:,:].flatten())[data.shape[1]/2]
##    plt.plot(range(m.shape[0]),m)
##    plt.title("LED off mean plot")
##    plt.show()
##    plt.plot(range(med.shape[0]),med)
##    plt.title("Pulsed LED median pixels")
##    plt.show()

def riseTime(filename):
    data = openfits(filename)
    newdata = numpy.zeros([data.shape[0]/10,data.shape[1],data.shape[2]])
    for i in range(newdata.shape[0]):
        for j in range(10):
            newdata[i,:,:] = data[i + (j * 10),:,:]
    m = numpy.zeros((newdata.shape[0]))
    for i in range(newdata.shape[0]):
        m[i] = newdata[i,:,:].mean()
    plt.plot(range(m.shape[0]),m)
    plt.show()
    cdsdata = cdsCube(newdata)
    m = numpy.zeros((cdsdata.shape[0]))
    for i in range(cdsdata.shape[0]):
        m[i] = cdsdata[i,:,:].mean()
    plt.plot(range(m.shape[0]),m)
    plt.show()
    
def freqResponseByTemp():
    navg = 2
    T60K = cdsCube(avgCube(openfits("140603_153626.fits"), avg = navg))
    T85K = cdsCube(avgCube(openfits("140604_093425.fits"), avg = navg)) * (65. / 110.)
    T60Kmeans = numpy.zeros(T60K.shape[0])
    T85Kmeans = numpy.zeros(T85K.shape[0])
    for i in range(T60Kmeans.shape[0]):
        T60Kmeans[i] = T60K[i,:,:].mean()
    for i in range(T85Kmeans.shape[0]):
        T85Kmeans[i] = T85K[i,:,:].mean()
    print T60Kmeans.shape
    plt.plot(range(4,(T60Kmeans.shape[0] + 1) * 4,4),T60Kmeans, label = "60K")
    plt.plot(range(4,(T85Kmeans.shape[0] + 1) * 4,4),T85Kmeans, label = "85K")
    plt.xlabel("ms")
    plt.legend()
    plt.show()

def freqResponseByTemp1Nov():
    navg = 1
    T60K = cdsCube(avgCube(openfits("140603_153626.fits"), avg = navg))
    T85K = cdsCube(avgCube(openfits("140604_093425.fits"), avg = navg)) * (65. / 110.)
    T60Kmeans = numpy.zeros(T60K.shape[0])
    T85Kmeans = numpy.zeros(T85K.shape[0])
    for i in range(T60Kmeans.shape[0]):
        T60Kmeans[i] = T60K[i,:,:].mean()
    for i in range(T85Kmeans.shape[0]):
        T85Kmeans[i] = T85K[i,:,:].mean()
    print T60Kmeans.shape
    plt.plot(range(4,(T60Kmeans.shape[0] + 1) * 4,4),T60Kmeans, label = "60K")
    plt.plot(range(4,(T85Kmeans.shape[0] + 1) * 4,4),T85Kmeans, label = "85K")
    plt.xlabel("ms")
    plt.legend()
    plt.show()
    
def pixelCosmetics():
    t = 25
    n0 = 2
    mark5 = 2.85 * subtractnth(openfits("140602_153209.fits"),n=n0)[t + n0,:,:]/(-t)
    mark3 = 2.85 * subtractnth(openfits("d:/SAPHIRA/140408_152433f.fits"),n=n0)[t + n0,:,:]/(-t)
    mark2 = 2.85 * subtractnth(openfits("d:/SAPHIRA/140313_140002f.fits"),n=n0)[t + n0,:,:]/(-t)
##    n, bins, patches = plt.hist(mark2.flatten(),bins = 30,label="M02775-10")
##    plt.hist(mark3.flatten(),bins = bins,label="M02815-12")
##    plt.hist(mark5.flatten(),bins = bins,label="M04055-06")
    plt.hist([mark2.flatten(),mark3.flatten(),mark5.flatten()], bins = 30, label = ["M02775-10 (T = 70K)","M02815-12 (T = 70K)","M04055-06 (T = 60K)","M06495-27 (T = 55K)"])
    plt.legend()
    plt.title("Comparison of pixel cosmetics for SAPHIRA detectors (Vbias = 11.5V)")
    plt.ylabel("#pixels")
    plt.xlabel("e-/s")
    plt.show()

def detectorCosmetics(mark = 2):
    t = 25
    n0 = 2
    if mark == 2:
        title = "M02775-10 (T = 70K)"
        V05  = 2.85 * subtractnth(openfits("d:/SAPHIRA/140313_142455f.fits"),n=n0)[t + n0,:,:]/(-t)
        V35  = 2.85 * subtractnth(openfits("d:/SAPHIRA/140313_141846f.fits"),n=n0)[t + n0,:,:]/(-t)
        V75  = 2.85 * subtractnth(openfits("d:/SAPHIRA/140313_182835f.fits"),n=n0)[t + n0,:,:]/(-t)
        V115 = 2.85 * subtractnth(openfits("d:/SAPHIRA/140313_140002f.fits"),n=n0)[t + n0,:,:]/(-t)
    elif mark == 3:
        title = "M02815-12 (T = 70K)"
        V05 = 2.85 * subtractnth(openfits("d:/SAPHIRA/140408_154959f.fits"),n=n0)[t + n0,:,:]/(-t)
        V35 = 2.85 * subtractnth(openfits("d:/SAPHIRA/140408_153711f.fits"),n=n0)[t + n0,:,:]/(-t)
        V75 = 2.85 * subtractnth(openfits("d:/SAPHIRA/140408_153055f.fits"),n=n0)[t + n0,:,:]/(-t)
        V115 = 2.85 * subtractnth(openfits("d:/SAPHIRA/140408_152433f.fits"),n=n0)[t + n0,:,:]/(-t)
    elif mark == 5:
        title = "M04055-06 (T = 60K)"
        V05 = 2.85 * subtractnth(openfits("140602_142251.fits"),n=n0)[t + n0,:,:]/(-t)
        V35 = 2.85 * subtractnth(openfits("140602_142621.fits"),n=n0)[t + n0,:,:]/(-t)
        V75 = 2.85 * subtractnth(openfits("140602_152216.fits"),n=n0)[t + n0,:,:]/(-t)
        V115 = 2.85 * subtractnth(openfits("140602_153209.fits"),n=n0)[t + n0,:,:]/(-t)
    plt.hist([V05.flatten(),V35.flatten(),V75.flatten(),V115.flatten()], bins = 20, label = ["0.5Vbias","3.5Vbias","7.5Vbias","11.5Vbias"])
    plt.title(title)
    plt.legend()
    plt.ylabel("#pixels")
    plt.xlabel("e-/s")
    plt.show()

def cycletime(cols,rows):
    Tpix = 3.76
    return (Tpix * (cols / 32) * rows) + (rows * 1.51) + 9.2

def findhighest(indata, length = 20):
    #First, copy the array.
    data = numpy.array(indata)
    values = numpy.sort(data, axis = None)
    i = 0
    ylist = []
    xlist = []
    while i < length:
        i += 1
        #print i, values[-i]
        #For some reason it throws up some nans every once in a while.
        if not math.isnan(values[-i]):
            #Find the current value.
            a = numpy.where(data + 0.00001 >= values[-i])
            #Convert this to coordinates we can use.
            y = a[0][0]
            x = a[1][0]
            #Wipe it from the array so we don't find it again.
            data[y,x] = 0
            #Add the coordinates to the list.
            ylist.append(y)
            xlist.append(x)
    return ylist, xlist

def centroid10pix(data):
    #Does the actual centroid from mean location of 10 brightest pixels.
    ylist, xlist = findhighest(data, length = 10)
    y = mean(ylist)
    x = mean(xlist)
    return y,x

def halve(inArray):
    return inArray[1:inArray.shape[0]/2]

def TTplot():
    #Takes a given ramp, performs a CDS on it, and then computes
    #the centroid via the 10 brightest pixels. Plots the centroid
    #across cds frames.
    frameTime = .0005882 #s
    filenames = []
    for f in listdir("./")[:]:
        if len(f) == 18:
            filenames.append(f)
    #Set up our centroid lists.
    cy = []
    cx = []
    d = cdsCube(openfits(filenames[0]))
    for n in range(152,d.shape[0]):
        try:
##            y, x = centroid10pix(d[n,:,:])
            y, x = centroid(d[n,:,:])
            cy.append(y)
            cx.append(x)
        except ZeroDivisionError:
            #Skip it.
            print "Skipping frame#",n
##    fy, Py_den = signal.periodogram(cy,(1 / frameTime))
##    fx, Px_den = signal.periodogram(cx,(1 / frameTime))
    spy = abs(halve(fft.fft(cy)))**2
    spx = abs(halve(fft.fft(cx)))**2
    freqy = halve(fft.fftfreq(len(cy), d = frameTime))
    freqx = halve(fft.fftfreq(len(cx), d = frameTime))
    
##    plt.plot(freqy, spy.real)
##    plt.plot(freqx, spx.real)
##    plt.xlabel('Freq (hz)')
##    plt.show()
    for f in filenames[1:30]:
        print f
        #Open and cds data in one line.
        d = cdsCube(openfits(f))
        cy = []
        cx = []
        if d.shape[0] > 200:
            for n in range(152,d.shape[0]):
                try:
##                    y, x = centroid10pix(d[n,:,:])
                    y, x = centroid(d[n,:,:])
                    cy.append(y)
                    cx.append(x)
                except ZeroDivisionError:
                    #Skip it.
                    print "Skipping frame#",n
    ##        fy1, Py_den1 = signal.periodogram(cy,(1 / frameTime))
    ##        fx1, Px_den1 = signal.periodogram(cx,(1 / frameTime))
            spy1 = abs(halve(fft.fft(cy)))**2
            spx1 = abs(halve(fft.fft(cx)))**2
            freqy1 = halve(fft.fftfreq(len(cy), d = frameTime))
            freqx1 = halve(fft.fftfreq(len(cx), d = frameTime))
            if len(freqy1) == len(freqy):
                for i, y in enumerate(freqy1):
                    j = 0
                    #Find the closest location.
                    while y > freqy[j] and j < len(freqy):
                        j += 1
                    #Make sure we haven't run off the end.
                    if j < len(freqy):
                        #print i,j
                        spy[j] += spy1[i]
                for i, x in enumerate(freqx1):
                    j = 0
                    #Find the closest location.
                    while x > freqx[j] and j < len(freqx):
                        j += 1
                    #Make sure we haven't run off the end.
                    if j < len(freqx):
                        spx[j] += spx1[i]
            else:
                print "failed"
        else:
            print "failed"
##        for i, y in enumerate(fy1):
##            j = 0
##            #Find the closest location.
##            while y > fy[j] and j < len(fy):
##                j += 1
##            #Make sure we haven't run off the end.
##            if j < len(fy):
##                print i,j
####                Py_den[j] += Py_den1[i]
##        for i, x in enumerate(fx1):
##            j = 0
##            #Find the closest location.
##            while x > fx[j] and j < len(fx):
##                j += 1
##            #Make sure we haven't run off the end.
##            if j < len(fx):
##                Px_den[j] += Px_den1[i]
    
        
##    plt.plot(range(152,d.shape[0]),cy,range(152,d.shape[0]),cx)
##    plt.show()
    
##    fy, Py_den = signal.periodogram(cy,(1 / frameTime))
##    fx, Px_den = signal.periodogram(cx,(1 / frameTime))
##    plt.plot(fy, Py_den)
##    plt.plot(fx, Px_den)
            
##    plt.plot(freqy, spy, label = "Y")
##    plt.plot(freqx, spx, label = "X")
    plt.loglog(freqy, spy, label = "Y")
    plt.loglog(freqx, spx, label = "X")
    plt.xlabel('Freq (hz)')
    plt.legend()
    plt.show()

def getMask(image, factor = 5):
    mask = numpy.ones(image.shape)
    for y in range(1, image.shape[0] - 1):
        for x in range(1, image.shape[1] - 1):
            i = image[y,x]
            up = image[y-1,x] * factor
            down = image[y+1,x] * factor
            left = image[y,x-1] * factor
            right = image[y,x+1] * factor
            if i > up and i > down and i > left and i > right:
                mask[y,x] = 0
    return mask

def background(indata, zeroScreen = True, getSigma = False):
    #Finds the background level of an image.
    #First, copy the array.
    data = numpy.array(indata)
    l = data.shape[0] * data.shape[1]
    sortedimage = numpy.sort(data, axis = None)
    #Find where the zeros stop, so we don't count them.
    #print numpy.where(sortedimage == 0)
    if zeroScreen:
        lastzero = numpy.where(sortedimage == 0)[-1][-1]
    else:
        lastzero = 0
    l = len(sortedimage[lastzero:])
    #Use the middle 2000 pixels to find the average background.
##    bkg = mean(sortedimage[l/2 + lastzero - 1000:l/2 + lastzero + 1000])
    bkg = numpy.mean(sortedimage[l/2 + lastzero - l/10:l/2 + lastzero + l/10])
    if not getSigma:
        return bkg
    else:
        #Find the standard deviation by taking a random sample of 1000 pixels.
        sample = []
        for dummy in range(3000):
            sample.append(sortedimage[lastzero + int(random.random() * len(sortedimage[lastzero:]))])
        sortedsample = numpy.sort(sample, axis = None)
        sd = stddev(sortedsample[100:-100]) #Screen outliers.
        return bkg, sd

def findFWHM(image, y, x):
    #Just finds the rough FWHM of the star at the given location in the given image.
    count = []
    i = 0
    while image[y + i,x] > (image[y,x] / 2):
        i += 1
    count.append(i)
    i = 0
    while image[y - i,x] > (image[y,x] / 2):
        i += 1
    count.append(i)
    i = 0
    while image[y,x + i] > (image[y,x] / 2):
        i += 1
    count.append(i)
    i = 0
    while image[y,x - i] > (image[y,x] / 2):
        i += 1
    count.append(i)
    FWHM = mean(count)
    return FWHM

def enhance(data, factor = 2, debug = False):
    if debug:
        print "Enhancing..."
        t = time()
    #Ups the resolution on the 2D input image by factor in each dimension.
    upres = numpy.zeros((data.shape[0] * factor,data.shape[1] * factor))
    for y in range(upres.shape[0]):
        for x in range(upres.shape[1]):
            upres[y,x] = data[y/factor,x/factor]
    if debug:
        print "Image enhanced, time elapsed", time() - t, "seconds."
    return upres

def getLucky():
    startFrame = 150 #150 for 64 x 64, 10 for full frame
    percentile = 90
##    nFiles = -1
    nFiles = 10
    rPSF = 2 #Was 2.
    nFrames = 425 #45 for 100 frames starting
    nSkip = 100
    #Default location of our guide star
    #M3
    gY = 36
    gX = 256
    #24 LMi
    gY = 32
    gX = 32
    gR = 20
    gY *= 2
    gX *= 2
    gR *= 2
    #Make a composite image from selected lucky images.
    frameTime = .0005882 #s
    filenames = []
    for f in listdir("./")[:]:
        if len(f) == 18:
            filenames.append(f)
    brightest = []
    #Make the mask.
    mask = enhance(getMask((-1 * cdsCube(openfits(filenames[0])[startFrame:]))[0,:,:]))
    for f in filenames[:nFiles]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category = UserWarning)
            print f
            try:
                data = -1 * cdsCube(openfits(f)[startFrame:])
        ##        savefits("testy.fits",data)
                if data.shape[0] == nFrames:
                    for i in range(nSkip, data.shape[0], nSkip):
            ##            cy, cx = centroid(data[i,:,:])
            ##            print cy, cx
                        d = enhance(data[i,:,:])
            ##            d = ma.masked_array(image, mask = mask)
                        for y in range(d.shape[0]):
                            for x in range(d.shape[1]):
                                d[y,x] = mask[y,x] * d[y,x]
                        d = d[5:-5,5:-5]
                        d -= background(d, zeroScreen = False)
            ##            s = findhighest(d,length = 1)
            ##            y = s[0][0]
            ##            x = s[1][0]
                        y,x = centroid(d[gY - gR - 5 : gY + gR - 4, gX - gR - 5 : gX + gR - 4])
                        y += gY - gR - 5
                        x += gX - gR - 5
                        #Let's do a small area around the brightest.
                        brightest.append(d[y - rPSF:y + rPSF + 1,x - rPSF:x + rPSF + 1].sum())
            except IndexError:
                print "Only one frame, skipping."
    brightest.sort()
    print len(brightest)
    threshhold =  brightest[int(round((len(brightest) * percentile) / 100))]
    print "Threshhold value:",threshhold
    d = openfits(filenames[0])
    full = numpy.zeros([d.shape[1]*4,d.shape[2]*4])
    lucky = numpy.zeros([d.shape[1]*4,d.shape[2]*4])
    y0 = full.shape[0] / 2
    x0 = full.shape[1] / 2
    passed = 0
    total = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category = UserWarning)
        for f in filenames[:nFiles]:
            print f
            try:
                data = -1 * cdsCube(openfits(f)[startFrame:])
                if data.shape[0] == nFrames:
                    for i in range(data.shape[0]):
                        d = enhance(data[i,:,:])
                        #d = data[i,5:-5,5:-5] * (data[i,5:-5,5:-5] & mask)
                        for y in range(d.shape[0]):
                            for x in range(d.shape[1]):
                                d[y,x] = mask[y,x] * d[y,x]
                        d = d[5:-5,5:-5]
                        d -= background(d, zeroScreen = False)
        ##                s = findhighest(d,length = 1)
        ##                y, x = centroid(d)
                        y,x = centroid(d[gY - gR - 5 : gY + gR - 4 , gX - gR - 5 : gX + gR - 4])
                        y += gY - gR - 5
                        x += gX - gR - 5
                        #Does it meet our threshold?
                        if d[y - rPSF:y + rPSF + 1,x - rPSF:x + rPSF + 1].sum() >= threshhold:
                            passed += 1
                            #If so, we want to add it to the lucky image. Find the centroid.
                            lucky[y0 - y:y0 - y + d.shape[0],x0 - x:x0 - x + d.shape[1]] += d
                        full[y0 - y:y0 - y + d.shape[0],x0 - x:x0 - x + d.shape[1]] += d
                        total += 1
            except IndexError:
                print "Only one frame, skipping."
            except ZeroDivisionError:
                print "No valid centroid measurements, skipping."
    #Do a quick FWHM measurement on both images.
    print "Lucky:",findFWHM(lucky,y0,x0),"pix"
    print "Full:",findFWHM(full,y0,x0),"pix"
    savefits("lucky.fits",lucky[lucky.shape[0] - gY - d.shape[0]:lucky.shape[0] - gY,lucky.shape[1] - gX - d.shape[1]:lucky.shape[1] - gX])
    savefits("full.fits",full[full.shape[0] - gY - d.shape[0]:full.shape[0] - gY,full.shape[1] - gX - d.shape[1]:full.shape[1] - gX])
    print "Selected",passed,"of",total,",",100 * (1 - passed/(float(total))),"% excluded"

def comparativeQE():
    data = numpy.array([["140413_113243f.fits", "140413_113317f.fits", 35000, "M02775-10", 0.01],\
                        ["140331_102726f.fits", "140331_102754f.fits", 32000, "M02815-12", 0.1],\
                        ["140601_110524.fits", "140601_110551.fits", 20000, "M04055-06", 0.11],\
                        ["141029_164912.fits", "141029_165014.fits", 20000, "M04055-39", 0.01]])
    window = [[32,-32],[32,-32]]
    for i in range(data.shape[0]):
        on = openfits(data[i,1])
        off = openfits(data[i,0])
        onsub = subtractnth(on,n=5)
        offsub = subtractnth(off,n=5)
        n = 1
        #Check for saturation.
        cutoff = data[i,2]
        while (mean(mean(on[-n,window[0][0]:window[0][1],window[1][0]:window[1][1]])) < cutoff) and (n < on.shape[0] - 3):
            n += 1
        if n >= on.shape[0] - 1:
            print "Image saturates immediately."
        else:
            #n = on.shape[0] * 4 / 5.
            n = onsub.shape[0] - 20
            print "Using frame #",on.shape[0]-n
            diff = offsub[-n,window[0][0]:window[0][1],window[1][0]:window[1][1]] - onsub[-n,window[0][0]:window[0][1],window[1][0]:window[1][1]]
            m = numpy.mean(diff)
            print "Difference mean:", m
            t = (onsub.shape[0] - n) * float(data[i,4])
            print "Time:", t
            gain = m/t
            print on.shape[0] - n
            print onsub.shape[0] - n
            print data[i,3], "measured gain:", gain
            plt.plot(range(offsub[:-n,window[0][0]:window[0][1],window[1][0]:window[1][1]].shape[0]),\
                     numpy.mean(offsub[:-n,window[0][0]:window[0][1],window[1][0]:window[1][1]], axis = (1,2)),\
                     label = data[i,3] + " off")
            plt.plot(range(onsub[:-n,window[0][0]:window[0][1],window[1][0]:window[1][1]].shape[0]),\
                     numpy.mean(onsub[:-n,window[0][0]:window[0][1],window[1][0]:window[1][1]], axis = (1,2)),\
                     label = data[i,3] + " on")
    plt.xlabel("frame #")
    plt.ylabel("counts")
    plt.legend()
    plt.show()

def freqResponse(filename = "160722_3.1umFR1.9V-17-0.fits", navg = 1):
##    T60K = cdsCube(avgCube(openfits("141031_132340.fits"), avg = navg))
##    T60K = cdsCube(avgCube(openfits("150928_111239.fits"), avg = navg))
    T60K = cdsCube(avgCube(openfits(filename), avg = navg))
    T60Kmeans = numpy.zeros(T60K.shape[0])
    for i in range(T60Kmeans.shape[0]):
        T60Kmeans[i] = numpy.median(T60K[i,64:-64,92:-92])
    print T60Kmeans.shape
    plt.plot(numpy.array(range(0,T60Kmeans.shape[0] * navg,navg)) * 5.0,T60Kmeans, label = "60K")
    plt.xlabel("ms")
    plt.legend()
    plt.show()

def SNRComparison(Nmax):
    dcs = []
    lineFitting = []
    fowler = []
    for N in range(2,Nmax):
        dcs.append(1)
        lineFitting.append(sqrt((5./6.) * (N**2 - 1) / (N**2 + 1)))
        dtTint = 1. / (2. * N)
        print (1 + (dtTint/3.) * (1 / (N / 2) - 4 * (N / 2)))
        fowler.append((1 - (N / 2) * dtTint) / sqrt(1 + (dtTint/3.) * (1. / (N / 2) - 4 * (N / 2))))
    plt.plot(range(2,Nmax),dcs)
    plt.plot(range(2,Nmax),lineFitting)
    plt.plot(range(2,Nmax),fowler)
    plt.show()

def makeDarkMap(filename, timing = 1, start = 5, end = -1, n = 20, chargeGain = 2.89):
    d = openfits(filename)
    print filename[:-5]
    e = numpy.zeros(d.shape[1:])
    if end < 0:
        end += d.shape[0]
    if n * 2 + start > end:
        return "Parameters not long enough for",n,"averages, quitting."
    medians = []
    print "Note: Medians are from top-center subarray. This change was made 24 Apr 2017."
    for i in range(start,end):
        medians.append(numpy.median(d[i,-32:,64:-64]))
    m, b = linearfit(range(start,end), medians)
    plt.plot(range(start,end), medians)
    plt.plot([start,end],[start * m + b, end * m + b], 'r--')
    plt.show()
    dummy = raw_input("Good?")
    for i in range(0,n):
        e += d[i + start,:,:]
        e -= d[i - n + end,:,:]
    e /= n
##    e /= d.shape[0] - (start + n) #Added 20 Jan 2015.
    e /= end - (start + n) #Modified 29 Feb 2016.
    e /= timing #Added 7 Apr 2015.
    print "Traditional median dark current:", numpy.median(e) * chargeGain, "e-/s"
    print "Line fit dark current:", -m / timing * chargeGain, "e-/s"
    print "Note: dark map is in ADU/s"
    savefits(filename[:-5] + "darkMap.fits",e)

def plotDarks():
    VDD = [3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.5, 5.0, 5.5]
    filenames = ["141125_140707darkMap.fits",
                 "141125_135131darkMap.fits",
                 "141125_142732darkMap.fits",
                 "141124_135157darkMap.fits",
                 "141125_141520darkMap.fits",
                 "141124_133620darkMap.fits",
                 "141124_145724darkMap.fits",
                 "141124_142709darkMap.fits",
                 "141124_145029darkMap.fits"]
    dark = []
    for f in filenames:
        d = openfits(f)
        med = numpy.median(d)
        dark.append((med * 2.9 / 275))
    plt.plot(VDD,dark,'o-')
    plt.xlabel("VDD (V)")
    plt.ylabel("median pixel (e-/read)")
    plt.show()

def plotDarksvsTime():
##    times = [1,0.75,0.5,0.3,0.2,0.1,0.01]
##    filenames = ["141126_082926darkMap.fits", #3.5V
##                 "141126_090259darkMap.fits",
##                 "141126_084042darkMap.fits",
##                 "141126_085047darkMap.fits",
##                 "141126_095035darkMap.fits",
##                 "141126_084614darkMap.fits",
##                 "141126_084421darkMap.fits"]
    times = [1, 0.5, 0.3, 0.2]
    filenames = ["141126_100535darkMap.fits", #5.5V
                 "141126_101218darkMap.fits",
                 "141126_101505darkMap.fits",
                 "141126_101652darkMap.fits"]
##    filenames = ["141126_132316darkMap.fits", #5.0V
##                 "141126_132948darkMap.fits",
##                 "141126_133313darkMap.fits",
##                 "141126_133536darkMap.fits"]
##    filenames = ["141126_134247darkMap.fits", #4.5V
##                 "141126_134809darkMap.fits",
##                 "141126_135053darkMap.fits",
##                 "141126_135256darkMap.fits"]
    dark = []
    for f, t in map(None, filenames, times):
        d = openfits(f)
        med = numpy.median(d)
        dark.append((med * 2.9 / (275 * t)))
        print t,"\t", dark[-1]
    m,b = linearfit(times, dark)
    #print m,b
    plt.plot(times,dark,'o-')
    #plt.plot(times,[m * x + b for x in times],'-')
    plt.xlabel("sampling (s)")
    plt.ylabel("median pixel (e-/s)")
    plt.title("Fit results: glow " + str(m)[:6] + "e-/read")
    plt.show()

def histogramDarks():
    filenames = ["141125_140707darkMap.fits",
                 "141124_133620darkMap.fits",
                 "141124_145724darkMap.fits",
                 "141124_142709darkMap.fits"]
    s = 221
    titles = ["VDD = 3.5V",
             "VDD = 4.0V",
             "VDD = 4.5V",
             "VDD = 5.0V"]
    for f,t in map(None, filenames, titles):
        plt.subplot(s)
        s += 1
        d = openfits(f)
        dSort = numpy.sort(d, axis = None)
        print dSort
        r = dSort.shape[0]
        minimum = dSort[int(r * 1. / 10.)]
        maximum = dSort[int(r * 9. / 10.)]
        plt.hist(d.flatten() * 2.9 / 2.75, bins = 20, range = [minimum,maximum])
        plt.xlabel("e-/read")
        plt.ylabel("N")
        plt.title(t)
    plt.show()

def diffFits():
##    data1 = ["141126_152708darkMap.fits",
##             "141126_161412darkMap.fits",
##             "141126_162659darkMap.fits",
##             "141126_163929darkMap.fits"]
##    data2 = ["141126_160605darkMap.fits",
##             "141126_161927darkMap.fits",
##             "141126_163243darkMap.fits",
##             "141126_164532darkMap.fits"]
##    data1 = ["141203_124735.fits",
##             "141203_131225.fits",
##             "141203_164831.fits",
##             "141203_165453.fits",
##             "141203_170108.fits",
##             "141203_170654.fits",
##             "141203_171332.fits",
##             "141203_172520.fits",
##             "141203_171941.fits",
##             "141203_173115.fits"]
##    data2 = ["141203_125447.fits",
##             "141203_131446.fits",
##             "141203_165051.fits",
##             "141203_165732.fits",
##             "141203_170347.fits",
##             "141203_170937.fits",
##             "141203_171600.fits",
##             "141203_172752.fits",
##             "141203_172155.fits",
##             "141203_173336.fits"]

    
##    data1 = ["141215_083751.fits",
##             "141215_125831.fits",
##             "141215_130715.fits",
##             "141215_131355.fits",
##             "141215_131904.fits",
##             "141215_132443.fits",
##             "141215_133110.fits",
##             "141215_133749.fits",
##             "141215_134616.fits",
##             "141215_135153.fits"]
##    data2 = ["141215_084610.fits",
##             "141215_130048.fits",
##             "141215_131013.fits",
##             "141215_131620.fits",
##             "141215_132136.fits",
##             "141215_132744.fits",
##             "141215_133401.fits",
##             "141215_134002.fits",
##             "141215_134845.fits",
##             "141215_135549.fits"]


##    data1 = ["150114_133930.fits",
##             "150114_134707.fits",
##             "150115_101702.fits",
##             "150114_140119.fits",
##             "150115_123623.fits",
##             "150114_140958.fits",
##             "150115_125715.fits",
##             "150114_154803.fits",
##             "150114_160249.fits",
##             "150114_160800.fits"]
##    data2 = ["150114_134403.fits",
##             "150114_134947.fits",
##             "150115_104201.fits",
##             "150114_140411.fits",
##             "150115_123921.fits",
##             "150114_141548.fits",
##             "150115_131014.fits",
##             "150114_155531.fits",
##             "150114_160500.fits",
##             "150114_161026.fits"]

##    data1 = ["150119_084638.fits",
##             "150119_092458.fits",
##             "150119_093033.fits",
##             "150119_094207.fits",
##             "150119_100312.fits",
##             "150119_102244.fits",
##             "150119_103726.fits",
##             "150119_110103.fits",
##             "150119_111757.fits",
##             "150119_113539.fits"]
##    data2 = ["150119_092105.fits",
##             "150119_092720.fits",
##             "150119_093401.fits",
##             "150119_095804.fits",
##             "150119_101606.fits",
##             "150119_103158.fits",
##             "150119_105709.fits",
##             "150119_111031.fits",
##             "150119_113246.fits",
##             "150119_114029.fits"]

##    data1 = ["150123_114918.fits",
##             "150123_120240.fits",
##             "150123_122250.fits",
##             "150123_150746.fits",
##             "150123_151543.fits",
##             "150123_152046.fits",
##             "150123_152528.fits",
##             "150123_153805.fits",
##             "150123_154314.fits"]
##    data2 = ["150123_115416.fits",
##             "150123_121629.fits",
##             "150123_125137.fits",
##             "150123_151018.fits",
##             "150123_151809.fits",
##             "150123_152258.fits",
##             "150123_153447.fits",
##             "150123_154038.fits",
##             "150123_154532.fits"]
    data1 = ["150217_103730.fits",
             "150217_104544.fits",
             "150217_105325.fits",
             "150217_110138.fits",
             "150217_102031.fits"]
    data2 = ["150217_104043.fits",
             "150217_104800.fits",
             "150217_105554.fits",
             "150217_110420.fits",
             "150217_101839.fits"]

    labels = ["M04055-06_VDD35V_32CH",
              "M04055-06_VDD35V_16CH",
              "M04055-06_VDD35V_8CH",
              "M04055-06_VDD35V_4CH",
              "M04055-06_VDD35V_VDDOPOFF"]
    
##    labels = ["VDD5V_VDDA&OP5V_VDDPIX5V",
##             "VDD5V_VDDA&OP45V_VDDPIX5V",
##             "VDD5V_VDDA&OP5V_VDDPIX45V",
##             "VDD45V_VDDA&OP45V_VDDPIX45V",
##             "VDD45V_VDDA&OP4V_VDDPIX45V",
##             "VDD45V_VDDA&OP45V_VDDPIX4V",
##             "VDD4V_VDDA&OP4V_VDDPIX4V",
##             "VDD4V_VDDA&OP4V_VDDPIX35V",
##             "VDD4V_VDDA&OP35V_VDDPIX4V",
##             "VDD35V_VDDA&OP35V_VDDPIX35V"]

    
##    labels = ["VDD35V_VDDA&OP35V_VDDPIX35V_16CH_1",
##              "VDD35V_VDDA&OP35V_VDDPIX35V_16CH_2",
##              "VDD35V_VDDA&OP35V_VDDPIX35V_16CH_3",
##              "VDD35V_VDDA&OP35V_VDDPIX35V_16CH_4",
##              "VDD35V_VDDA&OP35V_VDDPIX35V_32CH_1",
##              "VDD35V_VDDA&OP35V_VDDPIX35V_32CH_2",
##              "VDD35V_VDDA&OP35V_VDDPIX35V_32CH_3",
##              "VDD35V_VDDA&OP35V_VDDPIX35V_16CH_5",
##              "VDD35V_VDDA&OP35V_VDDPIX35V_16CH_6"]
    
    for slowFile, fastFile, label in map(None, data1, data2, labels):
##        d1 = openfits(f1) #1Hz data.
##        d2 = openfits(f2) #10Hz data.
##        glow = 2.9 * (d2 - d1) / (9 * 1175)
##        savefits("Diff" + V + ".fits", diff)
        glow, dark = glowReduction(slowFile, fastFile)
##        savefits("141207_Glow_" + label + ".fits", glow)
##        savefits("150119_Glow_" + label + ".fits", glow)
        savefits("150217_Glow_" + label + ".fits", glow)
        #dark = 2 * d1 - d2
##        dark = - 2.9 * (d2 - 10 * d1) / (9 * 1175)
##        savefits("141207_Dark_" + label + ".fits", dark)
##        savefits("150119_Dark_" + label + ".fits", dark)
        savefits("150217_Dark_" + label + ".fits", dark)

def glowReduction(slowFile, fastFile, slowRate = 1, fastRate = 10, exposureTime = 120):
    #Rates should be Hz, time should be s.
    #Calculate time.
    #slowT = exposureTime * slowRate - slowRate * 25 #From makeDarkMap.
    #fastT = exposureTime * fastRate - fastRate * 25
    #print "Slow time:", slowT, "fast time:", fastT
    #Load fits files.
##    slow = numpy.median(openfits(slowFile)) / slowRate
##    fast = numpy.median(openfits(fastFile)) / fastRate
    #Define window.
    y0 = 0
    y1 = 48
    #Let's just do the reduction here.
    print "Slow:", slowFile[:-5], "Fast:", fastFile[:-5]
    n = 20
    slowData = openfits(slowFile)[:,y0:y1,:]
    fastData = openfits(fastFile)[:,y0:y1,:]
    slowSum = numpy.zeros(slowData.shape[1:])
    fastSum = numpy.zeros(fastData.shape[1:])
    for i in range(0,n):
        slowSum += slowData[i + 5,:,:] - slowData[i - n,:,:]
        fastSum += fastData[i + 5,:,:] - fastData[i - n,:,:]
    slow = slowSum / (n * (slowData.shape[0] - (n + 5)) / slowRate)
    fast = fastSum / (n * (fastData.shape[0] - (n + 5)) / fastRate)
    #At this point both fast and slow should be in ADU/s.
    #Measurements of dark current should be per second.
    #glow = (fast - slow) / ((fastRate - slowRate) * (exposureTime - 2.5)) #For slow rate.
    glow = (fast - slow) / (fastRate - slowRate)
    #glow /= exposureTime
    #dark = (slow / (exposureTime - 25)) - glow
    dark = slow - glow
    #Convert ADU to e-
    glow *= 2.9
    dark *= 2.9
    print "Median glow:", numpy.median(glow), "e-/read"
    print "Median dark:", numpy.median(dark), "e-/s"
    return glow, dark

def medianPlot(filename, x = 0):
    data = openfits(filename)
    print data.shape
    m = []
    for i in range(data.shape[0]):
        try:
            m.append(numpy.median(data[i,:,x:]))
        except:
            print i
    plt.plot(range(data.shape[0]),m)
    plt.show()

def meanPlot(filename):
    data = openfits(filename)
    m = []
    for i in range(data.shape[0]):
        m.append(numpy.mean(data[i,:,:]))
    plt.plot(range(data.shape[0]),m)
    plt.show()

def medianPlots(filenames, window = [[0,256],[0,320]], align = False):
    for f in filenames:
        data = openfits(f)
        m = []
        for i in range(data.shape[0]):
            m.append(numpy.median(data[i,window[0][0]:window[0][1],window[1][0]:window[1][1]]))
        if align:
            offset = m[0]
            for i in range(len(m)):
                m[i] -= offset
        plt.plot(range(data.shape[0]),m)
        plt.ylabel("ADU")
        plt.xlabel("frame #")
    plt.show()

def photonCounting2(histogramRange = [-100,100], binspacing = 1.0, avg = 1, first32 = False, detrend = False,\
                    referencePixels = False, ratioplot = False, cutoff = 0, start = 0, rrr = False, \
                    plotSample = False, mask = False, avgSum = False, plotDifference = True, logPlot = True,
                    gainNormalizedPlot = False, electronsX = False, ymax = 30000, ymin = 0, model = False,
                    rawADU = False):
##    offFile = "150209_162545.fits"
##    onFile =  "150209_162612.fits"
##    offFile = "150413_113438.fits" #COMMON = -8V, mk10 M04935-17
##    onFile = "150413_113453.fits"
##    offFile = "150413_141134.fits" #COMMON = -8V, mk10 M04935-17, ref pixels
##    onFile = "150413_141147.fits"
##    offFile = "150413_162347.fits" #COMMON = -10V, mk10 M04935-17, ref pixels
##    onFile = "150413_162358.fits"
##    offFile = "150414_134745.fits" #COMMON = -11V, mk10 M04935-17, ref pixels
##    onFile = "150414_134803.fits"
##    offFile = "150414_144226.fits" #COMMON = -8V, mk10 M04935-17, ref pixels, 16ch
##    onFile = "150414_144243.fits"
##    offFile = "150422_091955.fits" #COMMON = -10V, mk10 M04935-17, ref pixels, 32ch, unmasked
##    onFile = "150422_092006.fits"
##    offFile = "150625_124409.fits" #COMMON = -4.5V, mk3 M02775-35, ref pixels, 32ch, unmasked
##    onFile = "150422_092006.fits"
##    offFile = "151008_093127.fits" #COMMON = -8V, mk12 M06495-27, no ref, 32 x 1
##    onFile = "151008_093152.fits"

    
##    offFile = "151109_065456.fits" #COMMON = -11V, mk14 M06715-29, no ref, 32 x 1
##    onFile = "151109_070334.fits" #LED 0.8V
##    onFile = "151109_071428.fits" #LED 1.0V
##    onFile = "151109_072656.fits" #LED 1.1V
##    title = "BIAS = 14.5V, M06715-29, Data Taken 9 Nov 2015"
##    chargeGain = 2.89 #e-/ADU  ME911
##    gain = 45.6
    

    #MARK 14 CANON
##    offFile = "151117_113112.fits" #COMMON = -11V, mk14 M06715-27, no ref, 32 x 1
##    onFile = "151117_113131.fits" #LED 2.812V
##    title = "BIAS = 14.5V, M06715-27, Data Taken 17 Nov 2015"
##    chargeGain = 2.89 #e-/ADU  ME911
##    gain = 56.4
    
##    offFile = "160109_155405.fits" #COMMON = -11V, mk14 M06665-12, no ref, 32 x 1
##    onFile = "160109_155441.fits" #LED1 = 1.2V
##    title = "BIAS = 14.5V, M06665-12, Data Taken 9 Jan 2016"
##    chargeGain = 2.89 #e-/ADU  ME911
##    gain = 77.5



##    offFile = "160415_LEDoffCOM-10V-16-0.fits" #COMMON = -10V, VDD = 5V, w/ PB, 64 x 1
##    onFile = "160415_LED0.9VCOM-10V-18-0.fits"
##
##    offFile = "CUBE_LED_OFF-3-0.fits" #COMMON = -15V, VDD = 5V, w/ PB, 32 x 1
##    onFile = "160422_LED3.2VCOM-15V-4-0.fits"
##    title = "BIAS = 19.5V, M06715-27, Data Taken 22 Apr 2016"
##    chargeGain = 2.89 #e-/ADU  ME911
##    gain = 400
    
##    offFile = "160413_64x1LEDoff-2-0.fits" #COMMON = -15V, VDD = 5V, w/ PB, 64 x 1
##    onFile = "160413_64x1LEDon-3-0.fits"
##    title = "BIAS = 19.5V, M06715-27, Data Taken 13 Apr 2016"
##    chargeGain = 2.89 #e-/ADU  ME911
##    gain = 400

##    offFile = "subarray_32x1_LEDOFF_4_28-1-0.fits" #COMMON = -8V, VDD = 5V, w/ PB, 32 x 1
##    onFile = "subarray_32x1_LED60mA_4_28-2-0.fits"

##    offFile = "160723_PCLEDoff2-29-0.fits" #COMMON = -15V, mk 13 M06665-25 w/ PB, no ref, 32 x 1 not RRR
##    onFile = "160723_PCLEDon-30-0.fits" #LED 3.1um 4.2V 100mA
##    title = "BIAS = 19.5V, M06665-25 w/ PB-32, Data Taken 24 Jul 2016"
##    chargeGain = 1.57 #e-/ADU  ME1000
##    gain = 400

##    offFile = "161012_134758.fits" #COMMON = -11V, mk13 M06665-25 ME1000, no ref, 32 x 1
##    onFile = "161012_134843.fits"
##    title = "$V_{bias} = 14.5\mathrm{V}$, M06665-25, 60K, Data Taken 12 Oct 2016"
##    chargeGain = 1.57 #e-/ADU ME1000
##    gain = 65.6

##    offFile = "161012_141312.fits" #COMMON = -11V, mk13 M06665-25 ME1000, no ref, 32 x 1, diff window
##    onFile = "161012_141410.fits"
##    title = "BIAS = 14.5V, M06665-25, Data Taken 12 Oct 2016"

##    offFile = "161016_140515.fits" #COMMON = -11V, mk13 M06665-23 ME1000, no ref, 32 x 1, diff window
##    onFile = "161016_140609.fits"
##    title = "BIAS = 14.5V, M06665-23, Data Taken 16 Oct 2016"
##    chargeGain = 1.57 #e-/ADU ME1000
##    gain = 65.6

##    offFile = "161106_105035.fits" #COMMON = -9V, mk13 M06665-23 ME1000, no ref, 32 x 1, diff window
##    onFile = "161106_105045.fits"
##    title = "BIAS = 12.5V, M06665-23, Data Taken 6 Nov 2016"

    #MARK 13 ME1000 CANON
    offFile = "161106_105115.fits" #COMMON = -11V, mk13 M06665-23 ME1000, no ref, 32 x 1, diff window
    onFile = "161106_105124.fits"
##    title = "$V_{bias} = 14.5\mathrm{V}$, M06665-23, 60K, Data Taken 6 Nov 2016"
    title = "Histogram of Pixel Response for Mark 13 M06665-23\n$V_{bias} = 14.5\mathrm{V}$, $T = 62.5\mathrm{K}$"
    chargeGain = 1.57 #e-/ADU ME1000
##    chargeGain = 2.89 #test
    gain = 65.6
##    gain = 1.0

##    offFile = "170308_123137.fits" #COMMON = -11V, mk13 M06665-23 ME1000, no ref, 32 x 1, diff window
##    onFile = "170308_123144.fits"
##    offFile = "170308_123154.fits" #COMMON = -11V, mk13 M06665-23 ME1000, no ref, 32 x 1, diff window
##    onFile = "170308_123205.fits"
##    offFile = "170308_123214.fits" #COMMON = -11V, mk13 M06665-23 ME1000, no ref, 32 x 1, diff window
##    onFile = "170308_123222.fits"
##    title = "$V_{bias} = 14.5\mathrm{V}$, M06665-23, 60K, Data Taken 8 Mar 2017"
##    chargeGain = 1.57 #e-/ADU ME1000
##    gain = 65.6

##    offFile = "170130_095713.fits"
##    onFile = "170130_095725.fits"
##    title = "who cares"
##    chargeGain = 1.57
##    gain = 65.6

    #MARK 13 ME1000 40K

    #Test to show noise difference
##    offFile = "151117_113112.fits"
##    onFile = "161012_141312.fits"
##    title = "Noise comparison between tests"

    #MARK 19 ME1001 60K
##    offFile = "170314_110753.fits" #COMMON = -11V, mk19 M09225-27 ME1001, no ref, 32 x 1, diff window
##    onFile = "170314_110802.fits" #NO LIGHT
##    title = "$V_{bias} = 14.5\mathrm{V}$, M09225-27, 60K, Data Taken 14 Mar 2017"
##    chargeGain = 5.18 #e-/ADU ME1001?
##    gain = 104.3

##    offFile = "170324_162320.fits"
##    onFile = "170324_162337.fits"
##    title = "$V_{bias} = 14.5\mathrm{V}$, M09225-11, 60K, Data Taken 24 Mar 2017"
##    chargeGain = 4.71 #e-/ADU ME1001
##    gain = 91.7
    
    off = openfits(offFile)
    on = openfits(onFile)

    

    histrange = numpy.array(histogramRange)

    if rrr:
        oldOff = numpy.array(off)
        off = numpy.zeros([oldOff.shape[0] / 2, oldOff.shape[1], oldOff.shape[2]])
        for n in range(2,oldOff.shape[0],2):
            off[n/2,:,:] = oldOff[n,:,:] - oldOff[n - 1,:,:]
        oldOn = numpy.array(on)
        on = numpy.zeros([oldOn.shape[0] / 2, oldOn.shape[1], oldOn.shape[2]])
        for n in range(2,oldOn.shape[0],2):
            on[n/2,:,:] = oldOn[n,:,:] - oldOn[n - 1,:,:]

    if first32:
        on = on[:,:,:32]
        off = off[:,:,:32]
    if cutoff > 0:
        on = on[:cutoff,:,:32]
        off = off[:cutoff,:,:32]
    if start > 0:
        on = on[start:,:,:]
        off = off[start:,:,:]
    
    #Subtract reference pixels if necessary.
    if referencePixels:
        print "Subtracting reference pixels."
##        offMedian1 = numpy.zeros([on.shape[0]])
##        offMedian2 = numpy.zeros([on.shape[0]])
##        offMedian3 = numpy.zeros([on.shape[0]])
##        offMedian4 = numpy.zeros([on.shape[0]])
##        for i in range(off.shape[0]):
##            offMedian1[i] = numpy.median(off[i,0,:32])
##            offMedian2[i] = numpy.median(off[i,0,32:])
##            offMedian3[i] = numpy.median(off[i,1,:32])
##            offMedian4[i] = numpy.median(off[i,1,32:])
##        plt.plot(range(offMedian1.shape[0]), offMedian1)
##        plt.plot(range(offMedian2.shape[0]), offMedian2)
##        plt.plot(range(offMedian3.shape[0]), offMedian3)
##        plt.plot(range(offMedian4.shape[0]), offMedian4)
##        plt.show()
        ch = on.shape[2] / 2
        onReferenced = numpy.zeros([on.shape[0] - 1, on.shape[1], ch])
        offReferenced = numpy.zeros([off.shape[0] - 1, off.shape[1], ch])
        for i in range(onReferenced.shape[0]):
##            onReferenced[i,1,:] = on[i + 1,1,:32] - ((on[i + 1,1,32:] + on[i,0,32:]) / 2)
##            onReferenced[i,0,:] = on[i + 1,0,:32] - ((on[i + 1,1,32:] + on[i + 1,0,32:]) / 2)
            onReferenced[i,1,:] = on[i,1,:ch] - on[i,1,ch:]
            onReferenced[i,0,:] = on[i,0,:ch] - on[i,0,ch:]
        for i in range(offReferenced.shape[0]):
##            offReferenced[i,1,:] = off[i + 1,1,:32] - ((off[i + 1,1,32:] + off[i,0,32:]) / 2)
##            offReferenced[i,0,:] = off[i + 1,0,:32] - ((off[i + 1,1,32:] + off[i + 1,0,32:]) / 2)
            offReferenced[i,1,:] = off[i,1,:ch] - off[i,1,ch:]
            offReferenced[i,0,:] = off[i,0,:ch] - off[i,0,ch:]
        on = onReferenced
        off = offReferenced
    #Check the lengths, trim from the end to match.
    if (off.shape[0] <> on.shape[0]):
        print "Length mismatch, trimming."
        if off.shape[0] > on.shape[0]:
            off = off[: - (off.shape[0] - on.shape[0]),:,:]
        else:
            on = on[: - (on.shape[0] - off.shape[0]),:,:]
    #Do some detrending with the median.
    if detrend:
        print "Detrending..."
        onMedian = []
        offMedian = []
        for i in range(off.shape[0]):
            #Only apply it to a given range of high pixel frames.
            offMed = numpy.median(off[i,0,:])
            off[i,:,:] -= offMed
            onMed = numpy.median(on[i,0,:])
            on[i,:,:] -= onMed
            offMedian.append(numpy.median(off[i,:,:]))
            onMedian.append(numpy.median(on[i,:,:]))
        
    #Show median plots.
    if plotSample:
        plt.plot(range(off.shape[0]), off[:,0,0], label = "LED off")
        plt.plot(range(on.shape[0]), on[:,0,0], label = "LED on")
        plt.legend()
        plt.xlabel("Frame #")
        plt.ylabel("ADU")
        plt.show()

    #Detrend with lowest values in each frame.
##    for i in range(off.shape[0]):
##        off[i,:,:] -= numpy.mean(numpy.sort(off[i,:,:])[-4:])
##    for i in range(on.shape[0]):
##        on[i,:,:] -= numpy.mean(numpy.sort(on[i,:,:])[-4:])

    
    #Let's generate a mask.
    if mask:
        pixels = []
        for x in range(off.shape[2]):
            for y in range(off.shape[1]):
                pixels.append(off[10,y,x] - off[off.shape[0]-10,y,x])
        sortPix = numpy.sort(numpy.array(pixels), axis = None)
        #print "Sort Pix:", sortPix
        cutoff = sortPix[sortPix.shape[0] / 2]
        #print "Cutoff:", cutoff
        mask = numpy.zeros([off.shape[1],off.shape[2]])
        for x in range(off.shape[2]):
            for y in range(off.shape[1]):
                if (off[10,y,x] - off[off.shape[0]-10,y,x] < cutoff):
                    mask[y,x] = 1
    #print "Mask:", mask

    #Try average before CDS, per Don.
    if avg > 1:
        off = avgCube(off, avg = avg)
        on = avgCube(on, avg = avg)
        if avgSum:
            off *= avg
            on *= avg

    #Perform subtraction and masking.
    offSub = numpy.zeros([off.shape[0] - 1,off.shape[1],off.shape[2]])
    onSub = numpy.zeros([on.shape[0] - 1,on.shape[1],on.shape[2]])
    
##    offCount = 0
##    onCount = 0
    for i in range(offSub.shape[0]):
        if mask:
            offSub[i,:,:] = (off[i,:,:] - off[i + 1,:,:]) * mask
            onSub[i,:,:] = (on[i,:,:] - on[i + 1,:,:]) * mask
        else:
            offSub[i,:,:] = (off[i,:,:] - off[i + 1,:,:])
            onSub[i,:,:] = (on[i,:,:] - on[i + 1,:,:])

    #Do averaging.
##    if avg > 1:
##        offSub = avgCube(offSub, avg = avg)
##        onSub = avgCube(onSub, avg = avg)
##        if avgSum:
##            offSub *= avg
##            onSub *= avg


    if not(rawADU):
        offSub *= chargeGain
        onSub *= chargeGain
        binspacing *= chargeGain
##        histrange[0] = int(chargeGain * histrange[0]) + 0.01
##        histrange[1] = int(chargeGain * histrange[1]) + 0.01

    print "histrange:", histrange
    print "binspacing:", binspacing

    #Compute bins.
    bins = int((histrange[1] - histrange[0]) / binspacing)
    print "bins:", bins

##    if electronsX:
##        offSub /= gain
##        onSub /= gain
##        histrange /= gain
        
    #Plot with difference line if requested.
    if plotDifference:
        hist1 = numpy.histogram([offSub.flatten()], bins = bins, range = histrange)
        hist2 = numpy.histogram([onSub.flatten()], bins = bins, range = histrange)
        print hist1[1][3] - hist1[1][2], "binspacing", binspacing

        if model:
            histmodel = modelAPD()
##            histmodel = numpy.array(histmodel) / gain
            histmodel = numpy.array(histmodel) / 2
##            histmodel = numpy.histogram(histmodel, bins = bins, range = histrange)
            histmodel = numpy.histogram(histmodel, bins = bins / 3, range = histrange)
            histmodel = numpy.array(histmodel)
##            histmodel[0] = histmodel[0] * 4.3 #MODEL INCREASE
##            histmodel[0] = histmodel[0] * 1.8 #MODEL INCREASE
            for j in range(len(histmodel[1])):
                histmodel[1][j] = histmodel[1][j] * chargeGain

##        hist1sum = numpy.max(hist1[0])
##        hist2sum = numpy.max(hist2[0])

##        if not(rawADU):
##            for j in range(len(hist1[1])):
##                print hist1[1][j]
##                hist1[1][j] = hist1[1][j] * chargeGain #REMOVED
####        for j in range(len(hist1[0])):
####            hist1[0][j] = hist1[0][j] / hist1sum
##            for j in range(len(hist2[1])):
##                hist2[1][j] = hist2[1][j] * chargeGain #REMOVED
####        for j in range(len(hist2[0])):
####            hist2[0][j] = hist2[0][j] / hist2sum

        diff = hist2[0] - hist1[0]

##        i = diff.shape[0] / 2
##        while diff[i] < hist2[0][i] / 2:
##            i += 1
##        print "Threshold:", hist2[1][i]
##        del i

##        #Screen zeros in on - off
##        for i in range(diff.shape[0]):
##            if diff[i] < 0:
##                diff[i] = 0
        if gainNormalizedPlot:
            for j in range(len(hist1[1])):
                hist1[1][j] = hist1[1][j] / gain
            for j in range(len(hist2[1])):
                hist2[1][j] = hist2[1][j] / gain
##            if model:
##                for j in range(len(histmodel[1])):
##                    histmodel[1][j] = histmodel[1][j] / gain
        
        if logPlot:
            plt.semilogy(hist1[1][:-1],hist1[0], 'b-')
            plt.semilogy(hist2[1][:-1],hist2[0], 'r-')
            plt.semilogy(hist1[1][:-1],diff, 'k-')
        else:
            plt.plot(hist1[1][:-1],hist1[0], 'b-')
            plt.plot(hist2[1][:-1],hist2[0], 'r-')
            plt.plot(hist1[1][:-1],diff, 'k-')
            if model:
##                print "fff"
##                print histmodel[0]
##                print "fff"
                plt.plot(histmodel[1][:-1],histmodel[0], 'g-')
            plt.ylim([ymin,ymax]) #YLIM

        #Fit to difference line.
##        fitStartValue = 100
##        fitEndValue = 250

##        if gainNormalizedPlot:
##            fitStartValue = 0.9
##            fitEndValue = 1.1
##        else:
##            fitStartValue = gain - 10
##            fitEndValue = gain + 10
##        fitStart = 0
##        while hist1[1][fitStart] < fitStartValue:
##            fitStart += 1
##        fitEnd = fitStart
####        while diff[fitEnd] > 0 and fitEnd < diff.shape[0] - 1:
##        while hist1[1][fitEnd] < fitEndValue and diff[fitEnd + 1] > 0:
##            fitEnd += 1
##            
##        print "hist1[1][fitStart]", hist1[1][fitStart]
##        m, b = linearfit(hist1[1][fitStart:fitEnd],numpy.log(diff[fitStart:fitEnd]))
##        print m,b
##        fitPlotStart = 0
##        fitPlotEnd = hist1[1][-1]
##        plt.plot([fitPlotStart, fitPlotEnd],[e**(fitPlotStart * m + b), e**(fitPlotEnd * m + b)], 'k--')

##        if gainNormalizedPlot:
##            plt.legend(["LED off","LED on","on - off"])
####            plt.legend(["LED off","LED on","on - off","$e^{"+str(round(m,3))+"}$"])
##        else:
        if model:
            plt.legend(["LED off","LED on","on - off","model"])
        else:
            plt.legend(["LED off","LED on","on - off"])
##            plt.legend(["LED off","LED on","on - off","$e^{"+str(round(gain * m,3))+"/"+str(gain)+"}$"])
        
    else:
        if not(rawADU):
            offSub *= chargeGain
            onSub *= chargeGain
            binspacing *= chargeGain
            histrange[0] = int(chargeGain * histrange[0])
            histrange[1] = int(chargeGain * histrange[1])
        if logPlot:
            plt.hist([offSub.flatten(),onSub.flatten()], bins = bins, range=histrange, color = ['b','r'], histtype = 'step', log = True)
        else:
            plt.hist([offSub.flatten(),onSub.flatten()], bins = bins, range=histrange, color = ['b','r'], histtype = 'step', log = False)
        plt.legend(["LED on","LED off"])
    
##    plt.xlabel("ADU")
##    plt.xlabel("$e^{-}$")
    if gainNormalizedPlot:
        plt.xlabel("Gain (normalized)")
##        plt.xlim([histrange[0]/gain,histrange[1]/gain])
        plt.xlim([-3,5])        
    else:
        if electronsX:
            plt.xlabel("$e^-$")
        else:
            if rawADU:
                plt.xlabel("ADU")
            else:
                plt.xlabel("Gain")
        plt.xlim([histrange[0],histrange[1]])
    plt.ylabel("number of reads $N$")
    if avg > 1:
        title += "\n" + str(avg) + " average(s)"
##    title += "\n bin size: " + str(round(binspacing * 100) / 100) + ", " + str(offSub.shape[0]) + " frames"
    if detrend:
        title += ", detrended"
    plt.title(title)
    #plt.gca().set_ylim(bottom = 1)
    plt.show()

##    return hist1[1][:-1], diff

    if ratioplot:
        hist1 = numpy.histogram([offSub.flatten()], bins = bins, range = histrange)
        hist2 = numpy.histogram([onSub.flatten()], bins = bins, range = histrange)
        #print hist1[0] / hist2[0]
        ax = plt.subplot(111)
        ax.bar(hist2[1][:-1], hist2[0] / hist1[0].astype(float), width = 0.5)
    ##    ax.bar(hist2[1][:-1], hist2[0] - hist1[0], width = 0.5)
##        plt.title(title + " ratio plot")
        plt.show()
    print off.shape
    print on.shape

    #x = 0, avg = 2, histrange = (0,30), binspacing = 0.5, rampplot = False, screenforpeaks = False, mediansubtraction = False,\
                 #fullstatistics = False, perpixelstatistics = False, ratioplot = False, meanplot = False, onoffplot = False,\
                 #writecsv = False, csvfile = "default.csv")

def PCAveragingInvestigation(histrange = [-100,200], binspacing = 1.0):
    filename = "151117_113131.fits" #LED 2.812V
    title = "BIAS = 14.5V, M06715-27, Data Taken 17 Nov 2015"

    print "Loading fits file..."
    d = openfits(filename)

    #Perform rolling CDS.
    sub = numpy.zeros([d.shape[0] - 1,d.shape[1],d.shape[2]])
    for i in range(sub.shape[0]):
        sub[i,:,:] = (d[i,:,:] - d[i + 1,:,:])

    print "Averaging..."
    averages = [1,2,4,8]
    #Do averaging.
    averaged = []
    for i,a in enumerate(averages):
        if a > 1:
            averaged.append((avgCube(sub, avg = a)*a))
##            averaged.append((avgCube(sub, avg = a) * a)[:,0,0])
        else:
            averaged.append(sub)
##            averaged.append(sub[:,0,0])
        temp = numpy.array(averaged[-1])
        averaged[-1] = numpy.zeros([temp.shape[0] * a, temp.shape[1], temp.shape[2]])
        for x in range(a):
##            print a, x, temp.shape, averaged[-1].shape
            averaged[-1][x * temp.shape[0]:(x + 1) * temp.shape[0],:,:] = temp

    for i,av in enumerate(averaged):
        av = (av.flatten()).flatten()
        averaged[i] = av

    print "Plotting..."
    #Compute bins.
    bins = (histrange[1] - histrange[0]) / binspacing
    plt.hist(averaged, bins = bins, range=histrange)
##    plt.hist([offSub[:,0,1].flatten()], bins = bins, range=histrange)
##    plt.legend(["LED off","LED on"])
    plt.xlabel("ADU")
    plt.ylabel("n")

    legend = []
    print averaged[-1].shape
    for i,a in enumerate(averages):
        legend.append("AVG" + str(a) +
                      " MEDIAN " + str(round(numpy.median(averaged[i]))))
    plt.legend(legend)
    
##    plt.title(title)
    plt.show()



def readNoise(filename, detrend = True, start = 0, gain = 2.89, plot = True):
##    d = cdsCube(openfits(filename))[:,:1,:32]
    d = openfits(filename)[start:,:,:]
    #CDS?
    #Detrend the data?
    if detrend:
        print "Detrending..."
        for n in range(d.shape[0]):
            for y in range(d.shape[1]):
                for x in range(d.shape[2]/32):
                    d[n,y,x*32:(x+1)*32] -= numpy.median(d[n,y,x*32:(x+1)*32])
    #Do a CDS, since that's how we get the actual read noise.
    cds = cdsCube(d)
    #Take the standard deviations.
    stddevs = numpy.zeros([cds.shape[1],cds.shape[2]])
    for y in range(cds.shape[1]):
        for x in range(cds.shape[2]):
            stddevs[y,x] = numpy.std(cds[:,y,x], ddof = 1)
            print y,",",x,stddevs[y,x]
    sd = numpy.median(stddevs.flatten())
    print "CDS read noise:"
    print sd, "ADU"
    print sd * gain, "e-"
    if plot:
        plt.hist(cds.flatten(), bins = 60, range = [-30,30])
        plt.show()
    return sd
    

    
def commonNoise(filename, rate = 265.e3):
    data = openfits(filename)[:,:,:]
    #We need to unpack the data.
    unpacked = numpy.zeros([data.shape[0] * 4, data.shape[1] / 2, data.shape[2] / 2])
    for n in range(data.shape[0]):
        unpacked[n * 4, :, :] = data[n, :1, :32]
        unpacked[n * 4 + 1, :, :] = data[n, :1, 32:]
        unpacked[n * 4 + 2, :, :] = data[n, 1:, :32]
        unpacked[n * 4 + 3, :, :] = data[n, 1:, 32:]
    d2 = cdsCube(unpacked)
    d1 = unpacked
##    d2 = cdsCube(openfits(filename))[:,1:,:32]
##    d1 = openfits(filename)[:,1:,:32]
    e1 = numpy.zeros([d1.shape[0]])
    e2 = numpy.zeros([d2.shape[0]])
    e3 = numpy.zeros([d2.shape[0]])
    for i in range(e1.shape[0]):
        e1[i] = numpy.median(d1[i,:,:])
    #plt.plot(e)
    for i in range(e2.shape[0]):
        e2[i] = numpy.median(d2[i,:,:])
    for i in range(e3.shape[0]):
        e3[i] = d2[i,0,0] - e2[i]
    #e3 = d2 - e2
    f1 = abs(numpy.fft.rfft(e1))
    f2 = abs(numpy.fft.rfft(e2))
    f3 = abs(numpy.fft.rfft(e3))
##    rate = 265.e3 / 4
##    rate = 265.e3
##    rate = 205.e3
    print "Sampling rate:", rate, "Hz"
    freq1 = numpy.fft.rfftfreq(e1.shape[0], d = 1. / rate)
    freq2 = numpy.fft.rfftfreq(e2.shape[0], d = 1. / (rate / 2))
    freq3 = numpy.fft.rfftfreq(e3.shape[0], d = 1. / (rate / 2))
    plt.subplot(311)
    plt.plot(freq1,f1 / numpy.sqrt(freq1))
##    plt.xlim([0,16000])
    plt.xlim([0,rate / 4])
    plt.ylim([0,2000])
##    plt.ylim([0,500])
    plt.ylabel("ADU/(Hz^1/2)")
    plt.title("Raw median")
    plt.subplot(312)
    plt.plot(freq2,f2 / numpy.sqrt(freq2))
##    plt.xlim([0,16000])
    plt.xlim([0,rate / 4])
    plt.ylim([0,1000])
    plt.ylabel("ADU/(Hz^1/2)")
    plt.title("CDS median")
    plt.subplot(313)
    plt.plot(freq3,f3 / numpy.sqrt(freq3))
##    plt.xlim([0,16000])
    plt.xlim([0,rate / 4])
    plt.ylim([0,500])
    plt.title("CDS subtracted")
    plt.xlabel("Hz")
    plt.ylabel("ADU/(Hz^1/2)")
    plt.show()

def commonNoiseComparison():
    filenames = [["150120_100815.fits","150120_100822.fits"],
                 ["150127_091133.fits","150127_091145.fits"],
                 ["150127_091214.fits","150127_091221.fits"],
                 ["150202_115240.fits","150202_115248.fits"]]
##                 ["150122_151139.fits","150122_151146.fits"],
##                 ["150122_151239.fits","150122_151246.fits"],
##                 ["150122_151326.fits","150122_151333.fits"]]
##                 ["150120_154913.fits","150120_154921.fits"],
##                 ["150120_155204.fits","150120_155213.fits"],
##                 ["150120_155613.fits","150120_155629.fits"],
##                 ["150121_091435.fits","150121_091506.fits"],
##                 ["150121_103205.fits","150121_103254.fits"],
##                 ["150121_112949.fits","150121_113001.fits"],
##                 ["150122_133602.fits","150122_133637.fits"]]
##                 ["150122_110027.fits","150122_110037.fits"]]
##                 ["150121_092139.fits","150121_092146.fits"],
##                 ["150121_092941.fits","150121_092955.fits"]]
##                 ["150120_100523.fits","150120_100629.fits"],
##                 ["150120_100844.fits","150120_100853.fits"],
##                 ["150120_100950.fits","150120_100956.fits"],
##                 ["150120_131747.fits","150120_131814.fits"],
##                 ["150120_133909.fits","150120_133919.fits"],
##                 ["150120_141103.fits","150120_141126.fits"]]
##    leg = ["Baseline",
##           "Shields Grounded",
##           "Cryopump Disconnected",
##           "LSP Disconnected"]
    leg = ["data"]
##           "60Hz",
##           "120Hz",
##           "1kHz"]
##           "ANU Board Filtering",
##           "Common Grounded",
##           "Common Ungrounded 0V",
##           "Capacitors Disconnected",
##           "Capacitors Disconnected for Realsies",
##           "LED Power Supply Off",
##           "Capacitor on COMMON"]
##           "PRV"]
##           "Capacitors Disconnected #2",
##           "Capacitors Disconnected #3"]
##           "Leach Power Supply",
##           "Lights Off",
##           "Lakeshore Off",
##           "Terminated",
##           "Ground Disconnected",
##           "LSP Terminated"]
           
    for pair in filenames:
##        p0 = cdsCube(openfits(pair[0]))
##        p1 = cdsCube(openfits(pair[1]))
        p0 = openfits(pair[0])
        p1 = openfits(pair[1])
        l = 0
##        l = p0.shape[0]
        if p0.shape[0] > p1.shape[0]:
            l = p1.shape[0]
        else:
            l = p0.shape[0]
        e = numpy.zeros(l)
        fft = numpy.zeros(numpy.fft.rfft(e).shape)
        for f in pair:
##        f = pair[0]
##            d = cdsCube(openfits(f))[:,:1,:8]
            d = openfits(f)[:,:1,:8]
            e = numpy.zeros(l)
            for i in range(l):
                e[i] += numpy.median(d[i,:,:])
            fft += abs(numpy.fft.rfft(e))
        fft /= 2
        rate = 265.e3 / 4.
##        freq = numpy.fft.rfftfreq(e.shape[0], d = 1. / (rate / 2))
        freq = numpy.fft.rfftfreq(e.shape[0], d = 1. / (rate))
        plt.plot( freq, fft / numpy.sqrt(freq))
    plt.xlim([0, 16000])
    plt.ylim([0.1,1500])
    plt.ylabel("ADU/(Hz^1/2)")
    plt.legend(leg)
    plt.show()
    

def diffTheFits():
    glow1 = ["141207_Glow_VDD5V_VDDA&OP5V_VDDPIX5V.fits",
             "141207_Glow_VDD5V_VDDA&OP45V_VDDPIX5V.fits",
             "141207_Glow_VDD5V_VDDA&OP5V_VDDPIX45V.fits"]
    glow2 = ["141207_Glow_VDD45V_VDDA&OP45V_VDDPIX45V.fits",
             "141207_Glow_VDD45V_VDDA&OP4V_VDDPIX45V.fits",
             "141207_Glow_VDD45V_VDDA&OP45V_VDDPIX4V.fits"]
    glow3 = ["141207_Glow_VDD4V_VDDA&OP4V_VDDPIX4V.fits",
             "141207_Glow_VDD4V_VDDA&OP35V_VDDPIX4V.fits",
             "141207_Glow_VDD4V_VDDA&OP4V_VDDPIX35V.fits"]
    glow4 = ["141207_Glow_VDD4V_VDDA&OP4V_VDDPIX4V.fits"]
    glow5 = ["141207_Glow_VDD35V_VDDA&OP35V_VDDPIX35V.fits"]
    dark1 = ["141207_Dark_VDD5V_VDDA&OP5V_VDDPIX5V.fits",
             "141207_Dark_VDD5V_VDDA&OP45V_VDDPIX5V.fits",
             "141207_Dark_VDD5V_VDDA&OP5V_VDDPIX45V.fits"]
    dark2 = ["141207_Dark_VDD45V_VDDA&OP45V_VDDPIX45V.fits",
             "141207_Dark_VDD45V_VDDA&OP4V_VDDPIX45V.fits",
             "141207_Dark_VDD45V_VDDA&OP45V_VDDPIX4V.fits"]
    dark3 = ["141207_Dark_VDD4V_VDDA&OP4V_VDDPIX4V.fits",
             "141207_Dark_VDD4V_VDDA&OP35V_VDDPIX4V.fits",
             "141207_Dark_VDD4V_VDDA&OP4V_VDDPIX35V.fits"]
    dark4 = ["141207_Dark_VDD4V_VDDA&OP4V_VDDPIX4V.fits"]
    dark5 = ["141207_Dark_VDD35V_VDDA&OP35V_VDDPIX35V.fits"]
    groups1 = [glow1, glow2, glow4, dark1, dark2, dark4]
    groups2 = [glow2, glow3, glow5, dark2, dark3, dark5]
    for a,b in map(None,groups1,groups2):
        for g1f, g2f in map(None, a, b):
            g1 = openfits(g1f)
            g2 = openfits(g2f)
            for y in range(g2.shape[0]):
                for x in range(g2.shape[1]):
                    if g2[y,x] == 0:
                        g2[y,x] = 0.0001
            g = g1 / g2
            nameEnd1 = g1f.index('.')
            nameEnd2 = g2f.index('.')
            filename = "141230_" +  g1f[7:nameEnd1] + "_" + g2f[12:nameEnd2] + ".fits"
            savefits(filename, g)

def comparativeDarkPlots():
    #27 Jan 2015
    #Goal is to plot measured median dark currents for different
    #output settings and voltages, and to perform a linear regression
    #fit.
    V35 = numpy.array([0.285,0.473,0.702,1.219])
    V50 = numpy.array([2.725, 4.457, 7.095, 13.359])
    outputs = numpy.array([4, 8, 16, 32])
    A = numpy.vstack([outputs, numpy.ones(len(outputs))]).T
    m35, b35 = numpy.linalg.lstsq(A, V35)[0]
    m50, b50 = numpy.linalg.lstsq(A, V50)[0]
    r = numpy.array(range(outputs[-1]))
    plt.plot(outputs, 10 * V35, 'o-')
    plt.plot(r, 10 * m35 * r + 10 * b35)
    plt.plot(outputs, V50, 'o-')
    plt.plot(r, m50 * r + b50, 'm')
    print b35, b50
    plt.xlabel("# output channels")
    plt.ylabel("dark current (e-/s")
    plt.legend(["10 * VDD 3.5V", "10 * 3.5V fitted", "VDD 5.0V", "5.0V fitted"], loc = 2)
    plt.title("3.5V slope (* 10): " + str(m35 * 10)[:5] + ", 5.0V slope: " + str(m50)[:5])
    plt.show()

def channelsVsDark():
    #Full list of # channels vs. Dark for both detectors.
    mark5at110K = [ 68.2,  68.3,  68.6, 55.65]
    mark5at85K  = [2.284, 1.831, 1.612, 1.273]
    mark5at60K  = [1.025, 0.623, 0.397, 0.246]
    mark3at60K  = [1.219, 0.702, 0.474, 0.285]
    mark2at60K  = [2.449, 1.773, 1.492, 0.949]
    channels = [32, 16, 8, 4]
    colors = ['b', 'g', 'r', 'k']
    for i,d in enumerate([mark3at60K, mark5at60K, mark5at85K]):
        plt.plot(channels, d, colors[i] + 'o-')
        m, b = linearfit(channels, d)
        x = numpy.array(range(0,channels[0]))
        y = m * x + b
        plt.plot(x, y, colors[i] + '--')
##    plt.semilogy(channels, mark5at85K, 'o-')
##    plt.semilogy(channels, mark5at110K, 'o-')
    plt.legend(["mk3 M02775-10 @ 60K", "mk3 60K fit",\
                "mk5 M04055-06 @ 60K", "mk5 60K fit",\
                "mk5 M04055-06 @ 85K", "mk5 85K fit"], loc = 2)
    plt.xlabel("# outputs")
    plt.ylabel("dark current (e-/s)")
    plt.ylim([0, 2.5])
    plt.title("Dark vs. outputs, all detectors")
    plt.show()

def mark2VsDark():
    #Full list of # channels vs. Dark for both detectors.
    mark2at60K = [2.449, 1.773, 1.492, 0.949]
    mark2at60Ktoggled = [0.932, 1.013, 0.786]
    mark2at85K = [9.880, 11.874, 11.882, 12.952]
    mark2at80K = [6.002, 5.164, 5.028, 4.138]
    mark2at80Ktoggled = [3.826, 4.358, 4.153, 4.242]
    mark2at85Ktoggled = [10.499, 10.859, 11.091, 9.628]
    channels = [32, 16, 8, 4]
    channelsno8 = [32, 16, 4]
    colors = ['b', 'g', 'r', 'k']
    for i, d in enumerate([mark2at60K, mark2at60Ktoggled, mark2at80K, mark2at80Ktoggled]):
        if len(d) == 4:
            ch = channels
        elif len(d) == 3:
            ch = channelsno8
        plt.plot(ch, d, colors[i] + 'o-')
        m, b = linearfit(ch, d)
        x = numpy.array(range(0,ch[0]))
        y = m * x + b
        plt.plot(x, y, colors[i] + '--')
    plt.legend(["60K", "60K fit",\
                "60K VDDOP toggled", "60K toggled fit",\
                "80K", "80K fit",\
                "80K VDDOP toggled", "80K toggled fit"], loc = 2)
    plt.xlabel("# outputs")
    plt.ylabel("dark current (e-/s)")
    plt.ylim([0, 7])
    plt.title("Dark vs. outputs, mk2 M02812-10")
    plt.show()

def mark2VsDarkHighTemp():
    #Full list of # channels vs. Dark for both detectors.
    mark2at60K = [2.449, 1.773, 1.492, 0.949]
    mark2at85K = [9.880, 11.874, 11.882, 12.952]
    mark2at80K = [6.002, 5.164, 5.028, 4.138]
    mark2at90K = [29.163, 27.607, 27.369, 22.644]
    mark2at60Ktoggled = [0.932, 1.013, 0.786]
    mark2at80Ktoggled = [3.826, 4.358, 4.153, 4.242]
    mark2at85Ktoggled = [10.499, 10.859, 11.091, 9.628]
    mark2at90Ktoggled = [22.183, 26.566, 26.374, 27.034]
    channels = [32, 16, 8, 4]
    channelsno8 = [32, 16, 4]
    colors = ['b', 'g', 'r', 'k', 'c', 'y', 'm', 'b']
    for i, d in enumerate([mark2at80K, mark2at80Ktoggled, mark2at85K, mark2at85Ktoggled, mark2at90K,\
                           mark2at90Ktoggled]):
        if len(d) == 4:
            ch = channels
        elif len(d) == 3:
            ch = channelsno8
        plt.plot(ch, d, colors[i] + 'o-')
        m, b = linearfit(ch, d)
        x = numpy.array(range(0,ch[0]))
        y = m * x + b
        plt.plot(x, y, colors[i] + '--')
    plt.legend(["80K", "80K fit",\
                "80K VDDOP toggled", "80K toggled fit"\
                "85K", "85K fit",\
                "85K VDDOP toggled", "85K toggled fit"\
                "90K", "90K fit",\
                "90K VDDOP toggled", "90K toggled fit"], loc = 2)
    plt.xlabel("# outputs")
    plt.ylabel("dark current (e-/s)")
##    plt.ylim([0, 7])
    plt.title("Dark vs. outputs, mk2 M02815-12")
    plt.show()

def mark2DarkVsTemp():
    ch32 = [2.449, 6.002, 9.880,29.163]
    ch16 = [1.773, 5.164,11.874,27.607]
    ch8  = [1.492, 5.028,11.882,27.369]
    ch4  = [0.949, 4.138,12.952,22.644]
    colors = ['b', 'g', 'r', 'k', 'c', 'y', 'm', 'b']
    temps = [60, 80, 85, 90]
    for i, d in enumerate([ch32, ch16, ch8, ch4]):
        plt.plot(temps, d, colors[i] + 'o-')
        #m, b = linearfit(temps, d)
        #x = numpy.array(range(0,temps[0]))
        #y = m * x + b
        #plt.plot(x, y, colors[i] + '--')
    plt.legend(["32ch",\
                "16ch",\
                "8ch",\
                "4ch"], loc = 2)
    plt.xlabel("temperature (K)")
    plt.ylabel("dark current (e-/s)")
##    plt.ylim([0, 7])
    plt.xlim([55,95])
    plt.title("Dark vs. temperature, mk2 M02815-12")
    plt.show()

def mark2DarkVsTemp32Only():
    ch32 = [2.634, 4.294, 4.129, 9.867, 67.642, 441.654]
    temps = 1000. / numpy.array([60, 80, 85, 90, 100, 110])
    
    plt.semilogy(temps, ch32, 'o-')
    plt.title("Median dark current vs. temperature mk2 M02815-12")
    plt.ylabel("dark current (e-/s)")
    plt.xlabel("1000 / T (mK^-1)")
    plt.show()
    

def compareAvalancheGain24Feb2015():
    #Just put these two measurements on the same plot.
    VDD4V =  [0.88, 1.0, 1.28, 1.74, 2.49, 3.46, 5.41, 8.71, 13.79, 21.97, 35.09]
    VDD35V = [0.86, 1.0, 1.31, 1.88, 2.88, 4.41, 6.66, 9.44, 14.18, 22.63, 36.6]
    biases = [ 1.5, 2.5, 3.5,  4.5,  5.5,  6.5,  7.5,  8.5,  9.5,   10.5,  11.5]
    plt.semilogy(biases, VDD4V, 'o-')
    plt.semilogy(biases, VDD35V, 'o-')
    plt.xlabel("bias (V)")
    plt.ylabel("avalanche gain")
    plt.title("Avalanche gain for M02815-12")
    plt.legend(["VDD = 4V, 14 Apr 2014","VDD = 3.5V, 23 Feb 2015"], loc = 2)
    plt.ylim([0.7,40])
    plt.show()

def compareAvalancheGain6Oct2015():
    #Just put these two measurements on the same plot.
    T40K =   [0.72, 1.0, 1.45, 2.14, 2.96, 4.34, 8.45, 14.71, 27.16, 53.27]
    T45K =   [1.17, 1.0, 1.43, 2.13, 2.89, 4.61, 8.31, 14.78, 25.25, 43.90, 137.83]
    T50K =   [0.76, 1.0, 1.39, 1.99, 2.66, 4.43, 7.53, 13.12, 24.00, 44.14, 169.25]
    T55K =   [0.79, 1.0, 1.34, 2.20, 3.55, 5.32, 9.50, 15.68, 25.59, 47.33, 108.23, 155.74, 595.05]
    biases = [ 1.5, 2.5,  3.5,  4.5,  5.5,  6.5,  7.5,   8.5,   9.5,  10.5,   11.5,   12.5, 13.5]
    plt.semilogy(biases[:len(T40K)], T40K, 'o-')
    plt.semilogy(biases[:len(T45K)], T45K, 'o-')
    plt.semilogy(biases[:len(T50K)], T50K, 'o-')
    plt.semilogy(biases[:len(T55K)], T55K, 'o-')
    plt.xlabel("bias (V)")
    plt.ylabel("avalanche gain")
    plt.title("Avalanche gain for M06495-27")
    plt.legend(["40K","45K","50K","55K"], loc = 2)
    plt.ylim([0.7,600])
    plt.show()

def highTempDarkDrift26Feb2015():
    run1 = [156.254, 91.930, 64.441, 49.632, 40.072, 33.676]
    run2 = [153.903, 90.697, 64.026, 49.292, 39.990, 33.598, 28.992, 25.506, 22.665, 20.390]
    run3 = [153.615, 90.522, 63.523, 49.153, 39.774, 33.518, 28.818, 25.361, 22.556, 20.222]
    colors = ['b', 'g', 'r', 'k', 'c', 'y', 'm', 'b']
    for i, d in enumerate([run1, run2, run3]):
        plt.plot(range(len(d)), d, colors[i] + 'o-')
    plt.xlabel("ramp #")
    plt.ylabel("dark current (e-/s)")
    plt.legend(["series #1",\
                "series #2",\
                "series #3"], loc = 1)
    plt.title("Dark current curve through continuous series of ramps, M02815-12 T = 100 K")
    plt.show()

def darkPlots26Feb2015():
##    files = ["150226_135051.fits",
##             "150226_135227.fits",
##             "150226_135402.fits",
##             "150226_135537.fits",
##             "150226_135712.fits",
##             "150226_135848.fits",
##             "150226_140023.fits",
##             "150226_140158.fits",
##             "150226_140333.fits",
##             "150226_140508.fits"]
##    files = ["150226_135051.fits",
##             "150226_135227.fits",
##             "150226_135402.fits",
##             "150226_135537.fits",
##             "150226_135712.fits",
##             "150226_135848.fits",
##             "150226_140023.fits",
##             "150226_140158.fits",
##             "150226_140333.fits",
##             "150226_140508.fits"]
##    files = ["150226_162609.fits",
##             "150226_162750.fits",
##             "150226_162930.fits",
##             "150226_163110.fits",
##             "150226_163250.fits",
##             "150226_163431.fits",
##             "150226_163611.fits",
##             "150226_163751.fits",
##             "150226_163931.fits",
####             "150226_164111.fits"]
##    files = ["150227_104238.fits",
##             "150227_104425.fits",
##             "150227_104612.fits",
##             "150227_104758.fits",
##             "150227_104945.fits",
##             "150227_105132.fits",
##             "150227_105318.fits",
##             "150227_105505.fits",
##             "150227_105652.fits",
##             "150227_105838.fits"]
##    files = ["150227_132048.fits",
##             "150227_132220.fits",
##             "150227_132356.fits",
##             "150227_132533.fits",
##             "150227_133937.fits",
##             "150227_134114.fits",
##             "150227_134251.fits",
##             "150227_134427.fits"]
    files = ["150305_151154.fits",
             "150305_151335.fits",
             "150305_151515.fits",
             "150305_151656.fits",
             "150305_151836.fits",
             "150305_152016.fits"]
    for f in files:
        data = []
        d = openfits(f)
        x = numpy.median(d[-1,0:48,32:-32])
        for i in range(d.shape[0]):
            data.append(numpy.median(d[i,0:48,32:-32]))
        plt.plot(range(len(data)), data)
    plt.show()

def darkPlots6Mar2015():
##    files = ["150306_083323.fits",
##             "150306_083509.fits",
##             "150306_083654.fits",
##             "150306_083840.fits",
##             "150306_084025.fits",
##             "150306_084211.fits"]
    files = ["150306_124425.fits",
             "150306_124629.fits",
             "150306_132707.fits",
             "150306_140610.fits"]
    for f in files:
        data = []
        d = openfits(f)
        x = numpy.median(d[-1,0:48,32:-32])
        for i in range(d.shape[0]):
            data.append(numpy.median(d[i,0:48,32:-32]) - x)
        plt.plot(range(len(data)), data)
    plt.show()

def darkPlots2Mar2015():
    ch4 =   numpy.array([1.076, 2.188, 2.374, 4.954, 6.482, 71.507, 441.654])
    temps = numpy.array([   60,    75,    80,    85,    90,    100,     110])
    thousandOverT = 1000. / temps
    plt.semilogy(thousandOverT, ch4, 'o-')
    plt.ylabel("glow/dark current (e-/s")
    plt.xlabel("1000/T (mK^-1)")
    plt.title("glow/dark vs. temperature M02812-15")
    plt.show()

def darkPlots11Apr2015():
    masked =   numpy.array([0.020, 0.044, 0.111, 0.287, 0.819, 1.672, 7.282])
    unmasked = numpy.array([0.057, 0.067, 0.111, 0.218, 0.503, 1.259, 5.423])
    biases =   numpy.array([  2.5,   5.5,   7.5,   8.5,   9.5,  10.5,  11.5])
    plt.semilogy(biases, masked, 'o-')
    plt.semilogy(biases, unmasked, 'o-')
    plt.ylabel("dark current (e-/s)")
    plt.xlabel("bias voltage (V)")
    plt.title("dark (unmasked) vs. bias M04935-17")
    plt.legend(["masked","unmasked"], loc = 2)
    plt.show()

def fuckyou(filename):
    d = cdsCube(openfits(filename))
    means = []
    medians = []
    stds = []
    for y in range(d.shape[0]):
        for x in range(d.shape[1]):
            means.append(numpy.mean(d[2:,y,x]))
            medians.append(numpy.median(d[2:,y,x]))
            stds.append(numpy.std(d[2:,y,x], ddof = 1))
    plt.plot(medians, stds, 'o')
    plt.ylabel("standard deviation")
    plt.xlabel("median (CDS)")
    plt.show()

def multiRampCDS():
    d1 = openfits("150304_122034.fits")
    d2 = openfits("150304_123451.fits")
    avg1 = numpy.sum(d1[2:,:,:], axis = 0)
    avg2 = numpy.sum(d2[2:,:,:], axis = 0)
    cdsFrame = avg1 - avg2
    savefits("150304_CDSframe.fits", cdsFrame)

def avg5CDS(filename):
    d = openfits(filename)
    avg1 = numpy.sum(d[5:10,:,:], axis = 0)
    avg2 = numpy.sum(d[-5:,:,:], axis = 0)
    cdsFrame = avg1 - avg2
    savefits(filename[:-5] + "avg5CDS.fits", cdsFrame)
    
def lightSensitivityComparison():
    offlist = ["150306_155822.fits",
               "150306_154126.fits",
               "150308_151834.fits",
               "150306_155422.fits",
               "150308_141729.fits",
               "150308_143036.fits",
               "150308_152037.fits",
               "150308_151218.fits",
               "150308_142659.fits",
               "150308_143228.fits",
               "150308_152201.fits",
               "150308_151427.fits"]
    onlist  = ["150306_155901.fits",
               "150306_154222.fits",
               "150308_151938.fits",
               "150306_155528.fits",
               "150308_142617.fits",
               "150308_143138.fits",
               "150308_152114.fits",
               "150308_151326.fits",
               "150308_142744.fits",
               "150308_143322.fits",
               "150308_152239.fits",
               "150308_151517.fits"]
    colors = ['b', 'g', 'r', 'k']
    j = 0
    for offFile, onFile in map(None, offlist, onlist):
        offImage = openfits(offFile)
        onImage = openfits(onFile)
        data = []
        for i in range(offImage.shape[0]):
##            data.append(numpy.median(offImage[i,64:-64,96:-96]) - numpy.median(onImage[i,64:-64,96:-96]))
            data.append(numpy.median(onImage[i,64:-64,96:-96]))
        plt.plot(range(len(data)), data, colors[j] + 'o-')
        j += 1
        if j > 3:
            j = 0
    plt.legend(["Vdd = 3.5V, COMMON = 1V",
                "Vdd = 4V, COMMON = 1V",
                "Vdd = 4.5V, COMMON = 1.5V",
                "Vdd = 5V, COMMON = 2V"], loc = 1)
    plt.ylim([0,65000])
    plt.ylabel("off - on (median ADU)")
    plt.xlabel("time (s)")
    plt.title("Operating Voltages vs. Light Sensitivity T = 60K M02815-12 8 Mar 2015")
    plt.show()

def voltageGain10Mar2015():
    VDD5V = ["150310_160148.fits",
             "150310_160154.fits",
             "150310_160240.fits",
             "150310_160300.fits",
             "150310_160333.fits",
             "150310_160341.fits",
             "150310_160453.fits",
             "150310_160508.fits",
             "150310_160544.fits",
             "150310_160551.fits"]
    VDD45V =["150310_160824.fits",
             "150310_160831.fits",
             "150310_160917.fits",
             "150310_160919.fits",
             "150310_160944.fits",
             "150310_160946.fits",
             "150310_161053.fits",
             "150310_161054.fits",
             "150310_161121.fits",
             "150310_161122.fits"]
    VDD4V = ["150310_161239.fits",
             "150310_161240.fits",
             "150310_161318.fits",
             "150310_161319.fits",
             "150310_161352.fits",
             "150310_161356.fits",
             "150310_161422.fits",
             "150310_161425.fits",
             "150310_161450.fits",
             "150310_161456.fits"]
    VDD35V =["150310_161611.fits",
             "150310_161646.fits",
             "150310_161707.fits",
             "150310_161711.fits",
             "150310_161736.fits",
             "150310_161740.fits",
             "150310_161830.fits",
             "150310_161834.fits",
             "150310_161857.fits",
             "150310_161902.fits"]
    PRV = numpy.array([4.5, 4.5, 4.45, 4.45, 4.4, 4.4, 4.35, 4.35, 4.3, 4.3])
    lists = [VDD5V, VDD45V, VDD4V, VDD35V]
    for files in lists:
        meds = []
        test = openfits(files[-1])
        x = numpy.median(test[2:,64:-64,96:-96])
        for f in files:
            images = openfits(f)
            meds.append(numpy.median(images[2:,64:-64,96:-96]) - x)
        plt.plot(PRV, meds)
        m, b = linearfit(PRV, meds)
        print m
        print 1/(m / 1e6)
##        if PRV[0] > 3.5:
##            PRV = PRV - 0.5
    plt.show()

##    print "Using",stop,"frames"
##    cts = mean(mean(d))
##    cts = (numpy.median(data[5,:,:]) - numpy.median(data[stop,:,:])) * 2.89 / timing
##    cts = numpy.median(d)
##    noise = stddev(d.flatten())
##    noise = numpy.std(d, ddof = 1)
##    print cts / (stop - 5) #, "+/-", noise / (stop - 5), "e- / sec"

def avalancheMaps():
##    lightsoff = ["150330_101851.fits","150330_102005.fits","150330_102056.fits","150330_102149.fits",\
##                 "150330_102240.fits","150330_102328.fits","150330_102421.fits","150330_102518.fits",\
##                 "150330_102603.fits","150330_102747.fits","150330_102843.fits","150330_102949.fits"]
##    lightson =  ["150330_101858.fits","150330_102017.fits","150330_102111.fits","150330_102159.fits",\
##                 "150330_102251.fits","150330_102340.fits","150330_102432.fits","150330_102527.fits",\
##                 "150330_102617.fits","150330_102759.fits","150330_102852.fits","150330_103001.fits"]
##    bias =   [ 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5,10.5,11.5,12.5]
##    cutoff = 15000        
##    window = [[64, -64],[96,-96]]

    lightsoff = ["150406_111215.fits","150406_111304.fits","150406_111407.fits","150406_111502.fits",\
                 "150406_111627.fits","150406_111718.fits","150406_111802.fits","150406_111846.fits",\
                 "150406_111932.fits","150406_112015.fits","150406_112059.fits","150406_112142.fits",\
                 "150406_112221.fits"]
    lightson =  ["150406_111229.fits","150406_111316.fits","150406_111415.fits","150406_111513.fits",\
                 "150406_111640.fits","150406_111726.fits","150406_111814.fits","150406_111902.fits",\
                 "150406_111940.fits","150406_112025.fits","150406_112108.fits","150406_112151.fits",\
                 "150406_112235.fits"]
    bias =   [ 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5,10.5,11.5,12.5,13.5]
    timing = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
    cutoff = 35000        
    window = numpy.array([[92,-92],[64,-64]])
    unity = 1
    start = 5
    filename = "150411_AvalancheMaps_M04935-17.fits"
    
    i = 0
    results = numpy.zeros([len(lightsoff),256,320])
    for offFile, onFile in map(None, lightsoff, lightson):
        off = openfits(offFile)
        on = openfits(onFile)
##        gainMap = (numpy.median(on[1:11], axis = 0) - numpy.median(on[-10:], axis = 0)) -\
##                  (numpy.median(off[1:11], axis = 0) - numpy.median(off[-10:], axis = 0))
        endFrame = 0
        #Find where we hit saturation/non-linearity for these files.
        for n in range(on.shape[0]):
            if numpy.median(on[n,window[0,0]:window[0,1],window[1,0]:window[1,1]]) > cutoff:
                endFrame = n
        if endFrame % 2 == 0:
            endFrame -= 1
        n = endFrame
        print n
        gainMap = (numpy.median(on[1:n/2], axis = 0) - numpy.median(on[n/2:n], axis = 0)) -\
                  (numpy.median(off[1:n/2], axis = 0) - numpy.median(off[n/2:n], axis = 0))
        gainMap /= (n/2) - 1
        results[i,:,:] = gainMap
        i += 1
    results /= numpy.median(results[unity,window[0,0]:window[0,1],window[1,0]:window[1,1]])
    for i in range(results.shape[0]):
        print numpy.median(results[i,window[0,0]:window[0,1],window[1,0]:window[1,1]]), "+/-", numpy.std( results[i,window[0,0]:window[0,1],window[1,0]:window[1,1]], ddof = 1)
    savefits(filename, results)

def multiDarkMap(savefile, filenames):
    n = 1
    darkMap = numpy.zeros([256,320])
    for f in filenames:
        d = openfits(f)
        print f[:-5]
        e = numpy.zeros(d.shape[1:])
        for i in range(0,n):
            e += d[i + 5,:,:]
            e -= d[i - n,:,:]
        e /= n #Divide by number of averaged frames.
        e /= d.shape[0] - (5 + n) #Divide by number of seconds.
        e *= 2.89 #Multiply by charge gain.
        darkMap += e
    darkMap /= len(filenames)
    savefits(savefile,e)

def medianPlots8Apr2015():
    data = openfits("150407_174019.fits")
    windows = numpy.array([[[0,256],[0,320]],\
                           [[0,30], [160,320]],\
                           [[64,-64],[160,-64]]])
    for j in range(windows.shape[0]):
        m = []
        for i in range(data.shape[0]):
            m.append(numpy.median(data[i,windows[j,0,0]:windows[j,0,1],windows[j,1,0]:windows[j,1,1]]))
        plt.plot(range(data.shape[0]),m)
    plt.ylim([52800,54300])
    plt.legend(["overall","masked","unmasked"])
    plt.show()

def medianPlots9Apr2015():
    filenames = ["150408_170446.fits",\
                 "150409_010612.fits"]
    for f in filenames:
        data = openfits(f)
        windows = numpy.array([[[0,256],[0,320]],\
                               [[0,30], [160,320]],\
                               [[64,-64],[160,-64]]])
        for j in range(windows.shape[0]):
            m = []
            for i in range(data.shape[0]):
                m.append(numpy.median(data[i,windows[j,0,0]:windows[j,0,1],windows[j,1,0]:windows[j,1,1]]))
            plt.plot(range(data.shape[0]),m)
##    plt.ylim([52800,54300])
    plt.legend(["overall #1","masked #1","unmasked #1","overall #2","masked #2","unmasked #2"])
    plt.show()

def medianPlots10Apr2015():
    filenames = ["150409_170728.fits",\
                 "150409_210836.fits"]
    for f in filenames:
        data = openfits(f)
        windows = numpy.array([[[0,256],[0,320]],\
                               [[0,30], [160,320]],\
                               [[64,-64],[160,-64]]])
        for j in range(windows.shape[0]):
            m = []
            for i in range(data.shape[0]):
                m.append(numpy.median(data[i,windows[j,0,0]:windows[j,0,1],windows[j,1,0]:windows[j,1,1]]))
            plt.plot(range(data.shape[0]),m)
##    plt.ylim([52800,54300])
    plt.legend(["overall #1","masked #1","unmasked #1","overall #2","masked #2","unmasked #2"])
    plt.show()

def medianPlots10Apr2015v2():
    filenames = ["150410_112312.fits",\
                 "150410_115405.fits"]
    for f in filenames:
        data = openfits(f)
        windows = numpy.array([[[0,256],[0,320]],\
                               [[0,30], [160,320]],\
                               [[64,-64],[160,-64]]])
        for j in range(windows.shape[0]):
            m = []
            for i in range(data.shape[0]):
                m.append(numpy.median(data[i,windows[j,0,0]:windows[j,0,1],windows[j,1,0]:windows[j,1,1]]))
            plt.plot(range(data.shape[0]),m)
##    plt.ylim([52800,54300])
    plt.legend(["overall #1","masked #1","unmasked #1","overall #2","masked #2","unmasked #2"])
    plt.show()

def avalancheGains11Apr2015():
    bias = [ 1.5,  2.5,  3.5,  4.5,  5.5,  6.5,  7.5,  8.5,   9.5,  10.5,  11.5,  12.5,   13.5]
    gain = [1.04, 1.00, 1.31, 1.81, 2.65, 3.56, 5.56, 8.13, 13.27, 21.35, 34.34, 58.79, 101.89]
    std  = [0.11, 0.14, 0.19, 0.27, 0.39, 0.57, 0.86, 1.33,  2.00,  3.32,  5.55,  9.82,  18.61]
##    fig, axs = plt.subplots(nrows = 1, ncols = 1)
##    ax = axs[0,0]
    plt.errorbar(bias, gain, fmt = 'o-', yerr = std)
    ax = plt.subplot(111)
    ax.set_yscale("log")
    for b, g in zip(bias,gain):
        ax.annotate('{}'.format(int(g * 100) / 100.),xy=(b,g),xytext = (-5,5), ha = 'right', textcoords = 'offset points')
    plt.ylabel("Avalanche Gain")
    plt.xlabel("Bias (V)")
    plt.show()

def analyzeDark(filename, timing = 1, start = 5, end = -5, gain = 1.0, n = 20, saveFile = False):
    d = openfits(filename)
    if end < 0:
        end = d.shape[0] + end
    print filename[:-5]
    e = numpy.zeros(d.shape[1:])
    if n * 2 + start > end or end > d.shape[0]:
        return "Exposure not long enough for specified parameters, quitting."
    for i in range(0,n):
        e += d[i + start,:,:]
        e -= d[i - n + end,:,:]
    e /= n
    e /= end - (start + n)
    e /= timing
    e /= gain
    e *= 2.89
    medians = []
    for i in range(d.shape[0]):
        medians.append(numpy.median(d[i,:,:]))
    plt.plot(range(d.shape[0]),medians)
    plt.plot([start,start],[numpy.min(medians),numpy.max(medians)])
    plt.plot([end,end],[numpy.min(medians),numpy.max(medians)])
    plt.xlim([start - 5, end + 5])
    plt.show()
    dummy = raw_input("Good?")
    print "Overall dark current:", numpy.median(e), "e-/s +/-", numpy.std(e, ddof = 1), "e-/s"
    print "Upper right strip (masked):", numpy.median(e[-32:,160:]), "e-/s +/-", numpy.std(e[-32:,160:], ddof = 1), "e-/s"
    print "Center (unmasked):", numpy.median(e[64:-64,92:-92]), "e-/s +/-", numpy.std(e[64:-64,92:-92], ddof = 1), "e-/s"
    if saveFile:
        savefits(filename[:-5] + "dark_gainAdjusted.fits",e)
    
def screenhot(indata, r = 1, debug = False):
    #Screens for hot pixels. A larger r increases the resilience to large blobs (cosmic rays),
    #but also the chance that exceptionally sharp stars will have their centers missing.
    if debug:
        print "Screening for hot pixels..."
        t = time()
    threshhold = 3 * r
    data = numpy.array(indata)
    for y in range(r, data.shape[0] - r):
        for x in range(r, data.shape[1] - r):
            #If it's ten times more than the pixel to the left
            if abs(data[y,x]) > threshhold * abs(data[y,x-r]):
                #Check if it's also ten times more than the pixel above.
##                if abs(data[y,x]) > 10 * abs(data[y,x+r]) and abs(data[y,x]) > 10 * abs(data[y-r,x]) and abs(data[y,x]) > 10 * abs(data[y+r,x]):
                if abs(data[y,x]) > threshhold * abs(data[y-r,x]):
                    #If so, blend it out.
##                    data[y,x] = mean([data[y,x-r],data[y,x+r],data[y-r,x],data[y+r,x]])
                    data[y,x] = numpy.median(data[y-r:y+r+1,x-r:x+r+1])
    if debug:
        print "Screened, time elapsed",time() - t,"seconds."
    return data
    
def convertToPhotons(filename, y, x):
    d = openfits(filename)
    results = []
    rMin = 5
    rMax = 25
    gain = 20.
    coadded = numpy.zeros([d.shape[1],d.shape[2]])
    for n in range(d.shape[0]):
##        coadded += screenhot(d[n,:,:])
        coadded += d[n,:,:]
    coadded -= background(coadded, zeroScreen = False)
##    savefits("testy.fits",coadded)
    for r in range(rMin, rMax):
##        trimmed = numpy.zeros([d.shape[0],r*2+1,r*2+1])
##        for n in range(d.shape[0]):
##        trimmed[n,:,:] = circtrim(d[n,:,:], r, y, x)
##        trimmed[n,:,:] = annulussubtract(d[n,:,:], r, r*2, r*3, y, x)
        trimmed = numpy.zeros([r*2+1,r*2+1])
        trimmed[:,:] = circtrim(coadded[:,:], r, y, x)
        sumADU = numpy.sum(trimmed)
        sumPhotons = sumADU * 2.89 / (gain)
        photonsPerSec = (sumPhotons / d.shape[0]) * 100.
        results.append(photonsPerSec)
        
        print ".",
##    savefits("testy.fits",trimmed)
    print "."
    plt.plot(range(rMin,rMax),results,'o-')
    plt.show()
    print numpy.median(photonsPerSec), "photons/sec"

def darkCurrents1May2015():
    bias = [ 1.5,  2.5,  3.5,  4.5,  5.5,  6.5,  7.5,  8.5,   9.5,  10.5,  11.5,  12.5,   13.5]
    gain = [1.04, 1.00, 1.31, 1.81, 2.65, 3.56, 5.56, 8.13, 13.27, 21.35, 34.34, 58.79, 101.89]
    std  = [0.11, 0.14, 0.19, 0.27, 0.39, 0.57, 0.86, 1.33,  2.00,  3.32,  5.55,  9.82,  18.61]
##    fig, axs = plt.subplots(nrows = 1, ncols = 1)
##    ax = axs[0,0]
    plt.errorbar(bias, gain, fmt = 'o-', yerr = std)
    ax = plt.subplot(111)
    ax.set_yscale("log")
    for b, g in zip(bias,gain):
        ax.annotate('{}'.format(int(g * 100) / 100.),xy=(b,g),xytext = (-5,5), ha = 'right', textcoords = 'offset points')
    plt.ylabel("Avalanche Gain")
    plt.xlabel("Bias (V)")
    plt.show()

def darkCurrents4May2015():
    #Let's do some comparative histograms.
    files = ["",""]

def freqResponseSmallWindow(filename, title = "Frequency Response", averages = 512):
    d = openfits(filename)
    averaged = avgCube(d, avg = averages)
    cds = cdsCube(averaged)
    print cds.shape
    clean = removeSpikes(cds, threshhold = 20, debug = True)
##    clean = cds
    meaned = numpy.mean(numpy.mean(clean, axis = 2), axis = 1)
    t = numpy.array(range(len(meaned)), dtype = float)
##    t *= 0.0475 * averages
    t *= 0.0244 * averages
    plt.plot(t, meaned)
    plt.title(title)
    plt.ylabel("delta ADU")
    plt.xlabel("ms")
    plt.show()

def freqResponseSmallWindow2Traces(filenames, title = "Frequency Response", averages = 512):
    for f in filenames:
        d = openfits(f)
        averaged = avgCube(d, avg = averages)
        cds = cdsCube(averaged)
        print cds.shape
        clean = removeSpikes(cds, threshhold = 20, debug = True)
    ##    clean = cds
        meaned = numpy.mean(numpy.mean(clean, axis = 2), axis = 1)
        t = numpy.array(range(len(meaned)), dtype = float)
    ##    t *= 0.0475 * averages
        t *= 0.0244 * averages
        plt.plot(t, meaned)
    plt.title(title)
    plt.ylabel("delta ADU")
    plt.xlabel("ms")
    plt.show()

def freqResponseStacked(filename, title = "Frequency Response", averages = 512):
    d = openfits(filename)
    averaged = avgCube(d, avg = averages)
    cds = cdsCube(averaged)
    print cds.shape
    clean = removeSpikes(cds, threshhold = 100, debug = True)
##    clean = cds
    meaned = numpy.mean(numpy.mean(clean, axis = 2), axis = 1)
    t = numpy.array(range(len(meaned)), dtype = float)
    t *= 0.0475 * averages
    plt.plot(t, meaned)
    plt.title(title)
    plt.ylabel("delta ADU")
    plt.xlabel("ms")
    plt.show()

def freqResponseSmallWindowDivided(filename, title = "Frequency Response", averages = 512):
    d = openfits(filename)
    #Do the reference subtraction of the 4 pixels on the right side.
    for n in range(d.shape[0]):
        for y in range(d.shape[1]):
            for x in range(32):
                d[n,y,x] -= d[n,y,x + 32]
        if n % 10000 == 0:
            print ".",
    print "."
    d = d[:,:,:32]
    averaged = avgCube(d, avg = averages)
    cds = cdsCube(averaged)
    clean = removeSpikes(cds, threshhold = 20, debug = True)
    plt.plot(numpy.mean(numpy.mean(clean, axis = 2), axis = 1))
    plt.title(title)
    plt.show()
    
def freqResponseSmallWindowMaskEdge(filename, title = "Frequency Response", averages = 512):
    d = openfits(filename)
    #Do the reference subtraction of the 4 pixels on the right side.
    for n in range(d.shape[0]):
        d[n,:,:] -= numpy.mean(d[n,:,-2:])
        if n % 10000 == 0:
            print ".",
    print "."
    d = d[:,:,:30]
    averaged = avgCube(d, avg = averages)
    cds = cdsCube(averaged)
    clean = removeSpikes(cds, threshhold = 20, debug = True)
    plt.plot(numpy.mean(numpy.mean(clean, axis = 2), axis = 1))
    plt.title(title)
    plt.show()

def freqResponse160x1(filename, title = "Frequency Response", averages = 64):
    d = openfits(filename)
    #Do the reference subtraction of the dark pixels.
    for i in range(1,d.shape[0]):
        d[i,:,:32] -= d[i - 1,:,-32:]
        if i % 10000 == 0:
            print ".",
    print "."
    d = d[1:,:,:32]
    averaged = avgCube(d, avg = averages)
    cds = cdsCube(averaged)
    clean = removeSpikes(cds, threshhold = 50, debug = True)
    t = numpy.array(range(clean.shape[0]), dtype = float)
    t *= 0.05409 * averages
    plt.plot(numpy.mean(clean, axis = 2))
    plt.title(title + " avg:" + str(averages))
    plt.xlabel("ms")
    plt.show()

def freqResponseLog(filename, title = "Frequency Response", averages = 32):
    d = openfits(filename)
    averaged = avgCube(d, avg = averages)
    cds = cdsCube(averaged)
    print cds.shape
    clean = removeSpikes(cds, threshhold = 20, debug = True)
##    clean = cds
    meaned = numpy.mean(numpy.mean(clean, axis = 2), axis = 1)
    t = numpy.array(range(len(meaned)), dtype = float)
##    t *= 0.0475 * averages
    t *= 0.0244 * averages
    #Phase stack.
##    period = 0
##    i = 0
##    while period == 0:
##        if t[i] > 4000.:
##            period = i
##        i += 1
##    del i
##    for cyc in range(1, len(meaned) / period, period):
##        print cyc
##        meaned[:period] += meaned[cyc * period:(cyc + 1) * period]
##    meaned = meaned[:period]
##    t = t[:period]
    logged = numpy.zeros(meaned.shape)
    for i in range(logged.shape[0]):
        logged[i] = numpy.log(abs(meaned[i]))
##    plt.plot(t, abs(meaned))
    print t.shape
    print logged.shape
    plt.plot(t, logged)
    plt.title(title)
    plt.ylabel("delta ADU")
    plt.xlabel("ms")
    plt.show()

def fitFreqResponse(filename, startTime, endTime, averages = 32):
    d = openfits(filename)
    averaged = avgCube(d, avg = averages)
    cds = cdsCube(averaged)
    clean = removeSpikes(cds, threshhold = 20, debug = True)
    meaned = numpy.mean(numpy.mean(clean, axis = 2), axis = 1)
    t = numpy.array(range(len(meaned)), dtype = float)
##    t *= 0.0475 * averages
    t *= 0.0244 * averages
    logged = numpy.zeros(meaned.shape)
    for i in range(logged.shape[0]):
        logged[i] = numpy.log(abs(meaned[i]))
    start = 0
    while t[start] < startTime:
        start += 1
    end = start
    while t[end] < endTime:
        end += 1
    m,b = linearfit(t[start:end], logged[start:end])
    print "m:", m
    print "1/m:", 1/m
    #Now we need to turn this into a half-life.
    halfValue = (e ** (logged[start])) / 2
    half = start
    while (e ** logged[half]) > halfValue:
        half += 1
    print "Half-life:", t[half] - t[start]
    


    

##def freqResponseMark3T60K():
##    filenames = ["150513_152459.fits","150513_152527.fits","150513_152608.fits","150513_152657.fits",\
##                 "150513_152714.fits","150513_152932.fits","150513_153038.fits","150513_153110.fits"]
##    thresh = [5, 10, 10, 30, 50, 70, 70, 70]
##    spikethreshhold = 5
##    for f,t in map(None,filenames,thresh):
##        d = openfits(f)
##        averaged = avgCube(d, avg = 512)
##        cds = cdsCube(averaged)
##        clean = removeSpikes(cds, threshhold = spikethreshhold, debug = True, delete = False, byFrame = True)
####        clean = cds
##        meaned = numpy.mean(numpy.mean(clean, axis = 2), axis = 1)
##        for n in range(meaned.shape[0]):
##            if meaned[n] > 0:
##                meaned[n] = 0
####        for n in range(1,meaned.shape[0] - 1):
####            upper = meaned[n] - meaned[n+1]
####            lower = meaned[n] - meaned[n-1]
####            if upper > t * 2 \
####               and lower > t * 2:
####                meaned[n] = (meaned[n - 1] + meaned[n + 1]) / 2.
##        jump = 0
##        delta = 2
##        while meaned[jump + delta] - meaned[jump] < t:
##            jump += 1
##            if jump + delta >= meaned.shape[0]:
##                print "Oh, poop."
##                plt.clf()
##                plt.plot(meaned,'o-')
##                plt.show()
##                return
##        plt.plot(range(-jump, meaned.shape[0] - jump), meaned)
##    plt.title("Frequency Response 60K M02775-35")
##    plt.show()

##def freqResponseMark3T85K():
##    filenames = ["150514_151125.fits","150514_151502.fits","150514_151546.fits","150514_151659.fits",\
##                 "150514_151753.fits","150514_151856.fits","150514_152048.fits","150514_152227.fits"]
##    thresh = [5, 10, 10, 30, 50, 70, 70, 70]
##    spikethreshhold = 5
##    for f,t in map(None,filenames,thresh):
##        d = openfits(f)
##        averaged = avgCube(d, avg = 512)
##        cds = cdsCube(averaged)
##        clean = removeSpikes(cds, threshhold = spikethreshhold, debug = True, delete = False, byFrame = True)
####        clean = cds
##        meaned = numpy.mean(numpy.mean(clean, axis = 2), axis = 1)
##        for n in range(meaned.shape[0]):
##            if meaned[n] > 0:
##                meaned[n] = 0
####        for n in range(1,meaned.shape[0] - 1):
####            upper = meaned[n] - meaned[n+1]
####            lower = meaned[n] - meaned[n-1]
####            if upper > t * 2 \
####               and lower > t * 2:
####                meaned[n] = (meaned[n - 1] + meaned[n + 1]) / 2.
##        jump = 0
##        delta = 2
##        while meaned[jump + delta] - meaned[jump] < t:
##            jump += 1
##            if jump + delta >= meaned.shape[0]:
##                print "Oh, poop."
##                plt.clf()
##                plt.plot(meaned,'o-')
##                plt.show()
##                return
##        plt.plot(range(-jump, meaned.shape[0] - jump), meaned)
##    plt.title("Frequency Response 85K M02775-35")
##    plt.show()

def freqResponseMark3T85KvsBias():
    filenames = ["150514_151433.fits","150514_162147.fits","150514_161151.fits","150514_162635.fits",\
                 "150514_161659.fits"]
    thresh = [5, 10, 100, 300, 500]
    spikethreshhold = 5
    for f,t in map(None,filenames,thresh):
        d = openfits(f)
        averaged = avgCube(d, avg = 512)
        cds = cdsCube(averaged)
        clean = removeSpikes(cds, threshhold = spikethreshhold, debug = True, delete = False, byFrame = True)
##        clean = cds
        meaned = numpy.mean(numpy.mean(clean, axis = 2), axis = 1)
        for n in range(meaned.shape[0]):
            if meaned[n] > 0:
                meaned[n] = 0
##        for n in range(1,meaned.shape[0] - 1):
##            upper = meaned[n] - meaned[n+1]
##            lower = meaned[n] - meaned[n-1]
##            if upper > t * 2 \
##               and lower > t * 2:
##                meaned[n] = (meaned[n - 1] + meaned[n + 1]) / 2.
        jump = 0
        delta = 2
        while meaned[jump + delta] - meaned[jump] < t:
            jump += 1
            if jump + delta >= meaned.shape[0]:
                print "Oh, poop."
                plt.clf()
                plt.plot(meaned,'o-')
                plt.show()
                return
        plt.plot(range(-jump, meaned.shape[0] - jump), meaned)
    plt.title("Frequency Response 85K M02775-35")
    plt.show()

##def freqResponseMark10T60K():
##    filenames = ["150519_090958.fits","150519_091056.fits","150519_091131.fits","150519_091226.fits",\
##                 "150519_091323.fits","150519_091400.fits","150519_091450.fits","150519_091511.fits"]
##    thresh = [3, 5, 10, 30, 50, 70, 70, 70]
##    spikethreshhold = 5
##    for f,t in map(None,filenames,thresh):
##        d = openfits(f)
##        averaged = avgCube(d, avg = 512)
##        cds = cdsCube(averaged)
##        clean = removeSpikes(cds, threshhold = spikethreshhold, debug = True, delete = False, byFrame = True)
####        clean = cds
##        meaned = numpy.mean(numpy.mean(clean, axis = 2), axis = 1)
##        for n in range(meaned.shape[0]):
##            if meaned[n] > 0:
##                meaned[n] = 0
####        for n in range(1,meaned.shape[0] - 1):
####            upper = meaned[n] - meaned[n+1]
####            lower = meaned[n] - meaned[n-1]
####            if upper > t * 2 \
####               and lower > t * 2:
####                meaned[n] = (meaned[n - 1] + meaned[n + 1]) / 2.
##        jump = 0
##        delta = 2
##        while meaned[jump + delta] - meaned[jump] < t:
##            jump += 1
##            if jump + delta >= meaned.shape[0]:
##                print "Oh, poop."
##                plt.clf()
##                plt.plot(meaned,'o-')
##                plt.show()
##                return
##        plt.plot(range(-jump, meaned.shape[0] - jump), meaned)
##    plt.title("Frequency Response 60K M04935-17")
##    plt.show()

def freqResponseVsLight(mark = 3, T = 60, averages = 512, vertAlign = False):
    if mark == 10:
        if T == 60:
            filenames = ["150519_090958.fits","150519_091056.fits","150519_091131.fits","150519_091226.fits",\
                         "150519_091323.fits","150519_091400.fits","150519_091450.fits","150519_091511.fits"]
            thresh = [3, 5, 10, 30, 50, 70, 70, 70]
            measurementDate = "19 May 2015"
            legends = ["0.8V","0.9V","1.0V","1.1V","1.2V","1.3V","1.4V","1.5V"]
            vsString = "Light Level"
        elif T == 85:
            filenames = ["150521_143428.fits","150521_143504.fits","150521_143549.fits","150521_143630.fits",\
                         "150521_143733.fits","150521_143820.fits","150521_143853.fits","150521_143925.fits"]
            thresh = [5, 10, 10, 30, 50, 70, 70, 70]
            measurementDate = "21 May 2015"
            legends = ["0.8V","0.9V","1.0V","1.1V","1.2V","1.3V","1.4V","1.5V"]
            vsString = "Light Level"
        else:
            print "Invalid temperature specified."
    elif mark == 3:
        if T == 60:
            filenames = ["150513_152459.fits","150513_152527.fits","150513_152608.fits","150513_152657.fits",\
                         "150513_152714.fits","150513_152932.fits","150513_153038.fits","150513_153110.fits"]
            thresh = [3, 10, 10, 30, 50, 70, 70, 70]
            measurementDate = "13 May 2015"
            legends = ["0.8V","0.9V","1.0V","1.1V","1.2V","1.3V","1.4V","1.5V"]
            vsString = "Light Level"
        elif T == 85:
            filenames = ["150514_151125.fits","150514_151502.fits","150514_151546.fits","150514_151659.fits",\
                         "150514_151753.fits","150514_151856.fits","150514_152048.fits","150514_152227.fits"]
            thresh = [5, 10, 10, 30, 50, 70, 70, 70]
            measurementDate = "14 May 2015"
            legends = ["0.8V","0.9V","1.0V","1.1V","1.2V","1.3V","1.4V","1.5V"]
            vsString = "Light Level"
        else:
            print "Invalid temperature specified."
    elif mark == 0:
        #Do a grab bag of them: both detectors, both temperatures.
        filenames = ["150513_152608.fits","150514_151546.fits","150519_091131.fits","150521_143549.fits"]
        thresh = [10, 10, 10, 10]
        measurementDate = "13-21 May 2015"
        legends = ["mk3 60K","mk3 85K","mk10 60K","mk10 85K"]
        vsString = "Detector & Temperature"
        T = 0 #We're doing all temperatures.
    else:
        print "Invalid detector mark # specified."
    correction = averages / 512.
    spikethreshhold = 5
    #Calculate frames-to-time conversion for x-axis.
    avg2ms = 3. * 1000. * averages / (64.e3)
    for f,t in map(None,filenames,thresh):
        t *= correction
        d = openfits(f)
        averaged = avgCube(d, avg = averages)
        cds = cdsCube(averaged)
        clean = removeSpikes(cds, threshhold = spikethreshhold, debug = True, delete = False, byFrame = True)
##        clean = cds
        meaned = numpy.mean(numpy.mean(clean, axis = 2), axis = 1)
        for n in range(meaned.shape[0]):
            if meaned[n] > 0:
                meaned[n] = 0
        jump = 0
        delta = 5
##        while -(meaned[jump] - meaned[jump + delta]) < t or -(meaned[jump + 1] - meaned[jump + delta + 1]) < t:
##            jump += 1
##            if jump + delta >= meaned.shape[0]:
##                print "Oh, poop."
##                plt.clf()
##                plt.plot(meaned,'o-')
##                plt.show()
##                return
        #Let's try an alternate means of horizontal location.
        sampleWidth = int(2050. / avg2ms)
        delta = int(sampleWidth / 10.)
        jump = int(sampleWidth) #Starting position.
        meaned *= (-2.9) * 1000. / avg2ms
        while delta > 1:
            here = numpy.sum(meaned[jump: jump + sampleWidth])
            up = numpy.sum(meaned[jump + delta: jump + delta + sampleWidth])
            down = numpy.sum(meaned[jump - delta: jump - delta + sampleWidth])
            if here > up and here > down:
                delta /= 2
            elif up > down:
                jump += delta
            elif down > up:
                jump -= delta
            if jump < 1:
                print "something dumb happened"
                plt.clf()
                return
        if vertAlign:
            meaned -= numpy.mean(meaned[jump + sampleWidth / 10:jump + sampleWidth * 9 / 10])
        xaxis = numpy.array(range(-jump, meaned.shape[0] - jump))
        plt.plot(xaxis * avg2ms, meaned)
    titleString = "Frequency Response vs. " + vsString + ": "
    if T > 0:
        titleString += str(T) + "K "
    if mark == 3:
        titleString += "M02775-35 mk3 "
    elif mark == 10:
        titleString += "M04935-17 mk10 "
    titleString += measurementDate
    plt.title(titleString)
    plt.legend(legends)
    plt.xlabel("ms")
    plt.ylabel("e-/sec")
    plt.show()

def freqResponseVsTemp(averages = 32, vertAlign = False):
    filenames = ["150928_115440.fits","151002_085634.fits","151001_140352.fits","151001_084640.fits"]
    thresh = 200
    measurementDate = "2 Oct 2015"
    legends = ["60K","65K","70K","75K"]
    correction = averages / 512.
    spikethreshhold = 8
    #Calculate frames-to-time conversion for x-axis.
    avg2ms = 3. * 1000. * averages / (64.e3)
    for f in filenames:
        thresh *= correction
        d = openfits(f)
        averaged = avgCube(d, avg = averages)
        cds = cdsCube(averaged)
        clean = removeSpikes(cds, threshhold = spikethreshhold, debug = True, delete = False, byFrame = True)
##        clean = cds
        meaned = numpy.mean(numpy.mean(clean, axis = 2), axis = 1)
        for n in range(meaned.shape[0]):
            if meaned[n] > 0:
                meaned[n] = 0
        jump = 0
        delta = 10
##        while -(meaned[jump] - meaned[jump + delta]) < t or -(meaned[jump + 1] - meaned[jump + delta + 1]) < t:
##            jump += 1
##            if jump + delta >= meaned.shape[0]:
##                print "Oh, poop."
##                plt.clf()
##                plt.plot(meaned,'o-')
##                plt.show()
##                return
        #Let's try an alternate means of horizontal location.
        sampleWidth = int(2050. / avg2ms)
        delta = int(sampleWidth / 10.)
        jump = int(sampleWidth) #Starting position.
        meaned *= (-2.9) * 1000. / avg2ms
        while delta > 1:
            here = numpy.sum(meaned[jump: jump + sampleWidth])
            up = numpy.sum(meaned[jump + delta: jump + delta + sampleWidth])
            down = numpy.sum(meaned[jump - delta: jump - delta + sampleWidth])
            if here > up and here > down:
                delta /= 2
            elif up > down:
                jump += delta
            elif down > up:
                jump -= delta
            if jump < 1:
                print "something dumb happened"
                plt.clf()
                return
        if vertAlign:
            meaned -= numpy.mean(meaned[jump + sampleWidth / 10:jump + sampleWidth * 9 / 10])
        xaxis = numpy.array(range(-jump, meaned.shape[0] - jump))
        plt.plot(xaxis * avg2ms, meaned)
    titleString = "Frequency Response vs. Temperature M06495-19 "
    titleString += measurementDate
    plt.title(titleString)
    plt.legend(legends)
    plt.xlabel("ms")
    plt.ylabel("e-/sec")
    plt.show()

def avalancheGains10Jun2015():
    bias = [  1.5,  2.5,  3.5,  4.5,  5.5,  6.5,  7.5,  8.5,   9.5,  10.5,  11.5,  12.5]
    gains =[[0.88, 1.00, 1.21, 1.58, 2.18, 3.31, 5.14, 8.18, 13.06, 20.72, 32.66, 51.03],#M02775-10
            [0.76, 1.00, 1.30, 1.65, 2.49, 3.67, 5.72, 8.48, 14.40, 24.15, 40.01, 72.96],#M02775-35
            [0.72, 1.00, 1.29, 1.76, 2.58, 3.40, 5.35, 8.65, 11.49, 20.23, 34.16, 54.71]]#M04935-17
##    fig, axs = plt.subplots(nrows = 1, ncols = 1)
##    ax = axs[0,0]
    ax = plt.subplot(111)
    ax.set_yscale("log")
    for gain in gains:
        plt.plot(bias,gain,'o-')
    plt.ylabel("Avalanche Gain")
    plt.xlabel("Bias (V)")
    plt.legend(["M02775-10","M02775-35","M04935-17"],loc = 2)
    plt.title("Avalanche Gain vs. Bias")
    plt.show()

def avalancheGains28Sep2015():
    bias = [ 2.5,  3.5,  4.5,  5.5,  6.5,  7.5,  8.5,   9.5,  10.5,  11.5]
    gains =[[1.00, 1.21, 1.58, 2.18, 3.31, 5.14, 8.18, 13.06, 20.72, 32.66],#M02775-10
            [1.00, 1.30, 1.65, 2.49, 3.67, 5.72, 8.48, 14.40, 24.15, 40.01],#M02775-35
            [1.00, 1.29, 1.76, 2.58, 3.40, 5.35, 8.65, 11.49, 20.23, 34.16],#M04935-17
            [1.00, 1.29, 1.51, 2.82, 4.02, 5.64,10.49, 12.08, 19.52, 41.21]]#M06495-19##
    fig, axs = plt.subplots(nrows = 1, ncols = 1)
##    ax = axs[0,0]
    ax = plt.subplot(111)
    ax.set_yscale("log")
    for gain in gains:
        plt.plot(bias,gain,'o-')
    plt.ylabel("Avalanche Gain")
    plt.xlabel("Bias (V)")
    plt.legend(["M02775-10","M02775-35","M04935-17","M06495-19"],loc = 2)
    plt.title("Avalanche Gain vs. Bias")
    plt.show()



def avalancheGains10Oct2015():
    bias = [ 2.5,  3.5,  4.5,  5.5,  6.5,  7.5,  8.5,   9.5,  10.5,  11.5]
    gains =[[1.00, 1.21, 1.58, 2.18, 3.31, 5.14, 8.18, 13.06, 20.72, 32.66],#M02775-10
            [1.00, 1.30, 1.65, 2.49, 3.67, 5.72, 8.48, 14.40, 24.15, 40.01],#M02775-35
            [1.00, 1.29, 1.76, 2.58, 3.40, 5.35, 8.65, 11.49, 20.23, 34.16],#M04935-17
            [1.00, 1.29, 1.51, 2.82, 4.02, 5.64,10.49, 12.08, 19.52, 41.21],#M06495-19
            [1.00, 1.34, 2.20, 3.55, 5.32, 9.50,15.68, 25.59, 47.33, 108.23]]#M06495-27
##    fig, axs = plt.subplots(nrows = 1, ncols = 1)
##    ax = axs[0,0]
    ax = plt.subplot(111)
    ax.set_yscale("log")
    for gain in gains:
        plt.plot(bias,gain,'o-')
    plt.ylabel("Avalanche Gain")
    plt.xlabel("Bias (V)")
    plt.legend(["M02775-10","M02775-35","M04935-17","M06495-19","M06495-27"],loc = 2)
    plt.title("Avalanche Gain vs. Bias")
    plt.show()

def avalancheGains19Oct2015():
    bias = [ 2.5,  3.5,  4.5,  5.5,  6.5,  7.5,  8.5,   9.5,  10.5,  11.5]
    gains =[[1.00, 1.21, 1.58, 2.18, 3.31, 5.14, 8.18, 13.06, 20.72, 32.66],#M02775-10
            [1.00, 1.30, 1.65, 2.49, 3.67, 5.72, 8.48, 14.40, 24.15, 40.01],#M02775-35
            [1.00, 1.29, 1.76, 2.58, 3.40, 5.35, 8.65, 11.49, 20.23, 34.16],#M04935-17
            [1.00, 1.29, 1.51, 2.82, 4.02, 5.64,10.49, 12.08, 19.52, 41.21],#M06495-19
            [1.00, 1.34, 2.20, 3.55, 5.32, 9.50,15.68, 25.59, 47.33, 108.23],#M06495-27
            [1.00, 1.32, 1.65, 2.02, 2.96, 4.56, 6.56, 9.81,  14.94, 22.90]]#M06665-12
##    fig, axs = plt.subplots(nrows = 1, ncols = 1)
##    ax = axs[0,0]
    ax = plt.subplot(111)
    ax.set_yscale("log")
    for gain in gains:
        plt.plot(bias,gain,'o-')
    plt.ylabel("Avalanche Gain")
    plt.xlabel("Bias (V)")
    plt.legend(["M02775-10","M02775-35","M04935-17","M06495-19","M06495-27","M06665-12"],loc = 2)
    plt.title("Avalanche Gain vs. Bias")
    plt.show()

def avalancheGains26Oct2015():
    bias = [ 2.5,  3.5,  4.5,  5.5,  6.5,  7.5,  8.5,   9.5,  10.5,  11.5]
    gains =[[1.00, 1.21, 1.58, 2.18, 3.31, 5.14, 8.18, 13.06, 20.72, 32.66],#M02775-10
            [1.00, 1.30, 1.65, 2.49, 3.67, 5.72, 8.48, 14.40, 24.15, 40.01],#M02775-35
            [1.00, 1.29, 1.76, 2.58, 3.40, 5.35, 8.65, 11.49, 20.23, 34.16],#M04935-17
            [1.00, 1.29, 1.51, 2.82, 4.02, 5.64,10.49, 12.08, 19.52, 41.21],#M06495-19
            [1.00, 1.34, 2.20, 3.55, 5.32, 9.50,15.68, 25.59, 47.33, 108.23],#M06495-27
            [1.00, 1.32, 1.65, 2.02, 2.96, 4.56, 6.56, 9.81,  14.94, 22.90],#M06665-12
            [1.00, 1.12, 1.46, 2.00, 2.96, 3.48, 5.21, 7.63,  12.46, 18.97]]#M06715-27
##    fig, axs = plt.subplots(nrows = 1, ncols = 1)
##    ax = axs[0,0]
    ax = plt.subplot(111)
    ax.set_yscale("log")
    ax.tick_params(labelright = True)
    for gain in gains:
        plt.plot(bias,gain,'o-')
    plt.ylabel("Avalanche Gain")
    plt.xlabel("Bias (V)")
    plt.legend(["Mark 3 M02775-10",
                "Mark 3 M02775-35",
                "Mark 5 M04935-17",
                "Mark 12 M06495-19",
                "Mark 12 M06495-27",
                "Mark 13 M06665-12",
                "Mark 14 M06715-27"],loc = 2)
    plt.title("Avalanche Gain vs. Bias")
    plt.show()

def avalancheGains30Nov2015():
    bias = [ 2.5,  3.5,  4.5,  5.5,  6.5,  7.5,  8.5,   9.5,  10.5,  11.5]
    gains =[[1.00, 1.21, 1.58, 2.18, 3.31, 5.14, 8.18, 13.06, 20.72, 32.66],#M02775-10
            [1.00, 1.30, 1.65, 2.49, 3.67, 5.72, 8.48, 14.40, 24.15, 40.01],#M02775-35
            [1.00, 1.29, 1.76, 2.58, 3.40, 5.35, 8.65, 11.49, 20.23, 34.16],#M04935-17
            [1.00, 1.29, 1.51, 2.82, 4.02, 5.64,10.49, 12.08, 19.52, 41.21],#M06495-19
            [1.00, 1.34, 2.20, 3.55, 5.32, 9.50,15.68, 25.59, 47.33, 108.23],#M06495-27
            [1.00, 1.32, 1.65, 2.02, 2.96, 4.56, 6.56, 9.81,  14.94, 22.90],#M06665-12
            [1.00, 1.12, 1.46, 2.00, 2.96, 3.48, 5.21, 7.63,  12.46, 18.97],#M06715-27
            [1.00, 1.35, 1.70, 2.46, 3.41, 4.81, 7.29, 10.56, 16.16, 24.41]]#M06715-34
    
##    fig, axs = plt.subplots(nrows = 1, ncols = 1)
##    ax = axs[0,0]
    ax = plt.subplot(111)
    ax.set_yscale("log")
    ax.tick_params(labelright = True)
    for gain in gains:
        plt.plot(bias,gain,'o-')
    plt.ylabel("Avalanche Gain")
    plt.xlabel("Bias (V)")
    plt.legend(["Mark 3 M02775-10",
                "Mark 3 M02775-35",
                "Mark 5 M04935-17",
                "Mark 12 M06495-19",
                "Mark 12 M06495-27",
                "Mark 13 M06665-12",
                "Mark 14 M06715-27"],loc = 2)
    plt.title("Avalanche Gain vs. Bias")
    plt.show()

def avalancheGains3Dec2015():
    bias = [ 2.5,  3.5,  4.5,  5.5,  6.5,  7.5,  8.5,   9.5,  10.5,  11.5]
    gains =[[1.00, 1.28, 1.59, 2.28, 2.84, 3.97, 6.02, 8.80,  13.74, 23.36],#M06665-03
            [1.00, 1.32, 1.65, 2.02, 2.96, 4.56, 6.56, 9.81,  14.94, 22.90],#M06665-12
            [1.00, 1.12, 1.46, 2.00, 2.96, 3.48, 5.21, 7.63,  12.46, 18.97],#M06715-27
            [1.00, 1.16, 1.60, 1.96, 2.38, 3.59, 4.53, 7.51,  10.75, 18.65],#M06715-29
            [1.00, 1.35, 1.70, 2.46, 3.41, 4.81, 7.29, 10.56, 16.16, 24.41]]#M06715-34
    
##    fig, axs = plt.subplots(nrows = 1, ncols = 1)
##    ax = axs[0,0]
    ax = plt.subplot(111)
    ax.set_yscale("log")
    ax.tick_params(labelright = True)
    for gain in gains:
        plt.plot(bias,gain,'o-')
    plt.ylabel("Avalanche Gain")
    plt.xlabel("Bias (V)")
    plt.legend(["Mark 13 M06665-03",
                "Mark 13 M06665-12",
                "Mark 14 M06715-27",
                "Mark 14 M06715-29",
                "Mark 14 M06715-34"],loc = 2)
    plt.title("Avalanche Gain vs. Bias")
    plt.show()

def avalancheGains18Mar2015():
    bias = [ 2.5,  3.5,  4.5,  5.5,  6.5,  7.5,  8.5,  9.5,   10.5,  11.5,  12.5,  13.5]
    gains =[[1.00, 1.31, 1.64, 2.15, 2.80, 5.46, 8.39, 14.23, 22.10, 34.70, 55.03, 88.70],#M02775-35
            [1.00, 1.12, 1.46, 2.00, 2.96, 3.48, 5.21, 7.63,  12.46, 18.97, 28.67, 47.17]]#M06715-27
    
##    fig, axs = plt.subplots(nrows = 1, ncols = 1)
##    ax = axs[0,0]
    ax = plt.subplot(111)
    ax.set_yscale("log")
    ax.tick_params(labelright = True)
    for gain in gains:
        plt.plot(bias,gain,'o-')
    plt.ylabel("Avalanche Gain")
    plt.xlabel("Bias (V)")
    plt.legend(["Mark 3 M02755-35",
                "Mark 14 M06715-27"],loc = 2)
    plt.title("Avalanche Gain vs. Bias")
    plt.xlim([0,14])
    plt.show()

def avalancheGains3Nov2016():
    bias = [ 2.5,  3.5,  4.5,  5.5,  6.5,  7.5,  8.5,   9.5,  10.5,  11.5]
    gains =[[1.00, 1.28, 1.59, 2.28, 2.84, 3.97, 6.02, 8.80,  13.74, 23.36],#M06665-03
            [1.00, 1.32, 1.65, 2.02, 2.96, 4.56, 6.56, 9.81,  14.94, 22.90],#M06665-12
            [1.00, 1.12, 1.46, 2.00, 2.96, 3.48, 5.21, 7.63,  12.46, 18.97],#M06715-27
            [1.00, 1.16, 1.60, 1.96, 2.38, 3.59, 4.53, 7.51,  10.75, 18.65],#M06715-29
            [1.00, 1.35, 1.70, 2.46, 3.41, 4.81, 7.29, 10.56, 16.16, 24.41],#M06715-34
            [1.00, 1.55, 1.74, 2.31, 3.32, 4.22, 5.33, 8.67,  12.35, 18.34]]#M06665-23
    
##    fig, axs = plt.subplots(nrows = 1, ncols = 1)
##    ax = axs[0,0]
    ax = plt.subplot(111)
    ax.set_yscale("log")
    ax.tick_params(labelright = True)
    for gain in gains:
        plt.plot(bias,gain,'o-')
    plt.ylabel("Avalanche Gain")
    plt.xlabel("Bias (V)")
    plt.legend(["Mark 13 M06665-03",
                "Mark 13 M06665-12",
                "Mark 14 M06715-27",
                "Mark 14 M06715-29",
                "Mark 14 M06715-34",
                "Mark 13 M06665-23"],loc = 2)
    plt.title("Avalanche Gain vs. Bias")
    plt.show()
    
def cornerGlowvsBias18Jun2015():
    bias = [  2.5,  4.5,  6.5,  8.5,  10.5,  11.5]
    gain =  [1.00, 1.65, 3.67, 8.48, 24.15, 40.01] #M02775-35
    files = ["150611_180439.fits", #2.5V
             "150616_124138.fits", #4.5V
             "150616_155602.fits", #6.5V
             "150618_171801.fits", #8.5V
             "150617_113427.fits", #10.5V
             "150617_135703.fits"] #11.5V
    timing = [10, 10, 10, 5, 5, 5]
    for i, f in enumerate(files):
        d = openfits(f)
        means = []
        zero = numpy.mean(d[0,:5,:5])
        for n in range(d.shape[0]):
            means.append((zero - numpy.mean(d[n,:5,:5])) / gain[i])
        plt.plot(range(0,d.shape[0] * timing[i], timing[i]),means)
    legends = []
    for b in bias:
        legends.append(str(b) + "V")
    plt.legend(legends, loc = 2)
    plt.xlabel("time (s)")
    plt.ylabel("ADU")
    plt.title("Lower left corner glow vs. bias")
    plt.show()
        
def cornerGlowvsBias19Jun2015():
    files = ["150619_095902.fits", 
             "150619_102701.fits", 
             "150619_105442.fits", 
             "150619_112208.fits", 
             "150619_131345.fits"] 
    timing = [1, 4, 16, 4, 1]
    for i, f in enumerate(files):
        d = openfits(f)
        means = []
        zero = numpy.mean(d[0,:5,:5])
        for n in range(d.shape[0]):
            means.append(zero - numpy.mean(d[n,:5,:5]))
        plt.plot(range(0,d.shape[0] * timing[i], timing[i]),means)
    legends = []
    for t in timing:
        legends.append(str(t) + "s")
    plt.legend(legends, loc = 2)
    plt.xlabel("time (s)")
    plt.ylabel("ADU")
    plt.title("Lower left corner glow vs. frame interval")
    plt.show()

def compareMedians(f1,f2):
    d1 = openfits(f1)
    d2 = openfits(f2)
    medians1 = []
    medians2 = []
    for f in [f1,f2]:
        d = openfits(f)
        medians = []
        for i in range(d1.shape[0]):
            medians.append(numpy.median(d[i,:,:]))
        plt.plot(medians)
    plt.show()

def findBestRows():
    #Let's find the best position to do photon counting in!
    f = "150624_darkMap_M02775-35_85V.fits"
    d = openfits(f)
    #Crop to the right side.
    d = d[:,160:]
    rowInfo = []
    threshhold = 2
    dtype = [('row', int),('hotpix', int)]
    rowList = []
    for y in range(d.shape[0]):
        count = 0
        for x in range(d.shape[1]):
            if d[y,x] > threshhold:
                count += 1
        rowList.append((y,count))
    rowList = numpy.array(rowList, dtype = dtype)
    print rowList
    sortedList = numpy.sort(rowList, order = 'hotpix')
    print "Best 10 rows:"
    for i in range(10):
        print sortedList[i]
    

def filterCommonNoise(filename):
    data = openfits(filename)
    print "Filtering common noise."
    for i in range(data.shape[0]):
        for y in range(data.shape[1]):
            for x in range(data.shape[2]/32):
                common = numpy.mean(numpy.sort(data[i,y,x * 32:(x + 1) * 32])[-4:])
                data[i,y,x * 32:(x + 1) * 32] -= common
    savefits(filename[:-5]+"filtered.fits", data)

def samplePixelPlot(filename, delta = 1, avg = 1, rolling = False, detrend = False):
    data = openfits(filename)
    cds = numpy.zeros([data.shape[0] - delta,data.shape[1],data.shape[2]])
    print "Performing correlated dual sampling, delta =",delta,"."
    for i in range(cds.shape[0]):
        cds[i,:,:] = data[i,:,:] - data[i + delta,:,:]
        if detrend:
            cds[i,:,:] -= numpy.mean(cds[i,:,:].flatten())
    data = cds
    data = avgCube(data, avg = avg, rolling = rolling)
    
    for x in range(16):
        plt.plot(data[:,0,x])
    plt.show()

def simplePhotonCounting(histrange = [0,30], binspacing = 0.5, avg = 1, rollingAvg = True, delta = 1):
    filename = "150625_131920filtered.fits" #COMMON = -4.5V, mk3 M02775-35, 32ch, masked, 03 E0 14x00 84 17x00
    data = openfits(filename)
    #Do the frame averaging.
    if avg > 1:
        print "Averaging by",avg,
        if not rollingAvg:
            print "exclusively."
            averaged = numpy.zeros([data.shape[0] / avg,data.shape[1],data.shape[2]])
            for i in range(averaged.shape[0]):
                averaged[i,:,:] = numpy.mean(data[i * avg:(i + 1) * avg,:,:], axis = 0)
        else:
            print "inclusively (rolling)."
            averaged = numpy.zeros([data.shape[0] - avg,data.shape[1],data.shape[2]])
            for i in range(averaged.shape[0]):
                averaged[i,:,:] = numpy.mean(data[i:i + avg,:,:], axis = 0)
            if delta < avg:
                print "Increasing delta to accomodate rolling average."
                delta = avg
        data = averaged
    #Now we do the CDS.
    cds = numpy.zeros([data.shape[0] - delta,data.shape[1],data.shape[2]])
    print "Performing correlated dual sampling, delta =",delta,"."
    for i in range(cds.shape[0]):
        cds[i,:,:] = data[i,:,:] - data[i + delta,:,:]
    data = cds
    #Show some trends.
##    for x in range(16):
##        plt.plot(data[:,0,x])
##    plt.show()
    #Compute bins.
    print "Plotting histogram."
    bins = (histrange[1] - histrange[0]) / binspacing
##    dtype = [('row', int),('column', int),('std', float)]
##    stddevs = []
##    for y in range(data.shape[1]):
##        for x in range(data.shape[2]):
##            stddev = numpy.std(data[:,y,x],ddof = 1)
##            print "Standard deviation for",y,",",x,":", stddev
##            stddevs.append((y,x,stddev))
##    stddevs = numpy.array(stddevs, dtype = dtype)
##    sortedStddevs = numpy.sort(stddevs, order = 'std')
##    mask = numpy.zeros([data.shape[1],data.shape[2]])
##    for pix in sortedStddevs[:32]:
##        mask[pix[0],pix[1]] = 1
##    for n in range(data.shape[0]):
##        data[n,:,:] *= mask
    plt.hist(data.flatten(), bins = bins, range=histrange)
    plt.legend(filename)
    plt.show()

def darkTrends26Jun2015():
    files = ["150624_darkMap_M02775-35_25V.fits",
             "150624_darkMap_M02775-35_35V.fits",
             "150624_darkMap_M02775-35_45V.fits",
             "150624_darkMap_M02775-35_55V.fits",
             "150624_darkMap_M02775-35_60V.fits",
             "150624_darkMap_M02775-35_65V.fits",
             "150624_darkMap_M02775-35_70V.fits",
             "150624_darkMap_M02775-35_75V.fits",
             "150624_darkMap_M02775-35_80V.fits",
             "150624_darkMap_M02775-35_85V.fits",
             "150624_darkMap_M02775-35_90V.fits",
             "150624_darkMap_M02775-35_95V.fits",
             "150624_darkMap_M02775-35_100V.fits",
             "150624_darkMap_M02775-35_105V.fits",
             "150624_darkMap_M02775-35_110V.fits",
             "150624_darkMap_M02775-35_115V.fits",
             "150629_100144darkMap.fits",
             "150629_102204darkMap.fits"]
    biases = [2.5, 3.5, 4.5, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.5, 13.5]
    medians = []
    for f in files:
        d = openfits(f)
        medians.append(numpy.median(d[-32:,160:192]) * 2.89)
    plt.semilogy(biases[:-2], medians[:-2], 'bo-')
    plt.semilogy(biases[-3:], medians[-3:], 'bo--')
    plt.xlabel("Bias Voltage (V)")
    plt.ylabel("e-/s")
    plt.title("Median Dark Current (no gain correction) vs. Bias")
    plt.show()
        
def darkTrends30Jun2015():
    files = ["150624_darkMap_M02775-35_25V.fits",
             "150624_darkMap_M02775-35_35V.fits",
             "150624_darkMap_M02775-35_45V.fits",
             "150624_darkMap_M02775-35_55V.fits",
             "150624_darkMap_M02775-35_60V.fits",
             "150624_darkMap_M02775-35_65V.fits",
             "150624_darkMap_M02775-35_70V.fits",
             "150624_darkMap_M02775-35_75V.fits",
             "150624_darkMap_M02775-35_80V.fits",
             "150624_darkMap_M02775-35_85V.fits",
             "150624_darkMap_M02775-35_90V.fits",
             "150624_darkMap_M02775-35_95V.fits",
             "150624_darkMap_M02775-35_100V.fits",
             "150624_darkMap_M02775-35_105V.fits"]
##             "150624_darkMap_M02775-35_110V.fits",
##             "150624_darkMap_M02775-35_115V.fits",
##             "150629_100144darkMap.fits",
##             "150629_102204darkMap.fits"]
    biases = [2.5, 3.5, 4.5, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5]#, 11.0, 11.5, 12.5, 13.5]
    percentage = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    colors = ['b','r','g','k','m','y','c']
    for i,p in enumerate(percentage):
        medians = []
        for f in files:
            d = openfits(f)
            sortedList = numpy.sort(d[-32:,160:192].flatten())
            dark = sortedList[int(p * len(sortedList))]
            print dark,
            medians.append(dark * 2.89)
##            medians.append(numpy.median(d[-32:,160:192]))
        plt.semilogy(biases, medians, colors[i] + 'o-')
##        plt.semilogy(biases[:-2], medians[:-2], colors[i] + 'o-')
##        plt.semilogy(biases[-3:], medians[-3:], colors[i] + 'o--')
    plt.xlabel("Bias Voltage (V)")
    plt.ylabel("e-/s")
    plt.legend(["5%","10%","25%","50%","75%","90%","95%"], loc = 2)
    plt.title("Median Dark Current (no gain correction) vs. Bias")
    plt.xlim([0,11])
    plt.show()

def darkTrends1Jul2015():
    files = ["150624_darkMap_M02775-35_25V.fits",
             "150624_darkMap_M02775-35_35V.fits",
             "150624_darkMap_M02775-35_45V.fits",
             "150624_darkMap_M02775-35_55V.fits",
             "150624_darkMap_M02775-35_60V.fits",
             "150624_darkMap_M02775-35_65V.fits",
             "150624_darkMap_M02775-35_70V.fits",
             "150624_darkMap_M02775-35_75V.fits",
             "150624_darkMap_M02775-35_80V.fits",
             "150624_darkMap_M02775-35_85V.fits",
             "150624_darkMap_M02775-35_90V.fits",
             "150624_darkMap_M02775-35_95V.fits",
             "150624_darkMap_M02775-35_100V.fits",
             "150624_darkMap_M02775-35_105V.fits"]
##             "150624_darkMap_M02775-35_110V.fits",
##             "150624_darkMap_M02775-35_115V.fits",
##             "150629_100144darkMap.fits",
##             "150629_102204darkMap.fits"]
    biases = [2.5, 3.5, 4.5, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5]#, 11.0, 11.5, 12.5, 13.5]
    colors = ['b','r','g','k','m','y','c']
    windows = [[[224,256],[160,192]], #Top center. (standard)
               [[224,256],[288,320]], #Top right.
               [[0,32],[288,320]], #Bottom right.
               [[178,210],[216,248]]] #Upper right corner of glow aperture.
    for i,w in enumerate(windows):
        medians = []
        for f in files:
            d = openfits(f)
            dark = numpy.median(d[w[0][0]:w[0][1],w[1][0]:w[1][1]])
            medians.append(dark * 2.89)
        plt.semilogy(biases, medians, colors[i] + 'o-')
    plt.xlabel("Bias Voltage (V)")
    plt.ylabel("e-/s")
    plt.legend(["Top Center","Top Right","Bottom Right","Glow-Illuminated"], loc = 2)
    plt.title("Median Dark Current (no gain correction) vs. Bias")
    plt.xlim([0,11])
    plt.ylim([0.002,3])
    plt.show()

def voltageGain1Jul2015():
    files = ["150701_150505.fits",
             "150701_151027.fits",
             "150701_150745.fits",
             "150701_150914.fits"]
    voltage = [0.0, 0.1, 0.2, 0.3]
    gain = []
    for x in range(8):
        medians = []
        for f in files:
            d = openfits(f)
            med = numpy.median(d[:,:,x])
            medians.append(med * 2.89)
        plt.plot(voltage,medians)
        m,b = linearfit(voltage, medians)
        gain.append(1e6/m)
        print "Voltage Gain:", (1e6)/m, "uV/ADU"
    print "Mean Voltage Gain:", numpy.mean(gain), "uV/ADU"
    plt.show()

def avalancheGainInvestigation1Jul2015():
    lightsoff1 = ["150629_153156.fits","150629_153249.fits","150629_151305.fits","150629_151348.fits",\
                 "150629_151422.fits","150629_151502.fits","150629_151548.fits","150629_151637.fits",\
                 "150629_151742.fits","150629_151829.fits","150629_151917.fits","150629_145916.fits","150629_150008.fits"]
    lightson1 =  ["150629_153217.fits","150629_153259.fits","150629_151316.fits","150629_151357.fits",\
                 "150629_151433.fits","150629_151512.fits","150629_151600.fits","150629_151648.fits",\
                 "150629_151754.fits","150629_151840.fits","150629_151928.fits","150629_145935.fits","150629_150020.fits"]
    bias =   [ 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5,10.5,11.5,12.5,13.5]
    #title = "Optical gain for SAPHIRA device M02775-35 (VDD = 3.5V) 29 Jun 2015"

    lightsoff2 = ["150330_101851.fits","150330_102005.fits","150330_102056.fits","150330_102149.fits",\
                 "150330_102240.fits","150330_102328.fits","150330_102421.fits","150330_102518.fits",\
                 "150330_102603.fits","150330_102747.fits","150330_102843.fits","150330_102949.fits"]
    lightson2 =  ["150330_101858.fits","150330_102017.fits","150330_102111.fits","150330_102159.fits",\
                 "150330_102251.fits","150330_102340.fits","150330_102432.fits","150330_102527.fits",\
                 "150330_102617.fits","150330_102759.fits","150330_102852.fits","150330_103001.fits"]
    #title = "Optical gain for SAPHIRA device M02775-35 (VDD = 3.5V) 30 Mar 2015"

    for i in range(len(lightsoff2)):
        d1 = openfits(lightson1[i])
        d2 = openfits(lightson2[i])
        medians1 = []
        medians2 = []
        for n in range(d1.shape[0]):
            medians1.append(numpy.median(d1[n,64:-64,92:-92]))
        for n in range(d2.shape[0]):
            medians2.append(numpy.median(d2[n,64:-64,92:-92]))
        plt.plot(range(len(medians1)),medians1 - medians1[0])
        plt.plot(range(len(medians2)),medians2 - medians2[0])
        plt.show()

def darkTrends5Jul2015():
    fileslist = [["150624_darkMap_M02775-35_25V.fits",
              "150624_darkMap_M02775-35_35V.fits",
              "150624_darkMap_M02775-35_45V.fits",
              "150624_darkMap_M02775-35_55V.fits",
              "150624_darkMap_M02775-35_60V.fits",
              "150624_darkMap_M02775-35_65V.fits",
              "150624_darkMap_M02775-35_70V.fits",
              "150624_darkMap_M02775-35_75V.fits",
              "150624_darkMap_M02775-35_80V.fits",
              "150624_darkMap_M02775-35_85V.fits",
              "150624_darkMap_M02775-35_90V.fits",
              "150624_darkMap_M02775-35_95V.fits",
              "150624_darkMap_M02775-35_100V.fits",
              "150624_darkMap_M02775-35_105V.fits"],
             ["150704_darkMap_M02775-35_25V_70K.fits",
              "150704_darkMap_M02775-35_45V_70K.fits",
              "150704_darkMap_M02775-35_60V_70K.fits",
              "150704_darkMap_M02775-35_70V_70K.fits",
              "150704_darkMap_M02775-35_80V_70K.fits",
              "150704_darkMap_M02775-35_90V_70K.fits",
              "150704_darkMap_M02775-35_100V_70K.fits"]]
    biases = [[2.5, 3.5, 4.5, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5],
              [2.5, 4.5, 6.0, 7.0, 8.0, 9.0, 10.0]]
    colors = ['b','r','g','k','m','y','c']
    w = [[224,256],[160,192]] #Top center. (standard)
    for b,files in map(None, biases, fileslist):
        medians = []
        for f in files:
            print "file:", f
            d = openfits(f)
            dark = numpy.median(d[w[0][0]:w[0][1],w[1][0]:w[1][1]])
            medians.append(dark * 2.89)
        plt.semilogy(b, medians, 'o-')
    plt.xlabel("Bias Voltage (V)")
    plt.ylabel("e-/s")
    plt.legend(["60K","70K"], loc = 2)
    plt.title("Median Dark Current (no gain correction) vs. Bias")
    plt.xlim([0,11])
    plt.ylim([0.004,5])
    plt.show()

def darkTrends6Jul2015():
    files = ["150704_darkMap_M02775-35_25V_70K.fits",
              "150704_darkMap_M02775-35_45V_70K.fits",
              "150704_darkMap_M02775-35_60V_70K.fits",
              "150704_darkMap_M02775-35_70V_70K.fits",
              "150704_darkMap_M02775-35_80V_70K.fits",
              "150704_darkMap_M02775-35_90V_70K.fits",
              "150704_darkMap_M02775-35_100V_70K.fits"]
    biases = [2.5, 4.5, 6.0, 7.0, 8.0, 9.0, 10.0]#, 11.0, 11.5, 12.5, 13.5]
    percentage = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    colors = ['b','r','g','k','m','y','c']
    for i,p in enumerate(percentage):
        medians = []
        for f in files:
            d = openfits(f)
            sortedList = numpy.sort(d[-32:,160:192].flatten())
            dark = sortedList[int(p * len(sortedList))]
            print dark,
            medians.append(dark * 2.89)
##            medians.append(numpy.median(d[-32:,160:192]))
        plt.semilogy(biases, medians, colors[i] + 'o-')
##        plt.semilogy(biases[:-2], medians[:-2], colors[i] + 'o-')
##        plt.semilogy(biases[-3:], medians[-3:], colors[i] + 'o--')
    plt.xlabel("Bias Voltage (V)")
    plt.ylabel("e-/s")
    plt.legend(["5%","10%","25%","50%","75%","90%","95%"], loc = 2)
    plt.title("Median Dark Current (no gain correction) vs. Bias")
    plt.xlim([0,11])
    plt.show()

def darkTrends10Jul2015():
    files = ["150708_darkMap_M02775-35_25V_50K.fits",
             "150708_darkMap_M02775-35_45V_50K.fits",
             "150708_darkMap_M02775-35_60V_50K.fits",
             "150708_darkMap_M02775-35_70V_50K.fits",
             "150708_darkMap_M02775-35_80V_50K.fits",
             "150708_darkMap_M02775-35_90V_50K.fits",
             "150708_darkMap_M02775-35_100V_50K.fits"]
    biases = [2.5, 4.5, 6.0, 7.0, 8.0, 9.0, 10.0]#, 11.0, 11.5, 12.5, 13.5]
    percentage = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    colors = ['b','r','g','k','m','y','c']
    for i,p in enumerate(percentage):
        medians = []
        for f in files:
            d = openfits(f)
            sortedList = numpy.sort(d[-32:,160:192].flatten())
            dark = sortedList[int(p * len(sortedList))]
            print dark,
            medians.append(dark * 2.89)
##            medians.append(numpy.median(d[-32:,160:192]))
        plt.semilogy(biases, medians, colors[i] + 'o-')
##        plt.semilogy(biases[:-2], medians[:-2], colors[i] + 'o-')
##        plt.semilogy(biases[-3:], medians[-3:], colors[i] + 'o--')
    plt.xlabel("Bias Voltage (V)")
    plt.ylabel("e-/s")
    plt.legend(["5%","10%","25%","50%","75%","90%","95%"], loc = 2)
    plt.title("Median Dark Current (no gain correction) vs. Bias")
    plt.xlim([0,11])
    plt.show()

def cosmeticTrends7Oct2015():
    files = ["151007_103936.fits"]
    percentage = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    colors = ['b','r','g','k','m','y','c']
    for i,p in enumerate(percentage):
        medians = []
        for f in files:
            d = openfits(f)
            for n in range(d.shape[0]):
                sortedList = numpy.sort(d[n,:,:].flatten())
                dark = sortedList[int(p * len(sortedList))]
                medians.append(dark * 2.89)
        plt.plot(medians, colors[i] + 'o-')
    plt.xlabel("Frame #")
    plt.ylabel("ADU")
    plt.legend(["5%","10%","25%","50%","75%","90%","95%"], loc = 1)
    plt.title("Median ADU vs. Frame #")
##    plt.xlim([0,11])
    plt.show()

def plot5pixels(filename):
    d = openfits(filename)
    d = avgCube(d, avg = 64)
    d = cdsCube(d)
    for i in range(8):
        plt.plot(d[:,0,i])
    plt.show()


def darkTrends9Jul2015():
    fileslist = [["150624_darkMap_M02775-35_25V.fits",
                  "150624_darkMap_M02775-35_35V.fits",
                  "150624_darkMap_M02775-35_45V.fits",
                  "150624_darkMap_M02775-35_55V.fits",
                  "150624_darkMap_M02775-35_60V.fits",
                  "150711_darkMap_M02775-35_60V_60K.fits",
                  "150624_darkMap_M02775-35_65V.fits",
                  "150711_darkMap_M02775-35_65V_60K.fits",
                  "150624_darkMap_M02775-35_70V.fits",
                  "150624_darkMap_M02775-35_75V.fits",
                  "150711_darkMap_M02775-35_75V_60K.fits",
                  "150624_darkMap_M02775-35_80V.fits",
                  "150624_darkMap_M02775-35_85V.fits",
                  "150624_darkMap_M02775-35_90V.fits",
                  "150624_darkMap_M02775-35_95V.fits",
                  "150624_darkMap_M02775-35_100V.fits",
                  "150624_darkMap_M02775-35_105V.fits"],
                 ["150704_darkMap_M02775-35_25V_70K.fits",
                  "150704_darkMap_M02775-35_45V_70K.fits",
                  "150704_darkMap_M02775-35_60V_70K.fits",
                  "150704_darkMap_M02775-35_70V_70K.fits",
                  "150704_darkMap_M02775-35_80V_70K.fits",
                  "150704_darkMap_M02775-35_90V_70K.fits",
                  "150704_darkMap_M02775-35_100V_70K.fits"],
                 ["150706_darkMap_M02775-35_25V_80K.fits",
                  "150706_darkMap_M02775-35_45V_80K.fits",
                  "150706_darkMap_M02775-35_80V_80K.fits"],
                 ["150708_darkMap_M02775-35_25V_50K.fits",
                  "150708_darkMap_M02775-35_45V_50K.fits",
                  "150708_darkMap_M02775-35_60V_50K.fits",
                  "150710_darkMap_M02775-35_60V_50K.fits",
                  "150708_darkMap_M02775-35_70V_50K.fits",
                  "150710_darkMap_M02775-35_75V_50K.fits",
                  "150708_darkMap_M02775-35_80V_50K.fits",
                  "150708_darkMap_M02775-35_90V_50K.fits",
                  "150708_darkMap_M02775-35_100V_50K.fits"]]
    biases = [[2.5, 3.5, 4.5, 5.5, 6.0, 6.0, 6.5, 6.5, 7.0, 7.5, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5],
              [2.5, 4.5, 6.0, 7.0, 8.0, 9.0, 10.0],
              [2.5, 4.5, 8.0],
              [2.5, 4.5, 6.0, 6.0, 7.0, 7.5, 8.0, 9.0, 10.0]]
    colors = ['b','r','g','k','m','y','c']
    w = [[224,256],[160,192]] #Top center. (standard)
    for b,files in map(None, biases, fileslist):
        medians = []
        for f in files:
            print "file:", f
            d = openfits(f)
            dark = numpy.median(d[w[0][0]:w[0][1],w[1][0]:w[1][1]])
            medians.append(dark * 2.89)
        plt.semilogy(b, medians, 'o')
    plt.xlabel("Bias Voltage (V)")
    plt.ylabel("e-/s")
    plt.legend(["60K","70K","80K","50K"], loc = 2)
    plt.title("Median Dark Current (no gain correction) vs. Bias")
    plt.xlim([0,11])
    plt.ylim([0.01,15])
    plt.show()

def darkAllRamps13Jul2015():
    fileslist =[["150611_163202darkMap.fits",
             "150611_170254darkMap.fits",
             "150611_173346darkMap.fits",
             "150611_180439darkMap.fits"],
            ["150707_102057darkMap.fits",
             "150707_103654darkMap.fits",
             "150707_105252darkMap.fits",
             "150707_110849darkMap.fits",
             "150707_112452darkMap.fits",
             "150707_114049darkMap.fits",
             "150707_115647darkMap.fits",
             "150707_121244darkMap.fits"]]
    timing =[30, 15]
    w = [[224,256],[160,192]] #Top center. (standard)
    for t,files in map(None, timing, fileslist):
        medians = []
        for f in files:
            print "file:", f
            d = openfits(f)
            dark = numpy.median(d[w[0][0]:w[0][1],w[1][0]:w[1][1]])
            medians.append(dark * 2.89)
        plt.semilogy(range(t, t * (len(medians) + 1), t), medians, 'o-')
    plt.xlabel("Time (min)")
    plt.ylabel("e-/s")
    plt.legend(["60K","50K"])
    plt.title("Median Dark Current (no gain correction) vs. Ramp #")
    plt.show()

def darkTrends15Jul2015():
    fileslist = [["150624_darkMap_M02775-35_25V.fits",
                  "150624_darkMap_M02775-35_35V.fits",
                  "150624_darkMap_M02775-35_45V.fits",
                  "150624_darkMap_M02775-35_55V.fits",
                  "150624_darkMap_M02775-35_60V.fits",
                  "150711_darkMap_M02775-35_60V_60K.fits",
                  "150624_darkMap_M02775-35_65V.fits",
                  "150711_darkMap_M02775-35_65V_60K.fits",
                  "150624_darkMap_M02775-35_70V.fits",
                  "150624_darkMap_M02775-35_75V.fits",
                  "150711_darkMap_M02775-35_75V_60K.fits",
                  "150624_darkMap_M02775-35_80V.fits",
                  "150624_darkMap_M02775-35_85V.fits",
                  "150624_darkMap_M02775-35_90V.fits",
                  "150624_darkMap_M02775-35_95V.fits",
                  "150624_darkMap_M02775-35_100V.fits",
                  "150624_darkMap_M02775-35_105V.fits"],
                 ["150708_darkMap_M02775-35_25V_50K.fits",
                  "150708_darkMap_M02775-35_45V_50K.fits",
                  "150708_darkMap_M02775-35_60V_50K.fits",
                  "150710_darkMap_M02775-35_60V_50K.fits",
                  "150708_darkMap_M02775-35_70V_50K.fits",
                  "150710_darkMap_M02775-35_75V_50K.fits",
                  "150708_darkMap_M02775-35_80V_50K.fits",
                  "150708_darkMap_M02775-35_90V_50K.fits",
                  "150708_darkMap_M02775-35_100V_50K.fits"]]
    biases = [[2.5, 3.5, 4.5, 5.5, 6.0, 6.0, 6.5, 6.5, 7.0, 7.5, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5],
              [2.5, 4.5, 6.0, 6.0, 7.0, 7.5, 8.0, 9.0, 10.0]]
    colors = ['b','r','g','k','m','y','c']
    w = [[224,256],[160,192]] #Top center. (standard)
    for b,files in map(None, biases, fileslist):
        medians = []
        for f in files:
            print "file:", f
            d = openfits(f)
            dark = numpy.median(d[w[0][0]:w[0][1],w[1][0]:w[1][1]])
            medians.append(dark * 2.89)
        plt.semilogy(b, medians, 'o')
    plt.xlabel("Bias Voltage (V)")
    plt.ylabel("e-/s")
    plt.legend(["60K","50K"], loc = 2)
    plt.title("Median Dark Current (no gain correction) vs. Bias")
    plt.xlim([0,11])
    plt.ylim([0.01,15])
    plt.show()

def darkAllRamps15Jul2015():
    fileslist =[["150611_163202darkMap.fits",
             "150611_170254darkMap.fits",
             "150611_173346darkMap.fits",
             "150611_180439darkMap.fits"],
            ["150707_102057darkMap.fits",
             "150707_103654darkMap.fits",
             "150707_105252darkMap.fits",
             "150707_110849darkMap.fits",
             "150707_112452darkMap.fits",
             "150707_114049darkMap.fits",
             "150707_115647darkMap.fits",
             "150707_121244darkMap.fits"]]
    rawfiles = ["150611_163202.fits",
                "150611_170254.fits",
                "150611_173346.fits",
                "150611_180439.fits"]
    timing =[30, 15]
    w = [[224,256],[160,192]] #Top center. (standard)
    for t,files in map(None, timing, fileslist):
        medians = []
        for f in files:
            print "file:", f
            d = openfits(f)
            dark = numpy.median(d[w[0][0]:w[0][1],w[1][0]:w[1][1]])
            medians.append(dark * 2.89)
        plt.semilogy(range(t, t * (len(medians) + 1), t), medians, 'o-')
    #Let's also do the 60K in half-sections.
    medians = []
    for f in rawfiles:
        print "file:", f
        d = openfits(f)
        starts = [20, 100]
        print d.shape[0]
        ends = [99, d.shape[0] - 1]
        for start, end in map(None, starts, ends):
            e = numpy.zeros(d.shape[1:])
            n = 20
            for i in range(0,n):
                e += d[i + start,:,:]
                e -= d[i - n + end,:,:]
            e /= n
            e /= end - (start + n)
            e /= 10
            medians.append(numpy.median(e[w[0][0]:w[0][1],w[1][0]:w[1][1]]) * 2.89)
    t = timing[0] / 2
    plt.semilogy(range(t, t * (len(medians) + 1), t), medians, 'o-')
##    medians = []
##    for f in rawfiles:
##        print "file:", f
##        d = openfits(f)
##        start = 20
##        end = d.shape[0]
##        e = numpy.zeros(d.shape[1:])
##        n = 20
##        for i in range(0,n):
##            e += d[i + start,:,:]
##            e -= d[i - n + end,:,:]
##        e /= n
##        e /= end - (start + n)
##        e /= 10
##        medians.append(numpy.median(e[w[0][0]:w[0][1],w[1][0]:w[1][1]]) * 2.89)
##    t = timing[0]
##    plt.semilogy(range(t, t * (len(medians) + 1), t), medians, 'o-')
    plt.xlabel("Time (min)")
    plt.ylabel("e-/s")
    plt.legend(["60K","50K","60K 15min"])
    plt.title("Median Dark Current (no gain correction) vs. Ramp #")
    plt.show()

def darkRamps22Nov2015():
    files =["151121_150033darkMap.fits",
            "151121_160434darkMap.fits",
            "151121_170834darkMap.fits",
            "151121_181230darkMap.fits",
            "151121_191625darkMap.fits",
            "151121_202025darkMap.fits",
            "151121_212421darkMap.fits",
            "151122_155032darkMap.fits"]
##    w = [[224,256],[160,192]] #Top center. (standard)
    w = [[224,256],[160,192]] #Top center. (standard)
    biases = [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 12.5]
    medians = []
    for f in files:
        print "file:", f
        d = openfits(f)
        dark = numpy.median(d[w[0][0]:w[0][1],w[1][0]:w[1][1]])
        medians.append(dark * 2.89)
    plt.plot(biases, medians, 'o-')
    plt.xlabel("Bias (V)")
    plt.ylabel("e-/s")
    plt.title("Median Dark Current (no gain correction) vs. Bias Voltage M06715-27")
    plt.show()

def darkRamps24Nov2015():
    files =["151123_180237darkMap.fits",
            "151123_190638darkMap.fits",
            "151123_201033darkMap.fits",
            "151123_211428darkMap.fits",
            "151123_221823darkMap.fits",
            "151123_232219darkMap.fits",
            "151124_002614darkMap.fits",
            "151124_013009darkMap.fits"]
##    w = [[224,256],[160,192]] #Top center. (standard)
    w = [[224,256],[160,192]] #Top center. (standard)
    biases = [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
    medians = []
    for f in files:
        print "file:", f
        d = openfits(f)
        dark = numpy.median(d[w[0][0]:w[0][1],w[1][0]:w[1][1]])
        medians.append(dark * 2.89)
    plt.plot(biases, medians, 'o-')
    plt.xlabel("Bias (V)")
    plt.ylabel("e-/s")
    plt.title("Median Dark Current (no gain correction) vs. Bias Voltage M06715-27")
    plt.show()

def darkRamps30Nov2015():
    files =["151130_125648darkMap.fits",
            "151129_160710darkMap.fits",
            "151129_171306darkMap.fits",
            "151129_181901darkMap.fits",
            "151129_192456darkMap.fits",
            "151129_203051darkMap.fits",
            "151129_213646darkMap.fits",
            "151129_224241darkMap.fits",
            "151129_234836darkMap.fits"]
##    w = [[224,256],[160,192]] #Top center. (standard)
    w = [[92,184],[256,-1]] #Right side. (standard)
    biases = [2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    medians = []
    for f in files:
        print "file:", f
        d = openfits(f)
        dark = numpy.median(d[w[0][0]:w[0][1],w[1][0]:w[1][1]])
        medians.append(dark * 2.89)
    plt.plot(biases, medians, 'o-')
    plt.xlabel("Bias (V)")
    plt.ylabel("e-/s")
    plt.title("Median Dark Current (no gain correction) vs. Bias Voltage M06715-27")
    plt.show()

def darkRamps3Dec2015():
    files =["151101_111735darkMap.fits",
            "151020_115216darkMap.fits",
            "151026_174613darkMap.fits",
            "151123_180237darkMap.fits",
            "151108_131417darkMap.fits",
            "151130_180141darkMap.fits"]
##    w = [[224,256],[160,192]] #Top center. (standard)
    w = [[92,184],[256,-1]] #Right side. (standard)
    medians = []
    for f in files:
        print "file:", f
        d = openfits(f)
        dark = 2.89 * numpy.median(d[w[0][0]:w[0][1],w[1][0]:w[1][1]])
        print f, dark

def plotMedian(filename):
    d = openfits(filename)
    m = numpy.mean(numpy.median(d, axis = 2), axis = 1)
    plt.plot(m)
    plt.show()

def responsivity():
    offFiles = ["151006_135413.fits",#45K
                "151006_093915.fits",#50K
                "151006_170127.fits",#55K
                "151008_182926.fits",#60K
                "151009_132513.fits",#70K
                "151009_160527.fits",#85K
                "151010_111756.fits",#110K
                "151010_165505.fits",#150K
                "151019_105533.fits"]#mk13
    onFiles =  ["151006_135432.fits",#45K
                "151006_093938.fits",#50K
                "151006_170139.fits",#55K
                "151008_182937.fits",#60K
                "151009_132547.fits",#70K
                "151009_160601.fits",#85K
                "151010_111813.fits",#110K
                "151010_165519.fits",#150K
                "151019_105545.fits"]#mk13
    temps = [45, 50, 55, 60, 70, 85, 110, 150]
    results = []
    for offFile, onFile in map(None, offFiles, onFiles):
        off = openfits(offFile)
        on = openfits(onFile)
        medians = []
        for n in range(off.shape[0]):
            offn = numpy.median(off[n,:,:])
            onn = numpy.median(on[n,:,:])
            medians.append(offn - onn)
        medians = numpy.array(medians)
        medians -= medians[0]
        results.append(medians[2])
        plt.plot(medians)
    plt.legend(["45K","50K","55K","60K","70K","85K","110K","150K"], loc = 2)
    plt.show()
    results = numpy.array(results)
    results *= (0.21 / results[3])
    plt.plot(temps, results, "o-")
    plt.xlabel("Temperature (K)")
    plt.ylabel("QE (approximate)")
    plt.show()

def responsivityVsTemp5Dec2015():
    offFiles = ["151202_093745.fits",#60K
                "151203_084055.fits",#85K
                "151203_164528.fits",#110K
                "151204_141122.fits",#130K
                "151205_172001.fits"]#150K
    onFiless =[["151202_093948.fits",#60K
                "151203_084155.fits",#85K
                "151203_164603.fits",#110K
                "151204_141259.fits",#130K
                "151205_172101.fits"],#150K
              ["151202_094038.fits",#60K
                "151203_084302.fits",#85K
                "151203_164718.fits",#110K
                "151204_141401.fits",#130K
                "151205_172148.fits"],#150K
              ["151202_094116.fits",#60K
                "151203_084341.fits",#85K
                "151203_164800.fits",#110K
                "151204_141441.fits",#130K
                "151205_172226.fits"]]#150K
    temps = [60, 85, 110, 130, 150]
    for onFiles in onFiless:
        results = []
        for offFile, onFile in map(None, offFiles, onFiles):
            off = openfits(offFile)
            on = openfits(onFile)
            medians = []
            for n in range(off.shape[0]):
                offn = numpy.median(off[n,:,:])
                onn = numpy.median(on[n,:,:])
                medians.append(offn - onn)
            medians = numpy.array(medians)
            medians -= medians[0]
            results.append(medians[20])
        results = numpy.array(results)
        results *= (1.00 / results[-1])
        plt.plot(temps, results, "o-")
    plt.xlabel("Temperature (K)")
    plt.ylabel("responsivity")
    plt.ylim([0.05,1.05])
    plt.xlim([55,155])
    plt.legend(["1.75um","1.30um","1.05um"], loc = 2)
    plt.show()

def responsivity22Oct2015():
    offFiles = ["151022_102156.fits",#
                "151022_102156.fits",#
                "151022_102156.fits"]#
    onFiles =  ["151022_102219.fits",#LED1
                "151022_102320.fits",#LED2
                "151022_102408.fits"]#LED3
    waAvelengths = [1.75, 1.30, 1.05]
    results = []
    power = [0.38, 2.00, 2.50]
    for offFile, onFile, p, w in map(None, offFiles, onFiles, power, wavelengths):
        off = openfits(offFile)[:,64:-64,92:-92]
        on = openfits(onFile)[:,64:-64,92:-92]
        medians = []
        for n in range(off.shape[0]):
            offn = numpy.median(off[n,:,:])
            onn = numpy.median(on[n,:,:])
            medians.append(offn - onn)
        medians = numpy.array(medians)
        medians -= medians[0]
        medians /= p
        t = numpy.array(range(medians.shape[0])) * 10
        plt.plot(t, medians)
        diff = off - on
        savefits("151022_Response_" + str(w) + "um.fits", diff[3] - diff[1])
    plt.legend(["1.75um","1.30um","1.05um"], loc = 2)
    plt.xlabel("ms")
    plt.ylabel("ADU")
    plt.xlim([0,30])
    plt.title("Mk13 M06665-12 Responsivity (Calibrated for LED Output)")
    plt.show()

def responsivity23Oct2015():
    offFiles = ["151023_103211.fits",#
                "151023_103211.fits",#
                "151023_103211.fits",
                "151006_093915.fits",
                "151005_163241.fits",
                "151005_163241.fits"]#
    onFiles =  ["151023_103127.fits",#LED1 mk13 0.8V
                "151023_103330.fits",#LED2 mk13 0.981V
                "151023_103504.fits",#LED3 mk13 1.15V
                "151006_093938.fits",#LED1 mk12 0.8V
                "151005_163604.fits",#LED2 mk12 0.981V
                "151005_163830.fits"]#LED3 mk12 1.15V
    wavelengths = [1.75, 1.30, 1.05]
    results = []
    power = [0.38, 2.00, 2.50, 0.38, 2.00, 2.50]
    medianses = []
    for offFile, onFile, p, w in map(None, offFiles, onFiles, power, wavelengths):
        off = openfits(offFile)[:,64:-64,92:-92]
        on = openfits(onFile)[:,64:-64,92:-92]
        medians = []
        for n in range(off.shape[0]):
            offn = numpy.median(off[n,:,:])
            onn = numpy.median(on[n,:,:])
            medians.append(offn - onn)
        medians = numpy.array(medians)
        medians -= medians[0]
        print offFile, onFile
        medians /= p
##        t = numpy.array(range(medians.shape[0])) * 10
##        plt.plot(t, medians)
        medianses.append(medians)
    for i in range(3):
        t = numpy.array(range(100)) * 10
        plt.plot(t,medianses[i][:100] / medianses[i+3][:100])
##    t = numpy.array(range(medians.shape[0])) * 10
##    plt.legend(["1.75um mk13","1.30um mk13","1.05um mk13","1.75um mk12","1.30um mk12","1.05um mk12"], loc = 2)
        plt.legend(["1.75um","1.30um","1.05um"], loc = 2)
    plt.xlabel("ms")
##    plt.ylabel("ADU")
    plt.ylabel("ratio")
    plt.xlim([0,300])
    plt.title("Mk13/Mk12 Comparative Responsivity (Calibrated for LED Output)")
    plt.show()

def responsivity26Oct2015():
    offFiles = ["151023_103211.fits",#
                "151023_103211.fits",#
                "151023_103211.fits",
##                "151006_093915.fits",
##                "151005_163241.fits",
##                "151005_163241.fits",
                "151026_142342.fits",
                "151026_142342.fits",
                "151026_142342.fits"]#
    onFiles =  ["151023_103127.fits",#LED1 mk13 0.8V
                "151023_103330.fits",#LED2 mk13 0.981V
                "151023_103504.fits",#LED3 mk13 1.15V
##                "151006_093938.fits",#LED1 mk12 0.8V
##                "151005_163604.fits",#LED2 mk12 0.981V
##                "151005_163830.fits",#LED3 mk12 1.15V
                "151026_142513.fits",#LED1 mk14 0.8V
                "151026_143013.fits",#LED2 mk14 0.981V
                "151026_143301.fits"]#LED3 mk14 1.15V
    wavelengths = [1.75, 1.30, 1.05]
    results = []
    power = [0.38, 2.00, 2.50, 0.38, 2.00, 2.50]#, 0.38, 2.00, 2.50]
    for offFile, onFile, p, w in map(None, offFiles, onFiles, power, wavelengths):
        off = openfits(offFile)[:,64:-64,92:-92]
        on = openfits(onFile)[:,64:-64,92:-92]
        medians = []
        for n in range(off.shape[0]):
            offn = numpy.median(off[n,:,:])
            onn = numpy.median(on[n,:,:])
            medians.append(offn - onn)
        medians = numpy.array(medians)
        medians -= medians[0]
        medians /= p
        t = numpy.array(range(medians.shape[0])) * 10
        plt.plot(t, medians)
        diff = off - on
    plt.legend(["1.75um mk13","1.30um mk13","1.05um mk13","1.75um mk14","1.30um mk14","1.05um mk14"], loc = 2)
    plt.xlabel("ms")
    plt.ylabel("ADU")
    plt.xlim([0,30])
    plt.title("Responsivity (Calibrated for LED Output)")
    plt.show()

def responsivity2Nov2015():
    offFiles = ["151102_120211.fits",#
                "151102_120211.fits",#
                "151102_120211.fits",
##                "151006_093915.fits",
##                "151005_163241.fits",
##                "151005_163241.fits",
                "151026_142342.fits",
                "151026_142342.fits",
                "151026_142342.fits"]#
    onFiles =  ["151102_120305.fits",#LED1 mk13 M06665-03 0.8V
                "151102_120522.fits",#LED2 mk13 M06665-03 0.981V
                "151102_120731.fits",#LED3 mk13 M06665-03 1.15V
##                "151006_093938.fits",#LED1 mk12 0.8V
##                "151005_163604.fits",#LED2 mk12 0.981V
##                "151005_163830.fits",#LED3 mk12 1.15V
                "151026_142513.fits",#LED1 mk14 0.8V
                "151026_143013.fits",#LED2 mk14 0.981V
                "151026_143301.fits"]#LED3 mk14 1.15V
    wavelengths = [1.75, 1.30, 1.05]
    results = []
    power = [0.38, 2.00, 2.50, 0.38, 2.00, 2.50]#, 0.38, 2.00, 2.50]
    for offFile, onFile, p, w in map(None, offFiles, onFiles, power, wavelengths):
        off = openfits(offFile)[:,64:-64,92:-92]
        on = openfits(onFile)[:,64:-64,92:-92]
        medians = []
        for n in range(off.shape[0]):
            offn = numpy.median(off[n,:,:])
            onn = numpy.median(on[n,:,:])
            medians.append(offn - onn)
        medians = numpy.array(medians)
        medians -= medians[0]
        medians /= p
        t = numpy.array(range(medians.shape[0])) * 10
        plt.plot(t, medians)
        diff = off - on
    plt.legend(["1.75um mk13","1.30um mk13","1.05um mk13","1.75um mk14","1.30um mk14","1.05um mk14"], loc = 2)
    plt.xlabel("ms")
    plt.ylabel("ADU")
    plt.xlim([0,30])
    plt.title("Responsivity (Calibrated for LED Output)")
    plt.show()

def responsivity8Nov2015():
    offFiles = ["151108_101424.fits",
                "151108_101424.fits",
                "151108_101424.fits",
                "151026_142342.fits",
                "151026_142342.fits",
                "151026_142342.fits"]#
    onFiles =  ["151108_101441.fits",#LED1 mk13 M06715-29 0.8V
                "151108_101526.fits",#LED2 mk13 M06715-29 0.981V
                "151108_101602.fits",#LED3 mk13 M06715-29 1.15V
                "151026_142513.fits",#LED1 mk14 0.8V
                "151026_143013.fits",#LED2 mk14 0.981V
                "151026_143301.fits"]#LED3 mk14 1.15V
    wavelengths = [1.75, 1.30, 1.05]
    results = []
    power = [0.38, 2.00, 2.50, 0.38, 2.00, 2.50]#, 0.38, 2.00, 2.50]
    for offFile, onFile, p, w in map(None, offFiles, onFiles, power, wavelengths):
        off = openfits(offFile)[:,64:-64,92:-92]
        on = openfits(onFile)[:,64:-64,92:-92]
        medians = []
        for n in range(off.shape[0]):
            offn = numpy.median(off[n,:,:])
            onn = numpy.median(on[n,:,:])
            medians.append(offn - onn)
        medians = numpy.array(medians)
        medians -= medians[0]
        medians /= p
        t = numpy.array(range(medians.shape[0])) * 10
        plt.plot(t, medians)
        diff = off - on
    plt.legend(["1.75um M06715-29","1.30um M06715-29","1.05um M06715-29","1.75um M06715-27","1.30um M06715-27","1.05um M06715-27"], loc = 2)
    plt.xlabel("ms")
    plt.ylabel("ADU")
    plt.xlim([0,30])
    plt.title("Responsivity (Calibrated for LED Output)")
    plt.show()

def responsivity26Oct2015ratios():
    offFiles = ["151023_103211.fits",#
                "151023_103211.fits",#
                "151023_103211.fits",
##                "151006_093915.fits",
##                "151005_163241.fits",
##                "151005_163241.fits",
                "151026_142342.fits",
                "151026_142342.fits",
                "151026_142342.fits"]#
    onFiles =  ["151023_103127.fits",#LED1 mk13 0.8V
                "151023_103330.fits",#LED2 mk13 0.981V
                "151023_103504.fits",#LED3 mk13 1.15V
##                "151006_093938.fits",#LED1 mk12 0.8V
##                "151005_163604.fits",#LED2 mk12 0.981V
##                "151005_163830.fits",#LED3 mk12 1.15V
                "151026_142513.fits",#LED1 mk14 0.8V
                "151026_143013.fits",#LED2 mk14 0.981V
                "151026_143301.fits"]#LED3 mk14 1.15V
    wavelengths = [1.75, 1.30, 1.05]
    results = []
    power = [0.38, 2.00, 2.50, 0.38, 2.00, 2.50]#, 0.38, 2.00, 2.50]
    medianses = []
    for offFile, onFile, p, w in map(None, offFiles, onFiles, power, wavelengths):
        off = openfits(offFile)[:,64:-64,92:-92]
        on = openfits(onFile)[:,64:-64,92:-92]
        medians = []
        for n in range(off.shape[0]):
            offn = numpy.median(off[n,:,:])
            onn = numpy.median(on[n,:,:])
            medians.append(offn - onn)
        medians = numpy.array(medians)
        medians -= medians[0]
        print offFile, onFile
        medians /= p
##        t = numpy.array(range(medians.shape[0])) * 10
##        plt.plot(t, medians)
        medianses.append(medians)
    for i in range(3):
        t = numpy.array(range(100)) * 10
        plt.plot(t,medianses[i+3][:100] / medianses[i][:100])
##    t = numpy.array(range(medians.shape[0])) * 10
##    plt.legend(["1.75um mk13","1.30um mk13","1.05um mk13","1.75um mk12","1.30um mk12","1.05um mk12"], loc = 2)
        plt.legend(["1.75um","1.30um","1.05um"], loc = 2)
    plt.xlabel("ms")
##    plt.ylabel("ADU")
    plt.ylabel("ratio")
    plt.xlim([0,300])
    plt.title("Mk14/Mk13 Comparative Responsivity (Calibrated for LED Output)")
    plt.show()

def responsivity16Nov2015():
    offFiles = ["151116_095249.fits",
                "151116_095249.fits",
                "151116_095249.fits"]#LED3 mk14 M06715-27
    onFiles =  ["151116_100340.fits",#LED1 mk14 M06715-27 0.8V
                "151116_095524.fits",#LED2 mk14 M06715-27 0.981V
                "151116_100218.fits"]#LED3 mk14 M06715-27 1.15V
    wavelengths = [1.75, 1.30, 1.05]
    results = []
    power = [0.38, 2.00, 2.50]#, 0.38, 2.00, 2.50]
    for offFile, onFile, p in map(None, offFiles, onFiles, power):
        off = openfits(offFile)[:,64:-64,92:-92]
        on = openfits(onFile)[:,64:-64,92:-92]
        medians = []
        for n in range(off.shape[0]):
            offn = numpy.median(off[n,:,:])
            onn = numpy.median(on[n,:,:])
            medians.append(offn - onn)
        medians = numpy.array(medians)
        medians -= medians[0]
        medians /= p
        t = numpy.array(range(medians.shape[0])) * 10
        plt.plot(t, medians)
        diff = off - on
    plt.legend(["1.75um M06715-27","1.30um M06715-27","1.05um M06715-27"], loc = 2)
    plt.xlabel("ms")
    plt.ylabel("ADU")
    plt.xlim([0,30])
    plt.title("Responsivity (Calibrated for LED Output)")
    plt.show()

def responsivity1Dec2015():
    offFiles = ["151116_095249.fits",
                "151116_095249.fits",
                "151116_095249.fits",#LED off mk14 M06715-27
                "151201_093509.fits",
                "151201_093509.fits",
                "151201_093509.fits"]#LED off mk14 M06715-34
    onFiles =  ["151116_100340.fits",#LED1 mk14 M06715-27 0.8V
                "151116_095524.fits",#LED2 mk14 M06715-27 0.981V
                "151116_100218.fits",#LED3 mk14 M06715-27 1.15V
                "151201_093526.fits",#LED1 mk14 M06715-34 0.8V
                "151201_093623.fits",#LED2 mk14 M06715-34 0.981V
                "151201_093704.fits"]#LED3 mk14 M06715-34 1.15V
    wavelengths = [1.75, 1.30, 1.05, 1.75, 1.30, 1.05]
    results = []
    power = [0.38, 2.00, 2.50, 0.38, 2.00, 2.50]
    for offFile, onFile, p in map(None, offFiles, onFiles, power):
        off = openfits(offFile)[:,64:-64,92:-92]
        on = openfits(onFile)[:,64:-64,92:-92]
        medians = []
        for n in range(off.shape[0]):
            offn = numpy.median(off[n,:,:])
            onn = numpy.median(on[n,:,:])
            medians.append(offn - onn)
        medians = numpy.array(medians)
        medians -= medians[0]
        medians /= p
        t = numpy.array(range(medians.shape[0])) * 10
        plt.plot(t, medians)
        diff = off - on
    plt.legend(["1.75um M06715-27","1.30um M06715-27","1.05um M06715-27"], loc = 2)
    plt.xlabel("ms")
    plt.ylabel("ADU")
##    plt.xlim([0,30])
    plt.title("Responsivity (Calibrated for LED Output)")
    plt.show()

def responsivity2Dec2015():
    offFiles = ["151116_095249.fits",
                "151116_095249.fits",
                "151116_095249.fits",#Mk14 M06715-27 LED off
                "151202_093745.fits",
                "151202_093745.fits",
                "151202_093745.fits"]#Mk14 M06715-34 LED off
    onFiles =  ["151116_111114.fits",#LED1 M06715-27 10.0mA
                "151116_130110.fits",#LED2 M06715-27 10.0mA
                "151116_130606.fits",#LED3 M06715-27 10.0mA
                "151202_093948.fits",#LED1 M06715-34 10.0mA
                "151202_094038.fits",#LED2 M06715-34 10.0mA
                "151202_094116.fits"]#LED3 M06715-34 10.0mA
    results = []
    power = [0.38, 2.00, 2.50, 0.38, 2.00, 2.50]#, 0.38, 2.00, 2.50]
    for offFile, onFile, p in map(None, offFiles, onFiles, power):
        off = openfits(offFile)[:,64:-64,92:-92]
        on = openfits(onFile)[:,64:-64,92:-92]
        medians = []
        for n in range(off.shape[0]):
            offn = numpy.median(off[n,:,:])
            onn = numpy.median(on[n,:,:])
            medians.append(offn - onn)
        medians = numpy.array(medians)
        medians -= medians[0]
        medians /= p
        t = numpy.array(range(medians.shape[0])) * 10
        plt.plot(t, medians)
        diff = off - on
    plt.legend(["1.75um M06715-27","1.30um M06715-27","1.05um M06715-27","1.75um M06715-34","1.30um M06715-34","1.05um M06715-34"], loc = 2)
    plt.xlabel("ms")
    plt.ylabel("ADU")
##    plt.xlim([0,30])
    plt.title("Responsivity (Calibrated for LED Output)")
    plt.show()

def responsivity2Dec2015AllDetectors():
    offFiles = ["141216_101824.fits",
                "150421_095357.fits",
                "150928_141041.fits",
                "151019_105533.fits",
                "151103_151338.fits",
                "151026_144307.fits",
                "151108_111141.fits",
                "151202_093745.fits"]
    onFiles =  ["141216_101837.fits",
                "150421_095410.fits",
                "150928_141102.fits",
                "151019_105545.fits",
                "151103_151353.fits",
                "151026_144330.fits",
                "151108_111159.fits",
                "151202_093948.fits"]
    results = []
    stds = []
    correction = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.555]
    for offFile, onFile, corr in map(None, offFiles, onFiles, correction):
        off = openfits(offFile)[:,64:-64,92:-92]
        on = openfits(onFile)[:,64:-64,92:-92]
        medians = []
        for n in range(off.shape[0]):
            offn = numpy.median(off[n,:,:])
            onn = numpy.median(on[n,:,:])
            medians.append(offn - onn)
        medians = numpy.array(medians)
        medians -= medians[0]
        medians *= corr
        t = numpy.array(range(medians.shape[0])) * 10
        plt.plot(t, medians)
        results.append(medians[99])
    results = numpy.array(results)
    results *= (0.30 / results[0])
    leg = ["mk3 M02775-10",
           "mk10 M04935-17",
           "mk12 M06495-19",
           "mk13 M06665-12",
           "mk13 M06665-03",
           "mk14 M06715-27",
           "mk14 M06715-29",
           "mk14 M06715-34"]
    for i in range(results.shape[0]):
        print leg[i], str(results[i])[:5]
    plt.legend(leg, loc = 2)
    plt.xlabel("ms")
    plt.ylabel("ADU")
    plt.xlim([0,1000])
    plt.ylim([0,3500])
    plt.title("Responsivity")
    plt.show()

def responsivity3Dec2015():
    offFiles = ["151202_093745.fits",
                "151202_093745.fits",
                "151202_093745.fits",#Mk14 M06715-34 LED off 60K
                "151203_084055.fits",
                "151203_084055.fits",
                "151203_084055.fits",#Mk14 M06715-34 LED off 85K
                "151203_164528.fits",
                "151203_164528.fits",
                "151203_164528.fits"]#Mk14 M06715-34 LED off 110K
    onFiles =  ["151202_093948.fits",#LED1 M06715-34 10.0mA 60K
                "151202_094038.fits",#LED2 M06715-34 10.0mA 60K
                "151202_094116.fits",#LED3 M06715-34 10.0mA 60K
                "151203_084155.fits",#LED1 M06715-34 10.0mA 85K
                "151203_084302.fits",#LED2 M06715-34 10.0mA 85K
                "151203_084341.fits",#LED3 M06715-34 10.0mA 85K
                "151203_164603.fits",#LED1 M06715-34 10.0mA 110K
                "151203_164718.fits",#LED2 M06715-34 10.0mA 110K
                "151203_164800.fits"]#LED3 M06715-34 10.0mA 110K
    results = []
    power = [0.38, 2.00, 2.50, 0.38, 2.00, 2.50, 0.38, 2.00, 2.50]
    for offFile, onFile, p in map(None, offFiles, onFiles, power):
        off = openfits(offFile)[:,64:-64,92:-92]
        on = openfits(onFile)[:,64:-64,92:-92]
        medians = []
        for n in range(off.shape[0]):
            offn = numpy.median(off[n,:,:])
            onn = numpy.median(on[n,:,:])
            medians.append(offn - onn)
        medians = numpy.array(medians)
        medians -= medians[0]
        medians /= p
        t = numpy.array(range(medians.shape[0])) * 10
        print medians[20]
        plt.plot(t, medians)
    plt.legend(["1.75um 60K","1.30um 60K","1.05um 60K",
                "1.75um 85K","1.30um 85K","1.05um 85K",
                "1.75um 110K","1.30um 110K","1.05um 110K",], loc = 2)
    plt.xlabel("ms")
    plt.ylabel("ADU")
##    plt.xlim([0,30])
    plt.title("Responsivity (Calibrated for LED Output)")
    plt.show()

def responsivity4Dec2015AllDetectors130um():
    offFiles = ["151102_120211.fits",#M06665-03
                "151023_103211.fits",#M06665-12
                "151026_142342.fits",#M06715-27
                "151108_101424.fits",#M06715-29
                "151116_095249.fits",#M06715-27 new app
                "151202_093745.fits"]#M06715-34 new app
    onFiles =  ["151102_120522.fits",#M06665-03
                "151023_103330.fits",#M06665-12
                "151026_143013.fits",#M06715-27
                "151108_101526.fits",#M06715-29
                "151116_130110.fits",#M06715-27 new app
                "151202_094038.fits"]#M06715-34 new app
    results = []
    stds = []
    correction = [1.0, 1.0, 1.0, 1.0, 0.531, 0.531]
    for offFile, onFile, corr in map(None, offFiles, onFiles, correction):
        off = openfits(offFile)[:,64:-64,92:-92]
        on = openfits(onFile)[:,64:-64,92:-92]
        medians = []
        for n in range(off.shape[0]):
            offn = numpy.median(off[n,:,:])
            onn = numpy.median(on[n,:,:])
            medians.append(offn - onn)
        medians = numpy.array(medians)
        medians -= medians[0]
        medians *= corr
        t = numpy.array(range(medians.shape[0])) * 10
        plt.plot(t, medians)
        results.append(medians[20])
    results = numpy.array(results)
    results *= (1.0 / results[-1])
    leg = ["mk13 M06665-12",
           "mk13 M06665-03",
           "mk14 M06715-27",
           "mk14 M06715-29",
           "mk14 M06715-27 new app",
           "mk14 M06715-34 new app"]
    for i in range(results.shape[0]):
        print leg[i], str(results[i])[:5]
    plt.legend(leg, loc = 2)
    plt.xlabel("ms")
    plt.ylabel("ADU")
    plt.xlim([0,1000])
##    plt.ylim([0,3500])
    plt.title("Responsivity")
    plt.show()

def responsivity4Dec2015AllDetectors105um():
    offFiles = ["151102_120211.fits",#M06665-03
                "151023_103211.fits",#M06665-12
                "151026_142342.fits",#M06715-27
                "151108_101424.fits",#M06715-29
                "151116_095249.fits",#M06715-27 new app
                "151202_093745.fits"]#M06715-34 new app
    onFiles =  ["151102_120731.fits",#M06665-03
                "151023_103504.fits",#M06665-12
                "151026_143301.fits",#M06715-27
                "151108_101602.fits",#M06715-29
                "151116_130606.fits",#M06715-27 new app
                "151202_094116.fits"]#M06715-34 new app
    results = []
    stds = []
    correction = [1.0, 1.0, 1.0, 1.0, 0.531, 0.531]
    for offFile, onFile, corr in map(None, offFiles, onFiles, correction):
        off = openfits(offFile)[:,64:-64,92:-92]
        on = openfits(onFile)[:,64:-64,92:-92]
        medians = []
        for n in range(off.shape[0]):
            offn = numpy.median(off[n,:,:])
            onn = numpy.median(on[n,:,:])
            medians.append(offn - onn)
        medians = numpy.array(medians)
        medians -= medians[0]
        medians *= corr
        t = numpy.array(range(medians.shape[0])) * 10
        plt.plot(t, medians)
        results.append(medians[20])
    results = numpy.array(results)
    results *= (1.0 / results[-1])
    leg = ["mk13 M06665-12",
           "mk13 M06665-03",
           "mk14 M06715-27",
           "mk14 M06715-29",
           "mk14 M06715-27 new app",
           "mk14 M06715-34 new app"]
    for i in range(results.shape[0]):
        print leg[i], str(results[i])[:5]
    plt.legend(leg, loc = 2)
    plt.xlabel("ms")
    plt.ylabel("ADU")
    plt.xlim([0,1000])
##    plt.ylim([0,3500])
    plt.title("Responsivity")
    plt.show()

def LEDlinearity16Nov2015():
    files = [["151116_110855.fits",
              "151116_110932.fits",
              "151116_111002.fits",
              "151116_111028.fits",
              "151116_111114.fits"],
             ["151116_130318.fits",
              "151116_130251.fits",
              "151116_130216.fits",
              "151116_130143.fits",
              "151116_130110.fits"],
             ["151116_130417.fits",
              "151116_130438.fits",
              "151116_130512.fits",
              "151116_130539.fits",
              "151116_130606.fits"]]
    currents = [50, 40, 30, 20, 10]
    ax = plt.gca()
    ax2 = ax.twinx()
    for fil in files:
        results = []
        for f, c in map(None,fil, currents):
            d = openfits(f)[:,64:-64,92:-92]
            medians = []
            for n in range(d.shape[0]):
                medians.append(-numpy.median(d[n,:,:]))
            medians = numpy.array(medians)
            medians -= medians[0]
            medians /= c #Correct for current.
            endframe = 7
            medians /= endframe * 0.01
            results.append(medians[endframe] - medians[0])
##        results.append(0)
        ax.plot(currents, results, 'o-')
    ax.set_ylim(ymax = 20000)
    ax.set_xlim(xmax = 50, xmin = 10)
    ax2.set_ylim(ymax = 35000 * 2.89)
    ax.set_ylabel("ADU s-1 mA-1")
    ax2.set_ylabel("e- s-1 mA-1")
    ax.set_xlabel("mA")
    ax.legend(["1.75um","1.30um","1.05um"], loc = 2)
##    plt.xlim([0,30])
##    plt.ylim([0,200])
##    ax.title("Responsivity (Calibrated for LED Output)")
    plt.show()

def justGetTheGoddamnNumbers():
    offFiles = ["151023_103211.fits",#
                "151023_103211.fits",#
                "151023_103211.fits",
                "151006_093915.fits",
                "151005_163241.fits",
                "151005_163241.fits",
                "151022_102156.fits",#
                "151022_102156.fits",#
                "151022_102156.fits",
                "151026_142342.fits",
                "151026_142342.fits",
                "151026_142342.fits"]#
    onFiles =  ["151023_103127.fits",#LED1 mk13 0.8V
                "151023_103330.fits",#LED2 mk13 0.981V
                "151023_103504.fits",#LED3 mk13 1.15V
                "151006_093938.fits",#LED1 mk12 0.8V
                "151005_163604.fits",#LED2 mk12 0.981V
                "151005_163830.fits",#LED3 mk12 1.15V
                "151022_102219.fits",#LED1 1.132V mk13 80K
                "151022_102320.fits",#LED2 1.639V mk13 80K
                "151022_102408.fits",#LED3 1.580V mk13 80K
                "151026_142513.fits",#LED1 mk14 0.8V
                "151026_143013.fits",#LED2 mk14 0.981V
                "151026_143301.fits"]#LED3 mk14 1.15V
    wavelengths = [1.75, 1.30, 1.05]
    results = []
    medianses = []
    for offFile, onFile, w in map(None, offFiles, onFiles, wavelengths):
        off = openfits(offFile)[:,64:-64,92:-92]
        on = openfits(onFile)[:,64:-64,92:-92]
        medians = []
        for n in range(off.shape[0]):
            offn = numpy.median(off[n,:,:])
            onn = numpy.median(on[n,:,:])
            medians.append(offn - onn)
        medians = numpy.array(medians)
        medians -= medians[0]
        print offFile, onFile
##        t = numpy.array(range(medians.shape[0])) * 10
##        plt.plot(t, medians)
        print medians[30]

def ADU2e(filename):
    #Just converts a fits file from ADU/s to e-/s.
    d = openfits(filename)
    d *= -2.89 #2.89 e-/ADU
    d = savefits(filename[:-5] + "es.fits", d)

def detectorHistograms(filenames, title):
    #Converts the second frame of a cds file to a histogram.
    for f in filenames:
        d = openfits(f)
        frame = d[2,:,:].flatten()
##        y, binedges = histogram(frame, bins = 60, range = [-100,1500])
        y, binedges = histogram(frame, bins = 60)
        bincenters = 0.5 * (binedges[1:] + binedges[:-1])
        plt.semilogy(bincenters, y, '-')
    plt.ylabel("# pixels")
##    plt.xlabel("e-/s")
    plt.xlabel("ADU")
    plt.title(title)
    plt.legend(["Mark 13","Mark 14"], loc = 2)
##    plt.legend(["Mark 13","Mark 14"])
    plt.ylim(10,100000)
##    plt.xlim([-100,1500])
    plt.xlim(44000,56000)
##    ax = plt.gca()
##    ax.set_axis_bgcolor('white')
    plt.show()


def chargeGain2(filename):
    if not path.isfile(filename):
        print "File not found. Please check filename/pathing."
        return
    
    data = openfits(filename)
    if data.shape[1] == 256 and data.shape[2] == 320:
        data = data[:,96:-96,128:-128]
        print "Trimming to aperture..."
    
##    medianPlot(file)

    start = 40
    data = data - data[start,:,:]
    data = -1 * data[start+1:,:,:]
    ylist = []
    xlist = []
    signals = []
    variances = []
    #Do column adjustments.
##    for x in range(data.shape[2]):
##        data
##        data[:,:,x]
    for i in range(data.shape[0]):
        signals.append(numpy.mean(data[i,:,:].flatten()))
##        variances.append(numpy.std(data[i,:,:].flatten(), ddof = 1))
        variances.append(numpy.var(data[i,:,:].flatten(), ddof = 1))
   
    cuton = 10000
    cutoff = 25000
    startIndex = 1
    while signals[startIndex] < cuton:
        startIndex += 1
    dropIndex = 1
    while signals[-dropIndex] > cutoff:
        dropIndex += 1
    print "Dropping", dropIndex, "frames."
    m, b = linearfit(signals[startIndex:-dropIndex], variances[startIndex:-dropIndex])
##    m, b = linearfit(signals[300:], variances[300:])
    vfit = []
    for s in signals:
        vfit.append(s * m + b)
    if not (m == 0):
        g = (1/m)
        print "Gain:", g, "e-/ADU"
        print "Read Noise:", (b/g), "e- rms"
    plt.plot(signals,variances, 'o-')
    plt.plot(signals, vfit)
    plt.xlabel("Signal (ADU)")
    plt.ylabel("Standard Deviation (ADU)")
    plt.show()

def linearity(filename):
    d = openfits(filename)[:,96:-96,128:-128]
    medians = []
    for i in range(d.shape[0]):
        medians.append(numpy.median(d[i,:,:].flatten()))
    medians = numpy.array(medians)
    medians -= medians[0]
    medians *= -1
    print medians.shape
    ax = plt.gca()
    ax2 = ax.twinx()
    ax.plot(medians)
    ax.set_ylim(ymax = 35000)
    ax.set_xlim(xmax = d.shape[0])
    ax2.set_ylim(ymax = 35000 * 2.89)
    ax.set_ylabel("ADU")
    ax2.set_ylabel("e-")
    ax.set_xlabel("Frame #")
    plt.show()

def readNoisePlots(filename):
    d = openfits(filename)
    histrange = [-40, 40]
    binspacing = 0.5
    bins = (histrange[1] - histrange[0]) / binspacing
    subplots = [221, 222, 223, 224]
    fig = plt.figure()
    #Detrend by median.
    for i in range(d.shape[0]):
        d[i,:,:] -= numpy.median(d[i,:,:])
    for x in range(4):
        pix = d[:,0,x]
        pix -= pix[1]
        pix = pix[1:]
        #Let's detrend.
        m,b = linearfit(range(pix.shape[0]),pix)
        print m,b
        for i in range(pix.shape[0]):
            pix[i] -= m * i + b
        #Generate a title that gives the RN.
        title = str(numpy.std(pix, ddof = 1)) + " ADU rms"
        ax = fig.add_subplot(subplots[x])
        ax.set_ylabel("n")
        ax.set_xlabel("ADU")
        ax.set_title(title)
        y,binedges = histogram(pix, bins = bins, range=histrange)
        bincenters = 0.5*(binedges[1:]+binedges[:-1])
        ax.plot(bincenters,y,'k-')
    plt.show()

def readNoisePlotsvsAverages(filename):
    d = openfits(filename)
    pixelIndices = [0,2,3,5]
    histrange = [-40, 40]
    binspacing = 0.5
    bins = (histrange[1] - histrange[0]) / binspacing
    subplots = [231, 232, 233, 234, 235, 236]
    averages = [1, 4, 16, 64, 256, 1024]
    fig = plt.figure()
    #Detrend by mean.
    for i in range(d.shape[0]):
        d[i,:,:] -= numpy.mean(d[i,:,:])
    for i, a in enumerate(averages):
        ax = fig.add_subplot(subplots[i])
        ax.set_ylabel("n")
        ax.set_xlabel("ADU")
        devs = []
        for x in pixelIndices:
            pix = numpy.array(d[:,0,x])
            pix -= pix[1]
            pix = pix[1:]
            #Detrend by fitted line.
            m,b = linearfit(range(pix.shape[0]),pix)
            print m,b
            for i in range(pix.shape[0]):
                pix[i] -= m * i + b
            avgpix = avg(pix, avg = a)
            y,binedges = histogram(avgpix, bins = bins, range=histrange)
            bincenters = 0.5*(binedges[1:]+binedges[:-1])
            ax.plot(bincenters,y,'-')
            devs.append(numpy.std(avgpix, ddof = 1))
        #Generate a title that gives the RN.
        title = str(a) + "avg: " + str(numpy.mean(devs))[:5] + " ADU rms"
        ax.set_title(title)
    plt.show()

def photonCounting3(histrange = [-100,100], binspacing = 1.0, avg = 1, detrend = False, plotTraces = False, skip = 3000):
##    offFile = "151117_113112.fits" #COMMON = -11V, mk14 M06715-27, no ref, 32 x 1
##    onFile = "151117_113131.fits" #LED 2.812V

##    offFile = "CUBE_LED_OFF-3-0.fits" #COMMON = -15V, mk14 M06715-27 w/ PB, no ref, 32 x 1
##    onFile = "160422_LED3.2VCOM-15V-4-0.fits"  #LED 3.2V

##    offFile = "160516_LEDoffCOM-3V-3-0.fits" #COMMON = -3V, mk14 M06715-34 w/ PB, no ref, 32 x 1
##    onFile = "160516_LED3.2VCOM-3V-2-0.fits"  #LED 3.2V

##    offFile = "160601_COM-3VLEDoff-3-0.fits" #COMMON = -3V, mk 14 M06715-34 w/ PB, no ref, 32 x 1 not RRR
##    onFile = "160601_COM-3VLED3.2V-4-0.fits"

##    offFile = "160723_PCLEDoff2-29-0.fits" #COMMON = -15V, mk 13 M06665-25 w/ PB, no ref, 32 x 1 not RRR
##    onFile = "160723_PCLEDon-30-0.fits" #LED 3.1um 4.2V 100mA

    offFile = "161106_105115.fits" #COMMON = -11V, mk13 M06665-23 ME1000, no ref, 32 x 1, diff window
    onFile = "161106_105124.fits"
    
    off = openfits(offFile)
    on = openfits(onFile)
    #Check the lengths, trim from the end to match.
    if (off.shape[0] <> on.shape[0]):
        print "Length mismatch, trimming."
        if off.shape[0] > on.shape[0]:
            off = off[: - (off.shape[0] - on.shape[0]),:,:]
        else:
            on = on[: - (on.shape[0] - off.shape[0]),:,:]

    if skip > 0:
        off = off[skip:,:,:]
        on = on[skip:,:,:]
            
    #Do some detrending with the median.
    onMedian = []
    offMedian = []
    for i in range(off.shape[0]):
        if detrend:
            off[i,:,:] -= numpy.median(off[i,:,:])
            on[i,:,:] -= numpy.median(on[i,:,:])
        offMedian.append(numpy.median(off[i,:,:]))
        onMedian.append(numpy.median(on[i,:,:]))
    
    #Show median plots.
    if plotTraces:
        plt.plot(range(len(offMedian)), offMedian)
        plt.plot(range(len(onMedian)), onMedian)
        plt.show()
    #Let's generate a mask.
    pixels = []
    for x in range(off.shape[2]):
        for y in range(off.shape[1]):
            pixels.append(off[10,y,x] - off[9900,y,x])
    sortPix = numpy.sort(numpy.array(pixels), axis = None)
    #print "Sort Pix:", sortPix
    cutoff = sortPix[sortPix.shape[0] / 2]
    #print "Cutoff:", cutoff
    mask = numpy.zeros([off.shape[1],off.shape[2]])
    for x in range(off.shape[2]):
        for y in range(off.shape[1]):
            if (off[10,y,x] - off[9900,y,x] < cutoff):
                mask[y,x] = 1
    #print "Mask:", mask

    #Perform subtraction and masking.
    offSub = numpy.zeros([off.shape[0] - 1,off.shape[1],off.shape[2]])
    onSub = numpy.zeros([on.shape[0] - 1,on.shape[1],on.shape[2]])
    for i in range(offSub.shape[0]):
        offSub[i,:,:] = (off[i,:,:] - off[i + 1,:,:]) * mask
        onSub[i,:,:] = (on[i,:,:] - on[i + 1,:,:]) * mask

    #Do averaging.
    if avg > 1:
        offSub = avgCube(offSub, avg = avg)
        onSub = avgCube(onSub, avg = avg)

    #Compute bins.
    bins = (histrange[1] - histrange[0]) / binspacing
    y,binedges = histogram([offSub[:,0,0].flatten()], bins = bins, range=histrange)
    bincenters = 0.5*(binedges[1:]+binedges[:-1])

    print "Fitting Gaussian..."
    gaussian = fitGaussianToNegative(y, bincenters, binspacing)

    y2,binedges2 = histogram([onSub[:,0,0].flatten()], bins = bins, range=histrange)
    if len(binedges) <> len(binedges2):
        print "Something's real screwed up dude!"
        return
    
    #So now we have a fit gaussian, we want to scale it down to the on dataset.
    scale = 0.7
    delta = 0.1
    margin = 7
    while delta > 0.000001:
        here = abs(numpy.sum(gaussian[:(len(bincenters) / 2) - margin] * scale - y2[:(len(bincenters) / 2) - margin]))
        down = abs(numpy.sum(gaussian[:(len(bincenters) / 2) - margin] * (scale - delta) - y2[:(len(bincenters) / 2) - margin]))
        up = abs(numpy.sum(gaussian[:(len(bincenters) / 2) - margin] * (scale + delta) - y2[:(len(bincenters) / 2) - margin]))
        if here > up:
            print "up!"
            scale += delta
        elif here > down:
            print "down!"
            scale -= delta
        else:
            delta /= 2.
    print scale

    print "Fitting photon curve..."
##    photons = y2[:len(bincenters)] - scale * gaussian[:len(bincenters)] - (y - gaussian[:len(bincenters)])
##    photonPeak = list(photons).index(numpy.amax(photons))
##    photonFit = fitGaussianToNegative(photons, bincenters, binspacing, photonPeak - len(bincenters)/2)[:len(bincenters)]

    plt.hist([offSub[:,0,0].flatten(),onSub[:,0,0].flatten()], bins = bins, range=histrange, label = ["off","on"])
    plt.plot(numpy.array(bincenters)[:gaussian.shape[0]], gaussian[:len(bincenters)], label = "off fit")

    #Plot the subtraction also.
    plt.plot(numpy.array(bincenters)[:gaussian.shape[0]], y - gaussian[:len(bincenters)], label = "off - fit")

    #Plot the on fitting.
    plt.plot(numpy.array(bincenters)[:len(bincenters)], scale * gaussian[:len(bincenters)], label = "on fit")
    plt.plot(numpy.array(bincenters)[:len(bincenters)], y2[:len(bincenters)] - scale * gaussian[:len(bincenters)], label = "on - fit")

    plt.plot(numpy.array(bincenters)[:len(bincenters)], y2[:len(bincenters)] - scale * gaussian[:len(bincenters)] - (y - gaussian[:len(bincenters)]), label = "photons")

##    plt.plot(numpy.array(bincenters)[:len(bincenters)], photonFit, label = "photon fit")
##    plt.plot(numpy.array(bincenters)[:len(bincenters)], photons - photonFit, label = "photons - fit")

    i = bincenters.shape[0] / 2

##    adjustment = 

##    plt.legend(["LED off",
##                "LED on",
##                "LED off fit",
##                "LED off - fit"])
    plt.legend()
                
    
    plt.show()

##    profile = signal.gaussian(size,FWHM)



def photonCounting4(histrange = [-400,400], binspacing = 1.0, avg = 1, plotTraces = False, skip = 3000,
                    readResetRead = True):
##    offFile = "CUBE_LED_OFF-3-0.fits" #COMMON = -15V, mk14 M06715-27 w/ PB, no ref, 32 x 1
##    onFile = "160422_LED3.2VCOM-15V-4-0.fits"  #LED 3.2V GOOD
##    title = "BIAS = 19.5V, M06715-27, Data Taken 22 Apr 2016"

##    offFile = "160516_LEDoffCOM-3V-3-0.fits" #COMMON = -3V, mk14 M06715-34 w/ PB, no ref, 32 x 1, RRR
##    onFile = "160516_LED3.2VCOM-3V-2-0.fits"  #LED 3.2V

##    offFile = "160518_COM-15VLEDoff-1-0.fits" #COMMON = -15V, mk14 M06715-34 w/ PB, no ref, 32 x 1, RRR
##    onFile = "160518_COM-15VLED3.2V-2-0.fits"  #LED 3.2V

##    offFile = "160601_COM-3VLEDoff-18-0.fits" #COMMON = -3V, mk 14 M06715-34 w/ PB, no ref, 32 x 1 not RRR
##    onFile = "160601_COM-3VLED3.2V-19-0.fits"
##    title = "BIAS = 7.5V, M06715-34, Data Taken 1 Jun 2016"

##    offFile = "160109_155405.fits" #COMMON = -11V, mk14 M06665-12, no ref, 32 x 1
##    onFile = "160109_155441.fits" #LED1 = 1.2V BAD?
##    title = "BIAS = 14.5V, M06665-12, Data Taken 9 Jan 2016"

    offFile = "151117_113112.fits" #COMMON = -11V, mk14 M06715-27, no ref, 32 x 1
    onFile = "151117_113131.fits" #LED 2.812V BAD?
    title = "BIAS = 14.5V, M06715-27, Data Taken 17 Nov 2015"

##    offFile = "161106_105115.fits" #COMMON = -11V, mk13 M06665-23 ME1000, no ref, 32 x 1, diff window
##    onFile = "161106_105124.fits"
##    title = "BIAS = 14.5V, M06665-23, Data Taken 6 Nov 2016"

##    offFile = "170324_162320.fits"
##    onFile = "170324_162337.fits"
##    title = "$V_{bias} = 14.5\mathrm{V}$, M09225-11, 60K, Data Taken 24 Mar 2017"  #ME1001
    
    off = openfits(offFile)
    on = openfits(onFile)

    print "off.shape:", off.shape
    print "on.shape:", on.shape
    print "histrange:", histrange #WHY IS THIS GOING UP BY THE CHARGE GAIN EVERY TIME
    
    #Check the lengths, trim from the end to match.
    if (off.shape[0] <> on.shape[0]):
        print "Length mismatch, trimming."
        if off.shape[0] > on.shape[0]:
            off = off[: - (off.shape[0] - on.shape[0]),:,:]
        else:
            on = on[: - (on.shape[0] - off.shape[0]),:,:]

    if skip > 0:
        off = off[skip:,:,:]
        on = on[skip:,:,:]


    #Show median plots.
    if plotTraces:
        offMedian = []
        onMedian = []
        for i in range(off.shape[0]):
            offMedian.append(numpy.median(off[i,:,:]))
            onMedian.append(numpy.median(on[i,:,:]))
        plt.plot(range(len(offMedian)), offMedian)
        plt.plot(range(len(onMedian)), onMedian)
        plt.show()

    #This is for datasets that have the old method.
    off_plain = numpy.array(off)
    on_plain = numpy.array(on)

    if avg > 1:
        off = avgCube(off, avg = avg)
        on = avgCube(on, avg = avg)
    
    if readResetRead:
        print "Processing as read-reset-read..."
        newOff = numpy.zeros([off.shape[0]/2 - 1,off.shape[1],off.shape[2]])
        for i in range(newOff.shape[0]):
            newOff[i,:,:] = (off[i*2,:,:] - off[i*2 - 1,:,:])
        offSub = newOff
        newOn = numpy.zeros([on.shape[0]/2 - 1,on.shape[1],on.shape[2]])
        for i in range(newOn.shape[0]):
            newOn[i,:,:] = (on[i*2,:,:] - on[i*2 - 1,:,:])
        onSub = newOn
    else:
        if avg == 1:
            offSub = numpy.zeros([off.shape[0] - 1,off.shape[1],off.shape[2]])
            onSub = numpy.zeros([on.shape[0] - 1,on.shape[1],on.shape[2]])
            for i in range(offSub.shape[0]):
                offSub[i,:,:] = (off[i + 1,:,:] - off[i,:,:])
                onSub[i,:,:] = (on[i + 1,:,:] - on[i,:,:])
        else:
            offSub = numpy.zeros([off.shape[0] - 2,off.shape[1],off.shape[2]])
            onSub = numpy.zeros([on.shape[0] - 2,on.shape[1],on.shape[2]])
            for i in range(offSub.shape[0]):
                offSub[i,:,:] = (off[i + 2,:,:] - off[i,:,:])
                onSub[i,:,:] = (on[i + 2,:,:] - on[i,:,:])

    #This is for datasets that have the old method.
    offSub_plain = numpy.zeros([off_plain.shape[0] - 2,off_plain.shape[1],off_plain.shape[2]])
    onSub_plain = numpy.zeros([on_plain.shape[0] - 2,on_plain.shape[1],on_plain.shape[2]])
    for i in range(offSub_plain.shape[0]):
        offSub_plain[i,:,:] = (off_plain[i + 1,:,:] - off_plain[i,:,:])
        onSub_plain[i,:,:] = (on_plain[i + 1,:,:] - on_plain[i,:,:])
    if avg > 1:
        offSub_plain = avgCube(offSub_plain, avg = avg)
        onSub_plain = avgCube(onSub_plain, avg = avg)
        

##    if medianSubtract:
##        print "Subtracting temporal median..."
##        for x in range(offSub.shape[2]):
##            xMedian =  numpy.median(offSub[:,:,x])
##            offSub[:,:,x] -= xMedian
##            onSub[:,:,x] -= xMedian

    #Do averaging.
##    if avg > 1:
##        offSub = avgCube(offSub, avg = avg) * avg
##        onSub = avgCube(onSub, avg = avg) * avg

    #Flip results for Leach data.
    offSub = -offSub
    onSub = -onSub
    print "Off Median:", numpy.median(offSub[:,:,:])
    print "Off Mean:", numpy.mean(offSub[:,:,:])
    print "On Median:", numpy.median(onSub[:,:,:])
    print "On Mean:", numpy.mean(onSub[:,:,:])

    #TEMPTEMPTEMP
##    plt.hist(onSub.flatten(), bins = 100, range = [-50,100])
##    plt.show()
##    return

    #Convert to e-
    chargeGain = 1.58 #e-/ADU ME1000
    offSub *= chargeGain
    onSub *= chargeGain
    offSub_plain *= chargeGain
    onSub_plain *= chargeGain
    binspacing *= chargeGain
    histRangeNew = [0, 0]
    histRangeNew[0] = int(chargeGain * histrange[0])
    histRangeNew[1] = int(chargeGain * histrange[1])

    #Compute bins.
    bins = (histRangeNew[1] - histRangeNew[0]) / binspacing
    y,binedges = histogram([offSub[:,:,1:].flatten()], bins = bins, range=histRangeNew)
    bincenters = 0.5*(binedges[1:]+binedges[:-1])
    y2,binedges2 = histogram([onSub[:,:,1:].flatten()], bins = bins, range=histRangeNew)

    y_plain,binedges_plain = histogram([offSub_plain[:,:,1:].flatten()], bins = bins, range=histRangeNew)
    y2_plain,binedges2_plain = histogram([onSub_plain[:,:,1:].flatten()], bins = bins, range=histRangeNew)

    print "y mean:", numpy.mean(y)
    print "y2 mean:", numpy.mean(y2)
    print "binedges[10]:", binedges[10]
    print "binedges2[10]:", binedges2[10]
    
##    y2,binedges2 = histogram(modelAPD(), bins = bins, range=histrange)

    if len(binedges) <> len(binedges2):
        print "Something's real screwed up dude!"
        return

##    print "Subtracting other side of curve..."

    ySub = numpy.array(y)
    i = 0
    while bincenters[i] < 0:
        ySub[-i] -= ySub[i]
        ySub[i] = 0
        i += 1
##
##    y2,binedges2 = histogram([onSub[:,:,1:].flatten()], bins = bins, range=histrange)
##    if len(binedges) <> len(binedges2):
##        print "Something's real screwed up dude!"
##        return
##
##    y2Sub = numpy.array(y2)
##    i = 0
##    while bincenters[i] < 0:
##        y2Sub[-i] -= y2Sub[i]
##        y2Sub[i] = 0
##        i += 1
##
##    print bincenters[len(bincenters)/2]
##
##    subsub = y2Sub - ySub

    #Histograms
##    plt.subplot(3,1,1)
##    plt.semilogy(bincenters, y, label = "LED off")
####    plt.plot(bincenters, ySub, label = "offsub")
##    plt.semilogy(bincenters, y2, label = "LED on")
####    plt.plot(bincenters, y2Sub, label = "onsub")
####    plt.plot(bincenters, subsub, label = "subsub")
####    plt.ylim([0,20000])
##    plt.xlim([0,histrange[1]])
##    plt.legend()
##
##    plt.title(title)

    #Analysis
##    plt.subplot(3,1,2)
####    plt.plot(bincenters[i:], subsub[i:] / y2[i:].astype('float'), label = "photons")
####    plt.plot(bincenters[i:], ySub[i:] / y2[i:].astype('float'), label = "dark")
####    plt.plot(bincenters[i:], (y2[i:] - y2Sub[i:]) / y2[i:].astype('float'), label = "rn")
##
##    #Where is zero here?
##    
##    
##    plt.semilogy(bincenters[i:], y2[i:] - y[i:], label = "photons")
##    plt.ylim([0,6000])
##    plt.legend(loc = 1)

    #Move i up to where the photons start.
    while y2[i] - y[i] < 0:
        i += 1
##    i = 1 #SPECIAL

    #Integrate the results to get false positive/false negative.
    falseNegatives = numpy.zeros(bincenters[i:].shape[0])
    falsePositives = numpy.zeros(bincenters[i:].shape[0])
    falsePositiveFrac = numpy.zeros(bincenters[i:].shape[0])
    falseNegatives_plain = numpy.zeros(bincenters[i:].shape[0])
    falsePositives_plain = numpy.zeros(bincenters[i:].shape[0])
    falsePositiveFrac_plain = numpy.zeros(bincenters[i:].shape[0])
    print "i:", i
    print "bincenters[i]:", bincenters[i]
    for j in range(i,len(bincenters)):
        FP = 0
        FN = 0
        P = 0
        N = 0
        P_plain = 0
        FP_plain = 0
        FN_plain = 0
        for k in range(i,j):
            #Below threshold values.
            if y2[k] - y[k] > 0:
                FN += y2[k] - y[k]
                FN_plain += y2_plain[k] - y_plain[k]
                P  += y2[k] - y[k]
                P_plain += y2_plain[k] - y_plain[k]
            N += y[k]
        for k in range(j,len(bincenters)):
            #Above threshold values.
            FP += y[k]
            FP_plain += y_plain[k]
            if y2[k] - y[k] > 0:
                P += y2[k] - y[k]
                P_plain += y2_plain[k] - y_plain[k]
            N += y[k]
        falsePositives[j - i] = float(FP) #/ float(P)
        falseNegatives[j - i] = float(FN) / float(P)
        falsePositiveFrac[j - i] = float(FP) / numpy.sum(y) #(float(P) + float(N))
        falsePositives_plain[j - i] = float(FP_plain) #/ float(P)
        falseNegatives_plain[j - i] = float(FN_plain) / float(P)
        falsePositiveFrac_plain[j - i] = float(FP_plain) / numpy.sum(y_plain) #(float(P) + float(N))

        
        
##        falsePositiveFrac[j - i] = float(FP) / 
##    plt.subplot(3,1,3)
##    plt.plot(bincenters[i:], falsePositives, label = "false positives")
##    plt.plot(bincenters[i:], falseNegatives, label = "false negatives")
##    plt.plot(bincenters[i:], falsePositiveFrac, label = "false positive frac")
##    plt.legend(loc = 1)
##    plt.ylim([0.0,1.0])
##    plt.show()

    
    falsePositiveFrac *= (256.e3 / (32. * avg))

    #Print out some quick stats for a table.
    findFNs = [0.50, 0.80,0.90,0.95,0.99,0.999]
    j = falseNegatives.shape[0] / 2
    for findValue in findFNs:
        while falseNegatives[j] > 1 - findValue:
            j -= 1
        print "Threshold:", round(bincenters[j + i]), round(1 - falseNegatives[j], 3), round(falsePositiveFrac[j], 2)
##        print "Threshold:", round(bincenters[j + i]), round(1 - falseNegatives_plain[j], 3), round(falsePositiveFrac_plain[j], 2)
   
##    plt.plot(bincenters[i:], falsePositives, 'b', label = "P(false positive)")
    fig, ax1 = plt.subplots()
##    ax1.plot(bincenters[i:], falseNegatives, 'b', label = "$P(\mathrm{False Negative})$")
    ax1.plot(bincenters[i:], 1 - falseNegatives, 'b', label = "Threshold Efficiency")
    ax1.plot(bincenters[i:], 1 - falseNegatives_plain, 'k', label = "TE No AVG")
    
##    plt.ylim([0.0,1.0])
    ax1.set_ylim([0.0,1.0])
    ax1.set_xlim([0,200])
    ax1.set_ylabel("Threshold Efficiency", color = 'b')
    ax1.set_xlabel("Threshold ($e^{-}$)")
    ax1.tick_params('y', colors = 'b')

    ax2 = ax1.twinx()
    ax2.set_ylim([0,20])
##    ax2.set_ylim([0,1])
    ax2.set_xlim([0,200])
    print falsePositiveFrac[0]
    ax2.plot(bincenters[i:], falsePositiveFrac, 'r', label = "False Positive Rate")
    ax2.plot(bincenters[i:], falsePositiveFrac_plain, 'g', label = "FP Rate No AVG")
    ax2.set_ylabel("False Positive Rate ($e^{-}s^{-1}pix^{-1}$)", color = 'r')
    ax2.tick_params('y', colors = 'r')
##    ax2.set_ylim([0,10])
    #plt.ylim([0,45])
    
##    plt.legend(loc = 2)
##    plt.xlim([0,200])
##    ax1.set_xlabel("Threshold ($e^{-}$)")
    plt.title("Threshold Performance for a 32 x 32 Subarray\n" +\
              "$V_{bias} = 14.5V$, M06665-23, $T = 62.5\mathrm{K}$, $N_{avg} = " + str(avg) + "$")
    plt.show()
    del falseNegatives, falsePositives, falsePositiveFrac, offSub, onSub, y, y2, histrange, fig, ax1, ax2,\
        binedges, binedges2, bincenters, binspacing, ySub, i, off, on
    

##    profile = signal.gaussian(size,FWHM)

def avalancheModel(V = 14.5):
    runs = 10000
    steps = 500
##    x = 0.36
    epsilon = 17.55

    umPerStep = 3.0 / steps
    
##    E_g = bandgap(x) #Cadmium fraction

    x = VfromCd(steps = steps)
    E_g = numpy.zeros([steps])
##    print x.shape, E_g.shape
    for i in range(steps):
        E_g[i] = bandgap(x[i])

    #Compute voltage from gain.
##    Vth = 6.8 * E_g #E_g in eV, Vth in V
##    Vth = 1.0
##    print "Vth:", Vth
##    V = (math.log(gain, 2) - 1) * (Vth / 2) + Vth
    

##    V *= 0.667

    #Calculate the probability that an electron is duplicated at each step.
##    pDup = math.log(gain, 2) / (steps - math.log(gain, 2) * E_g / (voltage / steps))
##    print "pDup:", pDup
##    print "pNull:", (1 - pDup)**steps

    Vth = 6.8 * E_g[int(0.64 * steps)]
    print "Vth:", Vth
    print "V:", V
    print "Predicted gain:", 2**((V - Vth)/(Vth / 2))

##    Vsteps = VfromCd(steps = (steps - 3000)) * V
##    print Vsteps.shape
    
    results = []
    avalanches = []

##    plt.plot(E_g)
##    plt.show()
    try:
        for dummy in range(runs):
            E = [0.0]
    ##        E = [V * 0.15 / 0.20]
    ##        startStep = int(steps * (0.44 + random.random() * (0.64 - 0.44)))
    ##        startStep = int(steps * 0.54)
            #Bring in the photon and wait for it to get absorbed.
            a = 0
            dL = 12.832 / steps #um
            wavelength = 1.7 #um of incident photon
            E_hv =  6.242e18 * h * c / (wavelength * 1e-6)#eV
            #Absorption coefficient equation from Kinch 2007.
    ##        while random.random()**2 > dL * 4 * (E_hv * (E_hv - E_g[a])) and a < steps - 1:
    ##            a += 1
    ##        print a, bandgap(x[a])
    ##        if a == steps:
    ##            print "Pass-through"
            a = int(0.64 * steps) #Just place it at the junction.
            #Now we have an electron, so run the avalanche.
            for s in range(a, steps):
                randoms = numpy.random.rand(len(E))
                for i in range(len(E)):
                    #Calculate full pDup
                    if E[i] > 0:
                        E_n = E[i] / E_g[s]
                        #Replace this equation. Not sure where I got it.
                        P = (2.25e14 * (E_g[s])**2 / (E_n**0.5 * epsilon**2)) * (E_n - 1)**3 #in s**-1
    ##                    print "P:", P
                        if P > 0:
                            #Need a time interval.
                            t = umPerStep * 1e-6 / ((2 * E[i] * 1.60218e-19) / 9.109e-31)**0.5
    ##                        print "P:", P
    ##                        print "t:", t
                            Pt = P * t
    ##                        print "E:",E[i]
    ##                        print "Pt:", Pt
                            if randoms[i] < Pt:
                                #Roll against energy loss function.
                                PEn = probDistEnergyLoss(E_n)
                                roll = random.random()
    ##                            dE_index = 0
    ##                            while PEn[dE_index] < roll and dE_index < len(PEn):
    ##                                dE_index += 1
                                dE_index = (numpy.abs(PEn - roll)).argmin()
                                dE = E_g[s] * (1 + dE_index / 100.)
    ##                            E[i] -= 2*E_g[s]
                                E[i] -= dE
                                E.append(E_g[s] * (1 + dE_index / 100.))
                if s < (0.87 * steps):
                    E = [Evalue + V / ((1.00 - 0.64) * steps) for Evalue in E] #Voltage is applied across junction & multiplication layer.
            results.append(len(E))
            if dummy % 10 == 0:
                print ".",
            if dummy % 100 == 0:
    ##            print ".",
                print len(E),
    except KeyboardInterrupt:
        print "Keyboard interrupt registered, plotting existing results."
        plt.hist(results, bins = numpy.max(results), range = [0,numpy.max(results)-1], log = True)
        plt.title("Monte Carlo Avalanche Simulation, Gain = " + str(int(round(meanGain))))
    ##    plt.xlabel("$e^{-}$")
        plt.xlabel("Avalanche Gain")
        plt.ylabel("$n$")
        plt.show()
            
    meanGain = numpy.mean(results)
    plt.hist(results, bins = numpy.max(results), range = [0,numpy.max(results)-1], log = True)
    plt.title("Monte Carlo Avalanche Simulation, Gain = " + str(int(round(meanGain))))
##    plt.xlabel("$e^{-}$")
    plt.xlabel("Avalanche Gain")
    plt.ylabel("$n$")
    plt.show()

def AvalancheProbabilityTest():
    q = 1.6e-19
    results = []
##    for i,E_n in enumerate(numpy.array(range(100,800)) / 100.):
    for i,E_n in enumerate([2,3,4,5,6,7,8]):
        integrand = []
        for x in (numpy.array(range(100,int(E_n * 100))) / 100.): #From Kinch book (7.9)
            top = ((2 * (E_n - x) + 1)**2 - 1)**0.5 * (2 * (E_n - x) + 1) * (2 * x - 1) * ((2 * x - 1)**2 - 1)**0.5
            bottom = x**4
##            integrand += (top / bottom) * 0.01
            integrand.append(top / bottom)
        plt.plot(numpy.array(range(100,int(E_n * 100))) / 100.,integrand)
##        print integrand
##        results.append(integrand)
##        P = 3.2e7 * q * (q / h)**3
##    
##    plt.plot(numpy.array(range(100,800)) / 100.,results)
##        plt.plot(numpy.array(range(100,800)) / 100.,results)
    plt.xlim([0,8])
##    plt.ylim([0,150])
    plt.xlabel("$\Delta{E}/E_{g}$")
    plt.ylabel("$F(\Delta{E}/E_{g})$")
    plt.show()

def probDistEnergyLoss(E_n):
    #Simple version that does not take into account |F1F2| integral variance with deltaK.
    #Update later.
    #Calculate the probability distribution against energy lost.
    integrand = []
    for x in (numpy.array(range(100,int(E_n * 100))) / 100.): #From Kinch book (7.9)
        top = ((2 * (E_n - x) + 1)**2 - 1)**0.5 * (2 * (E_n - x) + 1) * (2 * x - 1) * ((2 * x - 1)**2 - 1)**0.5
        bottom = x**4
        integrand.append(top / bottom)
    prob = integrand / numpy.sum(integrand)
    #Turn it into a cumulative probability.
    for i in range(1,len(prob)):
        prob[i] += prob[i - 1]
##    plt.plot(prob)
##    plt.show()
    return prob

def nu_sat(E, E_g):
    return (2 * E / (7e-2 * E_g * 9.109e-31 * (1 + E / E_g)))**0.5

def nusatTest():
    Eg = 0.531072544
    Erange = (numpy.array(range(0,1000)) / 100.) * Eg
    results = []
    for E in Erange:
        nusat = (2 * E / (7e-2 * Eg * 9.109e-31 * (1 + E / Eg)))**0.5
        results.append(nusat)
    plt.plot(Erange,results)
    plt.show()

def timeTest():
    import time
    start = time.time()
    roll = random.random()
    PEn = probDistEnergyLoss(8.0)
##    dE_index = 0
##    while PEn[dE_index] < roll and dE_index < len(PEn):
##        dE_index += 1
    dE_index = (numpy.abs(PEn - roll)).argmin()
    end = time.time()
    print end - start

def bandgap(x, T = 60):
    return -0.302 + 1.93 * x + 5.35e-4 * T * (1 - 2 * x) - 0.81 * x**2 + 0.832 * x**3

def VfromCd(steps = 1000, plot = False):
    #Define the series of composition changes.
    umReg = numpy.array([4.0, 0.333, 0.333, 0.333, 0.333, 0.333, 2.5, 0.667, 2.333, 0.667, 1.0]) #length in um
    xReg = numpy.array([0.66, 0.53, 0.512, 0.494, 0.476, 0.458, 0.44, 0.48, 0.36, 0.46, 0.476]) #Cd_x

    umTotal = numpy.sum(umReg)

    x = []
    for i, um in enumerate(umReg):
        for dummy in range(int(round((um * steps / umTotal)))):
            x.append(xReg[i])
    x = numpy.array(x)

    if x.shape[0] > steps:
        x = x[:steps]

    permittivity = 9.75 + ((x - 0.2) * 12.1875) #From Kinch 2007
    susceptibility = 1. / permittivity

    totalsusceptibility = numpy.sum(susceptibility)
    V = numpy.zeros(susceptibility.shape)
    V[0] = 1.0
    for i in range(1,susceptibility.shape[0]):
        V[i] = V[i - 1] - susceptibility[i] / totalsusceptibility

    if plot:
        plt.plot(x)
    ##    plt.plot(susceptibility)
        plt.plot(V)
        plt.show()
    else:
        return x

def absorptionProfile():
    absCoef = 1.138 #um^-1
    steps = 1000
    N = 1000
    E_g = 0.496 #eV
    absorbed = numpy.zeros([N])
    length = 2.5 #um
    for i in range(absorbed.shape[0]):
        depth = 0
        while random.random() > (length / steps) * absCoef:
            depth += length / steps
        absorbed[i] = depth
    plt.hist(absorbed)
    plt.show()

def fitGaussianToNegative(y, bincenters, spacing, pastMiddle = 1):
    #Fit a Gaussian profile to the negative side of a histogram.
    peak = max(y)
    #Now, go through the set until we find the FWHM on the negative side.
    i = 0
    FWHM = 0
    while y[i] < peak:
        i += 1
##        print y[i], peak
        if y[i] < (peak / 2):
            FWHM = bincenters[i] * -1
##            print "FWHM = ", FWHM
##        print "offset = ", i - len(bincenters) / 2
    offset = int(i - len(bincenters) / 2) + 8
##    print "offset:", offset
    size = len(bincenters)
    passes = 4
    fitSize = y.shape[0]/2 + pastMiddle #was 1
    for dummy in range(passes):
        #Fit the shape, then the amplitude.
        #Fit to offset.
        delta = 1.
        while delta >= 1:
            print delta
            gaussian = makeGaussian(size, FWHM, peak, offset, spacing)
            Rsquared = numpy.sum((y[:fitSize] - gaussian[:fitSize])**2)
            gaussian = makeGaussian(size, FWHM, peak, offset + delta, spacing)
            RsquaredUp = numpy.sum((y[:fitSize] - gaussian[:fitSize])**2)
            gaussian = makeGaussian(size, FWHM, peak, offset - delta, spacing)
            RsquaredDown = numpy.sum((y[:fitSize] - gaussian[:fitSize])**2)
            if RsquaredDown < Rsquared:
                offset -= delta
            elif RsquaredUp < Rsquared:
                offset += delta
            else:
                delta = 0.1
            print ".",
        print "!"
        #Fit to FWHM.
        delta = FWHM / 4.
        while delta > FWHM / 1.e5:
            gaussian = makeGaussian(size, FWHM, peak, offset, spacing)
            Rsquared = numpy.sum((y[:fitSize] - gaussian[:fitSize])**2)
            gaussian = makeGaussian(size, FWHM + delta, peak, offset, spacing)
            RsquaredUp = numpy.sum((y[:fitSize] - gaussian[:fitSize])**2)
            gaussian = makeGaussian(size, FWHM - delta, peak, offset, spacing)
            RsquaredDown = numpy.sum((y[:fitSize] - gaussian[:fitSize])**2)
            if RsquaredDown < Rsquared:
                FWHM -= delta
            elif RsquaredUp < Rsquared:
                FWHM += delta
            else:
                delta /= 2.
            print ".",
##            print "std:", FWHM
        print "!"
        delta = peak / 4.
        #Fit to peak.
        while delta > peak / 1.e5:
            gaussian = makeGaussian(size, FWHM, peak, offset, spacing)
            Rsquared = numpy.sum((y[:fitSize] - gaussian[:fitSize])**2)
            gaussian = makeGaussian(size, FWHM, peak + delta, offset, spacing)
            RsquaredUp = numpy.sum((y[:fitSize] - gaussian[:fitSize])**2)
            gaussian = makeGaussian(size, FWHM, peak - delta, offset, spacing)
            RsquaredDown = numpy.sum((y[:fitSize] - gaussian[:fitSize])**2)
            if RsquaredDown < Rsquared:
                peak -= delta
            elif RsquaredUp < Rsquared:
                peak += delta
            else:
                delta /= 2
            print ".",
##            print "Peak:", peak
        print "!"
    print "std:", FWHM
    gaussian = makeGaussian(size, FWHM, peak, offset, spacing)
    return gaussian

def makeGaussian(size, std, peak, offset, spacing):
    #size is size of gaussian array to be returned.
    #std is standard deviation = FWHM/2
    #peak is peak height
    #offset is x-offset, and can only be integers (# of points)
    #spacing is distance between points
    if offset == 0:
        profile = signal.gaussian(size, std / spacing) * peak
    elif offset < 0:
        profile = signal.gaussian(size + offset * -2, std / spacing) * peak
    else:
        profile = signal.gaussian(size + offset * 2, std / spacing)[offset:] * peak
    return profile

def Nov29Medians():
    filenames = ["151129_124917.fits",
             "151129_124926.fits",
             "151129_124931.fits",
             "151129_124936.fits",
             "151129_124941.fits",
             "151129_124946.fits"]
    align = True
    for f in filenames:
        data = openfits(f)
        m = []
        for i in range(data.shape[0]):
            m.append(numpy.median(data[i,:,:]))
        if align:
            offset = m[0]
            for i in range(len(m)):
                m[i] -= offset
        plt.plot(range(data.shape[0]),m)
    plt.ylabel("ADU")
    plt.xlabel("frame #")
    plt.legend(["Run #1",
                "Run #2",
                "Run #3",
                "Run #4",
                "Run #5",
                "Run #6"], loc = 3)
    plt.show()

def darkTrends31Jan2016(log = True):
    files = ["151228_144652darkMap.fits",
            "151228_165637darkMap.fits",
            "151228_190622darkMap.fits",
            "151228_211617darkMap.fits",
            "151228_232602darkMap.fits",
            "151229_013547darkMap.fits"]
##             "150624_darkMap_M02775-35_110V.fits",
##             "150624_darkMap_M02775-35_115V.fits",
##             "150629_100144darkMap.fits",
##             "150629_102204darkMap.fits"]
    biases = [2.5, 4.5, 6.5, 8.5, 9.5, 10.5]
    percentage = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    colors = ['b','r','g','k','m','y','c']
    for i,p in enumerate(percentage):
        medians = []
        for f in files:
            d = openfits(f)
            sortedList = numpy.sort(d[-32:,160:192].flatten())
            dark = sortedList[int(p * len(sortedList))]
            print dark,
            medians.append(dark * 2.89)
##            medians.append(numpy.median(d[-32:,160:192]))
        if log:
            plt.semilogy(biases, medians, colors[i] + 'o-')
        else:
            plt.plot(biases, medians, colors[i] + 'o-')
##        plt.semilogy(biases[:-2], medians[:-2], colors[i] + 'o-')
##        plt.semilogy(biases[-3:], medians[-3:], colors[i] + 'o--')
    plt.xlabel("Bias Voltage (V)")
    plt.ylabel("e-/s")
    plt.legend(["5%","10%","25%","50%","75%","90%","95%"], loc = 2)
    plt.title("Percentile Dark Current (no gain correction) vs. Bias\nM06715-27 @ 60K Data:28-29 Dec 2015")
    plt.xlim([0,11])
    plt.show()

def darkRamps31Jan2016():
    files =["151228_144652darkMap.fits",
            "151228_165637darkMap.fits",
            "151228_190622darkMap.fits",
            "151228_211617darkMap.fits",
            "151228_232602darkMap.fits",
            "151229_013547darkMap.fits"]
##    w = [[224,256],[160,192]] #Top center. (standard)
    w = [[224,256],[160,192]] #Top center. (standard)
    biases = [2.5, 4.5, 6.5, 8.5, 9.5, 10.5]
    medians = []
    for f in files:
        print "file:", f
        d = openfits(f)
        dark = numpy.median(d[w[0][0]:w[0][1],w[1][0]:w[1][1]])
        medians.append(dark * 2.89)
    plt.plot(biases, medians, 'o-')
    plt.xlabel("Bias (V)")
    plt.ylabel("e-/s")
    plt.title("Median Dark Current (no gain correction) vs. Bias Voltage\nM06715-27 @ 60K Data:28-29 Dec 2015")
    plt.show()

def darkRamps2Feb2016(log = False, gainCorrect = False, vsBiases = False):
    masterfiles =[
                  ["151227_113650darkMap.fits",#40K
                   "151227_134635darkMap.fits",
                   "151227_155625darkMap.fits",
                   "151227_180610darkMap.fits",
                   "151227_201555darkMap.fits",
                   "151227_222540darkMap.fits"],
                  ["151226_143121darkMap.fits",#50K
                   "151226_164111darkMap.fits",
                   "151226_185056darkMap.fits",
                   "151226_210041darkMap.fits",
                   "151226_231026darkMap.fits",
                   "151227_012012darkMap.fits"],
                  ["151228_144652darkMap.fits",#60K
                   "151228_165637darkMap.fits",
                   "151228_190622darkMap.fits",
                   "151228_211617darkMap.fits",
                   "151228_232602darkMap.fits",
                   "151229_013547darkMap.fits"],
                  ["151224_160532darkMap.fits",#65K
                   "151224_181517darkMap.fits",
                   "151224_202502darkMap.fits",
                   "151224_223447darkMap.fits",
                   "151225_004437darkMap.fits",
                   "151225_025432darkMap.fits"],
                  ["151222_170403darkMap.fits",#70K
                   "151222_191353darkMap.fits",
                   "151222_212338darkMap.fits",
                   "151222_233333darkMap.fits",
                   "151223_014323darkMap.fits",
                   "151223_035308darkMap.fits"],
                  ["151223_151856darkMap.fits",#75K
                   "151223_172841darkMap.fits",
                   "151223_193826darkMap.fits",
                   "151223_214816darkMap.fits",
                   "151223_235806darkMap.fits",
                   "151224_020756darkMap.fits"],
                  ["160223_123426darkMap.fits",#80K
                   "160223_144411darkMap.fits",
                   "160223_165356darkMap.fits",
                   "160223_190341darkMap.fits",
                   "160223_211326darkMap.fits",
                   "160223_232311darkMap.fits"]]
    w = [[224,256],[160,192]] #Top center. (standard)
    biases = [2.5, 4.5, 6.5, 8.5, 9.5, 10.5]
    gains =  [1.0, 1.5, 3.0, 5.2, 7.6, 12.5]
    for files in masterfiles:
        medians = []
        for f in files:
            print "file:", f
            d = openfits(f)
            dark = numpy.median(d[w[0][0]:w[0][1],w[1][0]:w[1][1]])
            medians.append(dark * 2.89)
        if gainCorrect:
            for i in range(len(medians)):
                medians[i] /= gains[i]
        if log:
            plt.semilogy(biases, medians, 'o-')
        else:
            plt.plot(biases, medians, 'o-')
    plt.xlabel("Bias (V)")
    plt.ylabel("e-/s")
    plt.legend(["40K","50K","60K","65K","70K","75K","80K"], loc = 2)
##    plt.ylim(0, 0.25)
    plt.xlim(0, 11)
    title = "Median Dark Current"
    if gainCorrect:
        title += " (gain corrected)"
    else:
        title += " (no gain correction)"
    title +=  " vs. Bias Voltage\nM06715-27 @ 40 - 65K Data: 22 - 29 Dec 2015"
    plt.title(title)
    plt.show()

def darkTrends2Feb2016(log = True, gainCorrect = False, T = 60, gainXaxis = False):
    if T == 40:
        files =["151227_113650darkMap.fits",#40K
                "151227_134635darkMap.fits",
                "151227_155625darkMap.fits",
                "151227_180610darkMap.fits",
                "151227_201555darkMap.fits",
                "151227_222540darkMap.fits"]
    elif T == 50:
        files =["151226_143121darkMap.fits",
                "151226_164111darkMap.fits",
                "151226_185056darkMap.fits",
                "151226_210041darkMap.fits",
                "151226_231026darkMap.fits",
                "151227_012012darkMap.fits"]
    elif T == 60:
        files =["151228_144652darkMap.fits",
                "151228_165637darkMap.fits",
                "151228_190622darkMap.fits",
                "151228_211617darkMap.fits",
                "151228_232602darkMap.fits",
                "151229_013547darkMap.fits"]
    elif T == 65:
        files =["151224_160532darkMap.fits",#65K
                "151224_181517darkMap.fits",
                "151224_202502darkMap.fits",
                "151224_223447darkMap.fits",
                "151225_004437darkMap.fits",
                "151225_025432darkMap.fits"]
    elif T == 70:
        files =["151222_170403darkMap.fits",#70K
                "151222_191353darkMap.fits",
                "151222_212338darkMap.fits",
                "151222_233333darkMap.fits",
                "151223_014323darkMap.fits",
                "151223_035308darkMap.fits"]
    elif T == 75:
        files =["151223_151856darkMap.fits",#75K
                "151223_172841darkMap.fits",
                "151223_193826darkMap.fits",
                "151223_214816darkMap.fits",
                "151223_235806darkMap.fits",
                "151224_020756darkMap.fits"]
    elif T == 85:
        files =["160229_123223darkMap.fits",
                "160229_144213darkMap.fits",
                "160229_165158darkMap.fits",
                "160229_190143darkMap.fits",
                "160229_211128darkMap.fits",
                "160229_232113darkMap.fits"]
    biases = [2.5, 4.5, 6.5, 8.5, 9.5, 10.5]
##    percentage = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    percentage = [0.25, 0.50, 0.75, 0.85, 0.90, 0.95, 0.99]
    gains =  [1.0, 1.5, 3.0, 5.2, 7.6, 12.5] #1.7um
##    gains =  [1.0, 1.0, 1.31, 1.67, 1.95, 2.52] #3.1um
##    colors = ['b','r','g','k','m','y','c']
    colors = ['m','k','b','c','g','y','r']
    for i,p in enumerate(percentage):
        medians = []
        for f in files:
            d = openfits(f)
            sortedList = numpy.sort(d[-32:,160:192].flatten())
            dark = sortedList[int(p * len(sortedList))]
            print dark,
            medians.append(dark * 2.89)
##            medians.append(numpy.median(d[-32:,160:192]))
        if gainCorrect:
            for m in range(len(medians)):
                medians[m] /= gains[m]
        if gainXaxis:
            Xaxis = gains
        else:
            Xaxis = biases
        if log:
            plt.semilogy(Xaxis, medians, colors[i] + 'o-')
        else:
            plt.plot(Xaxis, medians, colors[i] + 'o-')
##        plt.semilogy(biases[:-2], medians[:-2], colors[i] + 'o-')
##        plt.semilogy(biases[-3:], medians[-3:], colors[i] + 'o--')
    if gainXaxis:
        plt.xlabel("APD gain")
    else:
        plt.xlabel("$V_{bias}$")
    ylab = "$e^{-}/\mathrm{s}$"
    if gainCorrect:
        ylab += " (gain-corrected)"
    plt.ylabel(ylab)
##    plt.legend(["5%","10%","25%","50%","75%","90%","95%"], loc = 2)    
    plt.legend(["25%","50%","75%","85%","90%","95%","99%"], loc = 2)
    title = "Percentile Dark Current"
    if gainCorrect:
        title += " (gain-corrected)"
        ymax = 0.30
    else:
        title += " (no gain correction)"
        ymax = 100.0
##    title +=  " vs. Bias Voltage\nM06715-27 @ " + str(T) + "K, Data Recorded: "
    title +=  " vs. Bias Voltage\nMark 14 M06715-27, reset interval 5s, " + str(T) + "K"
##    ymax = 0.25
##    if T == 40:
##        title += "27 - 28 Dec 2015"
##    elif T == 50:
##        title += "26 - 27 Dec 2015"
##    elif T == 60:
##        title += "28 - 29 Dec 2015"
##    elif T == 65:
##        title += "24 - 25 Dec 2015"
##        ymax = 0.5
##    elif T == 70:
##        title += "22 - 23 Dec 2015"
##        ymax = 1.0
##    elif T == 75:
##        title += "23 - 24 Dec 2015"
##        ymax = 1.5
    plt.title(title)
##    plt.xlim([0,11])
    plt.xlim([2,11])
    plt.ylim([0, ymax])
    plt.show()

def darkRampsMark3OldData4Feb2016(log = False, gainCorrect = False):
    files = ["150624_darkMap_M02775-35_25V.fits",
             "150624_darkMap_M02775-35_35V.fits",
             "150624_darkMap_M02775-35_45V.fits",
             "150624_darkMap_M02775-35_55V.fits",
             "150624_darkMap_M02775-35_65V.fits",
             "150624_darkMap_M02775-35_75V.fits",
             "150624_darkMap_M02775-35_85V.fits",
             "150624_darkMap_M02775-35_95V.fits",
             "150624_darkMap_M02775-35_105V.fits",
             "150624_darkMap_M02775-35_115V.fits",
             "150629_darkMap_M02775-35_125V.fits",
             "150629_darkMap_M02775-35_135V.fits"]
    w = [[224,256],[160,192]] #Top center. (standard)
    biases = [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5]
    gains =  [1.0, 1.3, 1.6, 2.2, 2.8, 5.5, 8.4,14.2, 22.1, 34.7, 55.0, 88.7]
    medians = []
    for f in files:
        print "file:", f
        d = openfits(f)
        dark = numpy.median(d[w[0][0]:w[0][1],w[1][0]:w[1][1]])
        medians.append(dark * 2.89)
    if gainCorrect:
        for i in range(len(medians)):
            medians[i] /= gains[i]
    if log:
        plt.semilogy(biases, medians, 'o-')
    else:
        plt.plot(biases, medians, 'o-')
    plt.xlabel("Bias (V)")
    plt.ylabel("e-/s")
    plt.ylim(0, 8.0)
    plt.xlim(0, 14)
    title = "Median Dark Current"
    if gainCorrect:
        title += " (gain corrected)"
    else:
        title += " (no gain correction)"
    title +=  " vs. Bias Voltage\nM02775-35 @ 60K Data: 24, 29 Jun 2015"
    plt.title(title)
    plt.show()

def mk3vsmk14Dark9Feb2016(log = False, gainCorrect = True, mimicRaytheon = False):
    files = ["150624_darkMap_M02775-35_25V.fits",
             "150624_darkMap_M02775-35_35V.fits",
             "150624_darkMap_M02775-35_45V.fits",
             "150624_darkMap_M02775-35_55V.fits",
             "150624_darkMap_M02775-35_65V.fits",
             "150624_darkMap_M02775-35_75V.fits",
             "150624_darkMap_M02775-35_85V.fits",
             "150624_darkMap_M02775-35_95V.fits",
             "150624_darkMap_M02775-35_105V.fits",
             "150624_darkMap_M02775-35_115V.fits",
             "150629_darkMap_M02775-35_125V.fits",
             "150629_darkMap_M02775-35_135V.fits"]
    w = [[224,256],[160,192]] #Top center. (standard)
    biases = [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5]
    gains =  [1.0, 1.3, 1.6, 2.2, 2.8, 5.5, 8.4,14.2, 22.1, 34.7, 55.0, 88.7]
    medians = []
    for f in files:
        print "file:", f
        d = openfits(f)
        dark = numpy.median(d[w[0][0]:w[0][1],w[1][0]:w[1][1]])
        print dark * 2.89
        medians.append(dark * 2.89)
    if gainCorrect:
        for i in range(len(medians)):
            medians[i] /= gains[i]
            print medians[i]
    if mimicRaytheon:
        biases = -numpy.array(biases)
    if log:
        plt.semilogy(biases, medians, 'bo-')
    else:
        plt.plot(biases, medians, 'bo-')

    files =["151228_144652darkMap.fits",
            "151228_165637darkMap.fits",
            "151228_190622darkMap.fits",
            "151228_211617darkMap.fits",
            "151228_232602darkMap.fits",
            "151229_013547darkMap.fits"]
    biases = [2.5, 4.5, 6.5, 8.5, 9.5, 10.5]
    gains =  [1.0, 1.5, 3.0, 5.2, 7.6, 12.5]
    medians = []
    for f in files:
        print "file:", f
        d = openfits(f)
        dark = numpy.median(d[w[0][0]:w[0][1],w[1][0]:w[1][1]])
        print dark * 2.89
        medians.append(dark * 2.89)
    if gainCorrect:
        for i in range(len(medians)):
            medians[i] /= gains[i]
            print medians[i]
    if mimicRaytheon:
        biases = -numpy.array(biases)
    if log:
        plt.semilogy(biases, medians, 'ro-')
    else:
        plt.plot(biases, medians, 'ro-')
    
    plt.xlabel("$V_{bias}$")
    plt.ylabel("$e^{-}/\mathrm{s}$")
    title = "Median Dark Current"
    if gainCorrect:
        title += " (gain corrected)"
        plt.ylim(0, 8.0)
    else:
        title += " (no gain correction)"
        plt.ylim(0, 1000.0)
    if mimicRaytheon:
        plt.xlim(-20, 0)
        plt.legend(["Mk 3 M02775-35","Mk 14 M06715-27"], loc = 0)
        plt.ylim(1e-2, 1e5)
    else:
        plt.xlim(0, 14)
        plt.legend(["Mk 3 M02775-35","Mk 14 M06715-27"], loc = 2)
        
    title +=  " vs. Bias Voltage\n Mk 3 & 14 @ 60K Data: 24, 29 Jun, 28 - 29 Dec 2015"
    plt.title(title)
    plt.show()

def darkRamps1Mar2016(logY = False, logX = False, gainCorrect = False, gainXaxis = False):
    masterfiles =[
                  ["151227_113650darkMap.fits",#40K
                   "151227_134635darkMap.fits",
                   "151227_155625darkMap.fits",
                   "151227_180610darkMap.fits",
                   "151227_201555darkMap.fits",
                   "151227_222540darkMap.fits"],
                  ["151226_143121darkMap.fits",#50K
                   "151226_164111darkMap.fits",
                   "151226_185056darkMap.fits",
                   "151226_210041darkMap.fits",
                   "151226_231026darkMap.fits",
                   "151227_012012darkMap.fits"],
                  ["151228_144652darkMap.fits",#60K
                   "151228_165637darkMap.fits",
                   "151228_190622darkMap.fits",
                   "151228_211617darkMap.fits",
                   "151228_232602darkMap.fits",
                   "151229_013547darkMap.fits"],
                  ["151224_160532darkMap.fits",#65K
                   "151224_181517darkMap.fits",
                   "151224_202502darkMap.fits",
                   "151224_223447darkMap.fits",
                   "151225_004437darkMap.fits",
                   "151225_025432darkMap.fits"],
                  ["151222_170403darkMap.fits",#70K
                   "151222_191353darkMap.fits",
                   "151222_212338darkMap.fits",
                   "151222_233333darkMap.fits",
                   "151223_014323darkMap.fits",
                   "151223_035308darkMap.fits"],
                  ["151223_151856darkMap.fits",#75K
                   "151223_172841darkMap.fits",
                   "151223_193826darkMap.fits",
                   "151223_214816darkMap.fits",
                   "151223_235806darkMap.fits",
                   "151224_020756darkMap.fits"],
                  ["160229_123223darkMap.fits",#85K
                   "160229_144213darkMap.fits",
                   "160229_165158darkMap.fits",
                   "160229_190143darkMap.fits",
                   "160229_211128darkMap.fits"]]
##                   "160229_232113darkMap.fits"]]
    w = [[224,256],[160,192]] #Top center. (standard)
    biases = [2.5, 4.5, 6.5, 8.5, 9.5, 10.5]
    gains =  [1.0, 1.5, 3.0, 5.2, 7.6, 12.5]
    colors = ['m','b','k','c','g','y','r']
    for j,files in enumerate(masterfiles):
        medians = []
        for f in files:
            print "file:", f
            d = openfits(f)
            dark = numpy.median(d[w[0][0]:w[0][1],w[1][0]:w[1][1]])
            medians.append(dark * 2.89)
        if gainCorrect:
            for i in range(len(medians)):
                medians[i] /= gains[i]
        if gainXaxis:
            Xaxis = gains
        else:
            Xaxis = biases
        if logY:
            if logX:
                plt.loglog(Xaxis[:len(medians)], medians, colors[j] + 'o-')
            else:
                plt.semilogy(Xaxis[:len(medians)], medians, colors[j] + 'o-')
        else:
            if logX:
                plt.semilogx(Xaxis[:len(medians)], medians, colors[j] + 'o-')
            else:
                plt.plot(Xaxis[:len(medians)], medians, colors[j] + 'o-')
    ylab = "$e^{-}/\mathrm{s}$"
    if gainCorrect:
        ylab += " (gain-corrected)"
    plt.ylabel(ylab)
##    plt.legend(["40K","50K","60K","65K","70K","75K","85K"], loc = 2)
    plt.legend(["42.3K","52.4K","62.5K","67.6K","72.6K","77.7K","87.8K"], loc = 2)
##    plt.ylim(0, 500)
    title = "Median Dark Current"
    if gainCorrect:
        title += " (gain-corrected)"
        plt.ylim(0, 20.0)
    else:
##        title += " (no gain correction)"
        plt.ylim(0.01, 200.0)
    if gainXaxis:
        plt.xlim(0, 13)
        plt.xlabel("APD gain")
        title += " vs. APD Gain"
    else:
        plt.xlim(0, 11)
##        plt.xlim(2, 11)
        plt.xlabel("$V_{bias}$")
        title +=  " vs. Bias Voltage"
##    title += "\nMark 14 M06715-27, $T = 40 - 85\mathrm{K}$, Data: 22 - 29 Dec 2015 & 29 Feb 2016"
    title += "\nMark 14 M06715-27, read interval 5s, $T = 42 - 88\mathrm{K}$"
    plt.title(title)
    plt.show()

def tempCorrect():
    Tmeas = [60,100]
    Tdet = [62.5,102.9]
    m,b = linearfit(Tmeas,Tdet)
    Tnc = [40, 50, 60, 65, 70, 75, 85]
    for T in Tnc:
        print T, T * m + b

def cubePFfiles():
    chdir('./stablevoltage')
    fileList = listdir('./')
    first = openfits(fileList[0])
    cube = numpy.zeros([1, first.shape[0], first.shape[1]])
    cube[0,:,:] = first
    fileLength = len(fileList[0])
    for j in range(4):
        for i in range(1, len(fileList)):
            if len(fileList[i]) == fileLength + j:
                image = openfits(fileList[i])
                newCube = numpy.zeros([cube.shape[0] + 1, cube.shape[1], cube.shape[2]])
                newCube[:-1,:,:] = cube
                newCube[-1,:,:] = image
                del cube
                cube = newCube
    savefits("combined.fits",cube)

def rowMedianPlot(filename):
    d = openfits(filename)
    medians = []
    for i in range(d.shape[0]):
        for x in range(d.shape[1]):
            medians.append(numpy.median(d[i,x,:]))
    plt.plot(medians)
    plt.show()

def timeMedianPlot(filename):
    d = openfits(filename)
    medians = []
    for i in range(d.shape[0]):
        for y in range(d.shape[1]):
            for x in range(d.shape[2] / 32):
                medians.append(numpy.median(d[i,-y,x*32:(x+1)*32]))
    plt.plot(medians)
    plt.show()

def rowReadNoise(filename, detrend = True):
##    d = cdsCube(openfits(filename))[:,:1,:32]
    d = openfits(filename)
    #Detrend the data?
    if detrend:
        print "Detrending..."
        for n in range(d.shape[0]):
            for y in range(d.shape[1]):
                for x in range(d.shape[2]/32):
                    d[n,y,x*32:(x+1)*32] -= numpy.median(d[n,y,x*32:(x+1)*32])
    #Do a CDS, since that's how we get the actual read noise.
    cds = cdsCube(d)
    #Take the standard deviations.
    stddevs = numpy.zeros([cds.shape[2]])    
    for x in range(cds.shape[2]):
        stddevs[x] = numpy.std(cds[:,:,x], ddof = 1)
    sd = numpy.median(stddevs.flatten())
    print "CDS read noise:"
    print sd, "ADU rms"
    print sd * 2.89, "e- rms"
    
def convertToElectrons(filename, newfilename):
    d = openfits(filename)
    d *= 2.9
    savefits(newfilename, d)

def plot4():
    lightsoff = ["150404_COM2Voff-4-0.fits","150404_COM1Voff-6-0.fits","150404_COM0Voff-8-0.fits",
                 "150404_COM-1Voff-10-0.fits"]
    lightson =  ["150404_COM2Von-5-0.fits","150404_COM1Von-7-0.fits","150404_COM0Von-9-0.fits",
                 "150404_COM-1Von-11-0.fits"]
    f, axarr = plt.subplots(2,2)
    for i in range(len(lightsoff)):
        d1 = openfits(lightsoff[i])
        d2 = openfits(lightson[i])
        medians1 = []
        medians2 = []
        for n in range(d1.shape[0]):
            medians1.append(numpy.median(d1[n,:,d1.shape[2]/2:]))
            medians2.append(numpy.median(d2[n,:,d2.shape[2]/2:]))
        axarr[i/2,i%2].plot(medians1)
        axarr[i/2,i%2].plot(medians2)
        axarr[i/2,i%2].set_title(i)
    plt.show()

def PCPBtest():
##    off = openfits("PCoff-3-0.fits")
##    off = openfits("160408_highbiastest-15V-10-0.fits")
##    plt.title("1.000MHz COM = -15V (8 Apr 2016)")
##    off = openfits("160408_testrow-10V-12-0.fits")
##    off = openfits("160411_1.428MHztest-3-0.fits")
##    plt.title("1.428MHz COM = -15V (11 Apr 2016)")
##    off = openfits("160411_1.000MHztest-4-0.fits")
##    plt.title("1.000MHz COM = -15V (11 Apr 2016)")
    off = openfits("160411_1MHzTest-1-0.fits")
##    on = openfits("PCon-4-0.fits")
##    medians = []
##    onMed = []
    for x in range(8):
        medians = []
        for i in range(off.shape[0]):
    ##        offMed.append(numpy.mean(off[i]))
            medians.append(off[i,0,x])
        plt.plot(medians, label="channel #" + str(x))
        
##    for i in range(on.shape[0]):
####        onMed.append(numpy.mean(on[i]))
##        onMed.append(off[i,0])
##    plt.plot(onMed)
##    plt.legend(["LED off","LED on"], loc = 2)
    plt.legend(loc = 2)
    plt.ylabel("ADU")
    plt.xlabel("Frame #")
    plt.show()

def PCPBEveryOther():
##    off = openfits("PCoff-3-0.fits")
##    off = openfits("160408_highbiastest-15V-10-0.fits")
##    plt.title("1.000MHz COM = -15V (8 Apr 2016)")
##    off = openfits("160408_testrow-10V-12-0.fits")
##    off = openfits("160411_1.428MHztest-3-0.fits")
##    plt.title("1.428MHz COM = -15V (11 Apr 2016)")
##    off = openfits("160411_1.000MHztest-4-0.fits")
##    plt.title("1.000MHz COM = -15V (11 Apr 2016)")
    
##    off = openfits("160413_EveryOtherFIlterTest-2-0.fits")
    off = openfits("160413_64x1Test-1-0.fits")
    print off.shape
##    on = openfits("PCon-4-0.fits")
##    medians = []
##    onMed = []
    for x in range(0,8):
        medians = []
        for i in range(off.shape[0]):
    ##        offMed.append(numpy.mean(off[i]))
##            for n in range(0,2):
            medians.append(off[i,0,x])
        plt.plot(medians, label="channel #" + str(x))
        
##    for i in range(on.shape[0]):
####        onMed.append(numpy.mean(on[i]))
##        onMed.append(off[i,0])
##    plt.plot(onMed)
##    plt.legend(["LED off","LED on"], loc = 2)
    plt.legend(loc = 2)
    plt.ylabel("ADU")
    plt.xlabel("Frame #")
    plt.show()
    

def stdbypix():
##    d = openfits("PCoff-3-0.fits")
    d = openfits("PCoff-3-0.fits")
    d2 = openfits("PCon-4-0.fits")
    d3 = openfits("160408_highbiastest-15V-10-0.fits")
    d4 = openfits("160408_testrow128-11-0.fits")
    d5 = openfits("160408_testrow-10V-12-0.fits")
    d6 = openfits("160411_1.000MHztest-4-0.fits")
    for x in range(d.shape[2]):
##        print x,":", numpy.std(d[:,0,x], ddof = 1)
        print x,":", max(d[5000:5010,0,x]) - min(d[5000:5010,0,x]), " | ", \
              max(d2[5000:5010,0,x]) - min(d2[5000:5010,0,x]), " | ",\
              max(d3[5000:5010,0,x]) - min(d3[5000:5010,0,x]), " | ",\
              max(d4[5000:5010,0,x]) - min(d4[5000:5010,0,x]), " | ",\
              max(d5[5000:5010,0,x]) - min(d5[5000:5010,0,x]), " | ",\
              max(d6[5000:5010,0,x]) - min(d6[5000:5010,0,x])

def plotFrstPixels(filenames):
    for f in filenames:
        d = openfits(f)
        plt.plot(d[:,0,3], label = f)
    plt.legend()
    plt.show()

def compareNoises(offset = 0):
    baseFilename = "what_the_hell-02-00"
    points = []
##    offset = 2
    bits = numpy.zeros([16])
    for i in range(20):
        if len(str(i)) < 2:
            f = baseFilename + "0" + str(i) + ".fits"
        else:
            f = baseFilename + str(i) + ".fits"
        d = openfits(f)
        for y in range(d.shape[0]):
            for x in range(d.shape[1] / 32):
                points.append(d[y,x*32 + offset])
                for b in range(bits.shape[0]):
                    if d[y,x*32 + offset] % 2**(b+1) >= 2**b:
                        bits[b] += 1
    plt.plot(points)
    for b in range(bits.shape[0]):
        print b, bits[b]
    d = openfits("cube32x1-1-0.fits")
    points = []
    for i in range(d.shape[0]):
        points.append(d[i,0,0])
    plt.plot(points)
    plt.show()
    
def plotFirst8Pixels(f):
    d = openfits(f)
    for x in range(8):
        plt.plot(d[:,0,x], label = str(x))
    plt.legend()
    plt.show()

def analyzeRRRframes():
    fileList = listdir("./")
    Amedians = []
    Bmedians = []
    pix0off = []
    pix0on = []
    for f in fileList:
##        if "RR_offset_1A" in f:
##        if "full_frame_4_28" in f:
        if "RRR_full_frame_high_gain" in f:
            d = openfits(f)
            index = f[-9:-5]
            print index
            for y in range(96,160,2):
##                for y in range(2,256,2):
                Amedians.append(numpy.mean(d[y,160:192]))
                Bmedians.append(numpy.mean(d[y - 1, 160:192]))
##                pix0off.append(d[160,16])
##                pix0on.append(d[161,16])
                
    Amedians = Amedians[256:]
    Bmedians = Bmedians[:-256]
    plt.plot(Amedians)
    plt.plot(Bmedians)
    plt.plot(numpy.array(Amedians) - numpy.array(Bmedians))
    print numpy.std(numpy.array(Amedians) - numpy.array(Bmedians), ddof = 1)
    plt.show()

##    pix0off = pix0off[96:]
##    pix0on = pix0on[:-96]
            
##    pix0off = pix0off[1:]
##    pix0on = pix0on[:-1]
    
##    plt.plot(pix0off)
##    plt.plot(pix0on)
##    subtracted = numpy.array(pix0off) - numpy.array(pix0on)
##    print numpy.std(subtracted[20:], ddof = 1)
##    plt.plot(subtracted)
##    plt.show()
    
def analyze320x1():
##    d = openfits("sub_320x1_LEDOFF_4_28-5-0.fits")
##    d = openfits("reset_1us_320x1-2-0.fits") #low gain
##    d = openfits("reset_1us_320x1_high_gain-9-0.fits")
    d = openfits("reset_100ns_320x1_high_gain-1-0.fits")
##    d = openfits("reset_100ns_run3-03-0040.fits")
    A = []
    B = []
    print d.shape
    for n in range(2,d.shape[0],2):
        for y in range(d.shape[1]):
            for offset in range(10):
                A.append(d[n,y,offset * 32])
                B.append(d[n - 1,y,offset * 32])
    plt.plot(A, label = "pre-reset")
    plt.plot(B, label = "post-reset")
    plt.plot(numpy.array(A) - numpy.array(B), label = "subtracted")
    print numpy.std(numpy.array(A[370:]) - numpy.array(B[370:]), ddof = 1)
    plt.legend()
    plt.show()
    cds = numpy.array(A[:]) - numpy.array(B[:])
    cds = cds - cds[0]
##    plt.hist(cds, bins = 2000, range = [-1000,1000])
##    plt.show()

def noiseOn32x1():
    off = openfits("subarray_32x1_LEDOFF_4_28-1-0.fits")
    print off.shape
    medians = []
    for n in range(2,off.shape[0],2):
        medians.append(numpy.median(off[n, 0, :]) - numpy.median(off[n + 1, 0, 0]))
    plt.plot(medians)
    print numpy.std(numpy.array(medians).flatten(), ddof = 1)
    plt.show()

def fftPBdata():
    d = openfits("subarray_32x1_LEDOFF_4_28-1-0.fits")
    r = []
    for n in range(2, d.shape[0], 2):
        r.append(abs(d[n,0,0] - d[n + 1, 0,0]))
    r = numpy.array(r)
    f = abs(numpy.fft.rfft(r))
    freq = numpy.fft.rfftfreq(r.shape[0], d = 1. / 333.e3)
    plt.plot(freq, f / numpy.sqrt(freq))
    plt.show()

def resetEffectCheck():
##    d = openfits("reset_100ns_run3-03-0040.fits")
##    d = openfits("reset_1us_low_gain-01-0009.fits")
    d = openfits("reset_1us_320x1-2-0.fits")
    medians = []
    for n in range(2,d.shape[0],2):
        for xOff in range(0,320,32):
            medians.append(numpy.median(d[n,0,xOff:xOff + 32]) - numpy.median(d[n - 1,0,xOff:xOff + 32]))
    plt.plot(medians)
    plt.show()

def medianPlotRRR(filename):
    d = openfits(filename)
    medians = []
    for i in range(2,d.shape[0],2):
        medians.append(numpy.median(d[i,:,:] - d[i - 1,:,:]))
    plt.plot(medians)
    plt.show()
    plt.hist(medians, bins = 1000, range = [500,1500])
    plt.show()

def reducerrr(d):
    d2 = numpy.zeros([d.shape[0]/2 - 1,d.shape[1],d.shape[2]])
    for i in range(d2.shape[0]):
        d2[i,:,:] = d[i*2,:,:] - d[i*2 - 1,:,:]
    return d2


def excessNoisePlot20Jun2016():
    files = ["160619_PB32x1-7-0.fits",
             "160619_RN1V-10-0.fits",
             "160619_RN0V-11-0.fits",
             "160619_RN-1V-12-0.fits",
             "160619_RN-2V-13-0.fits",
             "160619_RN-3V-14-0.fits",
             "160619_RN-4V-15-0.fits",
             "160619_RN-5V-16-0.fits",
             "160619_RN-6V-17-0.fits",
             "160619_RN-7V-18-0.fits"]
    voltages = [1.0, 1.37, 1.60, 2.33, 3.35, 4.53, 6.52, 9.84, 14.38, 19.81]
    RNs = []
    for f in files:
        RNs.append(readNoise(f, start = 50000, gain = 1.5, plot = False))
    RNs = numpy.array(RNs)
    RNs /= RNs[0]
    plt.plot(voltages, RNs, 'ko-')
    plt.ylabel("Excess Noise Factor $F$")
    plt.xlabel("Gain")
    plt.title("Statistical Excess $1/f$ Noise in Mark 13 SAPHIRA")
    plt.show()
        

def mk3vsmk14thermal16Jul2016(log = False, gainCorrect = False):
    mk14files =[  ["151228_144652darkMap.fits",#60K
                   "151228_165637darkMap.fits",
                   "151228_190622darkMap.fits",
                   "151228_211617darkMap.fits",
                   "151228_232602darkMap.fits",
                   "151229_013547darkMap.fits"],
                  ["151222_170403darkMap.fits",#70K
                   "151222_191353darkMap.fits",
                   "151222_212338darkMap.fits",
                   "151222_233333darkMap.fits",
                   "151223_014323darkMap.fits",
                   "151223_035308darkMap.fits"],
                  ["160229_123223darkMap.fits",#85K
                   "160229_144213darkMap.fits",
                   "160229_165158darkMap.fits",
                   "160229_190143darkMap.fits",
                   "160229_211128darkMap.fits",
                   "160229_232113darkMap.fits"]]
    mk3files =  [["150624_darkMap_M02775-35_25V.fits",
                  "150624_darkMap_M02775-35_35V.fits",
                  "150624_darkMap_M02775-35_45V.fits",
                  "150624_darkMap_M02775-35_55V.fits",
                  "150624_darkMap_M02775-35_60V.fits",
                  "150711_darkMap_M02775-35_60V_60K.fits",
                  "150624_darkMap_M02775-35_65V.fits",
                  "150711_darkMap_M02775-35_65V_60K.fits",
                  "150624_darkMap_M02775-35_70V.fits",
                  "150624_darkMap_M02775-35_75V.fits",
                  "150711_darkMap_M02775-35_75V_60K.fits",
                  "150624_darkMap_M02775-35_80V.fits",
                  "150624_darkMap_M02775-35_85V.fits",
                  "150624_darkMap_M02775-35_90V.fits",
                  "150624_darkMap_M02775-35_95V.fits",
                  "150624_darkMap_M02775-35_100V.fits",
                  "150624_darkMap_M02775-35_105V.fits"],
                 ["150704_darkMap_M02775-35_25V_70K.fits",
                  "150704_darkMap_M02775-35_45V_70K.fits",
                  "150704_darkMap_M02775-35_60V_70K.fits",
                  "150704_darkMap_M02775-35_70V_70K.fits",
                  "150704_darkMap_M02775-35_80V_70K.fits",
                  "150704_darkMap_M02775-35_90V_70K.fits",
                  "150704_darkMap_M02775-35_100V_70K.fits"],
                 ["150706_darkMap_M02775-35_25V_80K.fits",
                  "150706_darkMap_M02775-35_45V_80K.fits",
                  "150706_darkMap_M02775-35_80V_80K.fits"]]
    w = [[224,256],[160,192]] #Top center. (standard)
    biases = [2.5, 4.5, 6.5, 8.5, 9.5, 10.5]
    gains =  [1.0, 1.5, 3.0, 5.2, 7.6, 12.5]
    for files in mk14files:
        medians = []
        for f in files:
            print "file:", f
            d = openfits(f)
            dark = numpy.median(d[w[0][0]:w[0][1],w[1][0]:w[1][1]])
            medians.append(dark * 2.89)
        if gainCorrect:
            for i in range(len(medians)):
                medians[i] /= gains[i]
        if log:
            plt.semilogy(biases, medians, 'o-')
        else:
            plt.plot(biases, medians, 'o-')
    plt.xlabel("Bias (V)")
    plt.ylabel("e-/s")
##    plt.ylim(0, 0.25)
    plt.xlim(2, 11)
    title = "Median Dark Current"
    if gainCorrect:
        title += " (gain corrected)"
    else:
        title += " (no gain correction)"
    title +=  " vs. Bias Voltage\nM06715-27 Data: 22 - 29 Dec 2015, 29 Feb 2016\nM02775-35 Data: 24 Jun - 11 Jul 2015"

    biases = [[2.5, 3.5, 4.5, 5.5, 6.0, 6.0, 6.5, 6.5, 7.0, 7.5, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5],
              [2.5, 4.5, 6.0, 7.0, 8.0, 9.0, 10.0],
              [2.5, 4.5, 8.0]]
    w = [[224,256],[160,192]] #Top center. (standard)
    for b,files in map(None, biases, mk3files):
        medians = []
        for f in files:
            print "file:", f
            d = openfits(f)
            dark = numpy.median(d[w[0][0]:w[0][1],w[1][0]:w[1][1]])
            medians.append(dark * 2.89)
        plt.semilogy(b, medians, 'o--')
    plt.legend(["Mk14 60K","Mk14 70K","Mk14 85K","Mk3 60K","Mk3 70K","Mk3 80K"], loc = 2)
    plt.title(title)
    plt.show()

def DonsSPIEFigures():
    markers = ["ko-","ks-","ko--","ks--"]
    x = [1,2,4,8,16,32]
    legends = [["128/128 Signal Ramp IRP Readout",
                "128/128 Signal Ramp Conventional Readout",
                "128/128 Reference Ramp IRP Readout",
                "128/128 Reference Ramp Conventional Readout"],
               ["128/128 Signal Ramp IRP Readout",
                "128/128 Signal Ramp Conventional Readout",
                "128/128 Reference Ramp IRP Readout",
                "128/128 Reference Ramp Conventional Readout"],
               ["128/128 Signal Ramp IRP Readout 16272",
                "128/128 Reference Ramp IRP Readout 16272",
                "128/128 Signal Ramp IRP Readout 18575",
                "128/128 Reference Ramp IRP Readout 18575"]]
    fillstyles = ['full','full','none','none']
    xlabels = ["CDS 1/1",
               "CDS 2/2",
               "CDS 4/4",
               "CDS 8/8",
               "CDS 16/16",
               "CDS 32/32"]
    dats = [[[14.1857, 10.4359, 7.8177, 5.8016, 4.43505],
             [18.72085, 13.6684, 9.9918, 6.9854, 5.05955],
             [10.48875, 7.514, 5.40205, 3.91445, 2.8435],
             [13.14885, 8.2059, 5.27845, 3.5562, 2.5315]],
            [[16.6777, 12.1712, 9.06439, 6.84766, 5.07783],
             [20.928, 14.4749, 10.1565, 7.58507, 5.05966],
             [12.2127, 8.85688, 6.49236, 4.79368, 3.52942],
             [17.0103, 10.2217, 6.89275, 4.80984, 3.16012]],
            [[14.1857, 10.4359, 7.8177, 5.8016, 4.43505],
             [10.48875, 7.514, 5.40205, 3.91445, 2.8435],
             [16.6777, 12.1712, 9.06439, 6.84766, 5.07783],
             [12.2127, 8.85688, 6.49236, 4.79368, 3.52942]]]
    dats = numpy.array(dats)
    #Don's fixes.
    dats[0,:,:] = dats[0,:,:] * 1.13
    dats[1,:,:] = dats[1,:,:] * 0.89
    dats[2,:2,:] = dats[2,:2,:] * 1.13
    dats[2,2:,:] = dats[2,2:,:] * 0.89
    titles = ["Read Noise for Phase 1 SN 16272",
              "Read Noise for Science Grade SN 18575",
              "Comparison of IRP CDS SN 16272 & 18575"]
    for data, title, legend in map(None, dats, titles, legends):
        for d, m, fill in map(None, data, markers, fillstyles):
            plt.loglog(x[:len(d)],d,m, fillstyle=fill, markersize = 8)
        plt.xticks(x, xlabels)
        plt.yticks([1,10],["1","10"])
        plt.legend(legend, loc = 3, prop = {'size':12})
        plt.ylim(1,30)
        plt.xlim(0.7,50)
        plt.ylabel("rms electrons ($e^{-}$)")
        plt.title(title)
        plt.grid(True)
        plt.grid(True, axis = 'y', which = 'minor')
        plt.tick_params(axis = 'x', which = 'minor', bottom = 'off')
        plt.show()
    #Now the other two.
    markers = ["ko-","ko--"]
    x = [1,2,4,8,16,32,64,128]
    legends = [["Signal $e^{-}$ @ 1.2/ADU",
                "Reference $e^{-}$ @ 1.2/ADU",],
               ["Signal $e^{-}$ @ 1.53/ADU",
                "Reference $e^{-}$ @ 1.53/ADU",]]
    fillstyles = ['full','none']
    xlabels = ["Del = 1",
               "Del = 2",
               "Del = 4",
               "Del = 8",
               "Del = 16",
               "Del = 32",
               "Del = 64",
               "Del = 128"]
    dats = [[[13.6, 13.8, 13.785, 14.047, 14.144, 14.413, 14.294, 14.1],
             [10.4, 10.4, 10.4, 10.412, 10.459, 10.5, 10.5, 10.5]],
            [[16.43348, 16.47804, 16.57494, 16.76848, 17.08436, 17.08436, 17.06588, 17.08991],
             [11.54149, 11.57145, 11.6535, 11.80599, 11.81842, 12.04371, 11.96868, 11.96435]]]
    dats = numpy.array(dats)
    dats[0,:,:] = dats[0,:,:] * 1.13
    dats[1,:,:] = dats[1,:,:] * 0.89
    titles = ["Phase 1 IRP Corrected 16272 Data",
              "IPR Corrected SN 18575 Data"]
    for data, title, legend in map(None, dats, titles, legends):
        for d, m, fill in map(None, data, markers, fillstyles):
            plt.semilogx(x[:len(d)],d,m, fillstyle=fill, markersize = 8)
        plt.xticks(x, xlabels)
        plt.yticks(range(0,20,2))
        plt.legend(legend, loc = 3, prop = {'size':15})
        plt.ylim(0,20)
        plt.xlim(0.7,180)
        plt.ylabel("rms electrons ($e^{-}$)")
        plt.title(title)
        plt.grid(True, axis = 'y')
        plt.tick_params(axis = 'x', which = 'minor', bottom = 'off')
        plt.show()
    



def detectorModel(gain = 20, readNoise = True, RNrms = 5.0, darkCurrent = True):
    #gain should be a mean gain.
    #readNoise should be e- rms
    photons = 10000
    totalframes = 30000
    steps = 1000
    histMax = 300
    binsize = 2.0
    darkphotons = 200

    #Calculate the probability that an electron is duplicated at each step.
    pDup = math.log(gain, 2) / steps
    print "pDup:", pDup
    print "pNull:", (1 - pDup)**steps
    
    results = []
    positives = []
    print "Avalanching..."
    for dummy in range(photons):
        electrons = 1
        for dummy2 in range(steps):
            randoms = numpy.random.rand(electrons)
            for r in randoms:
                if r < pDup:
                    electrons += 1
        results.append(electrons)
        positives.append(electrons)
        if dummy % 100 == 0:
            print ".",

    darks = []
    if darkCurrent:
        for dummy in range(darkphotons):
            electrons = 1
            darksteps = int(numpy.random.rand(darkphotons) * steps)
            for dummy2 in range(darksteps):
                randoms = numpy.random.rand(electrons)
                for r in randoms:
                    if r < pDup:
                        electrons += 1
            results.append(electrons)
            darks.append(electrons)
            if dummy % 100 == 0:
                print ".",
        

    if readNoise:
        print "Applying read noise..."
        RN = numpy.random.normal(scale = RNrms, size = len(results))
        for i,result in enumerate(results):
            result += RN[i]

    nondetects = numpy.random.normal(scale = RNrms, size = totalframes - electrons)

    print "Adding blank frames..."
    for n in nondetects:
        results.append(n)
    
    plt.hist(results, bins = histMax / binsize, range = [1,histMax], label = "Total")
    plt.hist(nondetects, bins = histMax / binsize, range = [1,histMax], label = "Non-Detects")
    plt.hist(positives, bins = histMax / binsize, range = [1,histMax], label = "Positives")
    plt.hist(darks, bins = histMax / binsize, range = [1,histMax], label = "Darks")
    plt.legend()
    plt.show()

def Oct2016DarkLongTrends():
    filelist = listdir("./")
    darks = []
##    startTime = 17
##    day1 = "161011"
##    day2 = "161012"
    
##    startTime = 19
##    day1 = "161012"
##    day2 = "161013"
    
##    startTime = 15
##    day1 = "161016"
##    day2 = "161017"
    
##    startTime = 19
##    day1 = "161019"
##    day2 = "161020"

    startTime = 8
    day1 = "161024"
    day2 = "161025"
    
    hours = 48
    times = numpy.array(range(0,hours*60,15))
    for f in filelist:
        if (".fits" in f) and (((day1 + "_" in f) and (int(f[7:9]) > startTime)) or (day2 + "_" in f)) and not ("darkMap" in f):
            print times[len(darks)], f
            d = openfits(f)
            e = numpy.zeros(d.shape[1:])
            start = 40
            for i in range(0,20):
                e += d[i + start,:,:]
                e -= d[i - 20 + d.shape[0],:,:]
            e /= 20
            e /= d.shape[0] - (start + 20)
            e /= 5 #5s interval
            darks.append(numpy.median(e[192:256,128:192].flatten()) * 2.89) #e- conversion
    times = times[:len(darks)]
    plt.semilogy(times,darks)
    plt.ylabel("e-/s")
    plt.xlabel("time (min)")
##    plt.title("12 hour darks, 60K, M06665-25 11-12 Oct 2016")
    plt.title("4 hour darks vs. bias, 60K, M06665-23 19-20 Oct 2016")
    plt.show()

def TrendsByDay():
    filelist = listdir("./")
    darks = []
##    day = "161018"
    day = "161019"
    times = numpy.array(range(0,12*60,15))
    for f in filelist:
        if (".fits" in f) and (day in f) and not ("darkMap" in f):
            print times[len(darks)], f
            d = openfits(f)
            e = numpy.zeros(d.shape[1:])
            start = 40
            for i in range(0,20):
                e += d[i + start,:,:]
                e -= d[i - 20 + d.shape[0],:,:]
            e /= 20
            e /= d.shape[0] - (start + 20)
            e /= 5 #5s interval
            darks.append(numpy.median(e[192:256,128:192].flatten()) * 2.89) #e- conversion
    times = times[:len(darks)]
    plt.semilogy(times,darks)
    plt.ylabel("e-/s")
    plt.xlabel("time (min)")
    plt.title("4 hour darks, 50K, M06665-23 18 Oct 2016")
    plt.show()

def OldDarkTrends13Oct2016():
    filelist = listdir("./")
    darks = []
    startTime =  92548
    endTime = 111735
    day = "151101"
    times = numpy.array(range(0,12*60,15))
    for f in filelist:
        if (".fits" in f) and ((day + "_" in f) and (int(f[7:13]) >= startTime) and (int(f[7:13]) <= endTime)) and not ("darkMap" in f):
            d = openfits(f)
            e = numpy.zeros(d.shape[1:])
            start = 40
            for i in range(0,20):
                e += d[i + start,:,:]
                e -= d[i - 20 + d.shape[0],:,:]
            e /= 20
            e /= d.shape[0] - (start + 20)
            e /= 5 #5s interval
            darks.append(numpy.median(e[96:168,288:].flatten()) * 2.89) #e- conversion
            print times[len(darks)], f, numpy.median(e[96:168,288:].flatten()) * 2.89
    times = times[:len(darks)]
    plt.semilogy(times,darks)
    plt.ylabel("e-/s")
    plt.xlabel("time (min)")
    plt.title("12 hour darks, 60K, M06665-25 11-12 Oct 2016")
    plt.show()

def voltageGain12Oct2016():
    files = ["161012_164943.fits",
             "161012_165111.fits",
             "161012_165146.fits",
             "161012_165217.fits",
             "161012_165245.fits"]
    voltage = [0.0, 0.05, 0.1, 0.15, 0.2]
    gain = []
    for x in range(8):
        medians = []
        for f in files:
            d = openfits(f)
            med = numpy.median(d[:,:,x])
            medians.append(med * 2.89)
        plt.plot(voltage,medians)
        m,b = linearfit(voltage, medians)
        gain.append(1e6/m)
        print "Voltage Gain:", (1e6)/m, "uV/ADU"
    print "Mean Voltage Gain:", numpy.mean(gain), "uV/ADU"
    plt.show()

def newChargeGain(filename, gain = 1.0, endFrame = 200):
##    filename = "161102_101504.fits"
##    filename = "151108_101441.fits"
    d = openfits(filename)
    #Clip data to mask aperture.
    d = d[:endFrame,64:-64,92:-92]
    
##    medians = []
##    for i in range(d.shape[0]):
##        medians.append(numpy.median(d[i,:,:]))
##    plt.plot(medians)
##    plt.show()
    d = -cdsCube(d)
    d /= gain
    means = []
    stds = []
    res = []
    for y in range(d.shape[1]):
        for x in range(d.shape[2]):
            means.append(numpy.mean(d[:,y,x]))
            stds.append(numpy.var(d[:,y,x], ddof = 1))
            res.append(numpy.mean(d[:,y,x])/numpy.var(d[:,y,x], ddof = 1))
    print numpy.median(means)
    print numpy.median(stds)
    print 1/numpy.median(res)
    print "Charge Gain:", (numpy.median(stds)) / numpy.median(means)
    plt.plot(means,stds,'o')
    plt.ylabel("$\sigma$")
    plt.xlabel("mean")
    plt.show()

def readNoisevBias():
    files = ["161106_104602.fits",
             "161106_104713.fits",
             "161106_104821.fits",
             "161106_104902.fits",
             "161106_104957.fits",
             "161106_105035.fits",
             "161106_105115.fits"]
    RNs = []
    for f in files:
        d = openfits(f)
        d = cdsCube(d)
        RN = numpy.std(d.flatten(), ddof = 1)
        while RN > 100:
            print "Correcting for glitch in", f
            for i in range(d.shape[0]):
                for x in range(d.shape[2]):
                    if abs(d[i,0,x]) > 100:
                        d[i,0,x] = 0
            RN = numpy.std(d.flatten(), ddof = 1)
        RNs.append(RN)
        del d
    print len(RNs)
    biases = [2.5, 4.5, 6.5, 8.5, 10.5, 12.5, 14.5]
    plt.plot(biases, RNs)
    plt.xlabel("bias (V)")
    plt.ylabel("RN (ADU rms)")
    plt.show()

def avStats():
    files = ["161106_155546.fits",
             "161106_155643.fits",
             "161106_155715.fits",
             "161106_155747.fits",
             "161106_155822.fits",
             "161106_155846.fits",
             "161106_155912.fits"]
    RNs = []
    for f in files:
        d = openfits(f)
        d = cdsCube(d)
        RN = numpy.std(d.flatten(), ddof = 1)
        while RN > 100:
            print "Correcting for glitch in", f
            for i in range(d.shape[0]):
                for x in range(d.shape[2]):
                    if abs(d[i,0,x]) > 100:
                        d[i,0,x] = 0
            RN = numpy.std(d.flatten(), ddof = 1)
        RNs.append(RN)
        del d
    print len(RNs)
    biases = [2.5, 4.5, 6.5, 8.5, 10.5, 12.5, 14.5]
    plt.plot(biases, RNs)
    plt.xlabel("bias (V)")
    plt.ylabel("RN (ADU rms)")
    plt.show()

def DarkVsBiasMk13ME100015Dec2015():
    startFile = "161024_092418.fits"
    endFile = "161025_105112.fits"
##    endFile = "161025_150835.fits"
    masterFile = listdir("./")
    startIndex = masterFile.index(startFile)
    endIndex = masterFile.index(endFile)
    files = masterFile[startIndex:endIndex + 1]
    timing = 5
    start = 40
    medians = []
    gains = [1.0, 1.7, 3.3, 5.3, 8.7, 12.]
    V = [2.5, 4.5, 6.5, 8.5, 9.5, 10.5]
    fullGains = numpy.zeros([len(gains) * 16])
    print fullGains.shape, len(files)
    for i in range(len(fullGains)):
        fullGains[i] = gains[i/16]
    for i,f in enumerate(files):
        d = openfits(f)
        cutoff = -1
        dark = 1.59 * (d[start,192:,128:-128] - d[-1,192:,128:-128]) / ((d.shape[0] - start) * timing)
        if i % 16 == 7:
            medians.append(numpy.median(dark) / fullGains[i])
            print numpy.median(dark) / fullGains[i]
    plt.plot(V, medians, 'o-')
    plt.ylabel("e-/s")
    plt.xlabel("Bias (V)")
    plt.title("Dark Vs. Bias, gain normalized, M06665-23 mk13 ME1000, data 24-25 Oct 2016")
    plt.show()

def mk3vsmk13vsmk14Dark16Dec2016():
    mk3  = [0.0195, 0.0284, 0.0163, 0.0191, 0.0182, 0.0109, 0.0183, 0.0642, 0.2520]
    mk3V = [   2.5,    3.5,    4.5,    5.5,    6.5,    7.5,    8.5,    9.5,   10.5]
    mk14 = [0.0349, 0.0302, 0.0200, 0.0249, 0.0480, 0.1668]
    mk14V =[   2.5,    4.5,    6.5,    8.5,    9.5,   10.5]
    mk13 = [0.0578, 0.0607, 0.0462, 0.0460, 0.0612, 0.1978]
    mk13V =[   2.5,    4.5,    6.5,    8.5,    9.5,   10.5]

    plt.semilogy(mk3V, mk3, 'o-', label = "M02775-35 mk3 ME911")
    plt.semilogy(mk13V, mk13, 'o-', label = "M06665-23 mk13 ME1000")
    plt.semilogy(mk14V, mk14, 'o-', label = "M06715-27 mk14 ME911")
    plt.ylabel("e-/s")
    plt.xlabel("Bias (V)")
    plt.legend(loc = 2)
    plt.title("Dark Vs. Bias, gain normalized")
    plt.show()

def pixelLevelChargeGain():
    startFile = "161016_182257.fits"
    endFile = "161017_044512.fits"
    masterFile = listdir("./")
    startIndex = masterFile.index(startFile)
    endIndex = masterFile.index(endFile)
    files = masterFile[startIndex:endIndex + 1]
    avg = 16
    test = openfits(files[0])
    masterCube = numpy.zeros([len(files), test.shape[1], test.shape[2]])
    del test
    for j,f in enumerate(files):
        d = openfits(f)
        avgFrame = numpy.zeros([d.shape[1],d.shape[2]])
        for i in range(16):
            avgFrame -= (d[d.shape[0] - avg + i,:,:] - d[15 + i,:,:])
        avgFrame /= avg
        masterCube[j,:,:] = avgFrame
    endFrame = numpy.zeros([masterCube.shape[1], masterCube.shape[2]])
    sigFrame = numpy.zeros([masterCube.shape[1], masterCube.shape[2]])
    varFrame = numpy.zeros([masterCube.shape[1], masterCube.shape[2]])
    for y in range(masterCube.shape[1]):
        for x in range(masterCube.shape[2]):
            avg = numpy.mean(masterCube[:,y,x])
            var = numpy.var(masterCube[:,y,x], ddof = 1)
            sigFrame[y,x] = avg
            varFrame[y,x] = var
            endFrame[y,x] = avg/var
    print numpy.mean(endFrame.flatten()), "+/-", numpy.std(endFrame, ddof = 1)
    if path.isfile("161216_ChargeGainFrame_Signal.fits"):
        remove("161216_ChargeGainFrame_Signal.fits")
    savefits("161216_ChargeGainFrame_Signal.fits", sigFrame)
    if path.isfile("161216_ChargeGainFrame_Variance.fits"):
        remove("161216_ChargeGainFrame_Variance.fits")
    savefits("161216_ChargeGainFrame_Variance.fits", varFrame)
    if path.isfile("161216_ChargeGainFrame.fits"):
        remove("161216_ChargeGainFrame.fits")
    savefits("161216_ChargeGainFrame.fits", endFrame)

def darkConsistency():
    filename1 = "151220_212305.fits"
    filename2 = "151220_210708.fits"
    d1 = openfits(filename1)
    d2 = openfits(filename2)

    start = 40
    end = -1
    n = 20
    timing = 5
    
    e1 = numpy.zeros(d1.shape[1:])
    e2 = numpy.zeros(d1.shape[1:])
    if end < 0:
        end += d1.shape[0]
    
    for i in range(0,n):
        e1[:,:] += d1[i + start,:,:]
        e1[:,:] -= d1[i - n + end,:,:]
        e2[:,:] += d2[i + start,:,:]
        e2[:,:] -= d2[i - n + end,:,:]
    e1 /= n
    e2 /= n
    e1 /= end - (start + n)
    e2 /= end - (start + n)
    e1 /= timing
    e2 /= timing
    e1 *= 2.89
    e2 *= 2.89

    plt.plot(e1.flatten(),e2.flatten(),'ko', markersize = 2)
    plt.ylabel("run #1, $e^{-}/\mathrm{s}$")
    plt.xlabel("run #2, $e^{-}/\mathrm{s}$")
    plt.title("Dark Current in Successive Runs")
    plt.ylim([-0.15,0.60])
    plt.xlim([-0.15,0.60])
    plt.show()

def darkvTemp7Jan2017(log = True):
    files =["151227_113650darkMap.fits",#40K
                "151226_143121darkMap.fits",#50K
                "151228_144652darkMap.fits",#60K
                "151224_160532darkMap.fits",#65K
                "151222_170403darkMap.fits",#70K
                "151223_151856darkMap.fits",#75K
                "160229_123223darkMap.fits"]#85K
    temps = [40, 50, 60, 65, 70, 75, 85]
    medians = []
    for f in files:
        d = openfits(f)
        medians.append(2.89 * numpy.median(d[-32:,160:192]))
    if log:
        plt.semilogy(temps, medians, 'ko-')
    else:
        plt.plot(temps, medians, 'o-')
##        plt.semilogy(biases[:-2], medians[:-2], colors[i] + 'o-')
##        plt.semilogy(biases[-3:], medians[-3:], colors[i] + 'o--')
    plt.xlabel("Temperature (K)")
    plt.ylabel("e-/s")
    title = "Dark Current"
    title +=  " vs. Temperature for M06715-27"
    plt.title(title)
    plt.xlim([35,90])
    plt.ylim([0.01, 1.0])
    plt.show()

def directTunnelingModel():
    E_g = 0.35#eV
    N = 5.e7
    N_t = 5.e6#cm-3
    m_e = 7.e-2 * E_g
    tsr = []
    tdir = []
    for V in range(1,10):
        Jtsr  = (1e-13 * N_t * V / E_g) * e**(-1.5e10 * pi * m_e**0.5 * E_g**1.5 / (N * (E_g + V))**0.5)
        Jtdir = 1.2e-2 * V * (N * (E_g + V))**0.5 * e**(-9.43e10 * m_e**0.5 * E_g**1.5 / (N * (E_g + V))**0.5)
        print Jtsr,Jtdir
        tsr.append(Jtsr)
        tdir.append(Jtdir)
    plt.plot(tsr)
    plt.plot(tdir)
    plt.show()

def readNoiseVsAvg(histrange = [-20,20], binspacing = 1.0):
    f = "161106_105115.fits" #COMMON = -11V, mk13 M06665-23 ME1000, no ref, 32 x 1, diff window
    d = openfits(f)
    averages = [1,2,4,8,16,32]
    #Perform subtraction and masking.
    cds = numpy.zeros([d.shape[0] - 1,d.shape[1],d.shape[2]])
    for i in range(cds.shape[0]):
        cds[i,:,:] = (d[i,:,:] - d[i + 1,:,:])
    averaged = []
    for a in averages:
        averaged.append(avgCube(cds, avg=a))
        print a, numpy.std(averaged[-1], ddof = 1)        
    bins = (histrange[1] - histrange[0]) / binspacing
    plt.hist(averaged, bins = bins, range=histrange)
    plt.xlabel("ADU")
    plt.ylabel("n")
    plt.show()

def measureF(hvPerFrame, offFile = "161106_105115.fits", onFile = "161106_105124.fits", gain = 82, histrange = [-30,200], binspacing = 1.0, plot = False):
##    offFile = "161106_105115.fits" #COMMON = -11V, mk13 M06665-23 ME1000, no ref, 32 x 1, diff window
##    onFile = "161106_105124.fits"

    chargeGain = 1.57 #e-/ADU ME1000
##    chargeGain = 1.04 #e-/ADU ME1000

    a = 2048
##    avg = [1,2,4,8,16,32,64,128,256,512,1024,2048,4196,8292]

    off = openfits(offFile)
    on = openfits(onFile)

    #Windowing.
##    on = on[:,:,4:14] #was 1:29
##    off = off[:,:,4:14]
    off = numpy.delete(off, [29], 2) #JUST THE DEAD ONE
    on = numpy.delete(on, [29], 2)
##    off = numpy.delete(off, [0,1,8,21,23,25,26,28,29], 2) #JUST THE REAL BAD ONES
##    on  = numpy.delete(on,  [0,1,8,21,23,25,26,28,29], 2)
##    off = numpy.delete(off, [0,8,14,15,17,18,21,23,24,25,26,28,29,30,31], 2) #HARSH
##    on  = numpy.delete(on,  [0,8,14,15,17,18,21,23,24,25,26,28,29,30,31], 2)
    
    #Perform subtraction and masking.
    offSub = numpy.zeros([off.shape[0] - 1,off.shape[1],off.shape[2]])
    onSub = numpy.zeros([on.shape[0] - 1,on.shape[1],on.shape[2]])
    for i in range(offSub.shape[0]):
        #Screen out resets.
        if abs(numpy.median(off[i,:,:]) - numpy.median(off[i + 1,:,:])) < 10:
            offSub[i,:,:] = (off[i,:,:] - off[i + 1,:,:])
        
    for i in range(onSub.shape[0]):
        if abs(numpy.median(on[i,:,:]) - numpy.median(on[i + 1,:,:])) < 10:
            onSub[i,:,:] = (on[i,:,:] - on[i + 1,:,:])


##    offSubCopy = numpy.copy(offSub)
##    onSubCopy  = numpy.copy(onSub)
##    for x in range(onSub.shape[2]):
##        offSub[:,:,x] -= numpy.median(numpy.delete(offSubCopy, [x], 2), axis = 2)
##        onSub[:,:,x] -= numpy.median(numpy.delete(onSubCopy, [x], 2), axis = 2)

##    plt.plot(off[:,0,0] - numpy.mean(off[:,0,0]))
##    plt.plot(offSub[:,0,0] - numpy.mean(offSub[:,0,0]))
##    plt.plot(numpy.median(offSub[:,:,1:], axis = 2)[:,0])
##    plt.show()

##    plt.plot(on[:,0,0] - numpy.mean(on[:,0,0]))
##    plt.plot(onSub[:,0,0] - numpy.mean(onSub[:,0,0]))
##    plt.show()

##    offSub = -cdsCube(off, rolling = True, delta = a)
##    onSub = -cdsCube(on, rolling = True, delta = a)

    offSub *= chargeGain
    onSub  *= chargeGain

    preSub = numpy.array(onSub)

    #Median adjustment for excursions from detector voltage noise etc.
##    for i in range(onSub.shape[0]):
##        onSub[i,:,:] -= numpy.median(onSub[i,:,:])
##    for i in range(offSub.shape[0]):
##        offSub[i,:,:] -= numpy.median(offSub[i,:,:])

    #Adjust for known incident hv.
##    hvMeasured = ((numpy.median(on[100:120,:,:]) - numpy.median(on[1500:1520,:,:])) - \
##                   (numpy.median(off[100:120,:,:]) - numpy.median(off[1500:1520,:,:]))) /\
##                   (1400)
##    hvMeasured = (numpy.sum(cdsCube(off).flatten()) / off.shape[0] - numpy.sum(cdsCube(on).flatten()) / on.shape[0]) / on.shape[2]
##    hvMeasured *= chargeGain
    
##    adjFactor = hvPerFrame / hvMeasured
##    print "Adjustment Factor:", adjFactor
##    print "Ignore adj factor, use gain."
##    onSub *= adjFactor
##    offSub *= adjFactor
##    onSub /= gain
##    offSub /= gain

    #Check individual channels' behavior.
##    for x in range(onSub.shape[2]):
##        print x, "mean:", numpy.mean(onSub[:,:,x]), "std:", numpy.std(onSub[:,:,x], ddof = 1)

##    means = numpy.mean(numpy.mean(onSub, axis = 2), axis = 1)
##    plt.plot(preSub[:,0,2])
##    plt.plot(onSub[:,0,2])
##    plt.show()

    #WE NEED TO AVERAGE A BUNCH OF THESE?
    #AND THEN WE CAN DO REAL F MEASUREMENT

    offSub = avgCube(offSub, avg = a)[1:,:,:]
    onSub = avgCube(onSub, avg = a)[1:,:,:]

    #Convert to photons.
    offSub *= a
    onSub *= a

    hvMeasured = numpy.median(onSub)
    print "hvMeasured", hvMeasured
    adjFactor = hvPerFrame / hvMeasured
    print "Adjustment Factor:", adjFactor
##    print "Ignore adj factor, use gain."
    onSub *= adjFactor
    offSub *= adjFactor

    

##    print onSub.shape
    
##    plt.plot(onSub[:,0,2])
##    plt.show()

##    offSub /= gain
##    onSub /= gain

    #Compute bins.
##    bins = (histrange[1] - histrange[0]) / (binspacing * 1.57)
    bins = (histrange[1] - histrange[0]) / (binspacing)

    hist1 = numpy.histogram([offSub.flatten()], bins = bins, range = histrange)
    hist2 = numpy.histogram([onSub.flatten()], bins = bins, range = histrange)
##    diff = hist2[0] - hist1[0]

    if plot:
        plt.hist([offSub.flatten(),onSub.flatten()], bins = bins, range=histrange, color = ['b','r'], histtype = 'step', log = False)
        plt.title("Gain:" + str(gain))
        plt.show()

    #Generate a dataset to fit the difference histogram.
        
    #Eliminate any results for 'negative photons' in case there's a reset artifact there.
##    diffZero = 0
##    z = 0
####    while histrange[0] + (z *  binspacing * 1.57) < 0:
##    while histrange[0] + (z *  binspacing) < 0:
##        z += 1
####    diffZero = histrange[0] + (z *  binspacing * 1.57)
##    diffZero = histrange[0] + (z *  binspacing)
##    diff = diff[z:]
##    
##    #Get rid of negative values.
##    for i in range(diff.shape[0]):
##        if diff[i] < 0:
##            diff[i] = 0
    #As a probability.
##    P = numpy.array(diff.astype(float)) / numpy.sum(diff)
##    for i in range(P.shape[0]-1):
##        P[-i] += numpy.sum(P[:-i])
##    N = 10000
##    diffData = numpy.zeros([N])
##    for i in range(N):
##        r = random.random()
##        j = 0
##        while P[j] < r:
##            j += 1
##        diffData[i] = diffZero + (binspacing*1.57) * j
##
##    hvMean = numpy.mean(diffData)
##    hvStd = numpy.std(diffData, ddof = 1)

    #Use compared FWHMs.
##    sumProb = 0
##    low = 0
##    high = 0
##    median = 0
##    for i in range(diff.shape[0]):
##        sumProb += float(diff[i]) / float(numpy.sum(diff))
##        if low == 0 and sumProb > 0.16:
##            low = i
##        elif median == 0 and sumProb > 0.5:
##            median = diffZero + i * binspacing
##        elif high == 0 and sumProb > 0.84:
##            high = i
##    
##    
##    print "high:", diffZero + (high * binspacing), "low:", diffZero + (low * binspacing)
##    phFWHM = (high - low) * binspacing
##
##    sumProb = 0
##    low = 0
##    high = 0
##    off = numpy.array(hist1[0])
##    for i in range(hist1[0].shape[0]):
##        sumProb += float(off[i]) / float(numpy.sum(off))
##        if low == 0 and sumProb > 0.16:
##            low = i
##        elif high == 0 and sumProb > 0.84:
##            high = i
##    rnFWHM = (high - low) * binspacing

    #Find where the bad stuff is coming from.
##    for i in range(offSub.shape[0]):
##        for x in range(offSub.shape[2]):
##            if offSub[i,0,x] < -100:
##                print "BAD:", i, x

    #Compile noise measurements separately for each channel before combining.
    phStds = []
    for x in range(onSub.shape[2]):
        phStds.append(numpy.std(onSub[:,:,x],ddof = 1))
    phStd = numpy.median(phStds)
    RNs = []
    for x in range(offSub.shape[2]):
        RNs.append(numpy.std(offSub[:,:,x], ddof = 1))
##        print x, RNs[x], phStds[x]
    RN = numpy.median(RNs)

    #Measure and subtract read noise.
    #Reset artifacts are killing us here. First, let's determine the edges of the central distribution.
##    lowRN = z
##    highRN = z
##    #print z
##    while hist1[0][lowRN] > 1:
##        lowRN -= 1
##    while hist1[0][highRN] > 1:
##        highRN += 1
##    print "lowRN", hist1[1][lowRN]
##    print "highRN", hist1[1][highRN]
##    if z - lowRN > highRN - z:
##        lowRN = z - (highRN - z)
##    else:
##        highRN = z + (z - lowRN)
##    print hist1[1][lowRN], hist1[1][highRN]
##    offList = []
##    for i in range(offSub.shape[0]):
##        for y in range(offSub.shape[1]):
##            for x in range(offSub.shape[2]):
##                if offSub[i,y,x] > hist1[1][lowRN] and offSub[i,y,x] < hist1[1][highRN]:
##                    offList.append(offSub[i,y,x])
##    RN = numpy.std(offList, ddof = 1)
    
##    print "RN:", RN
##    print "hvStd:", hvStd
##    print "hvMean:", hvMean
##    
##    F = ((hvStd**2 - RN**2) / (hvMean))**0.5
##    F = ((hvStd**2) / (hvMean))**0.5

##    shot = median ** 0.5
##    shot = (hvPerFrame * a)**0.5
    shot = numpy.median(onSub) ** 0.5

##    print "read noise FWHM:", rnFWHM
##    print "gain profile FWHM:", phFWHM
    print "median photoelectrons:", round(numpy.median(onSub), 2), "+/-", round(numpy.std(onSub, ddof = 1), 2)
    print "measured read noise:", round(RN,2), "+/-", round(numpy.std(RNs, ddof = 1),2)
    print "measured overall noise:", round(phStd,2), "+/-", round(numpy.std(phStds, ddof = 1),2)
    print "calculated shot noise:", round(shot, 2)
    
##    F = ((phFWHM/2.)**2 - (rnFWHM/2.)**2)**0.5 / shot
    if RN > phStd:
        print "Read noise greater than shot noise."
        F = 1.0
    else:
        F = ((phStd)**2 - (RN)**2)**0.5 / shot
        print "measured shot noise:", ((phStd)**2 - (RN)**2)**0.5
    
    print "F:", F

    Fchans = []
    for x in range(offSub.shape[2]):
        if RNs[x] < phStds[x]:
            Fchans.append(((phStds[x])**2 - (RNs[x])**2)**0.5 / shot)
    print "by-channel F:", numpy.median(Fchans), "+/-", numpy.std(Fchans, ddof = 1)

    
    return F

def plotF(plotSets = False):
##    offFiles = ["161106_104602.fits",
##                "161106_104713.fits",
##                "161106_104821.fits",
##                "161106_104902.fits",
##                "161106_104957.fits",
##                "161106_105035.fits",
##                "161106_105115.fits"]
##    onFiles =  ["161106_104631.fits",
##                "161106_104733.fits",
##                "161106_104828.fits",
##                "161106_104915.fits",
##                "161106_105004.fits",
##                "161106_105045.fits",
####                "161106_105124.fits"]
##    offFiles = ["170126_135911.fits",
##                "170126_140035.fits",
##                "170126_140145.fits",
##                "170126_140225.fits",
##                "170126_140323.fits",
##                "170126_140406.fits",
##                "170126_140449.fits",
##                "170126_140527.fits",
##                "170126_140616.fits",
##                "170126_140657.fits",
##                "170126_140737.fits"]
##    onFiles =  ["170126_135921.fits",
##                "170126_140051.fits",
##                "170126_140156.fits",
##                "170126_140235.fits",
##                "170126_140333.fits",
##                "170126_140414.fits",
##                "170126_140457.fits",
##                "170126_140537.fits",
##                "170126_140623.fits",
##                "170126_140704.fits",
##                "170126_140752.fits"]
    offFiles = ["170130_095713.fits",
                "170130_095809.fits",
                "170130_095912.fits",
                "170130_100009.fits",
                "170130_100105.fits",
                "170130_100152.fits",
                "170130_100233.fits",
                "170130_100322.fits",
                "170130_100402.fits",
                "170130_100455.fits",
                "170130_100554.fits"]
    onFiles =  ["170130_095725.fits",
                "170130_095820.fits",
                "170130_095922.fits",
                "170130_100019.fits",
                "170130_100118.fits",
                "170130_100159.fits",
                "170130_100242.fits",
                "170130_100332.fits",
                "170130_100421.fits",
                "170130_100510.fits",
                "170130_100602.fits"]
##    V =     [2.5, 4.5, 6.5, 8.5, 10.5, 12.5, 14.5]
##    gains = [1.0, 1.7, 3.3, 5.3, 12.,  35.,  82.]
##    V =     [2.5, 4.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5]
##    gains = [1.0, 1.7, 3.3, 4.2, 5.3, 8.7,  12.,  18.,  35.,  46.,  82.]
    V =     [2.5, 4.5, 6.5, 8.5, 10.5, 11.5, 12.5, 13.0, 13.5, 14.0, 14.5]
    gains = [1.0,1.53,1.97,4.33, 9.88, 14.7, 27.7, 31.0, 36.6,   50.,65.6]
##    V =     [2.5, 10.5, 11.5, 12.5, 13.0, 13.5, 14.0, 14.5]
##    gains = [1.0, 9.88, 14.7, 27.7, 31.0, 36.6,   50.,65.6]

    #First we measure the incident photon flux.
##    off = openfits(offFiles[0])[:,:,4:14]
##    on = openfits(onFiles[0])[:,:,4:14]
    off = openfits(offFiles[0])
    on = openfits(onFiles[0])
    
    off = numpy.delete(off, [29], 2) #JUST THE DEAD ONE
    on = numpy.delete(on, [29], 2)

##    hvPerFrame = ((numpy.median(on[100:120,:,:]) - numpy.median(on[1500:1520,:,:])) - \
##                               (numpy.median(off[100:120,:,:]) - numpy.median(off[1500:1520,:,:]))) /\
##                               (1400)
##    hvPerFrame = numpy.median(cdsCube(off[:1900,:,:])) - numpy.median(cdsCube(on[:1900,:,:]))
##    hvPerFrame *= 1.57
##    hvPerFrame *= 1.04
    
    onSub = numpy.zeros([on.shape[0] - 1,on.shape[1],on.shape[2]])
    for i in range(onSub.shape[0]):
        if abs(numpy.median(on[i,:,:]) - numpy.median(on[i + 1,:,:])) < 10:
            onSub[i,:,:] = (on[i,:,:] - on[i + 1,:,:])

    onSub *= 1.58

    a = 2048

    onSub = avgCube(onSub, avg = a)[1:,:,:]

    #Convert to photons.
    onSub *= a

    hvPerFrame = numpy.median(onSub)
    
    print "hvPerFrame:", hvPerFrame

    
    Flist = []
    for offFile,onFile, gain in map(None, offFiles, onFiles, gains):
##        F = measureF(offFile = offFile, onFile = onFile, gain = gain, plot = plotSets, histrange = [-800,800], hvPerFrame = hvPerFrame)
##        F = measureFbetter(offFile, onFile, gain)
##        F = measureFbest(offFile, onFile, gain)
        F = measureFbestest(offFile, onFile, gain)
        print "F:", F
        Flist.append(F)
##    plt.plot(V,Flist,'ko')
    plt.plot(gains,Flist,'ko')
    plt.ylabel("Statistical Excess Noise $F$")
##    plt.xlabel("$V_{bias}$")
    plt.xlabel("Gain")
    plt.ylim([0.8,2.0])
    plt.title("Excess Noise for Mark 13 M06665-23")
    plt.show()

def noiseByChannel():
    offFiles = ["170130_095713.fits",
                "170130_095809.fits",
                "170130_095912.fits",
                "170130_100009.fits",
                "170130_100105.fits",
                "170130_100152.fits",
                "170130_100233.fits",
                "170130_100322.fits",
                "170130_100402.fits",
                "170130_100455.fits",
                "170130_100554.fits"]
    noises = numpy.zeros([32,len(offFiles)])
    signals = numpy.zeros([32,len(offFiles)])
    for i,f in enumerate(offFiles):
        d = openfits(f)
        cds = cdsCube(d[1:,:,:])
        for x in range(32):
            noises[x,i] =  numpy.std(d[1:,:,x], ddof = 1)
            signals[x,i] = numpy.sum(cds[:,:,x])
    for x in range(32):
        print str(x) + ": sum " + str(numpy.mean(signals[x,:])) + " std " + str(numpy.mean(noises[x,:]))

def avgDemo(N = 64, A = 4, RN = 8, pulse = 16):
    #Averaging demo of noise analysis techniques to settle ongoing debate with Don.
    #N = dataset size
    #A = number of averages
    #RN = read noise rms
    #PULSE = pulse height

    #Create a dataset.
    data = numpy.zeros([N])
    #Start with read noise.
    for i in range(N):
        data[i] += random.normalvariate(0, RN)
    #Add a blind pulse.
    pulseLocation = int(random.random()*(N-2))
    data[pulseLocation:] += pulse

    print "RAW NOISE:", numpy.std(data, ddof = 1)
##    print "RAW PULSE:", data[pulseLocation]

    #Okay, let's do averaging.
    avgData = numpy.zeros([N/A])
    for i in range(N/A):
        avgData[i] = numpy.sum(data[i * A:(i + 1) * A]) / A

    print "AVG NOISE:", numpy.std(avgData, ddof = 1)
    #print "AVG PULSE:", avgData[pulseLocation / A]

    #Now the CDS.
    cdsData = numpy.zeros([(N/A) - 2])
    for i in range((N/A) - 2):
        cdsData[i] = avgData[i + 2] - avgData[i]

    print "CDS NOISE:", numpy.std(cdsData,ddof = 1)
    print "CDS PULSE:", cdsData[(pulseLocation / A) - 2]

    x = numpy.array(range(N))
    plt.plot(numpy.array(range(N)), data)
    plt.plot(numpy.array(range(0,N,A)), avgData)
    plt.plot(numpy.array(range(0,N-(A*2),A)), cdsData)
    plt.show()

def darkSeries(firstFile, lastFile, chargeGain = 2.89):
    files = listdir("./")
    if not ((firstFile in files) and (lastFile in files)):
        print "I didn't find one of endcap files."
    startIndex = files.index(firstFile)
    endIndex = files.index(lastFile)
    n = 20
    start = 40
    end = -1
    timing = 5
    darks = []
    for i in range(startIndex, endIndex + 1):
        if ".fits" in files[i]:
            try:
                d = openfits(files[i])
                e = numpy.zeros([d.shape[1],d.shape[2]])
                if end < 0:
                    end += d.shape[0]
                for j in range(0,n):
                    e += d[j + start,:,:]
                    e -= d[j - n + end,:,:]
                e /= n
                e /= end - (start + n) #Modified 29 Feb 2016.
                e /= timing #Added 7 Apr 2015.
                print "Median dark current:", numpy.median(e) * chargeGain, "e-/s"
                darks.append(numpy.median(e) * chargeGain)
            except IOError:
                print "Bad file:", files[i]
    plt.plot(numpy.array(range(0,len(darks) * 15, 15)), darks, 'ko-')
    plt.ylabel("$e^{-}\mathrm{s}^{-1}\mathrm{pix}^{-1}$")
    plt.xlabel("time (min)")
    plt.xlim([0, 15 * len(darks)])
##    plt.ylim([0,0.40])
    plt.title("Median Dark Current vs. Time\nMark 14 M06715-27, read interval 5s, $V_{bias} = 1.5V$, $T = 62.5\mathrm{K}$")
    print numpy.std(darks[len(darks)/2:], ddof = 1)
    plt.show()

def avalancheUniformity():
    lightsoff = ["161103_153049.fits",
                 "161103_153133.fits"]
    lightson =  ["161103_153102.fits",
                 "161103_153142.fits"]
    start = 3
    
    gain = []
    biases = []
    
    on1 = openfits(lightson[0])
    off1 = openfits(lightsoff[0])
    onsub1 = subtractnth(on1,n=start)
    offsub1 = subtractnth(off1,n=start)

    on2 = openfits(lightson[1])
    off2 = openfits(lightsoff[1])
    onsub2 = subtractnth(on2,n=start)
    offsub2 = subtractnth(off2,n=start)

    d1 = onsub1[8,96:192,128:256] - offsub1[8,96:192,128:256]
    d2 = onsub2[8,96:192,128:256] - offsub2[8,96:192,128:256]

    d1 /= numpy.median(d1)
    d2 /= numpy.median(d2)
        
    plt.plot(d1.flatten(),d2.flatten(), 'o')
    plt.ylim([0.6,1.4])
    plt.xlim([0.6,1.4])
    plt.show()

def fftData(filename):
    d = openfits(filename)
    r = []
    for n in range(d.shape[0] - 1):
        r.append(abs(d[n,0,0] - d[n + 1, 0,0]))
    r = numpy.array(r)
    f = abs(numpy.fft.rfft(r))
    freq = numpy.fft.rfftfreq(r.shape[0], d = 1. / 256.e3)
    plt.plot(freq, f / numpy.sqrt(freq))
    plt.show()

def DonsNoiseTest():
    d = cdsCube(openfits("170131_091117.fits"))
    pix1 = 0
    pix2 = 3
    summed = numpy.zeros([d.shape[0]])
    diffed = numpy.zeros([d.shape[0]])
    for i in range(d.shape[0]):
        summed[i] = (d[i,0,pix1] + d[i,0,pix2]) / 2
        diffed[i] = (d[i,0,pix1] - d[i,0,pix2]) / 2
    d2 = cdsCube(openfits("170131_091034.fits"))
    summed2 = numpy.zeros([d.shape[0]])
    for i in range(d2.shape[0]):
        summed2[i] = (d2[i,0,pix1] + d2[i,0,pix2]) / 2
    a = 512
    avSummed = numpy.zeros([summed.shape[0]/a])
    avDiffed = numpy.zeros([diffed.shape[0]/a])
    avSummed2= numpy.zeros([summed2.shape[0]/a])
    for i in range(avSummed.shape[0]):
        avSummed[i] = numpy.sum(summed[i * a:(i + 1) * a])
    for i in range(avDiffed.shape[0]):
        avDiffed[i] = numpy.sum(diffed[i * a:(i + 1) * a])
    for i in range(avSummed2.shape[0]):
        avSummed2[i] = numpy.sum(summed2[i * a:(i + 1) * a])
    
    print "mean(sum):", numpy.mean(avSummed2) - numpy.mean(avSummed)
    print "median(sum):", numpy.median(avSummed2) - numpy.median(avSummed)
    print "var(sum):", numpy.var(avSummed2)
    print "median(diff):", numpy.median(avDiffed)
    print "var(diff):", numpy.var(avDiffed, ddof = 1)

    plt.plot(diffed)
    plt.plot(summed)
    plt.legend(["difference","sum"])
##    plt.plot(d[:,0,pix1] - numpy.mean(d[:,0,pix1]))
    plt.show()


def freqResponsePlot(filename):
    d = openfits(filename)
    d = avgCube(d, avg = 1024)
##    for i in range(1,d.shape[0]):
##        if numpy.median(d[i,:,:]) > 100:
##            print "FART"
    d = cdsCube(d)
    medians = []
    for i in range(d.shape[0]):
        medians.append(numpy.median(d[i,:,:]))
    plt.plot(medians)
    plt.show()

def freqResponse11Feb2017():
    files = ["170207_152001.fits",
             "170209_160437.fits",
             "170210_194044.fits",
             "170211_110238.fits",
             "170212_104757.fits"] #Mark 13 M06665-23
##             "180306_LED_on-off_cube-8.fits"] #Mark 15 M09105-27
##             "180314_LED_on-off_JKHenriksen_set2_cube-0.fits"]
##             "180315_155932.fits"] #LEACH
    
##    offsets = [53, 64, 28, 71, 17, 1]
##    offsets = [106, 129, 55, 142, 34, 77]
    offsets = [106, 129, 55, 142, 34]## 49]
##    averages = [512, 512, 512, 512, 512, 2] #PIZZA BOX
    averages = [512, 512, 512, 512, 512]## 512]
    
    spikeThresh = 5
    for f,offset,a in map(None,files,offsets,averages):
##        print f,offset,a
        d = openfits(f)
##        d = cdsCube(d)
        print d.shape
        s = numpy.array(d)
        #Remove resets.
        if f == files[0]:
            interval = 10000
            i = 9999
        elif f == files[2]:
            interval = 100000
            i = 99935
        else:
            interval = 100000
            i = 99999

##        if f == files[-1]:
##            s = s[:,128:160,144:176] / (-49) #-209 old
            
        while i < d.shape[0] - 1:
##            if abs(numpy.mean(d[i + 1,:,:]) - numpy.mean(d[i,:,:])) > spikeThresh and\
##               abs(numpy.mean(d[i + 2,:,:]) - numpy.mean(d[i,:,:])) > spikeThresh:
            s[i + 1:,:,:] -= numpy.median(s[i + 1,:,:]) - numpy.median(s[i,:,:])
##            print "Reset found at", i
            i += interval
##                s = numpy.delete(s, i, axis = 0)
##        plt.plot(numpy.median(numpy.median(d, axis = 2), axis = 1))
##        plt.plot(numpy.median(numpy.median(s, axis = 2), axis = 1))
        s = avgCube(s, avg = a)
        s = cdsCube(s, rolling = True)
        #Remove spikes.
        for i in range(1,s.shape[0]-1):
            if abs(numpy.median(s[i,:,:]) - numpy.median(s[i + 1,:,:])) > spikeThresh and\
               abs(numpy.median(s[i,:,:]) - numpy.median(s[i - 1,:,:])) > spikeThresh and\
               abs(numpy.median(s[i + 1,:,:]) - numpy.median(s[i - 1,:,:])) < abs(numpy.median(s[i,:,:]) - numpy.median(s[i + 1,:,:])):
                s[i,:,:] = (s[i+1,:,:] + s[i-1,:,:]) / 2
##        plt.plot(numpy.array(range(0,s.shape[0] * a * 2,a * 2)), 100 * numpy.median(numpy.median(s, axis = 2), axis = 1))
        t = numpy.array(range(s.shape[0] - offset)) * 1000. / 160. #* a / 265
        plt.plot(t, -1.58 * numpy.median(numpy.median(s, axis = 2) / (3.77e-6 * a), axis = 1)[offset:])
        print ".",
##        plt.show()
    plt.title("Pulse Delay for Mark 13 M06665-23")
##    plt.legend(["62.5K","67.6K","72.6K","77.7K","82.7K","M09105-27 62.5K"], loc = 2)
    plt.legend(["62.5K","67.6K","72.6K","77.7K","82.7K"], loc = 2)
    plt.ylabel("$e^{-} \mathrm{sec}^{-1} \mathrm{pix}^{-1}$")
    plt.xlabel("Time (ms)")
    plt.xlim([0,2000])
    plt.ylim([0,60000])
    plt.show()

def avalancheGainComparison13Feb2017():
    biases = [  2.5,  3.5,  4.5,  5.5,  6.5,  7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5]
    gains17um=[ 1.00, 1.31, 1.43, 2.00, 2.84, 3.79,5.36,7.91, 11.4, 16.4, 25.6, 35.8] #1.7um
    gains31um =[1.00, 1.14, 0.99, 1.21, 1.31, 1.41,1.67,1.95, 2.52, 3.55, 4.64, 6.41] #3.1um
    plt.semilogy(biases, gains17um, 'bo-')
    plt.semilogy(biases, gains31um, 'ro-')
    plt.legend(["$1.7{\mu}\mathrm{m}$","$3.1{\mu}\mathrm{m}$"], loc = 2)
    plt.ylabel("Avalanche Gain")
    plt.xlabel("$V_{bias}$")
    plt.title("Avalanche Gain vs. Bias Voltage for $1.7\mu\mathrm{m}$ and $3.1\mu\mathrm{m}$ Incident Light\nMark 13 M06665-25, $T = 60\mathrm{K}$")
    plt.xlim([0,14])
    plt.ylim([0.8,100.])
    plt.show()

def darkTrends13Feb2017(log = True):
    files =["151228_144652darkMap.fits",
            "151228_165637darkMap.fits",
            "151228_190622darkMap.fits",
            "151228_211617darkMap.fits",
            "151228_232602darkMap.fits",
            "151229_013547darkMap.fits"]
    biases  =  [2.5, 4.5, 6.5, 8.5, 9.5, 10.5]
    gain17um = [1.0, 1.5, 3.0, 5.2, 7.6, 12.5] #1.7um
    gain31um = [1.0, 1.0, 1.31, 1.67, 1.95, 2.52] #3.1um
    colors = ['b','r','g']
    medians = []
    medians17um = []
    medians31um = []
    for i,f in enumerate(files):
        d = openfits(f)
        dark = numpy.median(d[-32:,160:192])
        print dark,
        medians.append(dark * 2.89)
        medians17um.append(dark * 2.89 / gain17um[i])
        medians31um.append(dark * 2.89 / gain31um[i])
    if log:
        plt.semilogy(biases, medians, colors[0] + 'o-')
        plt.semilogy(biases, medians31um, colors[1] + 'o-')
        plt.semilogy(biases, medians17um, colors[2] + 'o-')
    else:
        plt.plot(biases, medians, colors[0] + 'o-')
        plt.plot(biases, medians31um, colors[1] + 'o-')
        plt.plot(biases, medians17um, colors[2] + 'o-')
    plt.xlabel("$V_{bias}$")
    plt.ylabel("$e^{-}/\mathrm{s}$")
    title = "Median Dark Current vs. Bias Voltage for M06715-27 @ 60K"
    ymax = 3.0
    plt.title(title)
    plt.xlim([2,11])
    plt.ylim([0.01, ymax])
    plt.legend(["uncorrected","corrected for $3.1\mu\mathrm{m}$ gain", "corrected for $1.7\mu\mathrm{m}$ gain"], loc = 2)
    plt.show()

def photonCountingPredictor():
    #Determine how well we can measure photons at each bias voltage, given the gain and dark current.
    gains = [ 1.00, 1.45, 1.53, 1.97, 2.79, 3.48, 4.33, 6.93, 9.88, 14.7, 27.7, 36.6, 65.6] #M06665-23
    darks = [0.044,0.044,0.044,0.044,0.044,0.044,0.044,0.150,0.450,1.500,4.500,15.00,45.00] #M06665-23 rough
    biases = [ 2.5,  3.5,  4.5,  5.5,  6.5,  7.5,  8.5,  9.5, 10.5, 11.5, 12.5, 13.5, 14.5]
    RN = 32. #e- rms
    a = 8.
    #Averaging correction on read noise.
    RN /= a**0.5
    #Construct an artificial gain curve based on the 14.5V results.
    sampleGain = 65.6
    gainProfile = numpy.array([-1,      0,      1,      0,      0,      1,      4,      1,
                                0,      2,      5,      1,      1,      1,      2,      3,
                                2,      2,      0,      1,      1,      0,      2,      2,
                                2,      3,      0,      1,      3,      3,     -1,      4,
                                0,      3,      1,      1,      2,      0,      2,      1,
                                6,      2,     -2,     -2,     -1,      4,     -1,      1,
                               -1,    -16,     -7,    -10,     -3,     -9,    -10,     -1,
                               -4,     -3,     -2,     -7,      2,     -9,     -1,     -5,
                               -4,     -9,     -2,    -10,    -13,     -7,     -5,     -7,
                               -4,     -2,     -9,      7,     -9,     -8,     -9,    -13,
                              -11,    -10,     -6,     -2,     -2,     -3,    -17,     -6,
                              -11,    -21,    -11,    -14,     -9,      1,    -17,    -13,
                              -14,    -13,     -6,    -15,      3,      8,      7,     -4,
                               14,    -13,      9,     -4,    -37,     -1,     13,    -28,
                               -5,     15,     -2,     -5,    -26,      3,    -17,    -12,
                               -4,      2,      1,     -2,     13,    -54,    -60,    -16,
                             -128,   -159,   -305,   -494,   -617,   -896,  -1344,  -2049,
                            -2554,  -3904,  -5109,  -5935,  -7835,  -9406, -10323, -11724,
                           -11944, -11137, -10237, -10121,  -8750,  -6768,  -7560,  -8390,
                            -8629,  -8606,  -8486,  -8568,  -6868,  -5534,  -3379,  -1738,
                             -476,   1177,   1922,   2977,   3349,   3837,   4244,   4436,
                             4460,   4481,   4589,   4711,   4539,   4487,   4524,   4528,
                             4407,   4131,   4031,   4033,   3968,   3966,   3771,   3758,
                             3513,   3475,   3367,   3331,   3235,   3139,   3022,   2915,
                             2850,   2729,   2688,   2630,   2430,   2450,   2312,   2239,
                             2139,   2133,   2023,   1922,   1846,   1757,   1759,   1664,
                             1629,   1529,   1471,   1426,   1409,   1348,   1283,   1275,
                             1104,   1154,   1076,   1024,    973,    959,    901,    883,
                              826,    793,    749,    750,    733,    701,    623,    624,
                              595,    570,    583,    496,    491,    461,    457,    418,
                              415,    426,    400,    390,    333,    348,    303,    309,
                              282,    283,    270,    245,    269,    220,    187,    210,
                              189,    191,    207,    162,    162,    140,    155,    141,
                              138,    125,    130,    127,    114,    105,    102,     98,
                              114,    100,     83,     93,     94,     76,     77,     82,
                               59,     65,     54,     51,     62,     58,     40,     41,
                               47,     42,     38,     49,     29,     42,     36,     19,
                               30,     34,     39])
    #Clean it up a little.
    for i in range(gainProfile.shape[0]):
        if gainProfile[i] < 0:
            gainProfile[i] = 0
    gainProfile = gainProfile.astype(float)
    #Make it a probability.
    gainProfile /= numpy.sum(gainProfile)
    #Bin centers.
    gainProfileSampleX = numpy.array([ -235.        , -233.42809365, -231.85618729, -230.28428094,
                                       -228.71237458, -227.14046823, -225.56856187, -223.99665552,
                                       -222.42474916, -220.85284281, -219.28093645, -217.7090301 ,
                                       -216.13712375, -214.56521739, -212.99331104, -211.42140468,
                                       -209.84949833, -208.27759197, -206.70568562, -205.13377926,
                                       -203.56187291, -201.98996656, -200.4180602 , -198.84615385,
                                       -197.27424749, -195.70234114, -194.13043478, -192.55852843,
                                       -190.98662207, -189.41471572, -187.84280936, -186.27090301,
                                       -184.69899666, -183.1270903 , -181.55518395, -179.98327759,
                                       -178.41137124, -176.83946488, -175.26755853, -173.69565217,
                                       -172.12374582, -170.55183946, -168.97993311, -167.40802676,
                                       -165.8361204 , -164.26421405, -162.69230769, -161.12040134,
                                       -159.54849498, -157.97658863, -156.40468227, -154.83277592,
                                       -153.26086957, -151.68896321, -150.11705686, -148.5451505 ,
                                       -146.97324415, -145.40133779, -143.82943144, -142.25752508,
                                       -140.68561873, -139.11371237, -137.54180602, -135.96989967,
                                       -134.39799331, -132.82608696, -131.2541806 , -129.68227425,
                                       -128.11036789, -126.53846154, -124.96655518, -123.39464883,
                                       -121.82274247, -120.25083612, -118.67892977, -117.10702341,
                                       -115.53511706, -113.9632107 , -112.39130435, -110.81939799,
                                       -109.24749164, -107.67558528, -106.10367893, -104.53177258,
                                       -102.95986622, -101.38795987,  -99.81605351,  -98.24414716,
                                        -96.6722408 ,  -95.10033445,  -93.52842809,  -91.95652174,
                                        -90.38461538,  -88.81270903,  -87.24080268,  -85.66889632,
                                        -84.09698997,  -82.52508361,  -80.95317726,  -79.3812709 ,
                                        -77.80936455,  -76.23745819,  -74.66555184,  -73.09364548,
                                        -71.52173913,  -69.94983278,  -68.37792642,  -66.80602007,
                                        -65.23411371,  -63.66220736,  -62.090301  ,  -60.51839465,
                                        -58.94648829,  -57.37458194,  -55.80267559,  -54.23076923,
                                        -52.65886288,  -51.08695652,  -49.51505017,  -47.94314381,
                                        -46.37123746,  -44.7993311 ,  -43.22742475,  -41.65551839,
                                        -40.08361204,  -38.51170569,  -36.93979933,  -35.36789298,
                                        -33.79598662,  -32.22408027,  -30.65217391,  -29.08026756,
                                        -27.5083612 ,  -25.93645485,  -24.36454849,  -22.79264214,
                                        -21.22073579,  -19.64882943,  -18.07692308,  -16.50501672,
                                        -14.93311037,  -13.36120401,  -11.78929766,  -10.2173913 ,
                                         -8.64548495,   -7.0735786 ,   -5.50167224,   -3.92976589,
                                         -2.35785953,   -0.78595318,    0.78595318,    2.35785953,
                                          3.92976589,    5.50167224,    7.0735786 ,    8.64548495,
                                         10.2173913 ,   11.78929766,   13.36120401,   14.93311037,
                                         16.50501672,   18.07692308,   19.64882943,   21.22073579,
                                         22.79264214,   24.36454849,   25.93645485,   27.5083612 ,
                                         29.08026756,   30.65217391,   32.22408027,   33.79598662,
                                         35.36789298,   36.93979933,   38.51170569,   40.08361204,
                                         41.65551839,   43.22742475,   44.7993311 ,   46.37123746,
                                         47.94314381,   49.51505017,   51.08695652,   52.65886288,
                                         54.23076923,   55.80267559,   57.37458194,   58.94648829,
                                         60.51839465,   62.090301  ,   63.66220736,   65.23411371,
                                         66.80602007,   68.37792642,   69.94983278,   71.52173913,
                                         73.09364548,   74.66555184,   76.23745819,   77.80936455,
                                         79.3812709 ,   80.95317726,   82.52508361,   84.09698997,
                                         85.66889632,   87.24080268,   88.81270903,   90.38461538,
                                         91.95652174,   93.52842809,   95.10033445,   96.6722408 ,
                                         98.24414716,   99.81605351,  101.38795987,  102.95986622,
                                        104.53177258,  106.10367893,  107.67558528,  109.24749164,
                                        110.81939799,  112.39130435,  113.9632107 ,  115.53511706,
                                        117.10702341,  118.67892977,  120.25083612,  121.82274247,
                                        123.39464883,  124.96655518,  126.53846154,  128.11036789,
                                        129.68227425,  131.2541806 ,  132.82608696,  134.39799331,
                                        135.96989967,  137.54180602,  139.11371237,  140.68561873,
                                        142.25752508,  143.82943144,  145.40133779,  146.97324415,
                                        148.5451505 ,  150.11705686,  151.68896321,  153.26086957,
                                        154.83277592,  156.40468227,  157.97658863,  159.54849498,
                                        161.12040134,  162.69230769,  164.26421405,  165.8361204 ,
                                        167.40802676,  168.97993311,  170.55183946,  172.12374582,
                                        173.69565217,  175.26755853,  176.83946488,  178.41137124,
                                        179.98327759,  181.55518395,  183.1270903 ,  184.69899666,
                                        186.27090301,  187.84280936,  189.41471572,  190.98662207,
                                        192.55852843,  194.13043478,  195.70234114,  197.27424749,
                                        198.84615385,  200.4180602 ,  201.98996656,  203.56187291,
                                        205.13377926,  206.70568562,  208.27759197,  209.84949833,
                                        211.42140468,  212.99331104,  214.56521739,  216.13712375,
                                        217.7090301 ,  219.28093645,  220.85284281,  222.42474916,
                                        223.99665552,  225.56856187,  227.14046823,  228.71237458,
                                        230.28428094,  231.85618729,  233.42809365])
    for i in range(len(gains)):
        dark = darks[i]
        bias = biases[i]
        gain = gains[i]
        print str(bias) + "V"
        print "RN:", RN
        #Scale gain profile to current gain value.
        gainProfileX = gainProfileSampleX * (gain / sampleGain)
##        print gainProfileX[-1]
        RNprob = signal.gaussian(400, RN)
##        print numpy.sum(RNprob)
        RNprob /= numpy.sum(RNprob)
##        print RNprob
        RNprob = RNprob[RNprob.shape[0]/2:]
        FPs = [] #False positives as a % of reads
        TEs = [] #Threshold efficiencies
        for j in range(int(gainProfileX[-1])):
            FPs.append(numpy.sum(RNprob[j:]))
            gainIndex = 0
            while gainProfileX[gainIndex] < j:
                gainIndex += 1
            TEs.append(1. - numpy.sum(gainProfile[gainIndex:]))
##        plt.plot(RNprob)
##        plt.plot(numpy.array(range(int(gainProfileX[-1]))), FPs)
##        plt.plot(numpy.array(range(int(gainProfileX[-1]))), TEs)
##        plt.show()
        j = 0
        while TEs[j] < 0.95 and j + 1 < len(TEs):
            j += 1
        fullFP = FPs[j] * 8e6 #convert to Hz
        fullFP += dark
        print "False Positives:", str(100 * FPs[j])+"%"
        print "FP_RN:", FPs[j] * 8e6, "e-/s/pix"
        print "FP Rate:", fullFP, "e-/s/pix"
        
def fullDarkSeries(device = "M06715-27", T = 60):
    if device == "M06665-23" and T == 60:
        firstFile = "170302_161656.fits"
        lastFile = "170304_171239.fits"
        chargeGain = 1.59
        cutoff = 40000
    elif device == "M06665-23" and T == 40:
        firstFile = "170305_125411.fits"
        lastFile = "170307_134944.fits"
        chargeGain = 1.59
        cutoff = 40000
    elif device == "M06715-27" and T == 60:
        firstFile = "170218_224956.fits"
        lastFile = "170220_234534.fits"
        chargeGain = 2.89
        cutoff = 40000
    elif device == "M09225-27" and T == 60:
##        firstFile = "170314_113225.fits"
##        lastFile = "170316_124102.fits"
##        lastFile = "170317_123511.fits"
        firstFile = "170317_213936.fits"
        lastFile = "170320_093622.fits"
        chargeGain = 5.18
        cutoff = 20000
    elif device == "M09225-11" and T == 60:
        firstFile = "170328_141628.fits"
        lastFile = "170331_061524.fits"
        chargeGain = 4.71
        cutoff = 40000
    elif device == "M09215-10" and T == 60:
        firstFile = "170411_094206.fits"
        lastFile = "170413_080323.fits"
        chargeGain = 4.71
        cutoff = 40000
    else:
        print "No data for requested device."
        return
    files = listdir("./")
    if not ((firstFile in files) and (lastFile in files)):
        print "I didn't find one of endcap files."
    startIndex = files.index(firstFile)
    endIndex = files.index(lastFile)
    n = 100
##    start = 200
##    end = -1
    length = 200

    darks = []
    for i in range(startIndex, endIndex + 1):
        if ".fits" in files[i] and len(files[i]) == 18:
            d = openfits(files[i])
            if d.shape[0] == 360 or d.shape[0] == 300:
                timing = 5
            elif d.shape[0] > 590:
                timing = 0.2
            elif d.shape[0] == 120:
                timing = 30
            elif d.shape[0] == 24:
                timing = 300
                n = 10
                length = 20
            else:
                print "UHHH WHAT"
                print files[i]
                print d.shape[0]
                return
            if d.shape[1] == 256:
                d = d[:,224:,:]
            e = numpy.zeros([d.shape[1],d.shape[2]])
            k = 1
            while numpy.median(d[-k,:,:]) < cutoff and k < d.shape[0] - 1:
                k += 1
            if k >= d.shape[0] - 1:
                print "Bad reset?"
                k = 1
            end = d.shape[0] - k
            start = end - length
            if start < 0:
                start = 0
    ##            print "Warning, bad range."
            if end < 0:
                end += d.shape[0]
            for j in range(0,n):
                e += d[start + j,:,:]
                e -= d[end - j,:,:]
            e /= n
            e /= end - (start + n) #Modified 29 Feb 2016.
            e /= timing #Added 7 Apr 2015.
            print files[i], round(numpy.median(e) * chargeGain,3), "e-/s", end - start, "Frames", "Timing:", timing, "s"
            darks.append(numpy.median(e) * chargeGain)
    plt.plot(numpy.array(range(0,len(darks) * 15, 15)), darks, 'ko-')
    plt.ylabel("$e^{-}/\mathrm{s}$")
    plt.xlabel("time (min)")
    plt.xlim([0, 15 * len(darks)])
##    plt.ylim([0,0.40])
    title = "Measured Dark vs. $V_{bias}$ for " + device + " @ " + str(T) + "K"
    plt.title(title)
    plt.show()

def finalDarkPlots():
    Vbias =              [  1.0,  1.5,  2.5,  3.5,  4.5,  5.5,  6.5,  7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5]# 14.5]
    dark  = numpy.array([[0.025,0.024,0.025,0.029,0.036,0.045,0.070, 0.11,0.24,0.78,  4.4,  23., 127., 640.], #M06715-27
                         [0.035,0.063,0.031,0.036,0.063,0.058,0.074,0.079,0.12,0.20, 0.46,  1.2,  3.6,  8.6]])#M09225-27
    gain  = numpy.array([[ 1.00, 1.00, 1.00, 1.14, 1.51, 2.08, 2.73, 4.06,5.72,8.42, 12.6, 18.5, 27.6, 41.5],
                        [   0.8,  0.8,  1.0,  1.2, 1.52, 2.05, 2.86, 4.17,6.15, 9.2, 13.9, 21.1, 32.4, 55.7]])
    for i in range(dark.shape[0]):
        for d in range(dark.shape[1]):
            dark[i,d] /= gain[i,d]
        plt.semilogy(Vbias,dark[i,:],'o-')
    plt.ylabel("$e^{-} \mathrm{sec}^{-1} \mathrm{pix}^{-1}$ (gain-corrected)")
    plt.xlabel("$V_{bias}$")
    plt.legend(["M06715-27 ME911  mk14",
                "M09225-27 ME1001 mk19"], loc = 2)
    plt.show()

def fig2DarkCurrentPaper(gainCorrect = False):
##    Vbias = [  2.5,  3.5,  4.5,  5.5,  6.5,  7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5]
##    dark  = [0.018,0.040,0.033,0.049,0.058,0.142,0.53,3.17,17.37, 41.9,131.6,609.4] #M02775-35
##    gain  = [  1.0, 1.23, 1.47, 1.85, 2.36, 4.54,6.91,11.7, 18.1, 28.4, 45.1, 72.7]
      #          10s    5s   10s    5s   10s    5s   5s   5s    5s    5s
    Vbias = [  3.5,  5.5,  7.5, 8.5, 9.5, 10.5, 11.5]
    dark  = [0.040,0.049,0.142,0.53,3.17,17.37, 87.9] #M02775-35
    gain  = [ 1.22, 1.85, 4.54,6.91,11.7, 18.1, 28.4]
    dark = numpy.array(dark)
    for i in range(dark.shape[0]):
        if gainCorrect:
            dark[i] /= gain[i]
    plt.semilogy(Vbias,dark[:],'o-', label = "Mark 3 M02775-35")
    
    Vbias = [  1.0,  1.5,  2.5,  3.5,  4.5,  5.5,  6.5,  7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5]
    dark  = [0.026,0.028,0.028,0.029,0.041,0.046,0.060,0.082,0.15,0.38,  2.4,  11.,  62., 280.,1700.] #M06665-23
    gain  = [ 1.00, 1.00, 1.00, 1.31, 1.43, 2.00, 2.84, 3.79,5.36,7.91, 11.4, 16.4, 25.6, 35.8, 56.4]

    dark = numpy.array(dark)
    for i in range(dark.shape[0]):
        if gainCorrect:
            dark[i] /= gain[i]
    plt.semilogy(Vbias,dark[:],'o-', label = "Mark 13 M06665-23")
    
    Vbias = [  1.0,  1.5,  2.5,  3.5,  4.5,  5.5,  6.5,  7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5]
    dark  = [0.025,0.024,0.025,0.029,0.036,0.045,0.070, 0.11,0.24,0.78,  4.4,  23., 127., 640.,2300.] #M06715-27
    gain  = [ 1.00, 1.00, 1.00, 1.31, 1.43, 2.00, 2.84, 3.79,5.36,7.91, 11.4, 16.4, 25.6, 35.8, 56.4]

    dark = numpy.array(dark)
    for i in range(dark.shape[0]):
        if gainCorrect:
            dark[i] /= gain[i]
    plt.semilogy(Vbias,dark[:],'o-', label = "Mark 14 M06715-27")

##    Vbias = [  4.5,  5.5,  6.5,  7.5,  8.5,  9.5, 10.5, 11.5, 12.5, 13.5]
##    dark  = [0.014,0.012,0.015,0.016,0.039,0.054,0.100,0.308,0.704,1.843] #M09225-11
##    gain  = [ 1.84, 2.37, 3.24, 4.60, 6.76, 10.0, 15.0, 23.0, 35.3, 53.4]
####    gain = gain / gain[1]
##
##    dark = numpy.array(dark)
##    for i in range(dark.shape[0]):
##        if gainCorrect:
##            dark[i] /= gain[i]
##    plt.semilogy(Vbias,dark[:],'o-', label = "M09225-11 ME1001 mk19")

    Vbias = [  1.0,  2.5,  3.5,  4.5,  5.5,  6.5,  7.5, 8.5, 9.5, 10.5, 11.5, 12.5]
    dark  = [0.035,0.031,0.036,0.063,0.058,0.074,0.079,0.12,0.20, 0.46,  1.2,  3.6] #M09225-27
    gain  = [  0.8,  1.0,  1.2, 1.52, 2.05, 2.86, 4.17,6.15, 9.2, 13.9, 21.1, 32.4]

    dark = numpy.array(dark)
    for i in range(dark.shape[0]):
        if gainCorrect:
            dark[i] /= gain[i]
    plt.semilogy(Vbias,dark[:],'o-', label = "Mark 19 M09225-27")
    
    ylabel = "$e^{-} \mathrm{sec}^{-1} \mathrm{pix}^{-1}$"
    if gainCorrect:
        ylabel += " (gain-corrected)"
    
    plt.ylabel(ylabel)
    plt.xlabel("$V_{bias}$")
    plt.xlim([0,15])
    plt.legend(loc = 2)
    plt.title("Median Dark Current vs. Bias Voltage\nread interval 5s, $T = 62.5\mathrm{K}$")
    plt.show()

def timingDarkSeries(firstFile, lastFile, chargeGain = 2.89):
    files = listdir("./")
    if not ((firstFile in files) and (lastFile in files)):
        print "I didn't find one of endcap files."
    startIndex = files.index(firstFile)
    endIndex = files.index(lastFile)
##    timing = [5, 5, 5, 5, 5, 5, 5, 5,
##              3, 3, 3, 3,
##              1, 1, 1, 1,
##              0.3, 0.3, 0.3, 0.3]
##              5, 5, 5, 5]
##    timing = [30, 30, 100, 100, 300, 300]
##    timing = [5, 5, 5, 5, 4, 3, 2., 1.]
    timing = [10, 10, 10, 10, 10, 10, 10, 10,
              3, 3, 3, 3,
              1, 1, 1, 1,
              0.5, 0.5, 0.5, 0.5,
              0.3, 0.3, 0.3, 0.3]
              
    n = 10
##    start = 1
    end = -1
##    length = 1000
##    timing = 5
    darks = []
    for i in range(startIndex, endIndex + 1):
        if ".fits" in files[i]:
            start = 1
            end = -1
            try:
                d = openfits(files[i])[:,-32:,64:-64]
##                if d.shape[0] > 1000:
##                    d = d[1001:,:,:]
                d = d[d.shape[0] / 3:,:,:]
##                plt.plot(numpy.median(numpy.median(d, axis = 2), axis = 1))
##                plt.show()
                e = numpy.zeros([d.shape[1],d.shape[2]])
                if end < 0:
                    end += d.shape[0]
##                start = end - length
                if start < 1:
                    print "Possibly bad range."
                    start = 1
                for j in range(0,n):
                    e += d[j + start,:,:]
                    e -= d[j - n + end,:,:]
                e /= n
                e /= end - (start + n) #Modified 29 Feb 2016.
                e /= timing[0] #Added 7 Apr 2015.
                print files[i], "Timing", timing[0],"Median dark current:", numpy.median(e) * chargeGain, "e-/s"
                timing.pop(0)
                darks.append(numpy.median(e) * chargeGain)
            except IOError:
                print "Bad file:", files[i]
    plt.plot(numpy.array(range(0,len(darks) * 30, 30)), darks, 'ko-')
    plt.ylabel("$e^{-}/\mathrm{s}$")
    plt.xlabel("time (min)")
    plt.xlim([0, 30 * len(darks)])
##    plt.ylim([0,0.40])
##    plt.title("Measured Dark for M06715-27 @ 60K, $V_{bias} = 1.5V$")
    plt.show()

def DonsCombinedPlot28Feb2017():
    Vbias = [  1.0,  1.5,  2.5,  3.5,  4.5,  5.5,  6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5]
    dark  = [0.025,0.024,0.025,0.029,0.036,0.045,0.070,0.11,0.24,0.78,  4.4,  23., 127., 640.,2300.]
    gain  = [ 1.00, 1.00, 1.00, 1.31, 1.43, 2.00, 2.84, 3.79, 5.36, 7.91, 11.4, 16.4, 25.6, 35.8, 56.4]
    olddark = numpy.copy(dark)
    for d in range(len(dark)):
        dark[d] /= gain[d]
    fig, ax1 = plt.subplots()
    lns1 = ax1.semilogy(Vbias,gain,'bo-', label = "Avalanche Gain")
    ax1.set_ylim([0.1,1000])
    ax1.set_ylabel("Avalanche Gain")
    ax1.set_xlabel("$V_{bias}$")

    #Horizontal line at unity gain.
##    ax1.plot([0,14.5],[1.0,1.0],'b--', label = "Unity Gain")

    #Best-fit for exponential slope gain.
    m, b = linearfit(Vbias[4:],numpy.log10(gain[4:]))
    print 1/m
##    ax1.plot([2.5, 14.5],[10**(m * 2.5 + b), 10**(m * 14.5 + b)],'b-.', label = "10^(" + str(round(m,2)) + "$V_{bias}$" + str(round(b,2)) + "V)")
    
    ax2 = ax1.twinx()
    lns3 = ax2.semilogy(Vbias,olddark,'go-', label = "Dark Current")
    lns2 = ax2.semilogy(Vbias,dark,'ro-', label = "Gain-Normalized Dark")
    ax2.set_ylabel("Gain-Normalized Dark Current $e^{-}\mathrm{s}^{-1}\mathrm{pix}^{-1}$")
    ax2.set_ylim([0.01,10000])
    meanDark = numpy.mean(dark[:8])

    #Horizontal fit line for low-bias dark current.
##    ax2.plot([1,16],[meanDark,meanDark],'r--', label = "Mean Dark = " + str(round(meanDark,2)) + "$e^{-}\mathrm{s}^{-1}\mathrm{pix}^{-1}$")
    print round(meanDark,3), "+/-", numpy.std(dark[:8],ddof = 1)
    #Best-fit for high-bias dark.
    m, b = linearfit(Vbias[9:-1],numpy.log10(dark[9:-1]))
    print 1/m
##    ax2.plot([8., 14.5],[10**(m * 8. + b), 10**(m * 14.5 + b)],'r-.', label = "10^(" + str(round(m,2)) + "$V_{bias}$" + str(round(b,2)) + "V)")
    
    lns = lns1 + lns3 + lns2
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, loc = 2)
    plt.title("Avalanche Gain and Gain-Normalized Dark Current\nMark 14 M06715-27, read interval 5s, 60K")
    plt.show()

def TimingComparisonPlots9Mar2017():
    plt.plot([5,0.2],[0.8,1.1],  'r^-',  label = "M06715-27 $V_{bias} = 9.5\mathrm{V}$ 60K")
    plt.plot([5,0.2],[0.15,0.50],'ro-',  label = "M06665-23 $V_{bias} = 8.5\mathrm{V}$ 60K")
    plt.plot([5,0.2],[0.38,0.99],'ro--', label = "M06665-23 $V_{bias} = 9.5\mathrm{V}$ 60K")
    plt.plot([5,0.2],[0.10,0.70],'bo-',  label = "M06665-23 $V_{bias} = 8.5\mathrm{V}$ 40K")
    plt.plot([5,0.2],[0.39,1.50],'bo--', label = "M06665-23 $V_{bias} = 9.5\mathrm{V}$ 40K")
    plt.plot([5,2,1,0.5,0.2],[0.03,0.07,0.04,0.05,0.14],'ro:', label = "M06665-23 $V_{bias} = 2.5\mathrm{V}$ 60K")
    plt.legend(loc = 2)
    plt.xlim([5.2,0])
    plt.ylim([0.0,2.0])
    plt.title("Timing vs. Dark")
    plt.xlabel("Readout Interval")
    plt.ylabel("$e^{-} \mathrm{s}^{-1} \mathrm{pix}^{-1}$")
    plt.show()

def darkLongIntervalHistogram10Mar2017():
    d = openfits("170309_144226darkMap.fits")
    d *= 1.58 #charge gain
    print d.shape
    plt.hist([d[224:,32:-32],d[:32,32:-32]],bins = 40,range = [-0.01 * 1.58,0.03 * 1.58])
    plt.legend(["top: y>223, 288>x>32","bottom:32>y, 288>x>32"])
    plt.ylabel("n")
    plt.xlabel("$e^{-} \mathrm{sec}^{-1} \mathrm{pix}^{-1}$")
    plt.show()

def mk14darkTrends20Mar2017(log = True, gainCorrect = False, gainXaxis = False):
##    files =["170317_233436.fits", #1V
##            "170318_032643.fits", #1.5V
##            "170318_071850.fits", #2.5V
##            "170320_054414.fits", #3.5V
##            "170318_111057.fits", #4.5V
##            "170320_015208.fits", #5.5V
##            "170318_150304.fits", #6.5V
##            "170319_220000.fits", #7.5V
##            "170318_185511.fits", #8.5V
##            "170319_180754.fits", #9.5V
##            "170318_224718.fits"] #10.5V
    files = ["170219_022626.fits", #1V
             "170219_063558.fits", #1.5V
             "170219_104530.fits", #2.5V
             "170220_193602.fits", #3.5V
             "170219_145502.fits", #4.5V
             "170220_152630.fits", #5.5V
             "170219_190439.fits", #6.5V
             "170220_111658.fits", #7.5V
             "170219_231416.fits", #8.5V
             "170220_070726.fits", #9.5V
             "170220_004434.fits", #10.5V
             "170220_025334.fits", #11.5V
             "170220_011649.fits", #12.5V
             "170220_022119.fits", #13.5V
             "170220_014904.fits"] #14.5V
    
    biases = [  1.0,  1.5,  2.5,  3.5,  4.5,  5.5,  6.5,  7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5]
##    percentage = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    percentage = [0.25, 0.50, 0.75, 0.85, 0.90, 0.95, 0.99]
##    percentage = [0.50, 0.75, 0.90, 0.95, 0.99]
##    gains =  [  0.8,  0.8,  1.0,  1.2, 1.52, 2.05, 2.86, 4.17,6.15, 9.2, 13.9]
    gains =  [ 1.00, 1.00, 1.00, 1.31, 1.43, 2.00, 2.84, 3.79, 5.36, 7.91, 11.4, 16.4, 25.6, 35.8, 56.4]
##    colors = ['b','r','g','k','m','y','c']
    colors = ['m','k','b','c','g','y','r']
##    colors = ['k','b','g','y','r']
    cutoff = 40000

    chargeGain = 2.89
    timing = 5
    n = 20
    length = 200

    for i,p in enumerate(percentage):
        medians = []
        for f in files:
            d = openfits(f)
            if d.shape[0] == 360:
                timing = 5
            elif d.shape[0] > 590:
                timing = 0.2
            if d.shape[1] == 256:
                d = d[:,224:,32:-32]
            e = numpy.zeros([d.shape[1],d.shape[2]])
            k = 1
            while numpy.median(d[-k,:,:]) < cutoff:
                k += 1
            end = d.shape[0] - k
            start = end - length
            if start < 1:
                start = 1
    ##            print "Warning, bad rang180846e."
            if end < 0:
                end += d.shape[0]
            print f, end, start
            for j in range(0,n):
                e += d[start + j,:,:]
                e -= d[end - j,:,:]
            e /= n
            e /= end - (start + n)
            e /= timing
            e *= chargeGain
            
            sortedList = numpy.sort(e[-32:,160:192].flatten())
            dark = sortedList[int(p * len(sortedList))]
            medians.append(dark)
        if gainCorrect:
            for m in range(len(medians)):
                medians[m] /= gains[m]
        if gainXaxis:
            Xaxis = gains
        else:
            Xaxis = biases
        if log:
            plt.semilogy(Xaxis[:-3], medians[:-3], colors[i] + 'o-')
        else:
            plt.plot(Xaxis, medians, colors[i] + 'o-')
    if gainXaxis:
        plt.xlabel("APD gain")
    else:
        plt.xlabel("$V_{bias}$")
    ylab = "$e^{-}\mathrm{s}^{-1}\mathrm{pix}^{-1}$"
    if gainCorrect:
        ylab += " (gain-corrected)"
    plt.ylabel(ylab)
##    plt.legend(["5%","10%","25%","50%","75%","90%","95%"], loc = 2)    
    plt.legend(["25%","50%","75%","85%","90%","95%","99%"], loc = 2)
##    plt.legend(["50%","75%","90%","95%","99%"], loc = 2)
    title = "Percentile Dark Current"
    if gainCorrect:
        title += " (gain-corrected)"
        ymax = 100.00
    else:
        title += " (no gain correction)"
        ymax = 10000.0
##    title +=  " vs. Bias Voltage\nMark 19 M09225-27, readout interval 300s, $T = 60\mathrm{K}$"
    title +=  " vs. Bias Voltage\nMark 14 M06715-27, readout interval 5s/0.2s, $T = 60\mathrm{K}$"
    plt.title(title)
    plt.xlim([0,12])
    if log:
        plt.ylim([0.001, ymax])
    else:
        plt.ylim([0, ymax])
    plt.show()

def mk13darkTrends20Mar2017(log = True):
    biases = [  1.0,  1.5,  2.5,  3.5,  4.5,  5.5,  6.5,  7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5]
    darks =  [0.023,0.025,0.024,0.025,0.031,0.037,0.044,0.060,0.10,0.39,  4.0,  21., 123., 840.] #M06665-23
    gain17um=[ 1.00, 1.00, 1.00, 1.31, 1.43, 2.00, 2.84, 3.79,5.36,7.91, 11.4, 16.4, 25.6, 35.8] #1.7um
    gain31um = [1.0,  1.0, 1.00, 1.14, 0.99, 1.21, 1.31, 1.41,1.67,1.95, 2.52, 3.55, 4.64, 6.41] #3.1um
    colors = ['b','r','g']
    medians = []
    medians17um = []
    medians31um = []
    for i, dark in enumerate(darks):
        print dark,
        medians.append(dark)
        medians17um.append(dark / gain17um[i])
        medians31um.append(dark / gain31um[i])
    if log:
        plt.semilogy(biases, medians, colors[0] + 'o-')
        plt.semilogy(biases, medians31um, colors[1] + 'o-')
        plt.semilogy(biases, medians17um, colors[2] + 'o-')
    else:
        plt.plot(biases, medians, colors[0] + 'o-')
        plt.plot(biases, medians31um, colors[1] + 'o-')
        plt.plot(biases, medians17um, colors[2] + 'o-')
    plt.xlabel("$V_{bias}$")
    plt.ylabel("$e^{-}\mathrm{s}^{-1}\mathrm{pix}^{-1}$")
    title = "Median Dark Current vs. Bias Voltage, Varying Gain Correction\nMark 13 M06665-23, readout interval 5s, $T = 60\mathrm{K}$"
    ymax = 1000.0
    plt.title(title)
    plt.xlim([0,14])
    plt.ylim([0.01, ymax])
    plt.legend(["uncorrected","corrected for $3.1\mu\mathrm{m}$ gain", "corrected for $1.7\mu\mathrm{m}$ gain"], loc = 2)
    plt.show()

def mk19DarkDiagnosis():
    d = openfits("170315_200053.fits")
##    d = openfits("170306_211136.fits")
    i = 1
    while i < d.shape[0]:
        hist = numpy.histogram(d[i,:,:].flatten(), range = [20000,50000], bins = 30)
##        print len(hist[0])
##        print len(hist[1])
        plt.plot(hist[1][1:], hist[0], label = "Frame #" + str(i))
        i += 20
    plt.legend()
    plt.show()

def timingPlotForPaper():
##    intervals = [0.2, 0.5, 1, 2, 5, 300]
##    darks = [0.144, 0.050, 0.39, 0.069, 0.031, 0.007]
    intervals = [#5, 3, 1, 0.3,
                 5, 3, 1, 0.3, 5,
##                 5, 4, 3, 2, 1,
                 30, 100, 300,
                 10, 3, 1, 0.5, 0.3]
    darks =     [#0.074, 0.084, 0.103, 0.174,
                 0.047, 0.051, 0.056, 0.069, 0.039,
##                 0.059, 0.037, 0.033, 0.028, 0.029,
                 0.007, 0.008, 0.007,
                 0.042, 0.052, 0.064, 0.054, 0.078]
    intervals = 1 / numpy.array(intervals)
    plt.plot(intervals, darks, 'ko')
    plt.ylabel("$e^{-}\mathrm{s}^{-1}\mathrm{pix}^{-1}$")
##    plt.xlabel("Read Interval (s)")
    plt.xlabel("Read Rate (Hz)")    
##    plt.title("Median Dark Current vs. Read Interval\nMark 13 M06665-23, $V_{bias} = 2.5\mathrm{V}$, $T = 60\mathrm{K}$")
    plt.title("Median Dark Current vs. Read Interval\nMark 19 M09225-11, $V_{bias} = 2.5\mathrm{V}$, $T = 60\mathrm{K}$")
    plt.show()

def FalsePositivePlot():
    #Plot dark next to false positive rate due to threshold selection.
    Vbiases=[  1.0,  1.5,  2.5,  3.5,  4.5,  5.5,  6.5,  7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5]
    darks = [0.026,0.028,0.028,0.029,0.041,0.046,0.060,0.082,0.15,0.38,  2.4,  11.,  62., 280.,1700.] #M06665-23
    gains = [ 1.00, 1.00, 1.00, 1.31, 1.43, 2.00, 2.84, 3.79,5.36,7.91, 11.4, 16.4, 25.6, 35.8, 56.4]
    RN = 12 #e-RMS
    RNdist = signal.gaussian(gains[-1] * 2000, RN * 10)
    values = numpy.linspace(-gains[-1] * 100, gains[-1] * 100, gains[-1] * 2000)
    thresh0 = 20
    gain0 = 56.4
    readRate = 265.e3 / (8 * 32)

    #Project a value.
    Vbiases.append(Vbiases[-1] + 1.0)
    darks.append(darks[-1]**2 / darks[-2])
    gains.append(gains[-1]**2 / gains[-2])

    #Interpolate.
##    for i in range(

    FPrates = []
    for Vbias, dark, gain in map(None,Vbiases,darks,gains):
        thresh = (gain / gain0) * thresh0
        tIndex = 0
        while values[tIndex] < thresh:
            tIndex += 1
        FPprob = numpy.sum(RNdist[tIndex:]) / numpy.sum(RNdist)
        FPrates.append(FPprob * readRate)
        print Vbias, thresh, FPrates[-1]

    darks = numpy.array(darks) / numpy.array(gains)
    FPrates = numpy.array(FPrates)

    

##    projectedVbiases = [Vbiases[-1], Vbiases[-1] + 1.0, Vbiases[-1] + 2.0]
##    projectedDarks = [darks[-1],darks[-1]**2 / darks[-2]]
##    projectedDarks.append(projectedDarks[-1]**2 / projectedDarks[-2])
##    projectedGains = [gains[-1]**2 / gains[-2]]
##    projectedGains.append(projectedGains[-1]**2 / gains[-1])
##    
##    projectedFPrates = [FPrates[-1],FPrates[-1] **2 / FPrates[-2]]
##    projectedFPrates.append(projectedFPrates[-1]**2 / projectedFPrates[-2])

##    projectedDarks = numpy.array(projectedDarks)
##    projectedFPrates = numpy.array(projectedFPrates)

    plt.semilogy(gains, darks, 'bo-', label = "dark current")
    plt.semilogy(gains,FPrates,'ro-', label = "false positives")
    plt.semilogy(gains, darks + FPrates, 'ko-', label = "total")
##    plt.semilogy(projectedVbiases, projectedDarks, 'bo--')
##    plt.semilogy(projectedVbiases, projectedFPrates, 'ro--')
##    plt.semilogy(projectedVbiases, projectedDarks + projectedFPrates, 'ko--')
    plt.ylabel("$\mathrm{s}^{-1}\mathrm{pix}^{-1}$")
    plt.xlabel("Gain")
    plt.title("False Positive Rate in a 32 x 32 Subarray with 95% Threshold\nMark 13 M06665-23, Pixel Rate 256 kHz, $T = 62.5\mathrm{K}$")
    plt.legend(loc = 4)
    plt.show()


def measureFbetter(offFile = "170130_100554.fits",onFile = "170130_100602.fits",gain = 65.6):
##    offFile = "170130_095713.fits"
##    onFile =  "170130_095725.fits"
##    gain = 1.0
##    offFile = "170130_095809.fits"
##    onFile = "170130_095820.fits"
##    offFile = "170130_100455.fits"
##    onFile = "170130_100510.fits"
##    gain = 50.
##    offFile = "170130_100554.fits"
##    onFile = "170130_100602.fits"
##    gain = 65.6

    print offFile, onFile, gain
    
    avg = [1,2,4,8,16,32]
##    avg = [1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16284]
##    avg = [32,64,128,256,512,1024,2048,4096,8192]
    spikeCutoff = 500 #10
    threshold = 20
    
    off = openfits(offFile)
    on = openfits(onFile)
    off = numpy.delete(off, [29], 2) #JUST THE DEAD ONE
    on = numpy.delete(on, [29], 2)
##    off = numpy.delete(off, [0,1,8,21,23,25,26,28,29], 2) #JUST THE REAL BAD ONES
##    on  = numpy.delete(on,  [0,1,8,21,23,25,26,28,29], 2)
##    off = numpy.delete(off, [0,8,14,15,17,18,21,23,24,25,26,28,29,30,31], 2) #HARSH
##    on  = numpy.delete(on,  [0,8,14,15,17,18,21,23,24,25,26,28,29,30,31], 2)
    
    #Perform subtraction and masking.
    print "Subtraction and masking..."
##    deleteList = []
##    offSub = numpy.zeros([off.shape[0] - 1,off.shape[1],off.shape[2]])
    offSub = []
    for i in range(off.shape[0] - 1):
        #Screen out resets.
        if abs(numpy.median(off[i,:,:]) - numpy.median(off[i + 1,:,:])) < spikeCutoff:
            for j in range(off.shape[2]):
                offSub.append(off[i,0,j])
##            offSub[i,:,:] = (off[i,:,:] - off[i + 1,:,:])
##        else:
##            deleteList.append(i)
##    offSub = numpy.delete(offSub, deleteList, axis = 0)
##    deleteList = []
##    onSub = numpy.zeros([on.shape[0] - 1,on.shape[1],on.shape[2]])
    onSub = []
    for i in range(on.shape[0] - 1):
        #Screen out resets.
        if abs(numpy.median(on[i,:,:]) - numpy.median(on[i + 1,:,:])) < spikeCutoff:
            for j in range(on.shape[2]):
                if -(on[i,0,j]) > threshold:
                    onSub.append(on[i,0,j])
##            onSub[i,:,:] = (on[i,:,:] - on[i + 1,:,:])
##        else:
##            deleteList.append(i)
##    onSub = numpy.delete(onSub, deleteList, axis = 0)
    
##    offSub *= 1.58
##    onSub *= 1.58
    
##    offSub *= 2.89
##    onSub *= 2.89
    
    offstds = []
    onstds = []
    shots = []

##    offmeans = []
##    onmeans = []
    
    print "Averaging..."
    for a in avg:
##        offa = avgCube(offSub, a) * a / gain
##        ona = avgCube(onSub, a) * a / gain
##        offa = avgCube(offSub, a) * a
##        ona =  avgCube(onSub, a)  * a
        offa = avg(offSub) * a
        ona = avg(onSub) * a
        
##        print "off avg mean:", numpy.mean(offa)
##        print "on avg mean:", numpy.mean(ona)
        
##        plt.hist(ona.flatten(), bins = 1000, range = [numpy.mean(ona) - 3 * numpy.std(ona, ddof = 1),numpy.mean(ona) + 3 * numpy.std(ona, ddof = 1)])
##        plt.show()

##        if offa.shape[0] <= 20 or ona.shape <= 20:
##            print "Average", a, "removed, dataset not long enough"
##            avg = avg[:-1]
##        elif numpy.mean(ona) < 20:
##            print "Average", a, "removed, not enough light per read"
##            avg = avg[1:]
##        else:
        
##        offstds.append(numpy.std(offa, ddof = 1)) #OLD
##        onstds.append(numpy.std(ona, ddof = 1))
##        shots.append(numpy.sum(ona)**0.5)
        
        for i in range(len(offa)):
            offstds.append(numpy.std(offa))
##            for j in range(len(offa)):
##                offstds.append(numpy.std(offa[:,i,j], ddof = 1))
##                offmeans.append(numpy.mean(offa[:,i,j]))
        for i in range(len(ona)):
            onstds.append(numpy.std(ona))
##            for j in range(len(ona)):
##                onstds.append(numpy.std(ona[:,i,j], ddof = 1))
##                onmeans.append(numpy.mean(ona[:,i,j]))
##                shots.append(numpy.mean(ona[:,i,j]) ** 0.5)
##                print numpy.std(ona[:,i,j]), numpy.mean(ona[:,i,j])
##                print numpy.mean(ona[50,i,j]), numpy.mean(ona[51,i,j]), numpy.mean(ona[500,i,j])
        
##    print "Plotting..."
##    plt.loglog(avg, offstds, 'bo-')
##    print offstds
##    plt.loglog(avg, onstds, 'ro-')
##    print onstds
##    plt.loglog(avg, shots, 'mo-')
##    print shots
##    m1,b1 = linearfit(numpy.log(avg[:5]), numpy.log(onstds[:5]))
##    m2,b2 = linearfit(numpy.log(avg), numpy.log(shots))
##    print m1, m2, m1/m2
##    plt.loglog([avg[0],avg[-1]],[e**(b1 + m1 * numpy.log(avg[0])), e**(b1 + m1 * numpy.log(avg[-1]))],'k--')
##    plt.legend(["read noise", "photon noise", "calculated shot noise", "fit to photon noise"], loc = 2)
##    plt.ylabel("std. dev. ($e^{-}$ rms)")
##    plt.xlabel("$N_{avg}$")
##    plt.show()
##    return m1 / m2

##    for j in range(len(onstds)):
##        offstds[j] = round(offstds[j], 2)
##        onstds[j] = round(onstds[j], 2)
##        shots[j] = round(shots[j], 2)

##    print "offstds:", offstds
##    print "onstd:  ", onstds
##    print "shots:  ", shots
    print "mean offstds:", numpy.mean(offstds)
    print "mean onstds:", numpy.mean(onstds)
##    print "mean shots:", numpy.mean(shots)
##    print "mean offmeans:", numpy.mean(offmeans)
##    print "mean onmeans:", numpy.mean(onmeans)
##    print numpy.mean(onstds) - numpy.mean(offstds)
##    print numpy.mean(onmeans) - numpy.mean(offmeans)
##    return numpy.sum(onstds[:5]) / (numpy.sum(shots[:5]) + numpy.sum(offstds[:5])) #OLD
##    return numpy.mean(onstds) / numpy.mean(shots)
##    return numpy.mean(onstds) / numpy.std(shots, ddof = 1)
##    return (numpy.mean(onmeans) - numpy.mean(offmeans)) / numpy.mean(shots)
    return numpy.mean(onstds) / numpy.mean(offstds)

def measureFbest(offFile = "170130_100554.fits",onFile = "170130_100602.fits",gain = 65.6):

    print "offFile:", offFile
    print "onFile:", onFile
    print "gain:", gain
    
    spikeCutoff = 500 #10
    threshold = (20 * gain / 65.6)
    
    off = openfits(offFile)
    on = openfits(onFile)
##    off = numpy.delete(off, [29], 2) #JUST THE DEAD ONE
##    on = numpy.delete(on, [29], 2)
##    off = numpy.delete(off, [0,1,8,21,23,25,26,28,29], 2) #JUST THE REAL BAD ONES
##    on  = numpy.delete(on,  [0,1,8,21,23,25,26,28,29], 2)
    off = numpy.delete(off, [0,8,14,15,17,18,21,23,24,25,26,28,29,30,31], 2) #HARSH
    on  = numpy.delete(on,  [0,8,14,15,17,18,21,23,24,25,26,28,29,30,31], 2)
    
    #Perform subtraction and masking.
    print "Subtraction and masking..."
    onSub = []
    i = 0
    offset = 0
    onTemp = []
    while i < on.shape[0] - 100:
        i += 1
        if (i - offset) % 2000 < 100 or (i - offset) % 2000 > 1900:
            i += 200
            if len(onTemp) > 1800:
                onSub.append(numpy.sum(onTemp) * 1.58 / gain)
            onTemp = []
        elif abs(numpy.median(on[i,:,:]) - numpy.median(on[i + 1,:,:])) < spikeCutoff:
            for j in range(on.shape[2]):
                onTemp.append(on[i,0,j] - on[i + 1,0,j])
        elif i < 2000:
            offset = i
        else:
            print "ah crap", i
            plt.plot(on[:,0,0])
            plt.show()
    print onSub
    onLength = len(onSub)
    
    
    offSub = []
    i = 0
    offset = 0
    offTemp = []
    while i < off.shape[0] - 100:
        i += 1
        if (i - offset) % 2000 < 100 or (i - offset) % 2000 > 1900:
            i += 200
            if len(offTemp) > 1800:
                offSub.append(numpy.sum(offTemp) * 1.58 / gain)
            offTemp = []
        elif abs(numpy.median(off[i,:,:]) - numpy.median(off[i + 1,:,:])) < spikeCutoff:
            for j in range(off.shape[2]):
                offTemp.append(off[i,0,j] - off[i + 1,0,j])
        elif i < 2000:
            offset = i
        else:
            print "ah crap", i
    print offSub
    offLength = len(offSub)
    
    offstd = numpy.std(offSub)
    onstd = numpy.std(onSub)
    offmean = numpy.mean(offSub)
    onmean = numpy.mean(onSub)

    print "offstd:", offstd
    print "onstd:", onstd
    print "offmean:", offmean
    print "onmean:", onmean
    
    return (onstd**2 - offstd**2)**0.5 / (onmean - offmean)**0.5

def measureFbestest(offFile = "170130_100554.fits",onFile = "170130_100602.fits",gain = 65.6):

    print "offFile:", offFile
    print "onFile:", onFile
    print "gain:", gain
    
    spikeCutoff = 500 #10
    
    off = openfits(offFile)
    on = openfits(onFile)
##    off = numpy.delete(off, [29], 2) #JUST THE DEAD ONE
##    on = numpy.delete(on, [29], 2)
##    off = numpy.delete(off, [0,1,8,21,23,25,26,28,29], 2) #JUST THE REAL BAD ONES
##    on  = numpy.delete(on,  [0,1,8,21,23,25,26,28,29], 2)
    off = numpy.delete(off, [0,1,8,14,15,17,18,21,23,24,25,26,28,29,30,31], 2) #HARSH
    on  = numpy.delete(on,  [0,1,8,14,15,17,18,21,23,24,25,26,28,29,30,31], 2)
    
    #Perform subtraction and masking.
    print "Subtraction and masking..."
    onSub = []
    i = 0
    offset = 0
    onTemp = []
    while i < on.shape[0] - 100:
        i += 1
        if (i - offset) % 2000 < 100 or (i - offset) % 2000 > 1900:
            i += 200
            if len(onTemp) > 1800:
                for j in range(on.shape[2]):
                    onSub.append((onTemp[on.shape[2] * 1 + j] +
                                  onTemp[on.shape[2] * 2 + j] +
                                  onTemp[on.shape[2] * 3 + j] +
                                  onTemp[on.shape[2] * 4 + j] +
                                  onTemp[on.shape[2] * 5 + j] +
                                  onTemp[on.shape[2] * 6 + j] +
                                  onTemp[on.shape[2] * 7 + j] +
                                  onTemp[on.shape[2] * 8 + j] -
                                  onTemp[-on.shape[2] * 1 + j] -
                                  onTemp[-on.shape[2] * 2 + j] -
                                  onTemp[-on.shape[2] * 3 + j] -
                                  onTemp[-on.shape[2] * 4 + j] -
                                  onTemp[-on.shape[2] * 5 + j] -
                                  onTemp[-on.shape[2] * 6 + j] -
                                  onTemp[-on.shape[2] * 7 + j] -
                                  onTemp[-on.shape[2] * 8 + j]) / 8)
##                    print onTemp[on.shape[2] * 1 + j]
##                    print onTemp[-on.shape[2] * 8 + j]
##                onSub.append(numpy.sum(onTemp[on.shape[2] * 8:-on.shape[2] * 8]))
            onTemp = []
        elif abs(numpy.median(on[i,:,:]) - numpy.median(on[i + 1,:,:])) < spikeCutoff:
            for j in range(on.shape[2]):
                onTemp.append(on[i,0,j])
##                onTemp.append(on[i,0,j] - on[i + 1,0,j])
        elif i < 2000:
            offset = i
        else:
            print "ah crap", i
##            plt.plot(on[:,0,0])
##            plt.show()
##    print onSub
    onLength = len(onSub)
    
    offSub = []
    i = 0
    offset = 0
    offTemp = []
    while i < off.shape[0] - 100:
        i += 1
        if (i - offset) % 2000 < 100 or (i - offset) % 2000 > 1900:
            i += 200
            if len(offTemp) > 1800:
                for j in range(off.shape[2]):
                    offSub.append((offTemp[off.shape[2] * 1 + j] +
                                   offTemp[off.shape[2] * 2 + j] +
                                   offTemp[off.shape[2] * 3 + j] +
                                   offTemp[off.shape[2] * 4 + j] +
                                   offTemp[off.shape[2] * 5 + j] +
                                   offTemp[off.shape[2] * 6 + j] +
                                   offTemp[off.shape[2] * 7 + j] +
                                   offTemp[off.shape[2] * 8 + j] -
                                   offTemp[-off.shape[2] * 1 + j] -
                                   offTemp[-off.shape[2] * 2 + j] -
                                   offTemp[-off.shape[2] * 3 + j] -
                                   offTemp[-off.shape[2] * 4 + j] -
                                   offTemp[-off.shape[2] * 5 + j] -
                                   offTemp[-off.shape[2] * 6 + j] -
                                   offTemp[-off.shape[2] * 7 + j] -
                                   offTemp[-off.shape[2] * 8 + j]) / 8)
##                offSub.append(numpy.mean(offTemp[:off.shape[2] * 8]) - numpy.mean(offTemp[-8 * off.shape[2]:]))
##                offSub.append(numpy.sum(offTemp[off.shape[2] * 8:-off.shape[2] * 8]))
            offTemp = []
        elif abs(numpy.median(off[i,:,:]) - numpy.median(off[i + 1,:,:])) < spikeCutoff:
            for j in range(off.shape[2]):
                offTemp.append(off[i,0,j])
##                offTemp.append(off[i,0,j] - off[i + 1,0,j])
        elif i < 2000:
            offset = i
        else:
            print "ah crap", i
##    print offSub
    offLength = len(offSub)

##    print "off", offLength, "on", onLength
    print offSub
    print onSub

##    plt.hist(offSub, bins = 40, range = [-20,20], label = "LED off")
##    plt.legend()
##    plt.title("LED off plot, raw ADU, Gain:" + str(gain))
##    plt.show()
##    plt.hist(onSub, bins = 40, range = [numpy.mean(onSub) - 20 * gain, numpy.mean(onSub) + 20 * gain], label = "LED on")
##    plt.legend()
##    plt.title("LED on plot, raw ADU, Gain:" + str(gain))
##    plt.show()

##    print "offSub mean:", numpy.mean(offSub)
##    print "onSub mean:", numpy.mean(onSub) #0.319886 so times 3.126

##    offstd = numpy.std(numpy.array(offSub))# * 1.58 / gain) 
##    onstd = numpy.std(numpy.array(onSub))# * 1.58 / gain)
##    offmean = numpy.mean(offSub)
##    onmean = numpy.mean(onSub)
##
##    print "FOR DON <3 <3 <3"
##    print "offstd:", offstd
##    print "onstd:", onstd
##    print "offmean:", offmean
##    print "onmean:", onmean
    
    offstd = numpy.std(numpy.array(offSub)) * 1.58 / gain
    onstd = numpy.std(numpy.array(onSub)) * 1.58 / gain
    offmean = numpy.mean(offSub) * 1.58 / gain
    onmean = numpy.mean(onSub) * 1.58 / gain - offmean

    print "offstd:", offstd
    print "onstd:", onstd
    print "offmean:", offmean
    print "onmean:", onmean
    
    return ((onstd**2 - offstd**2)**0.5) / ((onmean - offmean)**0.5)
    

def Fspread():
    offFile = "170130_100551.fits"
    onFile = "170130_100602.fits"
    a = []
    on = openfits(onFile)
    for i in range(on.shape[0] - 1):
        a.append(numpy.median(on[i+1,:,:]) - numpy.median(on[i,:,:]))
    plt.plot(a)
    plt.show()

def excessNoiseFactorHypothetical():
    data = numpy.random.rand(100000,1,32)
    data = 2**data
    avg = [1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192]
    stds = []
    shots = []
    for a in avg:
        d = avgCube(data, a) * a
        stds.append(numpy.std(d, ddof = 1))
        shots.append(numpy.mean(d)**0.5)
    plt.loglog(avg, stds,'o-')
    plt.loglog(avg, shots,'o-')
    plt.show()

def manualF():
    bias = [ 1.0, 1.5, 2.0, 2.5, 3.5, 4.5, 5.5,  6.5,  7.5,  8.5,  9.5, 10.5, 11.5, 12.5]
    eADU = [3.80,4.59,5.36,6.11,7.35,8.53,9.65,10.12,10.56,10.87,10.79, 9.63, 6.22, 4.15]
    gain = [1.00,1.16,1.27,1.36,1.52,1.84,2.37, 3.24, 4.60, 6.76, 10.0, 15.0, 23.0, 35.3]
    cap  = [46.5,41.3,39.1,37.3,34.8,33.2,32.1, 31.4, 31.0, 30.7, 30.5, 30.5, 30.5, 30.5]

    bias = numpy.array(bias)
    eADU = numpy.array(eADU)
    gain = numpy.array(gain)
    cap  = numpy.array(cap)

    eADU /= gain
    eADU /= cap
    eADU /= eADU[0]

    plt.plot(bias, eADU,'o-')
    plt.ylabel("$V_{bias}$")
    plt.xlabel("F?")
    plt.show()

def DonsChargeGain(filename):
    d = openfits(filename)
    r = d[-24:-4,:,:] - d[4:24,:,:]

def avalancheGainFigure():
    bias  = [   1.5,  2.5,  3.5,  4.5,  5.5,  6.5,  7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5] 
    gain  =  [  1.03, 1.0, 1.23, 1.47, 1.85, 2.36, 4.54,6.91,11.7, 18.1, 28.4, 45.1, 72.7]

    plt.semilogy(bias, gain, 'bo-', label = "Mark 3 M02775-35")
    
    bias = [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5]
    gain = [1.0, 1.45, 1.53, 1.97, 2.79, 3.48, 4.33, 6.93, 9.88, 14.7, 27.7, 36.6, 65.6]

    plt.semilogy(bias, gain, 'go-', label = "Mark 13 M06665-23")

    gain = [1.0, 1.31, 1.43, 2.0, 2.84, 3.79, 5.36, 7.91, 11.4, 16.4, 25.6, 35.8, 56.4]

    plt.semilogy(bias, gain, 'ro-', label = "Mark 14 M06715-27")

    bias = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5]
    gain = [1.0, 1.16, 1.27, 1.36, 1.43, 1.52, 1.84, 2.37, 3.24, 4.6, 6.76, 10.0, 15.0, 23.0, 35.3, 53.4, 91.7]
    gain = numpy.array(gain)
    gain = gain / gain[3]

    plt.semilogy(bias, gain, 'co-', label = "Mark 19 M09225-11")
    plt.legend(loc = 2)
    plt.ylim([0.6,100])
##    plt.ylabel("Avalanche Gain")
    plt.ylabel("$e^{-}$")
    plt.xlabel("$V_{bias} (V)$")
    plt.title("Avalanche Gain vs. Bias Voltage")
    plt.show()

def darkHistogram():
##    filename = "170219_104530.fits"
##    title = "Histogram of Per-Pixel Dark Currents\nMark 14 M06715-27, $V_{bias} = 2.5\mathrm{V}$, $T = 62.5\mathrm{K}$"
##    filename = "170220_111658.fits"
##    title = "Histogram of Per-Pixel Dark Currents\nMark 14 M06715-27, $V_{bias} = 7.5\mathrm{V}$, $T = 60\mathrm{K}$"
    filename = "170220_025334.fits"
    title = "Histogram of Per-Pixel Dark Currents\nMark 14 M06715-27, $V_{bias} = 11.5\mathrm{V}$, $T = 62.5\mathrm{K}$"
##    filename = "170220_022119.fits"
##    title = "Histogram of Per-Pixel Dark Currents\nMark 14 M06715-27, $V_{bias} = 13.5\mathrm{V}$, $T = 60\mathrm{K}$"
    chargeGain = 2.89
    d = openfits(filename)[:,-32:,64:-64]
    cutoff = 20000
    n = 20
    length = 200
    if d.shape[0] == 360 or d.shape[0] == 300:
        timing = 5
    elif d.shape[0] > 590:
        timing = 0.2
    elif d.shape[0] == 120:
        timing = 30
    elif d.shape[0] == 24:
        timing = 300
        n = 10
        length = 20
    else:
        print "UHHH WHAT"
        print files[i]
        print d.shape[0]
        return
    if d.shape[1] == 256:
        d = d[:,224:,:]
    e = numpy.zeros([d.shape[1],d.shape[2]])
    k = 1
    while numpy.median(d[-k,:,:]) < cutoff and k < d.shape[0] - 1:
        k += 1
    if k >= d.shape[0] - 1:
        print "Bad reset?"
        k = 1
    end = d.shape[0] - k
    start = end - length
    if start < 0:
        start = 0
##            print "Warning, bad range."
    if end < 0:
        end += d.shape[0]
    for j in range(0,n):
        e += d[start + j,:,:]
        e -= d[end - j,:,:]
    e /= n
    e /= end - (start + n) #Modified 29 Feb 2016.
    e /= timing #Added 7 Apr 2015.
    e *= chargeGain
    print filename, round(numpy.median(e) * chargeGain,3), "e-/s", end - start, "Frames", "Timing:", timing, "s"
    median = numpy.median(e)
    std = numpy.std(e, ddof = 1)
    nstd = 5
    print "std:", std
    plt.hist(e.flatten(), bins = 200, range = [median - nstd*std, median + nstd*std], color = "w")
    plt.ylabel("$n$")
    plt.xlabel("$e^{-}\mathrm{s}^{-1}$")
    plt.title(title)
    plt.show()
    
def darkMedianPlot(filename):
    d = openfits(filename)
    d = d[:,-32:,64:-64]
    plt.plot(numpy.median(numpy.median(d, axis = 2), axis = 1))
    plt.show()

def darkMedianPlots(firstFile, lastFile, lineup = False):
    files = listdir("./")
    if not ((firstFile in files) and (lastFile in files)):
        print "I didn't find one of endcap files."
    startIndex = files.index(firstFile)
    endIndex = files.index(lastFile)
    for i in range(startIndex,endIndex + 1):
        if ".fits" in files[i]:
            d = openfits(files[i])
            if len(d.shape) == 3:
                d = d[:,-32:,64:-64]
                if lineup:
                    plt.plot(numpy.median(numpy.median(d, axis = 2), axis = 1) - numpy.median(numpy.median(d, axis = 2), axis = 1)[-1], label = files[i])
                else:
                    plt.plot(numpy.median(numpy.median(d, axis = 2), axis = 1), label = files[i])
    plt.legend()
    plt.ylabel("ADU (raw)")
    plt.xlabel("Frame #")
    plt.show()
        

def fullArrayNoise(filename):
    darkMedianPlot(filename)
    d = openfits(filename)
    print numpy.median(d[40:,:,:])
    stdarray = numpy.zeros([d.shape[1],d.shape[2]])
    for y in range(d.shape[1]):
        for x in range(d.shape[2]):
            stdarray[y,x] = numpy.std(d[40:,y,x], ddof = 1)
    savefits(filename[:-5] + "noise.fits", stdarray)

def voltageMedian(filename):
    d = openfits(filename)
    print numpy.median(d[40:,:,:])

def voltGainPlots21Apr2017():
##    medians = [[34802, 39420, 44014, 48710, 53568],
##               [33592, 38493, 43113, 47806, 52579],
##               [32625, 37163, 41967, 46798, 51515],
##               [40831, 45660, 50483, 54353, 58497],
##               [48493, 52735, 56832, 61627, 66311],
##               [47498, 51617, 55612, 60308, 65132, 70020, 74684]] #100K
    medians = [[9232, 13647, 18015, 22927, 27431],
               [8554, 12850, 17203, 21837, 26613],
               [8005, 11993, 16359, 20995, 25883],
               [15488, 20028, 24286, 28266, 32654],
               [23193, 27439, 31227, 36257, 40771],
               [21949, 26227, 30271, 35355, 40017, 44484, 49368]] #60K
    PRV = [[3.3050, 3.3550, 3.4056, 3.4554, 3.5052],
           [3.3050, 3.3550, 3.4056, 3.4554, 3.5052],
           [3.3050, 3.3550, 3.4056, 3.4554, 3.5052],
           [3.4056, 3.4554, 3.5052, 3.5477, 3.5902],
           [3.5052, 3.5477, 3.5902, 3.6400, 3.6898],
           [3.5052, 3.5477, 3.5902, 3.6400, 3.6898, 3.7397, 3.7894]]
    VDD = [3.5, 3.6, 3.7, 3.8, 3.9, 4.0]
    colors = ['b','g','r','c','m','k']

    for med,P,V,c in map(None, medians, PRV, VDD, colors):
        plt.plot(P, med, c + 'o-', label = "VDD = " + str(V) + "V")
        m, b = linearfit(P, med)
        plt.plot([P[0],P[-1]], [P[0] * m + b, P[-1] * m + b], c + '--', label = str(round(1.e6/m,2)) + " uV/ADU")
    plt.legend(loc = 2)
    plt.xlim([3.25,3.85])
    plt.xlabel("PRV (V)")
    plt.ylabel("median (ADU)")
##    plt.title("M06665-23 T=100K 21 Apr 2017 Voltage Gain Investigation")
    plt.title("M06665-23 T=60K 21 Apr 2017 Voltage Gain Investigation")
    plt.show()

def voltGainPlotsIanStyle21Apr2017():
##    medians = [[39420, 47806], #PRV = VDD - 0.15V
##               [34802, 43113, 51515, 58497, 66316, 74689], #PRV = VDD - 0.2V
##               [38493, 46798, 54353, 61632, 70025], #PRV = VDD - 0.25V
##               [33592, 41967, 50483, 56837, 65137], #PRV = VDD - 0.3V
##               [37163, 45660, 52740, 60313], #PRV = VDD - 0.35V
##               [32625, 40831, 48498, 55617]] #PRV = VDD - 0.4V    100K
    medians = [[13647, 21837], #PRV = VDD - 0.15V
               [9232, 17203, 25883, 32654, 40771, 49368], #PRV = VDD - 0.2V
               [12850, 20995, 28266, 36257, 44484], #PRV = VDD - 0.25V
               [8554, 16359, 24286, 31227, 40017], #PRV = VDD - 0.3V
               [11993, 20028, 27439, 35355], #PRV = VDD - 0.35V
               [8005, 15488, 23193, 30271]] #PRV = VDD - 0.4V    60K
    
    VDDi = [3.5, 3.5, 3.6, 3.6, 3.7, 3.7]
    voltDiffs = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    colors = ['b','g','r','c','m','k']

    for med, V, vD, c in map(None, medians, VDDi, voltDiffs, colors):
        VDD = []
        for i in range(len(med)):
            VDD.append(V + i * 0.1)
        plt.plot(VDD, med, c + 'o-', label = "PRV = VDD - " + str(vD) + "V")
        m, b = linearfit(VDD, med)
        plt.plot([VDD[0],VDD[-1]], [VDD[0] * m + b, VDD[-1] * m + b], c + '--', label = str(round(1.e6/m,2)) + " uV/ADU")
    plt.legend(loc = 2)
    plt.xlabel("VDD (V)")
    plt.ylabel("median (ADU)")
##    plt.title("M06665-23 T=100K 21 Apr 2017 Voltage Gain Investigation")
    plt.title("M06665-23 T=60K 21 Apr 2017 Voltage Gain Investigation")
    plt.show()

def darkvbias320x32():
    #From data 26-27 Apr 2017.
    biases = [2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
    darks = [0.0121, 0.0196, 0.0324, 0.0451, 0.0678, 0.0898]
    gains = [1.0, 1.45, 1.53, 1.97, 2.79, 3.48]
    darks = numpy.array(darks)
    gain = numpy.array(gains)
    plt.plot(biases, darks / gains, 'o-')
    plt.ylabel("$e^{-}\mathrm{sec}^{-1}\mathrm{pix}^{-1}$ (gain-corrected)")
    plt.xlabel("$V_{bias}$")
    plt.title("Dark vs. $V_{bias}$ for 320x32 Subarray M06665-23 26-27 Apr 2017")
    plt.show()

def darkRampMegaPlot():
    #Plot all measured darks over the last few days with M06665-23.
    darks = [0.0040, 0.0035, 0.0037, 0.0031, 0.0080, 0.0066, 0.0074, 0.0079, 0.0007, 0.0029,
             0.0049, 0.0060, 0.0045, 0.0049, 0.0051, 0.0078, 0.0034,
             0.0133, 0.0186,
             0.0205, 0.0145, 0.0310, 0.0177, 0.0167, 0.0133, 0.0273, 0.0123,
             0.3739, 0.1171, 0.0456, 0.0182, 0.0182, 0.0123, 0.0067, 0.0121, 0.0650, 0.0387, 0.0212, 0.0251, 0.0200, 0.0196, 0.0137, 0.0200,
             0.1177, 0.0587, 0.0546, 0.0391, 0.0268, 0.0396, 0.0244, 0.0324, 0.1717, 0.1000, 0.0684, 0.0602, 0.0666, 0.0457, 0.0451, 0.0373,
             0.2220, 0.1373, 0.1152, 0.0828, 0.0750, 0.0730, 0.0672, 0.0678, 0.3078, 0.2115, 0.1516, 0.1436, 0.1255, 0.1115, 0.0898, 0.0834]
    timings=[5, 5, 3, 3, 1, 1, 0.3, 0.3, 10, 30,
             30, 10, 5, 3, 1, 0.3, 5,
             5, 5,
             5, 5, 5, 5, 5, 5, 5, 5,
             5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
             5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
             5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    rows =  [32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
             32, 32, 32, 32, 32, 32, 32,
             128, 64,
             64, 64, 128, 128, 32, 32, 256, 256,
             32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
             32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
             32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32]
    timingList = [0.3, 1, 3, 5, 10, 30]
    rowsList = [32, 64, 128, 256]
    colors = ['b','g','r','c','m','k']
    markers = ['o', '^', 's', '*']
    firstFile = "170422_142515.fits"
    lastFile = "170427_082210.fits"
    files = listdir("./")
    if not ((firstFile in files) and (lastFile in files)):
        print "I didn't find one of endcap files."
    startIndex = files.index(firstFile)
    endIndex = files.index(lastFile)
    for i in range(startIndex,endIndex + 1):
        if ".fits" in files[i]:
            d = openfits(files[i])
            if len(d.shape) == 3:
                timeStr = files[i][4:-5]
                #Convert to decimal days.
                time = int(timeStr[:2]) + (int(timeStr[3:5]) * 1./24.) + (int(timeStr[5:7]) * 1./1440.) + (int(timeStr[7:]) * 1./86400.)
                dark = darks.pop(0)
                timing = timings.pop(0)
                row = rows.pop(0)
                plt.plot(time, dark, colors[timingList.index(timing)] + markers[rowsList.index(row)])
    plt.xlabel("Date/Time")
    plt.ylabel("$e^{-}\mathrm{sec}^{-1}\mathrm{pix}^{-1}$")
    plt.ylim([0, 0.4])
    plt.xlim([25,28])
    plt.show()

def darkRepeatTest():
    darks1= [0.0040, 0.0035, 0.0037, 0.0031, 0.0080, 0.0066, 0.0074, 0.0079, 0.0007, 0.0029,
             0.0049, 0.0060, 0.0045, 0.0049, 0.0051, 0.0078, 0.0034]
    darks2= [0.0058, 0.0053, 0.0051, 0.0042, 0.0079, 0.0107, 0.0075, 0.0077, 0.0000, 0.0029,
             0.0041, 0.0046, 0.0052, 0.0058, 0.0052, 0.0072, 0.0041]
    darks3= [0.0093, 0.0064, 0.0050, 0.0040, 0.0052, 0.0055, 0.0091, 0.0087, 0.0003, 0.0038,
             0.0046, 0.0042, 0.0049, 0.0041, 0.0051, 0.0056, 0.0033]
    plt.plot(darks1, 'o-', label = "60K run#1")
    plt.plot(darks2, 'o-', label = "60K run#2")
    plt.plot(darks3, 'o-', label = "40K")
##    plt.plot(numpy.array(darks3) - numpy.array(darks2))
    diff =  (numpy.array(darks1) + numpy.array(darks2)) / 2 - numpy.array(darks3)
    print numpy.mean(diff), numpy.std(diff, ddof = 1)
    plt.plot(diff, 'o-', label = "60K - 40K")
    plt.xlabel("Ramp #")
    plt.ylabel("$e^{-} \mathrm{sec}^{-1} \mathrm{pix}^{-1}$")
    plt.legend()
    plt.show()
                
def lowDarkPlot30Apr2017():
    #[0.0123, 0.0067, 0.0121,
    darks = [0.0079, 0.0107, 0.0075, 0.0077, 0.0080, 0.0066, 0.0074, 0.0079,
             0.0029, 0.0041, 0.0046, 0.0052, 0.0058, 0.0052, 0.0072, 0.0041, 0.0029, 0.0049, 0.0060, 0.0045, 0.0049, 0.0051, 0.0078, 0.0034,
             0.0058, 0.0053, 0.0051, 0.0042, 0.0040, 0.0035, 0.0037, 0.0031]

##        [30, 30, 30,             
    times = [60, 60, 60, 60, 60, 60, 60, 60,
             120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120,
             240, 240, 240, 240, 240, 240, 240, 240]
    print "1hr:", numpy.mean(darks[12:]), numpy.std(darks[12:], ddof = 1)
    plt.plot(times, darks, 'ro', label = "62.5K")
    darks = [0.0052, 0.0055, 0.0091, 0.0087,
             0.0038, 0.0046, 0.0042, 0.0049, 0.0041, 0.0051, 0.0056, 0.0033,
             0.0093, 0.0064, 0.0050, 0.0040]
    times = [60, 60, 60, 60,
             120, 120, 120, 120, 120, 120, 120, 120,
             240, 240, 240, 240]
    print numpy.mean(darks[4:]), numpy.std(darks[4:], ddof = 1)
    plt.plot(times, darks, 'bo', label = "42.3K")
    plt.ylabel("$e^{-} \mathrm{sec}^{-1} \mathrm{pix}^{-1}$")
    plt.xlabel("Ramp Length (min)")
    plt.legend()
    plt.xlim([10,280])
    plt.ylim([0,0.014])
    plt.title("Median Dark Current vs. Ramp Length for 320x32 Subarray\nMark 13 M06665-23, read interval 0.3-30s, $V_{bias} = 2.5\mathrm{V}$")
    plt.show()


def Figure2():
    d = openfits("161106_105124.fits")
    plt.plot(d[100:600,0,0])
    plt.xlabel("Read #")
    plt.ylabel("Raw (ADU)")
    plt.title("Single Pixel Readout for Mk13 M06665-2773")
    plt.show()

def modelAPD(debug = False):
    runs = 100000
##    runs = 3200000
    voltage = 14.5 #V
    maxDistance = 100
    cutoff = 3.5e-6
    avEnergy = (h * (c / cutoff)) / 1.6021e-19
    print avEnergy
    gains = []
    n = 0
    
    while n < runs:
        gains.append(startElectron(0, maxDistance, avEnergy, voltage))
        n += 1
    print numpy.mean(gains)
    if debug:
        plt.hist(gains, bins = 155)
        plt.xlim(-10, 300)
        plt.show()
    else:
        return gains


def startElectron(distance, maxDistance, cutoff, voltage):
    energy = 0
    electrons = 1
    while distance <= maxDistance:
        distance += 1
        energy += voltage / maxDistance
        if energy > cutoff:
            if random.random() > 0.957:
                electrons += startElectron(distance, maxDistance, cutoff, voltage)
                energy -= cutoff
    return electrons

def modelEfficiency():
    histrange = [-200,200]
    chargeGain = 1.58 #e-/ADU ME1000
    modelAPDdata = numpy.array(modelAPD())
    modelHist = histogram(modelAPDdata * chargeGain, bins = 200, range = histrange)
    binspacing = 1.0
    
    #Convert to e-
    binspacing *= chargeGain
    histrange[0] = int(chargeGain * histrange[0])
    histrange[1] = int(chargeGain * histrange[1])

    #Compute bins.
    bins = (histrange[1] - histrange[0]) / binspacing
    y,binedges = histogram(modelAPDdata, bins = 200, range=histrange)
    bincenters = 0.5*(binedges[1:]+binedges[:-1])
    
    for i in range(len(bincenters)):
##        print bincenters[i], modelHist[0][i]
        P = 0.
        FN = 0.
        for j in range(len(bincenters)):
            if j <= i:
                FN += modelHist[0][j]
            else:
                P += modelHist[0][j]
        print bincenters[i], P / (P + FN)


def compositeHistogram(histogramRange = [0,300], binspacing = 1.0, avg = 1,
                    referencePixels = False, cutoff = 0, start = 0,
                    avgSum = False, plotDifference = True, logPlot = True,
                    gainNormalizedPlot = False, electronsX = False, ymin = 0, ymax = 30000,
                    rawADU = False):
##    offFile = "150209_162545.fits"
##    onFile =  "150209_162612.fits"
##    offFile = "150413_113438.fits" #COMMON = -8V, mk10 M04935-17
##    onFile = "150413_113453.fits"
##    offFile = "150413_141134.fits" #COMMON = -8V, mk10 M04935-17, ref pixels
##    onFile = "150413_141147.fits"
##    offFile = "150413_162347.fits" #COMMON = -10V, mk10 M04935-17, ref pixels
##    onFile = "150413_162358.fits"
##    offFile = "150414_134745.fits" #COMMON = -11V, mk10 M04935-17, ref pixels
##    onFile = "150414_134803.fits"
##    offFile = "150414_144226.fits" #COMMON = -8V, mk10 M04935-17, ref pixels, 16ch
##    onFile = "150414_144243.fits"
##    offFile = "150422_091955.fits" #COMMON = -10V, mk10 M04935-17, ref pixels, 32ch, unmasked
##    onFile = "150422_092006.fits"
##    offFile = "151008_093127.fits" #COMMON = -8V, mk12 M06495-27, no ref, 32 x 1
##    onFile = "151008_093152.fits"

    
##    offFile = "151109_065456.fits" #COMMON = -11V, mk14 M06715-29, no ref, 32 x 1
##    onFile = "151109_070334.fits" #LED 0.8V
##    onFile = "151109_071428.fits" #LED 1.0V
##    onFile = "151109_072656.fits" #LED 1.1V
##    title = "BIAS = 14.5V, M06715-29, Data Taken 9 Nov 2015"
##    chargeGain = 2.89 #e-/ADU  ME911
##    gain = 45.6
    

    #MARK 14 CANON
    
##    offFile = "160109_155405.fits" #COMMON = -11V, mk14 M06665-12, no ref, 32 x 1
##    onFile = "160109_155441.fits" #LED1 = 1.2V
##    title = "BIAS = 14.5V, M06665-12, Data Taken 9 Jan 2016"
##    chargeGain = 2.89 #e-/ADU  ME911
##    gain = 77.5



##    offFile = "160415_LEDoffCOM-10V-16-0.fits" #COMMON = -10V, VDD = 5V, w/ PB, 64 x 1
##    onFile = "160415_LED0.9VCOM-10V-18-0.fits"
##
##    offFile = "CUBE_LED_OFF-3-0.fits" #COMMON = -15V, VDD = 5V, w/ PB, 32 x 1
##    onFile = "160422_LED3.2VCOM-15V-4-0.fits"
##    title = "BIAS = 19.5V, M06715-27, Data Taken 22 Apr 2016"
##    chargeGain = 2.89 #e-/ADU  ME911
##    gain = 400
    
##    offFile = "160413_64x1LEDoff-2-0.fits" #COMMON = -15V, VDD = 5V, w/ PB, 64 x 1
##    onFile = "160413_64x1LEDon-3-0.fits"
##    title = "BIAS = 19.5V, M06715-27, Data Taken 13 Apr 2016"
##    chargeGain = 2.89 #e-/ADU  ME911
##    gain = 400

##    offFile = "subarray_32x1_LEDOFF_4_28-1-0.fits" #COMMON = -8V, VDD = 5V, w/ PB, 32 x 1
##    onFile = "subarray_32x1_LED60mA_4_28-2-0.fits"

##    offFile = "160723_PCLEDoff2-29-0.fits" #COMMON = -15V, mk 13 M06665-25 w/ PB, no ref, 32 x 1 not RRR
##    onFile = "160723_PCLEDon-30-0.fits" #LED 3.1um 4.2V 100mA
##    title = "BIAS = 19.5V, M06665-25 w/ PB-32, Data Taken 24 Jul 2016"
##    chargeGain = 1.57 #e-/ADU  ME1000
##    gain = 400

##    offFile = "161012_134758.fits" #COMMON = -11V, mk13 M06665-25 ME1000, no ref, 32 x 1
##    onFile = "161012_134843.fits"
##    title = "$V_{bias} = 14.5\mathrm{V}$, M06665-25, 60K, Data Taken 12 Oct 2016"
##    chargeGain = 1.57 #e-/ADU ME1000
##    gain = 65.6

##    offFile = "161012_141312.fits" #COMMON = -11V, mk13 M06665-25 ME1000, no ref, 32 x 1, diff window
##    onFile = "161012_141410.fits"
##    title = "BIAS = 14.5V, M06665-25, Data Taken 12 Oct 2016"

##    offFile = "161016_140515.fits" #COMMON = -11V, mk13 M06665-23 ME1000, no ref, 32 x 1, diff window
##    onFile = "161016_140609.fits"
##    title = "BIAS = 14.5V, M06665-23, Data Taken 16 Oct 2016"
##    chargeGain = 1.57 #e-/ADU ME1000
##    gain = 65.6

##    offFile = "161106_105035.fits" #COMMON = -9V, mk13 M06665-23 ME1000, no ref, 32 x 1, diff window
##    onFile = "161106_105045.fits"
##    title = "BIAS = 12.5V, M06665-23, Data Taken 6 Nov 2016"

    #MARK 13 ME1000 CANON

##    offFile = "170308_123137.fits" #COMMON = -11V, mk13 M06665-23 ME1000, no ref, 32 x 1, diff window
##    onFile = "170308_123144.fits"
##    offFile = "170308_123154.fits" #COMMON = -11V, mk13 M06665-23 ME1000, no ref, 32 x 1, diff window
##    onFile = "170308_123205.fits"
##    offFile = "170308_123214.fits" #COMMON = -11V, mk13 M06665-23 ME1000, no ref, 32 x 1, diff window
##    onFile = "170308_123222.fits"
##    title = "$V_{bias} = 14.5\mathrm{V}$, M06665-23, 60K, Data Taken 8 Mar 2017"
##    chargeGain = 1.57 #e-/ADU ME1000
##    gain = 65.6

##    offFile = "170130_095713.fits"
##    onFile = "170130_095725.fits"
##    title = "who cares"
##    chargeGain = 1.57
##    gain = 65.6

    #MARK 13 ME1000 40K

    #Test to show noise difference
##    offFile = "151117_113112.fits"
##    onFile = "161012_141312.fits"
##    title = "Noise comparison between tests"

    #MARK 19 ME1001 60K
##    offFile = "170314_110753.fits" #COMMON = -11V, mk13 M06665-23 ME1000, no ref, 32 x 1, diff window
##    onFile = "170314_110802.fits"
##    title = "$V_{bias} = 14.5\mathrm{V}$, M09225-27, 60K, Data Taken 14 Mar 2017"
##    chargeGain = 5.18 #e-/ADU ME1001?
##    gain = 104.3


    detectors = ["M06665-23",
                 "M06715-27",
                 "M09225-11"]

    for detector in detectors:

##        if detector == "M02775-35":    
##            offFile = "150625_124409.fits" #COMMON = -4.5V, mk3 M02775-35, ref pixels, 32ch, unmasked
##            onFile = "150422_092006.fits"
##            title = "M02775-35"
##            chargeGain = 2.89
##            gain = 105.0

        if detector == "M06665-23":
            offFile = "161106_105115.fits" #COMMON = -11V, mk13 M06665-23 ME1000, no ref, 32 x 1, diff window
            onFile = "161106_105124.fits"
            title = "Mark 13 M06665-23"
            chargeGain = 1.57 #e-/ADU ME1000
            gain = 65.6

        elif detector == "M06715-27":
            offFile = "151117_113112.fits" #COMMON = -11V, mk14 M06715-27, no ref, 32 x 1
            onFile = "151117_113131.fits" #LED 2.812V
            title = "Mark 14 M06715-27"
            chargeGain = 2.89 #e-/ADU  ME911
            gain = 56.4

        elif detector == "M09225-11":
            offFile = "170324_162320.fits"
            onFile = "170324_162337.fits"
            title = "Mark 19 M09225-11"
            chargeGain = 4.71 #e-/ADU ME1001
            gain = 91.7
        
        off = openfits(offFile)
        on = openfits(onFile)
        histrange = numpy.array(histogramRange)

        if cutoff > 0:
            on = on[:cutoff,:,:32]
            off = off[:cutoff,:,:32]
        if start > 0:
            on = on[start:,:,:]
            off = off[start:,:,:]

        print "Off length:", off.shape[0]
        print "On length:", on.shape[0]
        
        #Check the lengths, trim from the end to match.
        if (off.shape[0] <> on.shape[0]):
            print "Length mismatch, trimming."
            if off.shape[0] > on.shape[0]:
                off = off[: - (off.shape[0] - on.shape[0]),:,:]
            else:
                on = on[: - (on.shape[0] - off.shape[0]),:,:]

        if detector == "M06665-23":
            trimLength = off.shape[0]
        elif detector == "M06715-27":
            print "Trimming to", trimLength, "frames"
            off = off[:trimLength, :, :]
            on = on[:trimLength, :, :]
            

        #Perform subtraction and masking.
        offSub = numpy.zeros([off.shape[0] - 1,off.shape[1],off.shape[2]])
        onSub = numpy.zeros([on.shape[0] - 1,on.shape[1],on.shape[2]])
        offCount = 0
        onCount = 0
        for i in range(offSub.shape[0]):
            offSub[i,:,:] = (off[i,:,:] - off[i + 1,:,:])
            onSub[i,:,:] = (on[i,:,:] - on[i + 1,:,:])

        #Do averaging.
        if avg > 1:
            offSub = avgCube(offSub, avg = avg)
            onSub = avgCube(onSub, avg = avg)
            if avgSum:
                offSub *= avg
                onSub *= avg

        if not(rawADU):
            offSub *= chargeGain
            onSub *= chargeGain
            binspacetemp = chargeGain * binspacing

        print "histrange:", histrange
        print "binspace:", binspacetemp

        #Compute bins.
        bins = int((histrange[1] - histrange[0]) / binspacetemp)
        print "bins:", bins

##        if electronsX:
##            offSub /= gain
##            onSub /= gain
##            histrange /= gain
            
        #Plot with difference line if requested.
        hist1 = numpy.histogram([offSub.flatten()], bins = bins, range = histrange)
        hist2 = numpy.histogram([onSub.flatten()], bins = bins, range = histrange)
        print hist1[1][3] - hist1[1][2], "binspacing", binspacetemp

        diff = hist2[0] - hist1[0]

        #Let's normalize these histograms.
##        diff /= numpy.max(diff) / 100

        if gainNormalizedPlot:
            for j in range(len(hist1[1])):
                hist1[1][j] = hist1[1][j] / gain
            for j in range(len(hist2[1])):
                hist2[1][j] = hist2[1][j] / gain
    ##            if model:
    ##                for j in range(len(histmodel[1])):
    ##                    histmodel[1][j] = histmodel[1][j] / gain
        
        if logPlot:
    ##        plt.semilogy(hist1[1][:-1],hist1[0], 'b-')
    ##        plt.semilogy(hist2[1][:-1],hist2[0], 'r-')
            plt.semilogy(hist1[1][:-1],diff, label = title)
        else:
    ##        plt.plot(hist1[1][:-1],hist1[0], 'b-')
    ##        plt.plot(hist2[1][:-1],hist2[0], 'r-')
            plt.plot(hist1[1][:-1],diff, label = title)

    plt.ylim([ymin,ymax]) #YLIM
        
    
    if gainNormalizedPlot:
        plt.xlabel("Gain (normalized)")
        plt.xlim([0,4])       
    else:
        if electronsX:
            plt.xlabel("$e^-$")
        else:
            if rawADU:
                plt.xlabel("ADU")
            else:
                plt.xlabel("Gain")
        plt.xlim([histrange[0],histrange[1]])
    plt.ylabel("number of reads")

    plt.title("Histogram of Pixel Response for Multiple Detectors\n" +
              "$V_{bias} = 14.5\mathrm{V}$, $T =$ 62.5K")

    plt.legend(loc = 3)
    plt.show()

       
chdir(defaultdir)
