#Measure the dark current from a recorded file.
import numpy
import pyfits as fits

def openfits(filename):
    #Simply wraps the fits reading stuff.
    hdulist = fits.open(filename,ignore_missing_end=True)
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
    hdu = fits.PrimaryHDU(data)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(filename)
    hdulist.close()
    return

def main():
    print "I'm the function that measures dark current from a file."
    print "The median dark current will be reported and a file will be saved with"
    print "the measured dark current for the given window."
    print "What file should I use? It should be in this directory. (I'll add the .fits to the end.)"
    filename = raw_input()
    filename += ".fits"
    
    try:
        data = openfits(str(filename))
        print "File loaded."
    except:
        print "Loading this file isn't working. Is the name right? I need the full name.fits."
        return
    
    timing = raw_input("What timing was this data taken with? ")
    if timing == 0:
        print "You can't take dark data with no timing!"
        return
    
    chargeGain = raw_input("What charge gain should I use (e-/ADU)? ME-911 is 2.1, ME-1000 is 1.6, and ME-1001 is 4.7. (Enter nothing for ME-1000 default.)")
    if not chargeGain:
        chargeGain = 1.6
        print "Using 1.6 e-/ADU, default for ME-1000."
    
    print "I need the window to measure dark current in, and check for non-linearity."
    x0 = raw_input("Please input the first value of the window, x0 (or put nothing for default:")
    x1 = None
    y0 = None
    y1 = None
    if not x0:
        print "Using default window, strip at top covered by mask [[175,195],[235,245]]."
        x0 = 175
        x1 = 195
        y0 = 235
        y1 = 245
    else:
        x1 = raw_input("Please input the second value of the window, x1:")
        y0 = raw_input("Please input the third value of the window, y0:")
        y1 = raw_input("Please input the fourth value of the window, y1:")
    if y0 > y1 or x0 > x1:
        #Correct values if put in the wrong order.
        temp = y1
        y1 = y0
        y0 = temp
        temp = x1
        x1 = x0
        x0 = temp
    window = [[x0,x1],[y0,y1]]
##    window = [[175,195],[235,245]]
##    window = [[155, 215],[205,275]]
##    window = [[70,110],[128,144]]
##    window = [[32,-32],[32,-32]]
##    window = [[0,48],[32,-32]] #bottom strip
##    window = [[64,-64],[96, -96]] #mask hole
#    window = [[-28, 256],[32, -32]] #mask covered
##    window = [[0, 32],[0, 32]] #specular glow
    
    cutoff = raw_input("Please input the cutoff at which to assume non-linearity and stop measuring (0 for no check): ")
#    cutoff = 20000
    windowedData = data[:,window[0][0]:window[0][1],window[1][0]:window[1][1]]
    print windowedData.shape
    stop = 0
    for i in range(windowedData.shape[0]):
        if numpy.median(windowedData[i,:,:]) > float(cutoff):
            stop = i
    print "stop:", stop
    print "Skipping first five frames and using next",stop - 5,"frames for measurement as per cutoff value."
    print numpy.median(windowedData[5,:,:]) - numpy.median(windowedData[stop,:,:])
    cts = (numpy.median(windowedData[5,:,:]) - numpy.median(windowedData[stop,:,:])) * chargeGain / float(timing)
    print "Measured median dark current in window is ", cts / (stop - 5), "e-/ADU."
    darkFrame = numpy.array(windowedData[5,:,:] - windowedData[stop,:,:]) * chargeGain / float(timing)
    darkFilename = filename[:-5] + "dark.fits"
    try:
        savefits(darkFilename, darkFrame)
    except:
        print "Tried to save dark file, and it didn't work! Is old dark file still present?"
        print "If you still want it, rename it, and then run this program again."
    print "Full dark frame saved as", darkFilename
    print "Note that non-linearity may be present outside of specified window, which makes measurement bad outside window!"
    print "(If you want to fix that, run this again with the window on the interested area.)"