import numpy as np
import math


class GaussianKernelCalculator:
    """A simple class to compute a zero-centered gaussian kernels in 1D and 2D.
    Inspired by http://dev.theomader.com/gaussian-kernel-calculator/
    """

    def __init__(self, sigma, kernelsize):

        if kernelsize <= 0 or kernelsize % 2 == 0 or kernelsize > 50 or sigma <= 0:
            print "Please use an positive odd kernelsize smaller then 50 and a positive sigma"
            return

        self.kernelSize = kernelsize
        self.sigma = sigma
        self.mu = 0.0
        self.sampleCount = 1000

        self.kernel1d = np.zeros(kernelsize)
        self.kernel2d = np.zeros((kernelsize, kernelsize))

        self.update_kernel()

    def gaussian_distribution(self, x):
        d = x - self.mu
        n = 1.0 / (math.sqrt(2 * math.pi) * self.sigma)
        return math.exp(-d*d/(2 * self.sigma * self.sigma)) * n

    def integrateSimphson(self, samples):
        result = samples[0][1] + samples[samples.shape[0]-1][1];

        for s in range(1, samples.shape[0]):
            sampleWeight = 4.0
            if (s % 2) == 0:
             sampleWeight = 2.0

            result += sampleWeight * samples[s][1]

        h = (samples[samples.shape[0]-1][0] - samples[0][0]) / (samples.shape[0]-1)
        return result * h / 3.0;

    def sample_interval(self, minInclusive, maxInclusive, sampleCount):
        result = np.zeros((sampleCount, 2))
        x = np.linspace(minInclusive, maxInclusive, sampleCount, endpoint=True)

        count = 0
        for i in x:
            y = self.gaussian_distribution(i)
            result[count] = [i, y]
            count += 1

        return result

    def round_to(self,num, decimals):
        shift = math.pow(10, decimals)
        return round(num * shift) / shift


    def update_kernel(self):
        samplesPerBin = math.ceil(self.sampleCount / self.kernelSize)
        # need an even number of intervals for simpson integration => odd number of samples
        if(samplesPerBin % 2 == 0):
            ++samplesPerBin

        weightSum = 0
        kernelLeft = -math.floor(self.kernelSize/2)

        # get samples left and right of kernel support first
        outsideSamplesLeft = self.sample_interval(-5 * self.sigma, kernelLeft - 0.5, samplesPerBin)
        outsideSamplesRight = self.sample_interval(-kernelLeft+0.5, 5 * self.sigma, samplesPerBin)

        #print outsideSamplesLeft
        #print outsideSamplesRight
        allSamples = [[outsideSamplesLeft, 0]]

        # now sample kernel taps and calculate tap weights
        for tap in range(0, self.kernelSize):

            left = kernelLeft - 0.5 + tap

            tapSamples = self.sample_interval(left, left+1, samplesPerBin)
            tapWeight = self.integrateSimphson(tapSamples)

            allSamples.append([tapSamples, tapWeight])
            weightSum += tapWeight


        allSamples.append([outsideSamplesRight, 0])

        # renormalize kernel and round to 6 decimals
        for i in range(0,len(allSamples)):
            allSamples[i][1] = self.round_to(allSamples[i][1] / weightSum, 6)

        for i in range(1, len(allSamples)-1):
            d1 = self.round_to(allSamples[i][1], 6)
            self.kernel1d[i-1] = d1

        for i in range(1, len(allSamples)-1):
            for j in range(1, len(allSamples)-1):
                d2 = self.round_to(allSamples[i][1] * allSamples[j][1], 6)

                self.kernel2d[i-1][j-1] = d2

        print np.sum(self.kernel1d)
        #print self.kernel2d


#g = GaussianKernelCalculator(1.0, 5)
