"""
MIT License

Copyright (c) 2017 Sameem Zahoor Taray

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import  numpy as np
import  cv2
from    scipy.ndimage.interpolation import shift
from    scipy.cluster.vq import kmeans2

class FlowHolder:
    def __init__(self, flow_size):
        # Flow size is the maximum frames we take into consideration prior to
        # Current frame
        self.flow_size = flow_size
        self.flow_count = 0
        self.flow_list = []
        self.residual_list = []
        # Standard Deviation test threshold
        # For residue based
        self.std_thres = 21.0
        # For optical flow based
        self.std_thres_flows = 3.0
        self.likelyhood_threshold = 2.5
        # Occlusion threshold
        self.beta = 30
        # Initialize probable change time to frame 0.
        self.tc = 0
        # List containing means. Use mean of Optical flow for now.
        # This has to be changed to means of the Residue
        self.means = np.zeros(self.flow_size)
        # Initialize fstat to low negative values so that triggering
        # during initialization is avoided
        self.fstat = -1 * np.ones(self.flow_size)

    def getCount(self):
        return self.flow_count

    def pushFlow(self, flow):
        if self.flow_count < self.flow_size:
            self.flow_list.append(flow) 
            self.flow_count += 1
        else:
            self.flow_list.pop(0)
            self.flow_list.append(flow)

    def composeFlow(self):     
        self.composed_flow = np.sum(self.flow_list[0:self.tc +1], axis=0)
    

    def getComposedFlow(self):
        return self.flow_list[self.tc]
    def getComposedResidual(self):
        return self.composed_residual

    def processFlow(self, flow):
        m = flow.mean()
        if self.flow_count < self.flow_size:
            self.flow_list.append(flow)
            self.flow_count += 1
            
            self.means[self.flow_count - 1] = m
            # Update fstat
            self.fstat[self.flow_count -1] = (self.flow_count - self.tc + 1) * (self.means[0:self.tc + 1].mean() - self.means[self.tc:self.flow_count].mean())**2
            print(self.tc)
        else:
            self.flow_list.pop(0)
            self.flow_list.append(flow)
            #Shif values to the left
            self.means = shift(self.means, -1, cval=0.0)
            self.means[self.flow_count - 1] = m
            self.fstat = shift(self.fstat, -1, cval=0.0)
            self.fstat[self.flow_count - 1] = (self.flow_count - self.tc + 1) * (self.means[0:self.tc + 1].mean() - self.means[self.tc:self.flow_count].mean()) **2
            # Check if tc is updated
            # print(self.tc)
            if self.tc != self.fstat.argmax():
                # Do standard deviation test.
                std = flow.std()
                print("Standard Deviation = " + str(std))
                # print("Change time = " + str(self.tc))
                if self.std_thres_flows < std:
                    self.tc = self.fstat.argmax()
                    # print("Change time = " + str(self.tc))
                    # Compose the flows.
                    return True
        return False

    def processFlowResidue(self, im1_g, im2_g, flow):
        # Compute statistics based on the residues.
        # Setup the displacements in the two images
        h, w = im1_g.shape[0], im1_g.shape[1]
        x = np.arange(w)
        y = np.arange(h)

        xx, yy = np.meshgrid(x, y)
        cord_x = xx + flow[:,:,0]
        cord_y = yy + flow[:,:,1]
        # Truncate outliers
        cord_x[cord_x < 0] = 0
        cord_x[cord_x > (w-1)] = w-1
        cord_y[cord_y < 0] = 0
        cord_y[cord_y > (h-1)] = h-1
        # Calculate the warp of image1 using the flow
        warp = im1_g[cord_y.astype(np.int), cord_x.astype(np.int)].astype(np.int)
        # Calulate the residue between image1 and image2
        residual = np.subtract(warp, im2_g.astype(np.int))
        # For non positive values
        residual = np.square(residual)
        # residual = np.absolute(residual)
        # Clip values greater than the Occlusion threshold
        # residual[residual > self.beta] = 0

        m = residual.mean()

        if self.flow_count < self.flow_size:
            self.residual_list.append(residual)
            self.flow_list.append(flow)
            self.flow_count += 1
            self.means[self.flow_count - 1] = m
            # Update fstat
            self.fstat[self.flow_count -1] = (self.flow_count - self.tc + 1) * (self.means[0:self.tc + 1].mean() - self.means[self.tc:self.flow_count].mean())**2
            # print(self.tc)
        else:
            self.residual_list.pop(0)
            self.residual_list.append(residual)
            self.flow_list.pop(0)
            self.flow_list.append(flow)

            # Calculate statistics
            # Shif values to the left
            self.means = shift(self.means, -1, cval=0.0)
            self.means[self.flow_count - 1] = m
            self.fstat = shift(self.fstat, -1, cval=0.0)
            self.fstat[self.flow_count - 1] = (self.flow_count - self.tc + 1) * (self.means[0:self.tc + 1].mean() - self.means[self.tc:self.flow_count].mean()) **2
            # Check if tc is updated
            # print(self.tc)
            if self.tc != self.fstat.argmax():
                # Do standard deviation test.
                std = residual.std()
                print("Standard Deviation = " + str(std))
                # print("Change time = " + str(self.tc))
                if self.std_thres < std:
                    self.tc = self.fstat.argmax()
                    # print("Change time = " + str(self.tc))
                    # Compose the flows.
                    return True
        return False

    def processFlowResidueMixed(self, im1_g, im2_g, flow):
        # Compute statistics based on the residues.
        # Setup the displacements in the two images
        h, w = im1_g.shape[0], im1_g.shape[1]
        x = np.arange(w)
        y = np.arange(h)

        xx, yy = np.meshgrid(x, y)
        cord_x = xx + flow[:,:,0]
        cord_y = yy + flow[:,:,1]
        # Truncate outliers
        cord_x[cord_x < 0] = 0
        cord_x[cord_x > (w-1)] = w-1
        cord_y[cord_y < 0] = 0
        cord_y[cord_y > (h-1)] = h-1
        # Calculate the warp of image1 using the flow
        warp = im1_g[cord_y.astype(np.int), cord_x.astype(np.int)].astype(np.int)
        # Calulate the residue between image1 and image2
        residual = np.subtract(warp, im2_g.astype(np.int))
        # For non positive values
        residual = np.square(residual)
        # residual = np.absolute(residual)
        # Clip values greater than the Occlusion threshold
        residual[residual > self.beta] = 0
        m = flow.mean()

        if self.flow_count < self.flow_size:
            self.flow_list.append(flow)
            self.residual_list.append(residual)
            self.flow_count += 1
            
            self.means[self.flow_count - 1] = m
            # Update fstat
            self.fstat[self.flow_count -1] = (self.flow_count - self.tc + 1) * (self.means[0:self.tc + 1].mean() - self.means[self.tc:self.flow_count].mean())**2
            print(self.tc)
        else:
            self.flow_list.pop(0)
            self.flow_list.append(flow)
            self.residual_list.pop(0)
            self.residual_list.append(residual)
            #Shif values to the left
            self.means = shift(self.means, -1, cval=0.0)
            self.means[self.flow_count - 1] = m
            self.fstat = shift(self.fstat, -1, cval=0.0)
            self.fstat[self.flow_count - 1] = (self.flow_count - self.tc + 1) * (self.means[0:self.tc + 1].mean() - self.means[self.tc:self.flow_count].mean()) **2
            # Check if tc is updated
            # print(self.tc)
            if self.tc != self.fstat.argmax():
                # Do standard deviation test.
                std = flow.std()
                # print("Standard Deviation = " + str(std))
                # print("Change time = " + str(self.tc))
                if self.std_thres_flows < std:
                    self.tc = self.fstat.argmax()
                    # print("Change time = " + str(self.tc))
                    # Compose the flows.
                    return True
        return False

    def composeFlowResidue(self):
        self.composed_flow = np.sum( self.flow_list[ 0:self.tc +1], axis=0 )
        # Use the mean of Residual as composed Residual
        self.composed_residual = np.sum(self.residual_list[self.tc: self.flow_size], axis=0 )
        # self.composed_residual[self.composed_residual < self.beta] = 0
    
    def likelyhoodTest(self, mask):
        mask = mask.astype(np.bool)
        # Hypothesis 1
        fg = self.composed_residual[mask]
        bg = self.composed_residual[np.bitwise_not(mask)]
        std_1 = fg.std() + bg.std()
        # Hypothesis 0
        std_2 = self.composed_residual.std()
        likelyhood = (std_2 - std_1)
        print(likelyhood)
        if likelyhood >= -150:
            return True
        return False

    def likelyhoodTestFlows(self, mask):
        mask = mask.astype(np.bool)
        # Hypothesis 1
        fg_u = self.composed_flow[:,:,0][mask]
        bg_u = self.composed_flow[:,:,0][np.bitwise_not(mask)]
        fg_v = self.composed_flow[:,:,1][mask]
        bg_v = self.composed_flow[:,:,1][np.bitwise_not(mask)]
        std_1 = fg_u.std()  + fg_v.std() + bg_u.std() + bg_v.std()
        # Hypothesis 0
        std_2 = self.composed_flow.std()

        likelyhood = (std_2 - std_1)
        print(likelyhood)
        if likelyhood >= 7:
            return True

        return False

    def refineSegmentation(self, img):
        data_matrix = np.vstack([300*self.composed_flow[:,:,0].flatten(), 300*self.composed_flow[:,:,1].flatten()\
        ,0.1*img[:,:,0].flatten(), 0.1*img[:,:,1].flatten(), 0.1*img[:,:,2].flatten()])
        res, idx = kmeans2(data_matrix.T, 2)
        return idx.astype(np.float32)

    def getTc(self):
        return self.tc

    def getResidual(self):
        return self.residual_list[self.tc].astype(np.uint8)
    

    def translateMask(self, mask, flow_index):
        h, w = mask.shape[0], mask.shape[1]
        x = np.arange(w)
        y = np.arange(h)

        xx, yy = np.meshgrid(x, y)
        cord_x = xx + self.flow_list[flow_index][:,:,0]
        cord_y = yy + self.flow_list[flow_index][:,:,1]
        # Truncate outliers
        cord_x[cord_x < 0] = 0
        cord_x[cord_x > (w-1)] = w-1
        cord_y[cord_y < 0] = 0
        cord_y[cord_y > (h-1)] = h-1

        translated_mask = mask[cord_y.astype(np.int), cord_x.astype(np.int)]
        return translated_mask
    
    def likelyhoodTestPropogated(self, segmentation_mask):
        translated_mask = segmentation_mask
        H0, H1 = 0.0, 0.0
        for i in range(self.tc, self.flow_size):
            translated_mask = self.translateMask(translated_mask, i).astype(np.bool)
            #  Hypothesis 1
            fg = self.residual_list[i][translated_mask]
            bg = self.residual_list[i][np.bitwise_not(translated_mask)]
            std_1 = fg.std() + bg.std()
            H1 += std_1
            # Hypothesis 0
            H0 += self.residual_list[i].std()
            likelyhood = (H0 - H1)
        
        

        if likelyhood == np.nan:
            return False
        elif likelyhood > -1 * i * 3:
            # print("Likelyhood ratio =\t" + str(likelyhood))
            return True
        return False

