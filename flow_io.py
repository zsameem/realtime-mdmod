"""
    Some helper functions to read, write and visualize optical flow
    Author: Samim Zahoor Taray
"""

def readFlow(name):
    import numpy as np
    if name.endswith('.pfm') or name.endswith('.PFM'):
        return readPFM(name)[0][:,:,0:2]

    f = open(name, 'rb')

    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')

    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()
    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))
    return flow.astype(np.float32)

def writeFlow(name, flow):
    import numpy
    f = open(name, 'wb')
    f.write('PIEH'.encode('utf-8'))
    np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
    flow = flow.astype(np.float32)
    flow.tofile(f)
    f.flush()
    f.close() 
# Create a 3 channel visualization of the optical flow based on the magnitude
# and Direction of the flow
def flow_to_color(optflow):
    import numpy as np
    import cv2
    optflow = np.absolute(optflow)
    # optflow[optflow < .75 ] = 0
    h = optflow.shape[0]
    w = optflow.shape[1]
    rho = np.sqrt(optflow[:,:,0]**2 + optflow[:,:,1]**2)
    rho /= rho.max()
    phi = np.arctan2(optflow[:,:,0], optflow[:,:,1]) 
    
    # phi = np.add(phi, np.pi/2)
    phi_degrees = 180.0/np.pi * phi
    
    # print(phi_degrees.min())
    hsvimg = np.stack( (phi_degrees, rho, np.ones((h,w))), axis = 2)
    rgbimg = cv2.cvtColor(hsvimg.astype(np.float32), cv2.COLOR_HSV2RGB)
    return rgbimg

def segment_flow(rgbimg):
    import numpy as np
    import cv2
    h = rgbimg.shape[0]
    w = rgbimg.shape[1]
    grayimg = cv2.cvtColor(rgbimg, cv2.COLOR_RGB2GRAY)
    out = np.zeros(grayimg.shape)
    out = cv2.normalize(grayimg, out, 1.0, 0.0, cv2.NORM_MINMAX)
    out = out * 255
    out = out.astype(np.uint8)
    
    # th3 = cv2.adaptiveThreshold(out,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #         cv2.THRESH_BINARY,11,2)
    
    ret,thr = cv2.threshold(out,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel  = np.ones((7,7), np.uint8)
    closing = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel) 
    closing = cv2.bitwise_not(closing)
    # h = np.histogram(closing, [0,1,2])
    # freq = h[0]
    # if freq[0] < freq[1]:
    #     closing = np.subtract(1,closing)
    # cnt = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    # cnt_mask = np.zeros(closing.shape, np.uint8)
    # cv2.drawContours(cnt_mask, cnt, -1, 255, -1)
    
    # Copy the thresholded image.
    # im_floodfill = closing.copy()
 
    # # Mask used to flood filling.
    # # Notice the size needs to be 2 pixels than the image.
    # mask = np.ones((h+2, w+2), np.uint8)
    # h_, w_ = h+2, w+2
    # mask[1:h_ - 1, 1: w_ - 1] = closing
    # # Floodfill from point (x, y) where x is the column and y is the row number
    # cv2.floodFill(im_floodfill, mask, (0,0), 0)
    # mask[1:h_ - 1, 1: w_ - 1] = im_floodfill

    # cv2.floodFill(im_floodfill, mask, (0, 479), 0)
    # mask[1:h_ - 1, 1: w_ - 1] = im_floodfill
    # cv2.floodFill(im_floodfill, mask, (639, 479), 0)
    # mask[1:h_ - 1, 1: w_ - 1] = im_floodfill
    # cv2.floodFill(im_floodfill, mask, (639, 0), 0)
    # mask[1:h_ - 1, 1: w_ - 1] = im_floodfill
    # # Invert floodfilled image
    # im_floodfill_inv = cv2.bitwise_not(im_floodfill)
 
    # # Combine the two images to get the foreground.
    # im_out = closing | im_floodfill_inv
    # opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel) 
    # print(out.mean(), out.std(), out.dtype)
    # print(th3.mean(), th3.std(), th3.dtype)
    # grayimg[grayimg < 200] = 0
    # grayimg[grayimg >=200] = 255
    return closing 
