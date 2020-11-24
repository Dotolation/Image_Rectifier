import numpy as np
import cv2

debug = False

def blah(str):
    if debug:
        print(str)
        

def build_A(pts1, pts2):
    #row and column counts
    rc, cc = pts1.shape
    
    p1, p2 = (pts1.tolist(), pts2.tolist()) # for speed
    #Append() of list is faster than np.concatenate
    A = []
    
    '''For each pts1-pts2 pair, following rows are pushed to A.:

       [x y 1 0 0 0 -(x'x) -(x'y) -x']
       [0 0 0 x y 1 -(y'x) -(y'x) -y']

       where (x, y) is a coordinate in pts1 and 
       (x', y') is a coordinate in pts2. 
    '''
    for i in range(0, rc):   
        p2x, p2y = (p2[i][0], p2[i][1])
        zeros = [0,0,0]
        #Cartesian -> Euclidian
        xy1 = p1[i] + [1]
        
        xi= xy1 + zeros + [e * -(p2x) for e in xy1]
        yi= zeros + xy1 + [e * -(p2y) for e in xy1]
        
        A.append(xi)
        A.append(yi)   
        
    blah("A sum")
    blah(np.array(A).sum())
    
    return np.array(A) #into np.array

def compute_H(pts1, pts2):
    # Construct the intermediate A matrix.
    A = build_A(pts1, pts2)
    
    # Compute the symmetric matrix AtA.
    AtA = np.matmul(np.transpose(A), A)
    blah("AtA sum")
    blah(AtA.sum())


    # Compute the eigenvalues and eigenvectors of AtA.
    e_val, e_vec = np.linalg.eig(AtA)
    e_val = np.absolute(e_val)
    blah("e_Val")
    blah(e_val)
    blah("e_Vec")
    blah(e_vec)
    mindex = np.argmin(e_val)
    return_v = e_vec[:,mindex] #weird indexing system
    
    # Return the eigenvector corresponding to the smallest eigenvalue, 
    # reshaped as a 3x3 matrix.
    blah(return_v.reshape(3,3))
    return return_v.reshape(3,3)


def bilinear_interp(image, points):
    y_border, x_border, c_channel = image.shape 

    if (c_channel == 1): #Greyscale
        blah("Greyscale.")
    else:
        blah("Vibrant c o l o r s !")

    y_size, x_size = points.shape[0:2]
    poINTs = points.astype(int)   
    Fxy = np.zeros((y_size, x_size, c_channel))

    #weight arrays (for a and b)
    a, b = np.split(points - poINTs, 2,  axis=2)
    a_in, b_in = (np.ones(a.shape) - a, np.ones(b.shape) - b)  # 1-a, 1-b.

    flattened = poINTs.reshape(-1, 2)
    flattened_x = flattened[:, 0]
    flattened_y = flattened[:, 1]
    blah("shape of flattened(Pre-Mask)")
    blah(flattened_x.shape)
    blah(flattened_y.shape)

    mask = (flattened_x >= 0) * (flattened_y >= 0) * (flattened_x < x_border - 1) * (flattened_y < y_border - 1)
    blah("mask")
    blah(mask)
    blah(mask.shape)

    y_coords, x_coords = np.meshgrid(np.arange(y_size), np.arange(x_size))
    y_coords = y_coords.T.reshape(-1)[mask]
    x_coords = x_coords.T.reshape(-1)[mask]
    blah("shape of x/y coordinates")
    blah(x_coords.shape)
    blah(y_coords.shape)

    flattened_y = flattened_y[mask]
    flattened_x = flattened_x[mask] 
    blah("shape/elements of flattened")
    blah(flattened_x.shape)
    blah(flattened_y.shape)
    blah(flattened_x)
    blah(flattened_y)

    '''Formula for the bilinear interpolation:
               f[y][x] * (1-a)(1-b) + f[y][x+1] * a(1-b)
               + f[y+1][x] * (1-a)b + f[y+1][x+1] * ab '''
    Fxy[y_coords, x_coords] = (
        + image[flattened_y, flattened_x] * (a_in * b_in)[y_coords, x_coords]
        + image[flattened_y, flattened_x+1] * (a * b_in)[y_coords, x_coords]
        + image[flattened_y+1, flattened_x] * (a_in * b)[y_coords, x_coords]
        + image[flattened_y+1, flattened_x+1] * (a * b)[y_coords, x_coords]
    )

    return np.rint(Fxy).astype(int)



def warp_homography(source, target_shape, Hinv):

    x_min, y_min, x_max, y_max = target_shape 
    x_size, y_size = (x_max - x_min, y_max - y_min)

    blah("New Image Size:")
    blah(x_size)
    blah(y_size)

    ones = np.ones((y_size, x_size,1)) #1s in [x, y, 1]
    #ys and xs in [x, y, 1], in a separate matrix.
    yy, xx = np.mgrid[y_min:y_max, x_min:x_max].reshape(2,y_size,x_size,1)
    grid = np.concatenate((xx,yy,ones),axis=2)

    warped = grid @ np.transpose(Hinv)
    
    coord, ones = (warped[:,:,0:2],warped[:,:,2].reshape(y_size, x_size, 1))
    coord = coord / ones


    """Pixel value retrieval & bilinear interpolation."""
    painted = bilinear_interp(source, coord)
    
    blah("Shape of the picture")
    blah(painted.shape)
    
    return painted
    

def rectify(image, planar_points, target_points):
    # Compute the rectifying homography that warps the planar points to the
    # target rectangular region.
    h = compute_H(planar_points, target_points)
    h_inv = compute_H(target_points, planar_points)  
      
    # Apply the rectifying homography to the bounding box of the planar image
    # to find its corresponding bounding box in the rectified space.
    y_max, x_max, unused = image.shape
    h_00 = h @[0,0,1]
    h_x0 = h @ [x_max, 0,1]
    h_0y = h @ [0, y_max,1]
    h_xy = h @ [x_max, y_max,1]
    h_minmax = np.concatenate(([h_00], [h_x0], [h_0y], [h_xy]))
    # Euclidian -> Cartesian
    bb = h_minmax[:,0:2] / (h_minmax[:,2].reshape(4,1))  
    
    bb_min = np.array([np.min(bb[:,0]), np.min(bb[:,1])])
    bb_max = np.array([np.max(bb[:,0]), np.max(bb[:,1])])
    bb_min = np.rint(bb_min).astype(int)
    bb_max = np.rint(bb_max).astype(int)
    shape = (bb_min[0], bb_min[1], bb_max[0], bb_max[1])

    blah("new bb_min:")
    blah(shape[0])
    blah(shape[1])
    blah("new bb_max:")
    blah(shape[2])
    blah(shape[3])    
    return warp_homography(image, shape, h_inv)

    # Offset the rectified bounding box such that its minimum point (the top
    # left corner) lies at the origin of the rectified space.

    # Compute the inverse homography to warp between the offset, rectified
    # bounding box and the bounding box of the input image.

    # Perform inverse warping and return the result.



