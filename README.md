# Image Rectifier.
This is an image rectification python module. It uses 2D homograph the warp the image coordinates.
###
Libraries Used
- Numpy (homography/bilinear interpolation)
- OpenCV(Image Display/Save, Coordinate Tracking)
- Tkinter (File Opener Dialogue Box)

# How to Use

1. Download this package, and open '''demo.py''' with Python IDLE. Press 'run module.'
2. Running the module will activate the file selection pop-up dialogue.
Choose a image that you want to rectify.
3. Choose 4 planar coordinates that you want to warp. (Pre-warp coordinates are in blue.)
4. Then, choose 4 destination coordinates to which the pre-warp coordinates will be warped.
   (Destination coordinates are marked with teal texts.)
5. Once rectification is done, you will find ```rectified.jpg``` in the folder.
6. Done!

## Sample Images
