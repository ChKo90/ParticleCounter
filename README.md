Installation:
- install Python 3.7 or newer
- download or clone this repository
- install project dependencies with the command line "pip install -r requirements.txt"

Usage:
- start main.py with Python
- select a workspace directory containing the czi files for evaluation
- Click in the group box "Preprocessing" at the start button to merge the Z-stacks and to compress the files to the subdirectory "preprocessed". This step is neccessary once, after opening a workspace the first time.
- configure the image settings for each channel, click at "show merged" to switch to the merged image view
- To detect particles in the image: remove the image background with the color settings in the "processed" tab and configure the detection in the "Particle Counter" tab. Click the start button in the "Detection" group box to test your setup with the selected image.
- in "Parameter Sets" you can save the whole image processing configuration.
- Click at "Export results" to export the result images of the current selected image or the whole workspace. Select the "Processed" tab to export the processed images (separate channels and/or merged). For exporting the particle detection, select the "Particle Counter" tab.
