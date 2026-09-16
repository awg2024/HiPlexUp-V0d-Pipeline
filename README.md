### V0d Histological Analysis Workflow 

The aim of this repository is to identify and quantify V0d interneurons from multiplex histological images

1. Python conversion from czi into a via read_czi.py into a 16-bit multichannel TIFF for analysis and an 8-bit composite preview image. 

2. FIJI image quality control, each converted image is checked for staining quality, tissue capture (both hemicords present?), imaging artifacts. Images failing a quality-control criteria will be flagged and remove before analysis. 
   
3. FIJI imaging analysis identifying; gray matter ROI, cell segmentation and measurements LABELLING : V0d cells, boardline cells and non-V0d cells for 20% of samples. This should be kept consistent across samples. In addition to these ROIs, tissue/background fluorescence is also measured so that a corrected marker intensities can be calculated (difference from cell intensity against background intensity)
   
4. Export CSV / XLSX in which we have one row per cell containing area; cell ID, X-Y coordinates, EVX1 corrected intensity, VGAT corrected intensity, PAX2 corrected intensity, DBX1 corrected intensity. We will need to configure Fiji to save and export this data.  

5. For each cell classification we can plot the distribution of each of these collected measurements. Plotting, for instance DBX1 intensity, we may discover that V0d cells lie in a range of 0-10 whereas non-V0d lie in a range of 10-20. Therefore we can place a boundary of 10 of V0d cell classification. Equivalent distributions are examined for EVX1, VGAT, PAX2, DBX1, cell area, morphology, normalised XY positions (this xy normalisation is incl in the HiPlex pipeline). 

6. Validation of parameters. The proposed thresholds are then tested against labelled cells that were not labelled. Here we can see under classification does our thresholds correctly identify the parameters?

6. Utilise pre-existing R HiPlexUp, modify for V0d, and utilise our newly defined thresholds of intensity and size. Most likely, the V0d end-definition will depend on a combination of marker intensities, morphologies and spatial positions. R pipeline then runs; 
- background-corrected fluorescence analysis;
- cell-size filtering;
- coordinate normalization;
- section rotation;
- left/right hemicord normalization;
- spatial normalization between sections;
- V0d classification using the experimentally derived thresholds.

8. Final V0d counts and intensity for WT and SOD1 mice. 
