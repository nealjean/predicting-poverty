# Combining satellite imagery and machine learning to predict poverty

The data and code in this repository allows users to generate figures appearing in the main text of the paper ***Combining satellite imagery and machine learning to predict poverty*** (except for Figure 2, which is constructed from specific satellite images). Paper figures may differ aesthetically due to post-processing.

Code was written in R 3.2.4 and Python 2.7.

Users of these data should cite Jean, Burke, et al. (2016). If you find an error or have a question, please submit an issue.

## Links to related projects

We are no longer maintaining this project, but will link to related projects as we learn of them.

Pytorch implementation: https://github.com/jmather625/predicting-poverty-replication

## Description of folders

- **data**: Input and output data stored here
- **figures**: Notebooks used to generate Figs. 3-5
- **scripts**: Scripts used to process data and produce Fig. 1
- **model**: Store parameters for trained convolutional neural network

## Packages required

**R**
- R.utils
- magrittr
- foreign
- raster
- readstata13
- plyr
- RColorBrewer
- sp
- lattice
- ggplot2
- grid
- gridExtra

The user can run the following command to automatically install the R packages
```
install.packages(c('R.utils', 'magrittr', 'foreign', 'raster', 'readstata13', 'plyr', 'RColorBrewer', 'sp', 'lattice', 'ggplot2', 'grid', 'gridExtra'), dependencies = T)
```
**Python**
- NumPy
- Pandas
- SciPy
- scikit-learn
- Seaborn
- Geospatial Data Abstraction Library (GDAL)
- Caffe

Caffe and pycaffe
- See [Caffe Installation](https://github.com/BVLC/caffe/wiki/Installation)

We recommend using the open data science platform [Anaconda](https://www.continuum.io/downloads).

## Instructions for processing survey data

Due to data access agreements, users need to independently download data files from the World Bank's Living Standards Measurement Surveys and the Demographic and Health Surveys websites. These two data sources require the user to fill in a Data User Agreement form. In the case of the DHS data, the user is also required to register for an account.

For all data processing scripts, the user needs to set the working directory to the repository root folder.

1. Download LSMS data
	1. Visit the [host website for the World Bank's LSMS-ISA data](http://econ.worldbank.org/WBSITE/EXTERNAL/EXTDEC/EXTRESEARCH/EXTLSMS/0,,contentMDK:23512006~pagePK:64168445~piPK:64168309~theSitePK:3358997,00.html):
	2. Download into **data/input/LSMS** the files corresponding to the following country-years:
 		1. Uganda 2011-2012
		2. Tanzania 2012-13
		3. Nigeria 2012-13
		4. Malawi 2013
		
**UPDATE (08/02/2017)**: The LSMS website has apparently recently removed two files from their database which contain crucial consumption aggregates for Uganda 2011-12 and Malawi 2013. Since we are not at liberty to share those files ourselves, this would inhibit replication of consumption analysis in those countries. We have reached out and will update this page according to their response.

**UPDATE (08/03/2017)**: The LSMS has informed us these files were inadvertently removed and will be restored unchanged as soon as possible.
		
	3. Unzip these files so that **data/input/LSMS** contains the following folders of data:
		1. UGA_2011_UNPS_v01_M_STATA
		2. TZA_2012_LSMS_v01_M_STATA_English_labels
		3. DATA (formerly NGA_2012_LSMS_v03_M_STATA before a re-upload in January 2016)
		4. MWI_2013_IHPS_v01_M_STATA
2. Download DHS data
	1. Visit the [host website for the Demographic and Health Surveys data](http://dhsprogram.com/data/dataset_admin/download-datasets.cfm)
	2. Download survey data into **data/input/DHS**. The relevant data are from the Standard DHS surveys corresponding to the following country-years:
		1. Uganda 2011
		2. Tanzania 2010
		3. Rwanda 2010
		4. Nigeria 2013
		5. Malawi 2010
	3. For each survey, the user should download its corresponding Household Recode files in Stata format as well as its corresponding geographic datasets
	4. Unzip these files so that **data/input/DHS** contains the following folders of data:
		1. UG_2011_DHS_01202016_171_86173
		2. TZ_2010_DHS_01202016_173_86173
		3. RW_2010_DHS_01312016_205_86173
		4. NG_2013_DHS_01202016_1716_86173
		5. MW_2010_DHS_01202016_1713_86173
		
		(Note that the names of these folders may vary slightly depending on the date the data is downloaded)
3. Run the following files in the script folder
	1. DownloadPublicData.R
	2. ProcessSurveyData.R
	3. save_survey_data.py

## Instructions for extracting satellite image features

1. Download the parameters of the trained CNN model [here](https://www.dropbox.com/s/4cmfgay9gm2fyj6/predicting_poverty_trained.caffemodel?dl=0) and save in the **model** directory.

2. Generate candidate locations to download using `get_image_download_locations.py`. This will generate locations meant to download 1x1 km RGB satellite images of size 400x400 pixels. For most of the countries, locations for about 100 images in a 10x10 km area around the cluster is generated. For Nigeria and Tanzania, we generate about 25 evenly spaced points in the 10x10 km area. The result of running this is a file for each country, for each dataset named `candidate_download_locs.txt`, in the following format for every line:
    ```
    [image_lat] [image_long] [cluster_lat] [cluster_long]
    ```
    For example, a line in this file may be 
    ```
    4.163456 6.083456 4.123456 6.123456
    ```
    Note that this requires GDAL and previously running `DownloadPublicData.R`.

3. Download imagery from locations of interest (e.g., cluster locations from Nigeria DHS survey). In this process, successfully downloaded images must then have a corresponding line in a output metadata file named `downloaded_locs.txt` (e.g., **data/output/DHS/nigeria/downloaded_locs.txt**). There will be one of these metadata files for each country. The format for each line of the metadata file must be, for each line:
    ```
    [absolute path to image] [image_lat] [image_long] [cluster_lat] [cluster_long]
    ```
    For example, a line in this file may be
    ```
    /abs/path/to/img.jpg 4.163456 6.083456 4.123456 6.123456
    ```
    Note that the last 4 fields in each line should be copied from the `candidate_download_locs.txt` file for each (country, dataset) pair. 

4. Extract cluster features from satellite images using `extract_features.py`. This will require installation of Caffe and pycaffe (see [Caffe Installation](https://github.com/BVLC/caffe/wiki/Installation)). This may also require setting pycaffe in your PYTHONPATH. In each country's data folder (e.g., **data/output/DHS/nigeria/**) we save two Numpy arrays: `conv_features.npy` and `image_counts.npy`. This process will be much faster if a sizable GPU is used, with `GPU=True` set in `extract_features.py`. 

## Instructions for producing figures

For all data processing scripts, the user needs to set the working directory to the repository root folder. If reproducing all figures, the user does *not* need to rerun the data processing scripts or the image feature extraction process (steps 1-2 for Fig. 1, steps 1-6 for Figs. 3-5).

To generate Figure 1, the user needs to run

1. DownloadPublicData.R
2. ProcessSurveyData.R
3. Fig1.R

To generate Figure 3, the user needs to run

1. DownloadPublicData.R
2. ProcessSurveyData.R
3. save_survey_data.py
4. get_image_download_locations.py
5. (download images)
6. extract_features.py
7. [Figure 3.ipynb](https://github.com/nealjean/predicting-poverty/blob/master/figures/Figure%203.ipynb)

To generate Figure 4, the user needs to run

1. DownloadPublicData.R
2. ProcessSurveyData.R
3. save_survey_data.py
4. get_image_download_locations.py
5. (download images)
6. extract_features.py
7. [Figure 4.ipynb](https://github.com/nealjean/predicting-poverty/blob/master/figures/Figure%204.ipynb)

To generate Figure 5, the user needs to run

1. DownloadPublicData.R
2. ProcessSurveyData.R
3. save_survey_data.py
4. get_image_download_locations.py
5. (download images)
6. extract_features.py
7. [Figure 5.ipynb](https://github.com/nealjean/predicting-poverty/blob/master/figures/Figure%205.ipynb)
