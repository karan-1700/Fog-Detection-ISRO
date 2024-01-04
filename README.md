# Fog-Detection-ISRO
Detection of Day Time Fog over India using INSAT-3DR Satellite Data

# Background
* Fog is a hazardous weather phenomenon that appears when water vapour near the surface is condensed to form suspended water droplets.
* The north-Indian region experiences dense fog during winter from November to February every year, reducing the horizontal visibility below 1 km range.
* Fog can impose serious danger to navigation for aviation and transportation sectors, which in turn can affect the economy and endanger lives.
* Therefore, **it is crucial to monitor the spatial and temporal extend of fog in order to take important decision which, in turn, can prevent occurrence of accidents.**
* The operational algorithm currently used for detecting fog sometimes lead to false detection or fails to detect fog patch.

# Objective
* In this project, **we propose an algorithm for automatic detection of day-time fog** over the Indian region using INSAT-3DR IMAGER data.
* Later on, using satellite data, we plan to predict the time when the fog patch would disappear completely.

# Abstract
The occurrence of fog is associated with several negative impacts as far as the life of people, their health and the socio-economic aspects are concerned. Therefore, it is crucial to observe and study the characteristics and physical behavior of fog to understand its complete life cycle. This information can assist us to better predict its occurrence and extent both spatially and temporally so as to mitigate some of the hazards associated with fog. In this report, a novel and simple remote sensing technique has been discussed which can be applied for any geostationary satellite data having at least one visible and one thermal infrared channel for detecting day time fog. The visible channel data of INSAT-3DR satellite is used to detect fog, while its thermal infrared channel observation is used to eliminate the medium- and high-level clouds and snow area by deriving various thresholds dynamically for each time instant. Spatial homogeneity, which is a unique characteristic property of fog, is also incorporated to detect fog efficiently. The fog maps generated using the proposed algorithm are validated qualitatively with the output of operational product from 0315UTC to 0945UTC for the whole month of January 2021. The algorithm is capable of detecting daytime fog without using the 3.9 μm channel that is contaminated by solar radiation during day time. Added to this, the proposed technique is developed to detect fog for each time of acquisition instead of depending on the previous day’s data unlike the operational product algorithm, thus making this algorithm suitable for operational use.

# Flowchart
![Project Flowchart](https://github.com/karan-1700/Fog-Detection-ISRO/blob/main/assets/images/18DCS055_Project_Report_Flowchart.png)

# Acknowledgement
* I agree to the Terms and Conditions of the MOSDAC for use and sharing of INSAT-3DR data.
* I would like to express my sincere gratitude to ISRO and, specifically, MOSDAC for providing access to the INSAT-3DR IMAGER database.
* I will use the INSAT-3DR database for research purposes only, not for any commercial use.

# References
1. Fundamentals of Remote Sensing [link](https://www.nrcan.gc.ca/maps-tools-and-publications/satellite-imagery-and-air-photos/tutorial-fundamentals-remote-sensing/9309).
2. “Detection of Day Time Fog Over India Using INSAT-3D Data” - Chaurasia, S., & Gohil, B. S. (2015) [link](https://ieeexplore.ieee.org/document/7328684).
3. “An objective method for detecting night time fog using MODIS data over northern India” - S. Chaurasia and B. S. Gohil (2016) [link](https://www.semanticscholar.org/paper/An-objective-method-for-detecting-night-time-fog-Chaurasia-Gohil/42e21f21c37209b879eeaf9248c8355f0b535dc8).
4. “Detection of Fog Using Temporally Consistent Algorithm With INSAT-3D Imager Data Over India” - Chaurasia, S., & Jenamani, R. K. (2017) [link](https://ieeexplore.ieee.org/document/8096990).
5. “Nighttime fog detection using MODIS data over Northern India” - S. Chaurasia et al. (2010) [link](https://www.researchgate.net/publication/227670489_Night_time_fog_detection_using_MODIS_data_over_Northern_India).
6. [Database link] INSAT-3DR ASIA MERCATOR, 2016. [Online]. Available: www.mosdac.gov.in
