## GNSS Data Transmission Setup

This project involves an Android application and a Python server for collecting and processing live raw GNSS data to calculate the precise location of the Android device.

## Overview

The system consists of three main components:

1. **Android Application**: Collects raw GNSS measurements and navigation messages from the device's GNSS receiver and sends this data to a Python server. 
2. **Python Server**: Receives the GNSS data, parses it, and calculates the precise location of the Android device using the provided GNSS measurements and navigation messages.
3. **GNSS Data Viewer**: A Python-based graphical interface using Pygame to visualize the latest GNSS data and calculated positions.

[Link to Android Repo](https://github.com/ofekrotem/Raw-GNSS-data-app) 


[Link to Python Server with parsing abilities and UI](https://github.com/ofekrotem/Live-Location-From-Gnss-data)
## Android Application
Screenshot from app:


![Screenshot from app](https://github.com/ofekrotem/Live-Location-From-Gnss-data/assets/92383710/a7fbad5b-8a79-472f-b47f-9058bfce9990)
### Features

- Collects raw GNSS measurements including pseudorange rates, satellite IDs, signal-to-noise ratios, and other relevant data.
- Transmits GNSS data to the Python server in real-time.


### Key Components

- **GNSS Measurements Collection**: Utilizes the Android GNSS API to gather raw measurements.
- **Data Transmission**: Sends the collected data to the Python server using a socket connection.


### Permissions

Ensure the following permissions are included in your `AndroidManifest.xml`:

- `ACCESS_FINE_LOCATION`
- `INTERNET`


### Data Format

GNSS data is sent as JSON objects with the following structure:

```json
{
  "pseudorangeRateMetersPerSecond": -210.8851486112045,
  "codeType": "C",
  "timeNanos": 275468000000,
  "biasNanos": 0.0,
  "constellationType": 1,
  "svid": 2,
  "accumulatedDeltaRangeState": 16,
  "receivedSvTimeNanos": 566476931246836,
  "pseudorangeRateUncertaintyMetersPerSecond": 1.0064293127089008,
  "accumulatedDeltaRangeMeters": -59388.61673446453,
  "accumulatedDeltaRangeUncertaintyMeters": 3.4028234663852886e+38,
  "carrierFrequencyHz": 1575420030.0,
  "receivedSvTimeUncertaintyNanos": 56,
  "cn0DbHz": 25.99774169921875,
  "fullBiasNanos": -1404307001532431398,
  "multipathIndicator": 0,
  "timeOffsetNanos": 0.0,
  "state": 16431
}
```

## Python Server

### Features

- Receives GNSS data from the Android application.
- Parses the incoming data and extracts relevant GNSS measurements.
- Calculates the precise location of the Android device using GNSS algorithms.


### Key Components

- **Data Reception**: Listens for incoming data from the Android application on a specified port.
- **Data Parsing**: Converts the received JSON data into a format suitable for GNSS calculations.
- **Location Calculation**: Implements algorithms to process the GNSS measurements and compute the device's location.


### Data Reception

Ensure the server is set to listen on the correct port and can handle multiple incoming connections if necessary. The server should validate the received data format and handle any potential errors or inconsistencies.


### Position Calculation Algorithms

The position calculation from GNSS data involves several key algorithms and processes:

1. **Pseudorange Calculation**:
   - Pseudorange is the distance between a satellite and the GNSS receiver, calculated using the time difference between the satellite's signal transmission and reception.
   - Adjustments are made for clock biases and signal propagation delays.

2. **Satellite Positioning**:
   - The precise positions of the satellites at the time of signal transmission are determined using the satellite ephemeris data.

3. **Trilateration**:
   - Using the pseudoranges and satellite positions, trilateration is performed to estimate the receiver's position.
   - This involves solving a set of nonlinear equations to find the intersection point of spheres centered at each satellite, with radii equal to the respective pseudoranges.

4. **Error Correction**:
   - Corrections are applied for various error sources, including atmospheric delays (ionospheric and tropospheric), multipath effects, and receiver clock biases.

5. **Least Squares Estimation**:
   - The final position is refined using a least squares estimation method to minimize the residual errors between the measured and predicted pseudoranges.


## GNSS Data Viewer
Screenshot from UI:

![Screenshot from parser ui](https://github.com/ofekrotem/Live-Location-From-Gnss-data/assets/92383710/b040f1e8-a376-4ba4-85de-5e3e985cbb4d)


### Features

- A graphical user interface built using Pygame to display the latest GNSS data and calculated positions.
- Fetches and displays data from the Python server in real-time.

### Key Components

- **Data Fetching**: Periodically requests the latest GNSS data from the server.
- **Display**: Renders the fetched data on the screen, including raw measurements and calculated positions.
- **Scrolling**: Supports vertical scrolling to view all data.

### Usage

1. Ensure the Python server is running and accessible.
2. Run the `parserUI.py` script to start the GNSS Data Viewer.
3. Use the mouse wheel to scroll through the displayed data.

### Example Data Display

The viewer displays the latest received GNSS measurements and the last calculated position, with the ability to scroll through the data if it exceeds the screen space.

### Error Handling

The viewer logs any errors encountered while fetching data, ensuring continuous operation even if data retrieval fails temporarily.


## Setup and Execution

### Prerequisites

- Android Studio for developing and deploying the Android application.
- Python 3.x for running the server-side scripts.
- Pygame library for the GNSS Data Viewer.

### Steps to Run the Project

1. **Android Application**:
   - Open the project in Android Studio.
   - Ensure all necessary permissions are granted.
   - Deploy the application on an Android device with GNSS capabilities.
   - Run the application to start collecting and sending GNSS data.

2. **Python Server**:
   - Set up a Python environment.
   - Ensure all necessary dependencies are installed (e.g., `socket` library for networking).
   - Run the server script to start listening for incoming GNSS data.
   - Monitor the server logs to verify data reception and processing.

3. **GNSS Data Viewer**:
   - Install the Pygame library: `pip install pygame`.
   - Run the `parserUI.py` script to start the GNSS Data Viewer.
   - View the latest GNSS measurements and calculated positions in the graphical interface.


### Logs

The server logs will display incoming GNSS measurements and the calculated location details. Here is an example log entry for received GNSS data:
```
Received GNSS measurements: [{'pseudorangeRateMetersPerSecond': -210.8851486112045, 'codeType': 'C', 'timeNanos': 275468000000, 'biasNanos': 0.0, 'constellationType': 1, 'svid': 2, 'accumulatedDeltaRangeState': 16, 'receivedSvTimeNanos': 566476931246836, 'pseudorangeRateUncertaintyMetersPerSecond': 1.0064293127089008, 'accumulatedDeltaRangeMeters': -59388.61673446453, 'accumulatedDeltaRangeUncertaintyMeters': 3.4028234663852886e+38, 'carrierFrequencyHz': 1575420030.0, 'receivedSvTimeUncertaintyNanos': 56, 'cn0DbHz': 25.99774169921875, 'fullBiasNanos': -1404307001532431398, 'multipathIndicator': 0, 'timeOffsetNanos': 0.0, 'state': 16431}]
           GPS time         Sat.X         Sat.Y         Sat.Z   pseudorange        cn0
satPRN                                                                                
G02     4876.931247  2.125484e+07  1.137998e+07  1.195455e+07  2.061986e+07  25.997742
G03     4876.929462  1.555347e+07 -3.338325e+05  2.134521e+07  2.140995e+07  33.724106
G04     4876.928287  2.603152e+07  1.502838e+06  5.444235e+06  2.174763e+07  27.396336
G08     4876.917000  2.070509e+07  7.645504e+06 -1.518714e+07  2.509086e+07  28.536171
G09     4876.917635  2.464024e+07 -7.975202e+06 -6.100307e+06  2.490059e+07  33.698151
G17     4876.919983  1.502237e+07 -1.327340e+07  1.792532e+07  2.432266e+07  24.078085
G19     4892.916715  5.241196e+06 -1.472178e+07  2.124746e+07  2.524938e+07  12.000000
G21     4876.929812  2.099442e+07  1.583235e+07  7.001847e+06  2.120426e+07  31.969954
G26     4876.915212 -2.895147e+06  2.529690e+07 -7.004190e+06  2.558558e+07  12.000000
G28     4876.924575 -1.023467e+06  1.916451e+07  1.835110e+07  2.264547e+07  31.847944
G31     4876.929283  8.071393e+06  2.304402e+07  9.743900e+06  2.126182e+07  35.402336
G32     4876.915267 -1.214113e+07  1.469668e+07  1.862456e+07  2.534785e+07  12.000000
Calculated position: (32.10117827996274, 35.20480623590327)
```



## Troubleshooting

- Ensure the Android device has a clear view of the sky for optimal GNSS signal reception.
- Verify network connectivity between the Android device and the Python server.
- Check for any permission-related issues on the Android device.
- Validate the data format and handle any parsing errors in the server script.
- Ensure the Pygame library is properly installed for the GNSS Data Viewer.


## Future Enhancements

- Implement advanced GNSS algorithms for improved location accuracy.
- Add support for additional GNSS constellations (e.g., GLONASS, Galileo).
- Enhance the Android application with a user interface to display real-time location data.
- Improve the GNSS Data Viewer with additional visualization options and user controls.
