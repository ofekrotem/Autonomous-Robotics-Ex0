# Autonomous Robotics - GNSS Raw Measurements

## Overview
This project is part of the Autonomous Robotics course and focuses on implementing a naive positioning algorithm using GNSS (Global Navigation Satellite System) raw measurements. The primary objective is to process raw GNSS data, compute positions in ECEF coordinates, and convert these to latitude, longitude, and altitude.

## Features
1. **Data Parsing**: Convert raw GNSS log files to a CSV format.
2. **Position Calculation**: Implement an iterative numerical minimal RMS algorithm on a weighted set of SatPRNs to compute positions.
3. **Coordinate Conversion**: Convert ECEF coordinates to latitude, longitude, and altitude.
4. **Output Generation**: Produce a KML file for visualization and a CSV file with computed positions.

## Requirements
- Python 3.8^
- pandas
- numpy
- navpy
- simplekml

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ofekrotem/Autonomous-Robotics-Ex0.git
   cd Autonomous-Robotics-Ex0
2. Install the required Python packages:
   ```bash
   pip install pandas numpy navpy simplekml

## Usage
1. **Prepare Data**:
   - Place your raw GNSS log file in the `data` directory.
   - Update the `input_filepath` in `Parser.py` with the path to your raw GNSS log file.

2. **Run the Script**:
   ```bash
   python Parser.py
3. **Outputs**:
   - `results/output_xyz.csv`: CSV file containing satellite positions.
   - `coordinates.kml`: KML file for visualization of the computed path.
   - `lla_coordinates.csv`: CSV file with additional columns for computed positions in latitude, longitude, and altitude.

## Notes
- The positioning algorithm is based on an iterative numerical minimal RMS algorithm on a weighted set of SatPRNs.
- The script includes provisions for handling missing measurement values.

## How to Run
1. Ensure that the raw GNSS log file is placed in the `data` directory and the `input_filepath` in `Parser.py` is updated accordingly.
2. Execute the script:
   ```bash
   python Parser.py
3. Check the results directory for the output_xyz.csv and the project root for the coordinates.kml and lla_coordinates.csv files.
