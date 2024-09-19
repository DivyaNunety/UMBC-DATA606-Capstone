# Project Title: Driver’s Fault Prediction 

**Prepared for**: UMBC Data Science Master’s Degree Capstone by Dr. Chaojie (Jay) Wang  
**Author**: Divya Sri Nunety  
**Semester**: Fall '24  

**GitHub repo of the project**: [UMBC-DATA606-Capstone](https://github.com/DivyaNunety/UMBC-DATA606-Capstone.git)  
**LinkedIn profile**: [Divya Sri Nunety](https://www.linkedin.com/in/divyasrinunety/)

## Background

### What is it About?
This project aims to predict the severity of road traffic accidents and identify the drivers at fault using machine learning techniques. The dataset obtained from the Automated Crash Reporting System (ACRS) of the Maryland State Police contains detailed records of traffic collisions on county and local roadways within Montgomery County. It includes information about each collision such as the type of vehicles involved, crash conditions, injury severity, and driver details. The project focuses on leveraging data-driven methods to understand the key factors contributing to accident severity and to build predictive models that can classify future accident outcomes and determine fault based on the available data.

### Why Does It Matter?
Here’s the paragraph broken down into key points:

- **Accurate predictions**:
  - Help emergency services allocate resources more effectively.
  - Ensure faster and more appropriate responses based on accident severity.

- **Understanding accident factors**:
  - Can guide **policy changes**.
  - Inform **infrastructure improvements**.
  - Support **targeted safety campaigns** to enhance road safety.

- **Identifying fault**:
  - Streamlines the **insurance claims process**.
  - Ensures **fair compensation**.
  - Reduces **fraud**.

- **Data-driven insights**:
  - Enable **government agencies** and **law enforcement** to make informed decisions for **better traffic management**.
  - Assist in **resource planning**.

- **Recognizing patterns** in driver faults and accident severity:
  - Helps implement **targeted interventions** to reduce **human error**.
  - Contributes to **safer roads** for everyone.
## Research Questions
1. How does driver behavior (e.g., speeding, distraction, alcohol use) influence the likelihood of being at fault in a traffic collision?
2. Can real-time data (e.g., weather, traffic flow) improve the accuracy of predictive models for accident severity and fault determination?
3. What combinations of vehicle-related factors (e.g., vehicle type, age, safety features) are most strongly associated with high-severity accidents?

## Data 

### Data Sources:
The dataset used in this project is sourced from the Automated Crash Reporting System (ACRS) ([Crash Reporting - Drivers Data](https://catalog.data.gov/dataset/crash-reporting-drivers-data)) maintained by the Maryland State Police, with collision data reported by Montgomery County Police, Gaithersburg Police, Rockville Police, and Maryland-National Capital Park Police. It includes detailed records of traffic collisions on county and local roadways.

### Data Size:
      82 MB

### Data Shape:
- **Rows**: 157,841
- **Columns**: 43

### Time Period:
    2006-2024

### Each Row Represent
Each row represents a motor vehicle operator (driver) involved in a traffic collision on a county or local roadway in Montgomery County. Each entry includes driver, vehicle, and accident details as well as severity and fault data.

## Data Dictionary

| Column Name              | Data Type | Definition                                                  | Potential Values                          |
|--------------------------|-----------|--------------------------------------------------------------|-------------------------------------------|
| Crash_ID                  | String    | Unique identifier for each reported crash.                    | Alphanumeric string (e.g., "ACRS202301")  |
| Crash_Date                | Date      | Date of the collision.                                        | Date format (e.g., "2023-01-15")          |
| Driver_ID                 | String    | Unique identifier for each driver involved in the crash.       | Alphanumeric string                       |
| Driver_Age                | Integer   | Age of the driver.                                            | Numeric value (e.g., 25, -1 for unknown)  |
| Driver_Gender             | Integer   | Gender of the driver.                                         | 1 = Male, 2 = Female, -1 = Unknown        |
| Vehicle_Type              | Integer   | Type of vehicle involved in the collision.                    | 1 = Car, 2 = Truck, 3 = Motorcycle, etc.  |
| Road_Condition            | Integer   | Road conditions at the time of the crash.                     | 1 = Dry, 2 = Wet, 3 = Snow/Ice, etc.      |
| Crash_Severity            | Integer   | Severity of the crash.                                        | 1 = Fatal, 2 = Serious, 3 = Minor         |
| Driver_Fault              | Integer   | Whether the driver was at fault.                              | 1 = At Fault, 0 = Not at Fault, -1 = Unknown |
| Weather_Condition         | Integer   | Weather conditions during the crash.                          | 1 = Clear, 2 = Rain, 3 = Fog, etc.        |
| Speed_Limit               | Integer   | Speed limit on the road where the crash occurred.              | Numeric value (e.g., 30, 45, etc.)        |
| Collision_Type            | Integer   | Type of collision (e.g., rear-end, head-on).                  | 1 = Rear-End, 2 = Head-On, 3 = Side-Swipe, etc. |
| Injury_Severity           | Integer   | Severity of injuries sustained by the driver or passengers.    | 1 = Fatal, 2 = Serious, 3 = Slight        |
| Light_Condition           | Integer   | Lighting conditions at the time of the crash.                 | 1 = Daylight, 2 = Dark - Street Lights On, etc. |
| Alcohol_Involved          | Integer   | Whether alcohol was involved in the crash.                    | 0 = No, 1 = Yes                           |
| Distraction_Involved      | Integer   | Whether distraction was involved in the crash.                | 0 = No, 1 = Yes                           |
| Driver_Education_Level    | Integer   | Education level of the driver.                                | 1 = High School, 2 = College, etc.        |
| Emergency_Response_Time   | Integer   | Time taken for emergency responders to arrive at the scene.    | Numeric value (e.g., 5, 15, etc.)         |

### Target Variable for Machine Learning
- **For Severity Prediction**: The target variable is `Crash_Severity`, representing the severity of the collision, categorized into:
  - 1 = Fatal
  - 2 = Serious
  - 3 = Minor

- **For Driver Fault Prediction**: The target variable is `Driver_Fault`, representing whether the driver was responsible for the accident:
  - 1 = At Fault
  - 0 = Not at Fault
  - -1 = Unknown

### Potential Features/Predictors for Machine Learning Models
- **Driver-Related Variables**:
  - Driver_Age
  - Driver_Gender
  - Driver_Education_Level
  - Alcohol_Involved
  - Distraction_Involved

- **Vehicle-Related Variables**:
  - Vehicle_Type
  - Speed_Limit
  - Collision_Type

- **Environmental Variables**:
  - Weather_Condition
  - Road_Condition
  - Light_Condition

- **Crash Variables**:
  - Emergency_Response_Time
  - Injury_Severity
  - Time_of_Day
