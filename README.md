# Edmonton Food Drive Project

![Edmonton Food Drive Logo](https://github.com/kmoreno013/MyProjects/blob/main/logo_efd.png?raw=true)

## Project Overview
The Edmonton Food Drive Project aims to develop a machine learning solution to optimize the management of food donation activities in Edmonton, AB. The project focuses on enhancing the efficiency and effectiveness of drop-off and pick-up processes, streamlining route planning, and improving resource allocation.

## Problem Statement
The current food donation management system in Edmonton faces challenges in coordinating drop-off locations, pick-up processes, and route planning. There is a need to automate and optimize these processes to ensure timely collection of donations and minimize logistical complexities.

## Objectives
- **Optimize Drop-off Locations**: Develop a machine learning model to identify the best drop-off locations based on geographic distribution and donation density.
- **Automate Pick-up Route Planning**: Implement a system for assigning and optimizing pick-up routes based on geographic structure and donation counts.
- **Enhance Stakeholder Coordination**: Streamline communication and coordination between Regional Coordinators, Stake Food Drive Representatives, and Ward Food Drive Representatives.
- **Improve Data Collection and Analysis**: Enhance data collection mechanisms to gain insights into donation patterns, resource utilization, and areas of improvement.

## Solution Approach
1. **Data Analysis and Machine Learning**: 
   - Utilize historical donation data and geographic information to identify optimal drop-off locations using clustering algorithms.
   
2. **Route Planning Algorithm**: 
   - Develop an efficient routing algorithm considering donation density, distance, and time constraints using VRP solutions.

3. **Data Collection Mechanism**: 
   - Integrate real-time data collection to monitor donation counts, route progress, and resource utilization.

## Data Sources
- 2023 and 2024 Edmonton Food Drive Dataset from Google and Microsoft Survey Forms

## EFD Dashboard
- Explore the [Edmonton Food Drive Dashboard](https://public.tableau.com/app/profile/kendrick.kent.moreno/viz/EFD2024Dashboard/EFDDashboard-Main) to gain insights into donation patterns and project performance.

## Contributors
- Kendrick Moreno - Team Member
- Roe Alincastre - Team Member
- Catrina Llamas - Team Member
- Professor Uchenna Mgbaja - Faculty Advisor

## Docker Information: How to Run the App

To run the Edmonton Food Drive API using Docker, follow the instructions below:
1. Prerequisites: Ensure you have Docker installed on your machine.

2. Clone the Repository: If you haven't cloned the project yet, do so by running:
```
git clone https://github.com/kmoreno013/EFD2024.git
cd EFD_2024
```

3. Build the Docker Container: In the root directory of the project (where the Dockerfile is located), run the following command to build the Docker Container:
```
docker compose up -d
```

4. Access the Application: Once the container is running, you can access the API through the following commands:
* Home Endpoint: `GET http://127.0.0.1:6060/efd2024_home`
* Health Status: `GET http://127.0.0.1:6060/health_status`
* Prediction (Polynomial Regression): `POST http://127.0.0.1:6060/v1/predict -H "Content-Type: application/json" -d @configs/request.json`
* Prediction (Decision Tree): `POST http://127.0.0.1:6060/v1/predict -H "Content-Type: application/json" -d @configs/request.json`

5. Shutting Down the Container: When you're done testing, you can stop the container by running:
```
docker stop <container_id>
docker ps
```
