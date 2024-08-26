# User Behavior Analytics
- This project is designed to analyze user behavior patterns and detect anomalies that might indicate security threats or unusual activities. By applying machine learning techniques such as K-means clustering, this system identifies user behaviors that significantly deviate from the norm, providing insights for cybersecurity and user monitoring purposes.
- User Behavior Analytics (UBA) is an essential part of modern cybersecurity strategies. It involves analyzing user activities to detect anomalies that may indicate potential security incidents. This project implements a simple UBA system using Python and K-means clustering to identify abnormal user behaviors based on login times and action counts.

## Features
- Load and preprocess user behavior data from a CSV file.
- Perform exploratory data analysis (EDA) to visualize user behavior patterns.
- Use K-means clustering to segment users into different behavior clusters.
- Identify and flag anomalous user behaviors based on their distance from cluster centers.
- Save the detected anomalies to a CSV file for further analysis.

## Installation
1. Clone the Repository:
`git clone https://github.com/yourusername/user-behavior-analytics.git`
2. Navigate to the Project Directory:
`cd user-behavior-analytics`
3. Create a Virtual Environment (Optional but recommended):
`python -m venv venv`
`source venv/bin/activate  # On Windows use venv\Scripts\activate`
4. Install Required Dependencies:
   `pip install -r requirements.txt`

## Usage
1. Prepare the Dataset:
- Place your user behavior data in a CSV file named `user_behavior.csv`.
- Ensure the CSV file has the following columns: `user_id`, `login_time`, `location`, `action_count`.
2. Run the Analysis:
`python user_behavior_analytics.py`
3. Review the Results:
- The script will output basic EDA results and visualization plots.
- Detected anomalies will be saved in a CSV file named `anomalous_user_behavior.csv`.

## Dataset
- The dataset should be a CSV file named `user_behavior.csv` with the following columns:
- `user_id`: Unique identifier for each user.
- `login_time`: Numeric value representing the time of login.
- `location`: Categorical value indicating the location of login (e.g., Office, Home, Cafe).
- `action_count`: Numeric value representing the number of actions performed by the user.

## Contributing
- Contributions are welcome! If you have any suggestions or improvements, please feel free to submit a pull request or open an issue.



