# Shopping Mall Shopper Profiling
## Built with
- Python
- scikit-learn

## Background
Understanding the behavior and preferences of shoppers in shopping malls is crucial for businesses to tailor their strategies effectively and enhance the shopping experience. Traditional methods of shopper profiling often rely on manual observation or simple metrics, which may not capture the complexity of shopper behavior.

## Description
To address this challenge, this project implements a comprehensive 3-stage machine learning solution for profiling shopping mall shoppers.

### Stage 1: Store Footfall Clustering
<img src="images/shopper.png" alt="Example image showing footall patterns of different stores in a shopping mall." width="500">

Example image showing footall patterns of different stores in a shopping mall.

In the first stage, the project employs clustering techniques to group stores based on their footfall patterns. This process categorizes stores into clusters with similar shopper traffic, facilitating the quantitative identification of distinct store groups within the mall.

### Stage 2: Rule-based Cluster Classification
Next, a rule-based classification approach is employed to categorize shoppers into segments based on their interaction with the clustered store groups. By analyzing shopper behavior within specific store clusters, this stage provides insights into the preferences and tendencies of different shopper segments.

### Stage 3: Shopper Footfall Forecasting
Finally, the project forecasts shopper footfall for each store cluster, leveraging historical data and machine learning models. By predicting future footfall patterns, businesses can optimize staffing, inventory management, and promotional strategies to meet anticipated demand.

Additionally, the solution is prototyped into a web application using Streamlit, providing an intuitive interface for data visualization and interaction. This facilitates seamless exploration of shopper profiles and insights, empowering businesses to make informed decisions to enhance the shopping experience and drive revenue.

