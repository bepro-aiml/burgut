# Module 1 | Class 1 Assignment
## ML Type Classification Exercise

Name: <Asadullo Ismoilov>  
Date: April 26, 2026

## Task 1: Classify 17 Real-World Scenarios

| Scenario # | Learning Type | Justification | Input | Output |
|---|---|---|---|---|
| 1 | Supervised Learning | Churn prediction uses historical labeled examples (churned vs not churned). The model learns mapping from customer behavior to known churn labels. | Customer usage history, billing, complaints, plan type, tenure | Probability or label of churn next month |
| 2 | Reinforcement Learning | The car learns by acting in an environment and receiving penalties/rewards for behavior. This trial-and-error feedback is the key RL signal. | Sensor/camera/LiDAR state, traffic context, map | Driving action (steer, accelerate, brake) / policy |
| 3 | Supervised Learning (common in practice) | Recommenders are often trained from historical interactions as implicit labels (clicked/bought vs not). The model predicts user-item relevance. | User profile, browsing history, purchase history, item features | Ranked list of recommended products |
| 4 | Unsupervised Learning | There are no predefined neighborhood categories. The goal is to discover structure/patterns in usage data. | Internet usage metrics by neighborhood (time, volume, app mix) | Clusters of neighborhoods with similar patterns |
| 5 | Supervised Learning | Spam filtering is typically trained on labeled emails (spam/not spam). The model learns a classification boundary. | Email text, sender metadata, links, headers | Spam / not spam label (or spam score) |
| 6 | Reinforcement Learning | The robot improves through trial-and-error with reward for correct sorting actions. This is a sequential decision process with feedback. | Camera/sensor view of ore, robot state | Pick/place action policy for sorting |
| 7 | Reinforcement Learning | Dynamic pricing can be framed as an agent choosing prices and receiving reward from revenue/conversion responses over time. It continuously adapts based on feedback. | Demand signals, user response, time, competitor/context signals | Selected price for each plan/time |
| 8 | Supervised Learning | Diabetes risk prediction uses past patient records with known outcomes. This is a labeled prediction task. | Age, labs, vitals, medical history, lifestyle factors | Diabetes risk score or class |
| 9 | Unsupervised Learning | Subscriber segmentation without predefined group count/labels is clustering. The objective is structure discovery in behavior data. | Call/data usage, recharge behavior, location/time patterns | Subscriber segments/clusters |
| 10 | Reinforcement Learning | Self-play chess engines learn by receiving outcome-based rewards and improving policy/value estimates. This is classic RL. | Board state | Best move (policy) / value of position |
| 11 | Unsupervised Learning (anomaly detection) | Fraud is flagged as behavior that deviates from normal patterns without reliable labels for all fraud cases. That fits unsupervised anomaly detection. | User transaction sequences, time/location/device patterns | Anomaly/fraud risk score or alerts |
| 12 | Unsupervised Learning | Article topic clustering is done without labeled topic tags. The system groups similar documents by latent structure. | Article text embeddings/keywords/metadata | Topic clusters |
| 13 | Supervised Learning | House pricing uses known sale prices as labels and predicts a continuous target. This is supervised regression. | Square footage, district, floor, property features | Predicted house price |
| 14 | Reinforcement Learning | The robot explores routes and gets time-based penalties, learning a policy that minimizes travel cost. Feedback from actions over time indicates RL. | Warehouse map, current position, obstacles, shelf goals | Next movement action / optimal route policy |
| 15 | Supervised Learning | Translation with sentence pairs uses labeled input-output examples (source sentence -> target sentence). The model learns sequence mapping. | Uzbek sentence tokens | English sentence tokens |
| 16 | Unsupervised Learning (collaborative filtering style) | "Find similar users and cross-recommend" often relies on structure in interaction matrices without explicit labels. Similarity-based collaborative filtering is commonly treated as unsupervised. | User listening histories, song/artist interaction matrix | Personalized playlist/recommended tracks |
| 17 | Unsupervised Learning | It explicitly states no labeled normal/anomaly examples. The model must detect unusual traffic patterns from structure/deviation. | Network traffic time-series and feature stats | Anomaly scores/alerts for traffic spikes |

## Task 2: Reflection Questions

### 1) Which scenarios were hardest to classify? What made them tricky?
The hardest ones were #7 (dynamic pricing), #11 (fraud detection), and #16 (Spotify recommendations). They are tricky because each can be implemented with more than one ML paradigm depending on data availability and system design. For example, fraud can be supervised when labeled fraud cases are strong, but often starts as unsupervised anomaly detection when labels are sparse or delayed. Similarly, recommendation can be framed as unsupervised similarity learning or supervised ranking with implicit feedback labels.

### 2) Can a single business problem use more than one type of ML at different stages? Give an example.
Yes, many real systems are multi-stage and mix ML types. For telecom churn, an unsupervised model can first segment customers into behavior groups, then a supervised model predicts churn risk inside each segment, and finally an RL policy can optimize retention offers over time. This combination improves personalization and can outperform a single-model approach.

### 3) If you had to pick one ML type most useful for Uzbekistan’s telecom industry right now, which would it be and why?
I would pick supervised learning as the most immediately useful type. Telecom companies usually have large historical datasets with labels for outcomes like churn, payment default, and service complaints, which makes supervised models practical and high-impact. It also aligns with business KPIs because predictions can be directly tied to actions such as targeted retention, fraud case prioritization, and demand forecasting.

## Notes
Some scenarios are intentionally ambiguous. I chose the most defensible framing for each and justified the key indicator (labels, structure discovery, or trial-and-error reward).

