# Module 1 | Class 2 Assignment
## Probability in Everyday Life

Name: <Asadullo Ismoilov>  
Date: April 26, 2026

## Task 1: 3 Real-World Probability Examples

### Example 1: Weather Forecasting
Weather services predict outcomes like the probability of rain tomorrow, for example a 70% chance of rain in Tashkent. They use historical weather records, satellite images, humidity, wind speed, and pressure trends as input data. Forecasts are never 100% certain because the atmosphere is chaotic and small measurement errors can change the final outcome, which is why even strong 24-hour forecasts are often around 80-90% accurate instead of perfect.

### Example 2: Medical Risk Prediction
Hospitals estimate the probability that a patient will develop a condition such as diabetes within the next few years. The model can use age, BMI, blood sugar level, family history, and lifestyle indicators to output a risk score, such as a 35% predicted risk. Certainty is impossible because medical data is noisy and incomplete, and two patients with similar records can still have different outcomes due to genetics, behavior changes, or unobserved factors.

### Example 3: Email Spam Filtering
An email system predicts whether an incoming message is spam, often assigning a probability such as 0.92 spam likelihood. It uses features like sender reputation, keywords, links, message structure, and user feedback labels from past emails. Perfect accuracy is not possible because spammers constantly change tactics and legitimate emails can sometimes look similar to spam, so strong production filters might still make mistakes on about 1-3% of messages.

## Task 2: Why ML Models Can Never Be 100% Accurate
ML models cannot be 100% accurate because uncertainty is built into both data and the real world. One key reason is distribution shift: patterns in future data are often different from patterns in training data. For example, a spam filter trained on 2023 emails may perform at 98% accuracy on old validation data but drop when new spam styles appear in 2024. Noise in data also adds unavoidable error, such as missing fields, incorrect labels, or inconsistent measurements from different sources. Data quality problems mean the model learns from imperfect signals, so some predictions will always be wrong. The bias-variance tradeoff is another limit, because making a model more flexible can reduce bias but increase variance and overfitting. In practice, ML aims for reliable probabilities and useful decisions under uncertainty, not perfect certainty.

