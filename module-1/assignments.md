# Module 1: Introduction to AI and ML — Activities & Assignments

---

## Activity 1: ML Type Classification Exercise

**Objective:** Given real-world scenarios, classify each into supervised, unsupervised, or reinforcement learning. Then justify your choice and define the input/output.

### Scenarios

For each scenario below, determine: (a) learning type, (b) why, (c) what the input and output would be.

1. Uztelecom predicts which customers will cancel their contract next month.
2. A self-driving car learns to navigate Tashkent traffic by repeatedly trying routes and receiving penalties for unsafe moves.
3. Uzum marketplace recommends products to users based on their browsing and purchase history.
4. A system groups Tashkent neighborhoods by internet usage patterns without any predefined categories.
5. An email provider classifies incoming messages as spam or not spam.
6. A robot arm in a Navoi mining facility learns to sort ore by trial and error, getting a reward signal for correct picks.
7. Dynamic pricing for Uztelecom mobile data plans — adjusting prices in real time based on demand signals and user response.
8. A hospital in Samarkand uses patient records to predict diabetes risk.
9. An algorithm segments Uztelecom's 10 million subscribers into marketing groups based on usage behavior — no one told it how many groups to make.
10. A chess engine improves its play by competing against itself over millions of games.
11. Detecting SIM card fraud at Uztelecom by flagging transactions that look different from a user's normal pattern.
12. A news aggregator clusters articles into topics automatically.
13. Predicting house prices in Tashkent from square footage, district, and floor number.
14. A warehouse robot learns the fastest route between shelves by exploring and receiving time-based penalties.
15. A model translates Uzbek text to English using a dataset of sentence pairs.
16. Spotify creates a "Discover Weekly" playlist by finding users with similar listening habits and cross-recommending.
17. An algorithm detects unusual spikes in Uztelecom's network traffic without labeled examples of "normal" vs "anomaly."

### Tips for Students

- If the data has clear labels (input -> known correct output), lean supervised.
- If the system learns from its own actions and feedback over time, think reinforcement.
- If there are no labels and the goal is to find structure, that is unsupervised.
- Some scenarios are deliberately ambiguous. A good answer acknowledges the ambiguity and argues for a choice — that matters more than picking the "right" one.
- Scenario 16 is a common trick: collaborative filtering can be framed as either supervised or unsupervised depending on implementation. Make your case.

### Reflection Questions

1. Which scenarios were hardest to classify? What made them tricky?
2. Can a single business problem use more than one type of ML at different stages? Give an example.
3. If you had to pick one ML type that is most useful for Uzbekistan's telecom industry right now, which would it be and why?

### Submission Requirements

- Table or numbered list with: scenario number, learning type, justification (1-2 sentences), input, output.
- At least 2 sentences on each reflection question.
- Format: PDF or Google Doc. No handwritten photos.

### What a Good Submission Looks Like

- Every classification has a clear justification, not just the label.
- Inputs and outputs are specific (e.g., "input: customer usage data over 6 months; output: binary churn/no-churn label"), not vague.
- Reflection answers show you actually thought about it, not just restated the definition.

---

## Activity 2: ITES Case Study — Recommendation or Fraud

**Objective:** Analyze a real ITES (IT-Enabled Services) ML application from both a technical and ethical angle.

### Choose One Scenario

**Option A: E-Commerce Recommendation System (Uzum-like platform)**
An Uzbek e-commerce platform wants to build a recommendation engine. Users browse thousands of products daily. The platform wants to show personalized suggestions on the homepage and in "you might also like" sections. They have 2 years of user clickstream data, purchase history, and product metadata.

**Option B: Mobile Payment Fraud Detection**
A mobile money service (like Payme or Click) processes millions of transactions monthly. They want an ML system that flags suspicious transactions in real time. They have historical transaction logs, user profiles, and a small set of confirmed fraud cases labeled by their security team.

### For Your Chosen Scenario, Answer These

1. **Problem type:** What kind of ML problem is this? (classification, regression, clustering, etc.) Why?
2. **Data requirements:** What specific data would the model need? List at least 8 data fields.
3. **Evaluation metric:** How would you measure if the model is working? Why that metric over others?
4. **Bias and fairness risks:** What could go wrong? Who might be unfairly affected?
5. **Privacy concerns:** What user data is involved, and what are the risks of collecting it?
6. **Your recommendation:** Should this system be deployed as-is, with guardrails, or not at all? Justify.

### Discussion Format

This activity has two parts:
- **In-class discussion** (20-30 min): Groups of 3-4 discuss their chosen scenario. Each group presents their key finding to the class.
- **Written component** (individual): 1-2 page writeup answering all 6 questions above.

### Tips for Students

- There is no single correct evaluation metric. What matters is your reasoning.
- For bias: think about who is in the training data and who is not. Uzbekistan has urban/rural divides, age gaps in tech adoption, and language differences (Uzbek vs Russian speakers). These all matter.
- For privacy: Uzbekistan's data protection landscape is still developing. Think about what should be protected even if current law does not require it.

### Reflection Questions

1. Would your ethical concerns change if this system were deployed in a country with strict data privacy laws (e.g., EU with GDPR)?
2. Is it possible to build a useful recommendation/fraud system without collecting personal data? What would you lose?
3. Who should be responsible when an ML system makes a wrong decision — the developer, the company, or the algorithm?

### Submission Requirements

- 1-2 page writeup covering all 6 questions.
- Cite at least one real-world example of a similar system (any country).
- Format: PDF or Google Doc.

---

## Activity 3: Real-World Integration — Telecom Churn Framing

**Objective:** Take a business problem and frame it as a machine learning problem. This is the single most important skill in applied ML — if you frame it wrong, nothing else matters.

### Scenario

An Uzbek telecom company (think Uztelecom, Beeline, or Ucell) is losing subscribers. Management says: "We want to know which customers will leave so we can offer them retention deals before they go." Your job is to turn this business request into a concrete ML problem definition.

### Tasks

Fill in the starter template below:

```
## Churn Prediction — ML Problem Framing

**Business objective:**
[What does the company actually want to achieve?]

**Target variable:**
[What exactly are you predicting? How do you define "churn"?]

**ML problem type:**
[Classification? Regression? Why?]

**Input features (list at least 10):**
1.
2.
3.
...

**Evaluation metric:**
[Which metric and why? Consider what matters more — catching churners or avoiding false alarms.]

**Class imbalance:**
[Is churn balanced 50/50? If not, what ratio would you expect, and how does that affect your approach?]

**Ethical considerations:**
[What could go wrong if this model is used to make decisions about customers?]

**Data availability:**
[Where would this data come from in an Uzbek telecom? What might be hard to get?]
```

### Discussion Format

- **Group work** (15-20 min): Teams of 3-4 fill in the template together.
- **Class discussion** (15 min): Compare definitions across groups. Focus on where groups disagree — that is where the learning happens.
- **Individual submission:** Your own completed template with at least 10 features and thoughtful responses to every section.

### Tips for Students

- Defining churn is harder than it sounds. Does "churn" mean the customer formally canceled? Stopped using data? Ported their number? Your definition shapes everything downstream.
- For features, think beyond obvious ones like "monthly spend." Consider: time since last recharge, number of support calls, network quality in their area, whether they have a contract or prepaid.
- Class imbalance is real. In most telecoms, churners are 2-5% of the base. If your model just predicts "no churn" for everyone, it gets 95%+ accuracy. That is useless. Think about what metric actually captures value.
- Ethical angle: if the model disproportionately flags low-income users as churn risks, and those users then get aggressive sales calls, that is a problem.

### Reflection Questions

1. Two groups might define "churn" differently and both be correct. What does that tell you about ML problem framing?
2. If you could only use 3 features to predict churn, which 3 would you pick and why?
3. Should a telecom company tell customers they are using ML to predict their behavior? Why or why not?

### Submission Requirements

- Completed template with all sections filled.
- At least 10 features with brief descriptions of each.
- Format: PDF or Google Doc.

---

## Activity 4: Quiz — 15 Multiple Choice Questions

Test your understanding of Module 1 concepts. Answer all 15 questions.

### Section 1: AI/ML Basics (Q1-Q3)

**Q1.** What is the relationship between AI and ML?
- a) AI and ML are the same thing
- b) ML is a subset of AI
- c) AI is a subset of ML
- d) They are completely unrelated

**Correct: b) ML is a subset of AI.**
ML is one approach to achieving AI. AI is the broader field; ML is a specific family of techniques within it that learn from data.

**Q2.** What differentiates deep learning from traditional ML?
- a) It uses less data
- b) It does not require computers
- c) It uses multi-layer neural networks
- d) It only works for image data

**Correct: c) It uses multi-layer neural networks.**
Deep learning models stack multiple layers of neurons to learn increasingly abstract representations. Traditional ML typically uses handcrafted features.

**Q3.** Supervised learning requires:
- a) Unlabeled data only
- b) A reward signal
- c) Labeled input-output pairs
- d) No data at all

**Correct: c) Labeled input-output pairs.**
The "supervised" part means someone has provided the correct answers (labels) for the training examples.

### Section 2: ML Types (Q4-Q7)

**Q4.** The goal of unsupervised learning is to:
- a) Predict a target variable
- b) Discover hidden structure in data
- c) Maximize a reward
- d) Minimize labeled data usage

**Correct: b) Discover hidden structure in data.**
Without labels, the model finds patterns, clusters, or groupings on its own.

**Q5.** Unsupervised learning is used to:
- a) Train with labeled examples
- b) Discover hidden structure in data
- c) Optimize a reward function
- d) Perform real-time predictions

**Correct: b) Discover hidden structure in data.**
Note: This question overlaps with Q4. Both emphasize the same core idea — unsupervised learning finds patterns without labels.

**Q6.** Reinforcement learning differs from other ML types because it:
- a) Uses labeled data
- b) Learns through trial-and-error with feedback
- c) Requires no data
- d) Only works with images

**Correct: b) Learns through trial-and-error with feedback.**
An RL agent takes actions, observes outcomes, and adjusts its strategy based on rewards or penalties.

**Q7.** Customer segmentation is an example of:
- a) Supervised learning
- b) Reinforcement learning
- c) Unsupervised learning
- d) Rule-based programming

**Correct: c) Unsupervised learning.**
You are grouping customers by behavior patterns without predefined categories — classic clustering.

### Section 3: Foundations (Q8-Q11)

**Q8.** The i.i.d. assumption means data points are:
- a) Dependent and differently distributed
- b) Independent and identically distributed
- c) Indexed and incrementally distributed
- d) Inferred and inductively derived

**Correct: b) Independent and identically distributed.**
Most ML algorithms assume each data point is drawn independently from the same underlying distribution. When this breaks (e.g., time-series data), you need specialized methods.

**Q9.** Empirical risk is:
- a) The theoretical minimum error
- b) Calculated using training data
- c) Always equal to true risk
- d) Independent of the model

**Correct: b) Calculated using training data.**
Empirical risk is the average loss computed over your actual training examples — it approximates the true risk you cannot directly measure.

**Q10.** A model with high variance is likely:
- a) Underfitting
- b) Perfectly balanced
- c) Overfitting
- d) Using too little data

**Correct: c) Overfitting.**
High variance means the model is too sensitive to training data — it memorizes noise instead of learning general patterns.

**Q11.** The purpose of optimization in ML is to:
- a) Collect more data
- b) Visualize results
- c) Find parameters that minimize error
- d) Remove outliers

**Correct: c) Find parameters that minimize error.**
Training an ML model is fundamentally an optimization problem: find the parameter values that make the loss function as small as possible.

### Section 4: Applications (Q12-Q13)

**Q12.** A fraud detection model has 99% accuracy but misses most fraud cases. This is likely due to:
- a) Too many features
- b) Class imbalance
- c) Too much training data
- d) High bias

**Correct: b) Class imbalance.**
If 99% of transactions are legitimate, a model that always predicts "not fraud" gets 99% accuracy while catching zero fraud. Accuracy is misleading here — you need precision, recall, or F1.

**Q13.** Collaborative filtering works by:
- a) Filtering out irrelevant data
- b) Using similarities between users or items to make recommendations
- c) Applying content-based rules only
- d) Removing duplicate data

**Correct: b) Using similarities between users or items to make recommendations.**
"Users who liked X also liked Y" is the core idea. It does not need to understand the content itself — just the patterns in user behavior.

### Section 5: Ethics (Q14-Q15)

**Q14.** An AI hiring tool rejects more candidates from certain demographics. This is an example of:
- a) Overfitting
- b) Data leakage
- c) Algorithmic bias
- d) Feature engineering

**Correct: c) Algorithmic bias.**
If the training data reflects historical hiring discrimination, the model learns and reproduces those biases. This is one of the most studied problems in ML ethics.

**Q15.** A bank uses ML to deny loans but refuses to explain why to applicants. This raises concerns about:
- a) Computational efficiency
- b) Transparency and explainability
- c) Data storage
- d) Model accuracy

**Correct: b) Transparency and explainability.**
People affected by ML decisions have a right to understand why. "The algorithm said so" is not an acceptable explanation, especially for high-stakes decisions like loans.

---

## Grading Rubric for Activities 1-3

| Criteria | Excellent (90-100%) | Good (70-89%) | Needs Work (<70%) |
|----------|-------------------|---------------|-------------------|
| **Correctness** | All classifications/framings are accurate with clear justification | Most are correct, minor gaps in reasoning | Multiple incorrect answers or missing justifications |
| **Depth of analysis** | Goes beyond definitions — considers edge cases, ambiguity, real-world nuance | Solid understanding, stays close to textbook answers | Surface-level or copy-pasted definitions |
| **Specificity** | Concrete examples, specific data fields, named metrics | Somewhat specific but occasionally vague | Generic or hand-wavy ("the model uses data") |
| **Ethical reasoning** | Identifies non-obvious risks, considers Uzbekistan context | Mentions ethics but stays general | Ethics section missing or one-liner |
| **Presentation** | Clean formatting, complete sections, easy to follow | Mostly complete, minor formatting issues | Missing sections, hard to read |

---

## Homework

**Due:** End of Module 1 (before Module 2 starts).

### Reading

Read the following and come prepared to discuss:
- [A Few Useful Things to Know About Machine Learning](https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf) — Pedro Domingos (focus on sections 1-4)
- [Machine Learning that Matters](https://icml.cc/2012/papers/298.pdf) — Kiri Wagstaff (skim for the main argument)

### Written Assignment

Identify **3 ML use cases** in Uzbekistan's ITES sector (telecom, fintech, e-commerce, government services, etc.).

For each use case:
1. Describe the business problem in 2-3 sentences.
2. What type of ML would you use? Why?
3. What data would you need?
4. What is one ethical risk?

**Format:** 1-2 pages, PDF or Google Doc. Cite your sources if you reference specific companies or statistics.
