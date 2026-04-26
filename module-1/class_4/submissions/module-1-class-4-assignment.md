# Module 1 | Class 4 Assignment
## ITES Case Study - Mobile Payment Fraud Detection

Name: <Asadullo Ismoilov>  
Date: April 26, 2026

Chosen scenario: **Option B - Mobile Payment Fraud Detection (Payme/Click-like service)**

## Task 1: Problem Type
This is primarily a **supervised binary classification** problem: each transaction is predicted as `fraud` or `not fraud`. The target is categorical, not numeric, so this is not regression. A secondary unsupervised anomaly detection layer can also help because fraud patterns evolve and labeled fraud cases are limited. The core production decision, however, is still binary classification at transaction time.

## Task 2: Data Requirements (at least 8 fields)
- `transaction_id`: Unique key to track each event, join logs, and investigate false positives/false negatives.
- `amount`: Fraud often appears as unusual transaction sizes relative to a user's normal spending pattern.
- `timestamp`: Time-of-day and day-of-week patterns help detect impossible or suspicious timing behavior.
- `sender_id`: Needed to build account-level historical behavior profiles and risk trends.
- `receiver_id` (merchant/account): Some destinations are statistically higher-risk and should raise model attention.
- `transaction_type` (P2P, bill pay, merchant payment, cash-out): Different channels have different fraud baselines.
- `device_fingerprint` (device model, app version, OS signals): New or rapidly changing devices are common account-takeover signals.
- `geo_location` (region, IP geolocation): Sudden location jumps or impossible travel speed are strong fraud indicators.
- `historical_transaction_frequency` (last 24h/7d): Bursty activity can indicate bot-driven abuse or stolen-account activity.
- `account_age`: Newly created accounts usually have less trust history and higher uncertainty.
- `failed_login_count` or `authentication_events`: Repeated failed access attempts can indicate takeover attempts before fraud.
- `label_fraud_confirmed` (from security team): Ground-truth labels are required to train and evaluate supervised models.

## Task 3: Evaluation Metric
My primary metric would be **Recall@ReviewCapacity** (recall measured at an operational alert threshold the risk team can handle), with **Precision** as a secondary guardrail. In fraud detection, missing real fraud (false negatives) is costly, so recall must be high. But if precision is too low, too many legitimate users are flagged, creating support burden and trust issues. A practical target could be: maintain recall above a defined floor while keeping precision high enough that analysts can review alerts in real time.

## Task 4: Bias and Fairness Risks
Bias can arise if training data over-represents some user groups and under-represents others, causing uneven error rates. In Uzbekistan, urban users may be over-represented in digital payments, so rural behavior could be misclassified as "abnormal" simply because it is less common in the data. Language and interface differences (Uzbek/Russian usage patterns) can indirectly correlate with risk signals and create proxy discrimination if not monitored. The model may also over-flag low-income users who transact in irregular small bursts, even when behavior is legitimate. Fairness checks should compare false positive and false negative rates by region, account age bands, and customer segments.

## Task 5: Privacy Concerns
This system processes highly sensitive personal and financial data: transaction histories, location traces, device signals, and identity-linked account behavior. If breached, bad actors could reconstruct routines, target social engineering attacks, or perform account takeover with contextual knowledge. Even without a breach, excessive retention of granular behavior data increases surveillance risk and potential internal misuse. Given Uzbekistan's evolving data protection environment, the company should still apply strong safeguards by default: data minimization, purpose limitation, encryption, access controls, and short retention windows for raw sensitive logs.

## Task 6: Recommendation
This system should be deployed **with guardrails**, not as-is. The business benefit is high (fraud loss reduction and user trust), but incorrect flags can harm legitimate users and create fairness concerns. Guardrails should include: human review for high-impact blocks, periodic bias audits by segment/region, strict privacy controls (minimization, encryption, role-based access), and user appeal channels for reversed decisions. The team should also run continuous monitoring for data drift and model recalibration, because fraud tactics change quickly. Deployment is justified only if these controls are enforced as operational policy, not optional best practice.

## Real-World Example (Required Citation)
A similar approach is used globally by payment platforms such as **Stripe Radar (US)**, which combines machine learning and rule-based controls to score transaction risk and block suspected fraud. This supports the same design pattern used in this analysis: probabilistic scoring + thresholding + analyst/operations workflow rather than fully autonomous irreversible blocking.

## Task 7: Reflection Questions
### 1) Would your ethical concerns change under strict privacy laws (e.g., GDPR)?
Yes, strict privacy laws would not remove ethical concerns, but they would force better controls and accountability. Requirements like purpose limitation, lawful basis, data subject rights, and auditability would reduce misuse risk and push teams to justify every data field they collect. I would still worry about fairness and false positives, but governance quality would likely improve.

### 2) Is it possible to build a useful fraud system without collecting personal data? What would you lose?
A partially useful system is possible using aggregated or pseudonymous behavioral signals, but performance would usually drop. Without account-linked history, device continuity, and contextual signals, the model loses important patterns needed to distinguish unusual but legitimate behavior from true fraud. You gain privacy, but you lose detection power and increase both missed fraud and uncertain alerts.

### 3) Who should be responsible when an ML system makes a wrong decision - developer, company, or algorithm?
The primary responsibility should be the **company**, because it deploys the system, sets policy, and controls remediation. Developers are responsible for technical quality and risk reporting, but they do not own final business decisions alone. The algorithm itself cannot be morally or legally responsible; it is a tool. Accountability should be shared operationally, but governance ownership must stay with the organization.

