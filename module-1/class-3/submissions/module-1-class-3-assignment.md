# Module 1 | Class 3 Assignment
## AI in Uzbekistan - Find It

Name: <Your Name>  
Date: April 26, 2026

## Task 1 + Task 2: 3 AI/ML Applications in Uzbekistan (Summary Table)

| Application | Sector | ML Type | Data Used |
|---|---|---|---|
| Uzum Market product recommendations ("You may also like" / personalized feed) | E-commerce | Combination: supervised ranking + unsupervised similarity (collaborative filtering) | Clicks, views, add-to-cart events, purchases, search queries, product metadata, session context |
| Payme transaction risk and fraud detection | Fintech / Digital payments | Combination: supervised classification + unsupervised anomaly detection | Transaction amount/time/location, device fingerprints, merchant category, account history, chargeback/fraud labels |
| Humans.uz customer support chatbot for service and billing questions | Telecom / Digital services | NLP (intent classification + entity extraction; often supervised) | Chat logs, FAQ/support tickets, language patterns (Uzbek/Russian), user account context, feedback on answer quality |

## Task 3: Explanation Paragraphs

### 1) Uzum Market recommendations
Uzum appears to use recommendation logic in its shopping experience, where product lists adapt to user behavior. This is likely a hybrid ML setup: supervised ranking predicts which items a user is most likely to click or buy, while unsupervised similarity helps group related users and products. The model would need event data such as views, clicks, purchases, and product attributes to learn relevance scores. Exact model architecture is not usually published, so the specific algorithm is an educated guess based on standard e-commerce recommendation systems.

### 2) Payme fraud detection
For a payment platform like Payme, fraud detection is a strong candidate for supervised classification because historical fraud cases can be labeled and used for training. At the same time, unsupervised anomaly detection is likely used to catch new fraud patterns that do not match old labels. Useful signals include transaction timing, amount, device behavior, location change, and account history over time. This classification is partly inferred from common fintech practice, since public sources typically describe outcomes (safer payments) more than technical details.

### 3) Humans.uz chatbot
Humans.uz offers digital services where automated customer support is a natural NLP application. A chatbot usually combines intent classification (for example, billing issue vs plan change), entity extraction (phone number, tariff, payment date), and retrieval/generation of responses. Training data would include historical support conversations, FAQ pairs, and user feedback about whether the answer solved the issue. If internal model details are not publicly confirmed, this remains an educated guess, but the ML type is most reasonably NLP with supervised components.

## Notes
Where implementation details are not publicly confirmed, the ML type choices above are reasoned estimates based on visible product behavior and standard ITES patterns from class (recommendations, fraud detection, chatbots).

