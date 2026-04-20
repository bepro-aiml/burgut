# Module 1: Introduction to AI and ML — Slide Blueprint

> 5 classes (4 lectures + 1 lab) | ~15-20 slides per class
> Each slide: Title, bullets, speaker notes, visual suggestion
> Activity integration points marked with [ACTIVITY]

---

## CLASS 1: What is AI? History + AI vs ML vs DL + ML Types (1.5h)

### Slide 1 — Title Slide
**Title:** Welcome to AI and Machine Learning
- Module 1, Class 1
- What is AI? Where did it come from? How do machines learn?
- Today: History, key ideas, AI/ML/DL, and the three types of ML

**Speaker notes:** Welcome everyone. This is the first class of an 8-module program that will take you from zero to building your own ML projects. Today we cover a lot of ground — the big picture of AI, how ML fits in, and the three fundamental ways machines learn. But first, let's see ML in action before we define a single term.

**Visual:** Clean title card with module number, class number.

---

### Slide 2 — [DEMO] Let's See ML Before We Define It
**Title:** Your First ML Model — Live Demo
- Open Google Colab
- Paste 5 lines of code
- Load a small dataset, train a Random Forest classifier, see predictions
- Total time: under 60 seconds

**Speaker notes:** Before I explain what machine learning is, I want you to see it. I'm going to open Colab, paste five lines, and run a working model. Watch what happens. It loads data, finds patterns, and predicts outcomes for new examples — no rules written by hand. You don't need to understand the code yet. Just watch it work.

**Visual:** Colab notebook projected live. Five lines of code visible. Output showing predictions.

---

### Slide 3 — [DEMO] What Just Happened?
**Title:** What You Just Saw
- We loaded data (input examples with known answers)
- We trained a model (the algorithm found patterns on its own)
- We made predictions on new data (the model applied what it learned)
- You don't understand it yet. That's fine.
- By Module 4, you will build this yourself from scratch.

**Speaker notes:** Three things happened. Data went in, the model learned patterns, and it made predictions. That's it. That's the entire loop of machine learning. Everything we cover today and for the next four modules is explaining why that worked and how to do it well. Hold onto this — it's your anchor for all the theory that follows.

**Visual:** Three-step diagram: Data In → Model Learns → Predictions Out. "You are here" marker at step 1.

---

### Slide 4 — What Comes to Mind?
**Title:** When you hear "Artificial Intelligence," what do you think of?
- Robots? Self-driving cars? ChatGPT?
- Movies vs reality — how different is it?
- Quick poll: "Name one AI thing you used today"

**Speaker notes:** Before we define anything, let's see what you already know. I want hands up or shout-outs — what comes to mind when you hear AI? Most people think Terminator or Iron Man. The reality is more boring but honestly more impressive.

**Visual:** Collage — movie AI (Terminator, JARVIS) on the left vs real AI (Google Translate, Spotify recommendations, face unlock) on the right.

> **Quick poll:** "Raise your hand if you used AI today — phone keyboard predictions, Google search, YouTube recommendations count."

---

### Slide 5 — Defining Artificial Intelligence
**Title:** What is AI? A Working Definition
- AI: Systems that perform tasks normally requiring human intelligence
- Perception, reasoning, learning, decision-making, language understanding
- Key insight: AI is a broad field, not a single technology
- Exists on a spectrum — narrow AI (today) vs general AI (theoretical)

**Speaker notes:** Here's the working definition we'll use. AI is any system that can do something that would normally need a human brain. That includes seeing, hearing, reading, deciding, and learning. Almost all AI today is narrow — it does one thing well. General AI that thinks like a human? Still science fiction.

**Visual:** Spectrum bar — left side "Narrow AI (calculator, spam filter, face recognition)" to right side "General AI (human-level reasoning)" with a marker showing "We are here" firmly on the narrow side.

---

### Slide 6 — Brief History of AI: The Early Days
**Title:** AI Timeline — 1950s to 1980s
- 1950: Turing Test — "Can machines think?"
- 1956: Dartmouth Conference — AI gets its name
- 1960s-70s: Rule-based systems, early optimism
- 1974-1980: First "AI Winter" — overpromised, underdelivered
- 1980s: Expert systems boom, then bust (second winter)

**Speaker notes:** AI didn't start with ChatGPT. It started in the 1950s when Alan Turing asked a simple question — can machines think? The Dartmouth Conference in 1956 is where the term "Artificial Intelligence" was literally invented. Early AI was all rules — if this, then that. People got very excited, governments poured money in, and then it didn't deliver. Twice. These crashes are called AI winters.

**Visual:** Horizontal timeline with key dates, icons for each event. Highlight the two "winter" periods in blue/cold colors.

---

### Slide 7 — Brief History of AI: The Modern Era
**Title:** AI Timeline — 1990s to Today
- 1997: IBM Deep Blue beats Kasparov at chess
- 2006: Geoffrey Hinton's deep learning breakthrough
- 2012: AlexNet wins ImageNet — deep learning goes mainstream
- 2016: AlphaGo beats world champion Lee Sedol
- 2022-2024: ChatGPT, GPT-4, Gemini — large language models explode

**Speaker notes:** The comeback started in the late 90s with better hardware and more data. Deep Blue beating Kasparov was a wake-up call. But the real turning point was 2012 — a neural network called AlexNet crushed the ImageNet competition and suddenly deep learning was the hottest thing in tech. Fast forward to today and you have ChatGPT, which most of you have probably used.

**Visual:** Continuation of the timeline from Slide 4. Include small images — Deep Blue, AlphaGo board, ChatGPT interface.

---

### Slide 8 — AI in Uzbekistan: What's Happening Now
**Title:** AI is Already Here in Uzbekistan
- Presidential decree on AI development (2023)
- Uztelecom: AI chatbots handling customer queries
- Banking: Fraud detection at Kapitalbank, Uzum
- Agriculture: Satellite imagery for crop monitoring
- Government: Digital ID systems, e-government portals

**Speaker notes:** This isn't just a Silicon Valley thing. Uzbekistan has a national AI strategy. Telecom companies are using chatbots right now — I work on one of these systems at Uztelecom. Banks are catching fraud with ML models. And farmers are starting to use satellite data to predict crop yields. You're entering this field at the right time for Uzbekistan.

**Visual:** Map of Uzbekistan with icons pinned to different sectors — telecom, banking, agriculture, government.

---

### Slide 9 — AI vs ML vs DL: The Nested Circles
**Title:** How AI, ML, and DL Relate
- AI: The broadest category — any intelligent system
- ML: A subset of AI — systems that learn from data (not hardcoded rules)
- DL: A subset of ML — uses neural networks with many layers
- All DL is ML, all ML is AI — but not the other way around

**Speaker notes:** This is probably the most important diagram in the entire module. Think of it as Russian dolls. AI is the biggest doll — it includes everything. Inside that is ML, which is a specific approach where the system learns patterns from data instead of being manually programmed. Inside ML is deep learning, which uses brain-inspired neural networks. We'll spend most of this course in the ML circle.

**Visual:** Nested concentric circles — outermost "Artificial Intelligence," middle "Machine Learning," innermost "Deep Learning." Each ring has 2-3 example technologies listed.

---

### Slide 10 — AI vs ML: What's the Difference?
**Title:** Traditional AI vs Machine Learning
- Traditional AI: Human writes the rules → system follows them
- ML: Human provides data → system discovers the rules
- Example: Spam filter
  - Rule-based: "If email contains 'lottery' → spam"
  - ML-based: Train on 10,000 labeled emails → learns patterns itself
- ML scales better — rules break when the world is complex

**Speaker notes:** Here's the clearest way to see it. Traditional AI means a programmer sits down and writes every rule. That works for simple stuff but breaks when things get messy. ML flips it — you give the system examples and it figures out the rules. Imagine writing rules for every possible spam email vs just showing it thousands of examples. Which sounds more practical?

**Visual:** Two-column comparison diagram. Left: "Human writes rules → Program → Output." Right: "Data + Answers → ML Algorithm → Rules/Model."

---

### Slide 11 — ML vs DL: When Do You Need Deep Learning?
**Title:** Machine Learning vs Deep Learning
- ML: Works well with structured data (tables, spreadsheets)
- DL: Excels with unstructured data (images, text, audio, video)
- ML: Needs manual feature engineering — you tell it what to look at
- DL: Learns features automatically from raw data
- Trade-off: DL needs much more data and compute power

**Speaker notes:** So when do you use plain ML vs deep learning? If your data fits in a spreadsheet — customer age, purchase amount, subscription type — classic ML works great and is simpler. But if your data is images, audio recordings, or raw text, deep learning can learn directly from that without you extracting features manually. The catch: deep learning is hungry — it needs a lot of data and serious computing power.

**Visual:** Two-column table. ML column: tabular data icon, feature engineering arrow, small dataset icon. DL column: image/audio icons, automatic feature learning, GPU cluster icon.

---

### Slide 12 — Data Science Venn Diagram
**Title:** Where Does Data Science Fit?
- Data Science = intersection of math/statistics + domain knowledge + programming
- ML is a tool within the data science toolkit
- Data scientists also do analysis, visualization, storytelling
- You don't need to be an expert in all three — but you need literacy in each

**Speaker notes:** You'll hear "data science" thrown around a lot, sometimes interchangeably with ML. They're related but not the same. Data science is broader — it's the intersection of math, coding, and knowing your business domain. ML is one of the tools a data scientist uses. Think of a carpenter — ML is like a power drill. Important, but not the only tool in the box.

**Visual:** Classic Venn diagram — three overlapping circles: "Mathematics & Statistics," "Computer Science & Programming," "Domain Expertise." Intersection labeled "Data Science." ML highlighted as a subset within the overlap.

---

### Slide 13 — Common Misconceptions
**Title:** Things People Get Wrong About AI
- "AI will replace all jobs" → It will change jobs, not eliminate them
- "AI understands things" → It finds patterns, it doesn't comprehend
- "More data always means better AI" → Bad data = bad results
- "AI is objective" → AI inherits biases from its training data

**Speaker notes:** Before we wrap up, let's clear up some myths. AI isn't coming for all jobs — it's coming for specific tasks within jobs. AI doesn't actually understand anything — it's very sophisticated pattern matching. And AI isn't neutral or objective. If you train it on biased data, you get biased results. We'll dig deep into this in Class 4 when we talk ethics.

**Visual:** Four myth cards — each with a "myth" crossed out in red and the "reality" in green below it.

---

### Slide 14 — Key Terms Glossary
**Title:** Terms You Should Know After Today
- **Artificial Intelligence** — systems performing tasks requiring human intelligence
- **Machine Learning** — learning patterns from data without explicit programming
- **Deep Learning** — ML using multi-layered neural networks
- **Narrow AI** — AI designed for a specific task
- **General AI** — hypothetical AI with human-level reasoning

**Speaker notes:** Quick reference slide. These five terms should be in your vocabulary by the time you walk out today. Don't worry about memorizing definitions — focus on understanding the relationships between them.

**Visual:** Clean glossary layout, each term with a small icon.

---

### Slide 15 — Discussion: AI Around You
**Title:** Think-Pair-Share: AI in Your Daily Life
- Think (1 min): List 3 AI-powered tools you use regularly
- Pair (2 min): Compare lists with your neighbor
- Share (3 min): Each pair shares their best example with the class

> **Think-pair-share:** Give students 1 minute individually, then pair up. Collect answers on the board.

**Speaker notes:** Let's make this real. Take one minute and think about your own life — what apps or services are using AI? Then turn to the person next to you and compare. We'll hear from a few pairs after. I bet we'll find AI in places you didn't expect.

**Visual:** Three-step diagram: Think (person icon) → Pair (two people) → Share (group icon).

---

### Slide 16 — Transition: How Do Machines Learn?
**Title:** Now That We Know What ML Is — How Does It Learn?
- AI > ML > DL — we've covered the what
- Now: HOW do machines learn? Three fundamentally different approaches
- Think of it like how humans learn — teacher, self-exploration, or trial and error

**Speaker notes:** We've covered the big picture — what AI is, where it came from, and how ML fits in. Now let's go one level deeper. There are three fundamentally different ways a machine can learn. By the end of this class, you should be able to look at a problem and say "that's supervised learning" or "that's unsupervised." That skill is worth more than you think.

**Visual:** Nested circles from earlier, with an arrow pointing into the ML circle saying "Now: inside here."

---

### Slide 17 — The Three Types at a Glance
**Title:** Three Ways Machines Learn
- **Supervised Learning:** Learn from labeled examples (teacher gives answers)
- **Unsupervised Learning:** Find patterns in unlabeled data (no teacher)
- **Reinforcement Learning:** Learn by trial and error (rewards and penalties)
- Each type fits different problems — there's no "best" one

**Speaker notes:** Here's the big picture. Supervised learning is like studying with an answer key — the data comes with correct answers and the model learns to predict them. Unsupervised is like being dropped in a foreign city with no map — you start finding patterns and grouping things on your own. Reinforcement learning is like training a dog — good behavior gets a treat, bad behavior gets nothing.

**Visual:** Three-column layout, each with an analogy icon: teacher with answer key, explorer with magnifying glass, dog with treat.

---

### Slide 18 — Supervised Learning: The Details
**Title:** Supervised Learning — Learning with Labels
- Training data includes input-output pairs (features → label)
- Model learns the mapping: given these inputs, predict this output
- Two sub-types: Classification (categories) and Regression (numbers)
- Most common type in industry — ~70% of ML applications

**Speaker notes:** Supervised learning is the workhorse of ML. You give it examples where you already know the answer, and it learns the pattern. Like showing a child hundreds of photos of cats and dogs with labels. After enough examples, the child can identify new animals they've never seen. There are two flavors — classification when you're predicting a category, regression when you're predicting a number.

**Visual:** Diagram showing training data (table with features + label column) → ML model → predictions on new data.

---

### Slide 19 — Supervised Learning Examples
**Title:** Supervised Learning in Action
- **Classification:** Email spam or not spam? Customer will churn or stay?
- **Regression:** What will the house price be? How much revenue next quarter?
- Uzbekistan example: Uztelecom predicting which customers might cancel their plan
- Uzbekistan example: Classifying customer complaints by category

**Speaker notes:** Classification means you're sorting things into buckets — spam vs not spam, fraud vs legitimate, will churn vs won't churn. Regression means you're predicting a number on a continuous scale — price, temperature, revenue. At Uztelecom, we use classification to sort customer messages into categories so agents know how to respond faster.

**Visual:** Two-column split: Classification (discrete buckets icon) vs Regression (continuous line graph icon). Uzbekistan examples highlighted.

---

### Slide 20 — Unsupervised Learning: The Details
**Title:** Unsupervised Learning — Finding Hidden Structure
- No labels in the data — model discovers patterns on its own
- Main tasks: Clustering (grouping similar items) and Dimensionality Reduction
- Useful for exploration — "What groups exist in this data?"
- Harder to evaluate — no "right answer" to compare against

**Speaker notes:** Unsupervised learning is trickier. You hand the model a pile of data with no labels and say "find me something interesting." The most common task is clustering — grouping similar things together. Imagine you dump all Uztelecom customers' usage data into a model with no labels. It might discover natural groups — heavy data users, voice-only users, international callers. You didn't tell it those groups exist. It found them.

**Visual:** Scatter plot showing unlabeled data points → same plot with colored clusters after unsupervised learning.

---

### Slide 21 — Unsupervised Learning Examples
**Title:** Unsupervised Learning in Action
- **Customer segmentation:** Group users by behavior (no predefined categories)
- **Anomaly detection:** Find unusual transactions in banking data
- **Topic modeling:** Discover themes in thousands of customer reviews
- Uzbekistan example: Grouping Uzum marketplace shoppers by purchasing patterns

**Speaker notes:** Customer segmentation is the classic use case. You don't decide the segments in advance — the algorithm finds them. Anomaly detection is another big one — a bank doesn't label every transaction as fraud or not. Instead, the model learns what "normal" looks like and flags anything weird. In Uzbekistan, online marketplaces like Uzum could use this to understand their customer base without manually categorizing everyone.

**Visual:** Three real-world icons: customer groups, suspicious transaction alert, word cloud from reviews.

---

### Slide 22 — Reinforcement Learning: The Details
**Title:** Reinforcement Learning — Learning by Doing
- Agent interacts with an environment, takes actions, gets rewards
- Goal: Maximize cumulative reward over time
- No labeled data needed — learns from experience
- Best for sequential decision-making problems

**Speaker notes:** Reinforcement learning is the most different from the other two. There's an agent — think of it as a player in a game. It takes actions, and the environment gives it a reward or a penalty. Over thousands of tries, it figures out the best strategy. This is how AlphaGo learned to beat the world champion — by playing millions of games against itself.

**Visual:** Loop diagram: Agent → Action → Environment → Reward → Agent (cycle).

---

### Slide 23 — Reinforcement Learning Examples
**Title:** Reinforcement Learning in Action
- Game playing: AlphaGo, Chess engines, Atari games
- Robotics: Robot learning to walk, pick up objects
- Recommendations: Netflix/YouTube optimizing what to show next
- Uzbekistan example: Optimizing traffic light timing in Tashkent intersections

**Speaker notes:** RL is behind some of the most impressive AI demos — game-playing agents, robots learning to walk. But it's also in everyday apps. YouTube's recommendation system uses RL to maximize your watch time. A practical Uzbekistan application: imagine RL optimizing traffic signals in Tashkent — the agent learns timing patterns that minimize congestion. That's a real problem worth solving.

**Visual:** Four quadrant grid with an icon and example for each use case.

---

### Slide 24 — Comparison Table
**Title:** Side-by-Side: Three Types of ML

| | Supervised | Unsupervised | Reinforcement |
|---|---|---|---|
| Data | Labeled | Unlabeled | No dataset — environment |
| Goal | Predict | Discover | Optimize |
| Feedback | Correct answers | None | Rewards/penalties |
| Example | Spam detection | Customer segmentation | Game AI |
| Difficulty | Easier to evaluate | Harder to evaluate | Hardest to train |

**Speaker notes:** Here's your cheat sheet. Keep this table in mind — it'll save you a lot of confusion. The key difference is the feedback loop. Supervised has explicit correct answers. Unsupervised has no feedback at all. Reinforcement has delayed feedback through rewards. Most of your work in this course will be supervised and unsupervised.

**Visual:** Clean comparison table, color-coded columns.

---

### Slide 25 — Decision Flowchart: Which ML Type?
**Title:** How to Choose the Right ML Approach
- Do you have labeled data? → Yes → **Supervised**
  - Predicting a category? → **Classification**
  - Predicting a number? → **Regression**
- Do you have unlabeled data? → Yes → **Unsupervised**
  - Looking for groups? → **Clustering**
  - Reducing complexity? → **Dimensionality Reduction**
- Is this a sequential decision problem? → **Reinforcement Learning**

**Speaker notes:** This flowchart is your decision-making tool. When you face an ML problem, start here. First question: do I have labels? If yes, supervised. If no, are you looking for structure or making sequential decisions? This seems simple, but trust me — even experienced practitioners use this mental model.

**Visual:** Flowchart diagram with decision diamonds and outcome boxes. Color-coded paths matching the comparison table.

---

### Slide 26 — [ACTIVITY] ML Type Classification Exercise
**Title:** Activity 1: Classify the Problem
- I'll describe 8 real-world scenarios
- For each one, decide: Supervised, Unsupervised, or Reinforcement?
- If supervised, is it Classification or Regression?
- Work in pairs, 7 minutes, then we discuss

> [ACTIVITY] Professor's Activity 1 — ML Type Classification Exercise

**Speaker notes:** Time to test your understanding. I'm going to give you eight scenarios — some from Uzbekistan, some international. With your partner, classify each one. Don't overthink it — use the flowchart. We'll discuss the tricky ones together afterward.

**Visual:** Activity instruction card. List of 8 scenarios on a handout or projected.

**Scenarios:**
1. Predicting house prices in Tashkent based on area, rooms, district
2. Grouping mobile subscribers by usage patterns
3. Teaching a robot to navigate a warehouse
4. Detecting fraudulent bank transactions
5. Recommending products on Uzum based on browsing history
6. Predicting next month's electricity demand in Samarkand
7. Finding topics in thousands of customer support tickets
8. A self-driving car learning to stay in its lane

---

### Slide 27 — Activity Debrief
**Title:** Let's Review the Answers
- Walk through each scenario with class input
- Highlight edge cases — some problems can use multiple approaches
- Key insight: The same problem can sometimes be framed differently

> **Ask the class:** "Which ones were tricky? Did any pair disagree?"

**Speaker notes:** Let's go through these. I'll ask a pair for each one. For the tricky ones — like recommendations or fraud — there are actually multiple valid approaches. Fraud can be supervised (if you have labeled fraud data) or unsupervised (anomaly detection). That's an important lesson: problem framing matters as much as the algorithm.

**Visual:** Answers revealed one by one, with the correct ML type highlighted.

---

### Slide 28 — Semi-Supervised and Self-Supervised (Bonus)
**Title:** Bonus: Other Learning Paradigms
- **Semi-supervised:** Small amount of labeled data + large amount of unlabeled
- **Self-supervised:** Model creates its own labels from the data (e.g., predict the next word)
- ChatGPT uses self-supervised learning on massive text data
- These are growing in importance — less labeling needed

**Speaker notes:** Quick bonus before we close. The three types we covered are the classic categories, but there are hybrid approaches. Semi-supervised mixes labeled and unlabeled data — useful when labeling is expensive. Self-supervised is how GPT models work — the model predicts the next word, so the text itself becomes the label. You'll see more of this in later modules.

**Visual:** Spectrum diagram: Supervised → Semi-supervised → Self-supervised → Unsupervised.

---

### Slide 29 — Key Takeaways
**Title:** What We Covered Today
- We saw a working ML model before defining anything — now you know what we're building toward
- AI is a broad field; ML and DL are specific approaches within it
- AI has a 70-year history with multiple boom-bust cycles
- Three types of ML: Supervised (labeled), Unsupervised (unlabeled), Reinforcement (rewards)
- Problem framing determines which approach you use
- Next class: Mathematical foundations — probability, statistics, and the ML workflow

**Speaker notes:** Big class today. We started by seeing ML in action, then covered the full landscape — what AI is, where it came from, how ML fits in, and the three ways machines learn. The skill you're building is learning to look at a problem and immediately know which bucket it falls into. Next time: the math intuition behind all of this, and the end-to-end ML workflow.

**Visual:** Summary icons for demo + AI history + ML types. Arrow to "Next: Foundations."

---

### Slide 30 — Homework
**Title:** Before Next Class
- Read: "A Brief History of AI" (linked article, ~15 min)
- Find: 3 examples of ML being used in Uzbekistan (any sector)
- For each example: classify as supervised, unsupervised, or reinforcement
- Review the comparison table and flowchart (study them)

**Speaker notes:** Read the article I'll share. Then find three real examples of ML in Uzbekistan and classify each one using what we learned today. Could be banking, telecom, agriculture, government. Bonus points if you find one that could be approached from two different angles.

**Visual:** Checklist format with due date.

---
---

## CLASS 2: Foundations — Probability, Stats, ML Workflow (1.5h)

### Slide 1 — Title Slide
**Title:** Mathematical Foundations and the ML Workflow
- Module 1, Class 2
- Probability and statistics — intuition, not formulas
- The end-to-end ML workflow: from problem to deployment

**Speaker notes:** Today might sound scary — math. But I promise, no formulas on the board today. We're building intuition. You need to understand why probability and statistics matter for ML, and how an ML project actually flows from start to finish. By the end, you'll have a mental map of the entire ML process.

**Visual:** Title card. Split visual: dice/charts on one side, workflow pipeline on the other.

---

### Slide 2 — Why Math Matters for ML
**Title:** You Don't Need to Be a Mathematician, But...
- ML is built on probability and statistics — you need the intuition
- Understanding the "why" behind models prevents blind usage
- Goal today: Understand concepts, not memorize formulas
- Module 2 will go deeper — today is the appetizer

**Speaker notes:** Let me be honest — you don't need a PhD in math to do ML. But you do need to understand what's happening under the hood at a basic level. Otherwise you're just pressing buttons and hoping for the best. Today is about building that intuition. We'll go deeper in Module 2.

**Visual:** Iceberg diagram — "What you see: predictions" above water, "What's underneath: probability, statistics, optimization" below water.

---

### Slide 3 — Probability: The Coin Flip
**Title:** Probability — How Likely Is It?
- Probability = how likely something is to happen (0 to 1)
- Coin flip: 50% heads, 50% tails — that's probability
- ML models output probabilities: "This email is 87% likely to be spam"
- The model doesn't say "yes" or "no" — it says "how confident am I?"

**Speaker notes:** Let's start with the simplest possible example. A coin flip. You know there's a 50% chance of heads. That's probability. ML models work the same way — they don't give you a hard yes or no. They give you a probability. "I'm 87% sure this is spam." Then you decide the threshold — maybe anything above 70% gets filtered. That threshold is your decision, not the model's.

**Visual:** Coin flip animation concept. Then a spam email with a confidence meter showing 87%.

---

### Slide 4 — Probability: The Dice Analogy
**Title:** Probability Distributions — What Could Happen?
- Roll a fair die: each face has equal probability (1 in 6)
- That's a **uniform distribution** — all outcomes equally likely
- Customer call duration: most calls are short, few are very long
- That's a **skewed distribution** — some outcomes much more common

**Speaker notes:** A die is a uniform distribution — every outcome is equally likely. But most real-world data isn't uniform. Think about how long customer calls last at a telecom company. Most calls are under 5 minutes, some are 10-15, and a few outliers go over an hour. That shape — tall on the left, trailing off to the right — is a distribution. Understanding the shape of your data is the first step to building good models.

**Visual:** Side by side: flat bar chart (die, uniform) vs right-skewed curve (call duration).

---

### Slide 5 — Statistics: Mean, Median, Spread
**Title:** Describing Data — Central Tendency and Spread
- **Mean:** Average value — sensitive to outliers
- **Median:** Middle value — robust to outliers
- **Standard deviation:** How spread out the data is
- Analogy: Average salary in Uzbekistan vs median salary — very different stories

**Speaker notes:** If I tell you the average salary in a company is 15 million soʻm, that sounds decent. But what if the CEO makes 500 million and everyone else makes 5 million? The average is misleading. The median — the middle person's salary — tells the real story. Standard deviation tells you how spread out the values are. These three numbers are your first tools for understanding any dataset.

**Visual:** Salary distribution with mean and median marked — showing how an outlier CEO pulls the mean right while median stays put.

---

### Slide 6 — Correlation: Things That Move Together
**Title:** Correlation — Not Causation
- Correlation: Two things tend to move together
- Temperature goes up → ice cream sales go up (positive correlation)
- Correlation does NOT mean causation
- Classic trap: Ice cream sales and drowning deaths both rise in summer

**Speaker notes:** Correlation measures whether two things move together. Temperature and ice cream sales are correlated. But here's the trap that catches even professionals — correlation is not causation. Ice cream sales and drowning deaths are correlated. Does ice cream cause drowning? No. Summer causes both. In ML, you'll find lots of correlations in your data. Your job is to think critically about which ones are meaningful.

**Visual:** Scatter plot showing positive correlation. Then a humorous "ice cream causes drowning?" diagram with the hidden variable (summer) revealed.

---

### Slide 7 — Gradient Descent: Walking Downhill
**Title:** How Models Learn — The Downhill Walk
- Imagine you're blindfolded on a hilly landscape
- Goal: Find the lowest point (minimum error)
- Strategy: Feel the slope under your feet, step downhill
- Repeat until you can't go any lower — that's gradient descent
- ML models do this with math, thousands of times per second

**Speaker notes:** This is how ML models actually learn. Imagine you're blindfolded on a mountain and you need to find the valley. What do you do? You feel which direction slopes downward and take a step. Then you check again and take another step. That's gradient descent. The "mountain" is the error landscape — the model is trying to minimize its mistakes. Every step is an adjustment to the model's parameters. It does this millions of times until it settles at a low point.

**Visual:** 3D surface plot (like a mountain landscape) with a ball rolling downhill toward the minimum. Arrow showing the path of gradient descent.

---

### Slide 8 — The ML Workflow: Big Picture
**Title:** The End-to-End ML Workflow
- Problem Definition → Data Collection → Data Preparation → Modeling → Evaluation → Deployment
- It's not linear — you loop back constantly
- Most time spent on data (cleaning, preparing) — not on the model
- The model is maybe 10-20% of the work

**Speaker notes:** Here's the full picture of how an ML project works. It starts with defining the problem, then collecting and cleaning data, then building a model, evaluating it, and deploying it. The dirty secret of ML — you'll spend 80% of your time on data, not models. And it's not a straight line. You'll build a model, realize your data is bad, go back and fix it, retrain, evaluate, repeat. That's normal.

**Visual:** Circular/iterative workflow diagram with all 6 steps. Arrows going forward AND backward to show the iterative nature. "Data" steps highlighted as the largest portion.

---

### Slide 9 — Step 1: Problem Framing
**Title:** Start with the Right Question
- "We want to use AI" is NOT a problem statement
- Good: "We want to predict which customers will cancel next month"
- Define: What are you predicting? What data do you have? What does success look like?
- Bad framing = wasted months of work

**Speaker notes:** The most important step is the one most people skip. Before you touch any data or code, you need to define the problem precisely. "We want to use AI" — I hear this constantly, and it means nothing. Compare that to "We want to predict which of our 500,000 subscribers will cancel in the next 30 days so we can offer them a retention deal." Now that's something you can build.

**Visual:** Bad problem statement (vague, crossed out) vs Good problem statement (specific, checkmarked).

---

### Slide 10 — Steps 2-3: Data Collection and Preparation
**Title:** Data — The Fuel for ML
- Collection: Where does the data come from? Databases, APIs, sensors, surveys?
- Preparation: Cleaning missing values, fixing errors, formatting
- Feature engineering: Creating useful inputs from raw data
- Reality: This is where 60-80% of project time goes

**Speaker notes:** Once you know the problem, you need data. And real data is messy. Missing values, duplicate records, wrong formats, inconsistent naming. At Uztelecom, customer data might have phone numbers in three different formats across three different systems. You have to clean all of that before any model can learn from it. Module 3 is entirely about this.

**Visual:** Funnel diagram: Raw Data (messy) → Cleaning → Feature Engineering → Clean Dataset (structured table).

---

### Slide 11 — Steps 4-5: Modeling and Evaluation
**Title:** Build It, Then Prove It Works
- Modeling: Choose an algorithm, train it on your data
- Start simple — a basic model often beats a complex one
- Evaluation: How do you measure success?
  - Accuracy, precision, recall, F1 score (we'll learn these in Module 4)
- Key rule: Always test on data the model has never seen

**Speaker notes:** Now the fun part — building the model. But here's a pro tip: start with the simplest possible model. A basic logistic regression often outperforms a fancy neural network, especially with small data. Then evaluate — and here's the critical thing — you must test on data the model has NEVER seen during training. Otherwise you're just testing memory, not learning.

**Visual:** Split: Training data going into model (left), Test data being used for evaluation with a score card (right). Wall between them labeled "Never mix these."

---

### Slide 12 — Step 6: Deployment and Monitoring
**Title:** A Model in a Notebook Is Useless
- Deployment: Put the model into a real system where it makes predictions
- The model needs to work in real-time, not just on your laptop
- Monitoring: Models degrade over time — data changes, patterns shift
- You need to retrain and update regularly

**Speaker notes:** A model sitting in a Jupyter notebook is a science experiment. A model running in production, making real predictions, affecting real decisions — that's engineering. Deployment is where most projects die. And even after deployment, you're not done. The world changes. Customer behavior shifts. Your model gets stale. You need monitoring and retraining. This is the MLOps side — which is a growing field.

**Visual:** Laptop with notebook → Arrow → Cloud/server → Arrow → Mobile app/dashboard showing predictions. Monitoring loop underneath.

---

### Slide 13 — Putting It All Together: Churn Example
**Title:** ML Workflow Applied: Telecom Churn Prediction
- **Problem:** Predict which subscribers will cancel next month
- **Data:** Call records, billing history, complaints, plan type
- **Preparation:** Clean missing values, create features like "days since last complaint"
- **Model:** Train classification model (supervised learning)
- **Evaluate:** Test on last month's actual churners
- **Deploy:** Flag at-risk customers for the retention team

**Speaker notes:** Let's walk through a real example end to end. Uztelecom wants to reduce customer churn. Step by step — define the problem, gather the data from billing and CRM systems, clean it up, engineer useful features, train a model, test it against reality, and deploy it so the retention team gets a daily list of at-risk customers. That's the full workflow in action. You'll actually frame this problem yourself in the lab session.

**Visual:** The workflow diagram from Slide 8, but each step now filled in with the churn example specifics.

---

### Slide 14 — Discussion: Frame a Problem
**Title:** Your Turn: Frame an ML Problem

> **Ask the class:** "Pick a local business or organization. What problem could ML solve for them? Walk through the first three steps of the workflow."

- Think about: What would you predict? Where would the data come from?
- 3 minutes with your neighbor, then share
- No wrong answers — we're practicing the thinking process

**Speaker notes:** Quick exercise. Think of any business in Uzbekistan — a restaurant, a bank, a university, a delivery service. What's a problem ML could help with? Don't worry about the model — focus on the problem definition and where the data would come from. Talk to your neighbor for 3 minutes. Let's hear some creative ideas.

**Visual:** Thought bubble icons with local business logos (bazaar, bank, university).

---

### Slide 15 — Key Takeaways
**Title:** What We Covered Today
- Probability and statistics give ML its foundation — intuition over formulas
- Gradient descent: models learn by walking downhill on the error landscape
- The ML workflow: Problem → Data → Preparation → Model → Evaluate → Deploy
- Most time is spent on data, not models
- Next class: Real-world ITES applications deep-dive

**Speaker notes:** Two big things from today. First, the math behind ML isn't scary — it's probability, statistics, and optimization, and you can understand the intuition without memorizing formulas. Second, the ML workflow is your roadmap for any project. Memorize those six steps. Next class we'll look at real applications in the ITES sector — recommendation systems, fraud detection, chatbots, churn prediction.

**Visual:** Recap slide with workflow diagram thumbnail and key stats icons.

---

### Slide 16 — Homework
**Title:** Before Next Class
- Review the ML workflow steps — be able to list them from memory
- Read the assigned article on ML in telecom/ITES
- Start thinking about Activity 3: What data would you need for churn prediction?

**Speaker notes:** Homework is light. Know the six workflow steps cold. Read the article I'll share on ML in telecom. And start thinking about churn prediction — what data would you need? What would you predict? We'll use this in the lab session at the end of the module.

**Visual:** Checklist format.

---
---

## CLASS 3: ITES Applications Deep-Dive (1.5h)

### Slide 1 — Title Slide
**Title:** AI/ML in the Real World — ITES Applications
- Module 1, Class 3
- Four case studies: Recommendations, Fraud Detection, Chatbots, Churn Prediction
- How it works, why it matters, what can go wrong

**Speaker notes:** Today is case study day. We're looking at four major applications of ML in the ITES sector — information technology enabled services. These are the systems running behind the scenes at banks, telecom companies, e-commerce platforms, and call centers. For each one, we'll cover how it works, why it matters for business, and what challenges come with it.

**Visual:** Four icons in a grid: shopping cart (recommendations), shield (fraud), chat bubble (chatbots), broken chain (churn).

---

### Slide 2 — Case Study 1: Recommendation Systems
**Title:** How Netflix, YouTube, and Uzum Know What You Want
- **How it works:** Collaborative filtering — "users like you also liked..."
- Content-based filtering — "because you watched/bought this..."
- Hybrid approaches combine both
- Uses: E-commerce, streaming, news feeds, music

**Speaker notes:** Every time YouTube suggests a video or Uzum shows you "recommended for you" products, that's a recommendation system. There are two main approaches. Collaborative filtering says "people similar to you bought these things." Content-based says "this product is similar to things you already liked." Most real systems use both. It's one of the most commercially valuable applications of ML.

**Visual:** Diagram showing two paths: User → Similar Users → Their Products (collaborative) and User → Their Products → Similar Products (content-based). Both paths merge.

---

### Slide 3 — Recommendations: Why It Matters + Challenges
**Title:** Recommendations — Business Impact and Challenges
- **Why it matters:** Amazon — 35% of revenue from recommendations. Netflix — saves $1B/year in retention
- **Uzbekistan angle:** Uzum, Korzinka Go, local streaming platforms
- **Challenges:**
  - Cold start problem — new users have no history
  - Filter bubbles — users only see what they already like
  - Data sparsity — most users interact with a tiny fraction of products

**Speaker notes:** The business impact is massive. A third of Amazon's revenue comes from "you might also like." Netflix estimates their recommendation engine saves a billion dollars a year in customer retention. In Uzbekistan, Uzum marketplace and Korzinka Go delivery could use the same technology. But it's not perfect — new users have no history, and there's a risk of creating filter bubbles where people only see what they already agree with.

**Visual:** Revenue impact stat callouts. Challenge cards below.

---

### Slide 4 — Case Study 2: Fraud Detection
**Title:** Catching Criminals with Machine Learning
- **How it works:** Model learns patterns of normal transactions
- Flags transactions that deviate significantly — anomaly detection
- Can also use supervised learning if you have labeled fraud data
- Real-time decisions needed — milliseconds to approve or block

**Speaker notes:** Banks process millions of transactions daily. Humans can't review each one. ML models learn what normal transactions look like for each customer — amount, location, time of day, merchant type. When something doesn't fit the pattern — like a large purchase in a foreign country at 3 AM — the system flags it. Some systems use supervised learning with labeled fraud/not-fraud data. Others use unsupervised anomaly detection. Often both.

**Visual:** Transaction flow diagram: Customer swipes card → ML model checks in milliseconds → Approve or Flag.

---

### Slide 5 — Fraud Detection: Why It Matters + Challenges
**Title:** Fraud Detection — Business Impact and Challenges
- **Why it matters:** Global card fraud exceeded $30B in 2023
- **Uzbekistan angle:** Growing digital payments (Payme, Click, Uzum Bank)
- **Challenges:**
  - Class imbalance — fraud is rare (0.1% of transactions), hard to learn from
  - False positives — blocking legitimate transactions frustrates customers
  - Adversarial attacks — fraudsters adapt to detection methods
  - Privacy — models need access to sensitive financial data

**Speaker notes:** As Uzbekistan moves toward digital payments — Payme, Click, Uzum Bank — fraud detection becomes critical. The biggest challenge is imbalance. Out of a million transactions, maybe a thousand are fraud. The model sees 99.9% normal data. It's like finding a needle in a haystack. And the needle keeps changing shape — fraudsters evolve their tactics. We'll learn about handling imbalanced data in Module 3.

**Visual:** Scales showing imbalance: huge pile of normal transactions vs tiny pile of fraud. Adaptation cycle: fraud detected → fraudster adapts → new fraud pattern.

---

### Slide 6 — Case Study 3: AI Chatbots
**Title:** Chatbots — Automating Customer Support
- **How it works:** NLP (Natural Language Processing) to understand user intent
- Match intent to a predefined answer or generate a response
- Simple chatbots: Rule-based decision trees
- Advanced chatbots: LLMs (like ChatGPT) for open-ended conversation

**Speaker notes:** Chatbots are the application I know best — we're building one at Uztelecom. A customer types "How do I check my balance?" and the bot needs to understand the intent — it's about balance checking, not billing or complaints. Simple bots use rules and keyword matching. Advanced bots use NLP and large language models to understand meaning even when the phrasing is unusual. The goal: handle routine questions automatically so human agents focus on complex issues.

**Visual:** Chat interface mockup showing a customer asking a question in Uzbek and the bot responding. Behind the interface: NLP pipeline (text → intent recognition → response selection).

---

### Slide 7 — Chatbots: Why It Matters + Challenges
**Title:** Chatbots — Business Impact and Challenges
- **Why it matters:** 24/7 availability, handles 60-80% of routine queries, reduces cost per interaction
- **Uzbekistan angle:** Uztelecom AI chatbot, banking bots, government e-services
- **Challenges:**
  - Understanding Uzbek language (Latin + Cyrillic, code-switching)
  - Handling ambiguous or emotional customer messages
  - Knowing when to escalate to a human agent
  - Maintaining accuracy — wrong answers erode trust fast

**Speaker notes:** A good chatbot handles the easy stuff — balance inquiries, plan details, service activation — so your human agents can focus on complaints and complex issues. At Uztelecom, the chatbot handles a significant portion of incoming queries. But Uzbek language presents unique challenges — people switch between Latin and Cyrillic, mix in Russian, use slang. And the bot needs to know its limits. Confidently giving a wrong answer is worse than saying "let me connect you to an agent."

**Visual:** Impact stats (queries handled, cost reduction). Challenge cards with Uzbek language examples.

---

### Slide 8 — Case Study 4: Churn Prediction
**Title:** Predicting Customer Churn — Before They Leave
- **How it works:** Supervised classification — will this customer churn (yes/no)?
- Features: Usage trends, billing patterns, complaint history, tenure
- Model assigns a churn probability to each customer
- Business action: Proactive retention offers to high-risk customers

**Speaker notes:** Churn prediction is the case study we'll work with most in this module. It's supervised classification — you take historical data on customers who left and who stayed, and the model learns the patterns. Declining usage, increasing complaints, payment delays — these are signals. The output is a probability. "Customer X has an 82% chance of churning next month." The retention team then reaches out before it happens. Prevention is cheaper than acquisition.

**Visual:** Customer journey diagram: Active → Warning signals (declining usage, complaints) → Churn risk score → Retention intervention.

---

### Slide 9 — Churn Prediction: Why It Matters + Challenges
**Title:** Churn Prediction — Business Impact and Challenges
- **Why it matters:** Acquiring a new customer costs 5-7x more than retaining one
- **Uzbekistan angle:** Uztelecom, Beeline, Ucell — competitive telecom market
- **Challenges:**
  - Defining "churn" — canceled? Inactive for 30 days? Reduced usage?
  - Data quality — customer records often incomplete or messy
  - Actionability — prediction is useless without a retention strategy
  - Model fairness — don't discriminate against certain customer groups

**Speaker notes:** In Uzbekistan's telecom market, customers can switch carriers easily. Losing a customer means lost revenue and the cost of acquiring a replacement. The first challenge is surprisingly basic — defining what churn means. Does a customer who reduces their plan count? What about someone who hasn't called in 30 days but still pays? These definitions change the entire model. And prediction without action is pointless — you need a retention team ready to act on the model's output.

**Visual:** Cost comparison: retention ($1) vs acquisition ($5-7). Decision tree for churn definition.

---

### Slide 10 — [ACTIVITY] ITES Case Study Discussion
**Title:** Activity 2: Pick a Case Study, Go Deeper
- Form 4 groups — each group gets one case study
- For your case study, discuss:
  1. What data would you need?
  2. What ML type would you use?
  3. Name one Uzbekistan company that could use this
  4. What's the biggest risk if the model is wrong?
- 8 minutes discussion, 3 minutes per group to present

> [ACTIVITY] Professor's Activity 2 — ITES Case Study

**Speaker notes:** Split into four groups. Each group takes one of the four case studies we just covered. Spend 8 minutes discussing those four questions. Then each group presents for 3 minutes. I want to hear specific Uzbekistan companies and specific risks. "The model could be wrong" isn't enough — tell me what happens to the business when it's wrong.

**Visual:** Group activity instruction card with the four questions.

---

### Slide 11 — Cross-Cutting Themes
**Title:** What All Four Case Studies Have in Common
- All need quality data — garbage in, garbage out
- All require careful problem framing before modeling
- All have ethical implications (privacy, fairness, transparency)
- All need human oversight — ML assists decisions, doesn't replace judgment
- All benefit from domain knowledge — knowing the business matters

**Speaker notes:** Looking across all four case studies, some themes keep coming up. Data quality is everything. Problem framing matters more than algorithm choice. Ethics can't be an afterthought. Humans need to stay in the loop. And domain expertise — actually understanding the business — is what separates a useful model from a toy. Keep these themes in mind for the rest of the course.

**Visual:** Central node "ML Applications" with five spokes: Data Quality, Problem Framing, Ethics, Human Oversight, Domain Knowledge.

---

### Slide 12 — Key Takeaways
**Title:** What We Covered Today
- Four major ITES applications: recommendations, fraud, chatbots, churn
- Each has a clear business case AND real challenges
- ML in Uzbekistan is growing — these applications are already here
- The "how it works" matters less than "what can go wrong"
- Next class: Ethics in AI — bias, fairness, and privacy

**Speaker notes:** Four applications, four different problem types, four sets of challenges. The technical "how" is important, but the "what could go wrong" is what separates a responsible AI practitioner from a reckless one. Next class, we go deeper into the ethics side — bias in training data, fairness in predictions, and privacy concerns. Bring your thinking caps.

**Visual:** Four case study icons with checkmarks. Arrow to "Next: Ethics."

---

### Slide 13 — Homework
**Title:** Before Next Class
- Pick one of the four case studies and write a one-page summary:
  - How it works (your own words)
  - One Uzbekistan application
  - Two potential risks
- Read the assigned article on AI ethics (linked)
- Come ready to discuss: "Should AI make decisions about people?"

**Speaker notes:** One page, your own words. Pick whichever case study interests you most. And read the ethics article before next class — we'll jump straight into discussion. The question to think about: should AI systems make decisions that affect people's lives? If yes, under what conditions? If no, why not?

**Visual:** Checklist with reading link.

---
---

## CLASS 4: Ethics in AI — Case Study Discussion (1.5h)

### Slide 1 — Title Slide
**Title:** Ethics in AI — Bias, Fairness, and Privacy
- Module 1, Class 4
- No right answers today — just important questions
- Interactive session: case studies and group discussion

**Speaker notes:** Today is different. Less lecturing, more discussing. AI ethics isn't a box you check — it's a set of questions you need to ask every time you build or deploy a system. We'll look at real cases where AI went wrong, discuss what happened and why, and think about what we'd do differently. I want to hear your opinions.

**Visual:** Title card with scales of justice + AI circuit brain icon.

---

### Slide 2 — Why Ethics in AI Matters
**Title:** The Power to Help or Harm
- AI systems make decisions affecting millions of people
- Hiring algorithms, loan approvals, criminal sentencing, medical diagnosis
- These systems can amplify existing inequalities
- "Move fast and break things" doesn't work when you're breaking people's lives

**Speaker notes:** When you build a spam filter that makes a mistake, an email goes to the wrong folder. Annoying, not life-changing. But when an AI system denies someone a loan, rejects their job application, or flags them as a criminal suspect — that's a different level of consequence. AI systems can scale discrimination at a speed and scope no human could match. That's why ethics isn't optional.

**Visual:** Gradient of impact: Low stakes (spam filter, recommendations) → High stakes (hiring, loans, criminal justice, healthcare). Color shifts from green to red.

---

### Slide 3 — Bias in AI: Where Does It Come From?
**Title:** AI Bias — It Starts with Us
- **Data bias:** Training data reflects historical discrimination
- **Selection bias:** Data doesn't represent the full population
- **Measurement bias:** What we measure isn't what we think we're measuring
- **Algorithmic bias:** Model design choices favor certain outcomes
- AI doesn't create bias — it inherits and amplifies it

**Speaker notes:** Here's the uncomfortable truth: AI bias comes from us. If your training data reflects a world where women were historically denied loans, the model learns that pattern and continues it. If your dataset only includes data from Tashkent, it won't work for Karakalpakstan. The model is a mirror — it reflects whatever you show it, including the ugly parts.

**Visual:** Pipeline diagram: Biased World → Biased Data → Biased Model → Biased Decisions → Reinforces Biased World (cycle).

---

### Slide 4 — Case Study: Amazon's Hiring Algorithm
**Title:** Case Study — Amazon's Resume Screening Tool
- Amazon built an AI to screen job applicants
- Trained on 10 years of hiring data — mostly male hires in tech
- Model learned to penalize resumes containing the word "women's"
- Downgraded graduates of all-women's colleges
- Amazon scrapped the tool in 2018

> **Ask the class:** "Whose fault is this? The algorithm? The data? The engineers? The company?"

**Speaker notes:** This is one of the most famous AI bias cases. Amazon trained a resume screener on a decade of their own hiring decisions. But those decisions were biased — they'd historically hired mostly men for technical roles. So the model learned that being female was a negative signal. It literally penalized resumes that mentioned women's organizations. They tried to fix it, couldn't guarantee fairness, and killed the project. The lesson: biased history creates biased AI.

**Visual:** Resume document with "women's chess club" highlighted and a red "penalized" stamp. Timeline of events.

---

### Slide 5 — Case Study: COMPAS Criminal Justice
**Title:** Case Study — Predicting Who Will Re-Offend
- COMPAS: Algorithm used in US courts to predict recidivism risk
- ProPublica investigation (2016) found racial bias
- Black defendants were twice as likely to be falsely flagged as high-risk
- White defendants were more likely to be falsely labeled as low-risk
- The algorithm's developers said it was "equally accurate" across races

> **Think-pair-share:** "Can an algorithm be equally accurate but still unfair? How?"

**Speaker notes:** This case will make you think. COMPAS was used by judges to help decide bail and sentencing. ProPublica found it was twice as likely to falsely label Black defendants as high-risk. The company said the overall accuracy was the same across races. Both statements were technically true — but they were measuring different things. This reveals something deep: fairness has multiple definitions, and they can contradict each other. There's no purely mathematical solution to fairness.

**Visual:** Side-by-side error rates by race. Highlight the different types of errors (false positives vs false negatives).

---

### Slide 6 — Fairness: Multiple Definitions
**Title:** What Does "Fair" Even Mean?
- **Equal accuracy:** Same overall accuracy across groups
- **Equal error rates:** Same false positive/negative rates across groups
- **Demographic parity:** Same proportion of positive outcomes across groups
- Mathematical proof: You cannot satisfy all fairness criteria simultaneously
- Choosing which fairness metric = a human value judgment, not a technical one

**Speaker notes:** This is the hardest slide in the whole module. There are multiple mathematical definitions of fairness, and — this has been mathematically proven — you cannot satisfy all of them at the same time. So you have to choose. And that choice isn't technical, it's ethical. Do you want equal accuracy or equal error rates? The answer depends on context, stakes, and values. This is why AI ethics needs diverse teams, not just engineers.

**Visual:** Three fairness definitions as overlapping circles that don't fully overlap — illustrating the impossibility of satisfying all simultaneously.

---

### Slide 7 — Privacy in AI
**Title:** AI and Your Data — Who Knows What?
- ML models need data — often personal data
- Facial recognition: Trained on billions of photos, often without consent
- Voice assistants: Recording conversations to improve models
- Data collection in Uzbekistan: What are people comfortable sharing?
- GDPR (Europe), CCPA (California) — regulation is catching up

**Speaker notes:** Every ML model we discussed this week needs data, and that data often comes from people. Your face, your voice, your purchase history, your location. Most people don't realize how much data is being collected or how it's being used. In Uzbekistan, digital services are growing fast — Payme, Uzum, government e-services. What data are they collecting? How is it stored? Who has access? These are questions every AI practitioner needs to ask.

**Visual:** Person silhouette with data streams flowing out: location, purchases, browsing, voice, face. Question marks over each stream.

---

### Slide 8 — Case Study: Facial Recognition Bias
**Title:** Case Study — Facial Recognition Doesn't See Everyone
- MIT study (2018): Commercial facial recognition systems tested
- Error rates: White men ~1%, darker-skinned women ~35%
- Cause: Training data dominated by lighter-skinned faces
- Impact: False arrests, surveillance targeting minorities
- Several US cities banned government use of facial recognition

> **Ask the class:** "Should Uzbekistan use facial recognition in public spaces? Under what conditions?"

**Speaker notes:** Facial recognition is the poster child for bias. MIT researcher Joy Buolamwini tested three commercial systems and found massive accuracy gaps — nearly perfect for white men, wrong a third of the time for darker-skinned women. This has real consequences. In the US, multiple people have been wrongly arrested based on faulty facial recognition. Some cities banned it entirely. What about Uzbekistan? As smart city projects expand, this is a conversation we need to have.

**Visual:** Accuracy comparison chart across demographic groups. Photos showing correct vs incorrect identifications.

---

### Slide 9 — [ACTIVITY] Group Discussion: Ethics Scenarios
**Title:** Group Discussion — What Would You Do?
- **Scenario A:** Your bank's loan model rejects more women than men. Accuracy is the same. What do you do?
- **Scenario B:** A school wants to use AI to predict which students will drop out. Good idea or bad idea?
- **Scenario C:** Government wants real-time facial recognition in Tashkent metro for security.
- **Scenario D:** A telecom uses customer data to predict churn and sells retention lists to third parties.

> [ACTIVITY] Ethics group discussion — 12 minutes in groups, then class-wide debrief

**Speaker notes:** Four scenarios, four groups. Each group takes one. No right answers — I want you to argue both sides. What are the benefits? What are the risks? What safeguards would you put in place? You have 12 minutes, then each group presents their position and the rest of the class challenges them. Be ready to defend your reasoning.

**Visual:** Four scenario cards, each color-coded. Timer visible.

---

### Slide 10 — Group Debrief
**Title:** What Did We Decide?
- Each group presents: 3 minutes per group
- Class challenges and asks questions
- Key themes to track: consent, transparency, accountability, alternatives

> **Ask the class:** "For each scenario, who should be held accountable if the AI causes harm?"

**Speaker notes:** Let's hear from each group. When a group presents, I want the rest of you thinking about these questions: Did they consider all stakeholders? What's the worst case scenario? And who is responsible if things go wrong — the developer, the company, the government, or the algorithm? Accountability is the question nobody wants to answer.

**Visual:** Discussion format slide — group numbers, time allocation, key questions listed.

---

### Slide 11 — Responsible AI Principles
**Title:** Building AI Responsibly — A Framework
- **Transparency:** People should know when AI is making decisions about them
- **Accountability:** Someone must be responsible when AI causes harm
- **Fairness:** Actively test for and mitigate bias
- **Privacy:** Collect only what you need, protect what you have
- **Human oversight:** Keep humans in the loop for high-stakes decisions

**Speaker notes:** After all that discussion, here's a framework. Five principles. They won't solve every problem, but they give you a starting point. Whenever you build an ML system, run it through these five checks. Is it transparent? Is someone accountable? Have we tested for bias? Are we protecting privacy? Is there a human in the loop? If you can answer yes to all five, you're ahead of most companies in the world.

**Visual:** Five principles as a pentagon/radar chart. Each principle with a one-line description.

---

### Slide 12 — AI Ethics in Uzbekistan's Context
**Title:** Ethics Considerations Specific to Uzbekistan
- Rapidly growing digital infrastructure — decisions made now set precedents
- Uzbek/Russian bilingual data — whose language gets priority?
- Rural vs urban digital divide — AI trained on Tashkent data won't work in Navoi
- Young population — most affected by AI decisions in education, jobs, services
- Opportunity: Uzbekistan can learn from other countries' mistakes

**Speaker notes:** Ethics isn't abstract — it's local. Uzbekistan is building digital infrastructure right now, which means the decisions made today will shape how AI affects people for decades. There are specific issues: the language divide between Uzbek and Russian, the urban-rural data gap, and a young population that will be most affected by these systems. The good news is we can learn from the mistakes the US and Europe already made. We don't have to repeat them.

**Visual:** Uzbekistan map with callouts for specific regional considerations.

---

### Slide 13 — Key Takeaways
**Title:** What We Covered Today
- AI bias comes from data and design choices — not from the algorithm being "evil"
- Fairness has multiple definitions that can conflict with each other
- Privacy and consent are foundational — not afterthoughts
- Real cases: Amazon hiring, COMPAS, facial recognition — all cautionary tales
- Your responsibility as a future practitioner: ask the hard questions
- Next class: Lab session — Activity 3 (churn framing) + Module Quiz

**Speaker notes:** Today's takeaway is simple: AI ethics is your responsibility. Not the government's, not your company's — yours. Every model you build reflects choices about who benefits and who gets harmed. The technical skills we're teaching you are powerful. Use them carefully. Next class is our lab — you'll frame a real ML problem and take the module quiz.

**Visual:** Key takeaway cards. Arrow pointing to "Next: Lab."

---

### Slide 14 — Homework (Prepare for Lab)
**Title:** Before the Lab Session
- Review the ML workflow (6 steps) — you'll need it
- Review all three ML types — quiz will cover them
- Think about churn prediction: What data? What features? What ML type?
- Bring a laptop if possible — or be ready to work in pairs

**Speaker notes:** Next class is the lab. Come prepared. You'll be doing Activity 3 — framing the telecom churn prediction problem from scratch. And there's a quiz at the end covering everything from Module 1. Review the ML types, the workflow, the ethics principles. If you paid attention these four classes, you'll be fine.

**Visual:** Preparation checklist.

---
---

## CLASS 5: Lab Session — Activity 3 + Quiz (1.5h)

### Slide 1 — Title Slide
**Title:** Module 1 Lab — Churn Prediction Framing + Quiz
- Module 1, Class 5 (Lab)
- Activity 3: Frame a real ML problem
- Module 1 Quiz (15 MCQs)
- Bring everything you've learned this module

**Speaker notes:** Last class of Module 1. Today is hands-on — you'll apply everything we've covered. First, a warm-up to get your brains going. Then Activity 3, which is the main event: framing a telecom churn prediction problem from scratch. After that, a quiz to see where you stand. No surprises — everything on the quiz was covered in class.

**Visual:** Lab title card with timer breakdown: Warm-up (10 min) → Activity 3 (50 min) → Quiz (20 min) → Wrap-up (10 min).

---

### Slide 2 — Lab Schedule
**Title:** Today's Plan — 90 Minutes
- **0:00-0:10** — Warm-up review exercise
- **0:10-0:20** — Activity 3 introduction and team formation
- **0:20-0:55** — Group work: Frame the churn problem
- **0:55-1:05** — Group presentations (2 min each)
- **1:05-1:25** — Module 1 Quiz (15 MCQs)
- **1:25-1:30** — Wrap-up and Module 2 preview

**Speaker notes:** Here's the timeline. We move fast today, so stay focused. Warm-up is 10 minutes to get everyone on the same page. Then I'll explain Activity 3, you'll form teams, and work for 35 minutes. Quick presentations, then the quiz. Let's go.

**Visual:** Timeline bar with time blocks color-coded by activity type.

---

### Slide 3 — Warm-Up: Rapid Review
**Title:** Warm-Up — Quick Fire Round
- I'll ask 10 rapid questions — shout out the answers
- Topics: AI vs ML vs DL, ML types, workflow steps, ethics
- No grades — just getting your brains warmed up

> **Ask the class (rapid fire):**
> 1. "ML is a subset of ___?"
> 2. "Supervised learning needs ___ data?"
> 3. "Name the 6 ML workflow steps."
> 4. "Clustering is what type of ML?"
> 5. "What does COMPAS stand for? (Trick question — did anyone read the article?)"

**Speaker notes:** Quick fire round. I'll throw questions and you shout answers. Don't be shy — wrong answers are fine, this is warm-up. Ready? Let's go.

**Visual:** Question cards that can be revealed one at a time. Game-show style.

---

### Slide 4 — Activity 3 Introduction
**Title:** [ACTIVITY] Real-World Integration: Telecom Churn Framing
- **Scenario:** You're the data team at UzTelCom (fictional company, based on real telecom)
- Management says: "We're losing customers. Can AI help?"
- Your job: Frame this as an ML problem — end to end
- Deliverable: A one-page problem framing document

> [ACTIVITY] Professor's Activity 3 — Real-World Integration: Telecom Churn Framing

**Speaker notes:** Here's the scenario. You work at a telecom company. Management comes to you and says "we're losing subscribers, fix it with AI." That's all they give you. Your job is to turn that vague request into a concrete ML problem. By the end, you'll have a document that a data science team could actually work from. This is the most important skill in ML — problem framing.

**Visual:** Scenario setup card with fictional company logo and management quote.

---

### Slide 5 — Activity 3: What to Deliver
**Title:** Your Problem Framing Document Must Include:
1. **Problem statement:** One clear sentence — what are you predicting?
2. **ML type:** Which type of ML and why?
3. **Target variable:** What exactly counts as "churn"? Define it.
4. **Features list:** What data would you need? List at least 8 features.
5. **Data source:** Where would this data come from in a real telecom?
6. **Success metric:** How would you measure if your model works?
7. **One ethical concern:** What could go wrong?

**Speaker notes:** Seven components. Each one matters. The most common mistake I see: people jump straight to features without defining the problem or the target variable. What counts as churn? Customer canceled? Inactive for 30 days? Ported their number? Your definition changes everything. Take 5 minutes to discuss the definition before you start listing features.

**Visual:** Numbered checklist — each item with a brief example.

---

### Slide 6 — Activity 3: Team Formation
**Title:** Form Teams of 3-4
- Mix it up — don't just sit with your usual group
- Assign roles:
  - **Lead:** Keeps the team focused, presents at the end
  - **Scribe:** Writes the document
  - **Challenger:** Questions every assumption
- You have 35 minutes — manage your time

**Speaker notes:** Get into groups of 3-4. I want you to mix up — sit with people you haven't worked with yet. Assign roles right now, don't waste time on that later. The challenger role is the most important — your job is to push back on every assumption. "Why that definition of churn? Why those features? How do we know this data exists?" That kind of questioning makes better ML projects.

**Visual:** Role cards with descriptions. Timer set to 35 minutes.

---

### Slide 7 — Activity 3: Hints and Guidance
**Title:** Things to Think About While Working
- Churn definition matters: "Canceled" vs "inactive 30 days" vs "downgraded plan"
- Feature categories: Usage data, billing, demographics, customer service interactions
- Think about what data a real telecom actually has (not wishful thinking)
- Time horizon: Predicting churn in the next 7 days vs 30 days vs 90 days?
- Uzbekistan context: Prepaid vs postpaid behave very differently

**Speaker notes:** Some hints while you work. Prepaid and postpaid customers churn in completely different ways. A postpaid customer cancels a contract. A prepaid customer just stops recharging — they don't tell you they're leaving. That changes your target variable. Also think about time horizon — predicting 7-day churn gives you less time to act but more accuracy. Ninety-day churn gives you time but less precision. Trade-offs everywhere.

**Visual:** Hint cards — expandable if projected.

---

### Slide 8 — [Group Work Period]
**Title:** Work Time — 35 Minutes
- Clock is running
- I'll circulate to each group
- If stuck, raise your hand — but try to solve it yourselves first
- Remember: No right answer, but your reasoning must be solid

**Speaker notes:** Go. I'll walk around and check in with each group. If you're stuck on the definition, that's normal — it's the hardest part. Push through it. Don't aim for perfect, aim for defensible. Can you explain why you made each choice?

**Visual:** Large timer. Minimal text — this is a working slide.

---

### Slide 9 — Group Presentations
**Title:** Present Your Problem Framing — 2 Minutes Per Group
- Lead presents the 7 components
- Class can ask one question per presentation
- Listen for: How did different groups define churn?
- Listen for: Which features did most groups agree on?

> **Ask the class:** After all groups present — "What was the biggest disagreement between groups?"

**Speaker notes:** Time's up. Let's hear from each group. Two minutes — be concise. I want the class listening for differences. Did every group define churn the same way? Which features showed up in every group? Where did groups disagree? The disagreements are the most interesting part.

**Visual:** Presentation format slide with timer.

---

### Slide 10 — Debrief: Common Patterns
**Title:** What We Learned from the Activity
- Problem framing is harder than it looks — most time spent on definitions
- Feature selection requires domain knowledge, not just technical skill
- Different churn definitions lead to fundamentally different models
- Real ML projects start exactly like this — before any code is written

**Speaker notes:** Notice how much time you spent just defining things. That's real ML work. In industry, the problem framing phase can take weeks. The groups that defined churn precisely had an easier time choosing features. The ones that stayed vague struggled. That's the lesson: precision in problem definition saves you time everywhere else.

**Visual:** Summary card with key insights from the activity.

---

### Slide 11 — Reflection Questions
**Title:** Before the Quiz — Reflect
- What was the hardest part of the framing exercise?
- What would you do differently if you started over?
- How does this connect to the ML workflow we learned in Class 2?
- What data would be hardest to actually get from a real company?

**Speaker notes:** Take 60 seconds to think about these questions. No need to share — this is for your own learning. The framing exercise maps directly to the first three steps of the ML workflow: problem definition, data identification, and preparation planning. Everything we've covered in Module 1 came together in that activity.

**Visual:** Four reflection questions, clean layout.

---

### Slide 12 — Quiz Time
**Title:** Module 1 Quiz — 15 Multiple Choice Questions
- Covers all 4 lectures: AI/ML types, foundations, applications, ethics
- 20 minutes — take your time
- No tricks — if you attended class, you'll do fine
- Closed notes (but the concepts should be in your head by now)

> [ACTIVITY] Professor's Activity 4 — Quiz (15 MCQ)

**Speaker notes:** Quiz time. Fifteen questions, twenty minutes. Everything is from what we covered in class. No surprises. Read each question carefully — some have "which of the following is NOT" phrasing. If you're stuck, skip it and come back.

**Visual:** Quiz instruction card. Timer set to 20 minutes.

---

### Slide 13 — Quiz Topics Preview
**Title:** Quiz Covers These Areas
- AI vs ML vs DL definitions and relationships (3 questions)
- Types of ML: supervised, unsupervised, reinforcement (3 questions)
- ML workflow steps (2 questions)
- Probability and statistics intuition (2 questions)
- ITES applications (3 questions)
- Ethics: bias, fairness, privacy (2 questions)

**Speaker notes:** Here's the breakdown so you can allocate your time. No section is weighted more than others. If you're strong on one area and weak on another, spend your time on the weak spots. Pencils up in 20 minutes.

**Visual:** Topic breakdown with question count per area.

---

### Slide 14 — Module 1 Wrap-Up
**Title:** Module 1 Complete — What You've Learned
- What AI, ML, and DL are — and how they relate
- Three types of ML and when to use each
- The end-to-end ML workflow
- Real-world applications in ITES and Uzbekistan
- Why ethics is not optional
- How to frame an ML problem from scratch

**Speaker notes:** That wraps Module 1. You came in two weeks ago knowing AI as a buzzword. Now you can define it, classify ML problems, walk through the workflow, discuss real applications, and argue about ethics. Most importantly, you just framed a real ML problem. In Module 2, we start building the technical skills — Python, math, and data manipulation. The foundation you built here is what everything else rests on.

**Visual:** Six achievement badges — one per class topic. Module 1 completion banner.

---

### Slide 15 — What's Next: Module 2 Preview
**Title:** Module 2: Programming and Mathematical Foundations
- Python crash course — data types, functions, control flow
- NumPy and Pandas — the tools you'll use every day
- Data visualization — Matplotlib, Seaborn
- Linear algebra and calculus for ML — intuition first, formulas second
- Starts next week — come ready to code

**Speaker notes:** Module 2 is where we go from concepts to code. If you've never programmed before, don't panic — we start from zero. If you have some experience, the first class will be review for you, and it picks up quickly after that. I'll share a pre-class Python setup guide so your environment is ready on day one. See you next week.

**Visual:** Module 2 preview card with topic icons. Setup instructions link.

---

### Slide 16 — Homework: Module 1 Assignment Final Step
**Title:** Final Assignment Step — Due Before Module 2 Starts
- Polish your group's churn framing document
- Individual submission: Write your own version (can build on group work)
- Add a "Lessons Learned" section: What would you do differently?
- Submit via LMS before the first class of Module 2

**Speaker notes:** Last thing. Take your group's work and create your own individual version. It's fine to build on what your group did, but add your own thinking. Include a lessons learned section — what would you change if you did this again? Submit on the LMS. This is your Module 1 assignment. After this, you start fresh with Module 2.

**Visual:** Submission checklist with deadline.

---

*End of Module 1 Slide Blueprint — 5 classes (4 lectures + 1 lab)*
