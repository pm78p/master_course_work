# Problems

- Finding appropriate dataset.
- Some datasets have textual and visual features, making it almost impossible to use a Bayesian model easily.
- Some others have a large number of features, making it too complex to draw a Bayesian relation.
- Other datasets had similar distributions in features, which caused the Conditional Probability Table (CPT) to have completely equivalent values (uniform distribution).

### Procedure

- We went through the following steps for different datasets:
  1. Clean the data.
  2. Completely preprocess the data.
  3. Draw a simple Bayesian network.
  4. Derive the initial CPT.
  5. Observe the behavior of the Bayesian network and the probabilities in the CPT table.

This procedure was repeated approximately 7-9 times.

---

# Solution

The dataset used in this study is the **SUPPORT (Study to Understand Prognoses, Preferences, Outcomes, and Risks of Treatments)** dataset. It contains information on critically ill patients and consists of 9,105 patients from five US medical centers, collected between 1989 and 1994. The dataset includes a wide range of variables related to patient demographics, physiological measurements, disease severity, and survival estimates.

## Dataset

### Demographic Variables:
- Age
- Sex
- Race

### Physiological Measurements:
- **Mean Blood Pressure (meanbp):** Average blood pressure.
- **White Blood Cell Count (wblc):** Number of white blood cells.
- **Heart Rate (hrt):** Patient's heart rate.
- **Respiration Rate (resp):** Rate of respiration.
- **Temperature (temp):** Body temperature.
- **Creatinine Level (crea):** Blood creatinine level.
- **Sodium Level (sod):** Blood sodium level.

### Disease Severity and Related Variables:
- **Disease Group (dzgroup):** Group categorizing the disease.
- **Disease Class (dzclass):** Class of the disease.
- **Number of Comorbidities (num.co):** Number of additional diseases.

### Survival Estimates:
- Estimates of survival at 2 months (`surv2m`) and 6 months (`surv6m`).

### Health Conditions:
- **Diabetes (diabetes):** Presence of diabetes.
- **Dementia (dementia):** Presence of dementia.
- **Cancer (ca):** Presence of cancer.

### Target Variable:
- **Functional Disability (sfdm2):** Measure of the patient’s functional disability.

---

# Data Preprocessing

- Drop unimportant columns.
- Drop columns with more than 1,000 missing values, or variables that do not contribute to the prediction of the target (e.g., identifiers or target variables that are not used).
  
### Discretization of Continuous Columns

Since the continuous features show a Gaussian-like distribution, we used **quantile-based discretization**.

#### Quantile-based Discretization

This technique divides the range of a continuous variable into intervals or "bins," where each bin contains approximately the same number of data points. This ensures each bin represents a quantile of the distribution.

We used **Sturges' formula** to estimate the number of bins, then adjusted the bin number by checking the histogram of the quantized version of the related feature.

---

# Bayesian Network

Once data manipulation was complete, we drew a Bayesian network using our intuition about the characteristics of the features.

---

# Estimators

- **Bayesian vs. Maximum Likelihood Estimators (MLE):**
  - Bayesian estimators are based on Bayesian probability theory and incorporate prior knowledge.
  - MLE focuses on the likelihood of the observed data.

### Priors

- **Dirichlet Prior:** A distribution with a given probability density function (PDF).
  
- **BDeu (Bayesian Dirichlet Equivalent Uniform):** A scoring method used for structure learning in Bayesian networks. It evaluates how well a given network structure fits the data by combining prior knowledge with observed data. It uses Dirichlet priors for the Conditional Probability Tables (CPTs).

---

# Hill Climbing

Hill Climbing is another method we used to create the Bayesian model. It starts from a simple base point and iteratively explores adjacent solutions. Instead of constructing the Bayesian Network purely from intuition, this method begins with a simple model and progressively complicates it until a good model is achieved.

### Hill Climbing Network

An intuitive network.

---

# Approximate Estimators (Expectation-Maximization)

**Expectation-Maximization (EM)** is an iterative method used to find maximum likelihood estimates of parameters in models with latent variables. It's particularly useful when dealing with incomplete data. 

EM is more flexible, handling complex models, missing data, and latent variables. However, it is computationally intensive and may require more tuning and longer run times. The accuracy achieved by EM was similar to the Bayesian estimator (valid accuracy = 64%).

---

# Review

Would you recommend this book?  
*Write your review here.*
