

# 📚 COMPLETE WALKTHROUGH – AI PROJECT MANAGEMENT (CPMAI METHOD)

---

## 🧩 USE CASE: Predicting Missed Follow-Up Appointments in a Public Hospital

The national hospital system in your country is overwhelmed with patient no-shows. Doctors lose time, and health outcomes worsen when patients fail to return for essential follow-up care. The hospital wants to use artificial intelligence (AI) to **predict which patients are likely to miss their next appointment**, so they can intervene early — by sending reminders, offering telehealth options, or assigning human follow-ups.

This is not just a technical task; it’s a management challenge.  
We will manage this AI project using the **CPMAI methodology**.

---

## 🧭 BEFORE STARTING: Why CPMAI?

### 🔹 What is CPMAI?

CPMAI stands for **Cognitive Project Management for AI**. It is a project management methodology specifically designed for AI systems. Unlike general software projects, AI projects are **data-centric**, **iterative**, and come with **risks** like ethical bias, performance drift, and explainability challenges. CPMAI is structured to address those challenges.

It consists of **six phases**, each of which is crucial for the success of AI systems in real-world contexts. It is:
- **Agile**, meaning it supports iteration and learning over time.
- **Cross-functional**, meaning it bridges technical and non-technical teams.
- **Ethically aware**, meaning it supports privacy, bias mitigation, and compliance.

### 🔹 Why not another method?

- **CRISP-DM** (Cross Industry Standard Process for Data Mining): This is the most widely adopted data mining framework, but it was developed in the 1990s, before modern AI challenges. It lacks agile iterations and focuses more on traditional mining than deployment.

- **TDSP** (Team Data Science Process): Developed by Microsoft, TDSP is great in enterprise settings and supports Agile development. However, it is very tool-specific and assumes that your infrastructure is built on Microsoft Azure, which may not be the case in a public hospital context.

- **SEMMA** (Sample, Explore, Modify, Model, Assess): This is a technical workflow by SAS, focused mostly on modeling. It lacks explicit steps for business alignment, stakeholder communication, or risk/ethics handling. It's more suitable for individual data scientists.

- **KDD** (Knowledge Discovery in Databases): This is an academic framework focused on discovering patterns in large data sources. It is not a project management method and does not support real-world product deployment, business collaboration, or lifecycle management.

👉 In our case — a public hospital, with multiple stakeholders, ethical risks, real-world deployment — CPMAI is the most suitable.

---

# 🧭 CPMAI Phase I: Business Understanding

## ✦ What is this phase?

Business understanding is the foundation of any AI project. We need to fully define what problem we’re solving, for whom, why, and how success will be measured. Most AI projects fail because they begin with **technical excitement** (like “let’s use machine learning!”) rather than **business purpose**.

## ✦ Application to Our Use Case

We begin by conducting **stakeholder interviews** with:
- Doctors
- Appointment managers
- IT team
- Ethics committee

We ask:
- Why do follow-up appointments matter?
- What happens when a patient doesn’t show up?
- What’s currently being done to reduce no-shows?

### 🎯 Reframed Objective:
Rather than just "predict no-shows", our business goal is to:
> Increase follow-up attendance by 30% through predictive and targeted intervention.

### 📈 Business KPIs (Key Performance Indicators):
- Reduce missed appointments by 30%
- Improve patient follow-up within 7 days of original date
- Lower average wait time for re-bookings

We also clarify:
- **Scope**: Pilot in 5 departments first (e.g., oncology, cardiology).
- **Constraints**: Must respect GDPR laws, avoid bias, and integrate with existing scheduling software.

---

# 📊 CPMAI Phase II: Data Understanding

## ✦ What is this phase?

This phase is about understanding **what data exists**, **how reliable it is**, and **whether it supports our problem definition**. AI without good data is useless. Here, we inspect, question, and validate all the sources of information.

## ✦ Application to Our Use Case

We discover:
- Structured patient data: age, gender, previous appointments, time between visits
- Doctor notes: semi-structured text data
- System logs: timestamps, check-in times, missed appointments
- No direct data on patient emotions, socioeconomic status, or transportation

We conduct a **Data Quality Audit**:
- Check for missing values (e.g., no shows often have incomplete histories)
- Check for data drift (e.g., after COVID, telehealth data becomes more common)
- Look for **label leakage** (e.g., can we predict no-shows just from metadata?)

We also explore **potential bias**:
- Are certain ethnicities, income brackets, or rural patients more likely to be flagged? If yes, we must be careful not to **automate discrimination**.

We create a **data schema** and document:
- Field names
- Types
- Source systems
- Reliability
- Update frequency

---

# 🧹 CPMAI Phase III: Data Preparation

## ✦ What is this phase?

Here, we transform raw data into a format that AI models can use. This includes cleaning, normalizing, encoding, imputing, and creating new features. A majority of AI project time (up to 80%) is spent here.

## ✦ Application to Our Use Case

1. **Handle Missing Data**: Some patients don’t have full histories.
   - For numeric fields: Use median imputation.
   - For categorical fields: Use a special value like “unknown”.

2. **Encoding Categorical Variables**:
   - Convert appointment type into one-hot encoding (e.g., [consultation, surgery, lab test])
   - Encode gender as binary or one-hot, depending on how inclusive the system is.

3. **Time Feature Engineering**:
   - Days since last appointment
   - Frequency of previous no-shows
   - Distance from hospital (estimated via ZIP code)

4. **Outlier Detection**:
   - Remove data entries with impossible values (e.g., negative age)

5. **Bias Mitigation**:
   - Exclude features like race or religion unless explicitly justified by medical standards
   - Use fairness analysis later in modeling

6. **Balancing the Dataset**:
   - No-shows may be rare (~15% of records).
   - We apply **SMOTE** (Synthetic Minority Oversampling Technique) to create synthetic samples of no-shows to balance the dataset for training.

---

# 🤖 CPMAI Phase IV: Model Development

## ✦ What is this phase?

We now select a model, train it, evaluate it, and interpret its predictions. In AI, modeling isn’t just about “accuracy” — it's about **explainability**, **fairness**, and **performance under constraints**.

## ✦ Application to Our Use Case

We consider:
- **Logistic Regression**: Simple, interpretable, but less powerful
- **Random Forest**: Non-linear, handles missing data, decent explainability
- **XGBoost**: Powerful, fast, supports missing data, but more complex

✅ We choose **XGBoost**, and pair it with **SHAP (SHapley Additive exPlanations)** to interpret model decisions per patient.

We:
- Split the data into train, validation, and test sets (e.g., 60/20/20)
- Train multiple models and compare performance
- Use evaluation metrics suited for imbalanced data:
  - Precision: How many flagged patients actually no-show?
  - Recall: How many no-shows did we catch?
  - F1-score: Balance of both
  - AUC-ROC: Overall predictive power

We log all experiments using **MLflow** to ensure reproducibility.

---

# ✅ CPMAI Phase V: Model Evaluation

## ✦ What is this phase?

Model evaluation is about more than numbers — it means validating whether the AI actually **works** in a human, social, legal, and business context.

## ✦ Application to Our Use Case

Technical Results:
- AUC = 0.91
- Precision = 84%
- Recall = 72%
- F1 = 0.77

Human Testing:
- We organize **feedback sessions** with scheduling staff and doctors
- They want to see **"Why this patient?"** → We use SHAP to generate patient-specific explanations

Fairness Check:
- We run subgroup analysis across gender, age, department
- If we find bias in predictions (e.g., female patients unfairly flagged), we investigate further

---

# 🚀 CPMAI Phase VI: Operationalization

## ✦ What is this phase?

This is the phase where the model is deployed in the real world — not just running on a laptop, but integrated into hospital systems. This includes **infrastructure**, **monitoring**, **versioning**, and **human processes**.

## ✦ Application to Our Use Case

### ⚙️ Model Deployment

We containerize the model using **Docker**.  
We expose it as a REST API using **FastAPI**, which integrates with the hospital’s scheduling platform.

### 🧪 CI/CD (Continuous Integration / Continuous Deployment)

We use:
- **GitHub Actions** to trigger tests on new model versions
- **DVC (Data Version Control)** to track changes in data
- **MLflow** to manage model lifecycle
- **Prometheus + Grafana** to monitor performance

CI/CD ensures:
- Every update to the model is tested
- Nothing is deployed unless it passes validation and fairness thresholds
- We can rollback to previous models if an error occurs

---

# 🧯 Risk Mitigation

We apply a structured **AI Risk Audit**, covering:

1. **Legal Risks** (GDPR):  
   - We log every prediction made, store it securely, and can explain every decision.

2. **Ethical Risks**:  
   - We allow doctors to override predictions manually.  
   - We present explanations clearly to avoid blind trust.

3. **Technical Risks**:  
   - We set up **drift detection**: if the model’s performance drops (e.g., after a new variant of flu), we retrain.

---

# 📈 Post-Deployment Monitoring

After deployment:
- We monitor number of flagged patients per day
- We calculate daily recall and precision
- We run weekly reviews with doctors
- If precision drops below 75% → send alert
- If flagged patients increase by 30% in a week → investigate data drift

---

# 🧑‍🤝‍🧑 TEAM STRUCTURE

- **Project Manager**: Aligns all stakeholders
- **Data Scientist**: Builds models, interprets results
- **ML Engineer**: Deploys, monitors, integrates
- **Healthcare Expert**: Validates features, ethical constraints
- **Legal Officer**: Reviews GDPR, privacy, consent
- **UI/UX Designer**: Makes model output useful for humans



