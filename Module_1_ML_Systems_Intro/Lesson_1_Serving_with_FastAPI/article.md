# Lesson 1.1 — Serving a Model in Production

> **Lesson 1.1** — What production ML actually is, why it is harder than training, and how to serve your first model as a real API inside a container.

| | |
|---|---|
| **Problem this solves** | A trained model sitting in a notebook delivers no value. Getting it into a running service that other systems can call is the first engineering problem in the ML lifecycle. |
| **Mental model** | Think of your model as a function: `f(features) → prediction`. Production ML is the engineering work of making that function callable, reliable, versioned, and monitorable at scale. |
| **What the lecture demonstrates** | Training a simple classifier → wrapping it in a FastAPI endpoint → containerizing it with Docker → calling it like any other web service |
| **Where this fits** | This module builds the **Serving Layer** in the system map — the last step the model reaches before it delivers value. Everything else in the course feeds into this layer. |

---

# Background and Foundations for Machine Learning in Production  
*MLOps Part 1: An introduction to machine learning in production, covering pitfalls, system-level concerns, and an overview of the full ML lifecycle.*

---

## Introduction  
So, you've trained your machine learning model and tested its inference capabilities.  
What comes next? Is your job done?  
**Not really.**  

What you've completed is only a small part of a much larger journey. If you plan to deploy the model in a real-world application, there are many additional steps to consider. This is where **MLOps** becomes essential, helping you transition from model development to a production-ready system.

Machine Learning Operations (MLOps) in production is about integrating ML models into real-world software systems. It's where machine learning meets software engineering, DevOps, and data engineering.

The goal is to reliably deliver ML-driven features—like recommendation engines, fraud detectors, voice assistants, etc.—to end-users at scale. Hence, a key realization is that the ML model or algorithm itself is only a small part of a production ML system.

> **Note:** In real-world deployments, a lot of "glue" is needed around the model to make a complete system, including data pipelines, feature engineering, model serving infrastructure, user interfaces, monitoring, and more. Only a tiny fraction of an "ML system" is the ML code; the vast surrounding infrastructure is much larger and more complex.

---

## Why Does MLOps Matter?  
Building a highly accurate model in a notebook is just the beginning. Once deployed, ML models face changing real-world conditions: users may behave differently over time, data may drift, and model performance can decay. A model's quality does not remain static after deployment.

In production, an ML system must run continuously, often 24/7, and handle evolving data and usage patterns—all while meeting requirements for latency, throughput, and quality.

Inadequate MLOps can lead to stale or incorrect models lingering in production, causing bad predictions that hurt the business. Without proper MLOps, teams often end up with manual, brittle processes for model deployment, resulting in slow iteration and error-prone deployments.

> **Example issues before mature MLOps:**  
> - Slow time to market: Weeks or months to deploy a new model.  
> - Fragile pipelines: Manual steps that easily break.  
> - Scaling issues: Hard to handle growing data or model complexity without automation.

Effective MLOps addresses these risks by establishing processes to continuously monitor, evaluate, and improve models after deployment.

---

## MLOps vs. DevOps and Traditional Software Systems  
Building and deploying ML systems extends traditional software engineering but introduces unique challenges.

| Aspect | Traditional Software | ML Systems |
|--------|----------------------|------------|
| **Development** | Deterministic process | Experimental, data-driven |
| **Versioning** | Code (Git, etc.) | Code, data, and models |
| **Testing** | Functional correctness, performance | Plus data validation, model evaluation, bias/fairness checks |
| **Monitoring** | Service health (latency, errors, CPU/memory) | Plus model predictions, data drift, accuracy over time |
| **Deployment** | Push code through CI/CD | Often involves retraining pipelines, continuous training (CT) |
| **Failure Modes** | Bugs, infrastructure issues | Bugs + model quality degradation (e.g., data drift) |

---

### Key Differences Explained  
#### 1. **Experimental vs. Deterministic Development**  
ML development is highly experimental—trying multiple algorithms, features, and hyperparameters to find what works best. This requires tracking experiments and ensuring reproducibility.

#### 2. **Testing Complexity**  
ML systems require:
- Unit tests for data preprocessing and code.
- Validation of data quality.
- Model performance testing (accuracy, data leakage checks).
- Handling training/serving skew.

> **Data leakage** occurs when information from outside the training dataset is unintentionally used, leading to overly optimistic performance estimates.

#### 3. **Deployment and Updates**  
In ML, a "new version" often means retraining a model with new data. Deployment may involve an automated pipeline for periodic retraining and deployment—known as **continuous training (CT)**.

#### 4. **Production Performance Degradation**  
ML models can degrade over time due to:
- **Data drift**: Changes in input data distribution.
- **Concept drift**: Changes in the relationship between inputs and outputs.

Thus, monitoring must include model quality metrics, not just system health.

#### 5. **Lifecycle Complexity**  
ML systems have a **cyclical lifecycle**: after deployment, you often loop back to data collection and model improvement, creating a feedback loop (the "ML flywheel").

---

## System-Level Concerns in Production ML  
When moving from lab to production, several system-level concerns emerge:

### 1. **Latency and Throughput**  
- **Latency**: Time to produce a prediction after receiving input.  
- **Throughput**: Number of predictions per unit time.

> **Example:** Amazon found every 100ms of extra latency cost 1% in sales.

**Engineering decisions** to address latency/throughput:
- Model simplification or compression.
- Faster hardware or scaling out.
- Model quantization, distillation.
- Choosing deployment architecture: online inference vs. batch processing.
- Caching strategies.

### 2. **Data and Concept Drift**  
- **Data drift**: Input data distribution changes over time.  
- **Concept drift**: The relationship between inputs and outputs changes.

> **Example:** A model trained on summer photos may struggle with winter photos.

**Handling drift** involves:
- Monitoring statistical properties of inputs.
- Setting thresholds and alerts.
- Periodic retraining.
- Online learning (with caution).
- Human-in-the-loop fallback plans.

### 3. **Feedback Loops**  
ML systems can influence their own future input data.  
> **Example:** A recommendation model that narrows content diversity over time.

**Managing feedback loops**:
- Explore-exploit trade-offs.
- Debiasing training data.
- Simulations before deployment.
- Monitoring secondary metrics (e.g., content diversity).

### 4. **Reproducibility**  
The ability to recreate model results reliably is critical for debugging, collaboration, and consistency across environments.

**Achieving reproducibility**:
- Version control for code, data, and models.
- Containerization (e.g., Docker).
- Tests for data and model pipelines.

---

## The Machine Learning System Lifecycle  
A typical ML project lifecycle is cyclical and includes:

1. **Project Scoping**  
   Decide if ML is appropriate and define success criteria.

2. **Data Processing**  
   - Data ingestion & collection.  
   - Data preparation & feature engineering (ETL).

3. **Modeling**  
   - Model training & experimentation.  
   - Model evaluation & validation.

4. **Deployment**  
   - Expose model via API, embed in application, or deploy to edge devices.  
   - Set up inference infrastructure.

5. **Monitoring & Observability**  
   - Track operational metrics (latency, throughput, errors).  
   - Monitor predictive performance and drift.

6. **Maintenance & Continuous Improvement**  
   - Collect new data, retrain, update features/models.  
   - Loop back to earlier stages.

> **Note:** This is often visualized as the **Continuous Delivery for Machine Learning (CD4ML)** cycle.

---

## Conclusion  
This foundational article introduced MLOps as the discipline of moving ML models from development to production. We covered:

- Why MLOps matters: models degrade, systems evolve.
- How MLOps differs from DevOps: experimental development, added testing, cyclical lifecycle.
- Key system-level concerns: latency, drift, feedback loops, reproducibility.
- The end-to-end ML lifecycle: scoping → data → modeling → deployment → monitoring → iteration.

The mindset shift is from **model-centric** to **systems engineering**, where reproducibility, automation, and monitoring are first-class citizens.

---

*This course is actively growing — more lessons, deeper dives, and real-world case studies are being added.*
