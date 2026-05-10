# POI Privacy: Membership Inference Attack on Next POI Prediction

## Overview
This project investigates privacy vulnerabilities in Next POI (Point-of-Interest) 
Prediction models using Membership Inference (MI) attacks.

We demonstrate that an adversary can determine whether a user's location trajectory 
was used to train a POI prediction model, potentially exposing sensitive information 
such as hospital visits, religious activities, and daily routines.

## Research Theme
**Privacy** — Investigating membership inference vulnerabilities in 
location-based AI models.

## Dataset
- **yjmob100k** (Yahoo Japan / LY Corporation)
- 100,000 users / 75 days
- 200×200 grid, 500m × 500m cell size
- 30-minute timeslots
- 85 POI categories

## Models
### Target Model: POITransformer
- Transformer-based Next POI Prediction model
- Input: past 10 POI category sequence
- Output: next POI category probabilities (86 classes)
- Performance: Acc@1: 0.7732, Acc@5: 0.9083

### Attack Model: AttackMLP
- Membership Inference Attack model
- Input: confidence score vector from Target Model (86-dim)
- Output: membership probability (Member / Non-member)
- Performance: AUC: 0.7830, Attack Accuracy: 0.7039

## Repository Structure
poi-privacy-mi/
├── README.md
├── data/
│   └── README.md        # Dataset description
└── src/
├── preprocess.py    # Data preprocessing
├── target_model.py  # POITransformer
└── attack_model.py  # Attack MLP

## References
- Wongso et al., "GenUP: Generative User Profilers as In-Context Learners 
  for Next POI Recommender Systems", SIGSPATIAL 2025
- Shokri et al., "Membership Inference Attacks Against Machine Learning Models", 
  IEEE S&P 2017
