# 🎥 YouTube Data Analysis Project  

## 📘 Overview
Our project presents a **data analysis of YouTube videos**, combining metadata from Kaggle with automatically extracted transcripts and English translations.  
The aim is to understand how **video content and categories** relate to their transcripts and metadata, using **machine learning (Bag-of-Words)** to classify video categories.  

We completed this project as part of **EGBI222 – Data Analytics** coursework.  
All analysis and results are summarized in the accompanying **2-page PDF report** (`YouTube-Data-Summary.pdf`).

---

## 👥 Group Members
| No. | Name | Student ID |
|----|------|-------------|
| 1 | Khemmanat Kiattikulpimol | 6713356 |
| 2 | Chithisa Chakpet | 6713357 |
| 3 | Thiptiyakorn Meechaiyo | 6713364 |
| 4 | Bongkodmas Tankul | 6713371 |
| 5 | Bhuritz Lertwanasiriwan  | 6713373 |
| 6 | Suchanaree Srichaloem | 6713394 |

---

## 🧩 Project Objectives
1. Download and analyze the **YouTube videos dataset** from Kaggle.  
2. Fetch or generate **transcripts** from YouTube videos using `youtube-transcript-api`.  
3. Detect language and **translate transcripts into English**.  
4. Perform **exploratory data analysis (EDA)** and visualization.  
5. Train a **Bag-of-Words classification model** to predict video categories.  

---

## 🗂️ Dataset
**Source:** [YouTube Videos Data for ML and Trend Analysis (Kaggle)](https://www.kaggle.com/datasets/cyberevil545/youtube-videos-data-for-ml-and-trend-analysis)

This dataset contains:
- Video metadata (ID, title, description, category, views, etc.)
- Our added columns:
  - `transcript` → Text extracted from YouTube captions (or fallback: title + description)
  - `lang_guess` → Detected language of transcript
  - `transcript_en` → English-translated transcript text  

---

## 🧠 Methodology

### Step 1: Download and Load Dataset
```python
import pandas as pd
df = pd.read_csv("youtube_data.csv")
