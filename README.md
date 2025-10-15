# 🎥 YouTube Data Analysis Project  

## 📘 Overview
Our project presents a **data analysis of YouTube videos**, combining metadata from Kaggle with automatically extracted transcripts and English translations.  
The aim is to understand how **video content and categories** relate to their transcripts and metadata, using **machine learning (Bag-of-Words)** to classify video categories.  

We completed this project as part of **EGBI222 – Data Analytics** coursework.  
All analysis and results are summarized in the accompanying **2-page PDF report** (`Group Project.pdf`).

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

This project focuses on analyzing YouTube data to uncover trends and patterns in video content, categories, and audience engagement. The workflow was designed to be modular, collaborative, and reproducible. The full process is outlined below.

### 1. Data Acquisition
The dataset was obtained from **Kaggle** (“YouTube Videos Data for ML and Trend Analysis”), containing information such as video IDs, titles, categories, views, likes, comments, and publish times. The dataset was imported into **Google Colab** and stored in **Google Drive** for team access.

### 2. Data Preprocessing
The dataset was cleaned to remove missing, duplicated, or irrelevant rows. Columns were standardized (e.g., lowercase naming, consistent encoding) and unnecessary characters were stripped. Each participant was assigned a portion of the dataset for distributed processing.

| Person | Data Range (Rows) | Input File | Output File |
|--------|------------------|-------------|--------------|
| 1 | 1–4,398 | transcripts_p1_1–4,398.csv | transcripts_en_p1_13,193–17,589.csv |
| 2 | 4,399–8,796 | transcripts_p2_4,399–8,796.csv | transcripts_p2_13,193–17,589.csv |
| 3 | 8,797–13,192 | transcripts_p3_8,797–13,192.csv | transcripts_p3_13,193–17,589.csv |
| 4 | 13,193–17,589 | transcripts_p4_13,193–17,589.csv | transcripts_p4_13,193–17,589.csv |

### 3. Audio Transcription
If a transcript column was missing or incomplete, audio data were processed using the **Whisper model** to automatically generate English transcripts. The model was executed in GPU mode (`device="cuda"`) for faster inference. Each output was saved in CSV format in a shared Drive folder.

### 4. Language Detection and Translation
To ensure consistency, all transcripts were converted into **English**. A language detection (`langdetect`) was used.  
- Rows already in English were skipped to save time.  
- Non-English texts were translated in small batches with retry logic for failed translations.  
- The new column `transcripts_en` was added to the DataFrame.

### 5. Merging and Data Consolidation
After all members completed their portions, **all CSV files were merged** into a single master dataset named:
This dataset contained both the original transcript (`transcript`) and the English-translated version (`transcript_en`).  
After merging, these two columns were **separated** to create two independent datasets:
- [transcript](https://drive.google.com/file/d/176C5GE1cDqjkVpf651AEjEUbh5oh3-Wf/view?usp=sharing.csv) → contained only the original transcripts  
- [transcript_en](https://drive.google.com/file/d/17VB14Gx84Cct2LfMZbUFAUScdMK8sfBs/view?usp=sharing) → contained only the English-translated transcripts  
This separation made later processing, language analysis, and model training more organized and efficient.

### 6. Text Vectorization (Bag-of-Words)
The text in `transcripts_en` was transformed into numerical features using the **Bag-of-Words (BoW)** model with `CountVectorizer`. This allowed the model to quantify word frequencies across all videos and categories. Stop words were removed, and both unigrams and bigrams were included.

### 7. Classification Model
A simple **Logistic Regression classifier** was trained to predict the video’s category based on its transcript.  
- Input: BoW features  
- Output: Predicted category  
Model accuracy and F1 scores were calculated to evaluate performance.

### 8. Visualization and Interpretation
Descriptive analysis and visualization were performed to understand category distribution and patterns:
- **Horizontal bar charts** for top video categories  
- **Pie charts** showing the proportion of each category  

### 9. Export and Reporting
All processed and translated data were exported back to **Google Drive** under the `/exports` folder. Each team member verified the data integrity, and the combined results were used to generate visual summaries and final insights.
