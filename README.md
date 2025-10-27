# EduHub Query Builder

A Streamlit-based self-service data exploration tool that allows MOE HQ officers to filter, sort, and download datasets without coding knowledge.  
The app dynamically validates defaults to prevent Streamlit exceptions when switching datasets or removing fields.

---

## 🚀 How to run locally

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/eduhub-query-builder.git
   cd eduhub-query-builder
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the app:
   ```bash
   streamlit run eduhub_query_builder_app_v9_16_6.py
   ```

---

## 📊 Demo datasets included
| Dataset                                            | Description                                 |
| -------------------------------------------------- | ------------------------------------------- |
| `sample_p6_scores.csv`                             | P6 students' 2022–2024 exam results         |
| `sample_psle_scores_v2.csv`                        | Extended PSLE performance dataset           |
| `sample_attendance_sg_v2.csv`                      | Student attendance records (v2)             |
| `sample_attendance_sg_v3.csv`                      | Enhanced attendance dataset with new fields |
| `GraduateEmploymentSurveyNTUNUSSITSMUSUSSSUTD.csv` | Public graduate employment survey dataset   |


---

⚙️ Features
Interactive field selection, filtering, and sorting
Dynamic validation of defaults to prevent Streamlit errors
SQL query preview and CSV download
Support for multiple datasets
Responsive sidebar and intuitive UX

---

## 🌐 Deployment
Deploy directly to Streamlit Community Cloud:
1. Push this repo to GitHub.
2. Visit share.streamlit.io.
3. Choose app.py as the main entry file.

---

## 🛠 Requirements
See [`requirements.txt`](requirements.txt).
