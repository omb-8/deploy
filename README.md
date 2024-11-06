# Chatbot

A custom question-answering chatbot that leverages Langchain and Gemini AI via Streamlit.

## Setup Instructions

### 1. Create Environment Variables

To keep your Google API key secure, create a `.env` file:

```bash
vim .env
```

Add your Google API key to this file:

```plaintext
GOOGLE_API_KEY=your_actual_api_key_here
```

### 2. Add `.env` to `.gitignore`

To prevent accidental distribution of your API key, add `.env` to your `.gitignore` file:

```bash
echo ".env" >> .gitignore
```

### 3. Install Required Packages

Install dependencies by running:

```bash
pip install -r requirements.txt
```

### 4. Configure File Path for Student Handbook

Edit the `backend.py` file to set the correct file path for the student handbook:

```python
# In backend.py
student_handbook_path = "your/directory/path/to/student_handbook"
```

Replace `"your/directory/path/to/student_handbook"` with the actual path where your student handbook is stored.

---

## Usage

After setup, you can run the chatbot application with Streamlit.
```bash
streamlit run app.py
```

