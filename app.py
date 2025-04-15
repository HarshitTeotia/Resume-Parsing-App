import streamlit as st
import pickle
import docx
import PyPDF2
import re

# Load ML model and preprocessing objects
svc_model = pickle.load(open(r'C:\Users\harsh\PycharmProjects\PythonProject\clf.pkl', 'rb'))
tfidf = pickle.load(open(r'C:\Users\harsh\PycharmProjects\PythonProject\tfidf.pkl', 'rb'))
le = pickle.load(open(r'C:\Users\harsh\PycharmProjects\PythonProject\encoder.pkl', 'rb'))

# Clean resume text
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

# Extract text from different file types
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text

def extract_text_from_txt(file):
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        text = file.read().decode('latin-1')
    return text

def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        return extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        return extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        return extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type.")

# Predict resume category
def pred(input_resume):
    cleaned_text = cleanResume(input_resume)
    vectorized_text = tfidf.transform([cleaned_text]).toarray()
    predicted_category = svc_model.predict(vectorized_text)
    predicted_category_name = le.inverse_transform(predicted_category)
    return predicted_category_name[0]

# Main Streamlit app
def main():
    st.set_page_config(page_title="Resume Category Filter", page_icon="🗂️", layout="wide")
    st.title("Resume Shortlisting Tool for Recruiters")
    st.markdown("Upload multiple resumes and filter them by a specific job role.")

    uploaded_files = st.file_uploader("Upload Resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True)

    # Display job roles from the label encoder as readable options
    job_roles_internal = le.classes_
    job_roles_display = [role.replace('_', ' ').title() for role in job_roles_internal]

    # Safe selection using index
    selected_index = st.selectbox("Select the Job Role to Filter", range(len(job_roles_internal)),
                                  format_func=lambda i: job_roles_display[i])
    target_label = selected_index  # Index matches LabelEncoder output

    if uploaded_files:
        st.subheader("All Resume Predictions")
        matching_resumes = []

        for idx, file in enumerate(uploaded_files):
            try:
                resume_text = handle_file_upload(file)
                category = pred(resume_text)
                st.markdown(f"📄 **{file.name}** → Predicted Category: **{category}**")

                if le.transform([category])[0] == target_label:
                    matching_resumes.append((file.name, category, resume_text))
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")

        st.markdown("---")
        st.subheader("Matching Resumes")

        if matching_resumes:
            st.success(f"Found {len(matching_resumes)} matching resume(s).")
            for i, (name, category, text) in enumerate(matching_resumes):
                with st.expander(f"📄 {name} — {category}"):
                    st.write(f"**Predicted Category:** {category}")
                    if st.checkbox(f"Show extracted text from {name}", key=i):
                        st.text_area("Resume Text", text, height=300)
        else:
            st.warning("No resumes matched the selected job role.")

if __name__ == "__main__":
    main()
