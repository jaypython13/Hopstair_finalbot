import PyPDF2
import openai
import streamlit as st

# Set your OpenAI API key
openai.api_key = 'your-api-key-here'

def extract_text_from_pdf(pdf_path):
    pdf_reader = PyPDF2.PdfFileReader(open(pdf_path, 'rb'))
    text = ""
    for page_num in range(pdf_reader.numPages):
        page = pdf_reader.getPage(page_num)
        text += page.extract_text()
    return text

# Extracted dataset from PDF
dataset_text = extract_text_from_pdf('custom_dataset.pdf')

def generate_nudge(user_story, dataset_text):
    prompt = f"User story: {user_story}\n\nBased on this story and the following dataset, provide an encouraging and personalized nudge to boost the user's confidence.\n\nDataset: {dataset_text}\n\nNudge:"
    
    response = openai.Completion.create(
        engine="gpt-3.5-turbo",
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )

    return response.choices[0].text.strip()

# Example user story
user_story = "I've been feeling really overwhelmed with my workload lately, and it's affecting my confidence."
nudge = generate_nudge(user_story, dataset_text)
print(nudge)



# Streamlit App
def main():
    st.title("Confidence Boosting Chatbot")
    st.write("Tell me your story and I'll provide some encouraging words!")

    user_story = st.text_area("Your Story", height=200)
    
    if st.button("Generate Nudge"):
        if user_story.strip() == "":
            st.warning("Please enter your story.")
        else:
            nudge = generate_nudge(user_story, dataset_text)
            st.success(nudge)

if __name__ == "__main__":
    main()

