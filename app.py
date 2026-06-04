import streamlit as st
from ask import answer

st.title("Chat with my papers")
st.write("Ask a question and get an answer grounded in your research papers.")

question = st.text_input("Your question")

if question:
    with st.spinner("Searching the papers and writing an answer..."):
        text, sources = answer(question)
    st.markdown(text)
    with st.expander("Sources retrieved"):
        for s in sorted(set(sources)):
            st.write(s)