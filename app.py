import os
import shutil
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from indexing import clear_database, retrieve_documents, partition_documents, store_in_chroma
from query_data import query_rag
from dotenv import load_dotenv

load_dotenv()



DATA_PATH = "new-data"



def main():
    st.title("PDF Indexing and Querying App")
    if os.path.exists(DATA_PATH):
            existing_pdfs = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
            if existing_pdfs:
                st.subheader("Existing indexed Files:")
                for idx, pdf in enumerate(existing_pdfs):

                    st.markdown(f"📄 {pdf}")

    uploaded_files = st.file_uploader("Upload up to 5 PDF files", type=["pdf"], accept_multiple_files=True)

    if st.button("Upload and Index"):
        if len(uploaded_files) > 5:
            st.error("Please upload a maximum of 5 PDF files.")
        else:

            clear_database()
            shutil.rmtree(DATA_PATH)
            os.makedirs(DATA_PATH, exist_ok=True)

            # Save uploaded files
            for uploaded_file in uploaded_files:
                with open(os.path.join(DATA_PATH, uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())


            documents = retrieve_documents(DATA_PATH)
            chunks = partition_documents(documents)
            store_in_chroma(chunks)

            st.success("Files indexed successfully!")


    query_text = st.text_input("Enter your query:")

    if st.button("Run Query"):
        with st.spinner("Searching and generating answer..."):
            if query_text:
                response, source = query_rag(query_text)
                st.write(response)
                if source:
                    st.write("Source:")
                    source = (source[0].split("/"))[1].split(":")
                    st.write(f"Book: {source[0]}, Page: {source[1]}, Chunk: {source[2]}")
            else:
                st.error("Please enter a query.")


if __name__ == "__main__":
    main()
