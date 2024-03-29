from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from openai import OpenAI
import openai
import pandas as pd
import streamlit as st
client = OpenAI()

def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model,dimensions=512).data[0].embedding

def search_term_cosine_similarity(query, documents, vectorizer):
    query_vector = vectorizer.transform([query])
    document_vectors = vectorizer.transform(documents)
    similarities = cosine_similarity(query_vector, document_vectors).flatten()
    return similarities

def find_closest_matches(similarities, paragraphs,page_numbers, document_names):
    # Sort by similarity in descending order
    sorting_indices = sorted(range(len(similarities)), key=lambda k: similarities[k], reverse=True)
    sorted_page_numbers = [page_numbers[i] for i in sorting_indices]
    sorted_document_names = [document_names[i] for i in sorting_indices]
    sorted_paragraphs = [paragraphs[i] for i in sorting_indices]
    sorted_similarities = [similarities[i] for i in sorting_indices]
    return sorted_paragraphs, sorted_page_numbers, sorted_document_names, sorting_indices, sorted_similarities

def GPTprompt(pageno):
    text1 = "Please could you answer my question based on the text from '" + sorted_document_names[pageno][0:-4] + "' on " + sorted_page_numbers[pageno] + "?\n"
    text2 = "Answer my questions based on the text. Start your response directly by giving information, no source. Only quote the source at the end of each response and nowhere else.\n"
    text3 = "If this text does not have enough information to answer in a full and unbiased way it is bad. This would happen for example if it comes from a page with not enough text where many images could have been. \nIn this case, only reply 'ERROR999' to me and nothing else. I will give you another page and we try again.\n"
    text4 = "\nMy question is:\n\n" + prompt + "\n\n"
    text5 = "Text is here:\n\n" + sorted_paragraphs[pageno]
    return(text1+text2+text3+text4+text5)

df = pd.read_csv('embedding.csv')
df['ada_embedding'] = df.ada_embedding.apply(eval).apply(np.array)
embedding_matrix = np.array(df['ada_embedding'].to_numpy().tolist())

st.title("Welsh AI assistant")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("How can I help?"):
    # Create chat history with system and user messages
    chat_history = [
        {"role": "system", "content": "You are a helpful assistant, working for the Welsh government providing people with information on saving energy."}]
    st.chat_message("user").write(prompt)
    with st.spinner('Wait for it...'):
        embedded_prompt = np.array(get_embedding(prompt)).reshape(1, -1)
        similarities = cosine_similarity(embedded_prompt, embedding_matrix).flatten()
        sorted_paragraphs, sorted_page_numbers, sorted_document_names, sorting_indices, sorted_similarities = find_closest_matches(similarities,
                                                                                                                                   df['Text'],
                                                                                                                                   df['Page'],
                                                                                                                                   df['Document']
                                                                                                                                   )
        for pageno in range(0,5):
            # Create chat history for each iteration and accumulate it
            chat_history_iteration = [{"role": "user", "content": GPTprompt(pageno)}]
            
            response = client.chat.completions.create(
                    model="gpt-3.5-turbo-0125",
                    messages= chat_history + chat_history_iteration
                )
            response_text = response.choices[0].message.content
            
            if response_text!="ERROR999":
                # If no ERROR999 encountered, return the response
                st.chat_message("assistant").write(response_text)
                chat_history.append({"role": "user", "content": prompt})
                chat_history.append({"role": "assistant", "content": response_text})
                with st.expander('Source'):
                    st.write(sorted_paragraphs[pageno])
                break

            elif pageno==4:
                st.chat_message("assistant").write("Sorry I don't know what you mean, I only answer questions regarding energy sustainability.")
                break
    st.session_state.messages.extend(chat_history[1:]) #Everything but first system message
