import pandas as pd 
import numpy as np 
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader #this takes the raw text of the books description and convert it into a format that langching can work with
from langchain_text_splitters import CharacterTextSplitter #split the whole doc containing all the description into meaningful chunks 
from langchain_openai import OpenAIEmbeddings #API to call the model
from langchain_chroma import Chroma #use to store the info's into the vector database 
from langchain_community.embeddings import HuggingFaceEmbeddings


import gradio as gr


load_dotenv()
books=pd.read_csv("books_with_emotion.csv")


books["large_thumbnail"]= books["thumbnail"]+"&fife=w800"
books["large_thumbnail"]= np.where(
    books["large_thumbnail"].isna(),
    "image.png",
    books["large_thumbnail"]
)

raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
document = text_splitter.split_documents(raw_documents)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db_books = Chroma.from_documents(document, embedding=embedding_model)

def retrieve_semantic_recomendations( query:str,category:str=None,tone:str=None,initial_top_k:int=50,final_top_k:int=16)-> pd.DataFrame:
    recs = db_books.similarity_search_with_score(query, k=initial_top_k)  # Fetch similar books
    books_list = [int(rec[0].page_content.strip('"').split()[0]) for rec in recs]

    books_recs= books[books["isbn13"].isin(books_list)].head(final_top_k)

    if category != "All":
        books_recs = books_recs[books_recs["Simple_categories"]== category].head(final_top_k)
    else:
        books_recs = books_recs.head(final_top_k)

    if tone=="Happy":
        books_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone=="Surprising":
        books_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone=="Angry":
        books_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone=="Suspense":
        books_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone=="Sad":
        books_recs.sort_values(by="sadness", ascending=False, inplace=True)
    
    return books_recs

def recomended_books(query:str, category:str, tone:str):
    recommendations = retrieve_semantic_recomendations(query,category,tone)
    results=[]

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30])+"......."

        author_split = row["authors"].split(";")
        if len(author_split)==2:
            author_split = f"{author_split[0]} and {author_split[1]}"
        elif len(author_split)>2:
            author_split = f"{', '.join(author_split[:-1])} and {author_split[-1]}"
        else:
            author_split = row["authors"]
        
        caption = f"{row['title']} by {author_split}:{truncated_description}"
        results.append((row["large_thumbnail"],caption))

    return results

categories = ["All"] + sorted(books["Simple_categories"].unique())
tones = ["All"]+["Happy","Surprising","Angry","Suspenseful","Sad"]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic book recommender")

    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a description of a book: ",
        placeholder="eg. A story about forgiveness...")
        category_dropdown = gr.Dropdown(choices=categories, label="Select a Category: ", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Select an emotional tone: ", value="All")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recomendations")
    output= gr.Gallery(label="Recommended books", columns=8, rows=2)

    submit_button.click(fn=recomended_books,
                        inputs=[user_query, category_dropdown, tone_dropdown],
                        outputs=output)



if __name__ == "__main__":
    dashboard.launch()

