import pickle
import numpy as np
# import requests
from flask import Flask,render_template,request
import sklearn

app = Flask(__name__)
model = pickle.load(open('artifacts/model.pkl','rb'))
book_names = pickle.load(open('artifacts/book_names.pkl','rb'))
ratings = pickle.load(open('artifacts/final_ratings.pkl','rb'))
pivot = pickle.load(open('artifacts/book_pivot.pkl','rb'))

def fetch_poster(suggestion):
    book_name = []
    ids_index = []
    poster_url = []

    for book_id in suggestion:
        book_name.append(pivot.index[book_id])

    for name in book_name[0]: 
        ids = np.where(ratings['title'] == name)[0][0]
        ids_index.append(ids)

    for idx in ids_index:
        url = ratings.iloc[idx]['url']
        poster_url.append(url)

    return poster_url



def recommend_book(book_name):
    books_list = []
    book_id = np.where(pivot.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(pivot.iloc[book_id,:].values.reshape(1,-1), n_neighbors=10 )

    poster_url = fetch_poster(suggestion)
    
    for i in range(len(suggestion)):
            books = pivot.index[suggestion[i]]
            for j in books:
                books_list.append(j)
    return books_list , poster_url       



@app.route('/',methods = ['POST','GET'])
def index():
    if request.method=="POST":
        selected_book = request.form.get("book")
        if selected_book=="":
                return render_template("index.html",books=book_names)
        recommended_books,posters = recommend_book(selected_book)        
        return render_template("index.html",books=book_names,recommended_books=recommended_books,posters=posters)
    else:
        return render_template("index.html",books=book_names)


if __name__ == "__main__":
    app.run(debug=True)