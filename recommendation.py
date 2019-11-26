import pandas as pd
from surprise import KNNWithMeans, KNNBaseline

ratings_df =pd.read_csv('data/BX-Book-Ratings.csv', sep=';', encoding = 'unicode_escape')
books_df= pd.read_csv('data/BX-Books.csv', sep=';' , error_bad_lines=False, encoding = "ISO-8859-1")
books_df = books_df.head()[['ISBN', 'Book-Title', 'Year-Of-Publication', 'Publisher','Book-Author']]
users_df= pd.read_csv('data/BX-Users.csv', sep=';' , error_bad_lines=False, encoding = "ISO-8859-1")

merged_df =books_df.merge(ratings_df).merge(users_df)

df = merged_df.groupby(['ISBN', 'Book-Title'])[['Age']]

pd.DataFrame().rename({'Book-Title':'Title', 'Year-Of-Publication':'Year', 'Book-Author':'Author',
       'User-ID':'UserId', 'Book-Rating':'Rating'})