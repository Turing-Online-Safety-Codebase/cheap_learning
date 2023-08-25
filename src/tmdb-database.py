import requests
from tqdm import tqdm
from datetime import datetime as dt
import pandas as pd
import numpy as np
import json
import argparse
from collections import Counter

def parse_args():
    """Parser function"""
    parser = argparse.ArgumentParser(description="TMDB movie sentiment databse")
    parser.add_argument("--path_to_headers", type=str, help="path to headers")
    pars_args = parser.parse_args()
    
    return pars_args

def url_maker(page, year):
    """Function to create URL to download movie IDs from TMDB"""
    url = f"https://api.themoviedb.org/3/discover/movie?include_adult=false&include_video=false&language=en-US&page={page}&sort_by=vote_count.asc&primary_release_year={year}"
    return url

def get_movie_ids(years, headers):
    """Function to obtain movie IDs given a list of years"""
    movie_ids = list()
    for year in tqdm(years):
        for page in range(1, 501):
            url = url_maker(page, year)
            response = requests.get(url, headers=headers)
            movie_list = response.json()
            movie_ids += [list_["id"] for list_ in movie_list["results"]]
    return movie_ids

def get_reviews_from_movie(dict_movie,
                           authors_activity_dictionary,
                           reviews,
                           ratings,
                           usernames,
                           movie_ids,
                           date_cutoff,
                           activity_limit = 10,
                           format_date = "%Y-%m-%d",
                           ):
    """Function to update IN PLACE the lists with the reviews and correspondent important info"""
    if len(dict_movie["results"]) > 0:
        dates_created = [d["created_at"] for d in dict_movie["results"]] #"2023-04-13T13:19:08.627Z"
        dates_formated = [dt.strptime(date[:10], format_date) for date in dates_created]

        usernames_movie = [d["author_details"]["username"] for d in dict_movie["results"]]
        contents_movie = [d["content"] for d in dict_movie["results"]]
        ratings_movie = [d["author_details"]["rating"] if d["author_details"]["rating"] is not None else 5 for d in dict_movie["results"]]
        movie_id = dict_movie["id"]


        for (i, username) in enumerate(usernames_movie):
            if dates_formated[i] > date_cutoff:
                if username in authors_activity_dictionary.keys() and authors_activity_dictionary[username] <= activity_limit:
                    reviews.append(contents_movie[i])
                    ratings.append(ratings_movie[i])
                    usernames.append(usernames_movie[i])
                    movie_ids.append(movie_id)
                    authors_activity_dictionary[username] +=1
                if username not in authors_activity_dictionary.keys():
                    reviews.append(contents_movie[i])
                    ratings.append(ratings_movie[i])
                    usernames.append(usernames_movie[i])
                    movie_ids.append(movie_id)
                    authors_activity_dictionary[username] = 1


def get_dataframe(movie_ids_years, headers, date_cutoff = "2021-09-01", format_date = "%Y-%m-%d"):
    """Function to get the DataFrame before cleaning"""
    reviews = list()
    usernames = list()
    ratings = list()
    movie_ids = list()
    authors_activity_dictionary = dict()

    date_cutoff_dt = dt.strptime(date_cutoff, format_date)

    for id_ in tqdm(movie_ids_years):
        url = f"https://api.themoviedb.org/3/movie/{id_}/reviews?language=en-US"

        response = requests.get(url, headers=headers)
        dict_movie =response.json()
        get_reviews_from_movie(dict_movie, authors_activity_dictionary,reviews, ratings, usernames, movie_ids, date_cutoff_dt)

    database = pd.DataFrame()
    database["review"] = reviews
    database["movie_id"] = movie_ids
    database["username"] = usernames
    database["rating"] = ratings

    return database

def process_database(database):
    """Function to process database"""
    # drop reviews with a 5 or 6
    database_filtered = database[(database["ratings"]>6) | (database["ratings"]<5)]

    # removing new lines and changing apostrophes
    database_filtered.reviews = database_filtered.reviews.apply(lambda x: x.replace("\n", " "))
    database_filtered.reviews = database_filtered.reviews.apply(lambda x: x.replace("\r", " "))
    database_filtered.reviews = database_filtered.reviews.apply(lambda x: x.replace("\'", "'"))

    # give reviews with a <=4 a Negative and >=7 a Positive rating
    database_filtered["binary_rating"] =  np.where(database_filtered['ratings']>=7, 'Positive', 'Negative')

    return database_filtered

def get_headers(path):
    """Get headers from path given"""
    file = open(path, "r")
    headers = json.load(file)

    return headers

def main(path):
    """Function to download movie reviews, process it and 
    get a movie sentiment database from TMDB"""
    headers = get_headers(path)
    years = [2021, 2022, 2023]
    movie_ids_years = get_movie_ids(years, headers)

    database = get_dataframe(movie_ids_years, headers)
    database = process_database(database)

    database.to_csv("movie_sentiment_tmdb.csv", index=False)

if __name__ == "__main__":
    args = parse_args()

    main(args.path_to_headers)
