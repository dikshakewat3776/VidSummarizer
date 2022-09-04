from summarizer.sbert import SBertSummarizer
from flask import Flask, render_template, jsonify, request, flash
from sentence_transformers import SentenceTransformer, util
from constants import *
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi
import speech_recognition as sr
import pymongo
from pymongo.errors import ConnectionFailure
from rake_nltk import Rake
import nltk
import os

TEMPLATE_DIR = os.path.abspath('templates')
STATIC_DIR = os.path.abspath('styles')
app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
r = sr.Recognizer()
model = SBertSummarizer('paraphrase-MiniLM-L6-v2')
summarizer = pipeline('summarization')
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 240 * 1024 * 1024


def mongo_ingestion():
    """Establishing Mongo Connection"""
    try:
        url = "mongodb+srv://laxmi:j3HPLDZxH6KTgeyB@cluster0.zgqcixi.mongodb.net"
        client = pymongo.MongoClient(url)
        db = client.Video_Summarization
        print("MongoDB connected successfully to timeseries db")
    except ConnectionFailure:
        print("Failed to connect To MongoDB")
        client.close()
    collection = db.Video_Sum
    return collection


@app.route("/search", methods=["POST"])
def search():
    """
    >> reference link : https://www.sbert.net/docs/pretrained_models.html
    """


    query_embedding = model.encode('How big is London')
    passage_embedding = model.encode(['London has 9,787,426 inhabitants at the 2011 census',
                                      'London is known for its finacial district'])
    collection = mongo_ingestion()
    unique_id = []
    for data in collection.find():
        data_corpous = data["video_text_corpous"]
        if util.dot_score(query_embedding, data_corpous) > 0.6:
            unique_id.append(data["id"])
    print(unique_id)
    print("Similarity:", util.dot_score(query_embedding, passage_embedding))


@app.route("/video_data", methods=["GET"])
def video_data():
    return render_template("Video_data.html")


def get_vid_summ(vid_url):
    try:
        result = ""
        video_id = vid_url.split("=")[1]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)


        for i in transcript:
            result += ' ' + i['text']
        print(len(result))

        num_iters = int(len(result) / 1000)

        summarized_text = []
        for i in range(0, num_iters + 1):
            start = 0
            start = i * 1000
            end = (i + 1) * 1000

            print("input text:" + result[start:end])

            out = summarizer(result[start:end])
            out = out[0]
            out = out['summary_text']

            print("summarized text:" + out)
            summarized_text.append(out)
        return summarized_text
    except Exception as e:
        print(e)


def get_entities(summary):
    try:
        r = Rake()
        r.extract_keywords_from_text(summary)
        res = r.get_ranked_phrases()[0:5]
        return res
    except Exception as e:
        print(e)




@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        video_url = request.form.get("url")
        text_generation = get_vid_summ(video_url)
        print("Speech recognizer activated:::::::::::::")
        text_summ = summ(text_generation[0])
        entities = get_entities(text_summ)
        db_data = dict()
        db_data["id"] = 1
        db_data["video_url"] = video_url
        db_data["video_text_corpous"] = text_generation
        db_data["video_summary"] = text_summ
        db_data["video_entities"] = entities
        collection = mongo_ingestion()
        collection.insert_one(db_data)
        return render_template('Video_data.html', context={"text_generation": text_generation, "text_summ": text_summ,"entities":entities})


@app.route("/", methods=["GET"])
def main_page():
    return render_template("upload_file.html")


def summ(body):
    k_value = len(body.split("."))
    try:
        optimal_lines = model.calculate_optimal_k(body, k_max=k_value//2)
        result = model(body, num_sentences=optimal_lines)
        return result
    except Exception as e:
        print(e)


if __name__ == "__main__":
    app.run(debug=True)
