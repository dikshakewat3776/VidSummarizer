# VidSummarizer

## System Overview

To build a video summarization model to extract a summary of a video. Summarized video content
to understand the video content without watching a video. To use summarised text to build a
video search engine, which will help to search a video just by the video’s description.

<img src=https://user-images.githubusercontent.com/48310000/188303366-af22bbf8-6bb0-4488-aebd-d3eb4bff1c21.png width=75% height=75%>

## Video Summarization Overview

<img src=https://user-images.githubusercontent.com/48310000/188303431-9d67aab7-727f-4257-b69a-340cfe17960e.png width=75% height=75%>


<img src=https://user-images.githubusercontent.com/48310000/188308245-85ea56b4-d1ab-4b71-888c-bec68c37b983.png width=75% height=75%>


<img src=https://user-images.githubusercontent.com/48310000/188308271-7d6771b2-02d7-45de-9595-8c037febf801.png width=75% height=75%>


### Demo Video Link
https://drive.google.com/file/d/1HuZycF3OiJ9CNuzk-7FXRSD4YWMTVG1b/view?usp=drivesdk


### Video Summarizer

<img src=https://user-images.githubusercontent.com/48310000/188303868-1f09e2f2-9447-45e2-b220-ad77b1d21b6e.png width=75% height=75%>

### Data 
We have used MSVD dataset by Microsoft obtained from
<a href = https://www.oreilly.com/library/view/intelligent-projects-using/9781788996921/6c15342b-1fd9-450d-b5fb-9309634597ec.xhtml>here</a>

This dataset was used to train our sequence-to-sequence video captioning model. The dataset is divided into training_data and testing_data. Each folder contains video that will be used for training as well as testing.
These folders also consist of features of video along with training and testing label json files. These json files contain the caption for each feature detected.

### Feature Extraction
1. We use this model for when user uploads a video to break the video into a sequence of frames and clusters the frames to create features per video.
2. The number of frames vary as per the length of videos and we store them in extracted_frames.
3. These frames are passed through a pre-trained VGG16 model saved as numpy arrays.
4. We pre-process the data to identify beginning and ending of the sentences and generate captions that is video text generation.

### Model Training
1. The model is trained on a sequence-to-sequence architecture. Since we are training sequence of images we use LSTMs.
2. LSTMs play a significant role in encoder and decoder where features are given to encoder as an input.

We run the model by:
1. Extracting features: python extract_features.py
2. Predict: python predict_realtime.py to perform the captioning in real time.
Note: This model works well only with videos similar to trained videos.Note: This model works well only with videos similar to trained videos
3. We elevate the transcription by using YouTube Transcript API for subtitles and transcripts for translations.

### Summarizer
1. We use pytorch transformers to run extractive summarizations. We embed the sentences using the Bert Summarizer to find optimal sentemces.
2. We use Rapid Automatic Keyword Extraction algorithm for keyword extraction algorithm which tries to determine key phrases in a body of text by analyzing the frequency of word appearance and its co-occurance with other words in the text.
3. This keyword extraction helps us in storing most relevant summaries based on their ranks and relevance scores to the subject.
4. We also store entity based keywords of a video for query engine.








