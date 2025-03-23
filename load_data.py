import pandas as pd

def load_data(csv_file):
    df = pd.read_csv(csv_file)
    df.dropna(inplace=True)
    questions = df['question'].tolist()
    answers = df['answer'].tolist()
    return questions, answers

csv_file = "qa.csv"
questions, answers = load_data(csv_file)