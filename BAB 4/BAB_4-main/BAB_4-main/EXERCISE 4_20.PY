
# Exercise 4.20
from transformers import pipeline

question_answerer = pipeline('question-answering', framework='pt')

result = question_answerer(
    question='What does the student study?',
    context='Sarah is a college student. She studies multimedia technology and enjoys learning about artificial intelligence.',
)

print(result)
