import streamlit as st
import joblib
import numpy as np
import re

model=joblib.load(open(r'E:\survey pro\best_model_catboost.pkl','rb'))
features=[
    'What is your gender?',
    'What is your age?',
    'What is your current academic level?',
    'What is your current CGPA?',
    'Are you aware of your university’s career services?',
    'What is your department?',
    'How many internships have you completed during your studies?',
    'Which types of internships have you completed during your studies, and how many for each type?  [Virtual/Remote Internship]',
    'Which types of internships have you completed during your studies, and how many for each type?  [Industry/Corporate Internship]',
    'Which types of internships have you completed during your studies, and how many for each type?  [Government Internship]',
    'How many hours per week do you spend on extracurricular ( Non-academic / Supplementary )activities?',
    'On average, how many hours per week do you spend on self-learning (such as online courses, tutorials, or independent study)?',
    'Have you attended any career-related workshops or training sessions?',
    'Do you have a part-time job while studying? If yes, how many hours per week do you work?',
    'On a scale of 0 to 5, how confident are you in your ability to secure a job after graduation, considering your skills, experience, and job market conditions?',
    'In the past 6 months, how many job applications have you submitted?',
    'How long do you expect it will take to secure a job after graduation?',
    'Have you received any job offers before graduation?',
    'What is your expected starting salary after graduation? Please enter the amount in USD ($).',
    'To what extent do you think your university’s career services have helped you prepare for the job market?',
    'On a scale of 1-5, how relevant do you think your university courses are to real-world job requirements?',
    'Which of the following skills do you think employers value the most in your field?',
    'How would you rate the amount of hands-on training provided by your university?',
    'On a scale of 1-5, how important do you think networking is in securing a job?',
    'How many certificates have you achieved so far?',
    'Which of the following career paths do you prefer?',
    'Reflecting on your studies, which technical skills do you feel most comfortable using or applying?',
    'Which elective course have you found most helpful for your career preparation and readiness?',
    'Have you taken any courses outside of the university that you found particularly impactful that should be added to the university curriculum?',
    'Which professional or technical skills do you feel are missing from your university education?',
    'What specific improvements would you like to see in your university’s career preparation programs?',
    'cleaned_skills',
    'student_skill_score'
]
cat_features=[
    'What is your gender?',
    'What is your age?',
    'What is your current academic level?',
    'Are you aware of your university’s career services?',
    'What is your department?',
    'How many internships have you completed during your studies?',
    'On average, how many hours per week do you spend on self-learning (such as online courses, tutorials, or independent study)?',
    'Have you attended any career-related workshops or training sessions?',
    'Do you have a part-time job while studying? If yes, how many hours per week do you work?',
    'In the past 6 months, how many job applications have you submitted?',
    'How long do you expect it will take to secure a job after graduation?',
    'Have you received any job offers before graduation?',
    'What is your expected starting salary after graduation? Please enter the amount in USD ($).',
    'To what extent do you think your university’s career services have helped you prepare for the job market?',
    'Which of the following skills do you think employers value the most in your field?',
    'How would you rate the amount of hands-on training provided by your university?',
    'How many certificates have you achieved so far?',
    'Which of the following career paths do you prefer?',
    'Reflecting on your studies, which technical skills do you feel most comfortable using or applying?',
    'Which elective course have you found most helpful for your career preparation and readiness?',
    'Have you taken any courses outside of the university that you found particularly impactful that should be added to the university curriculum?',
    'Which professional or technical skills do you feel are missing from your university education?',
    'What specific improvements would you like to see in your university’s career preparation programs?',
    'cleaned_skills'
]

def sanitize_filename(name):
    return re.sub(r'[<>:"/\\|?*]', '_', name)

def load_encoder(feature):
    safe_name = sanitize_filename(feature)
    return joblib.load(open(f'E:\\survey pro\\{safe_name}_encoder.pkl', 'rb'))


def main():
     st.set_page_config(layout='wide')
     st.title("pridect career readnies")
     inputs = {}
     for feature in features:
          if feature in cat_features:
               en=load_encoder(feature)
               if hasattr(en, 'categories_'):
                    options = en.categories_[0].tolist()
                    inputs[feature] = st.selectbox(feature, options)
               elif hasattr(en, 'classes_'):
                    options = en.classes_.tolist()
                    inputs[feature] = st.selectbox(feature, options)
          else:
                inputs[feature] = st.text_input(feature)

     if st.button('predict'):
        input_values = []
        for feature in features:
            value = inputs[feature]
            if feature in cat_features:
                en = load_encoder(feature)
                value = en.transform(np.array([[value]]))[0]
            input_values.append(value)

        input_values = np.array(input_values, dtype='object').reshape(1, -1)
        y_pred = model.predict(input_values)
        if y_pred == 1:
            st.success('Ready.')
        else:
            st.error('Not Ready')

if __name__ == '__main__':
    main()

               


