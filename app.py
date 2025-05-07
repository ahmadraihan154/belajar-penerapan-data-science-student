import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')
encoder = joblib.load('encoder.joblib')
selector = joblib.load('selector.joblib')


def predict_status(input_data):
    # Fitur kategorikal yang perlu di-encode
    categorical_features = [
        'Marital_status', 'Application_mode', 'Application_order', 'Course',
        'Daytime_evening_attendance', 'Previous_qualification',
        'Mothers_qualification', 'Fathers_qualification', 'Mothers_occupation',
        'Fathers_occupation', 'Displaced', 'Debtor', 'Tuition_fees_up_to_date',
        'Gender', 'Scholarship_holder'
    ]
   
    # Simpan hasil encoding + fitur numerik
    encoded_input = []
    for feature in categorical_features:
        value = input_data[feature]
       
        # Tangani jika value tidak dikenal oleh encoder
        if value not in encoder[feature].classes_:
            value = encoder[feature].classes_[0]  # Default ke kelas pertama
       
        # Transformasi label
        encoded_value = encoder[feature].transform([value])[0]
        encoded_input.append(encoded_value)
        
    # Tambahkan fitur numerik (dalam urutan yang benar!)
    numeric_features = [
        'Age_at_enrollment', 'Admission_grade', 'Previous_qualification_grade',
        'Curricular_units_1st_sem_enrolled', 'Curricular_units_1st_sem_approved',
        'Curricular_units_1st_sem_grade', 'Curricular_units_1st_sem_evaluations',
        'Curricular_units_1st_sem_credited', 'Curricular_units_1st_sem_without_evaluations', 
        'Curricular_units_2nd_sem_credited', 'Curricular_units_2nd_sem_enrolled', 'Curricular_units_2nd_sem_approved',
        'Curricular_units_2nd_sem_grade', 'Curricular_units_2nd_sem_evaluations',
        'Curricular_units_2nd_sem_without_evaluations', 'Unemployment_rate', 'Inflation_rate', 'GDP'
    ]
    
    for feature in numeric_features:
        encoded_input.append(input_data[feature])
        
    # Ubah ke array dan reshape
    input_array = np.array(encoded_input).reshape(1, -1)
    
    # Scaling
    scaled_input = scaler.transform(input_array)
    
    # Feature selection (jika digunakan)
    selected_input = selector.transform(scaled_input)
    
    # Prediksi
    prediction = model.predict(selected_input)
    return prediction

# Streamlit UI
st.title('Welcome to the Student prediction app!')
st.write('This app predicts the status of a student based on various input features')



# Input second semester curricular units
curricular_units_2nd_sem_approved = st.number_input('Curricular Units 2nd Semester Approved', min_value=0, max_value=20, placeholder="Enter number of approved curricular units")
curricular_units_2nd_sem_grade = st.number_input('Curricular Units 2nd Semester Grade', min_value=0, max_value=20, placeholder="Enter grade for curricular units")
curricular_units_2nd_sem_evaluations = st.number_input('Curricular Units 2nd Semester Evaluated', min_value=0, max_value=35, placeholder="Enter number of evaluated curricular units")
curricular_units_2nd_sem_enrolled = st.number_input('Curricular Units 2nd Semester Enrolled', min_value=0, max_value=25, placeholder="Enter number of enrolled curricular units")
curricular_units_2nd_sem_without_evaluations = st.number_input('Curricular Units 2nd Semester Without Evaluated', min_value=0, max_value=20, placeholder="Enter number of curricular units without evaluated")
curricular_units_2nd_sem_credited = st.number_input('Curricular Units 2nd Semester Credited', min_value=0, max_value=20, placeholder="Enter number of curricular units credited")

# Input first semseter curricular units
curricular_units_1st_sem_approved = st.number_input('Curricular Units 1st Semester Approved', min_value=0, max_value=20, placeholder="Enter number of approved curricular units")
curricular_units_1st_sem_grade = st.number_input('Curricular Units 1st Semester Grade', min_value=0, max_value=20, placeholder="Enter grade for curricular units")
curricular_units_1st_sem_evaluations = st.number_input('Curricular Units 1st Semester Evaluated', min_value=0, max_value=35, placeholder="Enter number of evaluated curricular units")
curricular_units_1st_sem_enrolled = st.number_input('Curricular Units 1st Semester Enrolled', min_value=0, max_value=25, placeholder="Enter number of enrolled curricular units")
curricular_units_1st_sem_without_evaluations = st.number_input('Curricular Units 1st Semester Without Evaluated', min_value=0, max_value=20, placeholder="Enter number of curricular units without evaluated")
curricular_units_1st_sem_credited = st.number_input('Curricular Units 1st Semester Credited', min_value=0, max_value=20, placeholder="Enter number of curricular units credited")

# Numeric inputs
age_at_enrollment = st.number_input('Age at Enrollment', min_value=0, max_value=80, step=1, placeholder="Enter your age when you enrolled")
admission_grade = st.number_input('Admission Grade', min_value=0.0, max_value=200.0, step=0.1, placeholder="Enter admission grade")
previous_qualification_grade = st.number_input('Previous Qualification Grade', min_value=0.0, max_value=200.0, step=0.1, placeholder="Enter previous qualification grade")

# Binary inputs
tuition_fees_up_to_date = st.selectbox('Are your tuition fees up to date?', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
scholarship_holder = st.selectbox('Are you a scholarship holder?', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
debtor = st.selectbox('Do you have any outstanding debts?', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
displaced = st.selectbox('Are you a displaced student?', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

# Gender input
gender = st.selectbox('Gender', [0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')

# Marital status input
marital_status = st.selectbox(
    'Marital Status',
    [1, 2, 3, 4, 5, 6],
    format_func=lambda x:
        'Single' if x == 1 else
        'Married' if x == 2 else
        'Windower' if x == 3 else
        'Divorced' if x == 4 else
        'Facto Union' if x == 5 else
        'Legally Separated'
)

# Application mode input
application_mode = st.selectbox(
    'Application Mode',
    [1, 2, 5, 7, 10, 15, 16, 17, 18, 26, 27, 39, 42, 43, 44, 51, 53, 57],
    format_func=lambda x:
        '1st phase - general contingent' if x == 1 else
        'Ordinance No. 612/93' if x == 2 else
        '1st phase - special contingent (Azores Island)' if x == 5 else
        'Holders of other higher courses' if x == 7 else
        'Ordinance No. 854-B/99' if x == 10 else
        'International student (bachelor)' if x == 15 else
        '1st phase - special contingent (Madeira Island)' if x == 16 else
        '2nd phase - general contingent' if x == 17 else
        '3rd phase - general contingent' if x == 18 else
        'Ordinance No. 533-A/99, item b2) (Different Plan)' if x == 26 else
        'Ordinance No. 533-A/99, item b3 (Other Institution)' if x == 27 else
        'Over 23 years old' if x == 39 else
        'Transfer' if x == 42 else
        'Change of course' if x == 43 else
        'Technological specialization diploma holders' if x == 44 else
        'Change of institution/course' if x == 51 else
        'Short cycle diploma holders' if x == 53 else
        'Change of institution/course (International)'
)

# Application order input
application_order = st.selectbox(
    'Application Order',
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    format_func=lambda x: f"{x+1}th choice" if x != 0 else "1st choice"
)

# Course input
course = st.selectbox(
    'Study Program',
    [33, 171, 8014, 9003, 9070, 9085, 9119, 9130, 9147, 9238, 9254, 9500, 9556, 9670, 9773, 9853, 9991],
    format_func=lambda x: 
        'Biofuel Production Technologies' if x == 33 else
        'Animation and Multimedia Design' if x == 171 else
        'Social Service (evening attendance)' if x == 8014 else
        'Agronomy' if x == 9003 else
        'Communication Design' if x == 9070 else
        'Veterinary Nursing' if x == 9085 else
        'Informatics Engineering' if x == 9119 else
        'Equinculture' if x == 9130 else
        'Management' if x == 9147 else
        'Social Service' if x == 9238 else
        'Tourism' if x == 9254 else
        'Nursing' if x == 9500 else
        'Oral Hygiene' if x == 9556 else
        'Advertising and Marketing Management' if x == 9670 else
        'Journalism and Communication' if x == 9773 else
        'Basic Education' if x == 9853 else
        'Management (evening attendance)'
)

# Daytime/evening attendance
daytime_evening_attendance = st.selectbox(
    'Attendance Time',
    [0, 1],
    format_func=lambda x: 'Evening' if x == 0 else 'Daytime'
)

# Previous qualification input
previous_qualification = st.selectbox(
    'Previous Qualification',
    [1, 2, 3, 4, 5, 6, 9, 10, 12, 14, 15, 19, 38, 39, 40, 42, 43],
    format_func=lambda x:
        'Secondary education' if x == 1 else
        'Higher education - bachelor\'s degree' if x == 2 else
        'Higher education - degree' if x == 3 else
        'Higher education - master\'s' if x == 4 else
        'Higher education - doctorate' if x == 5 else
        'Frequency of higher education' if x == 6 else
        '12th year - not completed' if x == 9 else
        '11th year - not completed' if x == 10 else
        'Other - 11th year' if x == 12 else
        '10th year' if x == 14 else
        '10th year - not completed' if x == 15 else
        'Basic education 3rd cycle (9th–11th)' if x == 19 else
        'Basic education 2nd cycle (6th–8th)' if x == 38 else
        'Technological specialization course' if x == 39 else
        'Higher education - degree (1st cycle)' if x == 40 else
        'Professional higher technical course' if x == 42 else
        'Higher education - master (2nd cycle)'
)

# Mother's qualification input
mothers_qualification = st.selectbox(
    "Mother's Qualification",
    [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 14, 18, 19, 22, 26, 27, 29, 30, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44],
    format_func=lambda x:
        "Secondary Education - 12th Year of Schooling or Eq." if x == 1 else
        "Higher Education - Bachelor's Degree" if x == 2 else
        "Higher Education - Degree" if x == 3 else
        "Higher Education - Master's" if x == 4 else
        "Higher Education - Doctorate" if x == 5 else
        "Frequency of Higher Education" if x == 6 else
        "12th Year of Schooling - Not Completed" if x == 9 else
        "11th Year of Schooling - Not Completed" if x == 10 else
        "7th Year (Old)" if x == 11 else
        "Other - 11th Year of Schooling" if x == 12 else
        "10th Year of Schooling" if x == 14 else
        "General Commerce Course" if x == 18 else
        "Basic Education 3rd Cycle (9th/10th/11th Year)" if x == 19 else
        "Technical-Professional Course" if x == 22 else
        "7th Year of Schooling" if x == 26 else
        "2nd Cycle of the General High School Course" if x == 27 else
        "9th Year of Schooling - Not Completed" if x == 29 else
        "8th Year of Schooling" if x == 30 else
        "Unknown" if x == 34 else
        "Can't read or write" if x == 35 else
        "Can read without having a 4th Year of Schooling" if x == 36 else
        "Basic Education 1st Cycle (4th/5th Year)" if x == 37 else
        "Basic Education 2nd Cycle (6th/7th/8th Year)" if x == 38 else
        "Technological Specialization Course" if x == 39 else
        "Higher Education - Degree (1st Cycle)" if x == 40 else
        "Specialized Higher Studies Course" if x == 41 else
        "Professional Higher Technical Course" if x == 42 else
        "Higher Education - Master (2nd Cycle)" if x == 43 else
        "Higher Education - Doctorate (3rd Cycle)"
)

# Father's qualification input
fathers_qualification = st.selectbox(
    "Father's Qualification",
    [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 18, 19, 20, 22, 25, 26, 27, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44],
    format_func=lambda x:
        "Secondary Education - 12th Year of Schooling or Eq." if x == 1 else
        "Higher Education - Bachelor's Degree" if x == 2 else
        "Higher Education - Degree" if x == 3 else
        "Higher Education - Master's" if x == 4 else
        "Higher Education - Doctorate" if x == 5 else
        "Frequency of Higher Education" if x == 6 else
        "12th Year of Schooling - Not Completed" if x == 9 else
        "11th Year of Schooling - Not Completed" if x == 10 else
        "7th Year (Old)" if x == 11 else
        "Other - 11th Year of Schooling" if x == 12 else
        "2nd Year Complementary High School Course" if x == 13 else
        "10th Year of Schooling" if x == 14 else
        "General Commerce Course" if x == 18 else
        "Basic Education 3rd Cycle (9th/10th/11th Year)" if x == 19 else
        "Complementary High School Course" if x == 20 else
        "Technical-Professional Course" if x == 22 else
        "Complementary High School Course - Not Concluded" if x == 25 else
        "7th Year of Schooling" if x == 26 else
        "2nd Cycle of the General High School Course" if x == 27 else
        "9th Year of Schooling - Not Completed" if x == 29 else
        "8th Year of Schooling" if x == 30 else
        "General Course of Administration and Commerce" if x == 31 else
        "Supplementary Accounting and Administration" if x == 33 else
        "Unknown" if x == 34 else
        "Can't Read or Write" if x == 35 else
        "Can Read Without Having a 4th Year of Schooling" if x == 36 else
        "Basic Education 1st Cycle (4th/5th Year)" if x == 37 else
        "Basic Education 2nd Cycle (6th/7th/8th Year)" if x == 38 else
        "Technological Specialization Course" if x == 39 else
        "Higher Education - Degree (1st Cycle)" if x == 40 else
        "Specialized Higher Studies Course" if x == 41 else
        "Professional Higher Technical Course" if x == 42 else
        "Higher Education - Master (2nd Cycle)" if x == 43 else
        "Higher Education - Doctorate (3rd Cycle)"
)

# Mother's occupation input
mothers_occupation = st.selectbox(
    "Mother's Occupation",
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 90, 99, 122, 123, 125, 131, 132, 134, 141, 143, 144, 151, 152, 153, 171, 173, 175, 191, 192, 193, 194],
    format_func=lambda x:
        "Student" if x == 0 else
        "Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers" if x == 1 else
        "Specialists in Intellectual and Scientific Activities" if x == 2 else
        "Intermediate Level Technicians and Professions" if x == 3 else
        "Administrative staff" if x == 4 else
        "Personal Services, Security and Safety Workers and Sellers" if x == 5 else
        "Farmers and Skilled Workers in Agriculture, Fisheries and Forestry" if x == 6 else
        "Skilled Workers in Industry, Construction and Craftsmen" if x == 7 else
        "Installation and Machine Operators and Assembly Workers" if x == 8 else
        "Unskilled Workers" if x == 9 else
        "Armed Forces Professions" if x == 10 else
        "Other Situation" if x == 90 else
        "(blank)" if x == 99 else
        "Health professionals" if x == 122 else
        "Teachers" if x == 123 else
        "Specialists in Information and Communication Technologies (ICT)" if x == 125 else
        "Intermediate level science and engineering technicians and professions" if x == 131 else
        "Technicians and professionals, of intermediate level of health" if x == 132 else
        "Intermediate level technicians from legal, social, sports, cultural and similar services" if x == 134 else
        "Office workers, secretaries in general and data processing operators" if x == 141 else
        "Data, accounting, statistical, financial services and registry-related operators" if x == 143 else
        "Other administrative support staff" if x == 144 else
        "Personal service workers" if x == 151 else
        "Sellers" if x == 152 else
        "Personal care workers and the like" if x == 153 else
        "Skilled construction workers and the like, except electricians" if x == 171 else
        "Skilled workers in printing, precision instrument manufacturing, jewelers, artisans and the like" if x == 173 else
        "Workers in food processing, woodworking, clothing and other industries and crafts" if x == 175 else
        "Cleaning workers" if x == 191 else
        "Unskilled workers in agriculture, animal production, fisheries and forestry" if x == 192 else
        "Unskilled workers in extractive industry, construction, manufacturing and transport" if x == 193 else
        "Meal preparation assistants"
)

# Father's occupation input
fathers_occupation = st.selectbox(
    "Father's Occupation",
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 90, 99, 101, 102, 103, 112, 114, 121, 122, 123, 124, 131, 132, 134, 135, 141, 143, 144, 151, 152, 153, 154, 161, 163, 171, 172, 174, 175, 181, 182, 183, 192, 193, 194, 195],
    format_func=lambda x:
        "Student" if x == 0 else
        "Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers" if x == 1 else
        "Specialists in Intellectual and Scientific Activities" if x == 2 else
        "Intermediate Level Technicians and Professions" if x == 3 else
        "Administrative staff" if x == 4 else
        "Personal Services, Security and Safety Workers and Sellers" if x == 5 else
        "Farmers and Skilled Workers in Agriculture, Fisheries and Forestry" if x == 6 else
        "Skilled Workers in Industry, Construction and Craftsmen" if x == 7 else
        "Installation and Machine Operators and Assembly Workers" if x == 8 else
        "Unskilled Workers" if x == 9 else
        "Armed Forces Professions" if x == 10 else
        "Other Situation" if x == 90 else
        "(blank)" if x == 99 else
        "Armed Forces Officers" if x == 101 else
        "Armed Forces Sergeants" if x == 102 else
        "Other Armed Forces personnel" if x == 103 else
        "Directors of administrative and commercial services" if x == 112 else
        "Hotel, catering, trade and other services directors" if x == 114 else
        "Specialists in the physical sciences, mathematics, engineering and related techniques" if x == 121 else
        "Health professionals" if x == 122 else
        "Teachers" if x == 123 else
        "Specialists in finance, accounting, administrative organization, public and commercial relations" if x == 124 else
        "Intermediate level science and engineering technicians and professions" if x == 131 else
        "Technicians and professionals, of intermediate level of health" if x == 132 else
        "Intermediate level technicians from legal, social, sports, cultural and similar services" if x == 134 else
        "Information and communication technology technicians" if x == 135 else
        "Office workers, secretaries in general and data processing operators" if x == 141 else
        "Data, accounting, statistical, financial services and registry-related operators" if x == 143 else
        "Other administrative support staff" if x == 144 else
        "Personal service workers" if x == 151 else
        "Sellers" if x == 152 else
        "Personal care workers and the like" if x == 153 else
        "Protection and security services personnel" if x == 154 else
        "Market-oriented farmers and skilled agricultural and animal production workers" if x == 161 else
        "Farmers, livestock keepers, fishermen, hunters and gatherers, subsistence" if x == 163 else
        "Skilled construction workers and the like, except electricians" if x == 171 else
        "Skilled workers in metallurgy, metalworking and similar" if x == 172 else
        "Skilled workers in electricity and electronics" if x == 174 else
        "Workers in food processing, woodworking, clothing and other industries and crafts" if x == 175 else
        "Fixed plant and machine operators" if x == 181 else
        "Assembly workers" if x == 182 else
        "Vehicle drivers and mobile equipment operators" if x == 183 else
        "Unskilled workers in agriculture, animal production, fisheries and forestry" if x == 192 else
        "Unskilled workers in extractive industry, construction, manufacturing and transport" if x == 193 else
        "Meal preparation assistants" if x == 194 else
        "Street vendors (except food) and street service providers"
)

# Input economic data
unemployment_rate = st.number_input('Unemployment Rate', min_value=0.0, max_value=100.0, value=5.0, step=0.1, placeholder="Enter unemployment rate")
inflation_rate = st.number_input('Inflation Rate', min_value=0.0, max_value=100.0, value=3.0, step=0.1, placeholder="Enter inflation rate")
gdp = st.number_input('GDP', min_value=-10.0, max_value=10.0, value=3.0, step=0.1, placeholder="Enter GDP")

# Map the inputs to the format expected by the model
input_data = {
    'Curricular_units_2nd_sem_approved': curricular_units_2nd_sem_approved,
    'Curricular_units_2nd_sem_grade': curricular_units_2nd_sem_grade,  
    'Curricular_units_2nd_sem_evaluations': curricular_units_2nd_sem_evaluations,  
    'Curricular_units_2nd_sem_enrolled': curricular_units_2nd_sem_enrolled,
    'Curricular_units_2nd_sem_credited': curricular_units_2nd_sem_credited,  
    'Curricular_units_2nd_sem_without_evaluations': curricular_units_2nd_sem_without_evaluations,  
    'Curricular_units_1st_sem_approved': curricular_units_1st_sem_approved,
    'Curricular_units_1st_sem_grade': curricular_units_1st_sem_grade,
    'Curricular_units_1st_sem_evaluations': curricular_units_1st_sem_evaluations,  
    'Curricular_units_1st_sem_enrolled': curricular_units_1st_sem_enrolled,
    'Curricular_units_1st_sem_credited': curricular_units_1st_sem_credited,  
    'Curricular_units_1st_sem_without_evaluations': curricular_units_1st_sem_without_evaluations,  
    'Tuition_fees_up_to_date': tuition_fees_up_to_date,
    'Scholarship_holder': scholarship_holder,
    'Debtor': debtor,
    'Gender': gender,
    'Course': course,
    'Daytime_evening_attendance': daytime_evening_attendance,
    'Marital_status': marital_status,
    'Application_mode': application_mode,
    'Application_order': application_order,
    'Admission_grade': admission_grade,
    'Previous_qualification': previous_qualification,
    'Previous_qualification_grade': previous_qualification_grade,
    'Mothers_qualification': mothers_qualification,
    'Fathers_qualification': fathers_qualification,
    'Mothers_occupation': mothers_occupation,
    'Fathers_occupation': fathers_occupation,
    'Displaced': displaced,
    'Unemployment_rate': unemployment_rate,
    'Inflation_rate': inflation_rate,
    'GDP': gdp,
    'Age_at_enrollment': age_at_enrollment  
}

import numpy as np
import streamlit as st

if st.button('Predict'):
    prediction = predict_status(input_data)
    prediction_array = np.array(prediction)


    status_dict = {
        0: 'Dropout',   # Gantilah label sesuai makna aslinya
        1: 'Enrolled',
        2: 'Graduate'
    }

    # Tangani format output
    if prediction_array.ndim == 1:
        # Prediksi berupa class langsung atau 1D probabilitas
        if np.all(prediction_array >= 0) and np.all(prediction_array <= 1):
            # Probabilitas
            predicted_status_index = np.argmax(prediction_array)
        else:
            # Sudah berupa kelas
            predicted_status_index = int(prediction_array[0])

    elif prediction_array.ndim == 2:

        predicted_status_index = np.argmax(prediction_array, axis=1)[0]
    else:
        st.error("Format prediksi tidak dikenali")
        st.stop()

    predicted_status = status_dict.get(predicted_status_index, "Unknown")
    st.write(f"The model predicts that the student is likely to be: **{predicted_status}**")