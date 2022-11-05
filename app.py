import pandas as pd
import streamlit as st
from pycaret.regression import load_model, predict_model

st.set_page_config(page_title="Beneficio en Supermercado")

@st.cache(allow_output_mutation=True)
def get_model():
    return load_model('super')

def predict(model, df):
    predictions = predict_model(model, data = df)
    return predictions['Label'][0]

model = get_model()

st.title("Benefits SuperMarket App")
st.markdown("Enter your personal details to get a prediction of your supermarket\
    Developed by: Yesner\
        LinkedIn: https://www.linkedin.com/in/yesner-salgado/")

form = st.form("charges")

gasto_publicidad = form.slider('Gasto de publicidad', min_value=0, max_value=200000, value=0)
gasto_promocion = form.slider('Gastos de promoción', min_value=0, max_value=200000, value=0)
gasto_administracion = form.slider('Gastos de administración', min_value=0, max_value=200000, value=0)
region_list = ['New York', 'California', 'Florida']
region = form.selectbox('Region', region_list)

predict_button = form.form_submit_button('Predict')

input_dict = {'gasto_publicidad' : gasto_publicidad, 'gasto_promocion' : gasto_promocion, 'gasto_administracion' : gasto_administracion,
'region' : region}
input_df = pd.DataFrame([input_dict])

if predict_button:
 out = predict(model, input_df)
 st.success('The benefit are ${:.2f}'.format(out))
