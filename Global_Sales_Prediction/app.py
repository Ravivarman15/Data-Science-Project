import streamlit as st
import pandas as pd 
import joblib

@st.cache_resource
def load_artifacts():
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    encoder = joblib.load('encoder.pkl')
    return model,scaler,encoder

model,scaler,encoder = load_artifacts()

num_cols = ['Price', 'Quantity', 'Discount (%)']
cat_cols = ['Region', 'Category']

st.set_page_config(page_title='Ecommerce Sales Predictiion',layout='centered')

st.title('Ecommerce Revenue Predictiion')
st.write('Enter the Details about the sales')

price=st.number_input('Sales Amount :')
quantity=st.slider('Quantity',1,5)
discount=st.slider('Discount',0,60)
region=st.selectbox('Region',['Africa','Asia','Australia','Europe','North America','South America'])
category = st.selectbox('Category',['Automotive','Books','Electronics','Fashion','Health & Beauty','Home & Kitchen','Sports & Outdoors','Toys & Games'])

input_data={
    'Price':price, 
    'Quantity':quantity, 
    'Discount (%)':discount,
    'Region':region, 
    'Category':category
}

input_df =pd.DataFrame([input_data])

encode_array = encoder.transform(input_df[cat_cols])
encoded_df = pd.DataFrame(
    encode_array,
    columns=encoder.get_feature_names_out(cat_cols),
    index=input_df.index
)

for col in scaler:
    if col in input_df.columns:
        input_df[col]=scaler[col].transform(input_df[[col]])

input_df = input_df.drop(columns=cat_cols)
input_df = pd.concat([input_df, encoded_df], axis=1)

if st.button("Predict"):
    prediction = model.predict(input_df)[0]

    st.metric(
        label="📈 Predicted Sales Value",
        value=f"{prediction:,.2f}"
    )


st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
    }
    label {
        font-size: 0.85rem !important;
    }
    .stButton>button {
        padding: 0.25rem 0.75rem;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)



