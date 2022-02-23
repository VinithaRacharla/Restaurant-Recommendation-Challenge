import numpy as np
import streamlit as st
import pickle

final_model = pickle.load(open('weighted_logit.pkl', 'rb'))

def Restaurant_recommendation(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    reshaped_ip_data = input_data_as_numpy_array.reshape(1,-1)
    prediction = final_model.predict(reshaped_ip_data)
    print(prediction)

    if (prediction[0] == 0):
      return 'The restaurant is not recommended'
    else:
      return 'The restaurant is recommended'

def main():
    st.title('Restaurant Recommendation Challenge')
    customer_id = st.text_input('Give Customer ID ')
    gender = st.text_input('Gender')
    location_number = st.text_input('location number')
    id = st.text_input('ID')
    vendor_category_id = st.text_input('Vendor category ID')
    primary_tags = st.text_input('Primary tag Number')
    city_id = st.text_input('City ID')
    device_type = st.text_input('Device Type')
    display_orders = st.text_input('Number of orders displayed')
    vendor_rating = st.text_input('Rating of the vendor')
    
    recommendation=''
    if st.button('SUBMIT'):
        recommendation = Restaurant_recommendation([customer_id, gender, location_number, id, vendor_category_id, primary_tags, city_id, device_type,display_orders,vendor_rating])
    st.success(recommendation)
if __name__ == '__main__':
    main()
