import time
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import altair as alt
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input
from PIL import Image



st.set_page_config(layout="wide")  # trình bày full trang 

st.title("CLINICAL DIAGNOSTIC SYSTEM FOR PNEUMONIA & COVID-19")
st.subheader("""Author : Vo Trong Luan """)
st.subheader('ID Student : 20146505')
st.subheader('Contact: luanvoak@gmail.com')



model = tf.keras.models.load_model('pneumonia.h5')   # load model


# tạo khung để upload file xử lý
uploaded_file = st.file_uploader("CHOOSE A X-RAY IMAGE FILE", type=["jpg","jpeg","png"])



#ma trận để thay thế cho việc tính laplace để giúp cpu tính toán nhanh hơn
k=[[0,1,0],[1,-4,1],[0,1,0]]

#tạo hàm làm sắc nét ảnh
def Sacnet(imgPIL):
    sacnet = Image.new(imgPIL.mode, imgPIL.size)
    width = sacnet.size[0]
    height = sacnet.size[1]
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            rs = 0
            gs = 0
            bs = 0
            rs1 = 0
            gs1 = 0
            bs1 = 0
            a = 0
            b = 0
            for i in range(x - 1, x + 1 + 1):
                for j in range(y - 1, y + 1 + 1):
                    color = imgPIL.getpixel((i, j))
                    R = color[0]
                    G = color[1]
                    B = color[2]
                    rs += R * k[a][b]
                    gs += G * k[a][b]
                    bs += B * k[a][b]
                    b += 1
                    if b == 3:
                        b = 0
                a += 1
                if a == 3:
                    a = 0
            R1, G1, B1 = imgPIL.getpixel((x, y))
            rs1 = R1 - rs
            gs1 = G1 - gs
            bs1 = B1 - bs

            if rs1 > 255:
                rs1 = 255
            elif rs1 < 0:
                rs1 = 0
            if gs1 > 255:
                gs1 = 255
            elif gs1 < 0:
                gs1 = 0
            if bs1 > 255:
                bs1 = 255
            elif bs1 < 0:
                bs1 = 0

            sacnet.putpixel((x, y), (bs1, gs1, rs1))
    return sacnet




    
if uploaded_file is not None:   # kiểm tra có file xử lý hay không 
    img = image.load_img(uploaded_file,target_size=(300,300))
    # chia đôi giao diện
    col1, col2 = st.columns(2) 
    with col1:
        st.write('**X-RAY IMAGE NON-PROCESS**')
        st.image(img, channels="RGB")   # hiển thị ảnh
        Process = st.button("**Pre-process & Predict**")

    if Process:
        #img_array = img_to_array(img)
        #img_array= np.array(img_array)
        img_array = Sacnet(img)
        img = image.array_to_img(img_array)

        with col2:
            st.write('**X-RAY IMAGE IS PROCESSED**')
            st.image(img, channels="RGB")
            img = img.resize((150,150))
            img = img_to_array(img)
            img = img.reshape(1,150,150,3)
            img = img.astype('float32')
            img = img / 255

            with st.spinner("Waiting !!!"):
                time.sleep(2)

            result = int(np.argmax(model.predict(img),axis =1))
            percent = model.predict(img)

            if result == 0:
                st.write("**Based on the x-ray image it is COVID19**")
                percent = (percent.max())*100
                st.write("**Accuracy:** ",percent,"%")
                st.write("""** You have been diagnosed with COVID-19.
                The Centers for Disease Control and Prevention (CDC) recommends that :
                you stay home except to get medical care. You should monitor your symptoms carefully and if they get worse, call your healthcare provider immediately.
                Get rest and stay hydrated. Sleep and rest as much as possible. Feeling weak and tired for a while is normal, but your energy will return over time.
                Keep track of your symptoms, which may include fever, cough, loss of taste and smell, difficulty breathing, among others**""")
            elif result == 1 :
                st.write("**Based on the x-ray image it is HEALTHY**")
                percent = (percent.max())*100
                st.write("**Accuracy:** ", percent,"%")
            else :
                st.write("**Based on the x-ray image it is PNEUMOIA**")
                percent = (percent.max())*100
                st.write("**Accuracy:** ", percent,"%")
                st.write("""** You have been diagnosed with PNEUMONIA.
                Most people who have pneumonia will be able to stay home. If your symptoms haven’t improved within the first 5 days of taking antibiotics or your symptoms get worse, contact the doctor. Sometimes you may need a change in the dose or type of antibiotic, or you may need more than one medicine.
                Some people will need to be treated in hospital. This is more common for people who are very old, very young or who have other illnesses. A person in hospital for pneumonia may need oxygen therapy, or other more intense forms of treatment.
                Getting plenty of rest, drinking plenty of fluids and taking paracetamol for the fever are also important. Some people may also need physiotherapy to help clear their lungs.
                Cough medicine is not recommended for people with pneumonia. Coughing can help move mucous plugs from the tubes and help clear the infection.
                People with pneumonia should quit smoking and keep well away from things that will irritate their lungs, such as smoke. Drink plenty of fluids and get lots of rest to help you recover.
                **""")

