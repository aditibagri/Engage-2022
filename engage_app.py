from sklearn import linear_model 
import math 
import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from streamlit_option_menu import option_menu
from PIL import Image




st.set_page_config(layout="wide")
scatter_column, settings_column = st.columns((4, 1))

settings_column.title("Settings")

uploaded_file = settings_column.file_uploader("Choose File",type = ['csv'])






#pcamaker function
#function

def pca_maker(data_import):
    numerical_columns_list = []
    categorical_columns_list = []

    for i in data_import.columns:
        if data_import[i].dtype == np.dtype("float64") or data_import[i].dtype == np.dtype("int64"):
            numerical_columns_list.append(data_import[i])
        else:
            categorical_columns_list.append(data_import[i])

    numerical_data = pd.concat(numerical_columns_list, axis=1)
    categorical_data = pd.concat(categorical_columns_list, axis=1)

    numerical_data = numerical_data.apply(lambda x: x.fillna(np.mean(x)))

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(numerical_data)


    pca = PCA()
    pca_data = pca.fit_transform(scaled_values)
    pca_data = pd.DataFrame(pca_data)

    new_column_names = ["PCA_" + str(i) for i in range(1, len(pca_data.columns) + 1)]
    column_mapper = dict(zip(list(pca_data.columns), new_column_names))

    pca_data = pca_data.rename(columns=column_mapper)
    output = pd.concat([data_import, pca_data], axis=1)
    return output, list(categorical_data.columns), new_column_names
     
    
#scatter plot function
#scatter plot
def scatter_plot() :
 scatter_column.title("MULTI DIMENSIONAL ANALYSIS")
 if uploaded_file is not None:
    data_import= pd.read_csv(uploaded_file)   
    @st.cache
    def ret_time(data_import):
      time.sleep(5)
      return time.time()

    if st.checkbox("cache data"):
      st.write(ret_time(1))

    pca_data, cat_cols, pca_cols = pca_maker(data_import)
    
    scatter_column.markdown("The following graph gives clusters of similar samples.")

    categorical_variable = settings_column.selectbox("Variable Select", options = cat_cols)
    categorical_variable_2 = settings_column.selectbox("Second Variable Select", options = cat_cols)

    pca_1 = settings_column.selectbox("First Principle Component", options=pca_cols, index=0)
    pca_cols.remove(pca_1)
    pca_2 = settings_column.selectbox("Second Principle Component", options=pca_cols)

    #this scatter plot will show how similar the samples are i.e. the closer two points are on the graph more similarities they possess
      
    scatter_column.plotly_chart(px.scatter(data_frame=pca_data, x=pca_1, y=pca_2, color=categorical_variable, template="simple_white", height=800, hover_data = [categorical_variable_2]), use_container_width=True)
     
    
 else:
    st.warning("Please upload a .csv file")

def sales_chart() :
     scatter_column.title("SALES CHART")
     if uploaded_file is not None :
         scatter_column.markdown("Analyse the car monthly or total car sales of each model or company by the graph shown below .")
         scatter_column.text('')
         data_import = pd.read_csv(uploaded_file)
         scatter_column.dataframe(data_import,width=650,height=400)       
         
         groupby_column = st.selectbox(
         'Analyse sales of:',('Model' ,'Company')
        
         )

         # -- GROUP DATAFRAME
         output=st.selectbox('Sales Month :',('Nov-21-Sales',
         'Dec-21-Sales',	'Jan-22-Sales','Feb-22-Sales','Mar-22-Sales','Total Sales'
         ))
         output_columns = [output]

         df_grouped = data_import.groupby(by=[groupby_column], as_index=False)[output_columns].sum()

         # -- PLOT DATAFRAME
         fig = px.bar(
           df_grouped,
           x=groupby_column,
           y=output,
           template='plotly_white',
           title=f'<b>Sales of {groupby_column}</b>'
           )
         fig.update_layout(height=800,width=800)
         st.plotly_chart(fig)

     else:
       st.warning("Please upload a .csv file")

def filter_data():
 scatter_column.title("DATA SET")
 if uploaded_file is not None:
     
     scatter_column.markdown("Given below is the complete database of the sales and specification of the cars \nYou can filter and sort the data you wish to view . ")
     scatter_column.text("Note : To sort data click on the column heading of the column you wish to sort. ")
     st.markdown('APPLY FILTERS :') 
     data_import = pd.read_csv(uploaded_file)
     company_name = data_import['Company'].unique().tolist()
     price = data_import['AveragePrice_inINRLakhs'].unique().tolist()
    
     price_selection = st.slider('Average Price:',
                        min_value= min(price),
                        max_value= max(price),
                        value=(min(price),max(price)))

     company_selection = st.multiselect('Company:',
                                    company_name,
                                    default=company_name)

# --- FILTER DATAFRAME BASED ON SELECTION
     mask = (data_import['AveragePrice_inINRLakhs'].between(*price_selection)) & (data_import['Company'].isin(company_selection))
     number_of_result = data_import[mask].shape[0]
     st.markdown(f'*Available Results: {number_of_result}*')
     
     scatter_column.dataframe(data_import[mask],width=650,height=400)  

 else:
         st.warning("Please upload a .csv file")
      

    

def predict_price() :
    scatter_column.title("PREDICT THE PRICE")

        
    if uploaded_file is not None:  
         scatter_column.markdown("Need to find the price of car that have the specifications you require!!! \nDon't worry just the specifications below and click on the button :)") 
         data_import = pd.read_csv(uploaded_file)
         data_import. to_csv (uploaded_file)
         reg=linear_model.LinearRegression()

         Horsepower = scatter_column.number_input('Enter Horsepower (in PS)')
         Torque = scatter_column.number_input('Enter Torque (in Nm)')
         Mileage  = scatter_column.number_input('Enter Mileage (in kmpl)')

         reg.fit(data_import[['Horsepower(PS)','Torque(Nm)','Mileage (kmpl)']],data_import.AveragePrice_inINRLakhs)
 
         #reg.coef_
         #reg.intercept_ 
 
            
         
         if scatter_column.button('Predict Price'):
                 scatter_column.text("Note:The price is in Lakhs(INR)")
                 prediction = reg.predict([[Horsepower,Torque,Mileage]])
                 scatter_column.success(prediction)
    else:
         st.warning("Please upload a .csv file")

              
   

#sidebar

with st.sidebar:
            selected = option_menu(
                menu_title="Main Menu",  # required
                options=["Home","Data Set", "Multi Dimensional Analysis","Price Predictor","Sales Chart"],  # required
                icons=["house","table","graph-up","currency-bitcoin","clipboard-data"],  # optional
                menu_icon="cast",  # optional
                default_index=0,  # optional
            )
if selected== "Home":
    scatter_column.title("HOME")
    scatter_column.subheader("WELCOME!!!")
    scatter_column.markdown("This site provides the sales data of cars and companies and has multiple other features like sales analysis ,price prediction and data analysis build using extensive machine learning algorithms .")
    image = Image.open(r'car.jpg')
    scatter_column.image(image)

if selected == "Data Set":
    filter_data()
    
if selected == "Multi Dimensional Analysis":
    scatter_plot()
    
if selected == "Price Predictor":    
    predict_price()  

if selected == "Sales Chart":
    sales_chart()





st.sidebar.title("WELCOME \n This site will provide you all the information you need about the sales and specifications of the cars so that it is easy for you analyse sales and compare cars :) ")


