import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from datetime import datetime

# Set page configuration for a professional look
st.set_page_config(page_title="AgriPreidct", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for enhanced styling
st.markdown("""
    <style>
    body {
        background-color: #f0f4f8;
    }
    .main {
        background-color: #ffffff;
    }
    .stButton>button {
        background: linear-gradient(to right, #43cea2, #185a9d);
        color: white;
        border-radius: 8px;
        font-weight: bold;
        padding: 0.5em 1em;
    }
    .stSelectbox, .stDateInput, .stNumberInput {
        border-radius: 8px;
        font-size: 16px !important;
    }
    .stSidebar {
        background-color: #edf7f6;
    }
    .stTabs [data-baseweb="tab"] {
        background: #e1f5fe;
        border-radius: 5px 5px 0 0;
        margin-right: 2px;
        padding: 0.5em 1em;
        font-weight: 500;
    }
    h1, h2, h3 {
        color: #2e7d32;
    }
    .stMetric {
        background-color: #f1f8e9;
        border-radius: 10px;
        padding: 1em;
    }
    .st-expander {
        background-color: #e8f5e9;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Load and cache data
@st.cache_data
def load_data():
    df_supply = pd.read_excel("FYP data.xlsx", sheet_name=0)
    df_price = pd.read_excel("FYP data.xlsx", sheet_name="Price")
    return df_supply, df_price

# Preprocess data with combined LabelEncoder for commodities/crops
def preprocess_data(df_supply, df_price):
    df_supply = df_supply.ffill()
    df_price = df_price.ffill()
    df_supply['Date'] = pd.to_datetime(df_supply['Date'])
    df_price['Date'] = pd.to_datetime(df_price['Date'])
    df_price['Avg Price'] = (df_price['Min Price'] + df_price['Max Price']) / 2
    
    all_commodities = pd.concat([df_supply['Commodity'], df_price['Crop']]).unique()
    le_commodity = LabelEncoder()
    le_commodity.fit(all_commodities)
    df_supply['Commodity'] = le_commodity.transform(df_supply['Commodity'])
    df_price['Crop'] = le_commodity.transform(df_price['Crop'])
    
    le_supply_city = LabelEncoder()
    df_supply['Supply City'] = le_supply_city.fit_transform(df_supply['Supply City'])
    le_target_city = LabelEncoder()
    df_supply['Target City'] = le_target_city.fit_transform(df_supply['Target City'])
    le_city_price = LabelEncoder()
    df_price['City'] = le_city_price.fit_transform(df_price['City'])
    
    return (df_supply, df_price, le_commodity, le_supply_city, le_target_city, le_city_price)

# Train models
@st.cache_resource
def train_models(df_supply, df_price, _le_commodity, _le_supply_city, _le_city_price):
    X_class = df_supply[['Date', 'Commodity', 'Supply City', 'Quantity']].copy()
    X_class['Date'] = X_class['Date'].map(pd.Timestamp.toordinal).astype(np.int64)
    y_class = df_supply['Target City']
    
    df_merged = pd.merge(df_price, df_supply[['Date', 'Commodity', 'Supply City', 'Quantity']], 
                         left_on=['Date', 'Crop', 'City'], 
                         right_on=['Date', 'Commodity', 'Supply City'], how='left')
    df_merged['Quantity'] = df_merged['Quantity'].fillna(df_merged['Quantity'].mean())
    X_reg = df_merged[['Date', 'Crop', 'City', 'Quantity']].copy()
    X_reg['Date'] = X_reg['Date'].map(pd.Timestamp.toordinal).astype(np.int64)
    y_reg = df_merged['Avg Price']
    
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_class, y_class)
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(X_reg, y_reg)
    
    return rf_classifier, rf_regressor

# Prediction functions
def predict_best_city(commodity_name, supply_city_name, future_date, quantity, 
                      le_commodity, le_supply_city, le_target_city, rf_classifier):
    try:
        commodity_encoded = le_commodity.transform([commodity_name])[0]
        supply_city_encoded = le_supply_city.transform([supply_city_name])[0]
        future_date_ordinal = pd.to_datetime(future_date).toordinal()
        input_data = np.array([[future_date_ordinal, commodity_encoded, supply_city_encoded, quantity]])
        predicted_city_encoded = rf_classifier.predict(input_data)[0]
        best_city = le_target_city.inverse_transform([predicted_city_encoded])[0]
        return best_city
    except Exception as e:
        return f"Error predicting city: {str(e)}"

def predict_price(commodity_name, supply_city_name, future_date, quantity, 
                  le_commodity, le_city_price, rf_regressor):
    try:
        commodity_encoded = le_commodity.transform([commodity_name])[0]
        city_encoded = le_city_price.transform([supply_city_name])[0]
        future_date_ordinal = pd.to_datetime(future_date).toordinal()
        input_data = np.array([[future_date_ordinal, commodity_encoded, city_encoded, quantity]])
        predicted_price = rf_regressor.predict(input_data)[0]
        return predicted_price
    except Exception as e:
        return f"Error predicting price: {str(e)}"

# Safe inverse transform function
def safe_inverse_transform(encoder, label):
    try:
        return encoder.inverse_transform([label])[0]
    except ValueError:
        return 'Unknown'

# Main dashboard function
def main():
    st.title("🌾 Agricultural Data Analysis Dashboard")
    st.markdown("Analyze crop prices, quantities, market trends, and predict future outcomes with ease.")

    # Load and preprocess data
    with st.spinner("Loading data..."):
        df_supply, df_price = load_data()
        (df_supply, df_price, le_commodity, le_supply_city, le_target_city, 
         le_city_price) = preprocess_data(df_supply, df_price)
    
    # Train models
    with st.spinner("Training models..."):
        rf_classifier, rf_regressor = train_models(df_supply, df_price, le_commodity, le_supply_city, le_city_price)
    
    common_crops = sorted(le_commodity.classes_)

    # Sidebar for user inputs
    st.sidebar.header("Filters")
    date_range = st.sidebar.date_input("Select Date Range", 
                                       value=[df_supply['Date'].min(), df_supply['Date'].max()],
                                       min_value=df_supply['Date'].min(),
                                       max_value=df_supply['Date'].max())
    selected_crops = st.sidebar.multiselect("Select Crops", common_crops, default=common_crops[:2])
    
    # Handle single date selection
    if len(date_range) == 1:
        start_date = pd.Timestamp(date_range[0])
        end_date = start_date
    elif len(date_range) == 2:
        start_date = pd.Timestamp(date_range[0])
        end_date = pd.Timestamp(date_range[1])
    else:
        start_date = df_supply['Date'].min()
        end_date = df_supply['Date'].max()
        st.warning("Invalid date range selected. Using full dataset range.")

    # Filter data
    df_supply_filtered = df_supply[(df_supply['Date'] >= start_date) & 
                                   (df_supply['Date'] <= end_date) & 
                                   (df_supply['Commodity'].isin([le_commodity.transform([crop])[0] 
                                                                 for crop in selected_crops]))]
    
    df_price_filtered = df_price[(df_price['Date'] >= start_date) & 
                                 (df_price['Date'] <= end_date) & 
                                 (df_price['Crop'].isin([le_commodity.transform([crop])[0] 
                                                         for crop in selected_crops]))]

    # Tabs with new Home tab
    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏠 Home", "📈 Price Analysis", "📊 Quantity Analysis", 
                                                   "🌍 Market Overview", "🔮 Predictions", "ℹ️ Data View"])

    # Home Tab with Donut Charts
    with tab0:
        st.header("Dashboard Overview")
        st.markdown("A quick snapshot of key metrics across all data.")

        col1, col2 = st.columns(2)
        with col1:
            # Donut chart: Total Price Distribution by Crop
            df_price_dist = df_price.groupby('Crop')['Avg Price'].sum().reset_index()
            df_price_dist['Crop'] = df_price_dist['Crop'].apply(lambda x: safe_inverse_transform(le_commodity, x))
            fig_donut_price = px.pie(df_price_dist, values='Avg Price', names='Crop', 
                                     title="Total Price Distribution by Crop", hole=0.4, template="plotly_white")
            st.plotly_chart(fig_donut_price, use_container_width=True)
        
        with col2:
            # Donut chart: Total Quantity Distribution by Commodity
            df_quantity_dist = df_supply.groupby('Commodity')['Quantity'].sum().reset_index()
            df_quantity_dist['Commodity'] = df_quantity_dist['Commodity'].apply(lambda x: safe_inverse_transform(le_commodity, x))
            fig_donut_quantity = px.pie(df_quantity_dist, values='Quantity', names='Commodity', 
                                        title="Total Quantity Distribution by Commodity", hole=0.4, template="plotly_white")
            st.plotly_chart(fig_donut_quantity, use_container_width=True)
        
        # Help Section with green styling
        with st.expander("How to Use This Dashboard"):
            st.markdown("""
            - **Filters**: Use the sidebar to select a date range and crops.
            - **Home**: Get an overview with donut charts showing total price and quantity distributions.
            - **Price Analysis**: View price trends and distributions for selected crops.
            - **Quantity Analysis**: Explore supply quantities over time and by city.
            - **Market Overview**: Compare prices and quantities across all crops.
            - **Predictions**: Input parameters to predict target cities and prices.
            - **Data View**: Inspect the raw filtered data.
            """)

    # Price Analysis Tab
    with tab1:
        st.header("Price Analysis")
        if not df_price_filtered.empty:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Average Price", f"{df_price_filtered['Avg Price'].mean():.2f} PKR")
            with col2:
                st.metric("Price Range", f"{df_price_filtered['Avg Price'].min():.2f} - {df_price_filtered['Avg Price'].max():.2f} PKR")
            
            df_price_grouped = df_price_filtered.groupby(['Date', 'Crop'])['Avg Price'].mean().reset_index()
            df_price_grouped['Crop'] = df_price_grouped['Crop'].apply(lambda x: safe_inverse_transform(le_commodity, x))
            fig_price = px.line(df_price_grouped, x='Date', y='Avg Price', color='Crop', 
                                title="Average Price Over Time", template="plotly_white")
            fig_price.update_layout(hovermode="x unified")
            st.plotly_chart(fig_price, use_container_width=True)
            
            df_price_filtered_copy = df_price_filtered.copy()
            df_price_filtered_copy['Crop'] = df_price_filtered_copy['Crop'].apply(lambda x: safe_inverse_transform(le_commodity, x))
            fig_box = px.box(df_price_filtered_copy, x='Crop', y='Avg Price', title="Price Distribution by Crop",
                             template="plotly_white")
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.warning("No price data available for the selected filters.")

    # Quantity Analysis Tab
    with tab2:
        st.header("Quantity Analysis")
        if not df_supply_filtered.empty:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Quantity", f"{int(df_supply_filtered['Quantity'].sum())} units")
            with col2:
                st.metric("Unique Supply Cities", len(df_supply_filtered['Supply City'].unique()))
            
            df_supply_grouped = df_supply_filtered.groupby(['Date', 'Commodity'])['Quantity'].sum().reset_index()
            df_supply_grouped['Commodity'] = df_supply_grouped['Commodity'].apply(lambda x: safe_inverse_transform(le_commodity, x))
            fig_quantity = px.line(df_supply_grouped, x='Date', y='Quantity', color='Commodity',
                                   title="Total Quantity Over Time", template="plotly_white")
            fig_quantity.update_layout(hovermode="x unified")
            st.plotly_chart(fig_quantity, use_container_width=True)
            
            df_city_quantity = df_supply_filtered.groupby(['Supply City', 'Commodity'])['Quantity'].sum().reset_index()
            df_city_quantity['Commodity'] = df_city_quantity['Commodity'].apply(lambda x: safe_inverse_transform(le_commodity, x))
            df_city_quantity['Supply City'] = df_city_quantity['Supply City'].apply(lambda x: safe_inverse_transform(le_supply_city, x))
            fig_bar = px.bar(df_city_quantity, x='Supply City', y='Quantity', color='Commodity',
                             title="Quantity by Supply City", template="plotly_white")
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("No supply data available for the selected filters.")

    # Market Overview Tab
    with tab3:
        st.header("Market Overview")
        df_avg_price = df_price.groupby('Crop')['Avg Price'].mean().reset_index()
        df_avg_price['Crop'] = df_avg_price['Crop'].apply(lambda x: safe_inverse_transform(le_commodity, x))
        fig_avg_price = px.bar(df_avg_price, x='Crop', y='Avg Price', title="Average Price Across All Crops",
                               template="plotly_white", color='Crop')
        st.plotly_chart(fig_avg_price, use_container_width=True)
        
        if not df_price_filtered.empty and not df_supply_filtered.empty:
            df_price_avg = df_price_filtered.groupby('Crop')['Avg Price'].mean().reset_index()
            df_quantity_sum = df_supply_filtered.groupby('Commodity')['Quantity'].sum().reset_index()
            df_price_avg['Crop'] = df_price_avg['Crop'].apply(lambda x: safe_inverse_transform(le_commodity, x))
            df_quantity_sum['Commodity'] = df_quantity_sum['Commodity'].apply(lambda x: safe_inverse_transform(le_commodity, x))
            df_scatter = pd.merge(df_price_avg, df_quantity_sum, left_on='Crop', right_on='Commodity')
            fig_scatter = px.scatter(df_scatter, x='Quantity', y='Avg Price', color='Crop',
                                     title="Price vs. Quantity for Selected Crops", template="plotly_white",
                                     hover_data=['Crop'])
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Select crops and date range to view Price vs. Quantity analysis.")

    # Predictions Tab
    with tab4:
        st.header("Predictions")
        st.markdown("Predict the best target city and average price for a future transaction.")
        
        col1, col2 = st.columns(2)
        with col1:
            commodity_pred = st.selectbox("Commodity", common_crops)
            supply_city_pred = st.selectbox("Supply City", le_supply_city.classes_)
        with col2:
            future_date = st.date_input("Future Date", min_value=datetime.today())
            quantity_pred = st.number_input("Quantity", min_value=0, value=100)
        
        if st.button("Get Predictions"):
            with st.spinner("Generating predictions..."):
                best_city = predict_best_city(commodity_pred, supply_city_pred, future_date, quantity_pred,
                                              le_commodity, le_supply_city, le_target_city, rf_classifier)
                predicted_price = predict_price(commodity_pred, supply_city_pred, future_date, quantity_pred,
                                                le_commodity, le_city_price, rf_regressor)
            
            st.subheader("Prediction Results")
            st.write(f"**Best Target City:** {best_city}")
            if isinstance(predicted_price, str):
                st.error(predicted_price)
            else:
                st.write(f"**Predicted Average Price:** {predicted_price:.2f} PKR")

    # Data View Tab
    with tab5:
        st.header("Raw Data View")
        st.markdown("Inspect the filtered data used in the analysis.")
        if not df_supply_filtered.empty or not df_price_filtered.empty:
            data_option = st.radio("Select Data to View", ("Supply Data", "Price Data"))
            if data_option == "Supply Data" and not df_supply_filtered.empty:
                df_display = df_supply_filtered.copy()
                df_display['Commodity'] = df_display['Commodity'].apply(lambda x: safe_inverse_transform(le_commodity, x))
                df_display['Supply City'] = df_display['Supply City'].apply(lambda x: safe_inverse_transform(le_supply_city, x))
                df_display['Target City'] = df_display['Target City'].apply(lambda x: safe_inverse_transform(le_target_city, x))
                st.dataframe(df_display, use_container_width=True)
            elif data_option == "Price Data" and not df_price_filtered.empty:
                df_display = df_price_filtered.copy()
                df_display['Crop'] = df_display['Crop'].apply(lambda x: safe_inverse_transform(le_commodity, x))
                df_display['City'] = df_display['City'].apply(lambda x: safe_inverse_transform(le_city_price, x))
                st.dataframe(df_display, use_container_width=True)
            else:
                st.warning("No data available for the selected filters.")
        else:
            st.warning("No data available for the selected filters.")

if __name__ == "__main__":
    main()