import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from nsepy import get_history
import yfinance as yf
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

plt.style.use("fivethirtyeight")
pd.set_option('display.max_columns', 500)
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]

st.set_page_config(layout="wide")  # Set the layout to wide
st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("""
    <style>
    .metric-box {
        border: 1px solid #e6e6e6;
        padding: 5px;  /* Reduced padding */
        border-radius: 5px;
        margin: 5px;  /* Reduced margin */
        text-align: center;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
    }
    .metric-box h3 {
        margin: 0;
        font-size: 16px;  /* Reduced font size */
    }
    .metric-box .metric-value {
        font-size: 18px;  /* Reduced font size */
        font-weight: bold;
        color: #4CAF50;
    }
    .metric-box .metric-date {
        font-size: 12px;  /* Reduced font size */
        color: #757575;
    }
    </style>
    """, unsafe_allow_html=True)




# Title and description
st.title('Stock Analysis App')
st.write('Welcome to the Stock Analysis and Visualization App! This interactive web application enables you to analyze various financial key performance indicators (KPIs) of stocks listed on the Indian stock markets. With advanced visualizations and machine learning models, this app provides deep insights and predictions to enhance your stock analysis experience.')

# Using st.columns to create two columns
col1, col2 = st.columns(2)

# Overview
with col1:
    st.subheader('Overview')
    st.write('The Stock Analysis and Visualization App offers a range of features to analyze and visualize stock data effectively. From volatility analysis to Bollinger Bands, RSI, and MACD analysis, this app provides a comprehensive toolkit for both individual investors and financial analysts.')

# Key Features
with col2:
    st.subheader('Key Features')
    st.write('Volatility Analysis: Visualize stock volatility with adjustable timeframes.\n\nBollinger Bands Analysis: Customize parameters and visualize Bollinger Bands.\n\nRSI Analysis: Identify overbought and oversold conditions with customizable parameters.\n\nMACD Analysis: Visualize MACD for trend changes and momentum analysis.')




# List of stock symbols
stock_symbols = [

    "ADANIPORTS", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO", "BAJFINANCE",
    "BAJAJFINSV", "BHARTIARTL", "BPCL", "BRITANNIA", "CIPLA",
    "COALINDIA", "DIVISLAB", "DRREDDY", "EICHERMOT", "GRASIM",
    "HCLTECH", "HDFC", "HDFCBANK", "HEROMOTOCO", "HINDALCO",
    "HINDUNILVR", "ICICIBANK", "INDUSINDBK", "INFY", "ITC",
    "JSWSTEEL", "KOTAKBANK", "LT", "M&M", "MARUTI",
    "NESTLEIND", "NTPC", "ONGC", "POWERGRID", "RELIANCE",
    "SBIN", "SHREECEM", "SUNPHARMA", "TCS", "TATACONSUM",
    "TATAMOTORS", "TATASTEEL", "TECHM", "TITAN", "ULTRACEMCO",
    "UPL", "WIPRO"
]


# Sidebar - Stock selection
selected_stock = st.selectbox("Select a stock to analyze", stock_symbols)

# Sidebar - Period selection
months = st.slider("Select the number of months for analysis", min_value=1, max_value=60, value=12)

if selected_stock and months:
    # Calculate start date
    end_date = datetime.today() - timedelta(days=1)
    start_date = end_date - relativedelta(months=months)

    # Fetch data from yfinance
    stock_data = yf.download(selected_stock + ".NS", start=start_date, end=end_date)

    stock_data['Date'] = stock_data.index  # Make 'Date' a column while keeping the index which is also in date format

    # Main content layout using containers
    with st.container():
        col1, col2 = st.columns([1, 6])

        # Display raw data
        with col1:
            
            st.markdown("<br>", unsafe_allow_html=True)  # Add empty space
            st.markdown("<br>", unsafe_allow_html=True)  # Add empty space
            st.markdown("<br>", unsafe_allow_html=True)  # Add empty space
            st.markdown("<br>", unsafe_allow_html=True)  # Add empty space
            st.markdown("<br>", unsafe_allow_html=True)  # Add empty space

            # Aggregation level selection
            st.write("Select Timeframe:")
            aggregation = st.radio("", ['Daily', 'Weekly', 'Monthly'])
            
            st.markdown("<br>", unsafe_allow_html=True)  # Add empty space

            #Chart type selection
            st.write("Select Chart Type:")
            chart_type = st.radio("", ['Line Chart', 'Candlestick Chart'])

        # Visualization
        with col2:
            st.subheader(f"{selected_stock}")

        
            if aggregation == 'Weekly':
                agg_stock_data = stock_data.resample('W').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                })
                agg_stock_data.reset_index(inplace=True)
            elif aggregation == 'Monthly':
                agg_stock_data = stock_data.resample('M').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                })
                agg_stock_data.reset_index(inplace=True)
            else:
                agg_stock_data = stock_data.copy()

            if chart_type == 'Line Chart':
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=agg_stock_data['Date'], y=agg_stock_data['Close'], mode='lines', name='Close'))
                fig.add_trace(go.Bar(x=agg_stock_data['Date'], y=agg_stock_data['Volume'], name='Volume', yaxis='y2', opacity=0.4))
                fig.update_layout(
                    title=f'{selected_stock} Closing Prices Over Time',
                    xaxis_title='Date',
                    yaxis_title='Price',
                    yaxis2=dict(
                        title='Volume',
                        overlaying='y',
                        side='right'
                    ),
                    legend=dict(x=0, y=1.2, orientation='h')
                    
                )
            elif chart_type == 'Candlestick Chart':
                fig = go.Figure(data=[go.Candlestick(
                    x=agg_stock_data['Date'],
                    open=agg_stock_data['Open'],
                    high=agg_stock_data['High'],
                    low=agg_stock_data['Low'],
                    close=agg_stock_data['Close']
                )])
                fig.add_trace(go.Bar(x=agg_stock_data['Date'], y=agg_stock_data['Volume'], name='Volume', yaxis='y2', opacity=0.4))
                fig.update_layout(
                    title=f'{selected_stock} Candlestick Chart Over Time',
                    xaxis_title='Date',
                    yaxis_title='Price',
                    yaxis2=dict(
                        title='Volume',
                        overlaying='y',
                        side='right'
                    ),
                    legend=dict(x=0, y=1.2, orientation='h')
                )

            st.plotly_chart(fig, use_container_width=True)










# Function  (to create features from data)
def new_columns(data):
    
    l1 = []
    l2 = []
    for i in range(len(data)):
        if i == 0:
            var1 = 0
            var2 = 0
        else:
            var1 = data.iloc[i,3] - data.iloc[i-1,3]
            var2 = ((data.iloc[i,3] - data.iloc[i-1,3])/data.iloc[i-1,3])*100
        l1.append(var1)
        l2.append(var2)
    data["amount_change"] = l1
    data["percent_change"] = l2

    data["year"] = data["Date"].dt.year
    data["month"] = data["Date"].dt.month_name()
    data["week"] = data["Date"].dt.isocalendar().week
    data["day"] = data["Date"].dt.day
    data["dayname"] = data["Date"].dt.day_name()
    data["Target"] = np.where(data["percent_change"] > 0, 1,0)
    
    return data

# adding new features
stock_data = new_columns(stock_data)


def plot2_1():
    temp_df1 = stock_data.copy()
    temp_df1['color'] = np.where(temp_df1['percent_change'] >= 0, 'skyblue', 'orange')
    colors =  np.where(temp_df1['percent_change'] >= 0, 'skyblue', 'orange')

    fig = px.bar(temp_df1, x= temp_df1.index, y='percent_change', title=f'{selected_stock} share daily percent changes')

    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside',marker_color= colors)
    fig.update_layout(
        xaxis_title="Day",
        yaxis_title="Daily Percent Changes in stock",
        width=1000, 
        height=500 
        
    )
    return fig

def plot2_2():

    temp_df2 = stock_data.groupby(["year","week"])["percent_change"].sum()
    temp_df2 = temp_df2.reset_index()
    temp_df2['year_week'] = temp_df2['year'].astype(str) + '-W' + temp_df2['week'].astype(str)
    colors =  np.where(temp_df2['percent_change'] >= 0, 'skyblue', 'orange')

    fig = px.bar(temp_df2 , x = "year_week",y = "percent_change",text='percent_change', title= f'{selected_stock} share weekly percent changes')
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside',marker_color= colors)

    fig.update_layout(
        xaxis_title="weeks",
        yaxis_title="wekly Percent Changes in stock",
        uniformtext_minsize=8, 
        uniformtext_mode='hide',
        width=1000, 
        height=500 
    )
    
    return fig

def plot2_3():
    temp_df3 = stock_data.groupby(["year",stock_data["Date"].dt.month])["percent_change"].sum()
    temp_df3 = temp_df3.reset_index()
    temp_df3.rename(columns ={"Date" : "Month_No"},inplace=True)
    temp_df3['year_month'] = temp_df3['year'].astype(str) + '-M' + temp_df3['Month_No'].astype(str)
    colors =  np.where(temp_df3['percent_change'] >= 0, 'skyblue', 'orange')

    fig = px.bar(temp_df3 , x = "year_month",y = "percent_change",text='percent_change', title=f'{selected_stock} share Monthly percent changes')
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside',marker_color= colors)

    fig.update_layout(
        xaxis_title="Months",
        yaxis_title="Monthly Percent Changes in stock",
        uniformtext_minsize=8, 
        uniformtext_mode='hide',
        width=1000, 
        height=500 
    )
    
    return fig

def plot2_4():
    temp_df3 = stock_data.groupby("year")["percent_change"].sum()
    temp_df3 = temp_df3.reset_index()
    colors = np.where(temp_df3['percent_change'] >= 0, 'skyblue', 'orange')

    fig = px.bar(temp_df3, x="year", y="percent_change", text='percent_change',
                 title=f'{selected_stock} share Yearly percent changes')
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside', marker_color=colors)

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Yearly Percent Changes in stock",
        uniformtext_minsize=8,
        uniformtext_mode='hide',
        width=1000,
        height=500
    )

    return fig


with st.container():
    
    selected_plot = "Daily"

    col1, col2 = st.columns([1, 6])
    
    with col1:
        
        st.markdown("<br>", unsafe_allow_html=True)  # Add empty space
        st.markdown("<br>", unsafe_allow_html=True)  # Add empty space
        st.markdown("<br>", unsafe_allow_html=True)  # Add empty space
        st.markdown("<br>", unsafe_allow_html=True)  # Add empty space

        # Aggregation level selection
        
        st.markdown("Select Timeframe:")
        st.markdown("<br>", unsafe_allow_html=True)  # Add empty space

        if st.button("Daily Percent Changes"):
            selected_plot = "Daily"
        if st.button("Weekly Percent Changes"):
            selected_plot = "Weekly"
        if st.button("Monthly Percent Changes"):
            selected_plot = "Monthly"
        if st.button("Yearly Percent Changes"):
            selected_plot = "Yearly"
        
        st.markdown("<br>", unsafe_allow_html=True)  # Add empty space
        st.markdown("<br>", unsafe_allow_html=True)  # Add empty space

        st.subheader("Key Metrics")
        def calculate_metrics(data):
            # Maximum single day gain
            Maximun_single_day_gain = data[data["percent_change"] == data["percent_change"].max()][["Date", "percent_change"]]

            # Maximum single day loss
            Maximun_single_day_loss = data[data["percent_change"] == data["percent_change"].min()][["Date", "percent_change"]]

            # Monthly gains
            monthly_gains = data.resample("M").agg({"percent_change": "sum"}).reset_index()

            # Maximum monthly gain
            Maximun_Monthly_Gain = monthly_gains[monthly_gains["percent_change"] == monthly_gains["percent_change"].max()][["Date", "percent_change"]]

            # Maximum monthly loss
            Maximum_Monthly_Loss = monthly_gains[monthly_gains["percent_change"] == monthly_gains["percent_change"].min()][["Date", "percent_change"]]

            # Average percent change
            Average_Percent_change = abs(data["percent_change"]).mean()

            return Maximun_single_day_gain, Maximun_single_day_loss, Maximun_Monthly_Gain, Maximum_Monthly_Loss, Average_Percent_change
        
        # Select year
        years = stock_data.index.year.unique()
        selected_year = st.selectbox("Select Year", years)

        # Filter data based on selected year
        filtered_data = stock_data[stock_data.index.year == selected_year]
        
        # Calculate metrics
        Maximun_single_day_gain, Maximun_single_day_loss, Maximun_Monthly_Gain, Maximum_Monthly_Loss, Average_Percent_change = calculate_metrics(filtered_data)

        
            
        #Display each metric in a box
        st.markdown(f"""
            <div class="metric-box">
                <h3>Max Single Day Gain</h3>
                <div class="metric-value">{Maximun_single_day_gain['percent_change'].values[0]:.2f}%</div>
                <div class="metric-date">Date: {Maximun_single_day_gain.index.strftime('%Y-%m-%d')[0]}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="metric-box">
                <h3>Max Single Day Loss</h3>
                <div class="metric-value">{Maximun_single_day_loss['percent_change'].values[0]:.2f}%</div>
                <div class="metric-date">Date: {Maximun_single_day_loss.index.strftime('%Y-%m-%d')[0]}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="metric-box">
                <h3>Max Monthly Gain</h3>
                <div class="metric-value">{Maximun_Monthly_Gain['percent_change'].values[0]:.2f}%</div>
                <div class="metric-date">Date: {Maximun_Monthly_Gain['Date'].dt.strftime('%Y-%m').values[0]}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="metric-box">
                <h3>Max Monthly Loss</h3>
                <div class="metric-value">{Maximum_Monthly_Loss['percent_change'].values[0]:.2f}%</div>
                <div class="metric-date">Date: {Maximum_Monthly_Loss['Date'].dt.strftime('%Y-%m').values[0]}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="metric-box">
                <h3>Average Percent Change</h3>
                <div class="metric-value">{Average_Percent_change:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
                
        
        
        
    with col2:
        
        st.subheader("Increase and decrease Percent Changes for different Timeframe:")
        if selected_plot == "Daily":
            st.plotly_chart(plot2_1(), use_container_width=True)
        elif selected_plot == "Weekly":
            st.plotly_chart(plot2_2(), use_container_width=True)
        elif selected_plot == "Monthly":
            st.plotly_chart(plot2_3(), use_container_width=True)
        elif selected_plot == "Yearly":
            st.plotly_chart(plot2_4(), use_container_width=True)
            
            
        with st.container():
            
            col1, col2,= st.columns([1,1])
            # Function to calculate and plot positive and negative days
            
            with col1:
                def plot_positive_negative_days(stock_data, selected_year):
                    # Filter data for the selected year
                    stock_data_year = stock_data[stock_data['Date'].dt.year == selected_year]

                    # Calculate positive and negative days
                    positive_days = stock_data_year[stock_data_year['percent_change'] > 0]
                    negative_days = stock_data_year[stock_data_year['percent_change'] < 0]

                    # Create a DataFrame with the results
                    temp = {
                        'Category': ['Positive Days', 'Negative Days'],
                        'Days': [positive_days['percent_change'].count(), negative_days['percent_change'].count()]
                    }
                    df = pd.DataFrame(temp)

                    # Calculate percentage
                    total_days = df['Days'].sum()
                    df['Percentage'] = round((df['Days'] / total_days) * 100, 2)

                    

                    
                    # Create a donut chart
                    fig = go.Figure(data=[go.Pie(
                        labels=df['Category'],
                        values=df['Days'],
                        hole=.3,
                        marker=dict(colors=['#66c2a5', '#fc8d62'], line=dict(color='#000000', width=2)),
                        textinfo='percent+label',
                        insidetextorientation='radial'
                    )])

                    # Update layout for better aesthetics
                    fig.update_layout(
                        title_text='Distribution of Positive and Negative Days',
                        showlegend=True,
                        legend=dict(x=1, y=1, traceorder='normal', font=dict(size=12)),
                        font=dict(size=16),
                        margin=dict(l=20, r=20, t=50, b=20),
                    )

                    # Highlight significant points with annotations
                    fig.add_annotation(
                        x=0.5, y=-0.1,
                        text=f"Total Days: {total_days}<br>Positive Days: {df.loc[df['Category'] == 'Positive Days', 'Days'].values[0]}<br>Negative Days: {df.loc[df['Category'] == 'Negative Days', 'Days'].values[0]}",
                        showarrow=False,
                        font=dict(size=14, color="white"),  # Change color to white
                        align="center",
                        xref="paper",
                        yref="paper"
                    )
                    
                    return fig

            

                # Add a description of the plot
                st.write(f"""
                ### Distribution of Positive and Negative Days for {selected_year}
                This plot shows the distribution of days with positive and negative changes in the stock's percent change for the selected year.
                """)


                # Display the donut chart
                st.plotly_chart(plot_positive_negative_days(stock_data, selected_year), use_container_width=True)
                
            with col2:
            
                
                                
                def streak_calculator(data, selected_year):
                    positive_streak_record = []
                    negative_streak_record = []

                    positive_streak = 0
                    negative_streak = 0
                    temp_df4 = data[data["Date"].dt.year == selected_year]
                    
                    for i in temp_df4.Target:
                        
                        if i == 1:
                            positive_streak += 1
                            if negative_streak > 2:
                                negative_streak_record.append(negative_streak)
                            negative_streak = 0
                        else:
                            negative_streak += 1
                            if positive_streak > 2:
                                positive_streak_record.append(positive_streak)
                            positive_streak = 0

                    # Ensure to catch the last streaks if they are ongoing
                    if positive_streak > 2:
                        positive_streak_record.append(positive_streak)
                    if negative_streak > 2:
                        negative_streak_record.append(negative_streak)

                    # Create a DataFrame with streak counts
                    positive_streak_df = pd.DataFrame({
                        'Streak Length': positive_streak_record
                    })

                    negative_streak_df = pd.DataFrame({
                        'Streak Length': negative_streak_record
                    })
                    
                    return positive_streak_df, negative_streak_df

               
                st.subheader("Stock Streaks Visualization")
                st.markdown("""
                    *These plots show the distribution of positive and negative streaks in stock movements for the selected year.
                    A positive streak is a series of consecutive days where the stock price increased, while a negative streak is
                    a series of consecutive days where the stock price decreased. The x-axis represents the length of the streak,
                    and the y-axis represents the frequency of each streak length.*
                """)
                
                streak_type = st.selectbox('Select Streak Type', ['Positive', 'Negative'])

                # Calculate streaks
                positive_streak_df, negative_streak_df = streak_calculator(stock_data, selected_year)

                # Count streak lengths
                positive_streak_counts = positive_streak_df['Streak Length'].value_counts().reset_index()
                positive_streak_counts.columns = ['Streak Length', 'Count']

                negative_streak_counts = negative_streak_df['Streak Length'].value_counts().reset_index()
                negative_streak_counts.columns = ['Streak Length', 'Count']

                if streak_type == "Positive":
                    fig_positive = px.histogram(positive_streak_df, x='Streak Length', 
                                labels={'Streak Length': 'Positive Streak Length', 'count': 'Frequency'},
                                title=f'Frequency of Positive Streaks in Stock Movements for {selected_year}',
                                color_discrete_sequence=['#87CEEB'],
                                nbins=10)

                    # Update layout for better aesthetics
                    fig_positive.update_layout(
                        xaxis_title="Streak Length",
                        yaxis_title="Frequency",
                        showlegend=False,
                        font=dict(size=12),
                        margin=dict(l=20, r=20, t=50, b=20),
                        width=600,  # Reduce the size of the plot
                        height=400,  # Reduce the size of the plot
                        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
                        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
                        title={
                            'y':0.9,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top',
                            'font_size': 16
                        }
                    )
                    st.plotly_chart(fig_positive)

                    
                if streak_type == "Negative":
                    # Plot negative streak counts
                    fig_negative = px.histogram(negative_streak_df, x='Streak Length', 
                                                labels={'Streak Length': 'Negative Streak Length', 'count': 'Frequency'},
                                                title=f'Frequency of Negative Streaks in Stock Movements for {selected_year}',
                                                color_discrete_sequence=['#FFA07A'],
                                                nbins=10)

                    # Update layout for better aesthetics
                    fig_negative.update_layout(
                        xaxis_title="Streak Length",
                        yaxis_title="Frequency",
                        showlegend=False,
                        font=dict(size=12),
                        margin=dict(l=20, r=20, t=50, b=20),
                        width=600,  # Reduce the size of the plot
                        height=400,  # Reduce the size of the plot
                        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
                        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
                        title={
                            'y':0.9,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top',
                            'font_size': 16
                        }
                    )
                    st.plotly_chart(fig_negative)
                    
                    
                    
                    
                    
                    
                    
                    
                    
# Function to set default layout for plotly figures
def set_default_layout(fig):
    fig.update_layout(
        template='plotly_white',
        font=dict(color='white'),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Date",
        yaxis_title="Value"
    )

def volatility(df, window=14):
    df['Standard_Deviation'] = df['Close'].rolling(window).std()
    df['High-Low'] = df['High'] - df['Low']
    df['High-PrevClose'] = np.abs(df['High'] - df['Close'].shift(1))
    df['Low-PrevClose'] = np.abs(df['Low'] - df['Close'].shift(1))
    df['True_Range'] = df[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
    df['ATR'] = df['True_Range'].rolling(window).mean()

    # Determine volatile periods
    volatility_threshold = df['ATR'].quantile(0.75)
    df['Volatile'] = df['ATR'] > volatility_threshold

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.02, 
                        subplot_titles=('Close Price with Volatility Highlighted', 'Volatility (Standard Deviation)'),
                        row_width=[0.2, 0.8])

    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close Price', line=dict(color='white')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Standard_Deviation'], mode='lines', name='Volatility', line=dict(color='yellow')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df[df['Volatile']]['Date'], y=df[df['Volatile']]['Close'], mode='lines', name='Volatile Period', line=dict(color='red', width=1)), row=1, col=1)

    # Calculate average volatility
    avg_volatility = df['ATR'].mean()
    
    # Determine current status of volatility
    current_volatility_status = "Highly Volatile" if df['ATR'].iloc[-1] > volatility_threshold else "Stable"

    fig.update_layout(title='Volatility Analysis', height=500)
    set_default_layout(fig)

    return fig, avg_volatility, current_volatility_status

def bollinger_bands_analysis(df, window, num_std):
    df['Rolling_Mean'] = df['Close'].rolling(window).mean()
    df['Bollinger_High'] = df['Rolling_Mean'] + (df['Close'].rolling(window).std() * num_std)
    df['Bollinger_Low'] = df['Rolling_Mean'] - (df['Close'].rolling(window).std() * num_std)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close Price', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Bollinger_High'], mode='lines', name='Bollinger High', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Bollinger_Low'], mode='lines', name='Bollinger Low', line=dict(color='green', dash='dash')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Bollinger_High'], fill=None, mode='lines', line=dict(color='rgba(255, 0, 0, 0)'), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Bollinger_Low'], fill='tonexty', mode='lines', line=dict(color='rgba(0, 255, 0, 0)'), fillcolor='rgba(0, 100, 80, 0.2)', showlegend=False, hoverinfo='skip'))

    percentage_above_band = (df['Close'] > df['Bollinger_High']).mean() * 100
    percentage_below_band = (df['Close'] < df['Bollinger_Low']).mean() * 100
    current_band_status = "Above Band" if df['Close'].iloc[-1] > df['Bollinger_High'].iloc[-1] else "Below Band" if df['Close'].iloc[-1] < df['Bollinger_Low'].iloc[-1] else "Within Band"

    return fig, percentage_above_band, percentage_below_band, current_band_status

def rsi_analysis(df, window, lower_bound, upper_bound):
    delta = df['Close'].diff(1)
    delta = delta.dropna()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # Create figure
    fig = go.Figure()

    # Add RSI line chart
    fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI_14'], mode='lines', name='RSI', line=dict(color='blue')))

    # Add markers for overbought and oversold conditions
    fig.add_trace(go.Scatter(x=df[df['RSI_14'] > upper_bound]['Date'], y=df[df['RSI_14'] > upper_bound]['RSI_14'],
                             mode='markers', marker=dict(color='red', size=7), name='Overbought'))
    fig.add_trace(go.Scatter(x=df[df['RSI_14'] < lower_bound]['Date'], y=df[df['RSI_14'] < lower_bound]['RSI_14'],
                             mode='markers', marker=dict(color='green', size=7), name='Oversold'))

    # Update layout
    fig.update_layout(title='RSI Analysis', xaxis_title='Date', yaxis_title='RSI', height=500)
    set_default_layout(fig)
    return fig




def macd_analysis(df, fastperiod=12, slowperiod=26, signalperiod=9):
    df['EMA_Fast'] = df['Close'].ewm(span=fastperiod, adjust=False).mean()
    df['EMA_Slow'] = df['Close'].ewm(span=slowperiod, adjust=False).mean()
    df['MACD'] = df['EMA_Fast'] - df['EMA_Slow']
    df['Signal_Line'] = df['MACD'].ewm(span=signalperiod, adjust=False).mean()

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], mode='lines', name='MACD', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Signal_Line'], mode='lines', name='Signal Line', line=dict(color='red')))

    fig.update_layout(title='MACD Analysis', xaxis_title='Date', yaxis_title='MACD', height=500)
    set_default_layout(fig)
    return fig
                  

with st.container():
    st.header("Volatility Analysis and Bollinger Bands")
    st.markdown("<br>", unsafe_allow_html=True)  # Add empty space

    # Data loading
    data = stock_data.copy()

    # Selection for analysis type
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("ATR & Standard Deviation")

        
        # Timeframe selection
        timeframe_options = ['1 month', '3 months', '6 months', '1 year', 'All time']
        selected_timeframe1 = col1.selectbox('Select Timeframe', timeframe_options, key='timeframe1')
        window_vol = col1.slider('Select Window for Volatility Calculation', 10, 50, 14, key='window_vol')
        
        st.markdown("<br>", unsafe_allow_html=True)  # Add empty space
        st.markdown("<br>", unsafe_allow_html=True)  # Add empty space
        st.markdown("<br>", unsafe_allow_html=True)  # Add empty space
        st.markdown("<br>", unsafe_allow_html=True)  # Add empty space


        if selected_timeframe1 != 'All time':
            end_date = data['Date'].max()
            start_date = end_date - pd.DateOffset(months=int(selected_timeframe1.split()[0]) * {'month': 1, 'months': 1, 'year': 12}[selected_timeframe1.split()[1]])
            data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
            
        analysis_fig, avg_volatility, current_volatility_status = volatility(data, window_vol)
        st.plotly_chart(analysis_fig, use_container_width=True)  

         
        
    with col2:
        st.subheader("Bollinger Bands")

        # Timeframe selection
        selected_timeframe2 = col2.selectbox('Select Timeframe', timeframe_options, key='timeframe2')
        window_bb = col2.slider('Select Window for Bollinger Bands Calculation', 10, 50, 20, key='window_bb')
        num_std = col2.slider('Select Number of Standard Deviations', 1, 3, 2, key='num_std')

        if selected_timeframe2 != 'All time':
            end_date = data['Date'].max()
            start_date = end_date - pd.DateOffset(months=int(selected_timeframe2.split()[0]) * {'month': 1, 'months': 1, 'year': 12}[selected_timeframe2.split()[1]])
            data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
            
        analysis_fig, percentage_above_band, percentage_below_band, current_band_status = bollinger_bands_analysis(data, window_bb, num_std)
        st.plotly_chart(analysis_fig, use_container_width=True)
            
with st.container():
    st.subheader("Key Metrics")
    
    col1,col2,col3,col4,col5 = st.columns(5)
    with col1:
        avg_vol_html = f"""
        <div style='background-color: white; padding: 5px; border-radius: 5px; width: 150px; margin: 5px;'>
            <h4 style='color: black;'>Average Volatility</h4>
            <p style='font-size: 20px; color: blue;'>{avg_volatility:.2f}</p>
        </div>
        """
        st.markdown(avg_vol_html, unsafe_allow_html=True)

    with col2:
        vol_status_html = f"""
        <div style='background-color: white; padding: 5px; border-radius: 5px; width: 150px; margin: 5px;'>
            <h4 style='color: black;'>Current Volatility Status</h4>
            <p style='font-size: 20px; color: green;'>{current_volatility_status}</p>
        </div>
        """
        st.markdown(vol_status_html, unsafe_allow_html=True)

    with col3:
        perc_above_band_html = f"""
        <div style='background-color: white; padding: 5px; border-radius: 5px; width: 150px; margin: 5px;'>
            <h4 style='color: black;'>Percentage Above Band</h4>
            <p style='font-size: 20px; color: blue;'>{percentage_above_band:.2f}%</p>
        </div>
        """
        st.markdown(perc_above_band_html, unsafe_allow_html=True)

    with col4:
        perc_below_band_html = f"""
        <div style='background-color: white; padding: 5px; border-radius: 5px; width: 150px; margin: 5px;'>
            <h4 style='color: black;'>Percentage Below Band</h4>
            <p style='font-size: 20px; color: blue;'>{percentage_below_band:.2f}%</p>
        </div>
        """
        st.markdown(perc_below_band_html, unsafe_allow_html=True)

    with col5:
        current_band_status_html = f"""
        <div style='background-color: white; padding: 5px; border-radius: 5px; width: 150px; margin: 5px;'>
            <h4 style='color: black;'>Current Band Status</h4>
            <p style='font-size: 20px; color: green;'>{current_band_status}</p>
        </div>
        """
        st.markdown(current_band_status_html, unsafe_allow_html=True)
    
        
st.markdown("<br>", unsafe_allow_html=True)  # Add empty space
st.markdown("<br>", unsafe_allow_html=True)  # Add empty space


with st.container():
    st.header("Strength Analysis:")

    col1, col2,col3,col4 = st.columns([0.10, 0.4,0.4,0.1])
    
    with col1:
        st.markdown("<br>", unsafe_allow_html=True)  # Add empty space
        st.markdown("<br>", unsafe_allow_html=True)  # Add empty space
        

        # Timeframe selection for RSI/MACD
        timeframe_options_rsi = ['1 month', '3 months', '6 months', '1 year', 'All time']
        selected_timeframe_rsi = st.selectbox('Select Timeframe for RSI/MACD', timeframe_options_rsi,key='timeframe3')

        if selected_timeframe_rsi != 'All time':
            end_date_rsi_macd = data['Date'].max()
            start_date_rsi_macd = end_date_rsi_macd - pd.DateOffset(months=int(selected_timeframe_rsi.split()[0]) * {'month': 1, 'months': 1, 'year': 12}[selected_timeframe_rsi.split()[1]])
            data_rsi_macd = data[(data['Date'] >= start_date_rsi_macd) & (data['Date'] <= end_date_rsi_macd)]
        else:
            data_rsi_macd = data.copy()
            
    with col2:
    # RSI Plot
        st.subheader("Relative Strength Index")

        rsi_window = col1.slider('Select Window for RSI Calculation', 10, 50, 14)
        lower_bound = col1.slider('Select Lower Bound for RSI', 10, 50, 30)
        upper_bound = col1.slider('Select Upper Bound for RSI', 50, 90, 70)
        rsi_fig = rsi_analysis(data_rsi_macd, rsi_window, lower_bound, upper_bound)
        
        st.plotly_chart(rsi_fig, use_container_width=True)

    with col3:
    # MACD plot  
        st.subheader("MACD")
        
         # Timeframe selection for RSI/MACD
        timeframe_options_macd = ['1 month', '3 months', '6 months', '1 year', 'All time']
        selected_timeframe_macd = col4.selectbox('Select Timeframe for RSI/MACD', timeframe_options_macd,key='timeframe4')

        if selected_timeframe_macd != 'All time':
            end_date_rsi_macd = data['Date'].max()
            start_date_rsi_macd = end_date_rsi_macd - pd.DateOffset(months=int(selected_timeframe_macd.split()[0]) * {'month': 1, 'months': 1, 'year': 12}[selected_timeframe_macd.split()[1]])
            data_macd = data[(data['Date'] >= start_date_rsi_macd) & (data['Date'] <= end_date_rsi_macd)]
        else:
            data_macd = data.copy()

        fastperiod = col4.slider('Select Fast Period for MACD', 5, 20, 12)
        slowperiod = col4.slider('Select Slow Period for MACD', 20, 50, 26)
        signalperiod = col4.slider('Select Signal Period for MACD', 5, 20, 9)
        
        macd_fig = macd_analysis(data_macd, fastperiod, slowperiod, signalperiod)

        st.plotly_chart(macd_fig, use_container_width=True)
        
    
        
        
        
        
