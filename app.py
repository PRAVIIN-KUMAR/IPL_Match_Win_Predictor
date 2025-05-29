import streamlit as st
import pickle
import pandas as pd

# Load the trained model pipeline
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Teams and cities
teams = [
    'Royal Challengers Bangalore', 'Punjab Kings', 'Lucknow Super Giants',
    'Mumbai Indians', 'Rajasthan Royals', 'Delhi Capitals',
    'Sunrisers Hyderabad', 'Gujarat Titans', 'Kolkata Knight Riders',
    'Chennai Super Kings'
]

cities = sorted([
    'Bangalore', 'Delhi', 'Mumbai', 'Kolkata', 'Jaipur', 'Hyderabad',
    'Chennai', 'Cape Town', 'Port Elizabeth', 'Durban', 'Centurion',
    'East London', 'Johannesburg', 'Kimberley', 'Bloemfontein',
    'Ahmedabad', 'Cuttack', 'Nagpur', 'Visakhapatnam', 'Pune',
    'Raipur', 'Ranchi', 'Abu Dhabi', 'Bengaluru', 'Chandigarh',
    'Indore', 'Sharjah', 'Dubai', 'Navi Mumbai', 'Lucknow', 'Guwahati',
    'Dharamsala', 'Mohali'
])

# Mapping team names to logo file paths
team_logos = {
    'Chennai Super Kings': 'logos/csk.png',
    'Rajasthan Royals': 'logos/rr.png',
    'Punjab Kings': 'logos/pk.png',
    'Sunrisers Hyderabad': 'logos/sh.png',
    'Delhi Capitals': 'logos/dc.png',
    'Gujarat Titans': 'logos/gt.png',
    'Kolkata Knight Riders': 'logos/kkr.png',
    'Lucknow Super Giants': 'logos/lsg.png',
    'Mumbai Indians': 'logos/mi.png',
    'Royal Challengers Bangalore': 'logos/rcb.png'
}
ipl_logo = 'logos/ipl.png'

# Sidebar: Project Description
st.sidebar.title("🏏 IPL Win Predictor")
st.sidebar.markdown("""
Predict the winning probability of an IPL team based on the current match scenario.  
Use this tool to get real-time insights into the game using an ML model trained on past IPL data.

---

### Features:
- Batting and bowling teams
- Match city
- Target runs
- Current score
- Overs completed
- Wickets lost

---

### Output:
- Winning probability for both teams in percentage
""")

# Main Title with IPL logo
colA, colB = st.columns([1, 8])
with colA:
    st.image(ipl_logo, width=60)
with colB:
    st.title("🏆 IPL Win Predictor")

# Select teams
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

# Select city and target runs
selected_city = st.selectbox('Select host city', cities)
target = st.number_input('Target Score', min_value=1, step=1)

# Inputs for current match state
col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('Current Score', min_value=0, step=1)
with col4:
    overs = st.number_input('Overs Completed', min_value=0.1, max_value=20.0, step=0.1)
with col5:
    wickets_out = st.number_input('Wickets Out', min_value=0, max_value=10, step=1)

# Predict button and validation
if st.button('Predict Probability'):

    # Validate inputs
    if batting_team == bowling_team:
        st.warning("⚠️ Batting and Bowling team must be different.")
    elif overs == 0:
        st.warning("⚠️ Overs completed must be greater than 0.")
    elif score >= target:
        st.warning("⚠️ Current score must be less than the target.")
    elif wickets_out >= 10:
        st.warning("⚠️ Wickets lost must be less than 10.")
    else:
        # Calculate derived features
        runs_left = target - score
        balls_left = 120 - (overs * 6)
        wickets_left = 10 - wickets_out
        crr = score / overs
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

        # Prepare input dataframe for model
        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets_left],
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        # Get prediction probabilities
        result = pipe.predict_proba(input_df)[0]
        loss_prob = result[0]
        win_prob = result[1]

        # Display results with logos
        st.subheader("📊 Winning Probability")

        col1, col2 = st.columns(2)

        with col1:
            st.image(team_logos[batting_team], width=80)
            st.success(f"{batting_team}: {round(win_prob * 100)}%")

        with col2:
            st.image(team_logos[bowling_team], width=80)
            st.error(f"{bowling_team}: {round(loss_prob * 100)}%")