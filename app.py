import yfinance as yf
from pytrends.request import TrendReq
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import correlation as corr
from PIL import Image

image = Image.open('img.PNG')
st.image(image, use_column_width=True)

st.write("""
# ★ Stock Price - Google Trends correlations 

Shown area the stock *opening price* and ***volume*** of the given ticker

""")


# https://towardsdatascience.com/how-to-get-stock-data-using-python-c0de1df17e75
# define the ticker symbol
tickerSymbol_default = 'AAPL' #'GOOGL'
tickerSymbol = st.text_input("Ticker Symbol", tickerSymbol_default)
#tickerSymbol = tickerSymbol.strip()

start_date = '2021-03-31' #'2010-5-31'
start_date = st.text_input("Starting date  [yyyy-mm-dd]", start_date)

end_date = '2022-04-01' #'2020-5-31'
end_date = st.text_input("Ending date  [yyyy-mm-dd]", end_date)

period = '1d'
period = st.text_input("Period", period)

#get data on this ticker
stock_Data = yf.Ticker(tickerSymbol)
# get the hystorical price for this ticker

stock_df = stock_Data.history(period=period, start=start_date, end=end_date)
# Open  High    Low Close   Volume  Dividends   Stock Splits


see_volume = st.checkbox("See Volume", value=False)
flat_stocks = st.checkbox("Flat Stocks")
normalize = st.checkbox("Normalize all", value=True)
polish_all = st.checkbox("Polish all", value=True)
if polish_all:
    polish_n = int(st.text_input("Polish n:", "10"))
st.write("----------")
st.write("**Correlations settings**")
max_shift = int(st.text_input("Max trends day-shift allowed:", "20"))
rtrends_numb = int(st.text_input("RTrends number:", "10"))


if stock_df.empty:
    st.write("**This ticker do not exists!**")
else:

    ####### STREND
    if see_volume:
        st.header("★ Volume and Stock Opening price")
        st.line_chart(stock_df.Volume)
    else:
        st.header("★ Stock Opening price")
    st.line_chart(stock_df.Open)
    #st.line_chart(stock_df.Open-stock_df.Close)
    

    min_open = np.min(stock_df["Open"])
    max_open = np.max(stock_df["Open"])
    init_stock_value = stock_df["Open"][0]

    st.write(f"""
    **Min open**:    {np.round(min_open,3)}
    
    **Max open**:    {np.round(max_open,3)}
    """)

    dates = corr.compute_dates_vec(stock_df.index)

    strend_pday = corr.fill_trend(dates, stock_df["Open"])
    if flat_stocks:
        strend_pday = corr.flatter(strend_pday)

    tmp = [i for i in range(dates[0], len(strend_pday)+dates[0])]
    dates_fill = []
    for d in tmp: dates_fill.append(corr.to_date(start_date, d))

    strend_pday = pd.DataFrame(
        strend_pday,
        columns=["Open"]
    )

    if polish_all:
        strend_pday["Unpolished ST Data"] = strend_pday.Open
    strend_pday.index = dates_fill
    
    st.subheader("Worked Stock Trend")
    if flat_stocks:
        min_open_flat = min(strend_pday.Open)
        max_open_flat = max(strend_pday.Open)
        if normalize:
            strend_pday = (strend_pday-min_open_flat)/(max_open_flat-min_open_flat)


    else:
        if normalize:
            strend_pday = (strend_pday-min_open)/(max_open-min_open) 


    if polish_all:
        strend_pday.Open = corr.polish(strend_pday.Open, polish_n)
    
    st.line_chart(strend_pday)

    if flat_stocks:
        st.write(f"""
            **Min open flat**:    {np.round(min_open_flat,3)}
            
            **Max open flat**:    {np.round(max_open_flat,3)}
            """)



    ####### GTREND

    st.header("★ Google Trend")
    pytrend = TrendReq()
    std_list = start_date.split("-")
    edd_list = end_date.split("-")
    trend_str = st.text_input("Google Trends query:", "apple")
    geo_tag = st.text_input("""Geographic location [e.g. None (global), GB, GB-ENG, US]:""", "GB") #"GB-ENG" #US
    gprop = st.text_input("""GT property [e.g. None (all channels), news, images, youtube or froogle]:""", "") #"news"


    empowered_gt = st.checkbox("empowered historical interest", value=False)
    if empowered_gt:
        gtrend_df = pytrend.get_historical_interest([trend_str],
                year_start=int(std_list[0]), month_start=int(std_list[1]), day_start=int(std_list[2]),
                year_end=int(edd_list[0]), month_end=int(edd_list[1]), day_end=int(edd_list[2]),
                frequency="daily", geo=geo_tag, gprop=gprop
            ) 
    else:
    
        pytrend.build_payload([trend_str], cat=0, timeframe=start_date+" "+end_date, geo=geo_tag, gprop=gprop)
        gtrend_df = pytrend.interest_over_time()

    if st.checkbox("see tabular data", value=False):
        gtrend_df

    if st.checkbox("compare with another", value=False):
        trend_str2 = st.text_input("Google Trends query:", "iphone")
        geo_tag2 = st.text_input("""Geographic location:""", "") #"GB-ENG" #US
        gprop2 = st.text_input("""GT property:""", "news") #"news"
        pytrend.build_payload([trend_str2], cat=0, timeframe=start_date+" "+end_date, geo=geo_tag2, gprop=gprop2)
        gtrend_df2 = pytrend.interest_over_time()
        st.line_chart(pd.concat([gtrend_df[trend_str], gtrend_df2[trend_str2]], axis=1))
    else:
        st.line_chart(gtrend_df[trend_str])



    min_gtrend = np.min(gtrend_df[trend_str])
    max_gtrend = np.max(gtrend_df[trend_str])
    st.write(f"""
    **GTrend Min**: {min_gtrend}
    
    **GTrend Max**: {max_gtrend}
    """)



    st.subheader("Worked Google Trend")
    if normalize:   
        gtrend_df[trend_str] = (gtrend_df[trend_str]-min_gtrend)/(max_gtrend-min_gtrend)
    
    if empowered_gt:
        gtrend_pday = pd.DataFrame(
            gtrend_df[trend_str],
            columns=[trend_str]
        )
    else:
        # gtrend_pday = []
        # rep = 7  #ogni dato del file csv trend viene ripetuto rep volte
        # for p in gtrend_df[trend_str]:
        #     for i in range(rep):
        #         gtrend_pday.append(p)

        interpol_type = st.checkbox("linear interpolation", value=False)
        dates_g = corr.compute_dates_vec(gtrend_df.index)
        gtrend_pday = corr.fill_trend(dates_g, gtrend_df[trend_str], line=interpol_type)


        tmp = [i for i in range(dates_g[0], len(gtrend_pday) + dates_g[0])]
        dates_fill = []
        for d in tmp: dates_fill.append(corr.to_date(start_date, d))
        
        gtrend_pday = pd.DataFrame(
            gtrend_pday,
            columns=[trend_str]
        )
        gtrend_pday.index = dates_fill
    
    flat_gtrend = st.checkbox("Flat GTrend")
    if flat_gtrend:       
        gtrend_pday[trend_str] = corr.flatter(gtrend_pday[trend_str])


    if polish_all:
        gtrend_pday["Unpolished GT Data"] = gtrend_pday[trend_str]
        gtrend_pday[trend_str] = corr.polish(gtrend_pday[trend_str], polish_n)


    # Alligning
    i0 = pd.Index.intersection(strend_pday.index,gtrend_pday.index)[0]
    strend_pday = strend_pday[i0:]
    gtrend_pday = gtrend_pday[i0:]


    st.line_chart(gtrend_pday)
    show_plot = st.checkbox("Show both Google and Stock trends", value=True)
    if show_plot:
        st.subheader("Google Trend vs Stock Open prices")
        if polish_all:
            if st.checkbox("Show unpolished", value=True):
                st.line_chart(pd.DataFrame.join(strend_pday, gtrend_pday))
            else:
                st.line_chart(pd.DataFrame.join(pd.DataFrame(gtrend_pday[trend_str]),strend_pday["Open"]))
        else:
            st.line_chart(pd.DataFrame.join(strend_pday, gtrend_pday))



    ####### RANDOM WALK
    st.header("★ Random Walk")
    rndWalk = strend_pday.copy()[[]] #copy stock_df and erase all columns
    max_length = int(st.text_input("max consecutive steps:", "1"))
    st.button('refresh')
    rndWalk["random walk"] = corr.brownian(init_stock_value, len(strend_pday.Open), max_length=max_length)
    min_open_rnd = min(rndWalk["random walk"])
    max_open_rnd = max(rndWalk["random walk"])
    if normalize:
            rndWalk["random walk"] = (rndWalk["random walk"]-min_open_rnd)/(max_open_rnd-min_open_rnd)  

    if polish_all:
        rndWalk["Unpolished RND Data"] = rndWalk["random walk"]
        rndWalk["random walk"] = corr.polish(rndWalk["random walk"], polish_n)

    st.line_chart(rndWalk)

    show_plot = st.checkbox("Show both Random Walk and Stock trends", value=True)
    if show_plot:
        st.subheader("Random Walk vs Stock Open prices")
        st.line_chart(pd.DataFrame.join(strend_pday, rndWalk))



    
    ####### Correlation G-S Trends
    st.header("★ G-S Trends Correlation")
    corr_g = []
    mcorr_g = []
    lk_g = []
    st.write("--------------")
    st.write(f'**Shift di x giorni:$~~~~~~~~~~$correlation , likelihood | magnitude-correlation**')
    
    for i in range (max_shift):
        r, lk = corr.correlation(strend_pday.Open, gtrend_pday[trend_str], i)
        mang_cor = corr.correlation_Magnitude(strend_pday.Open, gtrend_pday[trend_str], i)
        corr_g.append(r)
        mcorr_g.append(mang_cor)
        lk_g.append(lk)
        st.write(f'Shift di {i} giorni:$~~~~~~~~~~${np.round(r,3)} ,   {np.round(lk,3)} | {np.round(mang_cor,3)}')
    
    maxmin_stat_g = corr.maxmin_shift(strend_pday.Open, gtrend_pday[trend_str], max_shift=max_shift, corr_type="classic")
    maxmin_stat_mag_g = corr.maxmin_shift(strend_pday.Open, gtrend_pday[trend_str], max_shift=max_shift, corr_type="magnitude")
    st.write()
    st.write(f"""
    --------------

    **Max correlation**: {np.round(maxmin_stat_g[1],3)} [{maxmin_stat_g[0]} days shift]

    **Min correlation**: {np.round(maxmin_stat_g[3],3)} [{maxmin_stat_g[2]} days shift]
    """)

    st.write(f"""
    --------------

    **Max magnitude-correlation**: {np.round(maxmin_stat_mag_g[1],3)} [{maxmin_stat_mag_g[0]} days shift]

    **Min magnitude-correlation**: {np.round(maxmin_stat_mag_g[3],3)} [{maxmin_stat_mag_g[2]} days shift]

    --------------
    """)

    corr_g_df = pd.DataFrame(
    )
    corr_g_df["correlation"] = corr_g
    corr_g_df["likelihood"] = lk_g
    corr_g_df["magnitude-correlation"] = mcorr_g


    st.subheader("Magnitude Correlation")
    st.line_chart(corr_g_df["magnitude-correlation"])


    # corr_g_vec, lk_g_vec = shifted_coorelation(strend_pday, gtrend_pday, max_shift=max_shift, corr_type="classic"))
    # plt.figure()
    # plt.plot(corr_g_vec)
    # plt.figure()
    # plt.plot(lk_g_vec)


    ####### Correlation S-R Trends
    st.header("★ S-R Trends Correlation")
    corr_r = []
    lk_r = []
    mcorr_r = []
    stdv = 0
    st.write("--------------")
    st.write(f'**Shift di x giorni:$~~~~~~~~~~$correlation , likelihood | magnitude-correlation**')
    for i in range (max_shift):
        r, lk, inc_stdv = corr.correlation(strend_pday.Open, rndWalk["random walk"], i, return_std=True)
        stdv += inc_stdv
        corr_r.append(r)
        lk_r.append(lk)
        mang_cor = corr.correlation_Magnitude(strend_pday.Open, rndWalk["random walk"], i)
        mcorr_r.append(mang_cor)
        st.write(f'Shift di {i} giorni:$~~~~~~~~~~${np.round(r,3)} ,   {np.round(lk,3)} | {np.round(mang_cor,3)}')

        
    maxmin_stat_r = corr.maxmin_shift(strend_pday.Open, rndWalk["random walk"], max_shift=max_shift, corr_type="classic")
    maxmin_stat_mag_r = corr.maxmin_shift(strend_pday.Open, rndWalk["random walk"], max_shift=max_shift, corr_type="magnitude")
    st.write()
    st.write(f"""
    --------------

    **Max correlation**: {np.round(maxmin_stat_r[1],3)} [{maxmin_stat_r[0]} days shift]

    **Min correlation**: {np.round(maxmin_stat_r[3],3)} [{maxmin_stat_r[2]} days shift]
    """)

    st.write(f"""
    --------------

    **Max magnitude-correlation**: {np.round(maxmin_stat_mag_r[1],3)} [{maxmin_stat_mag_r[0]} days shift]

    **Min magnitude-correlation**: {np.round(maxmin_stat_mag_r[3],3)} [{maxmin_stat_mag_r[2]} days shift]

    --------------
    """)

    ####### Log
    st.header("★ Correlations Log")

    for i in range(rtrends_numb-1):
        #prv_rand = random_walk
        random_walk = corr.brownian(init_stock_value, len(strend_pday.Open), 0.5, max_length=max_length)
        if normalize:     
            min_rnd = np.min(random_walk)
            max_rnd = np.max(random_walk)
            random_walk = (random_walk -min_rnd)/(max_rnd-min_rnd)

        if polish_all:
            random_walk = corr.polish(random_walk, polish_n)

        for i in range (max_shift):
            #r, lk = correlation_noMagnitude(prv_rand, random_walk, i) 
            r, lk, inc_stdv = corr.correlation(strend_pday.Open, random_walk, i, return_std=True)
            mang_cor = corr.correlation_Magnitude(strend_pday.Open, random_walk, i)
            mcorr_r.append(mang_cor)
            stdv += inc_stdv
            corr_r.append(r)
            lk_r.append(lk)

    rtrends_numb = len(lk_r)
    stdv = stdv/rtrends_numb



    # Statistics Log

    st.write("""--------------""")
    st.write(f"""Mean correlation with GTrend:$~~~~~~~~~~${np.round(np.mean(corr_g),3)}""") 
    st.write(f"""Mean likely with GTrend:$~~~~~~~~~~~~~~~~${np.round(np.mean(lk_g),3)}""")
    st.write(f"""Mean magn correl with GTrend:$~~~~~~~~~~${np.round(np.mean(mcorr_g),3)}""")
    st.write(f"Max correlation with GTrend:$~~~~~~~~~~${np.round(maxmin_stat_g[1],3)}     [shift: {maxmin_stat_g[0]}]")
    st.write(f"Min correlation with GTrend:$~~~~~~~~~~${np.round(maxmin_stat_g[3],3)}     [shift: {maxmin_stat_g[2]}]")
    st.write("""--------------""")

    st.write(f"""Mean correlation with RndWlk:$~~~~~~~~~~${np.round(np.mean(corr_r),3)}""") 
    st.write(f"""Mean likely with RndWlk:$~~~~~~~~~~~~~~~~${np.round(np.mean(lk_r),3)}""")
    st.write(f"""Mean magn correl with RTrend:$~~~~~~~~~~${np.round(np.mean(mcorr_r),3)}""")
    st.write(f"Max correl with first RTrend:$~~~~~~~~~~${np.round(maxmin_stat_r[1],3)}     [shift: {maxmin_stat_r[0]}]")
    st.write(f"Min correl with first RTrend:$~~~~~~~~~~${np.round(maxmin_stat_r[3],3)}     [shift: {maxmin_stat_r[2]}]\n")
    st.write("""--------------""")


    st.subheader("Confidence from likelihood calculator")
    lk_target = float(st.text_input("reference likelihood:", "1.645"))
    # p =  0.05, 0.025,  0.01, 0.005, 0.001, 0.0005 for
    # z = 1.645, 1.960, 2.327, 2.576, 3.091,  3.291
    st.write(f"Total Random trends:$~~~~~~~~~~~~~~~~~~~~${rtrends_numb}")
    st.write(f"Standard Deviation RTrends:$~~~~~~~~~~${np.round(stdv,5)}")
    st.write(f"Expected probability for {lk_target}:$~~~~~~~${np.round(corr.conf_int_prob(lk_target),4)}")
    lkstdv = np.round(lk_target*stdv,4)
    st.write(f"Likelihood*stdv:$~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~${lkstdv}")
    st.write(f"Prob(correlation > 0.5 + {lkstdv}):$~~~~~~~~~~~${np.sum(np.array(lk_r)>lk_target)/rtrends_numb}")
    st.write(f"Prob(correlation < 0.5 - {lkstdv}):$~~~~~~~~~~~${np.sum(np.array(lk_r)<-lk_target)/rtrends_numb}")
    st.write(f"Prob(|correlation-0.5| > {lkstdv}):$~~~~~~~~~~~${(np.sum(np.array(lk_r)>lk_target) + np.sum(np.array(lk_r)<-lk_target))/rtrends_numb}")
    
