import numpy as np
import pandas as pd
from datetime import datetime
import math



def to_date(start_date, day_of_year):
    
    cnt = 0
    init_d = int(start_date[:4])
    p = 0
    while True:
        try:
            d = pd.to_datetime(str(init_d + cnt) + str(day_of_year - p), format='%Y%j')
            break
        except:
            # leap years
            if (init_d+cnt)%4 == 0:
                p += 366
            else:
                p += 365
            cnt += 1
    return d


def fill_trend(x_axis, trend, line=True):
    out = []
    prev = 0
    x0 = x_axis[0]
    cnt = 0
    skip = 0
    for i in range(x0, x_axis[-1]+1):
        if skip>0:
            skip -=1
            continue
        if i in x_axis:
            out.append(trend[cnt])
            prev = trend[cnt]
            cnt += 1
        else:
            if not line: 
                if i+1 in x_axis:
                    #out.append(prev) #trend[cnt] #fully constant
                    out.append((prev+trend[cnt])/2)
                else:
                    out.append(prev)
            else:
                for j in range(i+1, x_axis[-1]+1):
                    if j in x_axis:
                        for k in range(i, j):
                            out.append((1-(k-i+1)/(j-i+1))*prev + (k-i+1)/(j-i+1)*trend[cnt])
                        skip = j-i-1
                        break
                    elif j == x_axis[-1]+1:
                        for k in range(i, j):
                            out.append(prev)
                        return out
    return out


def brownian(x0, steps, alpha=0.5, p=0.5, max_length=5):

    res = []
    x = x0
    while len(res)<steps:
        n = np.random.randint(1,max_length+1)
        spin = 2*int(np.random.random()>p) -1
        res += [x+i*spin*alpha for i in range(1,n+1)]
        x += spin*n*alpha
    return res[:steps]


def trend_derivative(trend_pday, eps=1):
    d_trend = []
    t_prev = None
    for t in trend_pday:
        if t_prev is None:
            d_trend.append(0)
            t_prev = t
        else:
            d_trend.append((t-t_prev)/eps)
            t_prev = t
    return d_trend


def flatter(trend):
    a = np.arange(len(trend))
    b = a.reshape(len(trend),1)    
    
    alpha = np.linalg.lstsq(b, trend-trend[0], rcond=None)[0]
    
    for i in range(len(trend)):
        trend[i] = trend[i] - alpha*i    
    return (trend)
 
## Polish rende più smooth la serie trend che gli viene passata come argomento eseguendo la media pesata tra un punto e i suoi 
## vicini n volte  

def polish(trend, n):

    for k in range(n):
        for i in range(len(trend)):
            
            if i==0:
                trend[i] = 0.75*trend[i] + 0.25*trend[i+1]
            else: 
                if i==(len(trend)-1): 
                    trend[i] = 0.25*trend[i-1] + 0.75*trend[i] #0.5
                else:
                    trend[i] = 0.25*trend[i-1] + 0.5*trend[i] + 0.25*trend[i+1]
    return trend
            

def correlation (strend, gtrend, shift, return_std=False):

    series_strend = []
    series_gtrend = []
    
    ## Converto le serie originali in serie con valori {0, 1, -1} ove 0 significa nessuno incremento, 1 incremento rispetto al 
    ## valore precedente, -1 decremento rispetto al valore precedente
    
    for i in range(len(strend)):
        if i < (len(strend)-1):
            if strend[i] <= strend[i+1]:
                if strend[i] < strend[i+1]: 
                    series_strend.append(1)
                else:
                    series_strend.append(0)
            else:
                series_strend.append(-1)
 
 
    for i in range(len(gtrend)):
        if i < (len(gtrend)-1):
            if gtrend[i] <= gtrend[i+1]:
                if gtrend[i] < gtrend[i+1]: 
                    series_gtrend.append(1)
                else:
                    series_gtrend.append(0)
            else:
                series_gtrend.append(-1)    
    
    count = 0
    for i in range(min(len(series_strend), len(series_gtrend)) - shift):
        if series_strend[i+shift] == series_gtrend[i]:
            count += 1
        
    r = count / (min(len(series_strend), len(series_gtrend)) - shift)  
    

    n = min(len(series_strend), len(series_gtrend)) - shift - 1
    stdv = (0.25/n)**(1/2)  #standard deviation di una distribuzione binomiale = (1/n*p*(1-p))^(1/2) con p = 0.5

    
    likely = (r-0.5)/stdv
    if return_std:
        return(r, likely, stdv)
    else:
        return(r, likely)


def correlation_Magnitude_old (strend, gtrend, shift, m):

# m-1 channels up and m-1 channels down

    series_strend = []
    series_gtrend = []
    v = 0
    
    for i in range(len(strend)):
        
         
        v = (abs(strend[i]) // (1/m)) + 1
        
        if strend[i] >= 0 :
            if strend[i] == 0 : 
                series_strend.append(0)
            else:
                series_strend.append(v)
        else:
            series_strend.append(-v)
    
    
    for i in range(len(gtrend)):
        
        v = (abs(gtrend[i]) // (1/m)) + 1
        if v > m : v = m
        
        if gtrend[i] >= 0 :
            if gtrend[i] == 0 : 
                series_gtrend.append(0)
            else:
                series_gtrend.append(v)
        else:
            series_gtrend.append(-v)    
    
    print ('strend')
    print (strend)
    print ('gtrend')
    print (gtrend)
    print ('series_s')
    print (series_strend)
    print ('series_g')
    print (series_gtrend)
    
    count = 0 
    a = 0
    alpha = 1  # tradeoff tra magnitude e direzione del trend
    
    for i in range(min(len(series_strend), len(series_gtrend)) - shift):
        a = np.sign(series_strend[i+shift] * series_gtrend[i])
        a *= np.exp(-alpha* abs( abs(series_strend[i + shift]) - abs(series_gtrend[i]) ) )
        count += a
   
    return (count)

def correlation_Magnitude (strend, gtrend, shift):
 
    series_strend = []
    series_gtrend = []
 
 
    for i in range(len(strend)-1):
 
        if (strend[i] != 0):
            inc = (strend[i+1] - strend[i])/strend[i]   
        else:
            inc = strend[i+1] - strend[i]  # come se facesse /1 cioè l' incremento relativo risetto al max (DA CAMBIARE)
        series_strend.append(inc)
 
 
    for i in range(len(gtrend)-1):
        if (gtrend[i] != 0):
            inc = (gtrend[i+1] - gtrend[i])/gtrend[i]    
        else:
            inc = gtrend[i+1] - gtrend[i]  # come se facesse /1 cioè l' incremento relativo risetto al max (DA CAMBIARE)
        series_gtrend.append(inc)
 
    ## Testing/debugging    

    #    print ('strend')
    #    print (strend)
    #    print ('gtrend')
    #    print (gtrend)
    #    print ('series_s')
    #    print (series_strend)
    #    print ('series_g')
    #    print (series_gtrend)
 
    count = 0 
    a = 0
    alpha = 1  # tradeoff tra magnitude e direzione del trend
 
    for i in range(min(len(series_strend), len(series_gtrend)) - shift):
        a = np.sign(series_strend[i+shift] * series_gtrend[i])
        a *= math.exp(-alpha* abs( abs(series_strend[i + shift]) - abs(series_gtrend[i]) ) )
        count += a
 
    return count/(min(len(series_strend), len(series_gtrend)) - shift)


def compute_dates_vec_old(trend_dates):
    dates = []
    sw = 0
    prev = -1
    lprev = 0
    d0 = trend_dates[0]
    d0oy = d0.dayofyear
    for d in trend_dates:
        if type(d) == str:
            d = datetime.strptime(d,'%Y-%m-%d').timetuple().tm_yday
            # leap years
            if int(d[:4]) %4 == 0:
                l = 1
            else:
                l = 0 
        else:
            l = int(d.is_leap_year)
            doy = d.dayofyear
            # d = d.dayofyear
            # d = str(d).split()[0]
            # d = datetime.strptime(d,'%Y-%m-%d').timetuple().tm_yday
        if prev > doy: #prev > d.year:
            sw += 365 + lprev
            #sw += 365*(d.year-prev) + lprev
        # elif prev == 0:
        #     sw -= doy-1
        lprev = l
        prev = doy #d.year
        dates.append(doy+sw)

    return dates


def compute_dates_vec(trend_dates):
    dates = []
    if type(trend_dates[0]) == str:
        d0 = datetime.strptime(trend_dates[0],'%Y-%m-%d').timetuple().tm_yday
    else:
        d0 = trend_dates[0]
    for d in trend_dates:
        if type(d) == str:
            d = datetime.strptime(d,'%Y-%m-%d').timetuple().tm_yday
        doy = d.dayofyear     
        dates.append(d0.dayofyear + (d-d0).days)   

    return dates


def shifted_coorelation(trend, predictor_trend, max_shift=20, return_std=False, corr_type="classic"):
    dim = min(max_shift, len(trend))
    corr_vec = []
    lk_vec = []
    std_vec = []

    if corr_type == "classic":
        for shift in range(dim):
            if return_std:
                corr, lk, std = correlation(trend, predictor_trend, shift, return_std=return_std)
                std_vec.append(std)
            else:
                corr, lk = correlation(trend, predictor_trend, shift, return_std=return_std)
            corr_vec.append(corr)
            lk_vec.append(lk)

        if return_std:
            return corr_vec, lk_vec, std_vec
        else:
            return corr_vec, lk_vec
    elif corr_type == "magnitude":
        for shift in range(dim):
            corr = correlation_Magnitude(trend, predictor_trend, shift)
            corr_vec.append(corr)
        return corr_vec, None
    else:
        raise Exception("Unknown correlation fucntion")


def maxmin_shift(trend, predictor_trend, max_shift=20, return_lk=False, corr_type="classic"):
    corr_vec, lk_vec = shifted_coorelation(trend, predictor_trend, max_shift=max_shift, corr_type=corr_type)
    min_shift = np.argmin(corr_vec)
    min_c = corr_vec[min_shift]
    max_shift = np.argmax(corr_vec)
    max_c = corr_vec[max_shift]
    if return_lk:
        return max_shift, max_c, lk_vec[max_shift], min_shift, min_c, lk_vec[min_shift]
    else:
        return max_shift, max_c, min_shift, min_c

def conf_int_prob(z):
    return 1 - (1.0 + math.erf(z / math.sqrt(2.0)))/2.0
