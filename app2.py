import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from bokeh.layouts import row
from bokeh.models import Range1d, LabelSet
from bokeh.plotting import figure, output_notebook, show, curdoc
from bokeh.models.widgets import Panel
from bokeh.models.widgets import Tabs
from bokeh.models import ColumnDataSource
from bokeh.models.tools import HoverTool
from bokeh.models import BoxAnnotation
from bokeh.layouts import row
from bokeh.models import Range1d, LabelSet
from bokeh.plotting import figure, output_notebook, show,curdoc
from bokeh.models.widgets import TabPanel
from bokeh.models import ColumnDataSource
from bokeh.models.tools import HoverTool
from bokeh.palettes import inferno
from sklearn import model_selection
from sklearn import ensemble
from sklearn import svm
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from PIL import Image

st.set_page_config(page_title='Game Project Prediction', 
                   page_icon='https://cdn-icons-png.flaticon.com/512/3971/3971167.png', 
                   layout="centered", initial_sidebar_state="expanded",
                    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)
df = pd.read_csv('vgsales_all1.csv')
df_ml = pd.read_csv('df_ml.csv')
DICT_GENRE = {'Role-Playing': 'dodgerblue',
        'Action': 'tomato',
        'Shooter': 'mediumaquamarine',
        'Sports': 'mediumpurple',
        'Platform': 'sandybrown',
        'Racing': 'lightskyblue',
        'Adventure': 'hotpink',
        'Fighting': 'palegreen',
        'Misc': 'violet',
        'Strategy': 'gold',
        'Simulation': 'lavender',
        'Puzzle': 'salmon',
        'Autre': 'aquamarine'}
flierprops = dict(marker="X", markerfacecolor='darkviolet', markersize=12,
                  linestyle='none')


selected = option_menu(
    menu_title="Menu", #required
    options=['Projet','Contexte','Méthodologie','Analyse','Modélisation','Conclusion','Modelisation pack'],
    icons=['book','search',"eyedropper",'calculator','clipboard','controller'],
    menu_icon='compass',
    default_index=0,
    orientation='horizontal',
    styles={
                "container": {"padding": "0!important", "background-color": "#black"},
                "icon": {"color": "#blank", "font-size": "13px"},
                "nav-link": {
                    "font-size": "12px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#9400D3",
                },
                "nav-link-selected": {"background-color": "#9400D3"},
                
            },
)


if selected == "Projet":
    st.title("La Data Science sera t elle le \"cheat code\" de la vente d'un jeu vidéo ?")
    st.markdown("Estimer les ventes d'un produit avant son lancement peut être une véritable force pour la rentabilité d'une entreprise. Dans le cadre de ce projet nous allons essayer de déployer un modèle qui permettra de prédire les ventes d'un jeu.")
    
    image = 'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAoHCBQSFRUVFRYYGBgaGRwYGBgYGRgaGBgYGBwZGRgYGRgcIS4lHB4rIRgaJjgnKy8xNTU1GiQ7QDs0Py41NTEBDAwMEA8QHhISHjQnJSs0NDU1NDQ3NDYxNTE0NDQ0NDQ0NTQ3PzQ0NDQ0MTQxNDE0NDE0NDE0NDQxMTQ0NDExNP/AABEIALcBEwMBIgACEQEDEQH/xAAcAAABBAMBAAAAAAAAAAAAAAAABAUGBwECAwj/xABNEAACAQIDAwcHBwkECgMAAAABAgADEQQSIQUGMQcTIkFRYXEycoGRobHBFEJSkrLC0RUjMzRTc4Ki8ENiY7MWJCU1RIOTo9LhVGTi/8QAGgEBAAMBAQEAAAAAAAAAAAAAAAECAwQGBf/EACkRAQEAAgECBQQDAAMAAAAAAAABAhEDITEEEjJBcRNRkcEiQmEFFIH/2gAMAwEAAhEDEQA/AKcpoWNgCT2AXipNmVjboFb/AEiF9jEGSJL26J06rXA9GVv67IItuAHH5treoKLnw9cnQZU2O3EsLdqgsL9hY2UekxQmyUAuSzaX6gPX5P8ANHJV4EjWx1N831mzEfXmxXr7uP8A+jeNINFXZC8QxHDystte9io9RMT1NjuOBU9gOZWPgGAkgK2v26dx6uu4v65oEtewt0tbAi/iFy39Ib8WhGnwNUa5GI4XXpC446rcRMRaS/Jqp6wSAdL+ggX9RWZagWADAWsQM4Fr+L318QY0lDoSUVdnUDxCjT5ua9/AWU+i0UbM3LOMD8xUClMtxV0DZr8CoNuHYeMaEPhJLjtx8fSueZLgddMh7+CjpeyMGIwz0zldGQ9jKVPqMgcYQhAIQhAIQhAIR/2DsP5XRxbJmNWiiVEUWsy5iKgItcm1iLdltbxggEIQgEIR6/JIXBnEvmDNVWnSGgUrldnc6XOqgC1uB4wGWEIQCEIQCEIQCEkGzd0sXiACqZVOoZzluO0DVreiSTB8m/7SqT3IoHta/ulbnjPdeceV9leTpRou5sqsx7ACT6hLdwe42Fp2vTzHtck+zh7JIMJspKYsiKo7FUAeoTO809o0nDfeqXTdnGEX5h/SQPYTCXn8mWEj6tX+hj96ocNbUaTsmMdbWY6dtj74lvC86HKcU2kwFiqkd2n4zvS2ghtxGlrHUD+vCM95sh1HiPfAfGrpr5RvbTRR6Dc6egTDYrsUenXh7PZHWhsVTgXxRfg2QAHUPnAVCvzgUzMToRdOILWYxJQ6mu5+cR4ae6azUTYQMyweS3CCoMTrYjm7aX455X0nfJbtilRq1KDnK1bJzZNspZM3Q845tO21uNrhPK2zXUEggj1GMuJr03BVlDg9TKCp49TePZJjX8lvNPukKNBRSV79Mta3UVsb6dVrD60CI717CwhoVqiYdKbohZSmZbEf3VIU+qVTLm3o/VMR+7aUzKpEIQgEIQgTrklxGTF1B9LDv61KMPcY919m7Oxleo2GW7f2q5bUw1yM1MMOJIa9tNBaRHk7qkY6kB85aqH+Kk9vbaPHJnigj183Wiewn8eHVLzrEU7/AOidL9mn1RMf6J0v2a/VEln5QTtEPygnaI1BCdo7BwVBUfEKVQMAcgsWNmsrFdcumttdIj37xSthcMEy5GdmUKLDKiqFsOoWfhHXlHxqthkVSP0q+xXkP3jf/Vtnp180727mfKPseyTZqIiNwhCZrCEIoweEes4SmpdmNgqi5P4eMBPLK3F3FaoVr4hLLxSmw49jMOzsB9Mf9xuT9cPatiMrVeKjitPw+k3f1dXaZ8+Kp0hqRMs8/aN8OP3pMuzwJ0SgojZjNuk6It/YImSrWqHU5R3fjMdx0avueWyDQanu6pwKziKTgaTthlY8ZGzWmLQizmBCTpXzPNl4Xmt4XnY4m15mmekviPfNLzamekviPfAdZsJrNhJQyJtNRNoGZwxLlSpBII1BBsQRqCCNbztE+K6vT/V+qKL33R258twS1GPTUMlTz0HlW6swKt/FGESP8keOK1MTQv0Xpc4B1ZqZynxJDj6okhEgNG9X6piPMPvEpqXHvb+p4jzPvCU5ISIQhAIQhAfNyq3N4/CN/jIp8GYKffN8A5w1asgPklk+qxGvfpG7YtXJiKDfRq029TqY87xUcmPxi9XPVD45nzD2H2y+KKV/lVu2H5VbtjReYvA7baxrVFUE/Ovrw4Gbb3LlGBXswVL+Zqj/AHo3Y49ER55RUyYpU+hh6Ceqmp+MZdoRFZtaay0OS/Z1GulTnaaPlC2zqGtcve1/AeqZZZeWbbcXHM97utTasLSwuSuvTpnEMwGboWJ+ic9wOzUD2SyP9H8J/wDGpfUX8Jj8hYUf8PS+on4TLLl3NaXxxxl3u/h1w22RVDBATl0JtoCe/rhTwAqNma5I7fgOqKNnph8Mr9BKaeWSAFXgAS1uGgGvdOv5Ww5Jyuhy8QhzEX4Xy3twmfSzrXRLNbkchs5Q2g0ipEUESJbyb3vTdaVBOm4vnfgBe3kDUn0iGAFUUGqVnZ3INyeHHgqjQDwkbkX8ts3U2BWahBeVxVVmV6i6ZywA7uA9OntjBvXtvFYbmVpV6iXBBAYnyco6/GThfNdM88fLjaujSE83NvTjj/xVb/qN+MxN/JXN9SEd4Xmt4XmrFteb0j0l8R75yvN6XlL4j3wHgTaaibCShmZms2gZibF9Xp/q0URNi+r0xRIOTqoVx9MD5yVlPhzTvx8UEngkM5L8KXxjPbSnRqMT2FgEHhfM3qMmgkBl3v8A1PEeYPtLKclxb4/qWI80fbWU7ISIQhAIQhA2RiCCOINx4iTDfpQNpYg/SyMP4qVNyf675DZOd/7fK6Ldb4Wk1/4XX7oloio/CYhJGopZ6lFOtnC/WZR8Y78qH+8q47FpD/tU4m2FR5zGYRe2shH8LKWHqEWcqyW2piO8Uz/2kHwkZEQ+WjyY1GSjXZbXtT1IuFBdgzEddgSfRKultcj9MstYA2OVbG17HM3VMeX0urw/bP4/cSRNpYglOiDcgKMpHOgkgupvoAAG0vo077MxdSo9nItlLaLazdDMnit7enWNm8m9a4CotOqzFmQOCiAixJFtX46RkPKRQ7an1F/8pz6ys6Reyf5+U22nXNOk7gBiqkgHh6e7rPcDGDd7Fs7VKbonRKsDTQIOnmvmW/8Ad498ZW5RaB/afVT8Y5bq7bo4pnSgDTyjORkRQbm1+j1yLjlJuxfG4zC46lt99km9danQxWHeocqhDckE65j2RxbevArSYc+p4nKLk37ALdsifKPVLigzcShvbudhK/mvHxzLHanLy5Y6k+yy15QMMuVBQqFVHG6gk9trmwvrxkP3n218sqh1UoijKqk3PaST2n4RkhNphjj1jmy5Mspq0QhCWUKYTEJIzN6XlL4j3znN6PlL5w98B6mRMTIkoZmZiEDMTYs8PA/1fqimWHyXbFw9YVMRUph3puFQtchOiGzBeGa/Xa46ooduTrd1sHhalSouWpWGYqRYoiqcikHUHpMx84DqnMSc4nyH81vcZBhIDJvn+pYjzV+2sp2XFvn+pYjzV+2kp2QkQhCAQhCASfb9C67MqW8vCIt/N1+/IDLD3yQ/IdjuB/ZlSfFaBH2WloiopCEJIe9xaRfaeEFtMzN4ZUcn4TtywU8u0nP0qdM/yAfCKuS2lm2mh+jTdh3cFP2pty3U7Y+mfpYdD6nqr8IvYV1Lf5Fv7XzV+00qCW/yK/2vmj7RmPJ2dXh+2fx+4j3K3Vz4qkwFr0bdXU9QdXhIFJ1yqj/WKOlvzR7eqrV7ZBZOHpY8nq/DEn/JKfz1f92PtCQCTDk9qsjYgqcp5sa+Bv8ACRyTeNTx+r8l3KD5FDzW/wAxpAZP+UH9HQ8Kn+Y0gErw+lp4j+vwIQhNXOIQhAUQmIQMzaj5S+cPeJpNqPlL5w94gPHOdILbit7+Btb2zqIlP6UeafeIplkMzMxMwMy1eSP9XxH74fYWVWoJIABJPADUnwEs7k0xSYfD1xWbI3PeS4Ia2RNcvG3fAn+J8h/Nb3GQYRdt7ewZHTD2zMCA7i6r22T5xtfiV6uMiA2y6svFgLBhYXbTUiw0PXA676n/AFKv4L9tJT0t3fKqHwFVlNwQhB7i6SopFSIQhIBCEIBLL3oUnY+zmHEMi+hqb3+xK0lqbap59gYcj5ppH+aon35MEDhCEshOeR2jfHVn+jRsO7Myj7sUcu9Cz4N+1aiX81lYa/xmdeROl+exjf3aY9Ze49kc+XPD3wmHqW8ivl8BURj70Eewo6W/yK/2nm/elQS3uRU6v5v3hMeT0unw/wDb4/cR7lZ/T0OH6NuH72pIGZZHLNTtiKFhpzbcB/fY/GVzlPYZON6M85dtJNeTTDirVroSRene47mH4yG5D2eyTrknQ/Kaun9kftLI5L/Gp4pfN2+7pygi1Oj/AMz2VWlfSw+UIfmqXjW/zTK8leH0r+I/r8CEITVziEIQO0JiEDM2o+UvnD3iaTej5S+cPeIDmf0o8w++KYmP6QeYffFMshkGcWxigkWOnYeuFaplEaabXYnvkbSnG6iuTdVu50Hb6+oR32gXVyHYEg2IDKwU9nRNlPdIlQxr0qYZSRrrbsijZe1ErGoXXmyDmQUlCprZQpA4cOJ46njJQlKbWVcPUpMl3YjI4NsvSDMGHzuAserwvdir1B9IqSMyMvHMBmWxvoTbTvsOucWrEdZYXvlJsBwDWNu6d6VNBTzkrcuFC5wpGY8VWxLte56gADrqIK57Qxx+R4ii1/mMhPXd6ZIGmnEm3dfrkDk0xdPMpRibFcq5gAQBe3qv2n4SHOpBIPEGx9EqnbSEIQCEIQCW7iKRbd9QP2aH1Yhb+y8qKXXgFvsK3/1XP1WZ/uy0RVWCYUzImSbyRafIlRsmLftqKvqW/wB6SHlVw3ObLxFhcoUcfwuoY/VYxs5GqVsJWb6VY+xEHwkw3kwnPYTE0hxejUUecUbL7bR7DyrLZ5GnymoR1KePDisqeWtyPfP80+9JjydnT4f+3ws3FPSfV6dFraXZVa1+q54azj8mo9WHof8ATX8I04lXVa7Llyc4hYEXLE5FFmsAttTYg9RuLx56hMssdSKzL2cMQKVNWc0KNlBY2ppewF9PVGHd7eqnindUw6UygBN0TUEkWup7REe/GzMViGorh8wXpZyHKKAbeUFNzp3GRNt0sWt8lRA9jcBnQnUkajQ2vKySzrWure02c+ULACqcPSp2XMah1vYEsrH2kxjPJxWyFhWp3texDD29XqjrvHVqh8O7I35u+Y2uBfLxI8I/7N2wtSk5U3vf2afCRjncZqN8uGZ4y37aVTX3axiHKaDnS4KjMCO0FbiNleg6MVdSrDirAgj0GW7T2oyoQdGRvV1j3yH7+MKgpVBx1W/aDqPcfXNcOXzXTm5PD+XG2VDIQhNnM6wnfA4J67hKa5mPAcNO0mOrboY0EjmC1vouje5oDHNqPlL5w94jrU3Yxy3Jw1XTXRCfdEg2bXRlzUai9IeUjjr7xAUn9IPMPvigmJ2UioLgjoHiO+bVKg7ZITYqpEVDjOtduM5UID9h9UynhBcOtNWZR0rEjxt1TlhnsJ0NWSFaYVsnOgEqTbMx1HWBbq49kZ9psxZbX0OluIPdA48JdVzEXva5yX7QP/UR1cazd3h+MgSKrjGfKXYsQABmtew7bACR7aS2qN36+sa+28xh8SwNiSQe09c0xdTM1+4CBwhCOWxMItWqoe+QFS9jYlSypYd92EhMm7o3QlpYbdzZ75gtItlOU9N+Pcbxs2/uQgR6mFLXQFmoscxKjUlH0vYa5SLzOcuNunTn4Plxx82txX8vXYFPNsdF7cLVHrWpKKnoLdGnfZ2GXtoEfWDfjNo5KpVZkmaUjdV8JsZItTc/a5wGxvlAQOefIylsoOepkvex4SycBW5xEdhbMLkcbd0ql6eXd2l31Ub62IJlqbOFqaeHxMQeV9p4bma1Wl9Co6fVYr8JLNyN6qWz1YurMTcWXTiVN7/w+2NnKDhxT2ljFHXVZ/8AqWf70jkzyxl6Vpx8lx3r3mlvnlWw/wCwf1r+Edt3t8fyizph6HSVc5z1Mote2llOusomTfky3ko4DEPz9wlRQucAnKVNxcDXKbnh3TPLjmujTHklvWRbGD2y6Plq0MqHQurF8pHC4yjTjwv1aR0fDU6lmUA9dwQR6xO1IU6yK9NldGF1ZCCpB6wRG3E7NZWzIzI1/KXr8Rwb0zG7k06Z5b6ejltDZubqEZX2KEJKi1+NtL+Mkq411vnUHvXQ+o6e2dEr06luon5rCx/9yOi38ohjbFBYtre1rH1X8Y3Y7dtKqqjhiq+TY2I4+vjLFbCLET4QExOnZFu5qqzbcGh9Kr61/wDGEsn5EIS/my+7LyY/ZSO59TLVGumbX6rRWcGz1SGR1BZukUbvIOo1mmJIRqYRVSxt0FAJ8nUsNWPiZNle0644aiR2fk8moV/hI9xm61q6+TiXHhUce5pK+ePafWZhql+OvjrCUY/KeMH/ABLnznZvtAzk+1MSfKdG86nQb7SSR1KaHiiHxRD7xOLYakf7NPQij3CBG2xznyqGGfvNCmD/ACZZybE0/nYTD/wisp/lqSSvs/DnjTX0Fx7mnBtkYc/MI8Hf4tAYvlWH68KB5tWqv2s05PUwr6NSqp5tdT7GpfGPdTYtA8C48HX4qYlfd+mTcPU9JQ/dEG0Xx+HFKo6A3AOhPGxFxfvsZxo0nqMFRWdjwVQWJ9AjrvHgclRWBLc5wFtQRlFu/iJau7+xaWAohRlDWvVqGwLHrux4KOAHxluPjuVY83POLGXvb2iqBupjrXGGq/UPu4xGmyMQ2bLSc5WKtZToy8VPeJeYrGoA1J0IvqbZwR2DKwsfXIDtJrVqwH7R/ttL8nFMZLKz8P4jLktmU1pDPyFiv2FT6pjlsXZ1aizNUpuikIAzKQL87SIF++xjuXgh1Exs6O3C/wAokOJxnOlqdIgMr9K5IzBSC4OXVQbgX69Yq2dtAc4KZN3VgCdNTYEkLxy62vwM4Y7D5hZCEOZWJI0bKb2btHdFWDQLbUE6XawBOvunB5o9HeLPff8A9/z7a/ao9q0glesoFgtR1A7AGIAl+7nC2CwQ/wAKn7QD8ZQ23v1nEfvqn22l/brDLhMGOyhS+ws78Hm8/VVElMhK/RJX1G0w0U7STLXrr2Vag9TsIlfgZZVaW1Uy7v4bv+Tn1uDLLwH6NPCV7vPTybBwy9gw3vWWHgP0aeaIHnzlYpZdqYn+8KTeulTv7QZDZO+WRbbSfvp0z/Lb4SCStSIQhIEm3S3xxGzm6Bz0yenSYnK394H5rd49IMu/dje7DbSU82crgdOm9gy9/wDeXvHptPNcVYDG1KDrVpOUdTdWU2IP4dRB0Ilcsdr452PUj4VWid8EOyRXcff2lj15urlp4hQLi4CVONylze+mq+q/VNBUmFmuldGOVvWUlpoyaG7L7V8O7umz0he4nZqg64XErpfdJMndCK8sJOldvPOP8tPOPwkwzyH7R8tPOMk/OTscRRzkwakTF5qXgKC8aKm1Xys6hMoUuqs5zsgNs+UDQGLTUjE1JPk2awzCiVDddtdICqtvAEqsjr0Q2XOG18StvjHjnJCdpqhauxchw/RW2jA2ub/1wkoFXQeEBUzzUvEhqzBqwO9PJzwqMAebp1CoPU7AKrejUjw7pI97c9RMiAkZrsBxNr2067GxtIajZkxL9i5F9A/EmWJja6JfyWqZWZELAM5F7ADjx6508Mlxsrh8VbjljZN9zTunhcQpLOCiZcoDaFiSLHLxAAB1PbI9tT9NW/eP9oya7L2izonPqtKoxIVCbFgLdJVJvb8JCtqP+frDq5xz1fSPXK8kkxkhwZZZctuU9vYgfEqvFgO8mw6+v0GdUaM+Lw5KqoXMRlBFwASAy3vfwi9DYKOwAe6c17PoYeqHLb+LYuFBso9RbTifDheOmw67uLkWXohe8jjbu4ThWpA1FGQsDe7aWWwuLjrvHNWtYDunB57MJjY9P/18Mue8uOV6dLNdN6VzvVh8uIqsODVH9YdgZfeyky4fDr2UqY9SLKa3poZqbv1rWqj/ALr/AAMu2mmVEXsUD1ACfQw7PMcnqvyonb6ZcVih2Yit/mNGyqeifAx63tS2OxY/x3P1jm+MZKw6LeBllV0b9pl2NRXsOGHtWTrB+Qnmj3SG8pCW2Ug7Hw4/mUSaYUdBPNHugULyz/7xP7mn96QGT/lo/wB4/wDJp/ekAlb3SJNuTTdNNo13NbNzNJQXymxZ2uEW9uGhJ83vkJnoPkv2Z8n2dTa1nrMazanUE5UF+zIqn0mIHHD7o7OpgqmGRuPl56ncbZyR6os2bsrDIpC4aghDMDlpIOu66hfolZ2XNf61ukejrrr1yJPv7QpYiupV3p3XK6ZTmdVCucpI6OgsR2dhEv00hNxRT6CfUX8J0DDhYeoSEDlJwWt1rDxRde4Wf3zniOUnDZGNOnVL2OXOEVc3VdsxsPRISk+JwFDFIOfpI6gsUuOALWUqRqLjLOBwlLAYetUoJUZUUuaXOO/RXV8gcmxtc2BsbTXd3aKYjD03Q3ARUdb3KuqhShXx6+sEHrjojXYfR4WvpbrBHA3jLHG7+xjlZdobR5T9nFQS9RT1qabEjuOUkeqEp3enZZwuLxFC2iVGC+YdU/lIhMfpxr9XIvx7XZO5iJIS8Y69MMCewn131PsjnmmrF2LzQ1JyLzmXgdXxAFrm1+HfbjG+tSurItQBTcZSuYrnubAgiw48ZviWY2K9XV1HxPH+uBiY85c6i2tuF9NFv6CTA4V8AjuXL6N0soGpA0Ot+GnZHJsQDwI9HfwiCoGHAki506PDq4jhxvOQdgTe3dbxPH0WkBea8w1friDnJpiH6DeHvkh2oaYZB11GLH0yw8ds5Gc4gLmqopyAk5cwvkuB3mVylXMidigS0w3vPvnV4eSy7fP8blcbLP8ATVgNmtWyV8UgFVD0bHSym6lgDa9ydJEdsC9auP8AEfhx8oybjADn+fzvfLlyX6HAC9u3SQXazfn637x/tGRz46xnz+Twl82d6+3b2hE1MdreuCC3rvrMM80zzlfRxuqmhacK+JVBmY2A1J7hxkPG9jki6gC4uQTe3XaYqc5WapnN1RgABwa9yD36AeucE8Pnb16PS8n/ACnBjhfLu340dKLjE03H06ma3Zzjn/yl0tKLwFQU3ptwCup7rAgy8ka4Bn0ZNPNW7u1Kb6i2Pxfng+tEMj1QXUjtFpJd+0/2hivOpn10aZjFSpkug7XQetgIQu/lOX/ZxH+JQ+2sl9AdFfNHuiTbOyaeLpczVzZCyt0TY3Rgy6+Ii5RaBQPLT/vH/k0/vSAS0OV3ACpjybkHmaY7vnSsailSQeINj6JWxLAF9BPU+DoCjSp0l0FNEQDuRQvwnmDZ7AVaZPAOpPhmF56exNYAEk2A1J7pMRUL5Rdv8wi0EPTqauRxFMafzHTwDSsTUuPhrxinefa7VXqYjrqOVp3HBFAtoewFfSxkb+X1fpn2RaHV73nVH0F9O/8Ar+tIyfLan0z7JkY6oPnn2Rs0mm628HySuCT+aey1ey3zX7ip18Ly46L6iec9m4pncIxuGNhfqJ0XXsvp6Zcu5G0zVw6Kx6dM821+Nl8gn+Gw8QYlDjtfdCliqrVmAu2W/wDCoX4TMfg8JOhRVWlkt0Q19bNwN+o2INvTN6rXOawAPzRew8Lkm3phCQNC00YwhA5M05M8IQOLtEztMQkDW8HW4I7RCEDajijlK27jLA3X3iWuopsSKoHS0NmHHMD1HtHbCE24MrMujn8ThLh1SPPK32wx5+t57/aMzCb+K7Ry+C9d+DczGYLGYhON9JHpJtmYpeZAv0rnNoeqwXXwAhCJ3Kw7SxN2N+KTBKGIutTRVYAsr9l7aq3ot39UISRCN+Npq2PxDILrdBe7LcrTpqdP4TGbD7QIdGZeiHVj0idFYE6eiEIHpGtvBhUVS1SwIBHRc6EdyxC+++AU/pie4U6nxWEJIrvfbH0cViTWp3I5tFBYEare+nplZ7UAFRrd1/G0ISMiEctraO+yYjZ5sSKzqKTLZhZiLVGDWtYrmI16xCErEq22xWu4QcKahB3kElz6XLHwtG6EJAIQhAJYm6O3RTr5nNlroC2hNqqXzGwvoSHPpEISYiu21OU2sKrigq80DZS1wTYAEkdVzc+mEIRtL//Z'
    st.image(image=image,
             use_column_width ='always')
    col1, col2 = st.columns(2)
    with col1:
        url = "https://www.linkedin.com/in/celine-anselmo/"
        st.markdown("[Céline ANSELMO ](%s)" % url)
        url = "https://www.linkedin.com/in/karine-minatchy-6644a5136/"
        st.markdown("[Karine Minatchy ](%s)" % url)
        url = "https://www.linkedin.com/in/dorian-rivet/"
        st.markdown("[Dorian RIVET](%s)" % url)
        url = "https://www.linkedin.com/in/cindysotton/?originalSubdomain=fr"
        st.markdown("[Cindy Sotton](%s)" % url)      
if selected == "Contexte":
    st.title('VgChartz : Analyse des données')
    st.subheader("Ventes globales par zones géograpiques")
    df_areas = df[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]
    df_areas = df_areas.sum().reset_index()
    df_areas = df_areas.rename(columns={"index": "Areas", 0: "Sales"})
    labels = df_areas['Areas']
    sizes = df_areas['Sales']
    colors = ['darkviolet', 'royalblue', 'hotpink', 'aqua']
    fig = px.pie(df_areas,
             values=sizes,
             names=labels,
             color_discrete_sequence=colors
    )
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    st.markdown(
        """Nos ventes se concentrent sur trois principaux marchés : North America, Europe, Japon (≈90%). Les ventes sur d'autres marchés sont inférieures à 10%. A noter la concentration particulière d'une part avec : \n North Amercia qui réalise près de la moitié des ventes. \n Le Japon qui réalise plus de 10% des ventes à mettre en perspecitive avec le nombre d'habitants.""")
    st.markdown('## Anlyse des données')
    st.markdown("Estimer les ventes d'un produit avant de le lancer est une étape essentielle dans la vie d'un produit. C'est ce que nous allons essayer de faire dans le cadre de ce projet.  \n Notre étude nous portera dans l'univers du jeu vidéo.")
    st.write(df.drop(columns=['Year']).describe())
    st.subheader("Ventes globales par jeux")
    source = ColumnDataSource(df)
    hover = HoverTool(
       tooltips=[
           ("name", "@Name"),
           ("Genre", "@Genre"),
           ("Platform", "@Platform"),
           ("Studio", "@Studio"),
           ("Note", '@Critic_Score') ])
    p98 = figure(plot_width=800, plot_height=700,x_axis_label='Year', y_axis_label='Global_Sales')
    doc = curdoc()
    doc.theme = 'dark_minimal'
    doc.add_root(p98)
    p98.circle(x='Year',y='Global_Sales',source = source,color='darkviolet',size=10)
    p98.add_tools(hover)
    st.bokeh_chart(p98, use_container_width=True)
    st.subheader("Ventes globales par années")
    data_NA = df.groupby(by=['Year'])['NA_Sales'].sum()
    data_NA = data_NA.reset_index()
    data_EU = df.groupby(by=['Year'])['EU_Sales'].sum()
    data_EU = data_EU.reset_index()
    data_JP = df.groupby(by=['Year'])['JP_Sales'].sum()
    data_JP = data_JP.reset_index()
    data_Others = df.groupby(by=['Year'])['Other_Sales'].sum()
    data_Others = data_Others.reset_index()
    data_globales = df.groupby(by=['Year']).sum()
    data_globales = data_globales.reset_index()
    p1 = figure(plot_width = 600, plot_height = 400,x_axis_label='Year', y_axis_label='Sales',y_range=[0,600]) 
    p2 = figure(plot_width = 600, plot_height = 400,x_axis_label='Year', y_axis_label='Sales',y_range=[0,600]) 
    p3 = figure(plot_width = 600, plot_height = 400,x_axis_label='Year', y_axis_label='Sales',y_range=[0,600]) 
    p4 = figure(plot_width = 600, plot_height = 400,x_axis_label='Year', y_axis_label='Sales',y_range=[0,600]) 
    p5 = figure(plot_width = 600, plot_height = 400,x_axis_label='Year', y_axis_label='Sales',y_range=[0,600]) 
    source1 = ColumnDataSource(data_NA)
    source2 = ColumnDataSource(data_EU)
    source3 = ColumnDataSource(data_JP)
    source4 = ColumnDataSource(data_Others)
    source5= ColumnDataSource(data_globales)
    p1.line(x = "Year",
        y = "NA_Sales",
        line_width = 3,
        color = "darkviolet",
        source = source1)
    p5.line(x = "Year",
        y = "NA_Sales",
        line_width = 3,
        color = "darkviolet",
        source = source1,
       legend_label="NA_Sales")
    p2.line(x = "Year",
        y = "EU_Sales",
        line_width = 3,
        color = "royalblue",
        source = source2)
    p5.line(x = "Year",
        y = "EU_Sales",
        line_width = 3,
        color = "royalblue",
        source = source2,
       legend_label="EU_Sales")
    p3.line(x = "Year",
       y = "JP_Sales",
       line_width = 3,
       color = "hotpink",
       source = source3)
    p5.line(x = "Year",
       y = "JP_Sales",
       line_width = 3,
       color = "hotpink",
       source = source3,
       legend_label="JP_Sales")
    p4.line(x = "Year",
       y = "Other_Sales",
       line_width = 3,
       color = "aqua",
       source = source4)
    p5.line(x = "Year",
       y = "Other_Sales",
       line_width = 3,
       color = "aqua",
       source = source4,
       legend_label="Other_Sales")
    p5.line(x = "Year",
       y='Global_Sales',
       line_width = 3,
       color = "gray",
       source = source5,
       legend_label="Global_Sales")
    labels = LabelSet(x='weight', y='height', text='names',
              x_offset=5, y_offset=5, render_mode='canvas')
    p5.add_layout(labels)
    tab1 = Panel(child = p1,
            title = "NA_Sales")
    tab2 = Panel(child = p2,
            title = "EU_Sales")
    tab3 = Panel(child = p3,
            title = "JP_Sales")
    tab4 = Panel(child = p4,
            title = "Others_Sales")
    tab5 = Panel(child = p5,
            title = "Globales")
    tabs = Tabs(tabs = [tab1, tab2, tab3, tab4, tab5])
    doc = curdoc()
    doc.theme = 'dark_minimal'
    doc.add_root(tabs)
    st.bokeh_chart(tabs, use_container_width=True)
    st.markdown(
        """Le marché du jeu vidéo a commencé sa croissance à partir de la seconde moitié des années 90 dynamisé par le lancement de nouvelles plateformes: \n\n Sortie de la PlayStation en 1995 \n\n Nouvel élan dans les années 2000 avec la sortie de la Nintendo 64. \n\nAprès une forte croissance (2005 à 2010), le marché revient à sa tendance initiale. """)   
if selected == "Méthodologie":
    tab1, tab2, statistiques = st.tabs(["Extraction des données", "Data processing",'Statistiques'])
    with tab1:
        st.markdown('Nous avons scrappé le site Vgchart pour récupérer :  \n - Des données plus récentes  \n - Variables Critic Score et Studio en plus')
        st.markdown ('Voici notre nouveau dataset:') 
        st.write(df.head())
    with tab2:
        st.markdown("1 - Nettoyage de données:  \n- Formater le type des varibles si nécessaire (str,int,date)    \n- Remplacement et ou suppression des Nans      \n  \n   \n 2 - Transformation des données:  \n- Supprimer les outliers des variables explicatives  \n- Encodage de la variable plateforme  \n    - Application d'un get dummies sur la variable plateforme \n    - Reverse du get dummies pour obtenir un résultat tel que Multi-Plateforme ou le nom de la plateforme  \n- Pivot du dataset pour obtention d'une ligne par jeu") 
        st.divider()  
        st.markdown("Nous avons choisi d'appliquer des modèles de régression pour prédire notre variable en quantité (régression linéaire, arbre de décision, forêt aléatoire). ")
    with statistiques:
        #Analyses statistiques

        #Global_Sales
        st.markdown('### Corrélation avec la variable cible: Global_Sales:')
        st.markdown("""<style>.big-font {font-size:15px !important;}</style>""", unsafe_allow_html=True)
        st.markdown("<p class='big-font'>L'analyse de la variance ANOVA a été utilisée pour mettre en relation nos différentes variables explicatives: Platform, Genre, Studio, Publisher et notre variable cible Global_Sales.</p>", unsafe_allow_html=True)

        #image stats1
        image_path = "stats1.png"
        try:
            image = Image.open(image_path)
            st.image(image, caption="Anova")
        except:
            st.error("Impossible d'afficher l'image")

        st.markdown("""<style>.big-font {font-size:15px !important;}</style>""", unsafe_allow_html=True)
        st.markdown("<p class='big-font'>L'objectif du test est de comparer les moyennes des deux échantillons et de conclure sur l'influence d'une variable explicative catégorielle sur la loi dune variable continue à expliquer. Lorsque la p-value (PR(>F)) est inférieur à 5%, on rejette l'hypothèse selon laquelle la variable n'influe pas sur Global_Sales.</p>", unsafe_allow_html=True)
        st.markdown("""<style>.big-font {font-size:15px !important;}</style>""", unsafe_allow_html=True)
        st.markdown("<p class='big-font'>Nous notons que les autres variables explicatives influent sur la valeur cible. Nous procèderons donc à une analyse complémentaire pour identifier le poids des variables lors de la modélisation.</p>", unsafe_allow_html=True)

        #variables explicatives
        st.markdown('### Corrélation entre les variables explicatives:')
        st.markdown("""<style>.big-font {font-size:15px !important;}</style>""", unsafe_allow_html=True)
        st.markdown("<p class='big-font'>Nous avons utilisé la méthode statistique V de Cramer pour mesurer le niveau de corrélation entre nos variables explicatives de type qualitatives.</p>", unsafe_allow_html=True)
        
        #image stats2
        image_path = "stats2.png"
        try:
            image = Image.open(image_path)
            st.image(image, caption="V de Cramer")
        except:
            st.error("Impossible d'afficher l'image")

        st.markdown("""<style>.big-font {font-size:15px !important;}</style>""", unsafe_allow_html=True)
        st.markdown("<p class='big-font'>Cette analyse a permis de mettre en valeur des relations fortes comme le genre et le publisher et de mettre de côté des analyses comme platform et genre où il n'y a pas de corrélation.</p>", unsafe_allow_html=True)
if selected == "Modélisation":
    tab1, tab2, tab3 = st.tabs(["Introduction", "Etapes de la Modélisation", 'Résultats'])
    with tab1:
        st.markdown('### Dataset Modélisation:')
        st.text('Variable Cible')
        st.markdown('-  Global_Sales')
         
        
        st.text('Variables Explicatives')
            

        st.markdown("-  L'année de sortie (Year)  \n-  Le genre (Genre)  \n-  Le studio l’ayant développé (Studio)  \n-  L’éditeur l’ayant publié (Publisher)  \n-  La plateforme sur laquelle le jeu est sortie (Platform)  \n-  Les notes utilisateurs (Critic_Score)")  
        st.markdown("Nous avons choisi d'appliquer des modèles de régression pour prédire notre variable en quantité (régression linéaire, arbre de décision, forêt aléatoire). ")
        
    with tab2:
        #Préprocessing Etapes
        st.markdown('### Pre-processing:')
        st.markdown('Etape 1: Clustering des variables studio et publisher')
        st.markdown('-  Suite au nombre important de modalités dans ces trois colonnes (+700 pour studio), nous avons simplifié les variables en utilisant la méthode du clustering. Studio, Publisher: 1 à 4 suivant leur montant de Global_Sales')
        st.markdown('Etape 2: Suppression des variables non pertinentes pour la modélisation (name, region sales, description)  \n Etape 3: Encoding des variables catégorielles')
        #Modélisation Etapes
        st.markdown('### Modélisation:')
        st.markdown('Etape 1: Clustering des variables studio et publisher')
        st.markdown("-  Nous avons choisi d'appliquer des modèles de régression pour prédire notre variable en quantité (régression linéaire, arbre de décision, forêt aléatoire).")
        st.markdown("Etape 2: Analyse de l'importance des variables, Itération 2  \nEtape 3: Calcul des meilleurs hyperparamètres, pas d\'itération.")
        
    with tab3:
        st.markdown('### Résultats:')
        # Graph importance 
        v1 = ([0.33011073, 0.21852045, 0.00776532, 0.062745  , 0.0032006 ,
        0.00506196, 0.00514901, 0.01860793, 0.00530342, 0.00532782,
        0.01765334, 0.00209605, 0.01434875, 0.01864197, 0.02456184,
        0.01448861, 0.02698887, 0.00310434, 0.00044673, 0.03785596,
        0.00462839, 0.00362745, 0.09522153, 0.00242704, 0.00349292,
        0.00649888, 0.00348894, 0.00684423, 0.0031338 , 0.00686099,
        0.00162643, 0.02738152, 0.0014133 , 0.01008641, 0.00066476,
        0.00062469])

        v2 = (['Critic_Score', 'Year', 'cat_publi', 'cat_studio',
        'Genre_Action-Adventure', 'Genre_Adventure', 'Genre_Fighting',
        'Genre_Misc', 'Genre_Music', 'Genre_Party', 'Genre_Platform',
        'Genre_Puzzle', 'Genre_Racing', 'Genre_Role-Playing', 'Genre_Shooter',
        'Genre_Simulation', 'Genre_Sports', 'Genre_Strategy', 'Platform_DC',
        'Platform_DS', 'Platform_GBA', 'Platform_GC',
        'Multi_Plateforme', 'Platform_N64', 'Platform_NS',
        'Platform_PC', 'Platform_PS', 'Platform_PS2', 'Platform_PS3',
        'Platform_PS4', 'Platform_PSP', 'Platform_Wii', 'Platform_WiiU',
        'Platform_X360', 'Platform_XB', 'Platform_XOne'])

        importances = v1
        feat_importances = pd.Series(importances, index=v2)

        # Trier les variables par ordre d'importance décroissante
        feat_importances = feat_importances.sort_values(ascending=False)

        # Afficher le graphique en barres
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(feat_importances.index, feat_importances.values, color="darkviolet")
        ax.set_title("Importance de chaque variable")
        ax.set_ylabel("Importance")
        ax.tick_params(axis="x", rotation=90)
        

        # Afficher le graphique dans Streamlit
        st.pyplot(fig)
        st.markdown("""<style>.big-font {font-size:15px !important;}</style>""", unsafe_allow_html=True)
        st.markdown('Notre analyse nous indique que les variables Critic_score et Year sont celles qui ont le plus de poids.')
        
        # Graph Résultats
        x = ["Régression Logistique","Arbre de Décision","Random Forest"]
        y = [0.11884109217929484,-0.2,0.10571851988597436]
        fig, ax = plt.subplots()
        color = ['#EEA2AD','#87CEFA','#8470FF']
        ax.bar(x, y, color=color, width=0.6)
        ax.set_ylim(-0.5, 1)
        ax.grid(axis='y')
        ax.set_title("Résultats")
        ax.tick_params(axis="x", rotation=55)
        st.pyplot(fig)
        st.markdown("""<style>.big-font {font-size:15px !important;}</style>""", unsafe_allow_html=True)
        st.markdown('>Les performances des modèles ne sont pas bonnes.  \nNous ne pouvons donc prévoir les ventes !')
if selected == "Conclusion":
    st.markdown("Des difficultés ont été rencontrées pour prédire les ventes en quantité en partant de notre dataset.  \n  Les difficultés principales identifiées sont:\n- Le match de la collecte de données via scrapping  \n - Le non partiitionnement par la durée sur les ventes \n  \nLes approches que nous pourrions proposer pour pallier ces difficultés: \n- Trouver un positionnement sur le marché avec  \n     -  Point de vue du studio  \n   - Point de vue du publisher \n   - Point de vue sur une période de temps \n   - Interpréation différente entre les séries de jeux (GTA) et les one-shoot")
    st.markdown("Il faut également prendre en compte pour ces projections les éléments suivants:  \n - Analyse de sentiment pour prédire l'engouement avant la sortie des jeux (commentaires sur le trailer, nombre de vues)  \n - Analyse du marché: \n     - Concurrence: cours de la bourse sur les acteurs du jeu vidéo \n     -  Economique: pouvoir d'achat sur le marché (ex: crise économique) \n     -  Socioculturel: étude sur les comportements des consommateurs (ex: genre attendu par âge, pays) \n     - Technologique: innovations sur le marché  \n     - Saisonnalité")
if selected == "Analyse":
    genre = st.radio(
    "Variable:",
    ('Plateformes', 'Publishers', 'Studios','Genres','Notes'))
    st.divider()
    if genre == 'Notes':
        
        st.subheader("Répartition des notes")
        df['cat_Notes'] = pd.cut(df['Critic_Score'], bins = [0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10], labels=['0 à 1','1 à 2','2 à 3','3 à 4','4 à 5','5 à 6','6 à 7', '7 à 8', '8 à 9', '9 à 10'])
        # Remplacer les modalités peu nombreuse par Autre
        df['cat_Notes'] = df['cat_Notes'].replace(['2 à 3', '1 à 2', '0 à 1'],['Autre','Autre','Autre'])
        # Créer une serie pour analyser le genre
        df3 = df['cat_Notes']
        df3.str.split(',', expand=True).stack().reset_index(drop=True)
        color = ['dodgerblue','tomato','mediumaquamarine','mediumpurple','sandybrown',
                                                'lightskyblue','hotpink','palegreen','violet','gold','lavender',
                                                'salmon','aquamarine','plum','peachpuff']
        fig = px.pie(df3,
                    values=df3.value_counts(),
                    names=df3.value_counts().index,
                    color_discrete_sequence = color)
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        st.markdown("Nous avons représenté ici la répartition des notes de notre liste de jeux.Il sera intéressant d'observer par la suite si l'on peut s'appuyer sur les notes pour valider nos analyses sur les autres variables. On observe près de 75% des jeux qui ont une note comprise entre 5 et 8.")
        
    
        st.markdown("## Zoom sur les Notes")
        df['cat_Notes'] = pd.cut(df['Critic_Score'], bins = [0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10], labels=['0 à 1','1 à 2','2 à 3','3 à 4','4 à 5','5 à 6','6 à 7', '7 à 8', '8 à 9', '9 à 10'])
        # Remplacer les modalités peu nombreuse par Autre
        df['cat_Notes'] = df['cat_Notes'].replace(['2 à 3', '1 à 2', '0 à 1'],['Autre','Autre','Autre'])
        # Créer une serie pour analyser le genre
        df3 = df['cat_Notes']
        df3.str.split(',', expand=True).stack().reset_index(drop=True)
        DICT_NOTE = {'7 à 8': 'dodgerblue',
                '8 à 9': 'tomato',
                '6 à 7': 'mediumaquamarine',
                '5 à 6': 'mediumpurple',
                '4 à 5': 'sandybrown',
                '9 à 10': 'lightskyblue',
                '3 à 4': 'hotpink',
                'Autre': 'palegreen'}
        col1,col2 = st.columns(2)
        with col1:
            st.markdown("Notes top 10 modalités les plus fréquentes")
            sns.set(style="ticks", context="talk")
            plt.style.use("dark_background")
            fig = plt.figure(figsize=(20, 10))
            sns.barplot(y=df["cat_Notes"].value_counts().head(10).index,
                    x=df["cat_Notes"].value_counts().head(10).values,
                        palette = DICT_NOTE)
            st.pyplot(fig,theme="streamlit", use_container_width=True)
        with col2:
            st.markdown("Notes top 10 des ventes  \n ")
            sns.set(style="ticks", context="talk")
            plt.style.use("dark_background")
            fig = plt.figure(figsize=(20, 10))
            df_publisher = df[['cat_Notes', 'Global_Sales']]
            df_publisher = df_publisher.groupby('cat_Notes')['Global_Sales'].sum().sort_values(ascending=False).head(10)
            df_publisher = pd.DataFrame(df_publisher).reset_index()
            sns.barplot(y="cat_Notes", x="Global_Sales",data=df_publisher, palette = DICT_NOTE);
            st.pyplot(fig,theme="streamlit", use_container_width=True)
        st.markdown("On peut donc considérer que les jeux sont en règle général qualitatifs. De manière logique, les jeux les mieux notés sont ceux qui se vendent le plus.")
        st.subheader("Evolution des notes par genres et par années")
        fig, ax = plt.subplots(5, 1, sharex=True,sharey=True,
                            figsize=(10, 5))
        sns.set(style="ticks", context="talk")
        plt.style.use("dark_background")
        liste=['Misc','Action','Shooter','Adventure','Sports']
        color=['violet','tomato','mediumaquamarine','hotpink','mediumpurple']
        for index, i in enumerate(liste):
            dfsource =pd.DataFrame(df[df.Genre ==i].groupby(['Genre', 'Year']).mean()).reset_index()
            source = ColumnDataSource(dfsource)
            sns.lineplot(x = "Year",
                y = "Critic_Score",
                data=dfsource,
                color=color[index],
                label=i,
                        ax=ax[index])
            ax[index].set_ylabel('')
        st.pyplot(fig, theme="streamlit", use_container_width=True) 
        st.markdown("Nous observons sur le graphique ci-contre l'évolution de la note par année sur les genres que nous avions commenté dans la partie Genre. Les genres Misc, Action, Sports semblent afficher une continuité. Nous observons pour Shooter et Adventures des pic de décroissances qu'il pourra être intéressant d'analyser plus en détails. Il y a-t-il un jeu qui aurait été mal reçu du public sur ces pics.")
    if genre == 'Plateformes':
        st.markdown('Zoom sur les plateformes')
        st.subheader("Répartition des plateformes")
        df1 = df[df.columns[11:]]
        # Remplacer les petites valeurs par autre
        df1['Platform'] = df['Platform'].replace(['WiiU', 'PS4', 'XOne',
            'XB', 'DC'],['Autre','Autre','Autre','Autre','Autre'])
        # Remplacer les petites valeurs par autre aussi dans df
        df['Platform'] = df['Platform'].replace(['WiiU', 'PS4', 'XOne',
            'XB', 'DC'],['Autre','Autre','Autre','Autre','Autre'])
        df1 = pd.Series(df1["Platform"])
        df1.str.split(',', expand=True).stack().reset_index(drop=True)
        DICT_PLAT = {'Multi_Plateforme': 'dodgerblue',
                    'PSP': 'tomato',
                    'GBA': 'mediumaquamarine',
                    'PC': 'mediumpurple',
                    'DS': 'sandybrown',
                    'PS3': 'lightskyblue',
                    'GC': 'hotpink',
                    'PS': 'palegreen',
                    'Wii': 'violet',
                    'PS2': 'gold',
                    'Autre': 'lavender',
                    'X360': 'salmon',
                    '3DS': 'aquamarine',
                    'NS': 'plum',
                    'N64': 'peachpuff'}
        st.subheader("Ventes globales par zones géograpiques")
        color = ['dodgerblue','tomato','mediumaquamarine','mediumpurple','sandybrown',
                                        'lightskyblue','hotpink','palegreen','violet','gold','lavender',
                                        'salmon','aquamarine','plum','peachpuff']
        fig = px.pie(df1,
            values=df1.value_counts(),
            names=df1.value_counts().index,
            color_discrete_sequence = color)
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        st.markdown("Nous constatons que les parts de marché se répartissent de manière équilibré entre les plateformes.A noter que certaines plateformes tendent à disparaitre car remplacer par leur upgrade (PS2 qui devient la PS3).")

        col1,col2 = st.columns(2)
        with col1: 
            st.subheader("Plateforme ayant le plus de références vendues")
            sns.set(style="ticks", context="talk")
            plt.style.use("dark_background")
            fig = plt.figure(figsize=(20, 10))
            sns.barplot(y=df["Platform"].value_counts().head(10).index,
                    x=df["Platform"].value_counts().head(10).values, palette=DICT_PLAT);
            st.pyplot(fig,theme="streamlit", use_container_width=True)       

            st.subheader("Platforme ayant le plus de ventes")
            sns.set(style="ticks", context="talk")
            plt.style.use("dark_background")
            fig = plt.figure(figsize=(20, 10))
            df_publisher = df[['Platform', 'Global_Sales']]
            df_publisher = df_publisher.groupby('Platform')['Global_Sales'].sum().sort_values(ascending=False).head(10)
            df_publisher = pd.DataFrame(df_publisher).reset_index()
            sns.barplot(y="Platform", x="Global_Sales",palette=DICT_PLAT,data=df_publisher);
            st.pyplot(fig,theme="streamlit", use_container_width=True)
        st.markdown("Les jeux vendus sur plusieurs plateformes ont le meilleur ratio de vente par jeu. PSP et GBA sont les deux premiers en terme de quantités de jeux mais perdent leur place lorsque l'on regarde le nombre de vente total et médian. A part pour la PS3, l'ensemble des autres plateforme à le même nombre de vente médian. Néanmoins, à restituer avec le nombre de plateformes en exploitation à ce moment.  \n Certaines plateformes vont avoir des jeux qui auront des ventes disproportionnées par rapport au reste de leur catalogue d'où un écart important entre la moyenne et la médiane des ventes (ex : Wii). Le graphique suivant nous permettra de mieux observer  ")
        with col2: 
            st.subheader("Nombre de ventes median par plateforme")
            sns.set(style="ticks", context="talk")
            plt.style.use("dark_background")
            fig = plt.figure(figsize=(20, 10))
            df_publisher = df[['Platform', 'Global_Sales']]
            df_publisher = df_publisher.groupby('Platform')['Global_Sales'].median().sort_values(ascending=False).head(10)
            df_publisher = pd.DataFrame(df_publisher).reset_index()
            sns.barplot(y="Platform", x="Global_Sales",palette=DICT_PLAT,data=df_publisher);
            st.pyplot(fig,theme="streamlit", use_container_width=True)

            st.subheader("Nombre de ventes moyen par plateforme")
            fig = plt.figure(figsize=(20, 10))
            df_publisher = df[['Platform', 'Global_Sales']]
            df_publisher = df_publisher.groupby('Platform')['Global_Sales'].mean().sort_values(ascending=False)
            df_publisher = pd.DataFrame(df_publisher).reset_index()
            sns.barplot(y="Platform", x="Global_Sales",palette=DICT_PLAT,data=df_publisher);
            st.pyplot(fig,theme="streamlit", use_container_width=True)
        st.subheader("Analyse des valeurs extrêmes pour les plateformes")
        sns.set(style="ticks", context="talk")
        plt.style.use("dark_background")
        fig = plt.figure(figsize=(20, 10))
        
        df1 = df[df.columns[11:]]
        # Remplacer les petites valeurs par autre
        df1['Platform'] = df['Platform'].replace(['WiiU', 'PS4', 'XOne',
        'XB', 'DC'],['Autre','Autre','Autre','Autre','Autre'])
        # Remplacer les petites valeurs par autre aussi dans df
        df['Platform'] = df['Platform'].replace(['WiiU', 'PS4', 'XOne',
        'XB', 'DC'],['Autre','Autre','Autre','Autre','Autre'])
        df1 = pd.Series(df1["Platform"])
        df1.str.split(',', expand=True).stack().reset_index(drop=True)
            # Dictionnaire des couleurs par modalités pour retrouver les mêmes sur l'ensemble des graphiques
        DICT_PLAT = {'Multi_Plateforme': 'dodgerblue',
        'PSP': 'tomato',
        'GBA': 'mediumaquamarine',
        'PC': 'mediumpurple',
        'DS': 'sandybrown',
        'PS3': 'lightskyblue',
        'GC': 'hotpink',
        'PS': 'palegreen',
        'Wii': 'violet',
        'PS2': 'gold',
        'Autre': 'lavender',
        'X360': 'salmon',
        '3DS': 'aquamarine',
        'NS': 'plum',
        'N64': 'peachpuff'}
        sns.set_theme(style = 'darkgrid')
        plt.style.use('dark_background')
        bx=sns.boxplot(x='Platform',
                    y='Global_Sales',
                    palette = DICT_PLAT,
                    flierprops=flierprops,
                    data=df[df.Platform.isin(list(df.Platform.value_counts().index))])
        bx.set_xticklabels(bx.get_xticklabels(),rotation=75)
        sns.set(style="ticks", context="talk")
        plt.style.use("dark_background")
        fig = bx.get_figure()
        sns.set(style="ticks", context="talk")
        plt.style.use("dark_background")
        ax = bx.axes
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        st.pyplot(fig, use_container_width=True)
        st.markdown("Cette représentation graphique met en évidence le constat effectué précédemment à savoir que la plateforme Wii à un outlier. Il s'agit de Wii Sport qui fait des records de ventes par rapport aux autres jeux de Wii. Nos recherches nous ont indiqué que ce jeu est sorti en 2006 en même temps que la console Wii ce qui a participé à l'engouement et l'explosion des ventes. Le jeu faisait parti d'une offre bundle avec la Wii. La DS a également des valeurs extrêmes qu'il sera intéressant de regarder avec New Super Mario.")
        st.subheader("Analyse de la corrélation de la variable Platform")
        comp_platform = df[['Platform', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales',"Global_Sales"]]
        # comp_genre
        comp_map = comp_platform.groupby(by=['Platform']).sum().sort_values(by="Global_Sales",ascending=False)
            # comp_map
        plt.figure(figsize=(15, 10))
        sns.set(font_scale=1)
        ht = sns.heatmap(comp_map, annot = True, cmap ="cool", fmt = '.1f')
        fig2 = ht.get_figure()
        ax = ht.axes
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        st.pyplot(fig2,facecolor='black', use_container_width=True)

        
    if genre == "Publishers":
        st.markdown("Zoom sur les Publishers")
        st.markdown("Avant propos :  \n Nous avons plus de 200 modalités pour cette variable.Il sera intéressant d'observer par la suite les évolutions au cours du temps et les différences par pays.Les publishers les plus importants en terme de production de jeux sont Nintendo, Sony CE, Ubisoft ou encore Electronics Arts.")
        DICT_PUBLISHER = {'Nintendo' : 'dodgerblue','Sony Computer Entertainment':'tomato','Ubisoft':'mediumaquamarine','Electronic Arts':'mediumpurple',
            'Sega':'sandybrown','Konami':'lightskyblue','Activision':'hotpink','THQ':'palegreen','Capcom':'violet','Atlus':'gold',"Rockstar Games":'lavender',
            'Mojang':'salmon','RedOctane':'aquamarine','EA Sports' : 'darkseagreen','Microsoft Game Studios':'moccasin','Sony Computer Entertainment America':'rosybrown','Broderbund':'blue',
            'MTV Games':'plum','ASC Games':"turquoise",'Valve':'indianred','Hello Games':'peachpuff','Microsoft Studios':'lemonchiffon','Valve Corporation':'lightcoral',
            'Bethesda Softworks':'paleturquoise','LucasArts':'steelblue','Virgin Interactive':'chocolate','Sony Interactive Entertainment':'mediumorchid',
            'Blizzard Entertainment':'yellowgreen','City Interactive':'slategrey','Rare':'cornsilk','Square':'cadetblue','Warner Bros. Interactive':'pink'}
        col1,col2 = st.columns(2)
        with col1:
            st.subheader("Publisher top 10 modalités les plus fréquentes")
            sns.set(style="ticks", context="talk")
            plt.style.use("dark_background")
            fig = plt.figure(figsize=(20, 10))
            sns.barplot(y=df["Publisher"].value_counts().head(10).index,
                x=df["Publisher"].value_counts().head(10).values, palette = DICT_PUBLISHER );
            st.pyplot(fig,theme="streamlit", use_container_width=True)

            
            st.subheader("Nombre de ventes median par Publisher")
            sns.set(style="ticks", context="talk")
            plt.style.use("dark_background")
            fig = plt.figure(figsize=(20, 10))
            df_publisher = df[['Publisher', 'Global_Sales']]
            df_publisher = df_publisher.groupby('Publisher')['Global_Sales'].median().sort_values(ascending=False).head(10)
            df_publisher = pd.DataFrame(df_publisher).reset_index()
            sns.barplot(y="Publisher", x="Global_Sales",palette = DICT_PUBLISHER,data=df_publisher);
            st.pyplot(fig,theme="streamlit", use_container_width=True)
        st.markdown("Le top 4 est globalement constant en terme de quantité de jeux vendus et de production de jeux. Mais les 4 publishers perdent leur place lorsque l'on regarde le ratio ventes/quantités median. Nintendo se démarque en terme de référence sorties et de ventes. Mojang et RedOctane ont quand à eux un très bon ratio de vente médian alors qu'ils sont absents des deux premiers graphiques. On peut constater que les deux premiers Publisher ne sont pas sensibles à des valeurs extrêmes.En revanche Rockstar G, Sony IE, Bethesda S ou encore Nitendo ont quand à eux des jeux qui auront eu des ventes peu représentatives des ventes médianes de leur catalogue. ")
        with col2:
            st.subheader("Publisher top 10 des ventes")
            sns.set(style="ticks", context="talk")
            plt.style.use("dark_background")
            fig = plt.figure(figsize=(20, 10))
            df_publisher = df[['Publisher', 'Global_Sales']]
            df_publisher = df_publisher.groupby('Publisher')['Global_Sales'].sum().sort_values(ascending=False).head(10)
            df_publisher = pd.DataFrame(df_publisher).reset_index()
            sns.barplot(y="Publisher", x="Global_Sales",palette = DICT_PUBLISHER,data=df_publisher);
            st.pyplot(fig,theme="streamlit", use_container_width=True)


        

            st.subheader("Nombre de ventes moyen par Publisher")
            sns.set(style="ticks", context="talk")
            plt.style.use("dark_background")
            fig = plt.figure(figsize=(20, 10))
            df_publisher = df[['Publisher', 'Global_Sales']]
            df_publisher = df_publisher.groupby('Publisher')['Global_Sales'].mean().sort_values(ascending=False).head(20)
            df_publisher = pd.DataFrame(df_publisher).reset_index()
            sns.barplot(y="Publisher", x="Global_Sales",palette = DICT_PUBLISHER,data=df_publisher);
            st.pyplot(fig,theme="streamlit", use_container_width=True)
        st.subheader("Analyse des valeurs extrêmes pour les publisher")
        plt.figure(figsize=(15, 10))
        sns.set(style="dark", context="talk")
        plt.style.use("dark_background")
        df5= df.loc[(df['Publisher'] == 'Mojang') 
        | (df['Publisher'] =='RedOctane')
        | (df['Publisher'] =='Rockstar Games')
        | (df['Publisher'] =='Sony Interactive Entertainment')
        | (df['Publisher'] =='Bethesda Softworks')
        | (df['Publisher'] =='Nintendo')]
        # création du dico
        DICT_PUBLISHER = {'Nintendo' : 'dodgerblue','Sony Computer Entertainment':'tomato','Ubisoft':'mediumaquamarine','Electronic Arts':'mediumpurple',
                    'Sega':'sandybrown','Konami':'lightskyblue','Activision':'hotpink','THQ':'palegreen','Capcom':'violet','Atlus':'gold',"Rockstar Games":'lavender',
                    'Mojang':'salmon','RedOctane':'aquamarine','EA Sports' : 'darkseagreen','Microsoft Game Studios':'moccasin','Sony Computer Entertainment America':'rosybrown','Broderbund':'blue',
                    'MTV Games':'plum','ASC Games':"turquoise",'Valve':'indianred','Hello Games':'peachpuff','Microsoft Studios':'lemonchiffon','Valve Corporation':'lightcoral',
                    'Bethesda Softworks':'paleturquoise','LucasArts':'steelblue','Virgin Interactive':'chocolate','Sony Interactive Entertainment':'mediumorchid',
                    'Blizzard Entertainment':'yellowgreen','City Interactive':'slategrey','Rare':'cornsilk','Square':'cadetblue','Warner Bros. Interactive':'pink'}
        sns.set_theme(style = 'darkgrid')
        plt.style.use('dark_background')
        bx=sns.boxplot(x='Publisher',
                y='Global_Sales',
                palette = DICT_PUBLISHER,
                flierprops=flierprops,
                data=df5)
        bx.set_xticklabels(bx.get_xticklabels(),rotation=75)
        sns.set(style="ticks", context="talk")
        plt.style.use("dark_background")
        fig3 = bx.get_figure()
        sns.set(style="ticks", context="talk")
        plt.style.use("dark_background")
        ax = bx.axes
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        st.pyplot(fig3, use_container_width=True)
        st.markdown("Analyse des outliers :  \n Nintendo : Wii Sport sorti en 2006 la saga Mario  \n Rockstar Games : La saga GTA (dont le V sorti en 2014)  \n Bethseda Softworks :The Elder Scrolls sorti en 2011 Fallout 4 sorti en 2015.  \n On remarque pour Sony que celui ci n'est pas concerné par des valeurs extrêmes. Cependant en regardant la distribution de ces modalités, nous constatons que 50% de celles ci sont en dessous de 1,6M de vente et que sa valeur max. est de 10,33M. D'ou un écart entre la moyenne et la médiane.")
        st.subheader("Analyse de la corrélation de la variable Publisher")
        comp_platform = df[['Publisher', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', "Global_Sales"]]
        # comp_genre
        comp_map = comp_platform.groupby(by=['Publisher']).sum().sort_values(by = "Global_Sales", ascending=False).head(10)
        # comp_map
        plt.figure(figsize=(15, 10))
        sns.set(font_scale=1)
        ht = sns.heatmap(comp_map, annot = True, cmap ="cool", fmt = '.1f')
        fig2 = ht.get_figure()
        ax = ht.axes
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        st.pyplot(fig2,facecolor='black', use_container_width=True)
        
    if genre =="Genres":
        curdoc().theme = 'dark_minimal'
        liste=['Misc','Action','Shooter','Adventure','Sports']
        color=['violet','tomato','mediumaquamarine','hotpink','mediumpurple']
        df_bokeh = df.Year>1992
        p = figure(plot_width = 1000, plot_height = 600,x_axis_label='Year', y_axis_label='Genre')
        for index, i in enumerate(liste):
                dfsource =pd.DataFrame(df[df.Genre ==i].groupby(['Genre', 'Year']).count()).reset_index()
                source = ColumnDataSource(dfsource)
                p.line(x = "Year",
                    y = "Name",
                    line_width = 3,
                    color=color[index],
                    source = source,
                    legend_label=i)
                doc = curdoc()
                doc.theme = 'dark_minimal'
                doc.add_root(p)
                p.legend.click_policy="mute"
        st.bokeh_chart(p, use_container_width=True)
        st.markdown(
        "Role-Playing, Action, Shooter, Sports représentent la moitié des parts de marché.")
        st.subheader("Répartition des genres")
        # Remplacer les modalités peu nombreuse par Autre
        df['Genre'] = df['Genre'].replace(['Music', 'Party','Action-Adventure'],['Autre','Autre','Autre'])
        # Créer une serie pour analyser le genre
        df2 = df['Genre']
        df2.str.split(',', expand=True).stack().reset_index(drop=True)
        color = ['dodgerblue','tomato','mediumaquamarine','mediumpurple','sandybrown',
                                                'lightskyblue','hotpink','palegreen','violet','gold','lavender',
                                                'salmon','aquamarine','plum','peachpuff']
        fig = px.pie(df2,
                    values=df2.value_counts(),
                    names=df2.value_counts().index,
                    color_discrete_sequence = color)
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        
        
        
        
        st.markdown("Zoom sur les Genre")
        # Remplacer les modalités peu nombreuse par Autre
        df['Genre'] = df['Genre'].replace(['Music', 'Party','Action-Adventure'],['Autre','Autre','Autre'])
        # Créer une serie pour analyser le genre
        df2 = df['Genre']
        df2.str.split(',', expand=True).stack().reset_index(drop=True)
        DICT_GENRE = {'Role-Playing': 'dodgerblue',
                        'Action': 'tomato',
                        'Shooter': 'mediumaquamarine',
                        'Sports': 'mediumpurple',
                        'Platform': 'sandybrown',
                        'Racing': 'lightskyblue',
                        'Adventure': 'hotpink',
                        'Fighting': 'palegreen',
                        'Misc': 'violet',
                        'Strategy': 'gold',
                        'Simulation': 'lavender',
                        'Puzzle': 'salmon',
                        'Autre': 'aquamarine'}
        col1,col2 = st.columns(2)
        with col1:
            st.subheader("Genre ayant le plus de références vendues")
            sns.set(style="dark", context="talk")
            plt.style.use("dark_background")
            fig = plt.figure(figsize=(20, 10))
            sns.barplot(y=df["Genre"].value_counts().head(10).index,
                    x=df["Genre"].value_counts().head(10).values, palette =DICT_GENRE );
            st.pyplot(fig,theme="streamlit", use_container_width=True)
            st.subheader("Genre ayant le plus de ventes")
            sns.set(style="ticks", context="talk")
            plt.style.use("dark_background")
            fig = plt.figure(figsize=(20, 10))
            df_publisher = df[['Genre', 'Global_Sales']]
            df_publisher = df_publisher.groupby('Genre')['Global_Sales'].sum().sort_values(ascending=False).head(10)
            df_publisher = pd.DataFrame(df_publisher).reset_index()
            sns.barplot(y="Genre", x="Global_Sales",palette =DICT_GENRE,data=df_publisher);
            st.pyplot(fig,theme="streamlit", use_container_width=True)
        st.markdown("Le top 6 reste le même, même si les genres s'interchange. Les genres de \"niche\" à savoir l'étiquette Autre, sont les genres les plus \"rentables\". L'écart entre les ventes médianes et moyennes est considérable, nous pouvons donc affirmer que cette variable est très sujette à des outliers.")
        with col2:
            st.subheader("Nombre de ventes median par Genre")
            sns.set(style="dark", context="talk")
            plt.style.use("dark_background")
            fig = plt.figure(figsize=(20, 10))
            df_publisher = df[['Genre', 'Global_Sales']]
            df_publisher = df_publisher.groupby('Genre')['Global_Sales'].median().sort_values(ascending=False).head(10)
            df_publisher = pd.DataFrame(df_publisher).reset_index()
            sns.barplot(y="Genre", x="Global_Sales",palette =DICT_GENRE,data=df_publisher)
            st.pyplot(fig,theme="streamlit", use_container_width=True)
            st.subheader("Nombre de ventes moyen par Genre")
            sns.set(style="dark", context="talk")
            plt.style.use("dark_background")
            fig = plt.figure(figsize=(20, 10))
            df_publisher = df[['Genre', 'Global_Sales']]
            df_publisher = df_publisher.groupby('Genre')['Global_Sales'].mean().sort_values(ascending=False)
            df_publisher = pd.DataFrame(df_publisher).reset_index()
            sns.barplot(y="Genre", x="Global_Sales",palette =DICT_GENRE,data=df_publisher);
            st.pyplot(fig,theme="streamlit", use_container_width=True)
        st.subheader("Analyse de la corrélation de la variable Genre")
        comp_genre = df[['Genre', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]
        comp_map = comp_genre.groupby(by=['Genre']).sum()
        plt.figure(figsize=(20, 10))
        sns.set(font_scale=1)
        ht = sns.heatmap(comp_map, annot = True, cmap ="cool", fmt = '.1f')
        fig2 = ht.get_figure()
        ax = ht.axes
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        st.pyplot(fig2,facecolor='black', use_container_width=True) 
        

        st.subheader("Analyse des valeurs extrêmes pour les genre")
        # Remplacer les modalités peu nombreuse par Autre
        fig = plt.figure(figsize=(20, 10))
        sns.set(style="ticks", context="talk")
        plt.style.use("dark_background")
        df['Genre'] = df['Genre'].replace(['Music', 'Party','Action-Adventure'],['Autre','Autre','Autre'])
        # Créer une serie pour analyser le genre
        df2 = df['Genre']
        df2.str.split(',', expand=True).stack().reset_index(drop=True)
        # création d'un dictionnaire pour avoir les même couleurs
        DICT_GENRE = {'Role-Playing': 'dodgerblue',
        'Action': 'tomato',
        'Shooter': 'mediumaquamarine',
        'Sports': 'mediumpurple',
        'Platform': 'sandybrown',
        'Racing': 'lightskyblue',
        'Adventure': 'hotpink',
        'Fighting': 'palegreen',
        'Misc': 'violet',
        'Strategy': 'gold',
        'Simulation': 'lavender',
        'Puzzle': 'salmon',
        'Autre': 'aquamarine'}
        bx=sns.boxplot(x='Genre',
                y='Global_Sales',
                palette = DICT_GENRE,
                flierprops=flierprops,
                data=df[df.Genre.isin(list(df.Genre.value_counts().index))])
        bx.set_xticklabels(bx.get_xticklabels(),rotation=75)
        sns.set(style="dark")
        plt.style.use("dark_background")
        fig = bx.get_figure()
        ax = bx.axes
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        st.pyplot(fig, use_container_width=True)        
    if genre == "Studios":
        st.markdown("### Zoom sur les Publishers")
        st.markdown("Avant propos :  \n Cette variable possède plus de 1000 modalités. Pour pouvoir l'analyser, nous avons sélectionné des top selon la fréquence et les ventes.")
        DICT_STUDIO = {'Capcom': 'dodgerblue',
                        'Konami': 'tomato',
                        'Nintendo EAD': 'mediumaquamarine',
                        'EA Canada': 'mediumpurple',
                        'Square Enix': 'sandybrown',
                        'Ubisoft Montreal': 'lightskyblue',
                        'EA Tiburon': 'hotpink',
                        'Namco': 'palegreen',
                        'Ubisoft ': 'violet',
                        'Sonic Team': 'gold',
                        'Hudson Soft': 'lavender',
                        'Rare Ltd.': 'salmon',
                        'Atlus Co.': 'aquamarine',
                            'Ubisoft': "orchid",
                        'Game Freak': "plum",
                        'Rockstar North': "lavender",
                        'Infinity Ward':"magenta",
                        "Traveller's Tales":"blue",
                        'Treyarch':"mediumpurple",
                        'Good Science Studio':"turquoise",
                        'Nintendo SDD': "slateblue",
                        'Sledgehammer Games': "lightcoral",
                        'Dice': "peachpuff",
                        'Neversoft': "lemonchiffon",
                        'Nintendo EAD / Retro Studios': "mintcream",
                        'Rockstar Games': "powderblue",
                        'EA DICE': "navy",
                        'Bethesda Game Studios':"palegreen",
                        'Polyphony Digital': "blueviolet",
                        '4J Studios': "bisque",
                        'Nintendo EAD Tokyo':"azure",
                        'Bungie Studios': "steeblue",
                        'Project Sora':"chocolate",
                        'Naughty Dog': "mediumorchid",
                        'Team Bondi': "lightcyan",
                        'Level 5 / Armor Project': "tomato",
                        "Bungie":"deeppink"
            }
        col1,col2 = st.columns(2)
        with col1:
            st.subheader("Studio top 10 modalités les plus fréquentes")
            sns.set(style="ticks", context="talk")
            plt.style.use("dark_background")
            fig = plt.figure(figsize=(20, 10))
            sns.barplot(y=df["Studio"].value_counts().head(10).index,
                        x=df["Studio"].value_counts().head(10).values, palette = DICT_STUDIO );
            st.pyplot(fig,theme="streamlit", use_container_width=True)

            st.subheader("Studio top 10 des ventes")
            sns.set(style="ticks", context="talk")
            plt.style.use("dark_background")
            fig = plt.figure(figsize=(20, 10))
            df_studio = df[['Studio', 'Global_Sales']]
            df_studio = df_studio.groupby('Studio')['Global_Sales'].sum().sort_values(ascending=False).head(10)
            df_studio = pd.DataFrame(df_studio).reset_index()
            sns.barplot(y="Studio", x="Global_Sales",palette = DICT_STUDIO,data=df_studio);
            st.pyplot(fig,theme="streamlit", use_container_width=True)
        st.markdown("Nous remarquons que les répartitions sur les Studio n'est pas la même que sur les variables précédentes. En revanche le constat sur l'analyse du nombre de vente médian reste le même. On remarque que les échelles entre la médiane et la moyenne sont équivalentes.Infinity Ward perd 5M entre les 2 indicateurs et les 2 derniers Studios ne sont pas les mêmes. Nous pouvons en conclure que cette variable n'est pas très sensible aux valeurs extrêmes pour ces Studios. ")
        with col2:
            st.subheader("Nombre de ventes median par Studio")
            sns.set(style="ticks", context="talk")
            plt.style.use("dark_background")
            fig = plt.figure(figsize=(20, 11))
            df_studio = df[['Studio', 'Global_Sales']]
            df_studio = df_studio.groupby('Studio')['Global_Sales'].median().sort_values(ascending=False).head(10)
            df_studio = pd.DataFrame(df_studio).reset_index()
            sns.barplot(y="Studio", x="Global_Sales",palette = DICT_STUDIO,data=df_studio);
            st.pyplot(fig,theme="streamlit", use_container_width=True)
        
            st.subheader("Nombre de ventes moyen par Studio")
            sns.set(style="ticks", context="talk")
            plt.style.use("dark_background")
            fig = plt.figure(figsize=(20, 11))
            df_studio = df[['Studio', 'Global_Sales']]
            df_studio = df_studio.groupby('Studio')['Global_Sales'].mean().sort_values(ascending=False).head(15)
            df_studio = pd.DataFrame(df_studio).reset_index()
            sns.barplot(y="Studio", x="Global_Sales",palette = DICT_STUDIO, data=df_studio);
            st.pyplot(fig,theme="streamlit", use_container_width=True)
        st.subheader("Analyse des valeurs extrêmes pour les studios")
        plt.figure(figsize=(15, 10))
        sns.set(style="dark", context="talk")
        plt.style.use("dark_background")
        df6= df.loc[(df['Studio'] == 'Infinity Ward') 
        | (df['Studio'] =='Sledgehammer Games')
        | (df['Studio'] =='Nintendo SDD')
        | (df['Studio'] =='Good Science Studio')
        | (df['Studio'] =='Nintendo EAD')
        | (df['Studio'] =='Treyarch')
        | (df['Studio'] =='4J Studios')
        | (df['Studio'] =='Nintendo EAD Tokyo')]
        # création du dico
        DICT_STUDIO = {'Capcom': 'dodgerblue',
        'Konami': 'tomato',
        'Nintendo EAD': 'mediumaquamarine',
        'EA Canada': 'mediumpurple',
        'Square Enix': 'sandybrown',
        'Ubisoft Montreal': 'lightskyblue',
        'EA Tiburon': 'hotpink',
        'Namco': 'palegreen',
        'Ubisoft ': 'violet',
        'Sonic Team': 'gold',
        'Hudson Soft': 'lavender',
        'Rare Ltd.': 'salmon',
        'Atlus Co.': 'aquamarine',
        'Ubisoft': "orchid",
        'Game Freak': "plum",
        'Rockstar North': "lavender",
        'Infinity Ward':"magenta",
        "Traveller's Tales":"blue",
        'Treyarch':"mediumpurple",
        'Good Science Studio':"turquoise",
        'Nintendo SDD': "slateblue",
        'Sledgehammer Games': "lightcoral",
        'Dice': "peachpuff",
        'Neversoft': "lemonchiffon",
        'Nintendo EAD / Retro Studios': "mintcream",
        'Rockstar Games': "powderblue",
        'EA DICE': "navy",
        'Bethesda Game Studios':"palegreen",
        'Polyphony Digital': "blueviolet",
        '4J Studios': "bisque",
        'Nintendo EAD Tokyo':"azure",
        'Bungie Studios': "steeblue",
        'Project Sora':"chocolate",
        'Naughty Dog': "mediumorchid",
        'Team Bondi': "lightcyan",
        'Level 5 / Armor Project': "tomato",
        "Bungie":"deeppink"}
        sns.set_theme(style = 'darkgrid')
        plt.style.use('dark_background')
        bx=sns.boxplot(x='Studio',
                y='Global_Sales',
                palette = DICT_STUDIO,
                flierprops=flierprops,
                data=df6)
        bx.set_xticklabels(bx.get_xticklabels(),rotation=75)
        sns.set(style="ticks", context="talk")
        plt.style.use("dark_background")
        fig4 = bx.get_figure()
        sns.set(style="ticks", context="talk")
        plt.style.use("dark_background")
        ax = bx.axes
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        st.pyplot(fig4, use_container_width=True)
        st.markdown("Cette représentation graphique confirme les constats précédents.Pour Treyarch les outliers correspondent aux jeux :  \n Call of Duty: Black Ops sorti en 2010  \n Call of Duty: World at War : 2008   \n Pour Nintendo EAD : Wii Sport sorti en 2006  \n   -   La saga Mario constitue le reste des outliers")
        st.subheader("Analyse de la corrélation de la variable Studio")
        comp_platform = df[['Studio', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', "Global_Sales"]]
        # comp_genre
        comp_map = comp_platform.groupby(by=['Studio']).sum().sort_values(by = "Global_Sales", ascending=False).head(10)
        # comp_map
        plt.figure(figsize=(15, 10))
        sns.set(font_scale=1)
        ht = sns.heatmap(comp_map, annot = True, cmap ="cool", fmt = '.1f')
        fig2 = ht.get_figure()
        ax = ht.axes
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        st.pyplot(fig2,facecolor='black', use_container_width=True)
        st.markdown("Nintendo EAD est le leader du marché. En excluant Nintendo, on constate qu'il y'a un studio qui se démarque en fonction des régions :  \n -    NA : EA Tiburon\n -    EU : EA Canada\n -    JP : Game Freak et Capcom")
                 
if selected == 'Modelisation pack':
    col1,col2 = st.columns(2)
    with col1:
        values = st.slider('Taille du test dataset',
        min_value=0.2,max_value=0.9,value=0.5)
    with col2:
        critere = st.select_slider('Critère à maximiser',
        options=['squared_error', 'friedman_mse', 'poisson'],value='friedman_mse')
    random_state = st.number_input('Random State',min_value=0,max_value=999,step=1,value=42)
        
    
    
    data = df_ml.drop('Global_Sales',axis=1)
    # isoler la variable cible
    y = df_ml['Global_Sales']
    X_train, X_test, y_train, y_test = train_test_split(data,y,test_size=values,train_size=(1-values),random_state=random_state)

# Séparation des variables catégorielles et numértique
    num_train = X_train[['Critic_Score','Year','cat_publi','cat_studio']]
    num_test = X_test[['Critic_Score','Year','cat_publi','cat_studio']]

    cat_train = X_train.drop(['Critic_Score','Year','cat_publi','cat_studio','Name'], axis=1)
    cat_test = X_test.drop(['Critic_Score','Year','cat_publi','cat_studio','Name'], axis=1)
    oneh = OneHotEncoder(drop="first", sparse=False)

    cat_train = pd.DataFrame(oneh.fit_transform(cat_train), columns = oneh.get_feature_names_out())
    cat_test = pd.DataFrame(oneh.fit_transform(cat_test), columns = oneh.get_feature_names_out())

    X_train_new = pd.concat([num_train.reset_index() ,cat_train.reset_index()], axis = 1)
    X_test_new = pd.concat([num_test.reset_index() ,cat_test.reset_index()], axis = 1)
    X_test_new.head()
    #X_test_new.shape,X_train_new.shape,
    X_test_new = X_test_new.drop(columns='index')
    X_train_new = X_train_new.drop(columns='index')
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train_new)
    X_test_scaled = scaler.transform(X_test_new)

    
    st.divider()
    param_grid = {'criterion': [critere], 'max_depth': np.arange(3, 6)}
    cl1 = LinearRegression()
    cl1.fit(X_train_scaled, y_train)
    cl2 = DecisionTreeRegressor()
    clf_gs2 = GridSearchCV(cl2, param_grid, cv=10, verbose=1)
    clf_gs2.fit(X_train_scaled, y_train)
    cl2.fit(X_train_scaled, y_train)
    cl3 = RandomForestRegressor()
    clf_gs3 = GridSearchCV(cl3, param_grid, cv=10, verbose=1)
    clf_gs3.fit(X_train_scaled, y_train)
    cl3.fit(X_train_scaled, y_train)
    x = ["Régression Logistique","Arbre de Décision","Random Forest"]
    y=[round(cl1.score(X_test_scaled, y_test),4),round(cl2.score(X_test_scaled, y_test),4),round(cl3.score(X_test_scaled, y_test),4)]
    sns.set(style="ticks", context="talk")
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(10, 5))
    sns.barplot(x=x,y=y)
    plt.grid(True)
    st.pyplot(fig)
