import streamlit as st
import plotly.express as px
import pandas as pd
import logging
import data as d
# from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
# import time
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Classification App",
    layout="wide",
    initial_sidebar_state="expanded",
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s,%(message)s")


DATA_SOURCE = './data/iris_nonID.csv'

@st.cache
def load_full_data():
    data = pd.read_csv(DATA_SOURCE)
    df = data.dropna()
    return df

@st.cache 
def load_num_data():
    data = pd.read_csv(DATA_SOURCE)
    rows = ['Species']
    data = data.drop(rows, axis=1)
    return data

# @st.cache 
# def load_filtered_data(data, genre_filter):
#     # 数値でフィルター(何点以上)
#     # filtered_data = data[data['num_rooms'].between(rooms_filter[0], rooms_filter[1])]
#     grade_filter = []
#     gender_filter = []
#     for elem in genre_filter:
#         grade_filter.append(str(elem[0:2]))
#         gender_filter.append(str(elem[2]))

#     filtered_data = data[data['学年'].isin(grade_filter)]
#     filtered_data = filtered_data[filtered_data['性別'].isin(gender_filter)]

#     return filtered_data

@st.cache
def load_ML_data(feature1, feature2, train_num = 600):
    df = load_full_data()
    # X = df.drop('Species', axis=1)  # XはSpeciesの列以外の値
    X = df[[feature1, feature2]]
    y = df.Species  # yはSpeciesの列の値

    train_num = 600
    train_X = X[:train_num]
    test_X = X[train_num:]
    train_y = y[:train_num]
    test_y = y[train_num:]
    return (train_X, test_X, train_y, test_y)


def main():
    # # If username is already initialized, don't do anything
    # if 'username' not in st.session_state or st.session_state.username == 'default':
    #     st.session_state.username = 'default'
    #     input_name()
    #     st.stop()
    if 'username' not in st.session_state:
        st.session_state.username = 'test'
            
    if 'page' not in st.session_state:
        # st.session_state.page = 'input_name' # usernameつける時こっち
        st.session_state.page = 'deal_data'


    # --- page選択ラジオボタン
    st.sidebar.markdown('## ページを選択')
    page = st.sidebar.radio('', ('分類', 'データ可視化', '決定木'))

    if page == '分類':
        st.session_state.page = 'classfy'
        logging.info(',%s,分類,%s', st.session_state.username, page)
    elif page == 'データ可視化':
        st.session_state.page = 'vis'
        logging.info(',%s,ページ選択,%s', st.session_state.username, page)
    # elif page == 'テストデータ':
    #     st.session_state.page = 'test'
    #     logging.info(',%s,ページ選択,%s', st.session_state.username, page)
    elif page == '決定木':
        st.session_state.page = 'decision_tree'
        logging.info(',%s,ページ選択,%s', st.session_state.username, page)

    # --- page振り分け
    if st.session_state.page == 'input_name':
        input_name()
    elif st.session_state.page == 'classfy':
        classfy()
    elif st.session_state.page == 'vis':
        vis()
    # elif st.session_state.page == 'test':
    #     test()  
    elif st.session_state.page == 'decision_tree':
        decision_tree()        

# ---------------- usernameの登録 ----------------------------------
def input_name():
    # Input username
    with st.form("my_form"):
        inputname = st.text_input('username', 'ユーザ名')
        submitted = st.form_submit_button("Submit")
        if submitted: # Submit buttonn 押された時に
            if inputname == 'ユーザ名' or input_name == '': # nameが不適当なら
                submitted = False  # Submit 取り消し

        if submitted:
            st.session_state.username = inputname
            st.session_state.page = 'deal_data'
            st.write("名前: ", inputname)

# ---------------- 訓練データの加工 ----------------------------------
# def deal_data():
#     st.title("deal_data")

# ---------------- テストデータ　プロット ----------------------------------
def classfy():
    st.title('分類させてみる')
    test_idx = st.number_input("データ番号を入力(0~2)", min_value=0, max_value=2)

    # テストデータを取得
    test_data = [[4.4,2.9,1.4,0.2,"setosa"], [6.4,2.9,4.3,1.3,"versicolor"], [5.9,3,5.1,1.8,"virginica"]]
    test_df = pd.DataFrame(test_data,columns=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species'])
    st.dataframe(test_df)

    full_data = load_full_data()
    # test_numまでがテストデータ = 分割後もindexが揃う
    train, _ = train_test_split(full_data, test_size=0.3)
    train_X = train[["PetalWidthCm", "PetalLengthCm"]]
    train_y = train[["Species"]]

    # 学習
    # ここでは決定木を用います
    clf = DecisionTreeClassifier(random_state=0, max_depth=3)
    clf = clf.fit(train_X, train_y)
    # コンピューターの予測結果 
    tmp = test_df.copy()
    tmp = tmp.drop('Species', axis=1)
    pred = clf.predict(tmp[["PetalWidthCm", "PetalLengthCm"]])

    pred_btn = st.checkbox('予測結果をみる')
    if pred_btn:
        st.write('\n機械学習による予測結果は...')
        st.success(pred[test_idx: test_idx+1])

# ---------------- 決定木 : dtreeviz ----------------------------------
def decision_tree():
    st.title("分類を予測しよう")
    
    st.write('予測に使う変数を2つ選ぼう')
    left, right = st.beta_columns(2)
    features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    with left:
        feature1 = st.selectbox('予測に使う変数1',features)
    with right:
        feature2 = st.selectbox('予測に使う変数2',features)

    logging.info(',%s,決定木変数,%s', st.session_state.username, feature1+'_'+feature2)
    # 学習スタート
    started = st.button('学習スタート')
    if not started: 
        st.stop()
    
    # 決定木の可視化
    tree = d.my_dtree(feature1, feature2)
    st.image(tree, caption=feature1+'_'+feature2)

# ---------------- 可視化 :  各グラフを選択する ----------------------------------
def vis():
    st.title("iris データ")

    feature_data = load_num_data()
    full_data = load_full_data()
    label = feature_data.columns

    st.sidebar.markdown('## 様々なグラフを試してみよう')

    # sidebar でグラフを選択
    graph = st.sidebar.radio(
        'グラフの種類',
        ('棒グラフ', '箱ひげ図', '散布図')
    )
    
    # 棒グラフ
    if graph == "棒グラフ":
        st.markdown('## 各変数の分布を棒グラフを用いて調べる')

        with st.form("棒グラフ"):
            # 変数選択
            bar_val = st.selectbox('変数を選択',label)
            logging.info(',%s,棒グラフ,%s', st.session_state.username, bar_val)

            # Submitボタン
            plot_button = st.form_submit_button('グラフ表示')
            if plot_button:
                fig = px.bar(full_data, x='Species', y=bar_val, color=full_data.Species)
                st.plotly_chart(fig, use_container_width=True)

        # コードの表示
        code = st.sidebar.checkbox('コードを表示')
        if code:
            code_txt = "fig = px.bar(full_data, x='Species', y=" + bar_val + ", color=full_data.Species)"
            st.sidebar.markdown('---')
            st.sidebar.write(code_txt)
            st.sidebar.markdown('---')
    
    # 箱ひげ図
    elif graph == '箱ひげ図':
        st.markdown('## 各変数の分布を箱ひげ図を用いて調べる')
        with st.form("箱ひげ図"):
            # 変数選択
            box_val_y = st.selectbox('箱ひげ図にする変数を選択',label)
            logging.info(',%s,箱ひげ図,%s', st.session_state.username, box_val_y)

            # Submitボタン
            plot_button = st.form_submit_button('グラフ表示')
            if plot_button:
                # 箱ひげ図の表示
                fig = px.box(full_data, x='Species', y=box_val_y, color=full_data.Species)
                st.plotly_chart(fig, use_container_width=True)
                # コードの表示
        code = st.sidebar.checkbox('コードを表示')
        if code:
            code_txt = "fig = px.box(full_data, x='Species', y=" + box_val_y + ", color=full_data.Species)"
            st.sidebar.markdown('---')
            st.sidebar.markdown(code_txt)
            st.sidebar.markdown('---')
    
    # 散布図
    elif graph == '散布図':
        label = full_data.columns
        st.markdown('## 各変数の分布を散布図を用いて調べる')
        with st.form("散布図"):
            left, right = st.beta_columns(2)

            with left: # 変数選択 
                x_label = st.selectbox('横軸を選択',label)

            with right:
                y_label = st.selectbox('縦軸を選択',label)
            
            logging.info(',%s,散布図,%s', st.session_state.username, x_label+'_'+y_label)
                
            # Submitボタン
            plot_button = st.form_submit_button('グラフ表示')
            if plot_button:
                # 散布図表示
                fig = px.scatter(data_frame=full_data, x=x_label, y=y_label, color=full_data.Species)
                st.plotly_chart(fig, use_container_width=True)

        # コードの表示
        code = st.sidebar.checkbox('コードを表示')
        if code:
            code_txt = "fig = px.scatter(data_frame=full_data, x=" + x_label + ", y=" + y_label + ", color=full_data.Species)"
            st.sidebar.markdown('---')
            st.sidebar.write(code_txt)
            st.sidebar.markdown('---')

main()