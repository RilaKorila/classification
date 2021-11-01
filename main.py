import streamlit as st
import plotly.express as px
import pandas as pd
import logging
import data as d
# from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
# import time


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
    # page = st.sidebar.radio('', ('データ可視化', 'テストデータ', '決定木'))
    page = st.sidebar.radio('', ('データ可視化', '決定木'))

    # if page == 'データ加工':
    #     st.session_state.page = 'deal_data'
    #     logging.info(',%s,ページ選択,%s', st.session_state.username, page)
    if page == 'データ可視化':
        st.session_state.page = 'vis'
        logging.info(',%s,ページ選択,%s', st.session_state.username, page)
    elif page == 'テストデータ':
        st.session_state.page = 'test'
        logging.info(',%s,ページ選択,%s', st.session_state.username, page)
    elif page == '決定木':
        st.session_state.page = 'decision_tree'
        logging.info(',%s,ページ選択,%s', st.session_state.username, page)

    # --- page振り分け
    if st.session_state.page == 'input_name':
        input_name()
    # elif st.session_state.page == 'deal_data':
    #     deal_data()
    elif st.session_state.page == 'vis':
        vis()
    elif st.session_state.page == 'test':
        test()  
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
def test():
    st.title('テストデータ')
    test_idx = st.number_input("データ番号を入力(0~200)", min_value=0, max_value=200)

    # テストデータを取得
    full_data = load_full_data()
    test_num = 200
    # test_numまでがテストデータ = 分割後もindexが揃う
    train = full_data[test_num:]
    # test_num以降が訓練データ
    test = full_data[:test_num]
    train.drop('Species', axis=1) 
    
    # テストデータを取得
    test_df = full_data[test_idx: test_idx+1].drop('Species', axis=1)
    # 選択したデータの表示
    st.dataframe(test_df)

    # 学習
    # ここでは決定木を用います
    clf = DecisionTreeClassifier(random_state=0, max_depth=3)
    train_X = train.drop('Species', axis=1)
    train_y = train.Species
    clf = clf.fit(train_X, train_y)
    # コンピューターの予測結果  # 1が生存、0が死亡
    pred = clf.predict(test_df)

    pred_btn = st.checkbox('予測結果をみる')
    if pred_btn:
        st.write('\n機械学習による予測結果は...')
        if pred[0] == 1:
            st.success('生存！！')
        else:
            st.success('亡くなってしまうかも...')
        
        # その後、正解を見る
        ans = st.checkbox('正解をみる')
        if ans:
            st.write('\n実際は...')
            if test['Species'][test_idx] == 1:
                st.success('生存！！')
            else:
                st.success('亡くなってしまった...')

            test[test_idx: test_idx+1]


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
    
    # # データの取得
    # train_X, test_X, train_y, test_y = load_ML_data(feature1, feature2, train_num = 600)

    # # 木の深さを3に制限
    # clf = DecisionTreeClassifier(random_state=0, max_depth=3)
    # # 学習
    # clf = clf.fit(train_X, train_y)

    # # test_Xデータを全部予測する
    # pred = clf.predict(test_X)
    # # 正解率を計算する
    # acc = accuracy_score(pred, test_y)

    # st.success('学習終了！！')
    # st.write(f'accuracy: {acc:.5f}')

    # #　決定木の表示までにタイムラグがほしい
    # # 待たせられる
    # with st.spinner('Wait for it...'):
    #     time.sleep(3.5)

    # 決定木の可視化
    tree = d.my_dtree(feature1, feature2)
    st.image(tree, caption=feature1+'_'+feature2)

    # if vis_tree:
    #     viz = dtreeviz(
    #             clf,
    #             train_X, 
    #             train_y,
    #             target_name='Species',
    #             feature_names=train_X.columns,
    #             class_names=['Alive', 'Dead'],
    #         ) 

    #     viz.view()
    # st.set_option('deprecation.showPyplotGlobalUse', False)

    # viz = dtreeviz(
    #             clf,
    #             train_X, 
    #             train_y,
    #             target_name='Species',
    #             feature_names=train_X.columns,
    #             class_names=['Alive', 'Dead'],
    #         ) 
    # st.write("viz OK")

    # viz.view()
    # st.image(viz._repr_svg_(), use_column_width=True)
    # def st_dtree(viz, height=None):
    #     dtree_html = f"<body>{viz.svg()}</body>"
    #     components.html(dtree_html, height=height)

    # st_dtree(viz, 800)
    # st.write('end of code')
    # st.image(viz._repr_svg_(), use_column_width=True)

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